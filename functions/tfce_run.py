# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
tfce_run: TFCE group test (l2_tfce) on a finished heart depth run.

`main(argv)` is called by run_component_group_tests.py (steps tfce and confounds) with

    <run_dir> [--grains segment perbeat cell]
              [--models session unadj | all]
              [--measures hrsp hep] [--nperm 5000] [--jobs 4]
              [--dh 0.1] [--E 0.5] [--H 2.0] [--alpha 0.05]
              [--conventions sn mne]
              [--schemes unweighted precision_nseg ivw_shrunk ivw_own]

Reads <run_dir>/<grain>_wise/main/run_results.npz (the per-participant L1 depth-slope maps
of every model) and writes, next to the cluster tables:

    <grain>_wise/main/l2_tfce_summary_<model>.csv   rows = l2_tfce.TFCE_FIELDS
    <grain>_wise/main/l2_tfce_<model>.npz           t_obs, TFCE, p maps, null max,
                                                    weights, kept participants, and the
                                                    exact cells of every component

Participants and weights are those of the cluster L2. The `subjects` list is rebuilt from
the npz exactly as fit_l1 hands it to run_l2_heart (a participant with no map for a measure
is None), and then the same path runs: _stack_measure -> _listwise_keep -> build_weights,
with build_ivw_weights(subjects, 'hep') over the same model's participants for ivw_shrunk.
That is the model's own HEP slope variance, not the unadjusted model's, because
run_l2_heart is called per model with that model's participants. Schemes per grain are the
ones the run persisted (`schemes` in the npz: ivw_shrunk at the segment grain only), unless
--schemes names them.

--schemes accepts any scheme the grain persisted, plus `ivw_own`, which is available at every
grain: shrunk inverse-variance weights from the tested measure's own L1 slope variance in the
same model (l2_heart.build_own_ivw_weights through l2_tfce.prepare_measure). A persisted
scheme asked for at a grain that did not persist it is skipped with a message (e.g.
ivw_shrunk at perbeat). ivw_own has no cluster table to be checked against, so its maps are
verified through the unweighted rows (same participants, same keep rule, so the same X:
N_subjects, CFT, clusters, peaks and p at uniform weight), and its weighted t-map is checked
against the weighted t written out by hand. With --schemes the output merges: the (measure,
scheme) pairs this call does not recompute are carried over from the existing
l2_tfce_summary_<model>.csv / npz, so asking for ivw_own alone cannot wipe the unweighted
rows a figure reads. The carry-over is refused (nothing written for that model) when the
existing file was computed with other TFCE settings (nPerm, E, H, dh, alpha, seed,
conventions) or is older than run_results.npz. Without --schemes the files are rewritten
whole.

A model whose maps do not reproduce the cluster L2 is not written. For every measure x
scheme, before any TFCE is written:
  * N_subjects and N_eff_kish equal the cluster table's;
  * every cluster row's Peak_t equals the t-map here at its Peak_ms / Peak_Hz
    (to the CSV's 4-decimal rounding, |diff| <= 5e-5) and that cell is the
    max |t| over the cluster's exact persisted cells;
  * cluster_perm_weighted re-run on these X and w at the run's own nPerm, seed
    and alpha reproduces CFT, N_supra, N_clusters_total and every row's SumAbsT
    and perm_p (to the CSV rounding). The p depends on the weights, so this is
    the check that the weights are the same, not only the t-map.
A model with any failure is reported and not written.

Only the 'full' window. The cluster run's '@lit' window is a separate
correction space and is not re-tested here.
"""
from __future__ import annotations

import os
# One BLAS thread per worker: the workers already run in parallel, so more threads
# would oversubscribe the cores. Set before numpy is imported (spawned workers
# inherit the env).
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse                                                     # noqa: E402
import csv                                                          # noqa: E402
import json                                                         # noqa: E402
import sys                                                          # noqa: E402
import time                                                         # noqa: E402

import numpy as np                                                  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from l2_heart import (cluster_perm_weighted, build_ivw_weights,  # noqa: E402
                          kish_n_eff, SEED, MIN_SUB_L2, IVW_OWN,
                          KNOWN_SCHEMES)
from l2_tfce import (tfce_perm_weighted_multi, component_rows,  # noqa: E402
                         prepare_measure, TFCE_FIELDS, CONVENTIONS,
                         DEFAULT_E, DEFAULT_H, DEFAULT_DH)

GRAINS = ('segment', 'perbeat', 'cell')
TF_MEASURES = ('hrsp', 'hepc')
PEAK_TOL = 5e-5 + 1e-9          # Peak_t is written round(., 4)


# =============================================================================
def subjects_from_npz(z, model, measures):
    """The `subjects` list fit_l1 built, rebuilt from run_results.npz."""
    sids = [str(s) for s in z['subject_ids']]
    subs = [{'subject': s} for s in sids]
    for meas in measures:
        kb = f'L1_{model}_{meas}_beta'
        if kb not in z.files:
            continue
        B, V = z[kb], z[f'L1_{model}_{meas}_svar']
        nun, has = z[f'L1_{model}_{meas}_nunits'], z[f'L1_{model}_{meas}_hasmap']
        for i in range(len(sids)):
            subs[i][meas] = ((np.asarray(B[i], float), np.asarray(V[i], float),
                              int(nun[i])) if bool(has[i]) and B.ndim > 1
                             and B.shape[1] > 0 else None)
    return subs


def _read_cluster_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _load_cells(z):
    try:
        from heart_depth_run import load_l2_cells
        return load_l2_cells(z)
    except Exception:                                   # noqa: BLE001
        return None


def weighted_t_by_hand(X3, w):
    """The weighted one-sample t of the cluster L2, written out directly
    (not through l2_heart._null_t_weighted): m = sum(w x)/W, s_w^2 = sum(w (x - m)^2)
    n_eff / ((n_eff - 1) W), t = m / sqrt(s_w^2 / n_eff), n_eff = W^2/sum(w^2)."""
    X3 = np.asarray(X3, float)
    w = np.asarray(w, float)
    W = w.sum()
    neff = W ** 2 / np.sum(w ** 2)
    m = (X3 * w).sum(-1) / W
    sw2 = (w * (X3 - m[..., None]) ** 2).sum(-1) * neff / ((neff - 1.0) * W)
    return m / np.sqrt(sw2 / neff)


# =============================================================================
# THE WORKER: verify against the cluster L2, then TFCE
# =============================================================================
def run_task(task):
    """One (grain, model, measure, scheme). Returns a result dict; never raises
    for a verification failure (it is recorded), so one bad cell cannot kill the
    pool."""
    t0 = time.time()
    Xk, w = task['X'], task['w']
    times, freqs = task['times'], task['freqs']
    fails, notes = [], []
    n_used = Xk.shape[-1]
    crow = task['cluster_rows']
    X3 = Xk if Xk.ndim == 3 else Xk[None]
    # ivw_own has no cluster table: its maps are verified at uniform weight
    # against the unweighted rows (build_tasks hands those over), and its own
    # weighted t is checked by hand after the TFCE below.
    w_ver = task.get('w_verify')
    if w_ver is None:
        w_ver = w

    # -- verification against the cluster L2 ----------------------------------
    if crow is None:
        fails.append('no cluster rows for this measure x scheme in '
                     'l2_heart_summary_<model>.csv')
    else:
        r0 = crow[0]
        if int(_f(r0['N_subjects'])) != n_used:
            fails.append(f'N_subjects {n_used} != cluster {r0["N_subjects"]}')
        if abs(round(kish_n_eff(w_ver), 3) - _f(r0['N_eff_kish'])) > 1e-9:
            fails.append(f'N_eff_kish {kish_n_eff(w_ver):.3f} != cluster '
                         f'{r0["N_eff_kish"]}')
        t_cl, cft, cl, _ = cluster_perm_weighted(Xk, w_ver, task['cl_nperm'],
                                                 SEED, alpha=task['cl_alpha'])
        if abs(round(cft, 4) - _f(r0['CFT'])) > 1e-9:
            fails.append(f'CFT {cft:.4f} != cluster {r0["CFT"]}')
        n_supra = int(np.sum(np.abs(t_cl) >= cft))
        if str(r0.get('N_supra', '')) != '' and n_supra != int(_f(r0['N_supra'])):
            fails.append(f'N_supra {n_supra} != cluster {r0["N_supra"]}')
        ok_rows = [r for r in crow if r['Status'] == 'ok']
        if ok_rows and int(_f(ok_rows[0]['N_clusters_total'])) != len(cl):
            fails.append(f'N_clusters_total {len(cl)} != cluster '
                         f'{ok_rows[0]["N_clusters_total"]}')
        worst = 0.0
        for r in ok_rows:
            rank = int(_f(r['Cluster']))
            c = cl[rank - 1] if rank - 1 < len(cl) else None
            ti = int(np.argmin(np.abs(times - _f(r['Peak_ms']))))
            fi = (0 if freqs is None or r.get('Peak_Hz', '') == ''
                  else int(np.argmin(np.abs(freqs - _f(r['Peak_Hz'])))))
            if times[ti] != _f(r['Peak_ms']) or (
                    freqs is not None and freqs[fi] != _f(r['Peak_Hz'])):
                fails.append(f'cluster {rank}: peak location not on the axes')
                continue
            d = abs(float(t_cl[fi, ti]) - _f(r['Peak_t']))
            worst = max(worst, d)
            if d > PEAK_TOL:
                fails.append(f'cluster {rank}: Peak_t {t_cl[fi, ti]:.6f} != '
                             f'{r["Peak_t"]}')
            if c is None:
                fails.append(f'cluster {rank}: not reproduced')
                continue
            if abs(round(c['mass'], 4) - _f(r['SumAbsT'])) > 1e-9:
                fails.append(f'cluster {rank}: SumAbsT {c["mass"]:.4f} != '
                             f'{r["SumAbsT"]}')
            if abs(round(c['p'], 5) - _f(r['perm_p'])) > 1e-9:
                fails.append(f'cluster {rank}: perm_p {c["p"]:.5f} != '
                             f'{r["perm_p"]}')
            if (c['pf'], c['pt']) != (fi, ti):
                fails.append(f'cluster {rank}: peak cell ({c["pf"]},{c["pt"]}) '
                             f'!= CSV ({fi},{ti})')
            pc = task['persisted_cells'].get(rank)
            if pc is not None:
                if pc.shape[0] != c['n_cells'] or not np.array_equal(
                        np.asarray(sorted(map(tuple, pc))),
                        np.asarray(sorted(map(tuple, c['cells'])))):
                    fails.append(f'cluster {rank}: cells differ from the '
                                 f'persisted l2_cells')
                tv = np.abs(t_cl[pc[:, 0], pc[:, 1]])
                pk = pc[int(np.argmax(tv))]
                if (int(pk[0]), int(pk[1])) != (fi, ti):
                    fails.append(f'cluster {rank}: CSV peak is not the max |t| '
                                 f'of the persisted cells')
        notes.append(f'peak_t max|diff| {worst:.1e} over {len(ok_rows)} '
                     f'cluster row(s); cluster L2 reproduced at nPerm '
                     f'{task["cl_nperm"]}'
                     + ('' if task.get('w_verify') is None else
                        ' (maps checked against the UNWEIGHTED cluster table; '
                        'the %s weights have none)' % task['scheme']))

    out = dict(key=task['key'], fails=fails, notes=notes, n_used=n_used, w=w)
    if fails:
        out['elapsed'] = time.time() - t0
        return out

    # -- TFCE -----------------------------------------------------------------
    t_obs, res, _ = tfce_perm_weighted_multi(
        Xk, w, task['nperm'], SEED, task['E'], task['H'], task['dh'],
        task['conventions'])
    if task.get('w_verify') is None:
        if not np.allclose(t_obs, t_cl, rtol=0, atol=1e-12):
            out['fails'] = ['TFCE t_obs differs from the cluster driver t-map']
            return out
    else:
        t_h = weighted_t_by_hand(X3, w)
        d_h = float(np.max(np.abs(t_obs - t_h)))
        if not d_h <= 1e-9:
            out['fails'] = ['TFCE t_obs differs from the hand weighted t '
                            '(max |diff| %.3g)' % d_h]
            return out
        out['notes'].append('%s weighted t == hand formula (max |diff| %.1e)'
                            % (task['scheme'], d_h))
    rows_all, comps_all = [], {}
    for c in task['conventions']:
        meta = dict(Model=task['model'], Measure=task['measure'],
                    Scheme=task['scheme'], Convention=c,
                    N_dropped_nonfinite=task['n_drop'], nPerm=task['nperm'],
                    E=task['E'], H=task['H'], dh=task['dh'])
        rows, comps = component_rows(meta, t_obs, res[c]['tfce_obs'],
                                     res[c]['p_map'], times, freqs, w, n_used,
                                     task['alpha'])
        rows_all.extend(rows)
        comps_all[c] = comps
    out.update(t_obs=t_obs, res=res, rows=rows_all, comps=comps_all,
               elapsed=time.time() - t0)
    return out


# =============================================================================
def resolve_schemes(persisted, requested):
    """(schemes to run, [(scheme, why) not run]) for one grain.

    requested None -> the persisted list, unchanged. Otherwise each requested
    scheme runs if the grain persisted it or it is ivw_own (available at every
    grain); a persisted-elsewhere scheme is reported, never silently dropped.
    A name outside KNOWN_SCHEMES is refused by main() before anything runs.
    """
    if requested is None:
        return list(persisted), []
    run, skip = [], []
    for s in requested:
        if s in run:
            continue
        if s in persisted or s == IVW_OWN:
            run.append(s)
        else:
            skip.append((s, 'not persisted at this grain (%s)'
                         % ', '.join(persisted)))
    return run, skip


def build_tasks(main, z, model, measures, args, crows_all, cells_by_key):
    times = np.asarray(z['times'], float)
    freqs = np.asarray(z['freqs'], float)
    persisted = [str(s) for s in z['schemes']]
    schemes, _not_run = resolve_schemes(persisted, getattr(args, 'schemes', None))
    pj = json.loads(str(z['params_json']))
    subs = subjects_from_npz(z, model, measures)
    ivw_all = None
    if 'ivw_shrunk' in schemes:
        ivw_all, _ = build_ivw_weights(subs, pj.get('L2_IVW_MEASURE', 'hep'),
                                       verbose=False)
    tasks, skipped = [], []
    grain = str(z['grain'])
    pcl = (cells_by_key or {}).get(model, {})
    for meas in measures:
        if f'L1_{model}_{meas}_beta' not in z.files:
            skipped.append((meas, '*', 'no L1 map for this measure in the npz'))
            continue
        for sch in schemes:
            try:
                prep = prepare_measure(subs, meas, sch, ivw_all)
            except (ValueError, ZeroDivisionError) as exc:
                skipped.append((meas, sch, f'skipped:bad_weights {exc}'))
                continue
            if prep is None:
                skipped.append((meas, sch, 'skipped:no_maps'))
                continue
            Xk, w, n_drop, _kept = prep
            if Xk.shape[-1] < MIN_SUB_L2:
                skipped.append((meas, sch, 'skipped:too_few_subjects'))
                continue
            # ivw_own: verify the maps through the unweighted cluster rows at
            # uniform weight (same subjects and keep rule, hence the same X)
            vsch = 'unweighted' if sch == IVW_OWN else sch
            w_verify = np.ones(Xk.shape[-1]) if sch == IVW_OWN else None
            crow = [r for r in (crows_all or [])
                    if r['Measure'] == meas and r['Scheme'] == vsch] or None
            pc = {}
            for rec in pcl.get(f'{meas}|{vsch}', []):
                pc[int(rec['rank'])] = np.column_stack(
                    [np.asarray(rec['cells'])[:, 0], np.asarray(rec['t_index'])])
            kept_mask = np.zeros(len(subs), bool)
            kept_mask[_kept] = True
            tasks.append(dict(
                key=(grain, model, meas, sch), grain=grain, model=model,
                measure=meas, scheme=sch, X=Xk, w=w, w_verify=w_verify,
                n_drop=n_drop,
                kept=kept_mask, times=times,
                freqs=freqs if meas in TF_MEASURES else None,
                cluster_rows=crow, persisted_cells=pc,
                cl_nperm=int(pj['nPerm']), cl_alpha=float(pj['ALPHA']),
                nperm=args.nperm, E=args.E, H=args.H, dh=args.dh,
                alpha=args.alpha, conventions=tuple(args.conventions)))
    return tasks, skipped


_PAYLOAD_PREFIXES = ('t_obs', 'w', 'kept', 'tfce', 'p', 'nullmax', 'sigcells')


def _tfce_params(args):
    return dict(nperm=args.nperm, E=args.E, H=args.H, dh=args.dh,
                alpha=args.alpha, seed=SEED, conventions=list(args.conventions))


def carry_over(main, model, z, args, replaced):
    """(rows, payload, index, note) of the existing l2_tfce_<model> files for
    every (measure, scheme) pair not in `replaced`, or raises RuntimeError when
    those pairs exist but cannot be carried (different TFCE settings, a file
    older than run_results.npz, or different axes / subjects).

    `replaced` holds (measure, scheme) pairs; scheme '*' covers the measure.
    """
    csv_path = os.path.join(main, f'l2_tfce_summary_{model}.csv')
    npz_path = os.path.join(main, f'l2_tfce_{model}.npz')
    if not (os.path.isfile(csv_path) and os.path.isfile(npz_path)):
        return [], {}, {}, 'no existing file'

    def gone(meas, sch):
        return (meas, sch) in replaced or (meas, '*') in replaced

    old_rows = _read_cluster_csv(csv_path) or []
    keep_rows = [r for r in old_rows if not gone(r['Measure'], r['Scheme'])]
    if not keep_rows:
        return [], {}, {}, 'nothing to carry'
    pairs = sorted({(r['Measure'], r['Scheme']) for r in keep_rows})
    with np.load(npz_path, allow_pickle=False) as oz:
        old_par = json.loads(str(oz['params_json']))
        want = _tfce_params(args)
        diff = [k for k in want if old_par.get(k) != want[k]]
        if diff:
            raise RuntimeError(
                'existing %s holds %s computed with other TFCE settings (%s: '
                'file %s, now %s); recompute those schemes too or remove the file'
                % (os.path.basename(npz_path), pairs, ', '.join(diff),
                   [old_par.get(k) for k in diff], [want[k] for k in diff]))
        rr = os.path.join(main, 'run_results.npz')
        if os.path.getmtime(npz_path) < os.path.getmtime(rr):
            raise RuntimeError(
                'existing %s is OLDER than run_results.npz, so its %s may '
                'describe maps that no longer exist; recompute them'
                % (os.path.basename(npz_path), pairs))
        if not (np.array_equal(oz['times'], np.asarray(z['times'], float))
                and np.array_equal(oz['freqs'], np.asarray(z['freqs'], float))
                and np.array_equal(oz['subject_ids'].astype(str),
                                   np.asarray(z['subject_ids']).astype(str))):
            raise RuntimeError('existing %s is on other axes or subjects'
                               % os.path.basename(npz_path))
        payload = {}
        for k in oz.files:
            parts = k.split('__')
            if (parts[0] in _PAYLOAD_PREFIXES and len(parts) in (3, 4)
                    and (parts[1], parts[2]) in pairs):
                payload[k] = oz[k]
        old_index = json.loads(str(oz['sigcells_index_json']))
    index = {k: v for k, v in old_index.items()
             if tuple(k.split('|')[:2]) in pairs}
    return keep_rows, payload, index, 'carried %s' % pairs


def write_model(main, model, z, tasks, results, skipped, args):
    # with --schemes, the pairs this call does not recompute are carried over
    # from the existing files (see carry_over); without it, a whole rewrite
    carried_rows, carried_payload, carried_index, carry_note = [], {}, {}, ''
    if getattr(args, 'schemes', None) is not None:
        replaced = ({(t['measure'], t['scheme']) for t in tasks}
                    | {(m, s_) for m, s_, _w in skipped})
        carried_rows, carried_payload, carried_index, carry_note = carry_over(
            main, model, z, args, replaced)
    rows, payload, index = [], {}, {}
    order = {s: i for i, s in enumerate(str(s) for s in z['schemes'])}
    for t in sorted(tasks, key=lambda t: (t['measure'], order.get(t['scheme'], 9))):
        r = results[t['key']]
        rows.extend(r['rows'])
        tag = f'{t["measure"]}__{t["scheme"]}'
        payload[f't_obs__{tag}'] = np.squeeze(r['t_obs'])
        payload[f'w__{tag}'] = r['w']
        payload[f'kept__{tag}'] = t['kept']
        for c in t['conventions']:
            rc = r['res'][c]
            payload[f'tfce__{tag}__{c}'] = np.squeeze(rc['tfce_obs'])
            payload[f'p__{tag}__{c}'] = np.squeeze(rc['p_map'])
            payload[f'nullmax__{tag}__{c}'] = rc['null_max']
            recs, blocks, n0 = [], [], 0
            for i, comp in enumerate(r['comps'][c], 1):
                cc = comp['cells']
                blocks.append(cc)
                recs.append(dict(component=i, polarity=comp['polarity'],
                                 n_cells=int(cc.shape[0]), start=n0,
                                 stop=n0 + int(cc.shape[0]),
                                 min_p=comp['min_p']))
                n0 += int(cc.shape[0])
            payload[f'sigcells__{tag}__{c}'] = (np.concatenate(blocks)
                                                if blocks else
                                                np.zeros((0, 2), np.int32))
            index[f'{t["measure"]}|{t["scheme"]}|{c}'] = recs
    for meas, sch, why in skipped:
        rr = {k: '' for k in TFCE_FIELDS}
        rr.update(Model=model, Measure=meas, Scheme=sch, Status=why,
                  nPerm=args.nperm, E=args.E, H=args.H, dh=args.dh)
        rows.append(rr)
    if carried_rows:
        # stable: each pair keeps its own row order (components, conventions)
        rank = {s_: i for i, s_ in enumerate(KNOWN_SCHEMES)}
        rows = sorted(rows + carried_rows,
                      key=lambda r: (str(r['Measure']),
                                     rank.get(str(r['Scheme']), 99)))
        payload.update(carried_payload)
        for k, v in carried_index.items():
            index.setdefault(k, v)
        print(f'  {model}: {carry_note} from the existing file')
    payload.update(times=np.asarray(z['times'], float),
                   freqs=np.asarray(z['freqs'], float),
                   subject_ids=np.asarray(z['subject_ids']),
                   sigcells_index_json=np.array(json.dumps(index)),
                   params_json=np.array(json.dumps(dict(
                       nperm=args.nperm, E=args.E, H=args.H, dh=args.dh,
                       alpha=args.alpha, seed=SEED,
                       conventions=list(args.conventions),
                       cells_axes='(freq index, time index on the full times '
                                  'axis); HEP freq index is 0')))
                   )
    csv_path = os.path.join(main, f'l2_tfce_summary_{model}.csv')
    npz_path = os.path.join(main, f'l2_tfce_{model}.npz')
    tmp = csv_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(TFCE_FIELDS))
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, csv_path)
    tmpz = npz_path + '.tmp.npz'
    np.savez_compressed(tmpz, **payload)
    os.replace(tmpz, npz_path)
    return csv_path, npz_path, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description='TFCE L2 on a finished HEART run')
    ap.add_argument('run_dir')
    ap.add_argument('--grains', nargs='+', default=list(GRAINS))
    ap.add_argument('--models', nargs='+', default=['session', 'unadj'],
                    help="model names, or 'all' for every model in the npz")
    ap.add_argument('--measures', nargs='+', default=['hrsp', 'hep'])
    ap.add_argument('--nperm', type=int, default=5000)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--E', type=float, default=DEFAULT_E)
    ap.add_argument('--H', type=float, default=DEFAULT_H)
    ap.add_argument('--dh', type=float, default=DEFAULT_DH)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--conventions', nargs='+', default=list(CONVENTIONS))
    ap.add_argument('--no-write', action='store_true',
                    help='verify and compute, write nothing (smoke test)')
    ap.add_argument('--schemes', nargs='+', default=None,
                    help='schemes to run (default: those the grain persisted). '
                         'ivw_own is available at every grain. Pairs not '
                         'recomputed are carried over from the existing files.')
    args = ap.parse_args(argv)
    if args.schemes is not None:
        bad = [s for s in args.schemes if s not in KNOWN_SCHEMES]
        if bad:
            ap.error('unknown scheme(s) %s; known %s' % (bad, list(KNOWN_SCHEMES)))

    work = []                       # (main, model, z, tasks, skipped)
    for grain in args.grains:
        main = os.path.join(args.run_dir, f'{grain}_wise', 'main')
        npz = os.path.join(main, 'run_results.npz')
        if not os.path.isfile(npz):
            print(f'[{grain}] no run_results.npz in {main}: run the participant-level step '
                  f'first (run_component_analysis.py)')
            return 1
        z = np.load(npz, allow_pickle=False)
        cells_by_key = _load_cells(z)
        models = ([str(m) for m in z['models']] if args.models == ['all']
                  else args.models)
        meas = [m for m in args.measures if m in [str(x) for x in z['measures']]]
        _run, _not = resolve_schemes([str(x) for x in z['schemes']], args.schemes)
        for s_, why in _not:
            print(f'[{grain}] scheme {s_} skipped: {why}')
        for model in models:
            crows = _read_cluster_csv(os.path.join(
                main, f'l2_heart_summary_{model}.csv'))
            tasks, skipped = build_tasks(main, z, model, meas, args, crows,
                                         cells_by_key)
            work.append((main, model, z, tasks, skipped))
            print(f'[{grain}] {model}: {len(tasks)} task(s)'
                  + (f', skipped {skipped}' if skipped else ''), flush=True)

    all_tasks = [t for w_ in work for t in w_[3]]
    print(f'{len(all_tasks)} tasks, nperm {args.nperm}, jobs {args.jobs}',
          flush=True)
    results = {}
    t0 = time.time()
    if args.jobs <= 1:
        for t in all_tasks:
            r = run_task(t)
            results[r['key']] = r
            print(f'  done {r["key"]} {r.get("elapsed", 0):.0f}s '
                  f'{"FAIL " + "; ".join(r["fails"]) if r["fails"] else "ok"}',
                  flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_task, t) for t in all_tasks]
            for f in as_completed(futs):
                r = f.result()
                results[r['key']] = r
                print(f'  done {r["key"]} {r.get("elapsed", 0):.0f}s '
                      f'{"FAIL " + "; ".join(r["fails"]) if r["fails"] else "ok"}'
                      f' [{time.time() - t0:.0f}s]', flush=True)

    n_fail_models = 0
    for main, model, z, tasks, skipped in work:
        fails = [(t['key'], results[t['key']]['fails']) for t in tasks
                 if results[t['key']]['fails']]
        if fails:
            n_fail_models += 1
            print(f'REFUSED to write {main} / {model}: '
                  + ' | '.join(f'{k}: {"; ".join(v)}' for k, v in fails))
            continue
        for t in tasks:
            for nt in results[t['key']]['notes']:
                print(f'  verified {t["key"]}: {nt}')
            for r in results[t['key']]['rows']:
                if r['Convention'] == args.conventions[0] and r['Component'] in (0, 1):
                    print(f'    {r["Measure"]:4s} {r["Scheme"]:14s} {r["Convention"]}: '
                          f'best p {r["Best_p"]} n_sig {r["N_sig_total"]} '
                          f'{r["Start_ms"]}..{r["End_ms"]} ms {r["Start_Hz"]}..{r["End_Hz"]} Hz '
                          f'preR {r["N_cells_preR"]} g {r["HedgesG"]}')
        if args.no_write:
            print(f'(--no-write) {main} / {model} verified, nothing written')
            continue
        try:
            csv_path, npz_path, rows = write_model(main, model, z, tasks,
                                                   results, skipped, args)
        except RuntimeError as exc:
            n_fail_models += 1
            print(f'REFUSED to write {main} / {model}: {exc}')
            continue
        print(f'WROTE {csv_path} ({len(rows)} rows) + {os.path.basename(npz_path)}')
    return 1 if n_fail_models else 0
