# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
tfce_loso: leave-one-participant-out (LOSO) robustness of the weighted TFCE
group test (l2_tfce, convention 'sn') on a finished HEART depth run.

`main(argv)` is called by run_component_group_tests.py (step loso) with

    <run_dir> --grain perbeat --model session
              --measure hrsp --schemes unweighted precision_nseg
              [--nperm 5000] [--jobs 4] [--E .5 --H 2 --dh .1]
              [--alpha .05] [--ref-main <dir>]

The post-hoc battery's LOSO is cluster-based; this one is TFCE throughout.

Subjects and weights: the full-sample stack is built exactly as tfce_run builds it
(subjects_from_npz -> prepare_measure, i.e. _stack_measure -> _listwise_keep ->
build_weights, with build_ivw_weights over the model's subjects for ivw_shrunk). A fold
drops one participant from the `subjects` list (and the matching entry of the
full-sample ivw vector) and calls the same prepare_measure, so build_weights
re-normalises the remaining weights to mean 1.

ivw_shrunk: the fold keeps the full-sample shrunk ivw weights of the remaining
participants (the dropped participant's weight removed, the rest re-normalised). The
shrinkage prior is not re-estimated without the dropped participant. For unweighted and
precision_nseg the two readings are identical.

ivw_own (accepted although no run persists it): the weights are re-estimated in every
fold. prepare_measure computes them from the list it is handed
(l2_heart.build_own_ivw_weights: the tested measure's own L1 slope variance, shrinkage
c = the median over that list), and a fold hands it the list without the dropped
participant, so the median, and with it every remaining weight, is the fold's own.
Weight estimation is part of the procedure being left one out.

The output CSV merges: rows of (measure, scheme) pairs this call does not recompute are
kept from an existing l2_tfce_loso_<model>.csv, so a LOSO of ivw_own alone leaves the
unweighted folds a figure reads in place.

Per fold (dropped participant x scheme), same nPerm and SEED as the full run: min p, n
significant cells, n components, the largest significant component's time / frequency
span, n significant cells before 0 ms (whole map), and whether the full-sample primary
component's peak cell (component 1 of the full run: most cells) is still p < alpha, with
its p.

Writes <run_dir>/<grain>_wise/main/l2_tfce_loso_<model>.csv (measure in a column) and
prints a summary. The full sample is recomputed first and its min p checked against
l2_tfce_summary_<model>.csv (Convention sn) in --ref-main (default: the run's own main
folder); a mismatch aborts before any fold runs.
"""
from __future__ import annotations

import os
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

from l2_heart import (build_ivw_weights, kish_n_eff, SEED,     # noqa: E402
                          IVW_OWN)
from l2_tfce import (tfce_perm_weighted, tfce_components,       # noqa: E402
                         prepare_measure, DEFAULT_E, DEFAULT_H, DEFAULT_DH)
from tfce_run import subjects_from_npz, TF_MEASURES, _read_cluster_csv  # noqa: E402

CONVENTION = 'sn'
LOSO_FIELDS = (
    'Model', 'Measure', 'Scheme', 'Grain', 'Dropped', 'N_subjects', 'N_eff_kish',
    'nPerm', 'E', 'H', 'dh', 'Alpha', 'Min_p', 'N_sig', 'N_components',
    'Largest_N_cells', 'Largest_polarity', 'Largest_start_ms', 'Largest_end_ms',
    'Largest_extent_ms', 'Largest_start_Hz', 'Largest_end_Hz',
    'Largest_extent_Hz', 'N_sig_preR', 'Primary_peak_ms', 'Primary_peak_Hz',
    'Primary_peak_p', 'Primary_peak_sig', 'Full_min_p', 'Full_N_sig',
)


# =============================================================================
def fold_inputs(subs, meas, scheme, ivw_all, drop_gidx):
    """prepare_measure on `subs` without subject index drop_gidx (None = full).

    Returns (Xk, w, n_drop, kept) with `kept` as indices into the full `subs`.
    """
    if drop_gidx is None:
        return prepare_measure(subs, meas, scheme, ivw_all)
    keep = np.array([i for i in range(len(subs)) if i != drop_gidx], int)
    sub_r = [subs[i] for i in keep]
    ivw_r = None if ivw_all is None else np.asarray(ivw_all)[keep]
    prep = prepare_measure(sub_r, meas, scheme, ivw_r)
    if prep is None:
        return None
    Xk, w, n_drop, kg = prep
    return Xk, w, n_drop, keep[kg]


def summarise_map(t_obs, tf_obs, p_map, times, freqs, w, n_used, alpha,
                  primary=None):
    """Readout of one TFCE map (tfce_components), plus the primary-peak check."""
    comps, best_p, n_sig = tfce_components(t_obs, tf_obs, p_map, times, freqs,
                                           w, n_used, alpha)
    p2 = np.atleast_2d(p_map)
    sig = p2 < alpha
    times = np.asarray(times, float)
    out = dict(Min_p=round(best_p, 6), N_sig=n_sig, N_components=len(comps),
               N_sig_preR=int((sig & (times[None, :] < 0)).sum()),
               N_eff_kish=round(kish_n_eff(w), 3), N_subjects=n_used)
    if comps:
        c = comps[0]
        out.update(Largest_N_cells=c['n_cells'], Largest_polarity=c['polarity'],
                   Largest_start_ms=c['start_ms'], Largest_end_ms=c['end_ms'],
                   Largest_extent_ms=c['end_ms'] - c['start_ms'])
        if freqs is not None:
            out.update(Largest_start_Hz=c['start_hz'], Largest_end_Hz=c['end_hz'],
                       Largest_extent_Hz=c['end_hz'] - c['start_hz'])
    if primary is not None:
        pf, pt = primary
        pp = float(p2[pf, pt])
        out.update(Primary_peak_p=round(pp, 6), Primary_peak_sig=bool(pp < alpha))
    return out, comps


def run_fold(task):
    t0 = time.time()
    _t, tf, p, _nm, n_used = tfce_perm_weighted(
        task['X'], task['w'], task['nperm'], SEED, task['E'], task['H'],
        task['dh'], CONVENTION)
    s, comps = summarise_map(_t, tf, p, task['times'], task['freqs'], task['w'],
                             n_used, task['alpha'], task.get('primary'))
    return dict(key=task['key'], summary=s, comps=comps, p=p, t=_t,
                elapsed=time.time() - t0)


def _pool_map(tasks, jobs, label=''):
    res, t0 = {}, time.time()
    if jobs <= 1:
        for t in tasks:
            r = run_fold(t)
            res[r['key']] = r
            print(f'  {label}{r["key"]} min p {r["summary"]["Min_p"]} '
                  f'[{r["elapsed"]:.0f}s]', flush=True)
        return res
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(run_fold, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res[r['key']] = r
            print(f'  {label}{i}/{len(tasks)} {r["key"]} min p '
                  f'{r["summary"]["Min_p"]} n_sig {r["summary"]["N_sig"]} '
                  f'[{time.time() - t0:.0f}s]', flush=True)
    return res


def merge_loso_rows(out_csv, new_rows, measure, schemes):
    """new_rows preceded by the existing rows of every (measure, scheme) this
    call did not recompute. The file is per model, so a call that redoes one
    scheme must not delete the folds of the others."""
    done = {(measure, sch) for sch in schemes}
    old = [r for r in (_read_cluster_csv(out_csv) or [])
           if (r.get('Measure'), r.get('Scheme')) not in done]
    if old:
        print(f'  keeping {len(old)} existing row(s) of '
              f'{sorted({(r["Measure"], r["Scheme"]) for r in old})}')
    return old + list(new_rows)


# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description='TFCE LOSO on a finished HEART run')
    ap.add_argument('run_dir')
    ap.add_argument('--grain', required=True)
    ap.add_argument('--model', default='session')
    ap.add_argument('--measure', default='hrsp')
    ap.add_argument('--schemes', nargs='+', required=True)
    ap.add_argument('--nperm', type=int, default=5000)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--E', type=float, default=DEFAULT_E)
    ap.add_argument('--H', type=float, default=DEFAULT_H)
    ap.add_argument('--dh', type=float, default=DEFAULT_DH)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--ref-main', default=None,
                    help='folder holding l2_tfce_summary_<model>.csv for the '
                         'full-sample check (default: the run\'s own main)')
    args = ap.parse_args(argv)

    main_dir = os.path.join(args.run_dir, f'{args.grain}_wise', 'main')
    z = np.load(os.path.join(main_dir, 'run_results.npz'), allow_pickle=False)
    if str(z['grain']) != args.grain:
        raise SystemExit(f'npz grain {z["grain"]} != --grain {args.grain}')
    times = np.asarray(z['times'], float)
    freqs = (np.asarray(z['freqs'], float) if args.measure in TF_MEASURES
             else None)
    schemes_npz = [str(s) for s in z['schemes']]
    for sch in args.schemes:
        if sch not in schemes_npz and sch != IVW_OWN:
            raise SystemExit(f'scheme {sch} not persisted in this run '
                             f'({schemes_npz}); ivw_own is the only scheme '
                             f'accepted without being persisted')
    pj = json.loads(str(z['params_json']))
    subs = subjects_from_npz(z, args.model, [args.measure])
    ivw_all = None
    if 'ivw_shrunk' in args.schemes:
        # same call as tfce_run.build_tasks: over the model's own subjects
        subs_ivw = subjects_from_npz(z, args.model, ['hrsp', 'hep', 'hepc'])
        ivw_all, _ = build_ivw_weights(subs_ivw, pj.get('L2_IVW_MEASURE', 'hep'),
                                       verbose=False)
    sids = [s['subject'] for s in subs]
    common = dict(times=times, freqs=freqs, nperm=args.nperm, E=args.E, H=args.H,
                  dh=args.dh, alpha=args.alpha)

    # -- full sample, checked against the written TFCE summary -----------------
    full_in, full_tasks = {}, []
    for sch in args.schemes:
        prep = fold_inputs(subs, args.measure, sch, ivw_all, None)
        if prep is None:
            raise SystemExit(f'{args.measure}/{sch}: no maps')
        Xk, w, n_drop, kept = prep
        full_in[sch] = (Xk, w, kept)
        full_tasks.append(dict(common, key=(sch, 'none'), X=Xk, w=w))
    print(f'{main_dir}\n  {args.model}/{args.measure} schemes {args.schemes}, '
          f'nPerm {args.nperm}, seed {SEED}, convention {CONVENTION}', flush=True)
    print('full sample:', flush=True)
    full = _pool_map(full_tasks, args.jobs, 'full ')
    ref_main = args.ref_main or main_dir
    ref = _read_cluster_csv(os.path.join(ref_main,
                                         f'l2_tfce_summary_{args.model}.csv'))
    if ref is None:
        raise SystemExit(f'no l2_tfce_summary_{args.model}.csv in {ref_main}')
    primary = {}
    for sch in args.schemes:
        r = full[(sch, 'none')]
        rr = [x for x in ref if x['Measure'] == args.measure
              and x['Scheme'] == sch and x['Convention'] == CONVENTION]
        if not rr:
            raise SystemExit(f'no reference row for {args.measure}/{sch}/sn')
        ref_p, ref_n = float(rr[0]['Best_p']), int(rr[0]['N_sig_total'])
        got_p, got_n = r['summary']['Min_p'], r['summary']['N_sig']
        ok = abs(got_p - ref_p) < 1e-9 and got_n == ref_n
        print(f'  full {sch:14s}: min p {got_p} n_sig {got_n} | reference '
              f'{ref_p} / {ref_n} -> {"MATCH" if ok else "MISMATCH"}', flush=True)
        if not ok:
            raise SystemExit('full-sample TFCE does not reproduce the written '
                             'summary; refusing to run the folds')
        if r['comps']:
            c = r['comps'][0]
            primary[sch] = (c['pf'], c['pt'])
        else:
            primary[sch] = None

    # -- folds -----------------------------------------------------------------
    tasks = []
    for sch in args.schemes:
        Xk_full, _w, kept = full_in[sch]
        for j, g in enumerate(kept):
            Xf, wf, _nd, kg = fold_inputs(subs, args.measure, sch, ivw_all, int(g))
            # the fold stack must be the full stack minus that one column
            if (not np.array_equal(kg, np.delete(kept, j))
                    or not np.array_equal(Xf, np.delete(Xk_full, j, axis=-1))):
                raise SystemExit(f'fold {sids[g]}: stack is not the full stack '
                                 f'minus one column')
            tasks.append(dict(common, key=(sch, sids[g]), X=Xf, w=wf,
                              primary=primary[sch]))
    print(f'{len(tasks)} folds, jobs {args.jobs}', flush=True)
    folds = _pool_map(tasks, args.jobs)

    # -- write -----------------------------------------------------------------
    rows = []
    for sch in args.schemes:
        fs = full[(sch, 'none')]['summary']
        pk = primary[sch]
        for g in full_in[sch][2]:
            s = folds[(sch, sids[g])]['summary']
            r = {k: '' for k in LOSO_FIELDS}
            r.update(Model=args.model, Measure=args.measure, Scheme=sch,
                     Grain=args.grain, Dropped=sids[g], nPerm=args.nperm,
                     E=args.E, H=args.H, dh=args.dh, Alpha=args.alpha,
                     Full_min_p=fs['Min_p'], Full_N_sig=fs['N_sig'])
            r.update(s)
            if pk is not None:
                r.update(Primary_peak_ms=float(times[pk[1]]),
                         Primary_peak_Hz='' if freqs is None else float(freqs[pk[0]]))
            rows.append(r)
    out = os.path.join(main_dir, f'l2_tfce_loso_{args.model}.csv')
    rows = merge_loso_rows(out, rows, args.measure, args.schemes)
    tmp = out + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(LOSO_FIELDS))
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, out)
    print(f'WROTE {out} ({len(rows)} rows)')

    # -- summary ---------------------------------------------------------------
    print('\nSUMMARY (TFCE sn, LOSO)')
    for sch in args.schemes:
        rs = [r for r in rows if r['Scheme'] == sch
              and r['Measure'] == args.measure]
        mp = np.array([r['Min_p'] for r in rs], float)
        iw = int(np.argmax(mp))
        n05 = int((mp < args.alpha).sum())
        ext = [r['Largest_extent_ms'] for r in rs if r['Largest_extent_ms'] != '']
        exh = [r['Largest_extent_Hz'] for r in rs if r['Largest_extent_Hz'] != '']
        pre = [r['N_sig_preR'] for r in rs]
        psig = sum(1 for r in rs if r['Primary_peak_sig'] is True)
        print(f'  {sch:14s} full min p {rs[0]["Full_min_p"]} | worst fold p '
              f'{mp[iw]:.4f} (drop {rs[iw]["Dropped"]}) | folds p<{args.alpha}: '
              f'{n05}/{len(rs)} | min p range {mp.min():.4f}..{mp.max():.4f}')
        if ext:
            print(f'  {"":14s} largest-component extent {min(ext):.0f}..'
                  f'{max(ext):.0f} ms'
                  + (f', {min(exh):.0f}..{max(exh):.0f} Hz' if exh else '')
                  + f' | span start {min(r["Largest_start_ms"] for r in rs if r["Largest_start_ms"] != ""):.0f}'
                  f'..end {max(r["Largest_end_ms"] for r in rs if r["Largest_end_ms"] != ""):.0f} ms')
        print(f'  {"":14s} folds with sig cells before 0 ms: '
              f'{sum(1 for v in pre if v > 0)} (max {max(pre)} cells) | primary '
              f'peak still sig in {psig}/{len(rs)}')
    return 0
