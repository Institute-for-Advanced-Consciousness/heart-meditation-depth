# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
tfce_scalp: TFCE (Smith & Nichols 2009, E = .5, H = 2) as the group test of the
scalp arm (Analysis 1: channel x frequency x time maps), in place of its cluster-mass
test, so the paper uses TFCE throughout.

Per run dir (results/.../HEART_DEPTH_SCALP/<cell>):

    main/run_results.npz  l1__<measure>__<model>__<subject>
      -> subjects stacked exactly as the scalp L2 does (stack_subjects, as
         heart_scalp_pipeline._stack_scalp: sorted subject ids,
         listwise-finite), unweighted as in the scalp arm's cluster test, or
         with ivw_own weights (--scheme ivw_own, the setting run_scalp_analysis.py
         passes from config.SCALP_TFCE)
      -> one-sample t-map exactly as cluster_perm_3d (heart_scalp_pipeline.
         _null_t, identity flip column); weighted under ivw_own
      -> checked first against main/l2_scalp_summary.csv: the CFT (re-derived
         from the scalp arm's own flip matrix), N_supra, and the heaviest
         cluster's N_cells / SumAbsT / Peak_t. A mismatch refuses the measure.
      -> sign-flip null (l2_heart._make_flips, column 0 = identity, never
         in the null), graph TFCE of every map (l2_tfce.tfce_graph) on the
         channel x freq x time graph (l2_tfce.scalp_cell_edges over
         heart_scalp_pipeline.build_channel_adjacency)
      -> per-cell FWE p = (1 + #(null max|TFCE| >= |TFCE_obs|)) / nPerm
      -> main/l2_scalp_tfce_summary.csv  (one row per significant component,
                                          or one no_sig row per map)
         main/l2_scalp_tfce.npz          (t, tfce, p, null_max per map)
         (under ivw_own: l2_scalp_tfce_ivw_own_summary.csv, l2_scalp_tfce_ivw_own.npz)

The measures are HEP, HRSP and HEPC; HEPC, as the code names it, is the paper's HRPC
(heartbeat-related phase consistency).

The cell graph is not build_cluster_labeller's. That labeller also joins the last time
point of each frequency row to the first time point of the next row (a wrap-around at
the row bounds); scalp_cell_edges does not (see l2_tfce). The check against the cluster
CSV therefore uses the labeller itself, to reproduce what the cluster test did, and the
TFCE uses the graph without the wrap-around.

`main(argv)` is called by run_scalp_analysis.py (step tfce) with these arguments:
    <run_dir> [<run_dir> ...]
        [--measures hrsp hep hepc] [--models session unadj]
        [--nperm 5000] [--dh 0.1] [--jobs 5] [--check-only] [--scheme unweighted]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from l2_tfce import (tfce_graph, scalp_cell_edges, graph_components,  # noqa: E402
                         DEFAULT_E, DEFAULT_H, DEFAULT_DH)
from l2_heart import (_make_flips, _hedges_J, cfa_overlap, SEED,  # noqa: E402
                          _null_t_weighted, build_weights, kish_n_eff, IVW_OWN)
from fmt_depth_pipeline import shrunk_ivw                              # noqa: E402

FIELDS = (
    'Measure', 'Model', 'Status', 'N_subjects', 'nPerm', 'E', 'H', 'dh',
    'Alpha', 'Best_p', 'N_sig_total', 'N_components', 'Component', 'Polarity',
    'Start_ms', 'End_ms', 'Peak_ms', 'Start_Hz', 'End_Hz', 'Peak_Hz',
    'Channels', 'N_channels', 'Peak_channel', 'Peak_t', 'Peak_TFCE', 'Min_p',
    'N_cells', 'N_cells_preR', 'HedgesG', 'CFA_flag',
    'Cluster_Best_p', 'Cluster_N_survived', 'Note',
)


# =============================================================================
# reading a run, and the workers
# =============================================================================
_W = {}

def load_run(main_dir):
    """(z, times, freqs, ch_names, models) from a scalp run's run_results.npz.

    Raises when the saved time axis does not match the maps (every L1 map is cut to TLIMS;
    indexing a full epoch axis into it would shift every latency).
    """
    z = np.load(os.path.join(main_dir, 'run_results.npz'), allow_pickle=False)
    times = np.asarray(z['times'], float)
    freqs = np.asarray(z['freqs'], float)
    ch_names = [str(c) for c in z['ch_names']]
    l1 = [k for k in z.files if k.startswith('l1__')]
    models = sorted({k.split('__')[2] for k in l1}) or ['session', 'unadj']
    if l1:
        nT = int(np.asarray(z[l1[0]]).shape[-1])
        if nT != times.size:
            raise ValueError('%s: run_results.npz stores a %d-point time axis but the L1 maps '
                             'have %d points' % (main_dir, times.size, nT))
    return z, times, freqs, ch_names, tuple(models)


def stack_subjects(z, measure, model):
    """[nC x (nF) x nT x nSub], listwise-complete, as `_stack_scalp` builds it."""
    keys = sorted(k for k in z.files if k.startswith(f'l1__{measure}__{model}__'))
    if not keys:
        return None
    X = np.stack([np.asarray(z[k], float) for k in keys], axis=-1)
    finite = np.isfinite(X).all(axis=tuple(range(X.ndim - 1)))
    return X[..., finite]


def _init_worker(X2d, eu, ev, E, H, dh, w=None):
    _W.update(X2d=X2d, eu=eu, ev=ev, E=E, H=H, dh=dh, w=w)


def own_ivw_weights(z, meas, model):
    """ivw_own for the scalp arm: (w [nKept], kept subject ids).

    l2_heart.build_ivw_weights' arithmetic on the scalp run's stored
    `svarmean__<meas>__<model>__<sid>` (each participant's mean L1 slope
    variance over the whole channel x freq x time map): shrunk_ivw(v, 'median')
    = 1 / (v + median v), then build_weights -> mean 1. The subjects are
    stack_subjects' (sorted keys, listwise-finite maps), the median is taken
    over exactly those. Raises when the run has no svarmean keys.
    """
    keys = sorted(k for k in z.files if k.startswith(f'l1__{meas}__{model}__'))
    X = np.stack([np.asarray(z[k], float) for k in keys], axis=-1)
    finite = np.isfinite(X).all(axis=tuple(range(X.ndim - 1)))
    sids = [k.split('__')[-1] for k, f in zip(keys, finite) if f]
    miss = [s for s in sids if f'svarmean__{meas}__{model}__{s}' not in z.files]
    if miss:
        raise SystemExit(f'{meas}/{model}: no svarmean for {miss[:3]}... - this '
                         f'run does not store the slope variances; re-run the '
                         f'scalp cell so it stores them')
    sv = np.array([float(z[f'svarmean__{meas}__{model}__{s}']) for s in sids])
    w_all, info = shrunk_ivw(sv, 'median')
    if info['nUsed'] == 0:
        raise SystemExit(f'{meas}/{model}: no usable slope variance')
    return build_weights(IVW_OWN, np.ones(len(sids)), w_all), sids


def _work(args):
    """TFCE of the flip columns in `flips_blk`; returns (cols, max|TFCE|, obs map
    or None). Column index 0 is the identity: its full map is returned."""
    from heart_scalp_pipeline import _null_t
    cols, flips_blk = args
    if _W.get('w') is None:
        T = _null_t(_W['X2d'], flips_blk)              # [nCell x nBlk]
    else:
        T = _null_t_weighted(_W['X2d'], _W['w'], flips_blk)
    mx = np.empty(len(cols))
    obs = None
    for j, c in enumerate(cols):
        tf = tfce_graph(T[:, j], _W['eu'], _W['ev'], _W['E'], _W['H'],
                        _W['dh'])['sn']
        mx[j] = float(np.max(np.abs(tf)))
        if c == 0:
            obs = (tf, T[:, j].copy())
    return np.asarray(cols), mx, obs


def tfce_perm_graph(X2d, eu, ev, n_perm, seed=SEED, E=DEFAULT_E, H=DEFAULT_H,
                    dh=DEFAULT_DH, jobs=1, blk=20, verbose=True, w=None):
    """(tfce_obs, t_obs [nCell], null_max [n_perm-1], p [nCell], seconds) for
    X2d [nCell x nSub]. t_obs is column 0 (identity) of the same matmul as the null."""
    nSub = X2d.shape[1]
    flips = _make_flips(nSub, n_perm, seed)
    tasks = [(list(range(s, min(n_perm, s + blk))), flips[:, s:min(n_perm, s + blk)])
             for s in range(0, n_perm, blk)]
    null_max = np.full(n_perm, np.nan)
    tf_obs = None
    t0 = time.time()
    done = 0
    if jobs > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(jobs, _init_worker,
                                       (X2d, eu, ev, E, H, dh, w)) as pool:
            for cols, mx, obs in pool.imap_unordered(_work, tasks):
                null_max[cols] = mx
                if obs is not None:
                    tf_obs = obs
                done += len(cols)
                if verbose and (done % max(blk, n_perm // 10) < len(cols)):
                    el = time.time() - t0
                    print(f'      {done}/{n_perm} maps, {el:.0f} s '
                          f'({el / done * 1000:.0f} ms/map wall)', flush=True)
    else:
        _init_worker(X2d, eu, ev, E, H, dh, w)
        for tk in tasks:
            cols, mx, obs = _work(tk)
            null_max[cols] = mx
            if obs is not None:
                tf_obs = obs
    assert np.isfinite(null_max).all() and tf_obs is not None
    nm = null_max[1:]                                  # identity never in the null
    srt = np.sort(nm)
    a = np.abs(tf_obs[0])
    n_ge = srt.size - np.searchsorted(srt, a, side='left')
    p = (1.0 + n_ge) / n_perm
    tf_obs, t_obs = tf_obs
    return tf_obs, t_obs, nm, p, time.time() - t0


# =============================================================================
# the check against the cluster test's own table
# =============================================================================
def _read_cluster_csv(main_dir):
    with open(os.path.join(main_dir, 'l2_scalp_summary.csv'), newline='',
              encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault((r['Measure'], r['Model']), []).append(r)
    return by


def check_against_cluster(X, rows, ch_names, n_perm_cluster, verbose=True):
    """Re-derive the scalp cluster test's CFT / N_supra / heaviest cluster from
    X [nC x nF x nT x nSub] with its own flips and labeller. Returns (t_obs64,
    report dict). Raises on any mismatch."""
    from heart_scalp_pipeline import (_null_t, build_channel_adjacency,
                                          build_cluster_labeller, label_signed,
                                          MAX_NULL_CHUNK_GB)
    from heart_scalp_pipeline import SEED as SCALP_SEED
    nC, nF, nT, nSub = X.shape
    n_cell = nC * nF * nT
    X2d = X.reshape(n_cell, nSub)
    rng = np.random.default_rng(SCALP_SEED)             # cluster_perm_3d verbatim
    flips = rng.choice(np.array([-1.0, 1.0]), size=(nSub, n_perm_cluster))
    flips[:, 0] = 1.0
    cpc = max(1, int(MAX_NULL_CHUNK_GB * (1 << 30) / (n_perm_cluster * 8)))
    t32 = np.empty((n_cell, n_perm_cluster), dtype=np.float32)
    for s in range(0, n_cell, cpc):
        e = min(s + cpc, n_cell)
        t32[s:e] = _null_t(X2d[s:e], flips).astype(np.float32)
    t_obs32 = t32[:, 0].astype(float)
    np.abs(t32, out=t32)
    cft = float(np.percentile(t32, 95.0))
    del t32
    ok = [r for r in rows if r['Status'] in ('ok', 'no supra-threshold cells')]
    r0 = ok[0]
    rep = dict(cft=cft, cft_csv=float(r0['CFT']),
               n_supra=int(np.sum(np.abs(t_obs32) >= cft)),
               n_supra_csv=int(r0['N_supra']), n_sub=nSub,
               n_sub_csv=int(r0['N_subjects']))
    adj, _ = build_channel_adjacency(ch_names)
    lab = label_signed(t_obs32, cft, build_cluster_labeller(adj, nF, nT))
    pos = lab >= 0
    masses = np.bincount(lab[pos], weights=np.abs(t_obs32)[pos])
    top = int(np.argmax(masses))
    this = lab == top
    pk = int(np.argmax(np.where(this, np.abs(t_obs32), -np.inf)))
    c1 = next(r for r in ok if r['Cluster'] == '1')
    rep.update(top_cells=int(this.sum()), top_cells_csv=int(c1['N_cells']),
               top_mass=float(masses[top]), top_mass_csv=float(c1['SumAbsT']),
               top_peak_t=float(t_obs32[pk]), top_peak_t_csv=float(c1['Peak_t']),
               n_clusters=int(masses.size), n_clusters_csv=int(r0['N_clusters_total']))
    bad = []
    if rep['n_sub'] != rep['n_sub_csv']:
        bad.append('N_subjects')
    if abs(rep['cft'] - rep['cft_csv']) > 6e-5:
        bad.append('CFT')
    if rep['n_supra'] != rep['n_supra_csv']:
        bad.append('N_supra')
    if rep['top_cells'] != rep['top_cells_csv']:
        bad.append('N_cells')
    if abs(rep['top_mass'] - rep['top_mass_csv']) > 2e-3:       # csv: 3 dp
        bad.append('SumAbsT')
    if abs(rep['top_peak_t'] - rep['top_peak_t_csv']) > 6e-4:
        bad.append('Peak_t')
    if rep['n_clusters'] != rep['n_clusters_csv']:
        bad.append('N_clusters_total')
    rep['mismatch'] = bad
    # the t the TFCE uses: float64, the identity column of _null_t
    t64 = _null_t(X2d, np.ones((nSub, 1)))[:, 0]
    rep['max_abs_t32_vs_t64'] = float(np.max(np.abs(t64 - t_obs32)))
    if verbose:
        print(f'    check: CFT {cft:.4f} (csv {rep["cft_csv"]:.4f}), N_supra '
              f'{rep["n_supra"]} ({rep["n_supra_csv"]}), top cluster cells '
              f'{rep["top_cells"]} ({rep["top_cells_csv"]}), mass '
              f'{rep["top_mass"]:.3f} ({rep["top_mass_csv"]:.3f}), peak t '
              f'{rep["top_peak_t"]:.3f} ({rep["top_peak_t_csv"]:.3f}), '
              f'n clusters {rep["n_clusters"]} ({rep["n_clusters_csv"]}) -> '
              f'{"OK" if not bad else "MISMATCH " + ",".join(bad)}', flush=True)
    if bad:
        raise RuntimeError(f'cluster-table check failed: {bad} {rep}')
    return t64, rep


# =============================================================================
# readout
# =============================================================================
def readout_rows(meas, model, t, tf, p, eu, ev, times, freqs, ch_names, n_sub,
                 meta, alpha=0.05):
    """One row per significant component (per sign, on the TFCE graph),
    sorted by size; a single no_sig row when nothing is significant."""
    nC, nF, nT = t.shape
    sig = (p < alpha).ravel()
    base = {k: '' for k in FIELDS}
    base.update(meta)
    base.update(Measure=meas, Model=model, N_subjects=n_sub, Alpha=alpha,
                Best_p=round(float(p.min()), 6), N_sig_total=int(sig.sum()))
    tfl, tl, pl = tf.ravel(), t.ravel(), p.ravel()
    J = _hedges_J(n_sub)
    rows = []
    if sig.any():
        lab = graph_components(sig, np.sign(tl), eu, ev)
        ncomp = int(lab.max()) + 1
        comps = []
        for ci in range(ncomp):
            idx = np.flatnonzero(lab == ci)
            c, f, tt = np.unravel_index(idx, (nC, nF, nT))
            k = idx[int(np.argmax(np.abs(tl[idx])))]
            pc, pf, pt = np.unravel_index(k, (nC, nF, nT))
            s_ms, e_ms = float(times[tt.min()]), float(times[tt.max()])
            chans = [ch_names[i] for i in sorted(set(c.tolist()))]
            comps.append(dict(
                Polarity='pos' if tl[k] > 0 else 'neg',
                Start_ms=s_ms, End_ms=e_ms, Peak_ms=float(times[pt]),
                Start_Hz='' if freqs is None else float(freqs[f.min()]),
                End_Hz='' if freqs is None else float(freqs[f.max()]),
                Peak_Hz='' if freqs is None else float(freqs[pf]),
                Channels='+'.join(chans), N_channels=len(chans),
                Peak_channel=ch_names[pc], Peak_t=round(float(tl[k]), 4),
                Peak_TFCE=round(float(tfl[k]), 4),
                Min_p=round(float(pl[idx].min()), 6), N_cells=int(idx.size),
                N_cells_preR=int((times[tt] < 0).sum()),
                HedgesG=round(float(tl[k] / np.sqrt(n_sub) * J), 4),
                CFA_flag=bool(cfa_overlap(s_ms, e_ms))))
        comps.sort(key=lambda d: (-d['N_cells'], d['Min_p']))
        for i, d in enumerate(comps, 1):
            r = dict(base)
            r.update(d, Status='ok', Component=i, N_components=len(comps))
            rows.append(r)
    else:
        k = int(np.argmax(np.abs(tfl)))
        pc, pf, pt = np.unravel_index(k, (nC, nF, nT))
        r = dict(base)
        r.update(Status='no_sig', Component=0, N_components=0,
                 Peak_ms=float(times[pt]),
                 Peak_Hz='' if freqs is None else float(freqs[pf]),
                 Peak_channel=ch_names[pc], Peak_t=round(float(tl[k]), 4),
                 Peak_TFCE=round(float(tfl[k]), 4), Min_p=round(float(pl[k]), 6),
                 HedgesG=round(float(tl[k] / np.sqrt(n_sub) * J), 4),
                 Note='no cell with p < alpha; Peak_* = the max |TFCE| cell')
        rows.append(r)
    return rows


def _write_csv(path, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _write_npz(path, store):
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **store)
    os.replace(tmp, path)


# =============================================================================
# one run dir
# =============================================================================
def run_dir(rdir, measures, models, n_perm, dh, jobs, check_only=False,
            E=DEFAULT_E, H=DEFAULT_H, verbose=True, scheme='unweighted'):
    from heart_scalp_pipeline import build_channel_adjacency
    import mne
    mne.set_log_level('ERROR')
    main = os.path.join(rdir, 'main')
    z, times, freqs, ch_names, _models = load_run(main)
    clus = _read_cluster_csv(main)
    n_perm_cluster = int(rdir.rstrip('/\\').split('nperm')[-1])
    chan_adj, _ = build_channel_adjacency(ch_names)
    # the unweighted files keep their names; another scheme gets its own pair
    sfx = '' if scheme == 'unweighted' else '_' + scheme
    out_csv = os.path.join(main, f'l2_scalp_tfce{sfx}_summary.csv')
    out_npz = os.path.join(main, f'l2_scalp_tfce{sfx}.npz')
    # resume: keep maps already done at the same settings
    store, rows_by = {}, {}
    if os.path.isfile(out_npz) and not check_only:
        old = np.load(out_npz, allow_pickle=False)
        prm = json.loads(str(old['params']))
        if prm.get('nPerm') == n_perm and prm.get('dh') == dh:
            store = {k: old[k] for k in old.files}
            if os.path.isfile(out_csv):
                with open(out_csv, newline='', encoding='utf-8') as fh:
                    for r in csv.DictReader(fh):
                        rows_by.setdefault((r['Measure'], r['Model']), []).append(r)
        # release the handle: Windows refuses os.replace onto an open file, which
        # is what a rerun at a different nPerm then hits in _write_npz
        old.close()
    store.update(times=times, freqs=freqs, ch_names=np.asarray(ch_names),
                 params=np.asarray(json.dumps(dict(
                     nPerm=n_perm, dh=dh, E=E, H=H, seed=SEED, alpha=0.05,
                     scheme=scheme, graph='scalp_cell_edges (no row wrap)'))))
    checks = {}
    for model in models:
        for meas in measures:
            key = f'{meas}__{model}'
            if f'p__{key}' in store and (meas, model) in rows_by:
                if verbose:
                    print(f'  [{meas}/{model}] already done, skipped', flush=True)
                continue
            X = stack_subjects(z, meas, model)
            if X is None:
                continue
            if X.ndim == 3:
                X = X[:, None]                            # HEP -> [nC x 1 x nT x nSub]
            nC, nF, nT, nSub = X.shape
            if verbose:
                print(f'  [{meas}/{model}] {nC}ch x {nF}f x {nT}t, n={nSub}',
                      flush=True)
            t64, rep = check_against_cluster(X, clus[(meas, model)], ch_names,
                                             n_perm_cluster, verbose)
            checks[key] = rep
            if check_only:
                continue
            eu, ev = scalp_cell_edges(chan_adj, nF, nT)
            eu, ev = eu.astype(np.int32), ev.astype(np.int32)
            X2d = np.ascontiguousarray(X.reshape(nC * nF * nT, nSub))
            # small maps (HEP) run in-process: a worker's start-up (~50 s
            # under load) costs more than their whole null
            nj = jobs if nC * nF * nT > 50000 else 1
            w = None
            if scheme == IVW_OWN:
                w, _sids = own_ivw_weights(z, meas, model)
                assert w.size == nSub
                rep['n_eff_kish'] = kish_n_eff(w)
                rep['w_max_over_median'] = float(w.max() / np.median(w))
                if verbose:
                    print(f'    ivw_own: Kish n_eff {rep["n_eff_kish"]:.1f} of '
                          f'{nSub}, heaviest {rep["w_max_over_median"]:.2f}x '
                          f'median', flush=True)
            tf, t_w, nm, p, el = tfce_perm_graph(X2d, eu, ev, n_perm, SEED, E, H,
                                                 dh, jobs=nj, verbose=verbose, w=w)
            if w is None:
                rep['max_abs_t_perm_vs_check'] = float(np.max(np.abs(t_w - t64)))
            chk = tfce_graph(t_w, eu, ev, E, H, dh)['sn']
            assert np.max(np.abs(chk - tf)) < 1e-9 * max(1.0, np.abs(chk).max())
            t = t_w.reshape(nC, nF, nT)
            tf = tf.reshape(nC, nF, nT)
            p = p.reshape(nC, nF, nT)
            cr = clus[(meas, model)]
            cbest = min((float(r['perm_p']) for r in cr if r['perm_p']), default='')
            nsurv = sum(1 for r in cr if r['Survived'] == 'True')
            meta = dict(nPerm=n_perm, E=E, H=H, dh=dh, Cluster_Best_p=cbest,
                        Cluster_N_survived=nsurv)
            rows_by[(meas, model)] = readout_rows(
                meas, model, t, tf, p, eu, ev, times,
                None if meas == 'hep' else freqs, ch_names, nSub, meta)
            sq = (lambda a: a[:, 0]) if meas == 'hep' else (lambda a: a)
            store[f't__{key}'] = sq(t).astype(np.float32)
            store[f'tfce__{key}'] = sq(tf).astype(np.float32)
            store[f'p__{key}'] = sq(p).astype(np.float32)
            store[f'null_max__{key}'] = nm
            store[f'check__{key}'] = np.asarray(json.dumps(rep))
            allrows = [r for (ms, md) in [(a, b) for b in models for a in measures]
                       for r in rows_by.get((ms, md), [])]
            _write_csv(out_csv, allrows)
            _write_npz(out_npz, store)
            r0 = rows_by[(meas, model)][0]
            if verbose:
                print(f'    TFCE {el:.0f} s ({el / n_perm * 1000:.0f} ms/map wall): '
                      f'min p {r0["Best_p"]}, sig cells {r0["N_sig_total"]}, '
                      f'cluster best p {cbest}', flush=True)
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--measures', nargs='+', default=['hrsp', 'hep', 'hepc'])
    ap.add_argument('--models', nargs='+', default=['session', 'unadj'])
    ap.add_argument('--nperm', type=int, default=5000)
    ap.add_argument('--dh', type=float, default=DEFAULT_DH)
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--scheme', default='unweighted',
                    choices=['unweighted', IVW_OWN],
                    help='ivw_own needs a run that stores svarmean__ keys')
    a = ap.parse_args(argv)
    dirs = a.run_dirs
    for d in dirs:
        print(f'\n=== {os.path.basename(d.rstrip(os.sep))}', flush=True)
        t0 = time.time()
        run_dir(d, a.measures, a.models, a.nperm, a.dh, a.jobs, a.check_only,
                scheme=a.scheme)
        print(f'=== done in {time.time() - t0:.0f} s', flush=True)
    return 0
