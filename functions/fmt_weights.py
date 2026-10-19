# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""FMT depth effect with each participant weighted by the precision of their own slope.

The FMT depth run (fmt_depth_npz.py) stores each participant's depth slope and its
variance for every measure and model. This module adds the precision-weighted test of the
paper (`ivw_own`: shrunk inverse of the variance of the tested slope) beside the run's own
schemes. FMT is one number per participant, so the test is the weighted sign-flip t-test of
fmt_depth_pipeline, not TFCE. Nothing is refitted.

Before adding a row, the run's own unweighted, precision_nseg and ivw_shrunk rows of
fmt_depth_L2.csv are recomputed from the stored slopes; a mismatch stops the function.
"""
import csv
import io
import json
import os

import numpy as np

from fmt_depth_pipeline import shrunk_ivw, signflip_ttest

MEASURES = (('FMT_scalp', 'SCALP_BETA', 'SCALP_RV'), ('FMT_GED', 'GED_BETA', 'GED_RV'))
FIELDS = ('Cell', 'Measure', 'Model', 'Scheme', 'n', 'N_eff_kish', 't', 'p', 'CohenD',
          'Mean_slope')


def _test(betas, w, nperm):
    ok = np.isfinite(betas)
    t, p = signflip_ttest(betas[ok], np.asarray(w, float)[ok], nperm)
    wn = np.asarray(w, float)[ok]
    wn = wn / wn.mean()
    return float(t), float(p), int(ok.sum()), float(wn.sum() ** 2 / (wn ** 2).sum())


def weighted_rows(run_dir, verbose=True):
    """Rows of the weighted tests for one FMT depth run folder (the one holding main/)."""
    main_dir = os.path.join(run_dir, 'main')
    z = np.load(os.path.join(main_dir, 'run_results.npz'), allow_pickle=True)
    ref = list(csv.DictReader(io.open(os.path.join(main_dir, 'fmt_depth_L2.csv'),
                                      encoding='utf-8-sig')))
    js = json.load(io.open(os.path.join(main_dir, 'fmt_results.json'), encoding='utf-8'))
    nperm = None
    for k in ('nPerm', 'n_perm'):
        for src in (js, js.get('params', {}), js.get('P', {})):
            if isinstance(src, dict) and k in src:
                nperm = int(src[k])
    nseg = np.asarray(z['NSEG_SUB'], float)
    w_shr, _ = shrunk_ivw(np.asarray(z['SCALP_RV'], float)[0, :], 'median')
    cell = os.path.basename(os.path.normpath(run_dir))
    rows = []
    for mname, bk, vk in MEASURES:
        B, V = np.asarray(z[bk], float), np.asarray(z[vk], float)
        for model in dict.fromkeys(r['Model'] for r in ref if r['Measure'] == mname):
            want = {r['Scheme']: r for r in ref
                    if r['Measure'] == mname and r['Model'] == model}
            # the stored row of this model, and the permutation count: whichever
            # reproduces the run's own unweighted row
            hit = None
            for nper in ([nperm] if nperm else [1000, 5000, 10000]):
                for i in range(B.shape[0]):
                    if not np.isfinite(B[i]).any():
                        continue
                    t, p, n, _ne = _test(B[i], np.ones(B.shape[1]), nper)
                    if (abs(t - float(want['unweighted']['t'])) < 1e-9
                            and abs(p - float(want['unweighted']['p'])) < 1e-12):
                        hit = (i, nper)
            if hit is None:
                raise RuntimeError('%s %s/%s: cannot reproduce the unweighted row'
                                   % (cell, mname, model))
            i, nper = hit
            for sch, w in (('precision_nseg', nseg), ('ivw_shrunk', w_shr)):
                if sch in want:
                    t, p, _n, _ne = _test(B[i], w, nper)
                    if (abs(t - float(want[sch]['t'])) > 1e-9
                            or abs(p - float(want[sch]['p'])) > 1e-12):
                        raise RuntimeError('%s %s/%s/%s: %.4f/%.4f vs stored %s/%s'
                                           % (cell, mname, model, sch, t, p,
                                              want[sch]['t'], want[sch]['p']))
            w_own, info = shrunk_ivw(V[i], 'median')
            if info['nUsed'] == 0:
                raise RuntimeError('%s %s/%s: no usable slope variance' % (cell, mname, model))
            ok = np.isfinite(B[i])
            d_val = float(np.nanmean(B[i][ok]) / np.nanstd(B[i][ok]))
            for sch, w in (('unweighted', np.ones(B.shape[1])), ('precision_nseg', nseg),
                           ('ivw_shrunk', w_shr), ('ivw_own', w_own)):
                t, p, n, ne = _test(B[i], w, nper)
                wn = np.asarray(w, float)[ok]
                rows.append(dict(Cell=cell, Measure=mname, Model=model, Scheme=sch, n=n,
                                 N_eff_kish=round(ne, 2), t=round(t, 4), p=round(p, 4),
                                 CohenD=round(d_val, 4),
                                 Mean_slope=float(np.sum(B[i][ok] * wn) / wn.sum())))
                if verbose:
                    print('%-64s %-9s %-12s %-14s n %d (eff %.1f) t %+.3f p %.4f d %+.3f'
                          % (cell, mname, model, sch, n, ne, t, p, d_val), flush=True)
    return rows


def write_table(run_dirs, out_csv, verbose=True):
    """Weighted tests of several FMT depth runs, in one table."""
    rows = []
    for d in run_dirs:
        rows += weighted_rows(d, verbose=verbose)
    with io.open(out_csv, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print('wrote', out_csv)
    return rows
