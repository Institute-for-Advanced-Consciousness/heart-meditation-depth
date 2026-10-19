# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Model-adjusted power at each reported depth level, from the participant-level fit.

For every participant and every frequency, the participant-level fit is repeated exactly as the
analysis ran it (heart_depth_pipeline.subject_slopes, with the components the run selected),
and the design columns and coefficients of each call to l1_slopes.cell_ols are kept. Per
heartbeat, the session and pre-R-peak baseline terms are removed,

    y_adj = y - sesD @ b_session - bl * b_baseline,

and y_adj is averaged within each depth level with the fit's Huber weights, recomputed from
the unweighted fit's residuals by cell_ols's own rule. With these weights the depth slope of a
participant's level means is that participant's depth coefficient; build() measures the
difference and the figure scripts check it.

The result is cached next to the run's run_results.npz as panel_adjusted_<model>.npz.
"""
import os

import numpy as np

import heart_depth_pipeline as HP
import heart_depth_run as R
import l1_slopes as L1
from heart_depth_run import params_of_run, preselect_from_records

KEYS = ('adj_hw', 'cnt', 'levels', 'beta_fit', 'err_hw', 'subject_ids', 'times', 'freqs')


def cache_path(main_dir, model):
    return os.path.join(main_dir, 'panel_adjusted_%s.npz' % model)


def build(main_dir, data_dir, model='session', measure='hrsp', verbose=True):
    """Adjusted level means of one finished run (<run>/<grain>_wise/main).

    Returns adj_hw [participant, level, frequency, time] (uncentred), cnt [participant,
    level] (heartbeats per level), levels, beta_fit (the depth coefficients of the repeated
    fit, which must equal the stored ones), err_hw (per participant, the largest difference
    between the Huber-weighted slope of y_adj and the fitted slope), subject_ids, times and
    freqs.
    """
    p, grain, recs, _meas = params_of_run(main_dir)
    q = dict(p, IC_PRESELECT=preselect_from_records(recs), PIN_SELECTION=True,
             KEEP_PERBEAT=True)
    want_ic = {(str(r['subject']), int(r['session'])): int(r['ic'])
               for r in recs if (r.get('status') or '') == 'kept' and r.get('ic') is not None}
    all_sessions = R.list_sessions(data_dir)
    with np.load(os.path.join(main_dir, 'run_results.npz'), allow_pickle=False) as z:
        beta_run = np.asarray(z['L1_%s_%s_beta' % (model, measure)], float)   # [sub, f, t]
        sids = [str(s) for s in z['subject_ids']]
        times, freqs = np.asarray(z['times'], float), np.asarray(z['freqs'], float)
    levels = np.array(sorted(int(v) for v in p['DEPTH_LEVELS']))
    nS, nF, nT = beta_run.shape
    adj_hw = np.full((nS, levels.size, nF, nT), np.nan)
    cnt = np.zeros((nS, levels.size), np.int64)
    beta_fit = np.full((nS, nF, nT), np.nan)
    err_hw = np.zeros(nS)
    orig_list, orig_ols = R.list_sessions, HP.cell_ols
    try:
        for si, sid in enumerate(sids):
            mine = [s for s in all_sessions if str(s[0]) == sid]
            R.list_sessions = lambda _d, _m=mine: list(_m)
            sbs, _rec, _sub = R.build_cohort(data_dir, q, limit=None, verbose=False)
            sess = sbs.get(sid) or sbs.get(mine[0][0])
            if not sess:
                raise RuntimeError('%s: the rebuild kept no recording' % sid)
            for s in sess:
                k = (sid, int(s['session']))
                if int(s.get('ic', -1)) != want_ic[k]:
                    raise RuntimeError('%s session %d: rebuilt component %s, the run '
                                       'analysed %s' % (sid, k[1], s.get('ic'), want_ic[k]))
            calls = []

            def record(Y, dep, sesD=None, covC=None, ok=None, method='OLS', cluster=None,
                       target_col=1, _calls=calls):
                out = orig_ols(Y, dep, sesD=sesD, covC=covC, ok=ok, method=method,
                               cluster=cluster, target_col=target_col)
                # the same call unweighted: the fit the Huber weights are derived from
                ols = orig_ols(Y, dep, sesD=sesD, covC=covC, ok=ok, method='OLS',
                               cluster=None, target_col=target_col)
                _calls.append(dict(Y=Y, dep=np.asarray(dep, float), sesD=sesD, covC=covC,
                                   ok=ok, B=out[2], B_ols=ols[2], method=method,
                                   target_col=target_col))
                return out
            HP.cell_ols = record
            try:
                beta, _svar, _nu = HP.subject_slopes(sess, measure, q, model=model,
                                                     grain=grain)
            finally:
                HP.cell_ols = orig_ols
            if beta is None or len(calls) != nF:
                raise RuntimeError('%s: %d participant-level fits, expected %d'
                                   % (sid, len(calls), nF))
            beta_fit[si] = beta
            # depth level of every heartbeat, in the order the fit saw them
            dl = np.concatenate([HP._beats_of_session(s, measure, q)[1] for s in sess
                                 if HP._beats_of_session(s, measure, q)[0].size]).astype(int)
            cnt[si] = [(dl == lv).sum() for lv in levels]
            for f, c in enumerate(calls):
                assert c['target_col'] == 1 and str(c['method']).upper() == 'WLS'
                assert c['ok'] is None or np.asarray(c['ok'], bool).all()
                assert np.allclose(c['dep'], dl - dl.mean(), rtol=0, atol=1e-12)
                Y = np.asarray(c['Y'], float)
                cols = [np.asarray(x, float) for x in (c['sesD'], c['covC'])
                        if x is not None and np.size(x)]
                Bf, Bo = np.asarray(c['B'], float), np.asarray(c['B_ols'], float)
                assert Bf.shape == (2 + sum(x.shape[1] for x in cols), nT)
                Ya = Y - np.hstack(cols) @ Bf[2:] if cols else Y.copy()
                # Huber weights of cell_ols: w = min(1, 1.345 median|r| / |r|), with |r| the
                # residual norm of a beat across the time points of this frequency
                dc = c['dep'] - c['dep'].mean()
                Xd = np.hstack([np.ones((Y.shape[0], 1)), dc[:, None]] + cols)
                rn = np.linalg.norm(Y - Xd @ Bo, axis=1)
                hw = np.minimum(1.0, 1.345 * np.median(rn) / np.maximum(rn, L1.EPS))
                dw = dc - (hw * dc).sum() / hw.sum()
                s_w = ((hw * dw)[:, None] * Ya).sum(0) / (hw * dw ** 2).sum()
                err_hw[si] = max(err_hw[si], float(np.max(np.abs(s_w - Bf[1]))))
                for li, lv in enumerate(levels):
                    m = dl == lv
                    if m.any():
                        adj_hw[si, li, f] = (hw[m, None] * Ya[m]).sum(0) / hw[m].sum()
            if verbose:
                print('  participant %d of %d: %d heartbeats, levels %s, largest |slope - '
                      'stored| %.1e, weighted-slope error %.1e'
                      % (si + 1, nS, cnt[si].sum(), levels[cnt[si] > 0].tolist(),
                         float(np.nanmax(np.abs(beta - beta_run[si]))), err_hw[si]),
                      flush=True)
    finally:
        R.list_sessions, HP.cell_ols = orig_list, orig_ols
    return dict(adj_hw=adj_hw, cnt=cnt, levels=levels, beta_fit=beta_fit, err_hw=err_hw,
                subject_ids=np.array(sids), times=times, freqs=freqs)


def load(main_dir, data_dir, model='session', measure='hrsp', rebuild=False):
    """The cached adjusted level means of a run, built and cached when absent or older than
    the run (a cohort rebuild, one participant at a time; about ten minutes)."""
    fp = cache_path(main_dir, model)
    run_file = os.path.join(main_dir, 'run_results.npz')
    if os.path.isfile(fp) and not rebuild and os.path.getmtime(fp) >= os.path.getmtime(run_file):
        with np.load(fp, allow_pickle=False) as z:
            if all(k in z.files for k in KEYS):
                return {k: z[k] for k in KEYS}
    print('building the model-adjusted level means (one participant at a time)...', flush=True)
    out = build(main_dir, data_dir, model=model, measure=measure)
    tmp = fp[:-len('.npz')] + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, fp)
    return out


def centre_levels(A, have):
    """Centre A [participant, level, ...] within participant over the levels they have."""
    out = np.full_like(A, np.nan)
    for si in range(A.shape[0]):
        h = have[si]
        out[si, h] = A[si, h] - A[si, h].mean(axis=0, keepdims=True)
    return out


def level_stats(S, have):
    """Mean, standard error and n over the participants who have each level, S [participant,
    level, ...]."""
    mu, se, nn = [], [], []
    for li in range(S.shape[1]):
        v = S[have[:, li], li]
        nn.append(v.shape[0])
        mu.append(v.mean(axis=0))
        se.append(v.std(axis=0, ddof=1) / np.sqrt(v.shape[0]))
    return np.array(mu), np.array(se), np.array(nn)
