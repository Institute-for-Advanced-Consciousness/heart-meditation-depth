# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ongoing band power of the selected component against reported depth.

For each analysed segment, the power of the selected component in a frequency band over the
whole segment (not locked to the heartbeat), z-scored within session, as the component
analysis computes it for its band-power regressors. Each participant's depth slope comes from
an ordinary least-squares fit with session indicators, the same design as the session model;
the group test is a one-sample t test of the slopes, with a sign-flip permutation p and
Hedges' g.

The cohort is rebuilt with the components the analysis selected, read from its session
records, so the band power is that of the analysed component.
"""
import math

import numpy as np
from scipy import stats

from heart_depth_run import preselect_from_records
from heart_depth_run import build_cohort


def _session_dummies(ses):
    levels = np.unique(ses)
    if levels.size < 2:
        return np.zeros((ses.size, 0))
    D = np.stack([(ses == v).astype(float) for v in levels[1:]], axis=1)
    return D - D.mean(axis=0, keepdims=True)


def ols_slope(y, X, col=1):
    """Coefficient `col` of an ordinary least-squares fit; constant columns are dropped."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    keep = [0] + [j for j in range(1, X.shape[1]) if np.nanstd(X[:, j]) > 1e-12]
    if col not in keep or y.size < len(keep) + 2:
        return np.nan
    b = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    return float(b[keep.index(col)])


def depth_slopes(data_dir, params, session_records, band='beta'):
    """{participant: depth slope of the selected component's ongoing power in `band`}."""
    q = dict(params, KEEP_SEGCOV=True, PIN_SELECTION=True,
             IC_PRESELECT=preselect_from_records(session_records))
    cohort, _records, _summary = build_cohort(data_dir, q, limit=None, verbose=False)
    slopes = {}
    for sid, sessions in cohort.items():
        y, d, ses = [], [], []
        for s in sessions:
            n = len(s['depth'])
            y.append(np.asarray((s.get('segcov') or {}).get(band, np.full(n, np.nan)), float))
            d.append(np.asarray(s['depth'], float))
            ses.append(np.full(n, int(s['session'])))
        y, d, ses = np.concatenate(y), np.concatenate(d), np.concatenate(ses)
        X = np.column_stack([np.ones(d.size), d - d.mean()] + list(_session_dummies(ses).T))
        slopes[sid] = ols_slope(y, X)
    return slopes


def group_test(slopes, n_perm=10000, seed=20260924):
    """One-sample test of the participant slopes: t, p, sign-flip p, Hedges' g."""
    v = np.asarray([x for x in slopes if np.isfinite(x)], float)
    n = v.size
    m, sd = v.mean(), v.std(ddof=1)
    t = m / (sd / math.sqrt(n))
    flips = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_perm, n))
    null = np.abs((flips * v).mean(axis=1))
    return dict(n=n, mean_slope=float(m), t=float(t),
                p=float(2 * stats.t.sf(abs(t), n - 1)),
                p_signflip=float((1 + np.sum(null >= abs(m))) / (n_perm + 1)),
                hedges_g=float(m / sd * (1 - 3 / (4 * (n - 1) - 1))),
                n_negative=int((v < 0).sum()))
