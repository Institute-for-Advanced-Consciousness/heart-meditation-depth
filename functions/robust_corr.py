# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
robust_corr - skipped correlations with a percentile bootstrap, ported from
the Robust Correlation Toolbox (Pernet, Wilcox & Rousselet, 2013, Front. Psychol.
3:606).

WHAT IS PORTED, FILE BY FILE
  bivariate_outliers.m   MCD centre (h = floor((n + 2p + 1) / 2)), every point
                         projected on the line joining each point to the centre,
                         flagged when its distance exceeds median + gval * (ideal
                         fourths range), gval = sqrt(chi2inv(.975, 2)).
  idealf.m               the ideal fourths.
  skipped_correlation.m  Pearson and Spearman on the kept points; t with the full
                         n and Wilcox's critical value c = 6.947 / n + 2.3197
                         (alpha .05, n 10..200), which allows for the removal;
                         percentile bootstrap of the kept pairs.
  skipped_Pearson.m      the bootstrap p: 2 * min(Q, 1 - Q), Q = P(r* < 0).
  HZmvntest.m            Henze-Zirkler test of bivariate normality.

THREE DEPARTURES, each deliberate
  1. Both variables are standardised (median, MAD) before the outlier search, as
     Wilcox's own `outpro` does by default. The toolbox searches in the units it
     is handed, so its outliers change when a questionnaire is rescored 0-4
     instead of 0-100. With MAD = 0 (a scale at floor) the SD is used.
  2. The MCD is scikit-learn's FAST-MCD (best of `n_starts` random starts), not
     LIBRA's mcdcov; the two differ in their small-sample corrections, so the
     centre is close to, not identical with, the toolbox's. When the MCD cannot be
     computed (more than half the points on a line) the coordinate-wise median
     is the centre and `mcd_ok` is False.
  3. 10,000 bootstrap resamples instead of 1,000 (599 in skipped_Pearson.m), so
     that p is resolved finely enough for a false-discovery correction.
     skipped_Pearson.m resamples all rows and then keeps the positions of the
     kept points, which puts outliers back into the resamples; the older
     skipped_correlation.m resamples the kept pairs, and that is what is done.

The bootstrap does not redo the outlier search inside each resample (neither
does the toolbox), so its p is conditional on the removal; Wilcox's h is the
test that allows for it.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

GVAL = float(np.sqrt(stats.chi2.ppf(0.975, 2)))


def idealf(x):
    """(lower, upper) ideal fourths - idealf.m."""
    y = np.sort(np.asarray(x, float))
    n = y.size
    j = int(np.floor(n / 4 + 5 / 12))
    g = n / 4 - j + 5 / 12
    ql = (1 - g) * y[j - 1] + g * y[j]
    k = n - j + 1
    qu = (1 - g) * y[k - 1] + g * y[k - 2]
    return ql, qu


def _standardise(v):
    v = np.asarray(v, float)
    mad = stats.median_abs_deviation(v, scale='normal')
    if mad > 0:
        return (v - np.median(v)) / mad
    sd = v.std(ddof=1)
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def mcd_center(X, n_starts=20, seed=0):
    """(centre, ok): reweighted MCD location with h = floor((n + 2p + 1) / 2)."""
    from sklearn.covariance import MinCovDet
    n, p = X.shape
    h = int(np.floor((n + 2 * p + 1) / 2))
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for s in range(n_starts):
            try:
                m = MinCovDet(support_fraction=h / n, random_state=seed + s).fit(X)
            except Exception:
                continue
            det = float(np.linalg.det(m.raw_covariance_))
            if np.isfinite(det) and det > 1e-12 and (best is None or det < best[0]):
                best = (det, m.location_.copy())
    if best is None:
        return np.median(X, axis=0), False
    return best[1], True


def bivariate_outliers(x, y, seed=0, n_starts=20):
    """(flag [n] bool, mcd_ok) - bivariate_outliers.m on standardised margins."""
    X = np.column_stack([_standardise(x), _standardise(y)])
    n = X.shape[0]
    center, ok = mcd_center(X, n_starts=n_starts, seed=seed)
    D = X - center
    flag = np.zeros(n, bool)
    for i in range(n):
        bot = float(D[i] @ D[i])
        if bot == 0:
            continue
        dis = np.abs(D @ D[i]) / np.sqrt(bot)       # norm(A'*B/bot.*B)
        ql, qu = idealf(dis)
        flag |= dis > np.median(dis) + GVAL * (qu - ql)
    return flag, ok


def _pearson(a, b):
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a * b).sum(-1) / den


def _boot(a, b, rank, n_boot, rng, alpha):
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    A, B = a[idx], b[idx]
    if rank:
        A, B = stats.rankdata(A, axis=1), stats.rankdata(B, axis=1)
    r = _pearson(A, B)
    r = np.sort(r[np.isfinite(r)])
    if r.size < 100:                    # a constant variable after the removal
        return (np.nan, np.nan), np.nan
    low = int(round(alpha * r.size / 2))
    q = float((r < 0).mean())
    return (float(r[low - 1]), float(r[r.size - low - 1])), max(2 * min(q, 1 - q), 1 / r.size)


def skipped_correlation(x, y, n_boot=10000, alpha=0.05, seed=0):
    """Skipped Pearson and Spearman of two vectors (finite pairs only).

    Returns a dict: n, n_out, outliers (bool over the finite pairs), mcd_ok, and
    for each of 'pearson' / 'spearman': r, t (full n), h_wilcox (|t| >= c),
    ci (percentile bootstrap), p_boot.
    """
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 10:
        raise ValueError('skipped correlations need at least 10 pairs')
    flag, mcd_ok = bivariate_outliers(x, y, seed=seed)
    a, b = x[~flag], y[~flag]
    c = 6.947 / n + 2.3197
    out = dict(n=n, n_out=int(flag.sum()), outliers=flag, mcd_ok=mcd_ok, crit=c)
    rng = np.random.default_rng(seed)
    for name, rank in (('pearson', False), ('spearman', True)):
        aa, bb = (stats.rankdata(a), stats.rankdata(b)) if rank else (a, b)
        r = float(_pearson(aa, bb))
        if not np.isfinite(r):
            out[name] = dict(r=np.nan, t=np.nan, h_wilcox=False, ci=(np.nan, np.nan),
                             p_boot=np.nan)
            continue
        t = r * np.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else np.inf * np.sign(r)
        ci, p = _boot(a, b, rank, n_boot, rng, alpha)
        out[name] = dict(r=r, t=float(t), h_wilcox=bool(abs(t) >= c), ci=ci, p_boot=float(p))
    return out


def hz_test(x, y):
    """(HZ, p) Henze-Zirkler test of bivariate normality - HZmvntest.m (cov / n)."""
    X = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n, p = X.shape
    S = np.cov(X, rowvar=False, bias=True)
    if np.linalg.matrix_rank(S) < p:
        return n * 4.0, 0.0
    Si = np.linalg.inv(S)
    d = X - X.mean(axis=0)
    Dj = np.einsum('ij,jk,ik->i', d, Si, d)
    Y = X @ Si @ X.T
    Djk = -2 * Y.T + np.diag(Y)[:, None] + np.diag(Y)[None, :]
    b = 1 / np.sqrt(2) * ((2 * p + 1) / 4) ** (1 / (p + 4)) * n ** (1 / (p + 4))
    HZ = n * (np.exp(-b ** 2 / 2 * Djk).sum() / n ** 2
              - 2 * (1 + b ** 2) ** (-p / 2) * np.exp(-b ** 2 / (2 * (1 + b ** 2)) * Dj).mean()
              + (1 + 2 * b ** 2) ** (-p / 2))
    wb = (1 + b ** 2) * (1 + 3 * b ** 2)
    a = 1 + 2 * b ** 2
    mu = 1 - a ** (-p / 2) * (1 + p * b ** 2 / a + p * (p + 2) * b ** 4 / (2 * a ** 2))
    si2 = (2 * (1 + 4 * b ** 2) ** (-p / 2)
           + 2 * a ** (-p) * (1 + 2 * p * b ** 4 / a ** 2 + 3 * p * (p + 2) * b ** 8 / (4 * a ** 4))
           - 4 * wb ** (-p / 2) * (1 + 3 * p * b ** 4 / (2 * wb) + p * (p + 2) * b ** 8 / (2 * wb ** 2)))
    pmu = np.log(np.sqrt(mu ** 4 / (si2 + mu ** 2)))
    psi = np.sqrt(np.log((si2 + mu ** 2) / mu ** 2))
    return float(HZ), float(stats.lognorm.sf(HZ, s=psi, scale=np.exp(pmu)))
