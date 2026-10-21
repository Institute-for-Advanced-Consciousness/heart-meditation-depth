# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
l1_slopes: the shared stage-1 solver.

Fits per-subject depth slopes on [1, depth_centred, sessionDummies, covariates]. It is
the solver of the cell grain (units = session x depth cells), the segment grain (units =
depth-report segments) and the per-beat grain, where the caller passes `cluster` so the
slope variance is cluster-robust (beats cluster within segments).

Three things that look unusual and are deliberate:

  1. `svar = MSE * iXtX[1,1]`, the target coefficient's own diagonal element.
  2. Two ridge constants: 1e-8*trace for the OLS/initial solve and 1e-6*trace for
     the Huber-weighted solve. The solve is on unit-norm columns: `lam*trace` is a
     fraction of the largest column, so on raw columns it would delete any column far
     below O(1), such as the baseline column (see the `xsc` block).
  3. The refusal is a pure count of units against design columns. No rank check,
     no condition number, no pinv. The ridge means a singular design returns
     finite numbers, so the df count is the only path to NaN.

The identification boundary. The fit is refused only when nUnits < nP
(under-determined). At nUnits == nP the system is exactly identified: the slope is
estimable, so it is returned with svar = Inf. An exactly identified subject therefore
enters the unweighted group mean while carrying zero weight in IVW / precision-weighted
maps, so `unweighted`, `precision_nseg` and `ivw_shrunk` results in the same folder can
rest on different effective samples. If an effect appears under one weighting only,
compare N_subjects and N_eff_weighted across all three before calling it
weighting-robust.

The cell-grain df trap. Under a GLM baseline, HEPC (as the code names it; the paper's
HRPC, heartbeat-related phase consistency) never gets the Alday baseline column (unbiased
PPC over unit-length phasors has no amplitude scale to adjust), so
nP_hepc = nP_hep - 1 exactly. HEP/HRSP is therefore refused one unit sooner than HEPC in
the same folder. Confirm two measures were fit on the same n before comparing them.
Segment and per-beat grains are not affected (many more units per subject).
"""
from __future__ import annotations
import numpy as np

EPS = np.finfo(float).eps
__all__ = ['cell_ols', 'FitInfo']


class FitInfo:
    """Fit bookkeeping per model: nUnits, nP, df, ok, the refusal reason, and
    whether the fit is exactly identified. Read it: a refused cell-grain fit is
    otherwise invisible.
    """
    __slots__ = ('n_units', 'n_pred', 'df', 'ok', 'reason', 'exact')

    def __init__(self, n_units, n_pred, df, ok, reason='', exact=False):
        self.n_units, self.n_pred, self.df = n_units, n_pred, df
        self.ok, self.reason, self.exact = ok, reason, exact

    def __repr__(self):
        return (f'FitInfo(n_units={self.n_units}, nP={self.n_pred}, df={self.df}, '
                f'ok={self.ok}, exact={self.exact}'
                + (f', reason={self.reason!r}' if self.reason else '') + ')')


def _check_target_col(target_col, nP):
    """Refuse an out-of-range coefficient index by name.

    A design that loses a column (a degenerate covariate dropped upstream)
    would otherwise make target_col point past the end, and numpy would
    raise deep inside the solve with an index message that says nothing
    about models. Worse, a negative index would silently wrap and return
    some other coefficient as if it were the requested one.
    """
    tc = int(target_col)
    if not (0 <= tc < nP):
        raise IndexError(
            f'target_col={tc} is outside the design, which has {nP} column(s). '
            'The caller asked for a coefficient this model does not fit -- most '
            'likely a covariate column was dropped as degenerate and the '
            'interaction index was not recomputed.')
    return tc


def _crve_depth(X, resid, w, iXtWX, cluster, col=1):
    """Cluster-robust variance of the depth coefficient, one value per column.

    The CRVE with the Cameron & Miller (2015) small-sample factor. Only element
    [1,1] of

        V_j = iXtWX @ M_j @ iXtWX ,   M_j = sum_g (X_g' q_gj)(X_g' q_gj)'

    with q_ij = w_i * r_ij is used, so with a = iXtWX[:, 1]:

        V_j[1,1] = a' M_j a = sum_g ( (X_g a)' q_gj )^2 = sum_g ( sum_{i in g} u_i q_ij )^2

    with u = X @ a. That is one [n x P] elementwise product and one segment-sum,
    O(n*P) instead of O(nClusters * nP^2 * P), and it is exact (iXtWX is
    symmetric, so no transpose is being glossed over).

    Why the per-beat grain needs it: there the units are beats, and beats inside
    one depth-report segment are not independent; they share the segment's state,
    its artefacts and its depth label. An OLS variance over ~1200 correlated beats
    understates the standard error by roughly the square root of the beats per
    segment. The cluster is the segment, which is why `cluster` here is the
    segment id.

    `cluster` : [nUnits] integer group ids, already restricted to the used
                units. Returns None when there is nothing to cluster on, so the
                caller keeps the model-based variance rather than a silently
                degenerate robust one.
    """
    g = np.asarray(cluster).ravel()
    finite = np.isfinite(g.astype(float)) if g.dtype.kind == 'f' else np.ones(g.size, bool)
    uniq = np.unique(g[finite])
    n_g = uniq.size
    if n_g < 2:
        return None
    a = iXtWX[:, col]
    u = (X @ a) * w                                   # [n]
    q = resid * u[:, None]                            # [n x P]
    # sum within cluster: index once, add.at is the only scatter numpy offers
    idx = np.searchsorted(uniq, g)
    S = np.zeros((n_g, q.shape[1]), float)
    np.add.at(S, idx, q)
    V = (S ** 2).sum(axis=0)
    n, nP = X.shape
    # Cameron & Miller (2015) small-sample factor.
    c = (n_g / (n_g - 1.0)) * ((n - 1.0) / (n - nP))
    return V * c


def cell_ols(Yunits, dep, sesD=None, covC=None, ok=None, method='OLS',
             cluster=None, target_col=1):
    """Stage-1 depth-slope solve.

    Yunits : [nUnits x P]  measure at P grid points (time, or freq*time flattened)
    dep    : [nUnits]      depth per unit
    sesD   : [nUnits x k]  centred session blocking dummies (or None)
    covC   : [nUnits x c]  centred covariate columns (or None)
    ok     : [nUnits] bool units usable for this measure (default: all)
    method : 'OLS' (default) or 'WLS' (Huber). The pipeline passes L1_METHOD.
    cluster: [nUnits] group ids for a cluster-robust variance (or None).
             Used by the per-beat grain, where the units are beats and the
             cluster is the depth-report segment they came from. When given and
             usable, `svar` is the CRVE; otherwise it is the model-based
             MSE * iXtWX[1,1]. `slope` never moves: clustering changes the
             variance, not the point estimate.

    target_col : which coefficient of the design comes back as (slope, svar).
             Default 1 = the depth slope, which is column 1 of
             X = [1, depth, sesD..., covC...]. A moderation model passes the
             index of its depth x covariate column instead, because the quantity
             being tested there is 'does the depth slope change with the
             covariate', not 'what is the depth slope'.
             The variance follows the coefficient: svar reads
             iXtWX[target_col, target_col] and the CRVE is taken on the same
             column. Returning B[k] beside the depth column's variance would give
             a t-map with the right numerator and the wrong denominator, which no
             downstream check could catch.

    Returns (slope[P], svar[P], Bfull[nP x P] or None, FitInfo).
    svar is Inf at exact identification; slope/svar are NaN where the measure is
    non-finite for any used unit, and all-NaN when the fit is refused.
    """
    Y_all = np.asarray(Yunits, float)
    if Y_all.ndim == 1:
        Y_all = Y_all[:, None]
    dep = np.asarray(dep, float).ravel()
    n_all, P = Y_all.shape
    ok = np.ones(n_all, bool) if ok is None else np.asarray(ok, bool).ravel()

    slope = np.full(P, np.nan)
    svar = np.full(P, np.nan)

    d = dep[ok]
    # need >= 3 units and >= 2 distinct depths
    if d.size < 3 or np.unique(d).size < 2:
        return slope, svar, None, FitInfo(d.size, 0, np.nan, False,
                                          'need >=3 units and >=2 depths')

    dc = d - d.mean()                                   # depth centred
    cols = [np.ones((d.size, 1)), dc[:, None]]
    if sesD is not None and np.size(sesD):
        cols.append(np.asarray(sesD, float)[ok])
    if covC is not None and np.size(covC):
        cols.append(np.asarray(covC, float)[ok])
    X = np.hstack(cols)
    nP = X.shape[1]
    # -- Column scaling: without it the ridge below deletes small columns -----
    # `_ridge_inv` adds `lam * trace(A)` to every diagonal entry, so the ridge is only
    # "tiny" relative to the largest column in the design. A column whose scale sits far
    # below the O(1) intercept / depth / session columns would be annihilated by it. The
    # Alday baseline column is such a column on sensor data: pre-R-peak voltage for HEP
    # (~1e-7 V) and pre-R-peak power for HRSP (~1e-13 V^2), so its own X'X diagonal
    # entry (~n*scale^2) would fall 8 and 20 orders of magnitude below the WLS ridge
    # (1e-6*trace ~ 1e-5) and its coefficient would be shrunk to zero. Large columns are
    # the opposite case: `rr` (ms) and `respAmp` (raw respiration envelope) inflate
    # trace(X'X) and raise the ridge on the depth column. Covariates z-scored within
    # session (ecgAmp/ecgAmpT) or bounded (respPhase sin/cos) are O(1).
    #
    # Solving on unit-norm columns makes the ridge the uniform `lam * nP` relative
    # regulariser it is meant to be, and keeps its actual job (a singular design still
    # inverts). Rescaling a design column is exact in real arithmetic (it changes that
    # column's own coefficient scale and nothing else), so `B` and `svar` are mapped
    # straight back below, and a well-scaled design (every column O(1)) is unchanged.
    xsc = np.sqrt(np.sum(X * X, axis=0))
    xsc[~(xsc > 0)] = 1.0        # an all-zero column has no scale; leave it dead
    Xn = X / xsc                 # solve on this; never on X

    # Refuse only when under-determined (see the module docstring).
    if d.size < nP:
        return slope, svar, None, FitInfo(d.size, nP, d.size - nP, False,
                                          'under-determined (nUnits < nP)')
    exact = (d.size == nP)                              # 0 residual df

    Y = Y_all[ok]
    finite_all = np.all(np.isfinite(Y), axis=0)
    Yw = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    def _ridge_inv(A, lam):
        return np.linalg.solve(A + lam * np.trace(A) * np.eye(A.shape[0]),
                               np.eye(A.shape[0]))

    _tc = _check_target_col(target_col, nP)

    # Every solve below is on `Xn` (unit-norm columns), never on `X`. The
    # coefficients and the target variance come back to the caller's units in
    # the un-scaling step after the branches.
    if method.upper() == 'WLS' and not exact:
        # Huber: OLS fit -> residuals -> down-weight outlier units -> refit.
        XtX = Xn.T @ Xn
        B_ols = _ridge_inv(XtX, 1e-8) @ (Xn.T @ Yw)
        resid = Yw - Xn @ B_ols
        resid_norm = np.linalg.norm(resid, axis=1)      # per-unit, across P
        k_hub = 1.345 * np.median(resid_norm)
        if k_hub < EPS:
            k_hub = 1.0
        w = np.minimum(1.0, k_hub / np.maximum(resid_norm, EPS))
        XtWX = Xn.T @ (w[:, None] * Xn)
        iXtWX = _ridge_inv(XtWX, 1e-6)                  # note: 1e-6, not 1e-8
        B = iXtWX @ (Xn.T @ (w[:, None] * Yw))
        df = d.size - nP
        MSE = np.sum(((Yw - Xn @ B) ** 2) * w[:, None], axis=0) / df
        svar = MSE * iXtWX[_tc, _tc]
        if cluster is not None:
            v = _crve_depth(Xn, Yw - Xn @ B, w, iXtWX, np.asarray(cluster)[ok],
                            col=_tc)
            if v is not None:
                svar = v
    else:
        XtX = Xn.T @ Xn
        iXtX = _ridge_inv(XtX, 1e-8)
        B = iXtX @ (Xn.T @ Yw)
        df = d.size - nP
        if exact:
            # Estimable, but no residual df -> infinite variance. The subject
            # enters the unweighted mean and carries zero weight in IVW.
            svar = np.full(P, np.inf)
        else:
            MSE = np.sum((Yw - Xn @ B) ** 2, axis=0) / df
            svar = MSE * iXtX[_tc, _tc]
            if cluster is not None:
                v = _crve_depth(Xn, Yw - Xn @ B, np.ones(d.size),
                                iXtX, np.asarray(cluster)[ok], col=_tc)
                if v is not None:
                    svar = v

    # -- Un-scale: back to the caller's units ---------------------------------
    # X = Xn @ diag(xsc), so B_j = Bn_j / xsc_j exactly, and the variance of the
    # target coefficient carries its own column's square. This holds for the
    # model-based svar (MSE * iXtWX[tc,tc]) and for the CRVE alike: the CRVE's
    # `a = iXtWX[:,tc]` and `u = Xn @ a` both scale as 1/xsc[tc], so its V does
    # too, squared. `np.inf / x**2` stays Inf, so the exact-identification
    # contract is untouched.
    B = B / xsc[:, None]
    slope = B[_tc].copy()
    svar = svar / (xsc[_tc] ** 2)

    slope[~finite_all] = np.nan
    svar[~finite_all] = np.nan
    B = B.copy()
    B[:, ~finite_all] = np.nan
    return slope, svar, B, FitInfo(d.size, nP, d.size - nP, True, '', exact)
