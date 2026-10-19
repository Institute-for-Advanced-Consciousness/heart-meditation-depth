# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Threshold-free cluster enhancement (TFCE) as the group (L2) test of the heartbeat-locked depth
analysis, with the per-subject weighting schemes of the cluster test (unweighted /
precision_nseg / ivw_shrunk), plus ivw_own (the tested measure's own shrunk inverse slope
variance, estimated in prepare_measure from the subject list it is given).

    per-subject slope maps (listwise-kept, as l2_heart does)
      -> weighted sign-flip t-maps, observed + null in one matmul
         (l2_heart._make_flips + _null_t_weighted, imported)
      -> TFCE of every map (tfce_scores / the batched core below)
      -> null = max |TFCE| over the map for flips 1..n_perm-1
      -> per-cell FWE p = (1 + #(null_max >= |TFCE_obs|)) / n_perm
      -> readout: connected components of p < alpha, per polarity

Nothing statistical is re-derived here that exists elsewhere: the weights
(build_weights, build_ivw_weights -> fmt_depth_pipeline.shrunk_ivw), the
listwise keep rule, the weighted t, the flip matrix, Kish n_eff, Hedges' J,
the CFA window and the cluster labeller used for the readout are all imported
from l2_heart / l2_run.

==============================================================================
TWO TFCE CONVENTIONS, AND WHY BOTH ARE HERE
==============================================================================
TFCE (Smith & Nichols 2009) of a cell p with statistic t_p > 0:

    TFCE(p) = integral_0^{t_p} e(h)^E h^H dh
            ~ sum_k  e(h_k)^E * h_k^H * dh,     h_k = k*dh, h_k < t_p

where e(h) is the size (cell count) of the connected component of {t > h}
that contains p. Negative cells get the same on -t, with a negative sign.

MNE-Python's implementation (mne/stats/cluster_level.py, _find_clusters, with
threshold=dict(start=0, step=dh), checked on mne 1.11.0) sums instead

    sum_k  e(h_k)^E * (h_k - h_{k-1})^H          (and 0 at k = 0)

i.e. it raises the step to the power H, not the height. With a constant step that is
dh^H * sum_k e(h_k)^E: the height weighting is gone, and H only rescales every score by the same
constant. MNE's TFCE is therefore TFCE with H = 0 in all but name, whatever h_power says. For
example, an isolated cell of height x scores 0.09 / 0.19 / 0.39 at x = 1 / 2 / 4 (dh = .1) under
MNE (linear in x), against x^3/3 = 0.33 / 2.67 / 21.3 under Smith & Nichols.

Both are valid FWE procedures (any statistic tested against its own max null is), but they are
different statistics: the MNE form rewards extent at low heights much more, the S&N form
rewards height. So:

  convention='sn'   Smith & Nichols (the definition, and what E / H mean).
                    Primary (the default). Same threshold grid, same strict x > h, same
                    4-connectivity as the MNE form; only the per-level weight
                    differs (h_k^H * dh instead of (h_k - h_{k-1})^H).
  convention='mne'  MNE's sum, reproduced to 1e-12 (checked on mne 1.11.0). Kept for
                    comparison with MNE's TFCE.

Both are computed from ONE set of component labels per threshold, so asking
for both costs almost nothing extra.

MNE's 1-D path also ignores extent altogether. For a 1-D map with adjacency=None,
_find_clusters_1dir labels with ndimage.find_objects, which returns one tuple of one slice per
cluster, and the TFCE loop takes len(tuple) = 1 as the cluster size. Every 1-D cluster therefore
has extent 1: a plateau of height 2 scores 0.19 at width 1 and at width 9 on the 1-D path, and
0.57 at width 9 when the same map is passed as [1 x nT]. So MNE's 1-D TFCE
(`mne.stats` TFCE on a 1-D HEP map) is a pointwise max-|t|-like test, not TFCE. Here HEP is always
a [1 x nT] map, and the 'mne' convention reproduces MNE's 2-D path on that map, never its 1-D
one.

==============================================================================
CONNECTIVITY AND THE GRID
==============================================================================
  * 4-connectivity in freq x time: l2_run.CONN, the cross structuring
    element that _clusters uses, and also what scipy.ndimage.label defaults to
    in 2-D (and what MNE uses for a lattice when adjacency=None). HEP is a
    [1 x nT] map, so its neighbours are the adjacent time points.
  * Positive and negative tails are labelled separately (a positive and a
    negative patch that touch are never one component), as _clusters does.
  * Thresholds: np.arange(0, max|t|, dh), cells strictly above each threshold,
    exactly MNE's grid. A shared grid across a batch of maps gives each map the
    same prefix of that arange (numpy fills start + i*step), and levels at or
    above a map's own max contain none of its cells, so they add nothing.

==============================================================================
THE PERMUTATION p
==============================================================================
Same convention as l2_heart.cluster_perm_weighted: column 0 of the flip
matrix is the identity and supplies the observed map; the null is columns
1..n_perm-1 (n_perm - 1 members); p = (1 + #null >= obs) / n_perm, floor
1/n_perm (2e-4 at 5000). MNE's permutation_cluster_1samp_test draws n_perm-1
random flips and appends the observed statistic to H0, which is the same p.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Imported, not copied (see the module docstring).
from l2_run import CONN, _clusters                                  # noqa: E402
from l2_heart import (_make_flips, _null_t_weighted, _listwise_keep, _stack_measure, build_weights,
                      kish_n_eff, _hedges_J, _check_memory, cfa_overlap, SEED, MAX_NULL_GB,
                      build_own_ivw_weights, IVW_OWN)

__all__ = ['tfce_scores', 'tfce_perm_weighted', 'tfce_perm_weighted_multi',
           'tfce_graph', 'scalp_cell_edges', 'graph_edges', 'graph_components',
           'tfce_components', 'CONVENTIONS', 'TFCE_FIELDS']

CONVENTIONS = ('sn', 'mne')
DEFAULT_E, DEFAULT_H, DEFAULT_DH = 0.5, 2.0, 0.1


# =============================================================================
# THE TFCE CORE
# =============================================================================
def _level_weights(thresholds, H, dh, convention):
    """Per-threshold weight multiplying e(h_k)^E. See the module header."""
    thr = np.asarray(thresholds, float)
    if convention == 'sn':
        return (np.abs(thr) ** H) * dh
    if convention == 'mne':
        # verbatim MNE: h = |thresh| at k = 0, else |thresh_k - thresh_{k-1}|,
        # then h ** h_power
        h = np.empty(thr.size)
        if thr.size:
            h[0] = abs(thr[0])
            h[1:] = np.abs(np.diff(thr))
        return h ** H
    raise ValueError(f'unknown TFCE convention {convention!r}; known {CONVENTIONS}')


def _tfce_batch(T3, E, H, dh, conventions):
    """Unsigned TFCE of B maps at once, both tails, every convention.

    T3 : [B x nF x nT] statistic maps (finite).
    Returns {conv: (S_pos [B x nF x nT], S_neg [B x nF x nT])}, both >= 0.

    One ndimage.label call per threshold for the whole batch. The 2B maps (B positive, B negated)
    are stacked along the freq axis with a -inf separator row after each map; the cross
    structuring element cannot reach across a separator row, so no component can span two maps
    or two tails.
    """
    T3 = np.asarray(T3, float)
    B, nF, nT = T3.shape
    R = nF + 1
    P = np.full((2 * B, R, nT), -np.inf)
    P[:B, :nF] = T3
    P[B:, :nF] = -T3
    P2 = P.reshape(2 * B * R, nT)
    flat = P2.ravel()
    stop = float(np.max(np.abs(T3))) if T3.size else 0.0
    thresholds = np.arange(0.0, stop, dh, dtype=float)
    wts = {c: _level_weights(thresholds, H, dh, c) for c in conventions}
    S = {c: np.zeros(flat.size) for c in conventions}
    active = np.flatnonzero(flat > 0.0) if thresholds.size else np.zeros(0, int)
    mask = np.zeros(flat.size, bool)
    for k, h in enumerate(thresholds):
        active = active[flat[active] > h]           # strict, as MNE: x > thresh
        if active.size == 0:
            break
        mask[:] = False
        mask[active] = True
        lab, n = ndimage.label(mask.reshape(P2.shape), structure=CONN)
        labf = lab.ravel()[active]
        cnt = np.bincount(labf, minlength=n + 1).astype(float)
        ext = cnt ** E                              # e(h_k)^E per component
        e_act = ext[labf]
        for c in conventions:
            wk = wts[c][k]
            if wk != 0.0:
                S[c][active] += wk * e_act
    out = {}
    for c in conventions:
        s = S[c].reshape(2 * B, R, nT)[:, :nF]
        out[c] = (s[:B], s[B:])
    return out


def tfce_scores(tmap, E=DEFAULT_E, H=DEFAULT_H, dh=DEFAULT_DH, convention='sn'):
    """Signed TFCE of a 1-D [nT] or 2-D [nF x nT] statistic map.

    Positive and negative tails are enhanced separately; the score carries the
    sign of t (a cell at t = 0 scores 0). convention: 'sn' (Smith & Nichols,
    primary) or 'mne' (MNE's step^H form). See the module header.
    """
    t = np.asarray(tmap, float)
    if not np.isfinite(t).all():
        raise ValueError('tfce_scores needs a finite map')
    shp = t.shape
    t2 = t[None] if t.ndim == 1 else t
    if t2.ndim != 2:
        raise ValueError(f'map must be 1-D or 2-D, got shape {shp}')
    sp, sn_ = _tfce_batch(t2[None], E, H, dh, (convention,))[convention]
    return (sp[0] - sn_[0]).reshape(shp)


# =============================================================================
# THE WEIGHTED PERMUTATION TEST
# =============================================================================
def tfce_perm_weighted_multi(X, w, n_perm=5000, seed=SEED, E=DEFAULT_E,
                             H=DEFAULT_H, dh=DEFAULT_DH, conventions=CONVENTIONS,
                             batch=200, max_gb=MAX_NULL_GB):
    """Weighted sign-flip TFCE, every requested convention from one pass.

    X : [nF x nT x nSub] or [nT x nSub], finite, subjects matching w (apply
        _listwise_keep to X and w together first, as for
        l2_heart.cluster_perm_weighted).
    Returns (t_obs [nF x nT], {conv: dict(tfce_obs, p_map, null_max)}, n_used)
    with 1-D input giving [1 x nT] maps.
    """
    X = np.asarray(X, float)
    conventions = tuple(conventions)
    for c in conventions:
        if c not in CONVENTIONS:
            raise ValueError(f'unknown convention {c!r}; known {CONVENTIONS}')
    if n_perm < 2:
        raise ValueError(f'n_perm = {n_perm}: column 0 is the identity, so the '
                         f'null would be empty.')
    if X.ndim == 2:
        X = X[None]
    nF, nT, nSub = X.shape
    w = np.asarray(w, float)
    if nSub != len(w):
        raise ValueError(f'{nSub} subjects in X but {len(w)} weights.')
    if not np.isfinite(X).all():
        raise ValueError('tfce_perm_weighted got non-finite cells - apply '
                         '_listwise_keep to X and w together before calling.')
    _check_memory(nF * nT, n_perm, max_gb)

    flips = _make_flips(nSub, n_perm, seed)
    T = _null_t_weighted(X.reshape(nF * nT, nSub), w, flips)   # [nCell x P]
    t_obs = T[:, 0].reshape(nF, nT).copy()

    tf_obs = {c: None for c in conventions}
    null_max = {c: np.empty(n_perm - 1) for c in conventions}
    for b0 in range(0, n_perm, batch):
        b1 = min(n_perm, b0 + batch)
        T3 = T[:, b0:b1].T.reshape(b1 - b0, nF, nT)
        res = _tfce_batch(T3, E, H, dh, conventions)
        for c in conventions:
            sp, sn_ = res[c]
            mx = np.maximum(sp, sn_).reshape(b1 - b0, -1).max(axis=1)
            if b0 == 0:
                tf_obs[c] = (sp[0] - sn_[0]).copy()
                null_max[c][:b1 - 1] = mx[1:]
            else:
                null_max[c][b0 - 1:b1 - 1] = mx
    del T
    out = {}
    for c in conventions:
        srt = np.sort(null_max[c])
        a = np.abs(tf_obs[c]).ravel()
        n_ge = srt.size - np.searchsorted(srt, a, side='left')   # #(null >= a)
        p = ((1.0 + n_ge) / n_perm).reshape(nF, nT)
        out[c] = dict(tfce_obs=tf_obs[c], p_map=p, null_max=null_max[c])
    return t_obs, out, nSub


def tfce_perm_weighted(X, w, n_perm=5000, seed=SEED, E=DEFAULT_E, H=DEFAULT_H,
                       dh=DEFAULT_DH, convention='sn', batch=200,
                       max_gb=MAX_NULL_GB):
    """Single-convention form. Returns (t_obs, tfce_obs, p_map, null_max, n_used)."""
    t_obs, res, n_used = tfce_perm_weighted_multi(
        X, w, n_perm, seed, E, H, dh, (convention,), batch, max_gb)
    r = res[convention]
    return t_obs, r['tfce_obs'], r['p_map'], r['null_max'], n_used


# =============================================================================
# READOUT: the significant cells as rows comparable to the cluster table
# =============================================================================
TFCE_FIELDS = (
    'Model', 'Measure', 'Scheme', 'Convention', 'Status',
    'N_subjects', 'N_dropped_nonfinite', 'N_eff_kish', 'nPerm', 'E', 'H', 'dh',
    'Alpha', 'Best_p', 'N_sig_total', 'N_components',
    'Component', 'Polarity', 'Start_ms', 'End_ms', 'Peak_ms',
    'Start_Hz', 'End_Hz', 'Peak_Hz', 'Peak_t', 'Peak_TFCE', 'Min_p',
    'N_cells', 'N_cells_preR', 'CohenD', 'HedgesG', 'CohenD_neff',
    'HedgesG_neff', 'CFA_flag', 'Note',
)


def tfce_components(t_obs, tfce_obs, p_map, times, freqs, w, n_used,
                    alpha=0.05):
    """Connected components of p < alpha, per polarity, 4-connectivity.

    Uses l2_run._clusters on sign(t) * (p < alpha) with threshold 0.5, so
    the component rule (per-sign labelling, CONN) is the cluster test's own.
    Returns (components sorted by n cells desc, best_p, n_sig). Each component
    dict carries its exact cell list [n x 2] (freq index, time index).
    """
    t_obs = np.atleast_2d(np.asarray(t_obs, float))
    p_map = np.atleast_2d(np.asarray(p_map, float))
    tfce_obs = np.atleast_2d(np.asarray(tfce_obs, float))
    times = np.asarray(times, float)
    sig = p_map < alpha
    best_p = float(p_map.min())
    n_eff = kish_n_eff(w)
    comps = []
    if sig.any():
        _, lab = _clusters(np.sign(t_obs) * sig, 0.5)
        for ci in range(1, int(lab.max()) + 1):
            cells = np.argwhere(lab == ci)
            if not cells.size:
                continue
            tv = t_obs[cells[:, 0], cells[:, 1]]
            pk = cells[int(np.argmax(np.abs(tv)))]
            peak_t = float(t_obs[pk[0], pk[1]])
            d_obs = peak_t / np.sqrt(n_used)
            d_eff = peak_t / np.sqrt(n_eff)
            s_ms, e_ms = float(times[cells[:, 1].min()]), float(times[cells[:, 1].max()])
            rec = dict(
                cells=cells.astype(np.int32),
                polarity='pos' if peak_t > 0 else 'neg',
                start_ms=s_ms, end_ms=e_ms, peak_ms=float(times[pk[1]]),
                peak_t=peak_t, peak_tfce=float(tfce_obs[pk[0], pk[1]]),
                min_p=float(p_map[cells[:, 0], cells[:, 1]].min()),
                n_cells=int(cells.shape[0]),
                n_cells_preR=int((times[cells[:, 1]] < 0).sum()),
                cohen_d=float(d_obs), hedges_g=float(d_obs * _hedges_J(n_used)),
                cohen_d_neff=float(d_eff),
                hedges_g_neff=float(d_eff * _hedges_J(n_eff)),
                cfa_flag=cfa_overlap(s_ms, e_ms),
                pf=int(pk[0]), pt=int(pk[1]))
            if freqs is not None:
                fr = np.asarray(freqs, float)
                rec.update(start_hz=float(fr[cells[:, 0].min()]),
                           end_hz=float(fr[cells[:, 0].max()]),
                           peak_hz=float(fr[pk[0]]))
            else:
                rec.update(start_hz='', end_hz='', peak_hz='')
            comps.append(rec)
    comps.sort(key=lambda c: (-c['n_cells'], c['min_p']))
    return comps, best_p, int(sig.sum())


def component_rows(meta, t_obs, tfce_obs, p_map, times, freqs, w, n_used,
                   alpha=0.05):
    """Tidy rows (TFCE_FIELDS) for one model x measure x scheme x convention.

    meta: Model, Measure, Scheme, Convention, N_dropped_nonfinite, nPerm, E, H, dh.
    A map with no significant cell still gets one row (Status 'no_sig') whose
    Best_p is the map minimum and whose Peak_* locate the max |TFCE| cell, so a
    near-miss is visible and a null is never an absent row.
    """
    comps, best_p, n_sig = tfce_components(t_obs, tfce_obs, p_map, times, freqs,
                                           w, n_used, alpha)
    base = {k: '' for k in TFCE_FIELDS}
    base.update(meta)
    base.update(N_subjects=n_used, N_eff_kish=round(kish_n_eff(w), 3),
                Alpha=alpha, Best_p=round(best_p, 6), N_sig_total=n_sig,
                N_components=len(comps))
    rows = []
    if not comps:
        t2 = np.atleast_2d(t_obs)
        tf2 = np.atleast_2d(tfce_obs)
        pf, pt = np.unravel_index(int(np.argmax(np.abs(tf2))), tf2.shape)
        r = dict(base)
        r.update(Status='no_sig', Component=0,
                 Peak_ms=float(np.asarray(times)[pt]),
                 Peak_Hz='' if freqs is None else float(np.asarray(freqs)[pf]),
                 Peak_t=round(float(t2[pf, pt]), 6),
                 Peak_TFCE=round(float(tf2[pf, pt]), 6),
                 Min_p=round(best_p, 6), CFA_flag=False,
                 Note='no cell with p < alpha; Peak_* = the max |TFCE| cell')
        rows.append(r)
        return rows, comps
    for i, c in enumerate(comps, 1):
        r = dict(base)
        r.update(Status='ok', Component=i, Polarity=c['polarity'],
                 Start_ms=c['start_ms'], End_ms=c['end_ms'], Peak_ms=c['peak_ms'],
                 Start_Hz=c['start_hz'], End_Hz=c['end_hz'], Peak_Hz=c['peak_hz'],
                 Peak_t=round(c['peak_t'], 6), Peak_TFCE=round(c['peak_tfce'], 6),
                 Min_p=round(c['min_p'], 6), N_cells=c['n_cells'],
                 N_cells_preR=c['n_cells_preR'],
                 CohenD=round(c['cohen_d'], 4), HedgesG=round(c['hedges_g'], 4),
                 CohenD_neff=round(c['cohen_d_neff'], 4),
                 HedgesG_neff=round(c['hedges_g_neff'], 4),
                 CFA_flag=c['cfa_flag'])
        rows.append(r)
    return rows, comps


# =============================================================================
# SUBJECTS + WEIGHTS, exactly as run_l2_heart builds them
# =============================================================================
def prepare_measure(subjects, measure, scheme, ivw_all=None):
    """(Xk, w, n_drop, kept_gidx) for one measure x scheme, along run_l2_heart's path.

    _stack_measure (subjects with a map) -> _listwise_keep (finite map) ->
    build_weights(scheme, n_units of the kept, ivw_all[kept]).
    Raises ValueError / ZeroDivisionError where build_weights does.

    Under scheme 'ivw_own' the weights are build_own_ivw_weights(subjects, measure), the tested
    measure's own slope variance, estimated here from `subjects` itself; `ivw_all` is ignored.
    The weights therefore always belong to the list passed in: a LOSO fold (the list without one
    participant) re-estimates the shrinkage on the remaining participants, and a surrogate draw
    (its own L1 subject list) re-estimates it from that draw's own svar. Raises ValueError when
    no subject has a usable variance.
    """
    X, _svar, nseg, gidx = _stack_measure(subjects, measure)
    if X is None:
        return None
    keep = _listwise_keep(X)
    Xk = X[..., keep]
    if str(scheme).lower() == IVW_OWN:
        own_all, _info = build_own_ivw_weights(subjects, measure, verbose=False)
        w = build_weights(scheme, nseg[keep], own_all[gidx[keep]])
    else:
        w = build_weights(scheme, nseg[keep],
                          None if ivw_all is None else ivw_all[gidx[keep]])
    return Xk, w, int((~keep).sum()), gidx[keep]


# =============================================================================
# GRAPH TFCE: the scalp analysis (channel x freq x time)
# =============================================================================
# Same statistic as _tfce_batch (Smith & Nichols by default: sum over
# h_k = k*dh < |t| of e(h_k)^E * h_k^H * dh, strict x > h_k, tails separate),
# but on an arbitrary undirected graph given as an edge list, because the scalp
# map's channel neighbours come from the montage, not from a lattice.
#
# Algorithm (exact, not an approximation): thresholds are processed from high
# to low. A cell becomes active at the highest level k with h_k < |t|; an edge
# at the highest level at which both ends are active and of the same sign (so a
# positive and a negative patch never join). At level k the components of
# {|t| > h_k} are the level-(k+1) components plus the newly active cells, joined
# by the edges that became active at level k only; every older edge is already
# inside a component. So each level runs connected_components on a contracted
# graph (old components + new cells, new edges only), every edge is touched
# once, and the per-level work on cells is O(active cells). Total cost
# O(E + sum_cells |t|/dh) per map; about 60 ms for a 64 x 27 x 176 null map.
# Validated against a per-threshold brute force and against _tfce_batch on a
# pure grid.
def graph_edges(adj):
    """Unique undirected edges (u < v, int64) of a sparse adjacency; the
    diagonal (MNE's find_ch_adjacency sets it) and duplicates are dropped."""
    from scipy import sparse
    A = sparse.coo_matrix(adj)
    r, c = A.row.astype(np.int64), A.col.astype(np.int64)
    m = r != c
    u, v = np.minimum(r[m], c[m]), np.maximum(r[m], c[m])
    n = int(A.shape[0])
    key = np.unique(u * n + v)
    return key // n, key % n


def scalp_cell_edges(chan_adj, nF, nT):
    """Edges of the [nC x nF x nT] cell graph, flat index c*nF*nT + f*nT + t.

    Neighbours along exactly one dimension: the same (f, t) on a montage
    neighbour channel (chan_adj, from heart_scalp_pipeline.
    build_channel_adjacency), or the same channel with f +- 1 or t +- 1.

    This is not a copy of the edge list of heart_scalp_pipeline.build_cluster_labeller. That
    builder uses flat offsets +-1 in (f, t) with only a 0 <= dst < nF*nT bounds check, so it
    also joins (f, nT-1) to (f+1, 0), the last time point of one frequency row to the first of
    the next (a two-cell mask at (5, 3, 175) and (5, 4, 0) comes back as one cluster). Here the
    time edges stop at the row end. The two edge sets differ by exactly those nC * (nF - 1)
    wrap edges and nothing else.
    """
    cu, cv = graph_edges(chan_adj)
    nC = int(chan_adj.shape[0])
    FT = nF * nT
    base = (np.arange(nC, dtype=np.int64) * FT)[:, None]
    ft = np.arange(FT, dtype=np.int64).reshape(nF, nT)
    us, vs = [], []
    if nT > 1:                                     # time neighbours, same f
        a = ft[:, :-1].ravel(); us.append((base + a).ravel())
        vs.append((base + a + 1).ravel())
    if nF > 1:                                     # freq neighbours, same t
        a = ft[:-1, :].ravel(); us.append((base + a).ravel())
        vs.append((base + a + nT).ravel())
    if cu.size:                                    # channel neighbours, same f,t
        us.append((cu[:, None] * FT + np.arange(FT)).ravel())
        vs.append((cv[:, None] * FT + np.arange(FT)).ravel())
    if not us:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(us), np.concatenate(vs)


def _cc_pairs(a_, b_, cidx):
    """Connected components of the undirected graph with edges (a_[i], b_[i])
    over arbitrary non-negative node ids < cidx.size. Returns (nodes, (rmin,
    cc)): the distinct node ids, and per component its smallest node id plus
    each node's component index.

    No np.unique / searchsorted / sort. The graph handed to
    connected_components has one node per endpoint occurrence j in [a_, b_];
    row j links j to its partner occurrence and to the canonical occurrence
    of the same id (cidx, last write wins). Two entries per row, so the CSR
    is built directly.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    ne = a_.size
    ab = np.concatenate([a_, b_])
    m2 = ab.size
    jj = np.arange(m2, dtype=np.int64)
    cidx[ab] = jj
    canon = cidx[ab]
    ind = np.empty(2 * m2, np.int64)
    ind[0::2] = np.concatenate([jj[ne:], jj[:ne]])
    ind[1::2] = canon
    g = csr_matrix((np.ones(2 * m2), ind,
                    np.arange(0, 2 * m2 + 1, 2, dtype=np.int64)), shape=(m2, m2))
    ncc, cc = connected_components(g, directed=False)
    isc = canon == jj
    nodes = ab[isc]
    ccn = cc[isc]
    # renumber the components that hold canonical nodes (all do: every
    # occurrence is linked to its canonical one)
    rmin = np.full(ncc, np.iinfo(np.int64).max)
    np.minimum.at(rmin, ccn, nodes)
    return nodes, (rmin, ccn)


def tfce_graph(t, eu, ev, E=DEFAULT_E, H=DEFAULT_H, dh=DEFAULT_DH,
               conventions=('sn',)):
    """Signed TFCE of a flat map t [N] on the graph (eu, ev); {conv: [N]}.

    Exact per-threshold TFCE (see the block comment above): thresholds
    h_k = k*dh (np.arange's own values), cells strictly above, positive and
    negative tails enhanced separately, e = component size in cells.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    t = np.asarray(t, float).ravel()
    if not np.isfinite(t).all():
        raise ValueError('tfce_graph needs a finite map')
    N = t.size
    a = np.abs(t)
    out = {c: np.zeros(N) for c in conventions}
    amax = float(a.max()) if N else 0.0
    if amax <= 0.0:
        return out
    K = int(np.ceil(amax / dh)) + 2
    hgrid = np.arange(K, dtype=float) * dh          # == np.arange(0, stop, dh)
    L = np.searchsorted(hgrid, a, side='left') - 1  # top active level, -1 never
    Lmax = int(L.max())
    wts = {c: _level_weights(hgrid[:Lmax + 1], H, dh, c) for c in conventions}
    if Lmax >= 32767:
        raise ValueError(f'{Lmax + 1} TFCE levels: raise dh')
    # the lowest level any convention weights; below it nothing is added, so
    # neither its cells nor its merges need processing (k = 0 has h = 0, which
    # weighs 0 in both conventions whenever H > 0)
    live = [k for k in range(Lmax + 1) if any(wts[c][k] != 0.0 for c in conventions)]
    if not live:
        return out
    kmin = live[0]
    # cells ordered by level, highest first; everything below works in this
    # position space, so the active set at level k is the contiguous prefix
    # [:n_ge[k]] and the per-level updates are slices, not fancy indexing.
    # int16 keys: numpy's stable argsort is a radix sort for <= 16-bit ints.
    L16 = L.astype(np.int16)
    co = np.argsort(-L16, kind='stable').astype(np.int32)
    pos = np.empty(N, np.int32)
    pos[co] = np.arange(N, dtype=np.int32)
    # edges: active from min(L_u, L_v), only between same-sign cells. One
    # signed int16 code per cell (+-(L+1), 0 = never active) makes that two
    # gathers per edge instead of four.
    code = np.where(t > 0, L16 + 1, -(L16 + 1)).astype(np.int16)
    cu, cv = code[eu], code[ev]
    Le = np.minimum(np.abs(cu), np.abs(cv))
    Le -= 1
    ok = (Le >= kmin) & ((cu > 0) == (cv > 0))
    del cu, cv
    sel = np.flatnonzero(ok)
    Le = Le[sel]
    eo = np.argsort(-Le, kind='stable')
    sel = sel[eo]
    pu = pos[eu[sel]]
    pv = pos[ev[sel]]
    # n cells / edges with level >= k (reverse cumulative counts)
    cntL = np.bincount(L[L >= 0], minlength=Lmax + 1)
    n_ge = np.concatenate([np.cumsum(cntL[::-1])[::-1], [0]])
    cntE = np.bincount(Le, minlength=Lmax + 1)
    e_ge = np.concatenate([np.cumsum(cntE[::-1])[::-1], [0]])
    # A component is named by the position of one of its cells (its rep); a
    # merged component takes the smallest rep, so an old component (all its
    # cells at positions < n0) always keeps an old rep and the old cells need
    # relabelling only when two old components merge. size/ext are indexed by
    # rep and are stale for absorbed reps, which no active cell points to.
    #
    # Per level, the new edges each touch >= 1 new cell (Le = min(L_u, L_v)):
    #   A  new-old: a new cell touching ONE old component just joins it (no
    #      graph work); one touching several bridges them (a merge pair)
    #   B  new-new: pairs of provisional reps after A; equal ones drop out
    # Only the merge pairs (few) go to connected_components.
    rep = np.arange(N, dtype=np.int64)
    size = np.ones(N)
    ext = np.ones(N)                                 # size ** E
    remap = np.arange(N, dtype=np.int64)
    cidx = np.zeros(N, np.int64)                     # scratch for _cc_pairs
    BIG = np.int64(N)
    accp = {c: np.zeros(N) for c in conventions}
    for k in range(Lmax, kmin - 1, -1):
        n0, n1 = int(n_ge[k + 1]), int(n_ge[k])
        nn = n1 - n0
        e0, e1 = int(e_ge[k + 1]), int(e_ge[k])
        if e1 > e0:
            u = pu[e0:e1]; v = pv[e0:e1]
            un = u >= n0; vn = v >= n0
            both = un & vn
            one = un != vn
            prov = np.arange(n0, n1, dtype=np.int64)
            pa = pb = np.zeros(0, np.int64)
            if one.any():
                uo, vo, uno = u[one], v[one], un[one]
                xa = np.where(uno, uo, vo) - n0      # new endpoint, 0..nn-1
                ry = rep[np.where(uno, vo, uo)]      # old endpoint's rep
                mn = np.full(nn, BIG)
                np.minimum.at(mn, xa, ry)
                att = mn != BIG
                tgt = mn[att]
                prov[att] = tgt
                np.add.at(size, tgt, 1.0)
                ext[tgt] = size[tgt] ** E
                mxa = mn[xa]
                br = mxa != ry                       # bridges: merge pairs
                pa, pb = mxa[br], ry[br]
            if both.any():
                qa = prov[u[both] - n0]; qb = prov[v[both] - n0]
                dd = qa != qb
                pa = np.concatenate([pa, qa[dd]]); pb = np.concatenate([pb, qb[dd]])
            if pa.size:
                nodes, rr = _cc_pairs(pa, pb, cidx)
                csz = np.bincount(rr[1], weights=size[nodes], minlength=rr[0].size)
                rmin = rr[0]                         # min node per component
                size[rmin] = csz
                ext[rmin] = csz ** E
                newr = rmin[rr[1]]
                remap[nodes] = newr
                prov = remap[prov]
                if np.any((nodes < n0) & (newr != nodes)):
                    rep[:n0] = remap[rep[:n0]]
                remap[nodes] = nodes
            rep[n0:n1] = prov
        ea = ext[rep[:n1]]
        for c in conventions:
            wk = wts[c][k]
            if wk != 0.0:
                accp[c][:n1] += wk * ea
    acc = {}
    for c in conventions:
        acc[c] = np.empty(N)
        acc[c][co] = accp[c]
    sg = np.sign(t)
    for c in conventions:
        out[c] = acc[c] * sg
    return out


def graph_components(mask, sign, eu, ev):
    """Connected components of `mask` on the graph, separately per sign
    (sign = np.sign of the t map). Returns labels [N] int, -1 background."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    mask = np.asarray(mask, bool).ravel()
    sign = np.asarray(sign).ravel()
    N = mask.size
    keep = mask[eu] & mask[ev] & (sign[eu] == sign[ev])
    g = coo_matrix((np.ones(int(keep.sum())), (eu[keep], ev[keep])), shape=(N, N))
    _, lab = connected_components(g, directed=False)
    out = np.full(N, -1, np.int64)
    if mask.any():
        _, inv = np.unique(lab[mask], return_inverse=True)
        out[mask] = inv
    return out
