# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
lee_selection: scoring core of the Lee component selection.

Picks the IC(s) whose heartbeat-locked response is strongest, by scoring every
candidate inside a frequency x time box and taking the argmax (or everything
within ic_sel_thresh * max).

Norm modes 'raw', 'ztrial', 'sham', 'sham_raw' and 'blratio', under either criterion
('hrsp', 'ppc'); plus the selection/threshold rule and box_stat. 'blratio' is the median
over beats of log(mean power in t_sel / mean power in t_bl): scale-free (unlike 'raw'),
with no sham draws (cheap, and much less circular with the surrogate than 'sham_raw').

Read before trusting a selection:
  1. Under criterion='ppc' the norm mode has no effect. PPC divides amplitude out
     by construction (`z = Yc/|Yc|`), so 'raw', 'ztrial', 'sham_raw' all give the
     same score. Runs differing only in norm are one result printed N times, not
     replications, even though the folder tag differs (e.g. shamrawsel50). This is
     why `use_z`/`use_sham` below are gated on criterion=='hrsp'.
  2. The 'raw' score is tonic power, not heartbeat locking. It is mean |CWT|^2
     inside the box with no baseline and no contrast: a component with lots of
     flat alpha wins. 'ztrial' is what restores locking (per-trial z across the
     full epoch, then baseline subtraction).
  3. Lee selection is anti-correlated with component quality on this data: the
     per-trial z divides amplitude out, so channel-noise components can win.
     Selecting well is not the same as selecting brain.
  4. PPC here uses N = nBeats (all beats), not the per-cell finite count that
     heart_core.ppc_map uses. Equivalent only when no beat is NaN in the box.

The default norm mode is 'sham_raw', the pipeline's IC_SEL_NORM.

Sham shifts are drawn with NumPy's RandomState (MT19937) from a fixed seed
(`make_sham_shifts`); 'raw', 'ztrial' and 'blratio' draw nothing.
"""
from __future__ import annotations
import numpy as np

EPS = np.finfo(float).eps

__all__ = ['box_stat', 'lee_scores', 'select_ics_lee', 'make_sham_shifts']


def box_stat(co, ei, t_bl, t_sel, do_z):
    """Mean baseline-subtracted power of one IC at one frequency in the selection box.

    co    : complex CWT of one IC at one frequency, over the whole recording
    ei    : [nBeats x nT] sample indices of the epochs (0-based here)
    t_bl  : bool mask over nT, the pre-R-peak baseline window
    t_sel : bool mask over nT, the selection box in time
    do_z  : per-trial z-score across the full epoch before baselining
    """
    Yp = np.abs(co[ei]) ** 2                                  # [nBeats x nT]
    if do_z:
        sd = np.std(Yp, axis=1, ddof=1)                       # sample SD per beat
        Yp = (Yp - Yp.mean(axis=1, keepdims=True)) / np.maximum(sd, EPS)[:, None]
    Yp = Yp - Yp[:, t_bl].mean(axis=1, keepdims=True)
    return float(Yp[:, t_sel].mean())


def make_sham_shifts(n_sh, fs, shift_ms=(-500, 500), min_shift_ms=150, seed=20260728):
    """Rigid epoch shifts for the sham null.

    Draws n_sh shifts uniformly in shift_ms, rejecting any |shift| < min_shift_ms
    (too close to the real train to decorrelate), returned in samples.
    """
    rs = np.random.RandomState(seed)
    out = []
    while len(out) < n_sh:
        d = shift_ms[0] + (shift_ms[1] - shift_ms[0]) * rs.rand()
        if abs(d) < min_shift_ms:
            continue
        out.append(int(round(d / 1000.0 * fs)))
    return np.asarray(out, int)


def lee_scores(ic_cwt, ei, t_bl, t_sel, criterion='hrsp', norm_mode='sham_raw',
               n_pts=None, sham_opts=None):
    """Score every candidate IC.

    ic_cwt : [nIC x nFreqSel x nSamples] complex CWT at the selection frequencies
             (built with heart_core.morlet_cwt)
    ei     : [nBeats x nT] 0-based epoch sample indices
    t_bl,
    t_sel  : bool masks over nT
    Returns [nIC] scores (mean over selection frequencies).
    """
    criterion = criterion.lower()
    norm_mode = norm_mode.lower()
    if norm_mode not in ('raw', 'ztrial', 'sham', 'sham_raw', 'blratio'):
        raise ValueError(f"norm_mode must be raw|ztrial|sham|sham_raw|blratio, "
                         f"got {norm_mode!r}")

    # Both gates apply under criterion=='hrsp' only: under ppc the norm mode is inert.
    use_z = norm_mode in ('ztrial', 'sham') and criterion == 'hrsp'
    use_sham = norm_mode in ('sham', 'sham_raw') and criterion == 'hrsp'

    if (use_z or use_sham) and not np.any(t_bl):
        raise ValueError(f"norm_mode={norm_mode!r} needs a non-empty baseline mask t_bl")

    ic_cwt = np.asarray(ic_cwt)
    nIC, nFsel, _ = ic_cwt.shape
    ei = np.asarray(ei, int)
    nBeats, nT = ei.shape

    sh_idx, keep_b = [], np.ones(nBeats, bool)
    if use_sham:
        so = sham_opts or {}
        if 'fs' not in so:
            raise ValueError("sham modes require sham_opts['fs']")
        deltas = make_sham_shifts(so.get('n', 200), so['fs'],
                                  so.get('shift_ms', (-500, 500)),
                                  so.get('min_shift_ms', 150),
                                  so.get('seed', 20260728))
        if n_pts is None:
            raise ValueError('sham modes require n_pts (recording length in samples)')
        lo, hi = deltas.min(), deltas.max()
        keep_b = (ei[:, 0] + lo >= 0) & (ei[:, -1] + hi < n_pts)
        if keep_b.sum() < 10:
            raise ValueError(f'only {keep_b.sum()} beats survive the sham shift range')
        base = ei[keep_b]
        sh_idx = [base + d for d in deltas]

    scores = np.empty(nIC)
    for k in range(nIC):
        sc = 0.0
        for kf in range(nFsel):
            co = ic_cwt[k, kf]
            if criterion == 'hrsp':
                if use_sham:
                    m_real = box_stat(co, ei[keep_b], t_bl, t_sel, use_z)
                    m_sh = np.array([box_stat(co, s, t_bl, t_sel, use_z) for s in sh_idx])
                    sd = np.nanstd(m_sh, ddof=1)
                    sc += 0.0 if (not np.isfinite(sd) or sd <= 0) else \
                        (m_real - np.nanmean(m_sh)) / sd
                elif norm_mode == 'blratio':
                    # The scale-free, sham-free option. `raw` is tonic power (the
                    # loudest IC wins); baseline subtraction does not fix that,
                    # because a component twice as loud has twice the difference. A
                    # ratio is dimensionless: a flat component scores log(1) = 0
                    # however loud it is, so the score is "how much does this
                    # component rise after the R-peak". It needs no sham draws, so it
                    # is cheap and far less circular with the shifted-R-peak
                    # surrogate than `sham_raw`, which explicitly maximises what that
                    # control tests.
                    # Log, because a raw ratio is bounded below by 0 and unbounded
                    # above; its mean is dragged by a few beats with a near-zero
                    # denominator.
                    # Median across beats, not mean. A pre-R baseline is also
                    # post-previous-R: on this cohort about 19% of beats have the far
                    # edge of a [-150,-50] window inside the previous beat's 0-600 ms
                    # response. The median is drawn from the clean majority; a mean
                    # is not.
                    _P = np.abs(co[ei]) ** 2
                    _num = _P[:, t_sel].mean(axis=1)
                    _den = _P[:, t_bl].mean(axis=1)
                    _ok = np.isfinite(_num) & np.isfinite(_den) & (_den > EPS) & (_num > 0)
                    sc += (0.0 if not _ok.any()
                           else float(np.median(np.log(_num[_ok] / _den[_ok]))))
                elif use_z:
                    sc += box_stat(co, ei, t_bl, t_sel, True)
                else:
                    # 'raw': mean power in the box, no baseline, no contrast
                    sc += float((np.abs(co[ei][:, t_sel]) ** 2).mean())
            else:
                # 'ppc': amplitude divided out, so norm_mode is inert here
                Yc = co[ei][:, t_sel]
                z = Yc / (np.abs(Yc) + EPS)
                N = nBeats                      # all beats, not the per-cell finite count
                S = z.sum(axis=0)
                sc += float(((np.abs(S) ** 2 - N) / (N * (N - 1))).mean())
        scores[k] = sc / nFsel
    return scores


def select_ics_lee(ic_cwt, ei, t_bl, t_sel, criterion='hrsp', norm_mode='sham_raw',
                   ic_sel_thresh=1.0, ic_pool=None, n_pts=None, sham_opts=None):
    """Full selection. Returns (sel_ics, ic_weights, pool_scores).

    ic_sel_thresh=1.0 -> the single best IC, one per session.

    Selection rule: components scoring at or above ic_sel_thresh * max are kept,
    except that for any norm mode other than 'raw', when the max is <= 0 only the
    argmax is returned: scores can go negative ('ztrial'/'sham' can), and a
    threshold on a negative max would select the worst components.
    """
    scores = lee_scores(ic_cwt, ei, t_bl, t_sel, criterion, norm_mode, n_pts, sham_opts)
    pool = np.full(scores.shape, -np.inf)
    idx = np.arange(len(scores)) if ic_pool is None else np.asarray(ic_pool, int)
    pool[idx] = scores[idx]

    top = int(np.argmax(pool))
    mx = pool[top]
    if norm_mode.lower() == 'raw' or mx > 0:
        sel = np.flatnonzero((pool >= ic_sel_thresh * mx) & np.isfinite(pool))
    else:
        sel = np.array([top])
    sel = sel[np.argsort(pool[sel])[::-1]]
    return sel, pool[sel], pool
