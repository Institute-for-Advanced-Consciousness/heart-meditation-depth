# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
glm_baseline: Alday (2019) GLM baseline correction for the heartbeat-locked measures.

This is the pipeline's only baseline method. Instead of rewriting the data
(subtract, dB or percent), the pre-R-peak activity enters the stage-1 GLM as an
extra regressor, and its coefficient is estimated jointly with the depth slope.
beta_bl == 1 reproduces traditional subtraction, beta_bl == 0 reproduces no
correction, and anything in between is what the data support. Alday (2019), "How
much baseline correction do we need in ERP research?", Psychophysiology.

Nothing here subtracts anything. Every function returns columns to append to a
design matrix; the measure array is returned untouched. Subtract, dB and percent
baselines are not offered because they assume the baseline instead of estimating
it.

The column goes last, after the model covariates. Callers index coefficient
rows positionally (depth at index 1, covariates after it), so a column inserted
anywhere else renumbers their results.

One column for HEP, one per frequency for HRSP, none for HEPC (HEPC, as the code
names it, is the paper's HRPC, heartbeat-related phase consistency).
  HEP  : the baseline is a scalar per beat (mean pre-R-peak voltage), 1 column.
  HRSP : the baseline is the pre-R-peak power at that frequency, so the design
         changes with frequency and the time-frequency map is solved one
         frequency at a time, handing column f to the solve for frequency f.
  HEPC : no baseline. HEPC is unbiased PPC over unit-length phasors, so
         amplitude is divided out by construction and there is no power or
         amplitude scale to adjust. Asking for one raises.

Degrees of freedom. The baseline column costs one degree of freedom that HEPC
does not spend, so nP_hep == nP_hepc + 1 whenever the GLM baseline is on:
  nUnits <  nP_hep : HEP/HRSP is refused (all-NaN slope, the subject leaves the
                     sample) while HEPC, needing one df fewer, is fitted.
  nUnits == nP_hep : HEP/HRSP is exactly identified, so the slope is returned
                     with svar = Inf. The subject reaches the unweighted group
                     mean but carries zero weight in IVW, while HEPC gets a
                     finite variance and full weight.
This happens at a single cell count, not over a range, so a check of the depth
floor alone misses it for every covariate model. Read N_subjects per (measure,
model) from the l2_heart_summary_<model>.csv tables (heart_depth_run) before
comparing a HEP map with a HEPC map. Nothing here changes that arithmetic.
Running without the baseline (DO_BASELINE_CORRECT = False) removes the
asymmetry.

The handshake. 'glm' is the only baseline whose contract spans two places: the
pipeline computes the covariate and the estimator must consume it. A result
folder named for a baseline-adjusted analysis is not evidence that the column
reached the fit, which is why l1_fit_summary.csv records N_bl_cols and N_bl_dead
for every subject.

Order of operations (as in the pipeline; do not reorder)
   1. per-beat baseline   mean over the baseline mask, on the full epoch axis
                          (NaN-propagating, see beat_baseline)
   2. filter to kept beats (keep_bin: depth levels + INCLUDE_SAPO)
   3. centre per column across beats, then zero non-finite entries
   4. aggregate to units  cell/segment means, omitting NaN
   5. centre per column across units, then zero non-finite entries
   6. append last to the design
Steps 3 and 5 both centre. Centring an average of centred values changes it only
through unequal cell sizes.

Not in this module: the per-beat solver, the solve loop over frequencies, the stage-1 cell construction, PPC
and VIF, and the baseline window used for IC selection (lee_selection.py reuses
BASELINE_WIN for selection, not for correction).

Behaviour to know about
  1. Centring drops +/-Inf as well as NaN, for HEP and HRSP alike, so an
     infinite value loses only the offending unit. A mean of voltages or of
     |CWT|^2 cannot produce +/-Inf unless something upstream is broken.
  2. strict=True (the default) refuses a finished block that is entirely zero,
     with one message for an all-non-finite baseline window and another for a
     constant baseline (finite but identical across units, so it centres to
     zeros). Such a column would cost a df and carry nothing. strict=False
     returns it instead (cell_ols's 1e-8*trace ridge keeps X'X invertible and
     the column gets beta = 0); call dead_columns() on the result.
  3. A real HRSP array raises unless real_is_power=True. A real array could be
     power or amplitude, and passing abs(Yc) as if it were power would give a
     mean-amplitude baseline, breaking the beta_bl == 1 == subtraction
     correspondence with the |CWT|^2 DV.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    'BASELINE_WIN_DEFAULT',
    'GlmBaselineError',
    'baseline_mask', 'beat_baseline', 'centre_columns', 'aggregate_units',
    'glm_baseline_regressors', 'dead_columns',
]

# ── Constants ─────────────────────────────────────────────────────────────────
BASELINE_WIN_DEFAULT = (-150.0, -50.0)
# -150..-50 ms, not Park et al. 2018's [-300 -100]: the window sits close to the
# R-peak so that it does not reach back into the previous beat. It is a pipeline
# parameter; this constant is only the default for standalone use.


# Which measures take a baseline column at all. HEPC is False by construction.
_TAKES_BASELINE = {'HEP': True, 'HRSP': True, 'HEPC': False}


class GlmBaselineError(ValueError):
    """Raised when a baseline regressor cannot be built or has not been wired.

    Always carries a named reason and is never swallowed into a default: a run
    that silently loses its baseline looks exactly like a null result in a folder
    whose name claims a baseline-adjusted analysis.
    """


# ── Small NaN-aware helpers ───────────────────────────────────────────────────
def _finite_mean(A, axis):
    """Mean over `axis` ignoring non-finite entries; all-non-finite -> NaN.

    Written by hand rather than np.nanmean so that an all-NaN slice returns NaN
    quietly (no RuntimeWarning) and so that +/-Inf is dropped as well as NaN.
    """
    A = np.asarray(A, float)
    fin = np.isfinite(A)
    cnt = fin.sum(axis=axis)
    tot = np.where(fin, A, 0.0).sum(axis=axis)
    out = np.full(np.shape(cnt), np.nan, float)
    nz = cnt > 0
    out[nz] = tot[nz] / cnt[nz]
    return out


def _as_2d(A):
    A = np.asarray(A, float)
    return A[:, None] if A.ndim == 1 else A


# ── 1. the time mask ──────────────────────────────────────────────────────────
def baseline_mask(times_ms, win=BASELINE_WIN_DEFAULT):
    """Boolean mask selecting the baseline window on a full-epoch time axis.

    times_ms : [nT] epoch time axis in ms (the full epoch, before any trimming)
    win      : (lo, hi) ms, inclusive at both ends (t >= lo and t <= hi)

    Raises if the window selects no samples or extends beyond the epoch axis; the
    window is never clipped. A zero-width window (hi == lo) is allowed and selects
    the single sample at that latency (it raises if that latency falls between two
    samples). A reversed window (hi < lo) raises with its own message.
    """
    t = np.asarray(times_ms, float).ravel()
    lo, hi = float(win[0]), float(win[1])
    if not np.isfinite([lo, hi]).all():
        raise GlmBaselineError(f'baseline window {win!r} is not a finite (lo, hi) pair')
    if hi < lo:
        raise GlmBaselineError(
            f'baseline window {win!r} is REVERSED (hi < lo); it would select 0 samples. '
            'hi == lo is allowed and selects one sample (`t >= lo & t <= hi`).')
    if lo < t.min() or hi > t.max():
        raise GlmBaselineError(
            f'baseline window [{lo:g} {hi:g}] ms lies outside the epoch '
            f'[{t.min():g} {t.max():g}] ms -- extend the epoch, do not clip the window')
    m = (t >= lo) & (t <= hi)
    if not m.any():
        raise GlmBaselineError(
            f'baseline window [{lo:g} {hi:g}] ms selects 0 samples on a {t.size}-sample axis')
    return m


# ── 2. per-beat baseline ──────────────────────────────────────────────────────
def beat_baseline(Y, bl_mask, measure='HEP', *, real_is_power=False):
    """Per-beat baseline value(s): the mean of the measure over the baseline window.

    Y : HEP  -> [nBeats x nT]            real voltage, full epoch
        HRSP -> [nBeats x nFreq x nT]    complex CWT coefficients (squared to
                                         power here), or a real array with
                                         real_is_power=True. Full epoch either way.
    bl_mask : [nT] bool, from baseline_mask(), on the same axis as Y's last dim.
    real_is_power : HRSP only. A complex array is unambiguous and this flag is
        ignored. A real array could be power (|CWT|^2, the DV) or amplitude
        (|CWT|), which give different baselines, so a real array raises unless
        this is True. There is no default guess.

    Returns [nBeats x 1] for HEP, [nBeats x nFreq] for HRSP.

    The axis must be the full epoch: the pipeline computes the baselines from the
    untrimmed HEP epoch and CWT, while the DVs are the trimmed arrays. Handing
    this function a trimmed array silently shifts the window.

    The time mean propagates NaN on purpose, so one bad sample removes that
    beat's baseline. The beat-to-unit aggregation in aggregate_units() omits NaN
    instead: a beat whose baseline window is partly missing is not a beat whose
    baseline is known.
    """
    measure = str(measure).upper()
    if measure not in _TAKES_BASELINE:
        raise GlmBaselineError(f'unknown measure {measure!r}; expected one of {sorted(_TAKES_BASELINE)}')
    if not _TAKES_BASELINE[measure]:
        raise GlmBaselineError(
            'HEPC takes NO baseline column. It is unbiased PPC over UNIT-LENGTH '
            'phasors, so amplitude is divided out by construction and there is no '
            'power or amplitude scale left to adjust. This is a property of the '
            'measure, not an omission.')

    A = np.asarray(Y)
    m = np.asarray(bl_mask, bool).ravel()
    if not m.any():
        raise GlmBaselineError('bl_mask selects 0 samples')
    if A.shape[-1] != m.size:
        raise GlmBaselineError(
            f'bl_mask has {m.size} samples but the measure has {A.shape[-1]} on its last axis '
            '(the mask must be built on the SAME full-epoch axis as the data)')

    if measure == 'HEP':
        if A.ndim != 2:
            raise GlmBaselineError(f'HEP expects [nBeats x nT]; got shape {A.shape}')
        if np.iscomplexobj(A):
            raise GlmBaselineError('HEP is a real voltage; a complex array is not the HEP epoch')
        # Plain mean over the window -> [nBeats x 1]. NaN-propagating (see above).
        return np.mean(np.asarray(A, float)[:, m], axis=1)[:, None]

    # HRSP
    if A.ndim != 3:
        raise GlmBaselineError(f'HRSP expects [nBeats x nFreq x nT]; got shape {A.shape}')
    win = A[:, :, m]
    if np.iscomplexobj(win):
        # Power, not amplitude: it has to match the DV, abs(Yc)**2, or
        # beta_bl = 1 would no longer correspond to plain subtraction.
        win = np.abs(win) ** 2
    elif real_is_power:
        # The caller has stated that this is already power. Left unsquared on
        # purpose: squaring again here would give |CWT|^4.
        win = np.asarray(win, float)
    else:
        # A real array is ambiguous. The most likely caller error, handing over
        # abs(Yc), would give a plausible-looking mean-amplitude baseline and
        # break the beta_bl == 1 == subtraction correspondence with the |CWT|^2
        # DV. So the caller has to say which it is.
        raise GlmBaselineError(
            'HRSP was handed a REAL array, whose scale is ambiguous: power (|CWT|^2, '
            'what the DV is) and amplitude (|CWT|) give different baselines and this '
            'function cannot tell them apart. Pass the COMPLEX CWT, or pass '
            'real_is_power=True to state that you already '
            'squared. Passing abs(Yc) with real_is_power=True is a mean-AMPLITUDE '
            'baseline and is WRONG for the |CWT|^2 DV.')
    return np.mean(win, axis=2)          # [nBeats x nFreq], NaN-propagating


# ── 3./5. centring ────────────────────────────────────────────────────────────
def centre_columns(A):
    """Mean-centre each column (ignoring non-finite), then zero non-finite entries.

    The same treatment every covariate gets in this project: centre so the
    intercept stays the grand mean, then set missing values to 0 ("no deviation
    from the mean") so that a single missing unit does not delete the column.

    A column that was entirely non-finite becomes all zeros: collinear with the
    intercept, kept solvable by the solver's ridge, contributing nothing, and
    still costing a degree of freedom. dead_columns() detects it, and
    glm_baseline_regressors(strict=True) raises when the whole block is dead.
    """
    A = _as_2d(A).copy()
    mu = _finite_mean(A, axis=0)                 # [nCols], NaN if a column is dead
    out = A - mu[None, :]
    out[~np.isfinite(out)] = 0.0
    return out


# ── 4. beat -> unit aggregation ───────────────────────────────────────────────
def aggregate_units(bl_beats, units):
    """Average a per-beat baseline into per-unit (cell or segment) means.

    bl_beats : [nBeats x k]   from beat_baseline (k = 1 for HEP, nFreq for HRSP)
    units    : [nBeats] labels, or [nBeats x 2] rows. The pipeline keys cells on
               [session, depth] and orders them lexicographically ascending, as
               np.unique(axis=0) does here. The unit labels are returned so that
               the caller can build its measure matrix on the same ordering.

    Returns (block [nUnits x k], labels).

    Aggregation omits NaN here, unlike the time average in beat_baseline. A unit
    whose beats are all non-finite yields NaN, which the next centring step turns
    into 0.
    """
    B = _as_2d(bl_beats)
    U = np.asarray(units)
    if U.ndim == 1:
        U = U[:, None]
    if U.shape[0] != B.shape[0]:
        raise GlmBaselineError(f'units has {U.shape[0]} rows but the baseline has {B.shape[0]} beats')
    labels, gid = np.unique(U, axis=0, return_inverse=True)
    gid = np.asarray(gid).ravel()
    n_units = labels.shape[0]
    out = np.full((n_units, B.shape[1]), np.nan, float)
    for c in range(n_units):
        idx = gid == c
        if idx.any():
            out[c, :] = _finite_mean(B[idx, :], axis=0)
    return out, (labels.ravel() if labels.shape[1] == 1 else labels)


# ── the headline function ─────────────────────────────────────────────────────
def glm_baseline_regressors(Y, bl_mask, measure='HEP', *, units=None, keep=None,
                            strict=True, return_labels=False, real_is_power=False):
    """Build the Alday baseline regressor(s) to append to the L1 design matrix.

    Returns columns, never a corrected copy of Y. Y is not modified.

    Y        : the measure on the full epoch axis
               HEP  [nBeats x nT] voltage; HRSP [nBeats x nFreq x nT] complex
               CWT (or a real array, but only with real_is_power=True, because
               power and amplitude are not distinguishable from the array alone).
    bl_mask  : [nT] bool from baseline_mask()
    measure  : 'HEP' | 'HRSP'  ('HEPC' raises, with the reason)
    units    : optional [nBeats] or [nBeats x 2] unit keys. Given, the result is
               per unit (the cell/segment grain, [nUnits x k]). Omitted, the
               result is per beat ([nBeats x k], the per-beat grain).
    keep     : optional [nBeats] bool, the pipeline's keep_bin (depth levels +
               INCLUDE_SAPO), applied before the first centring. If `units` is
               given it is filtered by the same mask.
    strict   : raise if the finished block is entirely zero, i.e. carries no
               information while still costing one df per column. The two
               causes get different messages: an all-non-finite baseline window,
               and a constant (finite, identical across units) one. strict=False
               returns the all-zero block instead; pair it with dead_columns().
    real_is_power : HRSP only; forwarded to beat_baseline. A real HRSP array
               raises unless this is True.

    Returns [nUnits x k] (or [nBeats x k]), centred, non-finite zeroed, ready to
    be appended last to the design. k = 1 for HEP, nFreq for HRSP; for HRSP hand
    column f to the solve for frequency f.
    """
    Y_in = Y
    before = np.array(Y_in, copy=True) if isinstance(Y_in, np.ndarray) else None

    bl = beat_baseline(Y_in, bl_mask, measure, real_is_power=real_is_power)   # [nBeats x k]

    if keep is not None:
        k = np.asarray(keep, bool).ravel()
        if k.size != bl.shape[0]:
            raise GlmBaselineError(f'keep has {k.size} entries but there are {bl.shape[0]} beats')
        if not k.any():
            raise GlmBaselineError('keep excludes every beat -- no baseline can be built')
        bl = bl[k, :]
        if units is not None:
            u = np.asarray(units)
            units = u[k] if u.ndim == 1 else u[k, :]

    # Recorded BEFORE centring, because centring maps non-finite -> 0 and so
    # destroys the evidence needed to tell the two strict refusals apart.
    any_finite_pre = bool(np.any(np.isfinite(bl)))

    bl = centre_columns(bl)                             # step 3: centre + zero

    labels = None
    if units is not None:
        bl, labels = aggregate_units(bl, units)         # step 4: cell/segment means
        bl = centre_columns(bl)                         # step 5: centre again

    if strict and not np.any(bl != 0.0):
        # Both cases end as the same all-zero block but have different causes,
        # so they get different messages. strict=False keeps the block.
        if not any_finite_pre:
            raise GlmBaselineError(
                f'{measure} baseline is all non-finite: every beat baseline in the '
                'window is NaN/Inf, so the centred block is entirely zero -- a column '
                'that costs one degree of freedom and carries nothing. Check the '
                'baseline window, the beat filter, and the epoch axis. '
                '(Pass strict=False to keep the all-zero column.)')
        raise GlmBaselineError(
            f'{measure} baseline is constant across units: the values that reach the '
            'design (per unit if units= was given, else per beat) are finite but all '
            'equal, so centring leaves exact zeros -- a column collinear with the '
            'intercept that costs one degree of freedom and carries nothing. This is a '
            'DEGENERATE-BUT-LEGAL input (cell_ols\'s 1e-8*trace ridge keeps '
            'X\'X invertible and the dead column gets beta = 0), so pass strict=False '
            'if you deliberately want that; then check dead_columns().')

    # Contract check: we return regressors, we do not touch the data.
    if before is not None and not _same_array(before, np.asarray(Y_in)):
        raise GlmBaselineError('INTERNAL: the measure array was modified -- this method '
                               'estimates the baseline, it never subtracts it')
    return (bl, labels) if return_labels else bl


def _finite_equal(x, y):
    """Bit-equality on a real float array, counting NaN as equal to NaN.

    Hand-written rather than np.array_equal(..., equal_nan=True) so that the rule
    is the same on every numpy version: equal_nan raises TypeError on complex
    input in older numpy.
    """
    nx, ny = np.isnan(x), np.isnan(y)
    if not np.array_equal(nx, ny):          # NaNs must sit in the same places
        return False
    return bool(np.all(x[~nx] == y[~ny]))   # and everything else must be exact


def _same_array(a, b):
    """True iff `b` is bit-identical to `a`, NaN positions included.

    Complex arrays are compared part by part, so a NaN in either the real or the
    imaginary part must sit in the same place in both arrays. This catches an
    in-place NaN -> 0 rewrite of a CWT array, and a rewrite of the imaginary part
    of a value whose real part is NaN.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if np.iscomplexobj(a) != np.iscomplexobj(b):
        return False                        # a real<->complex recast IS a change
    if not (np.issubdtype(a.dtype, np.inexact) and np.issubdtype(b.dtype, np.inexact)):
        return bool(np.array_equal(a, b))   # int/bool arrays cannot hold NaN
    if np.iscomplexobj(a):
        return _finite_equal(a.real, b.real) and _finite_equal(a.imag, b.imag)
    return _finite_equal(a, b)


def dead_columns(bl):
    """Indices of baseline columns that carry no information (all zeros after centring).

    glm_baseline_regressors(strict=True) refuses only a block that is entirely
    zero, so a single dead column passes it while still spending a degree of
    freedom.
    """
    B = _as_2d(bl)
    return np.flatnonzero(~np.any(B != 0.0, axis=0))
