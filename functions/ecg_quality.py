# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
ECG quality measures for the heartbeat analyses.

The clean ratio (`clean_ratio_segments`) is the quantity the subject gate reads. It is not
a signal measure: it is the fraction of detected R-peaks that RR cleaning rejected, counted
only over beats inside the analysed segments,

    ratio = 1 - n_kept / n_all      over beats inside `spans`

and a subject is kept only when `ratio <= ecgMax` (ecgMax = 0.2). `ibi_short_prop`, the
fraction of short inter-beat intervals, is computed here as a reported diagnostic.

NaN handling. Every comparison against NaN is False, so the two natural spellings of one
gate behave oppositely on an unassessable subject:

    keep form   x <= ecgMax   ->  False for NaN  ->  dropped
    fail form   x >  ecgMax   ->  False for NaN  ->  kept

`ecg_ratio_passes_gate` tests `np.isfinite` first and returns False before any comparison,
and a non-finite threshold raises. `clean_ratio_segments` raises on a non-finite peak index
or span bound, which would otherwise drop out of both n_all and n_kept and shrink the
denominator. An unassessable quantity is NaN, never 0.

Indexing. `clean_ratio_segments` needs peaks and spans in the same convention, 0- or
1-based, so a one-sample shift (4 ms at 250 Hz) does not change it. `ibi_short_prop` uses
only differences of peak indices.

R-peak detection and RR cleaning are not in this module: it takes `rpeaks_all` /
`rpeaks_clean` as inputs. The segment-count criterion of the subject gate needs the beat
labelling and is not here either; the caller (heart_depth_run, heart_scalp_pipeline)
applies it.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    'EcgQualityError',
    'ECG_MAX_DEFAULT', 'MIN_IBI_MS_DEFAULT',
    'CleanRatio',
    'clean_ratio_segments',
    'ecg_ratio_passes_gate', 'ibi_short_prop', 'subject_ecg_clean_ratio',
]


class EcgQualityError(ValueError):
    """Raised instead of returning a plausible-looking default.

    An unusable input must fail loudly: a broken ECG that silently reads as clean would
    keep a subject who should be dropped.
    """


# -- Constants -----------------------------------------------------------------
# Maximum fraction of detected R-peaks rejected by RR cleaning (subject gate).
ECG_MAX_DEFAULT = 0.2
# Inter-beat intervals shorter than this (ms) count as short.
MIN_IBI_MS_DEFAULT = 600.0


# =============================================================================
# Input check, where MATLAB and numpy genuinely disagree
# =============================================================================


def _as_finite_1d(v, what):
    """Flatten to 1-D and refuse non-finite entries.

    MATLAB's `x(:)` accepts NaN, and a containment test then drops it from every count
    without a warning. Here it raises instead.
    """
    a = np.asarray(v, dtype=float).ravel()
    if a.size and not np.all(np.isfinite(a)):
        bad = int(np.count_nonzero(~np.isfinite(a)))
        raise EcgQualityError(
            f'{what}: {bad}/{a.size} entries are not finite. A non-finite sample '
            f'index would silently vanish from BOTH counts and shrink the '
            f'denominator, so it is refused rather than dropped.')
    return a


# =============================================================================
# 1. The clean ratio: the quantity the subject gate reads
# =============================================================================

class CleanRatio:
    """`clean_ratio_segments` output: the ratio plus the two counts behind it.

    Log n_all/n_kept beside the ratio: 1 - 1/1 = 0.0 and 1 - 5336/5339 = 0.0006 are both
    "clean", and only the counts tell one usable beat from an assessed session.
    """
    __slots__ = ('ratio', 'n_all', 'n_kept', 'scope')

    def __init__(self, ratio, n_all, n_kept, scope):
        self.ratio, self.n_all, self.n_kept, self.scope = ratio, n_all, n_kept, scope

    def __repr__(self):
        return (f'CleanRatio(ratio={self.ratio:.6g}, n_all={self.n_all}, '
                f'n_kept={self.n_kept}, scope={self.scope!r})')


def clean_ratio_segments(rpeaks_all, rpeaks_clean, spans=None):
    """Fraction of detected R-peaks that RR cleaning rejected, over beats inside `spans`.

    ratio = 1 - n_kept / n_all.

    Parameters
    ----------
    rpeaks_all   : all detected peaks (get_RR's `rPeaks`), sample indices.
    rpeaks_clean : the surviving subset (clean_rr's `nPeaks`), sample indices.
                   The pipelines pass `raw_abs_all` and `qa_abs_all` as these two.
    spans        : (nSeg, 2) inclusive sample ranges, or None.

                   `spans=None` gives the whole-recording ratio, which is biased because
                   most of the recording is not analysed. It is a reporting quantity
                   only and is never gated on. Do not pass None because segment spans
                   were inconvenient to obtain.

    Indices may be 0- or 1-based, as long as peaks and spans use the same convention.

    Returns
    -------
    CleanRatio, with `ratio` NaN when no detected beat falls inside any span. NaN means
    "cannot be assessed", not "clean": returning 0 there would let the session through
    the gate on no evidence. `ecg_ratio_passes_gate` fails NaN.
    """
    all_p = _as_finite_1d(rpeaks_all, 'clean_ratio_segments(rpeaks_all)')
    clean_p = _as_finite_1d(rpeaks_clean, 'clean_ratio_segments(rpeaks_clean)')

    if all_p.size == 0:
        return CleanRatio(np.nan, 0, 0, 'no detected beats')

    if spans is None or (hasattr(spans, '__len__') and len(spans) == 0):
        n_all = int(all_p.size)
        n_kept = int(clean_p.size)
        return CleanRatio(1.0 - n_kept / n_all, n_all, n_kept, 'whole-recording (biased)')

    s = np.asarray(spans, dtype=float)
    if s.ndim != 2 or s.shape[1] != 2:
        raise EcgQualityError(
            f'clean_ratio_segments: spans must be (nSeg, 2), got shape {s.shape}. '
            f'MATLAB v7.3 arrays read back from h5py TRANSPOSED - a (2, nSeg) '
            f'block_spans needs .T.')
    if not np.all(np.isfinite(s)):
        raise EcgQualityError(
            'clean_ratio_segments: spans contain non-finite bounds; every '
            'containment test against them is False, so the segment would '
            'silently contribute nothing instead of failing.')
    if np.any(s[:, 1] < s[:, 0]):
        bad = int(np.count_nonzero(s[:, 1] < s[:, 0]))
        raise EcgQualityError(
            f'clean_ratio_segments: {bad} span(s) have end < start. An inverted '
            f'span contains no beat and would quietly deflate n_all.')

    # Membership against the union of spans, inclusive on both ends. No tolerance: peaks
    # and spans are both integer sample indices. Overlapping spans do not double-count,
    # because `any` collapses them to the union.
    in_all = np.any((all_p[:, None] >= s[:, 0]) & (all_p[:, None] <= s[:, 1]), axis=1)
    in_clean = np.any((clean_p[:, None] >= s[:, 0]) & (clean_p[:, None] <= s[:, 1]),
                      axis=1) if clean_p.size else np.zeros(0, dtype=bool)

    n_all = int(np.count_nonzero(in_all))
    n_kept = int(np.count_nonzero(in_clean))

    if n_all == 0:
        return CleanRatio(np.nan, 0, n_kept, 'no beat inside any span (unassessable)')
    return CleanRatio(1.0 - n_kept / n_all, n_all, n_kept, 'segments')


# =============================================================================
# 2. The subject gate and the short-IBI diagnostic
# =============================================================================

def ecg_ratio_passes_gate(ratio, ecg_max=ECG_MAX_DEFAULT):
    """`ratio <= ecgMax`, with NaN failing. The one gate spelling used in this module.

    The fail masks `~(x <= ecgMax)` and `~isfinite(x) | x > ecgMax` agree on every value a
    fraction in [0, 1] can take. A bare `x > ecgMax` fail mask is False for NaN and
    therefore keeps an unassessable subject.

    Here the isfinite test runs first and returns, so no NaN reaches a comparison
    operator.
    """
    e = float(ecg_max)
    if not np.isfinite(e):
        # A NaN threshold would make every comparison False and fail the whole
        # cohort, which reads as a bad dataset instead of a bad config.
        raise EcgQualityError(f'ecg_ratio_passes_gate: ecg_max must be finite, got {ecg_max}')
    r = float(ratio)
    if not np.isfinite(r):
        return False              # unassessable is not clean
    return r <= e


def ibi_short_prop(rpeaks, fs, min_ibi_ms=MIN_IBI_MS_DEFAULT):
    """Fraction of inter-beat intervals shorter than `min_ibi_ms`, one session.

    `rpeaks` is the clean R-peak train (the pipeline's RPEAKS_STORE_RAW, which despite its
    name is the clean train).

    Returns NaN, not 0, when fewer than 2 peaks are available: a session with no
    measurable heartbeat must not score "no short IBIs at all".
    """
    p = _as_finite_1d(rpeaks, 'ibi_short_prop(rpeaks)')
    if p.size < 2:
        return np.nan
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise EcgQualityError(f'ibi_short_prop: fs must be finite and > 0, got {fs}')
    ibi_ms = np.diff(p) / fs * 1000.0
    return float(np.count_nonzero(ibi_ms < float(min_ibi_ms)) / ibi_ms.size)


def subject_ecg_clean_ratio(ratios):
    """Subject-level mean of per-observation clean ratios, over finite values only.

    A NaN observation had no detected beat inside any analysed segment and contributes no
    beats downstream anyway, so dropping it from the mean hides nothing. A subject with
    only such observations comes back NaN and fails, not 0.0 and passes.

    Returns (mean_ratio, n_unassessable).
    """
    r = np.asarray(ratios, dtype=float).ravel()
    finite = r[np.isfinite(r)]
    n_bad = int(r.size - finite.size)
    return (float(np.mean(finite)) if finite.size else np.nan), n_bad
