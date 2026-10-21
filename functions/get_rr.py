# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
get_rr: R-peak detection following the ECG branch (Pan-Tompkins) of BrainBeats' get_RR.

    peaks, info = get_rr(ecg, fs)

Why this detector. What rejects T-waves is the Pan-Tompkins energy front end:
differentiate, square, integrate over a 150 ms Hann window, and threshold at the 98th
percentile. That merges each QRS into one energy blob and leaves the T-wave (low slope,
hence low squared-derivative energy) below threshold. neurokit2's `neurokit` method uses a
different, gradient-based criterion which, on a slow heart with a tall T-wave, can admit
the T-wave as a second peak per beat (the inter-beat intervals then alternate between the
R-T and T-R intervals). The refractory period does not do this job: at its default of
0.25 s it is shorter than the R-T interval of a slow heart (~350 ms).

The steps, in get_RR's order:
  1. pchip-interpolate non-finite samples, then the 3-35 Hz zero-phase FIR
     bandpass (`rr_cleaning.qrs_bandpass`, which reproduces fir1's even-order
     bump, filtfilt's padlen and fir1's scaled Hamming window).
  2. flatline check (95th percentile of |x| < 0.05 -> error)
  3. P&T: [0; diff], square, zero-phase Hann integration over round(0.150*fs)
  4. energy threshold: 98th percentile of the integrated signal over samples
     fs..min(fs*90, N) (1-based, inclusive), times `peak_thresh`
  5. search-back: where an inter-candidate gap exceeds 1.5x the median, re-open
     that stretch at half the threshold
  6. QRS segment boundaries from the threshold mask
  7. polarity: one vote per segment on the dominant deflection, global majority,
     and the trace is flipped when the majority is negative
  8. peak localization: max (or min) of each segment per its polarity
  9. refractory dedup: within `ref_period`, keep the larger |amplitude|
 10. micro-refinement: snap to the nearest local extremum within 15 ms
 11. `unique(..., 'stable')`, then drop the first peak

Not included: the PPG branch, the adaptive chunked polarity mode (`ecg_adaptive_pol`),
and the progress printing.

Differences to know about:
* Scope. The pipeline calls this per analysed window (~60 beats at 60 s), not once over
  a whole recording block (~2000 beats), so step 4's percentile and step 5's median RR
  are estimated from ~60 beats. `info['en_thres']` carries the threshold actually used.
* `ref_period` defaults to 0.25 s, the value get_RR's code assigns; its header
  documents 0.35 s.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt
from scipy.signal.windows import hann

from rr_cleaning import qrs_bandpass

__all__ = ['get_rr', 'resolve_polarity', 'GetRRError']

# get_RR's code defaults (not its header's).
BP_CUTOFF_DEFAULT = (3.0, 35.0)
PEAK_THRESH_DEFAULT = 0.35
SEARCH_BACK_DEFAULT = True
REF_PERIOD_DEFAULT = 0.25          # seconds; see the module docstring
MICRO_REFINE_SEC = 0.015
INTEGRATION_SEC = 0.150            # the 150 ms integration window
ENERGY_PCTILE = 0.98
ENERGY_MAX_SEC = 90                # min(fs*90, nSamp)
SEARCHBACK_RATIO = 1.5
FLATLINE_P95 = 0.05


class GetRRError(RuntimeError):
    """Raised where get_RR raises an error: a flat line, or no usable signal."""


def _pchip_fill(x):
    """Interpolate non-finite samples before filtering, as get_RR does.

    `qrs_bandpass` raises on non-finite input rather than guessing, so the
    interpolation that get_RR does inline has to happen here, before it.
    """
    x = np.asarray(x, dtype=float).ravel()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x, 0
    if bad.all():
        raise GetRRError('get_rr: every ECG sample is non-finite.')
    from scipy.interpolate import PchipInterpolator
    t = np.arange(x.size, dtype=float)
    f = PchipInterpolator(t[~bad], x[~bad], extrapolate=True)
    out = x.copy()
    out[bad] = f(t[bad])
    return out, int(bad.sum())


def get_rr(signal, fs, times=None, *,
           bandpass=BP_CUTOFF_DEFAULT,
           peak_thresh=PEAK_THRESH_DEFAULT,
           search_back=SEARCH_BACK_DEFAULT,
           ref_period=REF_PERIOD_DEFAULT,
           polarity=None,
           flip_signal=True,
           drop_first=True):
    """R-peak sample indices, following the ECG branch of BrainBeats' get_RR.

    Parameters
    ----------
    signal : (N,) array. Raw or pre-filtered ECG. Non-finite samples are
        pchip-interpolated first, as get_RR does.
    fs : float. Sampling rate (Hz).
    times : (N,) array, optional. Sample times in seconds, used only by the
        search-back step. Defaults to `arange(N)/fs`, which is what a uniform
        axis gives and what the pipeline passes.
    bandpass : (hp, lp) Hz, or False to skip (already band-limited).
    peak_thresh : P&T energy threshold multiplier. get_RR's default is 0.35.
    ref_period : refractory period in seconds. get_RR's header (0.35) and its
        code (0.25) disagree; this follows the code. Do not raise it to 0.35
        without re-validating: on a slow heart the T-wave can fall between the
        two values (R-T ~348 ms), so 0.35 would start deduplicating real beats
        against their own T-wave, and it would move peak trains on ordinary
        recordings too. What rejects T-waves here is the P&T energy front end,
        not this parameter.
    polarity : None to vote (the default), or +1 / -1 to force. The pipeline
        forces it: polarity is a property of the electrode montage, not of a
        60 s window, and resolving it once per recording is both cheaper and
        steadier than voting per window on ~60 beats. `resolve_polarity` below
        is the session-level resolver.
    drop_first : get_RR drops the first peak on output so that
        `len(peaks) == len(RR)`. clean_rr's contract depends on it. Set False
        only if you want the untruncated train.

    Returns
    -------
    peaks : (M,) int array of sample indices into the original `signal`.
    info : dict with 'rr' (s), 'rr_t' (s), 'hr' (bpm), 'polarity',
        'en_thres', 'n_segments', 'n_dedup', 'n_nonfinite', 'filtered'.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise GetRRError(f'get_rr: fs must be finite and > 0, got {fs}')

    raw, n_nonfinite = _pchip_fill(signal)
    n = raw.size
    if times is None:
        times = np.arange(n, dtype=float) / fs
    else:
        times = np.asarray(times, dtype=float).ravel()
        if times.size != n:
            raise GetRRError(
                f'get_rr: times has {times.size} samples, signal has {n}.')

    # -- 1. BANDPASS --------------------------------------------------------
    if bandpass is False or bandpass is None:
        sig = raw.copy()
    else:
        sig = qrs_bandpass(raw, fs, bp=tuple(float(v) for v in bandpass))

    # -- 2. FLATLINE CHECK --------------------------------------------------
    if np.percentile(np.abs(sig), 95) < FLATLINE_P95:
        raise GetRRError('get_rr: ECG amplitude too small - likely a flat line.')

    # -- 3. P&T: DIFFERENTIATE, SQUARE, INTEGRATE ---------------------------
    # `[0; diff(signal)]` prepends a zero, so the derivative is the same
    # length as the signal and every later index stays in the signal's frame.
    d = np.concatenate(([0.0], np.diff(sig)))
    sq = d ** 2
    n_int = int(round(INTEGRATION_SEC * fs))
    if n_int < 2:
        raise GetRRError(f'get_rr: fs={fs} is too low for a 150 ms integrator.')
    # The integrator is the symmetric Hann window (zero at both ends), scipy's
    # hann(..., sym=True), as in get_RR; another window is a different filter.
    b_int = hann(n_int, sym=True)
    b_int = b_int / b_int.sum()
    # MATLAB's filtfilt pads with 3*(ntaps-1); scipy defaults to 3*ntaps. Passed
    # explicitly, the same correction qrs_bandpass makes.
    sig_int = filtfilt(b_int, 1.0, sq, padlen=3 * (b_int.size - 1))

    # -- 4. ENERGY THRESHOLD ------------------------------------------------
    # get_RR's `signal_filt(fs : min(fs*90, nSamp))` is 1-based and inclusive
    # (MATLAB indexing): elements fs..M become python [fs-1 : M]. Skips the
    # first second, caps at 90 s of statistics.
    lo = max(int(round(fs)) - 1, 0)
    hi = min(int(round(fs * ENERGY_MAX_SEC)), n)
    seg = sig_int[lo:hi] if hi > lo else sig_int
    if seg.size == 0:
        raise GetRRError('get_rr: no samples available for the energy threshold.')
    xs = np.sort(seg)
    en_thres = float(xs[int(np.ceil(ENERGY_PCTILE * xs.size)) - 1])
    poss = sig_int > (peak_thresh * en_thres)
    if not poss.any():
        raise GetRRError('get_rr: no QRS candidates found - check signal quality.')

    # -- 5. SEARCH-BACK FOR MISSED BEATS ------------------------------------
    if search_back:
        ind = np.flatnonzero(poss)
        if ind.size > 2:
            gaps = np.diff(times[ind])
            rrv = gaps[gaps > 0.01]
            if rrv.size:
                med = float(np.median(rrv))
                missed = np.flatnonzero(gaps > SEARCHBACK_RATIO * med)
                for m in missed:
                    a, b = ind[m], ind[m + 1]
                    poss[a:b + 1] |= sig_int[a:b + 1] > (0.5 * peak_thresh * en_thres)

    # -- 6. SEGMENT BOUNDARIES ----------------------------------------------
    pi = poss.astype(np.int8)
    left = np.flatnonzero(np.diff(np.concatenate(([0], pi))) == 1)
    right = np.flatnonzero(np.diff(np.concatenate((pi, [0]))) == -1)
    nb = int(min(left.size, right.size))
    if nb == 0:
        raise GetRRError('get_rr: no QRS segments found after thresholding.')
    left, right = left[:nb], right[:nb]

    # -- 7. POLARITY --------------------------------------------------------
    if polarity is not None:
        pol_global = int(np.sign(polarity)) or 1
    else:
        # One vote per QRS segment on the dominant deflection, majority wins.
        # Operates on the bandpassed windows, which is what makes it robust to
        # baseline wander and to T/P waves.
        votes = np.empty(nb, dtype=float)
        for i in range(nb):
            s = sig[left[i]:right[i] + 1]
            v = np.sign(s.max() - abs(s.min()))
            votes[i] = v if v != 0 else 1.0
        pol_global = int(np.sign(votes.sum())) or 1

    if pol_global < 0 and flip_signal:
        # Flip the trace so R is positive, then treat every
        # segment as positive. The time base is untouched, so returned indices
        # remain indices into the caller's original array.
        sig = -sig
        pol_seg = 1
    else:
        pol_seg = pol_global

    # -- 8. PEAK LOCALIZATION -----------------------------------------------
    peaks = np.empty(nb, dtype=np.int64)
    pkval = np.empty(nb, dtype=float)
    for i in range(nb):
        a, b = int(left[i]), int(right[i])
        s = sig[a:b + 1]
        j = int(np.argmax(s)) if pol_seg > 0 else int(np.argmin(s))
        peaks[i] = a + j
        pkval[i] = s[j]

    # -- 9. REFRACTORY DEDUP BY AMPLITUDE -----------------------------------
    order = np.argsort(peaks, kind='stable')
    peaks, pkval = peaks[order], pkval[order]
    keep = np.ones(peaks.size, dtype=bool)
    ref_n = int(round(float(ref_period) * fs))
    # Sequential and order-dependent, as in get_RR: each comparison is against
    # peaks[k-1] whether or not that one has already been dropped. Vectorising
    # this changes which of a close triplet survives.
    for k in range(1, peaks.size):
        if peaks[k] - peaks[k - 1] < ref_n:
            if abs(pkval[k]) > abs(pkval[k - 1]):
                keep[k - 1] = False
            else:
                keep[k] = False
    n_dedup = int((~keep).sum())
    peaks = peaks[keep]

    # -- 10. MICRO-REFINEMENT, +/-15 ms -------------------------------------
    micro = max(1, int(round(MICRO_REFINE_SEC * fs)))
    for k in range(peaks.size):
        w1 = max(0, int(peaks[k]) - micro)
        w2 = min(n, int(peaks[k]) + micro + 1)
        s = sig[w1:w2]
        j = int(np.argmax(s)) if pol_seg > 0 else int(np.argmin(s))
        peaks[k] = w1 + j

    # unique(..., 'stable') keeps the first occurrence, which after a sort is
    # the same as sorted-unique here, but spell it the same way.
    _, first = np.unique(peaks, return_index=True)
    peaks = peaks[np.sort(first)]

    # -- 11. RR / HR, AND THE DROPPED FIRST PEAK ----------------------------
    rr = np.diff(peaks) / fs
    rr_t = times[peaks] if peaks.size else np.array([], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        hr = 60.0 / rr if rr.size else np.array([], float)
    if drop_first and peaks.size:
        # get_RR drops the first peak so that len(peaks) == len(RR), which is
        # the contract clean_rr checks. Dropping it here rather than in the
        # caller keeps the two trains the same length by construction.
        peaks = peaks[1:]
        rr_t = rr_t[1:]

    info = dict(rr=rr, rr_t=rr_t, hr=hr, polarity=pol_global,
                en_thres=en_thres, n_segments=nb, n_dedup=n_dedup,
                n_nonfinite=n_nonfinite, filtered=sig)
    return peaks.astype(np.int64), info


def resolve_polarity(ecg, fs, *, default=1, **kw):
    """+1 if the R-wave points up in this recording, -1 if the lead is inverted.

    Session-level resolver. Call it once on a concatenation of the recording's
    analysed segments, then pass `polarity=1` to `get_rr` on the pre-flipped
    windows. Voting inside every 45-60 s window instead would decide on ~50
    beats rather than several hundred and could disagree with itself between
    two windows of one recording; polarity is a property of the electrode
    montage, not of a window.

    This is get_RR's own rule: it runs the same Pan-Tompkins front end the
    pipeline detects with, then takes one vote per QRS segment on the dominant
    deflection of the 3-35 Hz bandpassed trace and lets the global majority
    decide (`get_rr` step 7). The neurokit2 detector (`PEAK_SOURCE='neurokit2'`,
    `heart_core.detect_rpeaks`) uses `heart_core.ecg_polarity` instead: a
    detector should resolve polarity with its own front end, not another one's.

    Parameters
    ----------
    ecg : (N,) array. Raw ECG, not pre-flipped.
    fs : float. Sampling rate (Hz).
    default : returned when the vote cannot be taken (flat line, no QRS
        candidate above threshold). +1 matches `ecg_polarity`'s own error path;
        a session too poor to vote on is one the beat gates drop anyway.
    **kw : forwarded to `get_rr` (e.g. `bandpass`).

    Returns
    -------
    int, +1 or -1.
    """
    try:
        _, info = get_rr(np.asarray(ecg, float).ravel(), fs,
                         polarity=None, drop_first=False, **kw)
    except GetRRError:
        return int(default)
    pol = int(np.sign(info.get('polarity', default)))
    return pol if pol != 0 else int(default)
