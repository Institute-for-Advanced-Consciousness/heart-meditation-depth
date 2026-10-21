# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
RR-interval cleaning: a Python implementation of `clean_rr` from the BrainBeats toolbox.

Cleans an RR-interval series and optionally interpolates missed heartbeats, returning the NN
series and the surviving R-peak sample indices. It turns the raw detections of `get_rr` into the
beat train that the heartbeat-locked analyses epoch on, so every HEP, HRSP and HEPC latency
depends on it (HEPC, as the code names it, is the paper's HRPC, heartbeat-related phase
consistency).

CALL USED IN THIS PROJECT
  Every threshold at its default (mad_threshold 10, amp_mad_threshold 15, win_size 15 s,
  gap_ratio 1.5) and interpolate_missing=False. Step 5 therefore never runs in the analyses; it
  is implemented anyway.

VALIDATION
  On 70 reference recordings cleaned by BrainBeats, this module reproduces the cleaned peaks, the
  gap flags and the NN series exactly, element by element.

STEP 2 NEEDS peak_amp
  Step 2 reads peak amplitudes off the 3-35 Hz bandpassed trace that get_rr detects on.
  `qrs_bandpass()` rebuilds that trace so peak_amp can be computed from the ECG. Step 2 is not
  inert: on the reference recordings it removed 200 beats, 185 of which Step 3 does not catch.
  The polarity flip of get_rr is not needed, because abs(a - median(a)) > k*mad(a) is
  invariant under a global sign change.

BEHAVIOUR KEPT FROM clean_rr
  1. `max_rr` is computed and reported but never used. Step 3 is `rr < min_rr` only: long
     intervals are never removed, only flagged (Step 4) and, if interpolation is on, filled.
  2. The lower limit is min_rr = max(med - 10*MAD, 0.30 s). On a low-variability recording
     10*MAD is small, so min_rr can lie far above 0.30 s (up to 0.900 s, about 67 bpm, on the
     reference recordings). This short-interval rule is the main removal mechanism: 1381 removals
     against 185 amplitude-only removals over 277320 intervals.
  3. After gap filling, the filled interval is neither `bad` nor `interp`. Step 5 clears
     idx_bad[i] and writes the sub-interval into slot i, then marks only the n_missing inserted
     slots as interp, so the summary counts the original gap slot as clean (see `_fill_gaps`).

OTHER BEHAVIOUR
  * NaN RR intervals survive. `NaN < min_rr` and `NaN > 1.5*local_med` are both False, so a
    non-finite interval is neither removed nor flagged and counts as clean.
    info['n_nonfinite_rr'] reports them.
  * A MAD of 0 inverts the test. If more than half the intervals (or amplitudes) are identical,
    MAD is 0 and the threshold collapses onto the median: for RR this deletes everything below
    the median, for amplitude everything not exactly at the median. This is flagged in
    info['mad_rr_zero'] / info['amp_mad_zero'] and by a RuntimeWarning.
  * Peak indices are 1-based by default, as in BrainBeats. The module returns peaks in the base
    it is given (`index_base` only affects synthetic peaks created in Step 5 and the default
    rr_t). Subtract 1 before indexing a numpy array with 1-based peaks.
  * A median RR of 0 is legal and does not raise: the verbose printout goes through `_bpm()`,
    which returns Inf instead of dividing by zero.
  * `peaks_clean` is returned as float64 because Step 5 can emit NaN peaks (interpolation without
    an ECG). Cast to int64 after checking isfinite().

Ported with stricter input checks
  Two calls that BrainBeats' clean_rr accepts are rejected here:
  * `fs` is required (third positional argument) and must be finite and > 0. In BrainBeats fs is
    optional and only places synthetic peaks, never a threshold. As a consequence the
    `np.isfinite(fs)` term in `use_ecg` is always True.
  * `len(sig_t) != len(ecg_signal)` raises. BrainBeats never compares them.

NOT IMPLEMENTED
  * R-peak detection (see `get_rr`). The one piece of it reproduced here is `qrs_bandpass`,
    because peak_amp cannot be computed without it.
  * The legacy PhysioNet cleaning path of BrainBeats.
  * MATLAB's input validators. Type errors surface anyway; the length and finiteness checks that
    matter are explicit and raise.
  * The progress printout is opt-in (`verbose=True`); the same numbers are always in `info`.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ['clean_rr', 'qrs_bandpass',
           'DEFAULTS', 'RR_HARD_MIN', 'RR_HARD_MAX']

# Hard physiological limits (200 bpm / 30 bpm), applied after the data-driven MAD limits. They
# are not tuning parameters. RR_HARD_MAX only feeds `max_rr`, which is never used (see the module
# docstring); it is kept so info['max_rr'] reports the same value as the BrainBeats printout.
RR_HARD_MIN = 0.30   # s -> 200 bpm
RR_HARD_MAX = 2.00   # s ->  30 bpm

# Defaults of BrainBeats' clean_rr. The analyses override only `interpolate_missing`.
DEFAULTS = dict(
    interpolate_missing=True,
    max_missing=np.inf,
    mad_threshold=10.0,
    amp_mad_threshold=15.0,
    win_size=15.0,
    gap_ratio=1.5,
)


# ---------------------------------------------------------------------------
# MATLAB-semantics primitives
# ---------------------------------------------------------------------------
def _matlab_round(x):
    """MATLAB `round`: half away from zero (numpy rounds half to even).

    Step 5 uses `round(gap / ref_rr) - 1` to decide how many beats to insert; at
    gap/ref_rr == 2.5 this inserts 2 beats where np.round would insert 1.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def _nanmedian(x):
    """median(x, 'omitnan'). Returns NaN, without a warning, for an all-NaN or empty input.

    The consequence is exposed through info['mad_rr_zero'] and the emptiness checks instead.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmedian(x))


def _mad1(x):
    """MATLAB `mad(x, 1)`: the median absolute deviation, not scaled by 1.4826.

    MATLAB's `mad(x)` without the flag is the mean absolute deviation; clean_rr uses the median
    form in both places. Using the wrong one changes min_rr by about 25% and so how many beats
    are removed. NaNs are omitted.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return _nanmedian(np.abs(x - _nanmedian(x)))


def _matlab_max2(a, b):
    """Two-argument MATLAB `max`: NaN is ignored, not propagated.

    max(NaN, 0.30) is 0.30 in MATLAB and NaN under np.maximum. This branch is taken when every RR
    is NaN, and makes that case remove nothing instead of failing on a NaN comparison.
    """
    if np.isnan(a):
        return float(b)
    if np.isnan(b):
        return float(a)
    return float(max(a, b))


def _matlab_min2(a, b):
    """Two-argument MATLAB `min`: NaN ignored. See _matlab_max2."""
    if np.isnan(a):
        return float(b)
    if np.isnan(b):
        return float(a)
    return float(min(a, b))


def _bpm(rr_seconds):
    """60 / rr in bpm, returning +/-Inf for 0 instead of raising.

    Only the verbose printout uses it. `60/med_rr` and `60/max_rr` can both divide by zero on a
    degenerate but legal input: if more than half the intervals are 0 s, med_rr = 0,
    mad_rr = 0 and max_rr = min(0 + 10*0, 2.00) = 0, and the series is still cleaned
    (rr=[0 0 0 1 1 0] returns nn=[1 1]). The f-strings passed to `log` are built even when
    verbose=False, so plain division would abort the call. min_rr is clamped to >= 0.30 s and
    cannot hit this; it goes through _bpm for uniformity.
    """
    rr_seconds = float(rr_seconds)
    if rr_seconds == 0.0:
        # copysign so -0.0 gives -Inf, as 60/-0 does.
        return float(np.copysign(np.inf, rr_seconds))
    return 60.0 / rr_seconds


def _find_nearest_sample(sig_t, t, index_base=1):
    """Index of the sample of `sig_t` nearest to time `t`.

    On a tie the first index is returned (np.argmin). Returned in the caller's index base so a
    synthetic peak is directly comparable with the real ones.
    """
    return int(np.argmin(np.abs(np.asarray(sig_t, dtype=float) - float(t)))) + index_base


# ---------------------------------------------------------------------------
# Cached FIR initial conditions
# ---------------------------------------------------------------------------
# `scipy.signal.filtfilt(..., method='pad')` calls `lfilter_zi(b, a)` on every call, which solves
# a dense (n-1)x(n-1) system: its cost is O(order^3) and does not depend on the signal length.
# The highpass of `qrs_bandpass` has 251 taps at 250 Hz, so computing the initial conditions
# takes about 90% of each filtfilt call, and `qrs_bandpass` runs once per segment and again
# inside `get_rr`, always with the same filter.
#
# The result is bit-identical, not merely close: `zi` is a pure function of (b, a), so the cache
# returns the same vector, and `_filtfilt_fir` is scipy's own 1-D `filtfilt` method='pad' path,
# with the same operations in the same order.
_ZI_CACHE = {}


def _fir_zi_cached(b):
    """`lfilter_zi(b, [1.0])`, computed once per coefficient vector.

    Keyed on the coefficient bytes, not on (fs, cutoff): two designs that produce the same taps
    share an entry, and two that differ anywhere do not. `b.tobytes()` is exact on both counts.
    """
    from scipy.signal import lfilter_zi
    key = b.tobytes()
    zi = _ZI_CACHE.get(key)
    if zi is None:
        zi = lfilter_zi(b, [1.0])
        _ZI_CACHE[key] = zi
    return zi


def _filtfilt_fir(b, x, padlen):
    """`scipy.signal.filtfilt(b, [1.0], x, padtype='odd', padlen=padlen)`.

    Same operations in the same order, with the `lfilter_zi` call served from `_ZI_CACHE`. FIR
    only (a = [1.0]), which is all this module filters with.
    """
    from scipy.signal import lfilter
    from scipy.signal._arraytools import odd_ext, axis_slice, axis_reverse
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    if padlen >= x.shape[-1]:
        # scipy raises for this; let it raise with its own message rather than return
        # something shorter.
        from scipy.signal import filtfilt as _ff
        return _ff(b, [1.0], x, padtype='odd', padlen=padlen)
    ext = odd_ext(x, padlen, axis=-1)
    zi = _fir_zi_cached(b)
    zi_shape = [1] * x.ndim
    zi_shape[-1] = zi.size
    zi = np.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=-1)
    y, _zf = lfilter(b, [1.0], ext, axis=-1, zi=zi * x0)
    y0 = axis_slice(y, start=-1, axis=-1)
    y, _zf = lfilter(b, [1.0], axis_reverse(y, axis=-1), axis=-1, zi=zi * y0)
    y = axis_reverse(y, axis=-1)
    return axis_slice(y, start=padlen, stop=-padlen, axis=-1)


# ---------------------------------------------------------------------------
# QRS-band filter of get_rr, needed to rebuild peak_amp
# ---------------------------------------------------------------------------
def qrs_bandpass(signal, fs, bp=(3.0, 35.0)):
    """The 3-35 Hz zero-phase FIR bandpass that `get_rr` detects R peaks on.

    Two cascaded `fir1` + `filtfilt` stages (highpass then lowpass) of orders 3*round(fs/hp) and
    5*round(fs/lp). This is not a peak detector: it exists because clean_rr's Step 2 reads
    amplitudes off this filtered trace, so peak_amp cannot be computed without it.

    Three details reproduce MATLAB's filter design:
      * fir1 refuses an even tap count for a highpass (zero gain at Nyquist) and raises the order
        by one: fs=250 gives hp_order 249 -> 250, so 251 taps. Without that step every amplitude
        is wrong.
      * MATLAB filtfilt pads with padlen = 3*(ntaps-1); scipy defaults to 3*ntaps. Passed
        explicitly.
      * fir1's default window is a symmetric Hamming with scaling on, which is scipy's
        firwin(window='hamming', scale=True).

    The polarity flip of get_rr is not applied: Step 2's test is sign-invariant (see the module
    docstring).

    Parameters
    ----------
    signal : 1-D array. The trace given to get_rr (a slice of EEG.ECG.filt_data).
    fs     : Hz.
    bp     : (hp, lp) cutoffs in Hz. Defaults are those of get_rr.

    Returns
    -------
    float64 array, same length as `signal`.
    """
    from scipy.signal import filtfilt, firwin  # local: keeps import cost off the core path

    x = np.asarray(signal, dtype=float).ravel()
    fs = float(fs)
    if x.size == 0:
        raise ValueError('qrs_bandpass: empty signal.')
    if not np.all(np.isfinite(x)):
        # get_rr pchip-interpolates non-finite samples before filtering. This function does not
        # guess: a NaN here means the caller passed a different signal, and filtfilt would
        # return an all-NaN trace anyway.
        raise ValueError(f'qrs_bandpass: {int((~np.isfinite(x)).sum())} non-finite '
                         'samples. get_RR interpolates these before filtering; '
                         'do the same upstream rather than letting NaN spread.')

    hp_order = 3 * int(_matlab_round(fs / bp[0]))
    if hp_order % 2:                      # fir1: a highpass needs an odd tap count
        hp_order += 1
    b_hp = firwin(hp_order + 1, bp[0] / (fs / 2.0),
                  window='hamming', pass_zero=False, scale=True)
    y = _filtfilt_fir(b_hp, x, 3 * (len(b_hp) - 1))

    lp_order = 5 * int(_matlab_round(fs / bp[1]))
    b_lp = firwin(lp_order + 1, bp[1] / (fs / 2.0),
                  window='hamming', pass_zero=True, scale=True)
    return _filtfilt_fir(b_lp, y, 3 * (len(b_lp) - 1))


# ---------------------------------------------------------------------------
# Step 5 (gap filling), kept separate because the analyses do not use it
# ---------------------------------------------------------------------------
def _fill_gaps(nn, nn_t, npk, idx_bad, idx_interp, local_med,
               max_missing, use_ecg, sig_t, index_base, log):
    """Step 5 of clean_rr: reverse-order insertion of synthetic beats.

    Gaps are iterated from last to first because inserting at index i shifts everything above it,
    so a forward loop would corrupt the remaining indices.

    The filled slot is neither flagged nor marked synthetic: `nn[i]` is overwritten with the
    sub-interval and `idx_bad[i]` cleared, but only the n_missing inserted slots get
    idx_interp=True. So one of the (n_missing+1) sub-intervals of each gap is counted as real
    data in the summary. This is the behaviour of BrainBeats' clean_rr; changing it changes
    every downstream count.
    """
    gap_indices = np.flatnonzero(idx_bad)
    n_interp_total = 0

    for i in gap_indices[::-1]:                       # reverse: see note above
        if i < 1:                                     # the first interval has no previous
            continue                                  # beat to anchor on (see below)
        gap = float(nn[i])
        ref_rr_i = float(local_med[i])
        # A non-finite gap or a non-positive local median gives an infinite or undefined number
        # of beats to insert, so raise rather than return an unfilled series that looks filled.
        # This needs a non-positive or non-finite interval in the input, which get_rr output
        # cannot produce (peaks unique and sorted, so rr >= 1/fs).
        if not np.isfinite(gap) or not np.isfinite(ref_rr_i) or ref_rr_i <= 0:
            raise ValueError(
                f'clean_rr: cannot fill the gap at index {int(i)} '
                f'(t={nn_t[i]:.3f} s): gap={gap!r} s, local median={ref_rr_i!r} s. '
                'round(gap / local_median) is not a finite beat count, so there is '
                'no defined number of beats to insert (MATLAB errors here too). '
                'This needs a non-positive or non-finite RR interval in the input '
                '- fix the beat train upstream instead of interpolating over it.')
        n_missing = int(_matlab_round(gap / ref_rr_i)) - 1
        if n_missing < 1:
            continue
        if n_missing > max_missing:
            log(f'  Gap at t={nn_t[i]:.2f} s: {n_missing} beats missing, exceeds '
                f'max_missing={max_missing:g} - skipping')
            continue

        j_vec = np.arange(1, n_missing + 1, dtype=float)
        sub_iv = gap / (n_missing + 1)
        # Anchored on nn_t[i-1] (the previous beat), not nn_t[i]; hence i >= 1 above.
        new_t = nn_t[i - 1] + j_vec * sub_iv

        if use_ecg:
            new_peaks = np.array([_find_nearest_sample(sig_t, t, index_base)
                                  for t in new_t], dtype=float)
        else:
            new_peaks = np.full(n_missing, np.nan)

        nn[i] = sub_iv
        idx_bad[i] = False
        nn = np.concatenate([nn[:i], np.full(n_missing, sub_iv), nn[i:]])
        nn_t = np.concatenate([nn_t[:i], new_t, nn_t[i:]])
        npk = np.concatenate([npk[:i], new_peaks, npk[i:]])
        idx_bad = np.concatenate([idx_bad[:i], np.zeros(n_missing, bool), idx_bad[i:]])
        idx_interp = np.concatenate([idx_interp[:i], np.ones(n_missing, bool), idx_interp[i:]])
        local_med = np.concatenate([local_med[:i], np.full(n_missing, ref_rr_i), local_med[i:]])

        n_interp_total += n_missing
        log(f'  Gap at t={nn_t[i + n_missing]:.2f} s ({gap:.3f} s): inserted '
            f'{n_missing} beat(s) at {sub_iv:.3f} s intervals')

    return nn, nn_t, npk, idx_bad, idx_interp, local_med, n_interp_total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def clean_rr(rr, peaks, fs, *,
             rr_t=None,
             peak_amp=None,
             ecg_signal=None,
             sig_t=None,
             interpolate_missing=None,
             max_missing=None,
             mad_threshold=None,
             amp_mad_threshold=None,
             win_size=None,
             gap_ratio=None,
             index_base=1,
             verbose=False):
    """Clean an RR series (BrainBeats clean_rr).

    Parameters
    ----------
    rr : (N,) float
        RR interval durations in seconds.
    peaks : (N,) int/float
        R-peak sample indices, one per interval. Each interval is paired with its trailing peak
        (get_rr drops the first peak and the first RR_t on output), so len(peaks) == len(rr).
        1-based by default; see `index_base`.
    fs : float
        Sampling rate (Hz). Used for the default `rr_t`/`sig_t` axes only; the thresholds are
        all in seconds.
    rr_t : (N,) float, optional
        Interval timestamps in seconds. Defaults to `(peaks - index_base) / fs`. Only
        differences of rr_t are used (Step 4's +/-7.5 s window), so any common origin gives
        identical results.
    peak_amp : (N,) float, optional
        Amplitude of each peak on the bandpassed trace of get_rr. Rebuild it with
        `qrs_bandpass(ecg, fs)[peaks - index_base]`. If omitted, Step 2 is skipped, which is far
        from inert: on the reference recordings Step 2 removed 200 beats, 185 of them beats
        that Step 3 does not catch. A skip is recorded in
        info['amp_step'].
    ecg_signal, sig_t : optional
        Only used by Step 5 to place synthetic peaks. `sig_t` defaults to `(0:len(ecg)-1)/fs`
        when `ecg_signal` is given. Without an ECG, synthetic peaks are NaN.
    interpolate_missing, max_missing, mad_threshold, amp_mad_threshold,
    win_size, gap_ratio :
        Options of clean_rr; None means "use DEFAULTS". The analyses pass
        interpolate_missing=False and take every other default.
    index_base : int
        1 (default) or 0. Affects only the default `rr_t` and the sample indices invented for
        synthetic beats.
    verbose : bool
        Print the cleaning summary on stdout.

    Returns
    -------
    rr_clean : (M,) float64
        The NN series.
    peaks_clean : (M,) float64
        Surviving peak indices, same base as the input. float64 (not int) because Step 5 can
        emit NaN; cast after checking isfinite().
    idx_bad : (M,) bool
        Intervals still flagged as gaps after any filling.
    info : dict
        `nn_t`, `idx_interp`, plus diagnostics: med_rr / mad_rr / min_rr / max_rr,
        per-original-index masks (idx_too_short, idx_amp_outlier, idx_removed), local_med,
        counts, clean_ratio, amp_step, and degeneracy flags.

    Raises
    ------
    ValueError
        On empty input, a length mismatch between rr/peaks/rr_t or between sig_t/ecg_signal, or
        a non-positive or non-finite fs; and from Step 5 when a flagged gap has a non-finite or
        non-positive local median, so the number of beats to insert is undefined. See "PORTED
        with stricter input checks" in the module docstring for the two checks
        BrainBeats does not make.
    """
    log = (lambda s: print(s)) if verbose else (lambda s: None)

    p = dict(DEFAULTS)
    for k, v in (('interpolate_missing', interpolate_missing),
                 ('max_missing', max_missing),
                 ('mad_threshold', mad_threshold),
                 ('amp_mad_threshold', amp_mad_threshold),
                 ('win_size', win_size),
                 ('gap_ratio', gap_ratio)):
        if v is not None:
            p[k] = v
    p['interpolate_missing'] = bool(p['interpolate_missing'])

    # ---- input contract (raise on a broken call) -----------------------------
    rr = np.asarray(rr, dtype=float).ravel()
    npk = np.asarray(peaks, dtype=float).ravel()
    if rr.size == 0:
        raise ValueError('clean_rr: rr is empty. Nothing to clean - this is a '
                         'caller bug (an empty block should be skipped upstream), '
                         'not cleaned into an empty null.')
    if npk.size != rr.size:
        raise ValueError(f'clean_rr: len(peaks)={npk.size} != len(rr)={rr.size}. '
                         'get_RR drops the first peak so these must match; a '
                         'mismatch means the first-beat convention was applied '
                         'twice, or not at all.')
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'clean_rr: fs must be finite and > 0 (got {fs!r}).')
    if index_base not in (0, 1):
        raise ValueError(f'clean_rr: index_base must be 0 or 1 (got {index_base!r}).')

    if rr_t is None:
        # See the parameter docs: only differences of rr_t matter, so the
        # peak-derived axis is exact for a uniformly sampled recording.
        nn_t = (npk - index_base) / fs
    else:
        nn_t = np.asarray(rr_t, dtype=float).ravel()
        if nn_t.size != rr.size:
            raise ValueError(f'clean_rr: len(rr_t)={nn_t.size} != len(rr)={rr.size}.')

    if ecg_signal is not None:
        ecg_signal = np.asarray(ecg_signal, dtype=float).ravel()
    if sig_t is not None:
        sig_t = np.asarray(sig_t, dtype=float).ravel()

    # use_ecg gates synthetic-peak placement.
    use_ecg = ecg_signal is not None and ecg_signal.size > 0 and np.isfinite(fs)
    if use_ecg and sig_t is None:
        sig_t = np.arange(ecg_signal.size, dtype=float) / fs
    if use_ecg and sig_t is not None and sig_t.size != ecg_signal.size:
        raise ValueError(f'clean_rr: len(sig_t)={sig_t.size} != '
                         f'len(ecg_signal)={ecg_signal.size}.')

    n_orig = rr.size
    nn = rr.copy()
    npk = npk.copy()
    idx_bad = np.zeros(n_orig, dtype=bool)
    idx_interp = np.zeros(n_orig, dtype=bool)
    n_nonfinite_rr = int((~np.isfinite(rr)).sum())

    # ---- Step 1: data-driven physiological limits -------------------------
    med_rr = _nanmedian(nn)
    mad_rr = _mad1(nn)
    min_rr = med_rr - p['mad_threshold'] * mad_rr
    max_rr = med_rr + p['mad_threshold'] * mad_rr
    # Two-argument max/min that ignore NaN, so an all-NaN series removes nothing.
    min_rr = _matlab_max2(min_rr, RR_HARD_MIN)
    max_rr = _matlab_min2(max_rr, RR_HARD_MAX)

    mad_rr_zero = bool(np.isfinite(mad_rr) and mad_rr == 0)
    if mad_rr_zero:
        warnings.warn(
            'clean_rr: MAD of the RR series is exactly 0, so min_rr collapses '
            f'onto the median ({med_rr:.3f} s) and Step 3 will delete EVERY '
            'interval below it. This is faithful to clean_rr.m but it is a '
            'degenerate input, not a clean recording.', RuntimeWarning, stacklevel=2)

    log('\n--- clean_rr ---')
    # _bpm, not `60 / x`: med_rr and max_rr can legitimately be 0, and these f-strings are built
    # whatever `verbose` says (see _bpm).
    log(f'  Median RR:    {med_rr:.3f} s ({_bpm(med_rr):.0f} bpm)')
    log(f'  MAD:          {mad_rr:.3f} s')
    log(f'  Min RR (data-derived): {min_rr:.3f} s ({_bpm(min_rr):.0f} bpm)')
    log(f'  Max RR (data-derived): {max_rr:.3f} s ({_bpm(max_rr):.0f} bpm)')

    # ---- Step 2: abnormal peak amplitude ----------------------------------
    # Deliberately lenient (15 MAD): respiration and posture modulate ECG amplitude constantly,
    # so only disconnections and rails should be caught. Set amp_mad_threshold=inf to disable.
    amp_med = np.nan
    amp_mad = np.nan
    amp_mad_zero = False
    if peak_amp is None:
        amp_outliers = np.zeros(n_orig, dtype=bool)
        amp_step = 'skipped:not_provided'
    elif not np.isfinite(p['amp_mad_threshold']):
        amp_outliers = np.zeros(n_orig, dtype=bool)
        amp_step = 'skipped:threshold_inf'
    else:
        amp = np.asarray(peak_amp, dtype=float).ravel()
        if amp.size != n_orig:
            # A length mismatch skips Step 2, as in BrainBeats' clean_rr. Here the skip is warned
            # about and recorded in info['amp_step'], so the output cannot pass for an
            # amplitude-screened series.
            warnings.warn(
                f'clean_rr: len(peak_amp)={amp.size} != len(rr)={n_orig}. MATLAB '
                'silently skips the entire amplitude step here - reproduced, but '
                'this means Step 2 DID NOT RUN. Fix the caller; do not read the '
                'output as an amplitude-screened series.', RuntimeWarning, stacklevel=2)
            amp_outliers = np.zeros(n_orig, dtype=bool)
            amp_step = 'skipped:length_mismatch'
        else:
            amp_med = _nanmedian(amp)
            amp_mad = _mad1(amp)
            amp_mad_zero = bool(np.isfinite(amp_mad) and amp_mad == 0)
            if amp_mad_zero:
                warnings.warn(
                    'clean_rr: MAD of peak amplitudes is exactly 0, so the '
                    'amplitude test degenerates to "not equal to the median" and '
                    'will remove nearly every beat. Degenerate input.',
                    RuntimeWarning, stacklevel=2)
            # NaN amplitudes compare False and therefore survive.
            with np.errstate(invalid='ignore'):
                amp_outliers = np.abs(amp - amp_med) > p['amp_mad_threshold'] * amp_mad
            amp_outliers = np.asarray(amp_outliers, dtype=bool)
            amp_step = 'applied'
            if amp_outliers.any():
                log(f'  Removing {int(amp_outliers.sum())} beats with abnormal '
                    f"amplitude (>{p['amp_mad_threshold']:.0f} MAD from median)")

    # ---- Step 3: remove impossibly SHORT intervals ------------------------
    # Only the short side. `max_rr` above is computed, reported and never used again: long
    # intervals are handled by Steps 4-5 (flag, then optionally fill), never deleted.
    with np.errstate(invalid='ignore'):
        too_short = np.asarray(nn < min_rr, dtype=bool)   # NaN -> False
    if too_short.any():
        log(f'  Removing {int(too_short.sum())} intervals below min RR ({min_rr:.3f} s)')

    to_remove = amp_outliers | too_short
    n_removed = int(to_remove.sum())
    keep = ~to_remove
    nn = nn[keep]
    nn_t = nn_t[keep]
    npk = npk[keep]
    idx_bad = idx_bad[keep]
    idx_interp = idx_interp[keep]

    info = dict(
        nn_t=nn_t, idx_interp=idx_interp,
        med_rr=med_rr, mad_rr=mad_rr, min_rr=min_rr, max_rr=max_rr,
        idx_too_short=too_short, idx_amp_outlier=amp_outliers, idx_removed=to_remove,
        amp_med=amp_med, amp_mad=amp_mad, amp_step=amp_step,
        amp_mad_zero=amp_mad_zero, mad_rr_zero=mad_rr_zero,
        n_orig=n_orig, n_removed=n_removed, n_nonfinite_rr=n_nonfinite_rr,
        local_med=np.zeros(0), n_final=0, n_flagged=0, n_interp=0, n_clean=0,
        clean_ratio=1.0, params=dict(p), index_base=index_base, reason=None,
    )

    if nn.size == 0:
        # Warn and return, with a named reason rather than an empty array that a caller could
        # mistake for "nothing to do".
        info['reason'] = 'all_intervals_removed'
        warnings.warn(f'clean_rr: all {n_orig} intervals removed - check data '
                      'quality (info["reason"]="all_intervals_removed").',
                      RuntimeWarning, stacklevel=2)
        return nn, npk, idx_bad, info

    # ---- Step 4: gap detection via sliding-window local median ------------
    # Window is +/- win_size/2 (7.5 s by default) around each interval's own timestamp,
    # excluding itself. When no neighbour is in range the global median from Step 1 is used,
    # which is the median of the series before removal.
    n = nn.size
    local_med = np.empty(n, dtype=float)
    half_win = p['win_size'] / 2.0
    for i in range(n):
        in_win = np.abs(nn_t - nn_t[i]) <= half_win
        in_win[i] = False
        local_med[i] = _nanmedian(nn[in_win]) if in_win.any() else med_rr

    with np.errstate(invalid='ignore'):
        idx_bad = np.asarray(nn > p['gap_ratio'] * local_med, dtype=bool)

    if idx_bad.any():
        log(f"  Flagging {int(idx_bad.sum())} gaps (interval > "
            f"{p['gap_ratio']:.1f}x local median)")
    else:
        log('  No gaps detected')

    # ---- Step 5 (optional): fill gaps with synthetic beats ----------------
    n_interp_total = 0
    if p['interpolate_missing'] and idx_bad.any():
        nn, nn_t, npk, idx_bad, idx_interp, local_med, n_interp_total = _fill_gaps(
            nn, nn_t, npk, idx_bad, idx_interp, local_med,
            p['max_missing'], use_ecg, sig_t, index_base, log)
        if n_interp_total > 0:
            log(f'  Total synthetic beats inserted: {n_interp_total}')
            if not use_ecg:
                log("  Pass 'ecg_signal' and 'fs' for accurate peak placement.")
        else:
            log('  No gaps were suitable for filling')

    # ---- Summary ----------------------------------------------------------
    n_final = nn.size
    n_flagged = int(idx_bad.sum())
    n_interp = int(idx_interp.sum())
    n_clean = int((~idx_bad & ~idx_interp).sum())
    log(f'  Removed:   {n_removed}/{n_orig} ({100 * n_removed / n_orig:.1f}%)')
    log(f'  Flagged:   {n_flagged}/{n_final} ({100 * n_flagged / n_final:.1f}%)')
    log(f'  Synthetic: {n_interp}/{n_final} ({100 * n_interp / n_final:.1f}%)')
    log(f'  Clean:     {n_clean}/{n_final} ({100 * n_clean / n_final:.1f}%)')
    log('----------------')

    info.update(nn_t=nn_t, idx_interp=idx_interp, local_med=local_med,
                n_final=n_final, n_flagged=n_flagged, n_interp=n_interp,
                n_clean=n_clean,
                # The fraction removed, despite the name.
                clean_ratio=1.0 - n_final / max(n_orig, 1))
    return nn, npk, idx_bad, info
