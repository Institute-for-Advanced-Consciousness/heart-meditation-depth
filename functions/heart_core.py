# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
heart_core: core signal computations of the heart-locked pipeline.

Turns R-peak-locked epochs into the three measures (HEP, HRSP, HEPC; HEPC, as the code
names it, is the paper's HRPC, heartbeat-related phase consistency):
    morlet_kernels   complex Morlet kernels, amplitude-normalised
    morlet_cwt       FFT convolution with each kernel centred at index 0 (zero phase)
    ppc_map          unbiased pairwise phase consistency (Vinck et al. 2010)
    hep_average      time-domain mean over beats
    epoch_beats      fixed window around each R-peak
    reconstruct_ics  IC activations as EEGLAB computes them, sign-corrected
plus a cardiac-field topography diagnostic (cfa_topo_template, cfa_topo_r2) and the
neurokit2 R-peak detector with polarity handling (detect_rpeaks, ecg_polarity). The
pipelines' default R-peak detector is get_rr (BrainBeats); neurokit2 is the alternative.

Zero phase matters. A naive FFT convolution delays the output by
3*sigma = 3*width/(2*pi*f), which is frequency dependent: 596 ms at 4 Hz, 80 ms at 30 Hz
(width=5, fs=250), and it shifts every HRSP/HEPC latency. `morlet_cwt` has no delay.
"""
from __future__ import annotations
import numpy as np

__all__ = ['morlet_kernels', 'morlet_cwt', 'ppc_map', 'hep_average',
           'epoch_beats', 'reconstruct_ics', 'detect_rpeaks',
           'cfa_topo_template', 'cfa_topo_r2']

EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# ICA reconstruction
# ---------------------------------------------------------------------------
def reconstruct_ics(eeg, icaweights, icasphere=None, icachansind=None,
                    icawinv=None):
    """IC activations as EEGLAB computes them, sign-corrected.

        ic_act = icaweights * icasphere * data(icachansind, :)

    `icasphere` is the whitening matrix. In this study's preprocessing it was the
    identity in every session checked, so omitting it is harmless here, but not in
    general. Passing None assumes identity and is only safe when you have checked.

    `icawinv` fixes the sign convention. An IC's sign is arbitrary (ICA can return a
    component and its negation with equal likelihood), so without a convention a
    subject's two sessions can enter the pooled L1 slope with opposite signs and partly
    cancel, and the HEP map of a subject can be inverted. The convention: negate any IC
    whose peak-absolute scalp weight in `icawinv` is negative. A magnitude tie takes the
    first maximum (np.argmax).

    HRSP and HEPC are unaffected: HRSP is |CWT|^2 and HEPC is phase clustering on
    unit-length phasors, both exactly sign-invariant. Passing ``icawinv=None`` skips the
    correction, which is acceptable only for such sign-invariant measures.

    eeg : [nChan x nSamples]  ->  returns [nIC x nSamples]
    """
    X = eeg if icachansind is None or len(icachansind) == 0 else eeg[icachansind]
    W = icaweights if icasphere is None or icasphere.size == 0 else icaweights @ icasphere
    act = W @ X
    if icawinv is None:
        return act
    winv = np.asarray(icawinv, float)
    if winv.ndim != 2 or winv.shape[1] != act.shape[0]:
        raise ValueError(
            f'icawinv is {winv.shape} but there are {act.shape[0]} IC(s); it must '
            f'be [nChan x nIC] with one COLUMN per component. A transposed matrix '
            f'here would sign-correct against the wrong axis, silently.')
    peak = np.argmax(np.abs(winv), axis=0)                 # first maximum on ties
    flip = winv[peak, np.arange(winv.shape[1])] < 0
    if np.any(flip):
        act = act.copy()
        act[flip] = -act[flip]
    return act


# ---------------------------------------------------------------------------
# Morlet wavelets
# ---------------------------------------------------------------------------
def morlet_kernels(freqs, fs, width=5.0):
    """Complex Morlet kernels, one per frequency.

    Built as
        st = width / (2*pi*f)
        tk = -3*st : 1/fs : 3*st
        k  = (1/(st*sqrt(2*pi))) * exp(-tk^2/(2*st^2)) * exp(1i*2*pi*f*tk)
        k  = k / sum(abs(k))                        # amplitude normalisation

    Returns (kernels, zero_idx): list of complex arrays, and the index within
    each kernel corresponding to t = 0 (its centre).
    """
    kernels, zero_idx = [], []
    for f in np.atleast_1d(freqs).astype(float):
        st = width / (2.0 * np.pi * f)
        # the range -3*st : 1/fs : 3*st, inclusive, stopping at <= 3*st
        n = int(np.floor((6.0 * st) * fs)) + 1
        tk = -3.0 * st + np.arange(n) / fs
        k = (1.0 / (st * np.sqrt(2.0 * np.pi))) * np.exp(-tk ** 2 / (2.0 * st ** 2)) \
            * np.exp(1j * 2.0 * np.pi * f * tk)
        k = k / np.sum(np.abs(k))
        kernels.append(k)
        zero_idx.append(int(np.argmin(np.abs(tk))))   # t = 0 within the kernel
    return kernels, zero_idx


# ===========================================================================
# FFT backend
# ===========================================================================
# scipy.fft with next_fast_len is about 2-2.7x faster end to end through morlet_cwt than
# numpy.fft with a power-of-two length, on shapes up to 60 signals x 45 min x 5
# frequencies (the shape of the Lee selector's pooled CWT, the dominant cost of a `lee`
# run). The gain comes from scipy's pocketfft and from `next_fast_len` choosing a
# 5-smooth length over a power of 2 (at 45 min, 677 376 points against 1 048 576), not
# from threading; `workers=-1` is kept because it costs nothing.
#
# The two backends are not bit-identical (relative differences ~1e-15, from a different
# summation order inside the FFT). That moves no statistic, and the per-signal power
# ranking that the Lee selector takes an argmax over is the same under both. An
# ImportError of scipy.fft falls back to numpy.
HEART_FFT_BACKEND = 'scipy'          # 'scipy' (fast) | 'numpy' (power-of-two length)


def _fft_backend(min_len):
    """(fft, ifft, nfft) for a transform that must hold at least `min_len` points."""
    if HEART_FFT_BACKEND == 'scipy':
        try:
            import scipy.fft as _sf

            def _f(a, n=None, axis=-1):
                return _sf.fft(a, n=n, axis=axis, workers=-1)

            def _i(a, n=None, axis=-1):
                return _sf.ifft(a, n=n, axis=axis, workers=-1)

            return _f, _i, int(_sf.next_fast_len(int(min_len)))
        except ImportError:
            pass
    return (lambda a, n=None, axis=-1: np.fft.fft(a, n, axis=axis),
            lambda a, n=None, axis=-1: np.fft.ifft(a, n, axis=axis),
            int(2 ** np.ceil(np.log2(min_len))))


def morlet_cwt(x, freqs, fs, width=5.0):
    """Continuous wavelet transform by FFT convolution.

    x : [nSamples] or [nSignals x nSamples]
    -> complex coefficients [nFreq x nSamples] or [nSignals x nFreq x nSamples]

    Each kernel's centre is wrapped to index 0 (a circular shift), so the convolution has
    no group delay.
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    X = x[None, :] if single else x
    nSig, nPts = X.shape

    kernels, zidx = morlet_kernels(freqs, fs, width)
    maxL = max(len(k) for k in kernels)
    _fft, _ifft, nfft = _fft_backend(nPts + maxL)

    Xf = _fft(X, nfft, axis=1)
    out = np.empty((nSig, len(kernels), nPts), dtype=complex)
    for i, (k, zi) in enumerate(zip(kernels, zidx)):
        wp = np.zeros(nfft, dtype=complex)
        wp[:len(k)] = k
        wp = np.roll(wp, -zi)                     # kernel centre -> index 0
        K = _fft(wp, nfft)
        out[:, i, :] = _ifft(Xf * K[None, :], axis=1)[:, :nPts]
    return out[0] if single else out


# ---------------------------------------------------------------------------
# The three measures
# ---------------------------------------------------------------------------
def ppc_map(z):
    """Unbiased pairwise phase consistency across beats (Vinck 2010).

        PPC = (|sum(z)|^2 - N) / (N*(N-1))
    z : complex unit phasors [nBeats x nFreq x nTime]; non-finite dropped per
        cell. Cells with N < 2 valid beats return NaN. Range [-1/(N-1), 1].

    z must be unit-length phasors (divide out amplitude first). That is what makes HEPC
    amplitude-invariant, and also what makes it maximally sensitive to anything perfectly
    phase-locked to the R-peak, i.e. the cardiac field. Amplitude-invariant is not
    artifact-free.
    """
    z = np.asarray(z)
    valid = np.isfinite(z)
    N = valid.sum(axis=0).astype(float)
    z = np.where(valid, z, 0)
    R2 = np.abs(z.sum(axis=0)) ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        ppc = (R2 - N) / (N * (N - 1))
    ppc[N < 2] = np.nan
    return ppc


def hep_average(epochs):
    """HEP: time-domain mean across beats. epochs [nBeats x nTime] -> [nTime]."""
    return np.nanmean(np.asarray(epochs, dtype=float), axis=0)


def epoch_beats(x, peak_samples, win_ms, fs):
    """Cut a fixed window around each R-peak.

    x            : [nSamples] or [nSignals x nSamples]
    peak_samples : R-peak sample indices
    win_ms       : (t0, t1) e.g. (-300, 700), the pipeline's hep_win
    Returns (epochs, times_ms). Beats whose window falls outside the recording
    are dropped, not zero-padded: padding would bias the average.
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    X = x[None, :] if single else x
    nPts = X.shape[1]

    i0 = int(round(win_ms[0] * fs / 1000.0))
    i1 = int(round(win_ms[1] * fs / 1000.0))
    times = np.arange(i0, i1 + 1) / fs * 1000.0

    keep = [p for p in np.asarray(peak_samples, dtype=int)
            if p + i0 >= 0 and p + i1 < nPts]
    if not keep:
        return np.empty((0, X.shape[0], len(times))), times
    ep = np.stack([X[:, p + i0: p + i1 + 1] for p in keep], axis=0)
    return (ep[:, 0, :] if single else ep), times


def clear_of_joins(peaks, i0, i1, joins, guard):
    """True for each peak whose samples peak+i0 .. peak+i1, widened by `guard` samples on
    both sides, contain no join. `joins` are the first samples after each join (where
    removed data were cut out), in the same coordinates as `peaks`."""
    pk = np.asarray(peaks, dtype=np.int64)
    j = np.sort(np.asarray(joins, dtype=np.int64).ravel())
    if pk.size == 0 or j.size == 0:
        return np.ones(pk.size, bool)
    lo = pk + int(i0) - int(guard)
    hi = pk + int(i1) + int(guard)
    # a join at sample s separates s-1 from s: it is inside when lo < s <= hi
    return np.searchsorted(j, lo, side='right') == np.searchsorted(j, hi, side='right')


def intervals_clear_of_joins(peaks, joins):
    """True for each interval between consecutive peaks that no join falls inside."""
    pk = np.sort(np.asarray(peaks, dtype=np.int64))
    j = np.sort(np.asarray(joins, dtype=np.int64).ravel())
    if pk.size < 2 or j.size == 0:
        return np.ones(max(pk.size - 1, 0), bool)
    return np.searchsorted(j, pk[:-1], side='right') == np.searchsorted(j, pk[1:], side='right')


# ---------------------------------------------------------------------------
# Cardiac field topography (diagnostic)
# ---------------------------------------------------------------------------
def cfa_topo_template(sig_rows, topos, rp):
    """Empirical cardiac-field scalp map: the beat-averaged topography at the R-peak.

    At t = 0 the QRS dominates every other source by an order of magnitude, so
    averaging the scalp map over beats at that instant measures the cardiac
    field of this recording and this montage, with no forward model.

    sig_rows : [nRows x nPts] signal rows (independent-component activations).
    topos    : [nChan x nRows] scalp projection of each row.
    rp       : R-peak sample indices (0-based). Out-of-range peaks are dropped
               here, not by the caller.

    Returns [nChan] mean-centered, or None if no peak survived the range check.
    Centered on return because cfa_topo_r2 does not re-center.

    Pass broadband rows: narrowbanding the QRS smears it across ~200 ms and
    changes what the map describes.
    """
    sig_rows = np.asarray(sig_rows, float)
    rp = np.asarray(rp, int).ravel()
    rp = rp[(rp >= 0) & (rp <= sig_rows.shape[1] - 1)]
    if rp.size == 0:
        return None
    cfa_src = sig_rows[:, rp].mean(axis=1)             # [nRows]
    cfa_topo = np.asarray(topos, float) @ cfa_src      # [nChan] scalp CFA map
    return cfa_topo - cfa_topo.mean()


def cfa_topo_r2(topo_cand, cfa_topo):
    """Squared spatial correlation of a topography with the cardiac-field map.

    A spatial diagnostic, never a gate. It is independent of the ECG-amplitude
    regressors (which adjust for the size of the field) and of the surrogate
    control (which tests heartbeat-locking; the cardiac field is the most
    R-peak-locked signal in the data). A high r2 does not prove artifact and a
    low r2 does not clear it: the field can project onto a component whose
    overall map is dissimilar.

    Both inputs must already be mean-centered; this does not center (see
    cfa_topo_template).

    Returns r2 in [0, 1], or nan if either input is empty or non-finite. nan
    means "not measured" and must never be read as 0.
    """
    if topo_cand is None or cfa_topo is None:
        return float('nan')
    a = np.asarray(topo_cand, float).ravel()
    b = np.asarray(cfa_topo, float).ravel()
    if (a.size == 0 or b.size == 0 or not np.all(np.isfinite(a))
            or not np.all(np.isfinite(b))):
        return float('nan')
    return float((a @ b / ((np.linalg.norm(a) + EPS)
                           * (np.linalg.norm(b) + EPS))) ** 2)


# ---------------------------------------------------------------------------
# R-peak detection with neurokit2 (the pipelines default to get_rr)
# ---------------------------------------------------------------------------
def ecg_polarity(ecg, fs, band=(3.0, 35.0)):
    """+1 if the R-wave points up in this recording, -1 if the trace is inverted.

    neurokit2 assumes a positive-going R and, on an inverted trace, locks onto the
    inverted S instead: on this dataset the peaks land 28-32 ms late with a 99.7-100%
    beat match rate. Same beats, wrong fiducial, so every check that counts beats passes
    while every HEP latency is shifted. BrainBeats' get_RR handles polarity itself; this
    function is that step for neurokit2.

    Decided by outcome, not by waveform shape. Peaks are detected both ways and the
    winner is the orientation whose peaks sit on the larger excursion of the QRS-band
    signal: the R-wave is taller than the S-wave, so the correct fiducial scores higher.
    This costs a second detection pass. Waveform heuristics are not reliable here: a
    near-symmetric trace can give a skewness near zero and tied upper percentiles for +x
    and -x while the R-wave still points up.

    Returns +1 (upright) or -1 (inverted).
    """
    from scipy.signal import butter, filtfilt
    x = np.asarray(ecg, dtype=float).ravel()
    x = np.where(np.isfinite(x), x, 0.0)
    if x.size < int(fs * 5):
        return 1
    nyq = fs / 2.0
    try:
        b, a = butter(3, [band[0] / nyq, min(band[1] / nyq, 0.99)], btype='band')
        xf = filtfilt(b, a, x)
    except Exception:
        xf = x - np.mean(x)
    sd = np.std(xf)
    if sd <= 0:
        return 1
    xf = xf / sd

    score = {}
    for sign in (1, -1):
        try:
            pk = _detect_rpeaks_raw(sign * x, fs)
        except Exception:
            pk = np.array([], dtype=int)
        if pk.size < 10:
            score[sign] = -np.inf
            continue
        pk = pk[(pk >= 0) & (pk < xf.size)]
        # median height in the orientation that was searched
        score[sign] = float(np.median(sign * xf[pk])) if pk.size else -np.inf
    if not np.isfinite(score[1]) and not np.isfinite(score[-1]):
        return 1
    return 1 if score[1] >= score[-1] else -1


def _detect_rpeaks_raw(ecg, fs, method='neurokit'):
    """neurokit2 detection with no polarity handling. Internal."""
    import neurokit2 as nk
    ecg = np.asarray(ecg, dtype=float).ravel()
    clean = nk.ecg_clean(ecg, sampling_rate=fs, method=method)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method=method)
    return np.asarray(info['ECG_R_Peaks'], dtype=int)


def detect_rpeaks(ecg, fs, method='neurokit', fix_polarity=True):
    """R-peak detection via neurokit2, with the inverted-ECG case handled.

    Read ecg_polarity before setting fix_polarity=False: on an inverted trace neurokit2
    returns the right number of beats at the wrong fiducial, 28-32 ms late, which no
    beat-count check can catch.

    HEP is an evoked average locked to these samples, so a systematic offset shifts every
    latency. The pipelines' default detector is get_rr (BrainBeats); check this one
    against it (median sample offset ~0 and agreeing beat counts) before trusting a HEP
    result. The filter and threshold parameters are neurokit2's defaults.

    Returns peak sample indices into the original `ecg` array (flipping is internal and
    does not shift the time base).
    """
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError('detect_rpeaks needs neurokit2:  pip install neurokit2')
    ecg = np.asarray(ecg, dtype=float).ravel()
    sign = ecg_polarity(ecg, fs) if fix_polarity else 1
    return _detect_rpeaks_raw(sign * ecg, fs, method=method)
