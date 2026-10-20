# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
heart_depth_pipeline: the HEART depth pipeline, from the extracted .npz to L1 slopes.

    segment files (data/segments/<subject>/ses<N>.npz, config.DATA_DIR)
      -> R-peaks           (get_rr by default; see PEAK_SOURCE)
      -> IC activations    (heart_core.reconstruct_ics)
      -> beat epochs
      -> HEP / HRSP / HEPC (heart_core.hep_average, morlet_cwt, ppc_map)
      -> one or more ICs per session (lee_selection.select_ics_lee, or a
                             recorded selection replayed from p['IC_PRESELECT'])
      -> depth-segment units
      -> L1 depth slopes   (l1_slopes.cell_ols)
      -> L2                (the caller: heart_depth_run, l2_heart)

Every numerical step is in one of the helper modules; this file is the
orchestration, plus the construction of the depth units.

HEPC, as the code names it, is the paper's HRPC (heartbeat-related phase
consistency).

The helpers it relies on
  * R-peaks: get_rr, a Python implementation of BrainBeats' get_RR detector
    (PEAK_SOURCE='brainbeats', the default). 'neurokit2' uses neurokit2 plus a
    per-session polarity fix.
  * RR cleaning: rr_cleaning.clean_rr, run per meditation block, not per
    segment. Step 1's min_rr/max_rr come from the median and MAD of the whole RR
    series, so a 60-90 s window (~60 beats instead of the block's ~2000) would
    pick different thresholds. Stage B below pools the beats into their block in
    absolute samples before cleaning.
  * ECG quality: ecg_quality.clean_ratio_segments and ibi_short_prop, reported
    per session for the subject gate the caller applies.
  * Baseline: glm_baseline (Alday 2019), DO_BASELINE_CORRECT. A regressor
    column, never a subtraction. HEPC takes none (PPC is over unit-length
    phasors), so HEP/HRSP spend one df that HEPC does not. A time x frequency
    map is fit one frequency at a time, each with only that frequency's baseline
    column; all nFreq columns at once is a different model costing nFreq df,
    and at this cohort's unit counts it refuses the fit (17 units against
    2 + 27 columns gives an all-NaN HRSP map).

IC_SELECT_METHOD 'lee' is the only selection method implemented here.
"""
from __future__ import annotations
import os
import glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from heart_core import (reconstruct_ics, morlet_cwt, ppc_map, hep_average, ecg_polarity,
                        _detect_rpeaks_raw, cfa_topo_template, cfa_topo_r2)
from lee_selection import select_ics_lee
from l1_slopes import cell_ols
from rr_cleaning import clean_rr, qrs_bandpass
from get_rr import get_rr, resolve_polarity, GetRRError
# Both feed the per-session halves of the pre-L1 subject QA gate, which the
# caller applies.
from ecg_quality import clean_ratio_segments, ibi_short_prop
from glm_baseline import (glm_baseline_regressors, baseline_mask,
                             dead_columns)

EPS = np.finfo(float).eps

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_PARAMS = dict(
    # --- epoching / measures ---
    HEP_WIN=[-300, 700],           # ms around the R-peak
    HRSP_FREQS=list(range(4, 31)), # 4:1:30 Hz
    HRSP_WAVE_WIDTH=5,

    # --- Lee IC selection ---
    HRSP_SEL_BANDS=[4, 20],
    HRSP_SEL_WIN=[300, 550],
    HRSP_SEL_CRITERION='hrsp',     # 'hrsp' | 'ppc'
    IC_SEL_NORM='sham_raw',        # 'raw' | 'ztrial' | 'sham' | 'sham_raw'
    IC_SEL_THRESH=1.0,             # 1 = single best IC per session
    # 50, not lee_selection's own 200 default: the number of sham draws when a
    # sham mode is the selector (folder token `shamrawsel50`). The draw count
    # sets the sham SD every score is divided by, so it is part of the
    # statistic, not just sampling effort. It is also the dominant cost of the
    # pipeline (15.0 s per segment measured at 200 draws).
    IC_SEL_SHAM_N=50,
    BASELINE_WIN=[-150, -50],      # ms; the ztrial baseline for selection

    # --- IC-selection method ---
    # 'lee': functional selection, the component whose heartbeat-locked power is
    # strongest inside HRSP_SEL_BANDS x HRSP_SEL_WIN. The only method implemented.
    IC_SELECT_METHOD='lee',
    # {(subject_id, session): [0-based IC indices]}: a recorded selection to
    # replay. Read only when PIN_SELECTION is set (surrogate control, adjusted
    # depth means, ongoing power); a normal run selects from the data.
    IC_PRESELECT=None,

    # --- Analysis window, cut out of the 90 s extracted segment ---
    # The npz segments are not the analysis window. extract_npz writes
    # [report - EXTRACT_PRE_END - EXTRACT_EPOCH_DUR, report - EXTRACT_PRE_END],
    # i.e. 90 s ending 2 s before the report; the analysis uses
    # [report - PRE_END - EPOCH_DUR, report - PRE_END], cut from inside it.
    EPOCH_DUR=60,
    PRE_END=3,                     # s before the report
    EXTRACT_EPOCH_DUR=90,          # what the .npz on disk actually holds
    EXTRACT_PRE_END=2,             # as written by extract_npz.py
    # What the Morlet transform runs on, selection and measures alike.
    # 'window': the analysis window only, so the beats nearest its two ends sit
    # within the wavelet's reach of a zero-padded edge (3 SD, 0.6 s at 4 Hz).
    # 'segment': the whole stored segment, then the window is cut from the
    # transform, so every beat in the window sees real data on both sides
    # (at PRE_END=10, 22 s before the window and 8 s after it). Folder token
    # 'cwtseg'.
    WAVELET_ON='window',

    # --- The statistical window (trim) ---
    # The measures are computed over HEP_WIN and the model is fit over TLIMS:
    # the full [-300 700] epoch is built, the Alday baseline is taken from it,
    # and the DV is trimmed to [-100 600] before cell_ols. The trimmed map is
    # the multiple-comparison space the L2 cluster test corrects over, so the
    # trim changes every cluster-forming threshold and permutation p. It also
    # changes the L1 betas under L1_METHOD='WLS': the Huber weight of a unit is
    # 1.345*median / ||resid||, and ||resid|| is taken across the columns of
    # the fit, so which columns are present changes the weights.
    # The baseline window [-150 -50] straddles the trim on purpose: the
    # baseline regressor is computed on the full axis before the trim, so the
    # baseline cells outside [-100 600] inform the model but are never tested.
    # Trimming first would shorten the baseline to [-100 -50] and put its cells
    # inside the tested map.
    TLIMS=[-100, 600],

    # --- cohort gates (kept in step with the FMT arm by hand) ---
    MIN_SESSION_MIN=45,
    MIN_DEPTHS=2,
    # A beat-level gate: a beat whose preceding inter-beat interval is shorter
    # than this is dropped. The first beat of the train has a NaN interval and
    # is kept.
    MIN_IBI_MS=600,
    minSeg=5,                      # per SUBJECT, pooled over sessions
    MIN_SEG_SESSION=0,             # per-session floor; 0 = off
    MIN_CONF=4,                    # confidence rating floor (applied at extraction)
    DEPTH_LEVELS=[1, 2, 3, 4, 5],
    INCLUDE_SAPO=True,

    # --- beats ---
    # A segment needs at least this many epochable beats to be a unit, for all
    # three measures. Short IBIs are already removed beat by beat by
    # MIN_IBI_MS, so what is checked here is how many heartbeats are left in the
    # segment. At 60 s and 46-95 bpm a segment carries ~46-95 beats, so the
    # floor bites only on truncated or heavily cleaned segments. 20 is also the
    # floor the unbiased PPC estimator (HEPC) needs; a mean voltage or a mean
    # power does not need one, so lowering it removes HEPC's guard.
    MIN_BEATS_UNIT=20,

    # --- Per-segment covariates (opt-in; see segment_covariates) ---
    # Off by default: the default models (`session`, `unadj`) do not read them,
    # so computing them for every run would be pure cost. The covariate models
    # need them and refuse to run without them.
    KEEP_SEGCOV=False,
    # --- State (tonic) band power, the theta/alpha/beta covariates: mean power
    # inside the analysed window, one value per segment per band, logged. They
    # ask whether the heartbeat-locked difference is the same variance as the
    # ongoing state difference, which is the tonic question, so read a loss of
    # the effect under them as decomposition, not as a failed control. The .npz
    # holds segments only, so the filter runs per segment and the edge trim
    # below covers the transient.
    STATE_BANDS=((5.0, 8.0), (8.0, 13.0), (13.0, 30.0)),   # theta, alpha, beta
    STATE_BAND_TRIM_S=2.0,
    # 'component' (default): the tonic power of the analysed signal itself,
    # averaged over the whole window. That is the tonic-vs-phasic
    # decomposition: does the heartbeat-locked difference exceed the
    # segment-mean difference of the same signal? It shares variance with HRSP
    # by construction, so read a loss of the effect under it as 'the
    # difference is tonic', never as a failed artefact control.
    # 'channels': mean power over all analysed channels, which avoids that
    # shared variance but only rules out a global state-power account.
    STATE_BAND_SOURCE='component',
    # [-50 50] is the R-wave and [200 400] the T-wave. They are separate
    # models, never fitted together (one dipole drives both, so together they
    # are collinear). R-wave amplitude tracks the whole cardiac dipole and is a
    # global cardiac-field magnitude proxy; the T-wave is the deflection inside
    # the part of the HEP the literature interprets. Read a surviving `ecgAmp`
    # as "survives control for cardiac-field magnitude", never as "the T-wave
    # was removed".
    ECG_AMP_WIN=[-50, 50],         # ms, R-wave
    ECG_AMP_WIN_T=[200, 400],      # ms, T-wave
    # Respiration band for the envelope/phase covariates (a single-pass FIR
    # band-pass, then hilbert; see _resp_env_phase).
    RESP_BAND=[0.1, 1.0],
    # Divide each selected IC by its SD over the analysed segments, once per
    # session. See PASS 2 for why this is not a harmless global constant.
    IC_UNIT_VAR_NORM=True,
    # R-peak detector:
    #   'brainbeats' get_rr, the Python implementation of BrainBeats' get_RR
    #                (Pan-Tompkins).
    #   'neurokit2'  neurokit2 plus the per-session polarity fix. Its timing is
    #                exact but its beat set is not: on a slow heart with a tall
    #                T-wave its gradient criterion accepts the T-wave as an
    #                R-peak (on one participant, 1.5 and 1.8 times the true beat
    #                count in the two sessions).
    PEAK_SOURCE='brainbeats',      # 'brainbeats' | 'neurokit2'
    DO_CLEAN_RR=True,              # BrainBeats RR-interval rejection
    # Subject-level, not segment-level: the subject is gated on the mean ECG
    # clean ratio over its observations, in keep form, so an unassessable (NaN)
    # subject fails. The caller applies the gate.
    ECG_MAX=0.2,
    # There is no subject-level short-IBI gate: MIN_IBI_MS removes short
    # intervals beat by beat, and MIN_BEATS_UNIT then checks how many beats are
    # left in each segment. Gating the subject on a statistic of beats already
    # discarded would count the same defect twice. `ibi_short_prop` is still
    # computed and written to cohort_sessions.csv / cohort_subjects.csv as a
    # diagnostic (it exposes T-wave over-detection), but nothing gates on it.
    # The caller gates on the ECG half only (ecg_quality.ecg_ratio_passes_gate).

    # --- baseline (Alday GLM) ---
    DO_BASELINE_CORRECT=True,
    BASELINE_METHOD='glm',         # the only method: 'subtract'/'db'/'percent'
                                   # rewrite the data before the fit instead of
                                   # estimating the baseline.

    # --- L1 ---
    L1_METHOD='WLS',               # 'OLS' | 'WLS' (Huber)
    # False for HEART, unlike FMT. HEART carries `session` as a design
    # column (see subject_slopes); z-scoring within session on top of that would
    # remove the same effect twice and rescale sessions against each other.
    # Set True only if you deliberately want the FMT convention.
    STANDARDIZE_WITHIN_SESSION=False,
)


# ═══════════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════════
def list_sessions(data_dir):
    """[(subject_id, session, path), ...] for every <subject>/ses<N>.npz in data_dir."""
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, '*', 'ses*.npz'))):
        sid = os.path.basename(os.path.dirname(f))
        ses = int(os.path.basename(f).replace('ses', '').replace('.npz', ''))
        out.append((sid, ses, f))
    return out



# ═══════════════════════════════════════════════════════════════════════════
# THE ANALYSIS WINDOW: ONE IMPLEMENTATION, USED BY EVERYONE
# ═══════════════════════════════════════════════════════════════════════════
def segment_band_power(eeg, fs, bands, trim_s=2.0):
    """Log mean band power of one segment, over its rows: the tonic state.

    Band-pass every row of `eeg` (all channels, or the analysed component as a
    single row; see STATE_BAND_SOURCE), square, average over rows, drop
    `trim_s` at each edge, average over time, take log10. The .npz stores
    segments, so the filter runs per segment; `trim_s` (2 s,
    STATE_BAND_TRIM_S) covers the filter transient at both edges. The filter is
    a zero-phase 4th-order Butterworth.

    Parameters
    ----------
    eeg : [nChan x nPts] array. One segment, channels as analysed.
    fs : float. Sampling rate (Hz).
    bands : sequence of (lo, hi) Hz. Row order is the output order.
    trim_s : float. Seconds dropped at each edge before averaging.

    Returns
    -------
    [nBands] array of log10 mean power, NaN for a band the segment is too short
    for (nothing left after trimming).
    """
    from scipy.signal import butter, filtfilt
    x = np.asarray(eeg, float)
    out = np.full(len(bands), np.nan)
    trim = int(round(trim_s * fs))
    if x.ndim != 2 or x.shape[1] <= 2 * trim + 1:
        return out
    for b, (lo, hi) in enumerate(bands):
        sos_b, sos_a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='bandpass')
        y = filtfilt(sos_b, sos_a, x, axis=1)
        pw = np.nanmean(y ** 2, axis=0)[trim:-trim]
        m = np.nanmean(pw)
        out[b] = np.log10(m) if np.isfinite(m) and m > 0 else np.nan
    return out



def window_indices(seg_len, fs, epoch_dur, pre_end,
                   extract_epoch_dur=None, extract_pre_end=None):
    """Sample bounds of the analysis window inside each stored segment.

    Returns ``(lo, hi, usable)``. ``lo`` and ``hi`` are scalars (the window sits
    at a fixed offset from every segment's start, see below) and ``usable`` is a
    [nSeg] bool saying whether that segment contains sample ``hi-1``. The window
    for segment k is ``seg[..., lo:hi]``, and only for usable k.

    The stored segment is anchored at its start, not its end. extract_npz.py
    writes
        start = round((report - EXTRACT_PRE_END - EXTRACT_EPOCH_DUR) * fs)
        end   = min(round((report - EXTRACT_PRE_END) * fs),
                    next_epoch_start - 1,
                    n_total - 1)
    and its overlap resolution moves only the end. So `report` is recoverable
    from the start, and not from the end whenever two depth reports fall less
    than EXTRACT_EPOCH_DUR apart. Anchoring on the end would cut the window from
    the wrong place in an end-truncated segment while reporting it as full
    length, because a length test never checks anchoring.

    A segment that does not reach `hi` is refused, not clamped. Under the start
    anchor a short segment is short at its tail: it is missing the seconds
    nearest the report, which are the part the depth model is about. Those
    samples are not in the .npz, so the caller must drop the segment with a
    named reason. With extract_npz.RESOLVE_OVERLAPS = False no stored segment
    is end-truncated (each ends EXTRACT_PRE_END before its report), so the
    cohort does not exercise this path.

    Bounds are inclusive at both ends, so the window is ``epoch_dur*fs + 1``
    samples (15001 at 60 s / 250 Hz), not ``epoch_dur*fs``.
    """
    if extract_epoch_dur is None:
        extract_epoch_dur = DEFAULT_PARAMS['EXTRACT_EPOCH_DUR']
    if extract_pre_end is None:
        extract_pre_end = DEFAULT_PARAMS['EXTRACT_PRE_END']
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'window_indices: fs must be finite and > 0, got {fs}')
    if float(epoch_dur) > float(extract_epoch_dur):
        raise ValueError(
            f'EPOCH_DUR={epoch_dur} > {float(extract_epoch_dur):g} s stored per '
            f'segment by extract_npz.py; that data is not in the .npz.')
    trim = int(round((float(pre_end) - float(extract_pre_end)) * fs))
    if trim < 0:
        raise ValueError(
            f'PRE_END={pre_end} < {float(extract_pre_end):g} s used at '
            f'extraction: the samples between them were never written to the '
            f'.npz, so this window cannot be reconstructed.')
    epoch_samples = int(round(float(epoch_dur) * fs))
    # index of `report - PRE_END` inside the segment, counted from the START
    hi_incl = int(round(float(extract_epoch_dur) * fs)) - trim
    lo = hi_incl - epoch_samples
    if lo < 0:
        # Not a per-segment problem: this parameter pair cannot be honoured by
        # any segment, because the requested window starts before the stored
        # one. Both numbers in the message are measured from the report.
        raise ValueError(
            f'EPOCH_DUR={epoch_dur} with PRE_END={pre_end} needs '
            f'{(float(epoch_dur) + float(pre_end)):g} s '
            f'of data before the report, but extract_npz.py stored only '
            f'{float(extract_epoch_dur) + float(extract_pre_end):g} s. '
            f'The ceiling at PRE_END={pre_end} is EPOCH_DUR='
            f'{(float(extract_epoch_dur) + float(extract_pre_end) - float(pre_end)):g}. '
            f'Re-extract, do not pad.')
    hi = hi_incl + 1                          # exclusive slice bound
    sl = np.asarray(seg_len).astype(np.int64)
    return lo, hi, (sl >= hi)


def _resp_env_phase(x, fs, band=(0.1, 1.0)):
    """Respiration envelope and phase for one stored segment.

    Band-pass 0.1-1 Hz with EEGLAB's ``pop_eegfiltnew`` design (Hamming-windowed
    FIR with its transition-band and filter-order rules), then hilbert ->
    ``abs`` and ``angle``.

    One pass, not filtfilt. `scipy.signal.filtfilt` runs the kernel twice, which
    squares the magnitude response and narrows the effective band (a nominal
    4.26-8.74 Hz theta band passes only 5.22-7.78 Hz that way). pop_eegfiltnew
    is a single pass of a linear-phase FIR, made zero-phase by discarding the
    group delay, and that is what this reproduces.

    The .npz stores only segments, so this filters the ~90 s stored segment
    rather than the continuous recording. At a 0.1 Hz high-pass the kernel is
    3.3*fs/0.1 samples (about 33 s at 250 Hz), so a bare convolution would leave
    the last ~16 s of every segment edge-contaminated, and the analysed window
    ends only ~1 s before the segment does. The segment is therefore
    reflect-padded by the kernel half-length before convolution, which is
    standard practice for a quasi-periodic signal and what MNE does by default.
    It is a mitigation, not an equivalence: the slowest components of the
    respiratory drift within a segment are not recoverable from that segment
    alone. Respiration itself (~0.2-0.4 Hz) sits well inside the reliable range.

    Returns (env, phase), both the length of `x`, or (None, None) when the
    segment is constant or too short for the kernel.
    """
    from scipy.signal import firwin, fftconvolve, hilbert
    x = np.asarray(x, float)
    n = x.size
    lo, hi = float(band[0]), float(band[1])
    nyq = fs / 2.0
    if n < 16 or not np.isfinite(x).all() or np.std(x) == 0:
        return None, None
    # EEGLAB pop_eegfiltnew's transition bandwidth rule, then its
    # `filtorder = 3.3 / (df / srate)` and the round-up to an even order (an
    # odd number of taps), which is what makes the delay an exact integer.
    df_hp = min(max(lo * 0.25, 2.0), lo) if lo > 0 else 0.0
    df_lp = min(max(hi * 0.25, 2.0), nyq - hi) if hi < nyq else 0.0
    df = min([v for v in (df_hp, df_lp) if v > 0], default=2.0)
    ntaps = int(np.ceil(3.3 / (df / fs)))
    ntaps += (ntaps % 2 == 0)                       # odd -> integer group delay
    ntaps = min(ntaps, 2 * n - 1)                   # cannot exceed the padding
    ntaps += (ntaps % 2 == 0)
    if ntaps < 3:
        return None, None
    k = firwin(ntaps, [lo, hi], pass_zero=False, fs=fs, window='hamming')
    half = ntaps // 2
    pad = min(half, n - 1)
    xp = np.concatenate([x[pad:0:-1], x, x[-2:-2 - pad:-1]]) if pad > 0 else x
    y = fftconvolve(xp, k, mode='same')
    y = y[pad:pad + n] if pad > 0 else y
    z = hilbert(y)
    return np.abs(z), np.angle(z)


def tlims_mask(times, tlims):
    """Boolean mask of the analysed time points, inclusive at both ends.

    Raises on an empty or out-of-epoch window rather than returning an empty
    mask: an all-False mask makes every map zero-width and the L2 then reports
    "no cluster" for a run that tested nothing.
    """
    t = np.asarray(times, float)
    lo, hi = float(tlims[0]), float(tlims[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        raise ValueError(f'TLIMS={list(tlims)!r} is not an increasing finite '
                         f'window in ms.')
    m = (t >= lo) & (t <= hi)
    if not m.any():
        raise ValueError(
            f'TLIMS={list(tlims)!r} selects no sample of the epoch axis '
            f'[{t[0]:g}, {t[-1]:g}] ms. Widen HEP_WIN or fix TLIMS.')
    return m


# ═══════════════════════════════════════════════════════════════════════════
# PER-SESSION PROCESSING
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# PER-SEGMENT COVARIATES
# ═══════════════════════════════════════════════════════════════════════════
def segment_covariates(p, d, idx_keep, good, seg_peaks, seg_ecg, seg_eeg,
                       seg_len, times, fs, w0, w1, lo, hi, comp_sig, notes):
    """The `segcov` dict for one session, or None when KEEP_SEGCOV is off.

    `comp_sig` is the {segment index -> 1-D analysed-window signal} mapping used
    when STATE_BAND_SOURCE == 'component' (session_measures passes its combined
    IC signal). Pass {} to make 'channels' the only usable source; a segment
    missing from the mapping gets NaN state-band power.
    """
    _xs = comp_sig
    segcov = None
    if bool(p.get('KEEP_SEGCOV', False)):
        segcov = {}
        nS_ = len(idx_keep)
        rr_seg = np.full(nS_, np.nan)
        amp_seg = np.full(nS_, np.nan)
        ampt_seg = np.full(nS_, np.nan)
        ramp_seg = np.full(nS_, np.nan)
        rsin_seg = np.full(nS_, np.nan)
        rcos_seg = np.full(nS_, np.nan)
        a_win = np.asarray(p.get('ECG_AMP_WIN', [-50, 50]), float)
        t_win = np.asarray(p.get('ECG_AMP_WIN_T', [200, 400]), float)
        # Checked, not assumed. Outside HEP_WIN these masks select nothing,
        # every beat's amplitude is NaN, and the ecgAmp rows would vanish from
        # the survival table with no message, the model silently degrading to
        # unadjusted.
        for _nm, _w in (('ECG_AMP_WIN', a_win), ('ECG_AMP_WIN_T', t_win)):
            if _w[0] < times[0] or _w[1] > times[-1]:
                raise ValueError(
                    f'{_nm}={list(_w)} lies outside HEP_WIN '
                    f'[{times[0]:g}, {times[-1]:g}] ms, so it would select no '
                    f'sample and every beat amplitude would be NaN. Widen '
                    f'HEP_WIN or fix {_nm}.')
        a_ix = (times >= a_win[0]) & (times <= a_win[1])
        t_ix = (times >= t_win[0]) & (times <= t_win[1])
        # The respiration envelope is averaged over the analysed part of the
        # epoch (TLIMS); the phase is read at the R-peak.
        e_ix = tlims_mask(times, p['TLIMS'])
        has_resp = 'seg_resp' in d.files
        if not has_resp:
            # A NAMED refusal. An .npz without seg_resp cannot supply respAmp /
            # respPhase, and returning them as all-NaN would read downstream as
            # "measured and uninformative" rather than "never measured".
            notes.setdefault('segcov_absent', []).append(
                'respAmp/respPhase: this .npz carries no `seg_resp`')
        seg_resp_all = d['seg_resp'] if has_resp else None
        sub_all = (np.asarray(d['seg_subcond']).astype(float)
                   if 'seg_subcond' in d.files else None)
        pb_amp, pb_ampt, pb_seg_ix = [], [], []
        for j, k in enumerate(idx_keep):
            pk = seg_peaks[j]
            if pk.size == 0:
                continue
            eix = np.stack([np.arange(q + lo, q + hi + 1) for q in pk])
            # ── RR: within this segment only ────────────────────────────────
            # diff() over a train pooled across segments would measure the gap
            # between depth reports (~147 s at each segment's first beat), not
            # heart rate. The first beat of a segment has no preceding interval
            # inside the segment and is NaN, never 0.
            if pk.size > 1:
                rr_ms = np.diff(np.sort(pk)) / fs * 1000.0
                rr_seg[j] = float(np.mean(rr_ms))
            # ── ECG amplitude, RAW trace ────────────────────────────────────
            # The raw ECG, not the 3-35 Hz QRS-detection trace: that high-pass
            # strongly attenuates the low-frequency T-wave, which is what
            # ecgAmpT measures.
            ecg_w = np.asarray(seg_ecg[k, w0:w1], float)
            ep = ecg_w[eix]
            pb_amp.append(ep[:, a_ix].max(1) - ep[:, a_ix].min(1))
            pb_ampt.append(ep[:, t_ix].max(1) - ep[:, t_ix].min(1))
            pb_seg_ix.append(np.full(pk.size, j, np.int64))
            # ── respiration ─────────────────────────────────────────────────
            if has_resp:
                env, ph = _resp_env_phase(
                    np.asarray(seg_resp_all[k, :int(seg_len[k])], float), fs,
                    p.get('RESP_BAND', (0.1, 1.0)))
                if env is not None:
                    env_w, ph_w = env[w0:w1], ph[w0:w1]
                    ramp_seg[j] = float(np.nanmean(env_w[eix][:, e_ix]))
                    rsin_seg[j] = float(np.nanmean(np.sin(ph_w[pk])))
                    rcos_seg[j] = float(np.nanmean(np.cos(ph_w[pk])))
        # Z-scored within session. The raw peak-to-peak scale differs between
        # days (electrode placement, impedance) and a subject's two sessions
        # are pooled downstream, so an un-z-scored column would be partly a
        # session indicator, and FWL would then partial out session variance
        # the depth predictor also carries. Centring the pooled vector later
        # removes the grand mean, not the between-session step; only the
        # within-session z removes that.
        if pb_amp:
            bs = np.concatenate(pb_seg_ix)
            for src, dst in ((pb_amp, amp_seg), (pb_ampt, ampt_seg)):
                v = np.concatenate(src)
                ok_ = np.isfinite(v)
                if ok_.sum() > 1 and np.std(v[ok_]) > 0:
                    v = (v - v[ok_].mean()) / v[ok_].std()
                for j in range(nS_):
                    m_ = (bs == j) & np.isfinite(v)
                    if m_.any():
                        dst[j] = float(v[m_].mean())
        # STATE (TONIC) BAND POWER, one value per segment per band. Computed
        # on the analysed channels of the same window the measures use, then
        # z-scored within session like the other per-segment covariates: the
        # absolute power scale differs between days (impedance, placement) and
        # a subject's sessions are pooled downstream, so an un-z-scored column
        # would be partly a session indicator.
        bands = [tuple(float(v) for v in b) for b in p.get('STATE_BANDS',
                                                          ((5., 8.), (8., 13.), (13., 30.)))]
        src = str(p.get('STATE_BAND_SOURCE', 'component')).lower()
        if src not in ('component', 'channels'):
            raise ValueError("STATE_BAND_SOURCE must be 'component' or 'channels', "
                             f'got {src!r}.')
        state_seg = np.full((nS_, len(bands)), np.nan)
        trim_s = float(p.get('STATE_BAND_TRIM_S', 2.0))
        for j, k_ in enumerate(idx_keep):
            if src == 'component':
                if j not in _xs:
                    continue          # segment dropped before the measures
                sig_j = np.asarray(_xs[j], float)[None, :]
            else:
                sig_j = np.asarray(seg_eeg[k_, :, w0:w1], float)
            state_seg[j] = segment_band_power(sig_j, fs, bands, trim_s)
        for b, nm in enumerate(('theta', 'alpha', 'beta')[:len(bands)]):
            v = state_seg[:, b]
            ok_ = np.isfinite(v)
            if ok_.sum() > 1 and np.std(v[ok_]) > 0:
                v = (v - v[ok_].mean()) / v[ok_].std()
            segcov[nm] = v[good]

        segcov['rr'] = rr_seg[good]
        segcov['ecgAmp'] = amp_seg[good]
        segcov['ecgAmpT'] = ampt_seg[good]
        if has_resp:
            segcov['respAmp'] = ramp_seg[good]
            segcov['respPhase_sin'] = rsin_seg[good]
            segcov['respPhase_cos'] = rcos_seg[good]
        if sub_all is not None:
            # 1 = probe, 0 = emerge, -1 = neither (as written by extract_npz.py).
            # -1 is NaN here: it is "unknown", not a third level, and averaging
            # it in as -1 would make the column a nonsense contrast.
            rm = sub_all[idx_keep][good].astype(float)
            rm[rm < 0] = np.nan
            segcov['reportmethod'] = rm
        notes['segcov_keys'] = sorted(segcov)
    return segcov


def session_measures(npz_path, p, verbose=True, notes=None):
    """One session -> per-segment HEP / HRSP / HEPC on the selected IC.

    Returns None when the session fails a gate, else a dict with:
        depth   [nSeg]            depth level per retained segment
        cond    [nSeg]            condition code
        hep     [nSeg x nT]
        hrsp    [nSeg x nF x nT]
        hepc    [nSeg x nF x nT]
        ic      int               the top IC (highest-scoring)
        ic_all  list[int]         every selected IC, 0-based. 1 element under
                                  'lee' at IC_SEL_THRESH=1; several are combined
                                  as in PASS 2 (polarity-aligned to the top IC,
                                  per-IC unit-variance normalised, then
                                  averaged)
        ic_method str             which IC_SELECT_METHOD produced them
        times   [nT]              ms  (the full HEP_WIN axis; the TLIMS trim is
                                       applied by `subject_slopes`)
        freqs   [nF]              Hz
        n_beats [nSeg]            epochable beats per retained segment

    `notes` : an optional dict this function fills in. Every gate writes its
    counts there and a refusal writes `notes['skip_reason']`, so the caller can
    say why a session vanished instead of guessing; returning None with no
    reason is how a cohort silently shrinks. Keys always present after a call:
    n_seg_stored, n_seg_cond, n_seg_window_ok, n_seg_window_refused,
    n_beats_detected, n_beats_after_cleanrr, n_beats_after_min_ibi,
    n_beats_analysed, ecg_clean_ratio, ibi_short_prop.
    """
    if notes is None:
        notes = {}
    notes.setdefault('skip_reason', '')
    d = np.load(npz_path, allow_pickle=True)
    sid = str(d['subject_id']); ses = int(d['session']); fs = float(d['fs'])
    tag = f'{sid} ses{ses}'

    # ── SURROGATE_DROP: a session this draw cannot shift is excluded ────────
    # A session that admits no shift in the permitted band must not contribute
    # its real, unshifted epochs to the surrogate null (that would pull the null
    # toward the real statistic). The driver names such sessions in
    # SURROGATE_DROP and they are dropped here, so a draw is built on shifted
    # data only.
    _drop = p.get('SURROGATE_DROP')
    if _drop and ((str(sid), int(ses)) in _drop or (sid, ses) in _drop):
        notes['skip_reason'] = (
            'SURROGATE_DROP: this session admitted no shift in the permitted '
            'band for this draw, so it is EXCLUDED rather than analysed at '
            'shift 0 with its real peaks')
        return None

    seg_eeg = d['seg_eeg']                       # [nSeg x nChan x nPts]
    seg_ecg = d['seg_ecg']                       # [nSeg x nPts]
    depth = np.asarray(d['seg_depth']).astype(int)
    cond = np.asarray(d['seg_cond']).astype(int)
    seg_len = np.asarray(d['seg_len']).astype(int)

    notes['n_seg_stored'] = int(len(depth))

    # ── GATE 1: condition ───────────────────────────────────────────────────
    keep = np.ones(len(depth), bool)
    if not p['INCLUDE_SAPO']:
        keep &= (cond == 0)
    keep &= np.isin(depth, p['DEPTH_LEVELS'])
    notes['n_seg_cond'] = int(keep.sum())
    if keep.sum() == 0:
        notes['skip_reason'] = 'no segment in the analysed condition/depth set'
        if verbose: print(f'  {tag}: no segments in DEPTH_LEVELS — skipped')
        return None

    # ── GATE 1b: is the requested window actually stored for this segment? ──
    # Anchoring, not length (see window_indices): the stored segment is anchored
    # at its start and its end moves whenever two depth reports fall closer
    # together than EXTRACT_EPOCH_DUR, so a segment can be long enough and still
    # not contain the seconds this window asks for. A segment that does not
    # reach `w1` is dropped with a reason, never clamped. This runs before
    # MIN_DEPTHS, so a refused segment never counts toward a depth level.
    w0, w1, w_ok = window_indices(seg_len, fs, p['EPOCH_DUR'], p['PRE_END'],
                                  p['EXTRACT_EPOCH_DUR'], p['EXTRACT_PRE_END'])
    n_ref = int((keep & ~w_ok).sum())
    notes['n_seg_window_refused'] = n_ref
    if n_ref:
        lost = (np.asarray(seg_len)[keep & ~w_ok] / fs)
        notes['window_refused_dur_s'] = [round(float(v), 1) for v in lost]
        if verbose:
            print(f'  {tag}: {n_ref} segment(s) REFUSED — the stored segment is '
                  f'end-truncated ({lost.min():.1f}-{lost.max():.1f} s of '
                  f'{(p["EXTRACT_EPOCH_DUR"] + 1.0 / fs):.1f} s) so it does not '
                  f'contain the {float(p["EPOCH_DUR"]):g} s ending '
                  f'{float(p["PRE_END"]):g} s before the report')
    keep &= w_ok
    notes['n_seg_window_ok'] = int(keep.sum())
    _won = str(p.get('WAVELET_ON', 'window'))
    if _won not in ('window', 'segment'):
        raise ValueError(f"WAVELET_ON must be 'window' or 'segment', got {_won!r}")
    _cwt_seg = _won == 'segment'

    # ── GATE 1c: resolve overlaps at the analysis duration ──────────────────
    # The .npz may hold overlapping segments: extract_npz writes
    # `overlap_resolved=False` and leaves them whole, because trimming at its
    # own 90 s would discard seconds a shorter analysis window is entitled to.
    # Two analysis windows that overlap share samples, and counting those
    # samples in two "independent" units would inflate n, so overlaps are
    # resolved here.
    #
    # A file without the `overlap_resolved` key is assumed resolved at
    # extraction, and the assumption is announced rather than made silently,
    # because a wrong guess double-counts data.
    ov_key = 'overlap_resolved' in d.files
    already = bool(d['overlap_resolved']) if ov_key else True
    notes['overlap_resolved_at_extraction'] = already
    if not ov_key and verbose:
        print(f'  {tag}: .npz predates the overlap-provenance key — ASSUMING it '
              f'was resolved at extraction (it was, at '
              f'{float(p["EXTRACT_EPOCH_DUR"]):g} s). Re-extract to remove the '
              f'assumption.')
    n_ov = 0
    if not already and keep.sum() > 1:
        # Absolute bounds of each kept analysis window, in report order.
        span_all = np.asarray(d['seg_span'])
        kept_ix = np.where(keep)[0]
        kept_ix = kept_ix[np.argsort(span_all[kept_ix, 0])]
        starts = span_all[kept_ix, 0] + w0
        ends = span_all[kept_ix, 0] + w1                 # exclusive
        drop = []
        for a_ in range(len(kept_ix) - 1):
            if ends[a_] > starts[a_ + 1]:
                drop.append(kept_ix[a_])
                n_ov += 1
        # The earlier window of an overlapping pair is dropped, not trimmed.
        # Trimming would make w0/w1 vary per segment, and every downstream index
        # here (the selection epoch matrix, the measures pass) relies on their
        # being the same for every segment. A dropped segment is counted and
        # named (`n_window_overlaps`); a silently variable window would not be.
        if drop:
            keep[np.asarray(drop)] = False
            notes['n_seg_window_ok'] = int(keep.sum())
            if verbose:
                print(f'  {tag}: {n_ov} analysis window(s) OVERLAP at '
                      f'{float(p["EPOCH_DUR"]):g} s — the earlier of each pair is '
                      f'dropped (trimming, then dropping under '
                      f'{float(p["EPOCH_DUR"]) - 3:g} s, would give the same result '
                      f'above a 3 s overlap)')
    notes['n_window_overlaps'] = n_ov
    if keep.sum() == 0:
        notes['skip_reason'] = (
            f'every segment is end-truncated: none contains the '
            f'{float(p["EPOCH_DUR"]):g} s window ending {float(p["PRE_END"]):g} s '
            f'before the report')
        if verbose: print(f'  {tag}: no segment holds the window — skipped')
        return None

    # ── GATE 2: MIN_DEPTHS, per session (matches the FMT arm) ───────────────
    if len(set(depth[keep].tolist())) < p['MIN_DEPTHS']:
        notes['skip_reason'] = (f'MIN_DEPTHS: {len(set(depth[keep].tolist()))} '
                                f'distinct depth(s) < {int(p["MIN_DEPTHS"])}')
        if verbose:
            print(f'  {tag}: {len(set(depth[keep].tolist()))} distinct depth(s) '
                  f'< MIN_DEPTHS {p["MIN_DEPTHS"]} — session skipped')
        return None

    idx_keep = np.where(keep)[0]

    # ── THE SELECTION METHOD ───────────────────────────────────────────────
    method = str(p.get('IC_SELECT_METHOD', 'lee'))
    if method != 'lee':
        raise ValueError(f"IC_SELECT_METHOD must be 'lee', got {method!r}.")
    if str(p.get('HRSP_SEL_CRITERION', 'hrsp')) not in ('hrsp', 'ppc'):
        raise ValueError("HRSP_SEL_CRITERION must be 'hrsp' or 'ppc', got "
                         f"{p.get('HRSP_SEL_CRITERION')!r}.")

    # ── IC decomposition ───────────────────────────────────────────────────
    # icasphere is the identity in this study's preprocessing (verified over 16
    # sessions), but it is passed anyway: omitting it is not safe in general.
    icaw = np.asarray(d['icaweights'], float)
    icas = np.asarray(d['icasphere'], float) if 'icasphere' in d.files else None
    icach = np.asarray(d['icachansind'], int) if 'icachansind' in d.files else None
    # Refused, not defaulted: without icawinv the IC sign convention cannot be
    # applied, and falling back to "no sign correction" for this session only
    # would mix two sign conventions in one cohort.
    if 'icawinv' not in d.files:
        raise KeyError(
            f'{tag}: this .npz has no `icawinv`, so the IC sign '
            f'convention cannot be applied and HEP would be arbitrarily '
            f'signed for this session. Re-export with extract_npz.py,'
            f' which writes it.')
    icawinv = np.asarray(d['icawinv'], float)
    if icach is not None and icach.size and icach.max() >= seg_eeg.shape[1]:
        icach = icach - 1                        # 1-based export

    # ═══════════════════════════════════════════════════════════════════════
    # A RECORDED SELECTION TO REPLAY (PIN_SELECTION)
    # ═══════════════════════════════════════════════════════════════════════
    # Normally the component is selected from the data, below. With PIN_SELECTION
    # and an IC_PRESELECT table the recorded component is used instead. The
    # surrogate control needs this: it must hold the analysed component FIXED and
    # vary only the R-peak timing, or it stops being a null for the fit it is
    # compared against (re-selecting on SHIFTED peaks is a different question).
    # It also spares the pooled selection CWT, the dominant cost of a run.
    # OFF unless explicitly set: a normal run must keep selecting.
    #
    # A missing entry is a refusal, not an empty selection. An entry holding an
    # empty list means no component was recorded for the session (a named
    # drop); a session absent from the table is a wiring bug.
    presel = None
    if p.get('PIN_SELECTION') and isinstance(p.get('IC_PRESELECT'), dict):
        table = p['IC_PRESELECT']
        key = (sid, ses)
        if key not in table:
            raise KeyError(
                f'{tag}: no IC_PRESELECT entry for this session. An entry with '
                'an EMPTY list means "no component recorded" and is a valid '
                'result; a MISSING entry means the table was built without this '
                'session, which is a wiring bug.')
        presel = np.asarray(table[key], dtype=int).ravel()
        if presel.size == 0:
            notes['skip_reason'] = ('pinned selection: no component recorded '
                                    'for this session')
            if verbose:
                print(f'  {tag}: {notes["skip_reason"]} — session skipped')
            return None
        nIC_here = int(np.asarray(icaw).shape[0])
        if presel.min() < 0 or presel.max() >= nIC_here:
            raise IndexError(
                f'{tag}: IC_PRESELECT holds index/indices outside 0..{nIC_here - 1} '
                f'({presel.tolist()}). Indices are 0-BASED here; EEGLAB / MATLAB '
                'indices are 1-BASED and must have 1 subtracted, or one component '
                'is analysed in place of another and nothing looks wrong.')

    # A pinned run neither builds the selection CWT nor selects. Both gates
    # below test this one flag, so they cannot disagree.
    _pinned = presel is not None

    freqs = np.asarray(p['HRSP_FREQS'], float)
    win = p['HEP_WIN']
    nT = int(round((win[1] - win[0]) / 1000.0 * fs)) + 1
    times = win[0] + np.arange(nT) / fs * 1000.0

    # ── per segment: peaks -> IC epochs -> CWT ──────────────────────────────

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 1: peaks per segment, then the continuous CWT at the selection
    # frequencies only, concatenated across segments for one session-level
    # selection.
    #
    # select_ics_lee takes the continuous CWT [nIC x nFsel x nSamples] plus an
    # epoch-index matrix ei [nBeats x nT], not pre-epoched data. The sham norm
    # modes rigidly shift every epoch by up to ±500 ms and score the shifted
    # windows, so they need the samples around each beat to exist. Handing it
    # epoched data would remove the null it scores against (and epoching first
    # costs ~61x the memory).
    #
    # Selection frequencies only, and complex64: the full 4-30 Hz stack over all
    # 61 ICs for a whole session is tens of GB.
    # ═══════════════════════════════════════════════════════════════════════
    f_sel = (freqs >= p['HRSP_SEL_BANDS'][0]) & (freqs <= p['HRSP_SEL_BANDS'][1])
    lo = int(round(win[0] / 1000.0 * fs)); hi = int(round(win[1] / 1000.0 * fs))

    # ── ECG polarity: once per session ──────────────────────────────────────
    # Decided on a concatenation of the retained segments so the vote sees a few
    # hundred beats rather than one window's worth. The resolver matches the
    # detector: under 'brainbeats' it is get_rr's own vote (resolve_polarity);
    # under 'neurokit2' it is heart_core.ecg_polarity, which votes with two
    # neurokit2 passes. The two resolvers agree on every analysed recording of
    # the depth cohort.
    pol = 1
    _cat = np.concatenate([np.asarray(seg_ecg[k, :int(seg_len[k])], float)
                           for k in idx_keep[:6]])
    if p['PEAK_SOURCE'] == 'brainbeats':
        pol = resolve_polarity(_cat, fs)
    else:
        pol = ecg_polarity(_cat, fs)
    if verbose:
        print(f'  {tag}: ECG polarity {"upright" if pol > 0 else "INVERTED"}')

    # ── STAGE A: R-peaks per segment, in ABSOLUTE session samples ───────────
    seg_abs = np.asarray(d['seg_span'])            # [nSeg x 2] absolute spans
    # ── THE ANALYSIS WINDOW ─────────────────────────────────────────────────
    # Computed once, by `window_indices`, and the same [w0, w1) for every
    # segment: the window sits at a fixed offset from the segment's start,
    # because the start is the only end of a stored segment that reliably
    # encodes the report time (extract_npz.py's overlap resolution moves the
    # end). Segments that do not reach w1 were refused at gate 1b above, so
    # every index in idx_keep holds [w0, w1).
    n = w1 - w0                        # constant window length, EPOCH_DUR*fs + 1

    seg_peaks_abs = []
    # The peak amplitudes too. clean_rr's Step 2 (amplitude-outlier rejection)
    # needs the height of each R peak on the band-passed trace, and it is
    # skipped entirely when `peak_amp` is omitted (on the reference recordings
    # of rr_cleaning, Step 2 removes beats that Step 3 does not catch).
    #
    # They are taken from the same polarity-corrected window the peaks were
    # found in, band-passed with qrs_bandpass. Using the raw stored ECG instead
    # would take the amplitudes off a differently filtered trace than the
    # detection used.
    seg_peaks_amp = []
    for k in idx_keep:
        a = int(seg_abs[k][0]) + w0
        _win = pol * np.asarray(seg_ecg[k, w0:w1], float)
        if p['PEAK_SOURCE'] == 'brainbeats':
            # Polarity is forced to +1 because `_win` is already upright (it is
            # `pol * seg_ecg[...]`, with `pol` resolved once per session above).
            # get_rr would otherwise re-vote per window, on ~60 beats instead of
            # the session's few hundred, and could disagree with itself between
            # two windows of one recording.
            # drop_first=False: get_RR drops its first peak so that
            # len(peaks) == len(RR). Called once per 60 s window here, that
            # would cost a beat per segment; the clean_rr call below
            # re-establishes the convention itself.
            # A bad window must not kill the session. get_rr raises on a flat
            # line or when no QRS candidate is above threshold; an empty train
            # here flows into the MIN_BEATS_UNIT gate, which drops
            # the segment with a named reason, and the failure
            # is recorded.
            try:
                pk, _ = get_rr(_win, fs, polarity=1, drop_first=False)
            except GetRRError as exc:
                pk = np.array([], np.int64)
                notes.setdefault('getrr_failed_windows', []).append(
                    f'seg {int(k)}: {exc}')
        else:
            # Polarity is resolved once per session, above, not here.
            # ecg_polarity detects peaks both ways to decide, so calling
            # detect_rpeaks() per segment would double the cost (6.6 s vs 3.3 s
            # per segment measured) for an answer that cannot change within a
            # recording: polarity is a property of the electrode montage.
            pk = _detect_rpeaks_raw(_win, fs)
        seg_peaks_abs.append(pk.astype(np.int64) + a)
        _bp = qrs_bandpass(_win, fs)
        seg_peaks_amp.append(_bp[pk.astype(np.int64)]
                             if pk.size else np.array([], float))

    # Keep the pre-cleaning train. `ecg_quality.clean_ratio_segments` is
    # `1 - n_kept/n_all` over the analysed spans, so it needs both trains, and
    # `n_all` cannot be recovered after clean_rr has run. The spans are the
    # analysed windows (inclusive), not the stored segments.
    raw_abs_all = (np.concatenate(seg_peaks_abs) if seg_peaks_abs
                   else np.array([], np.int64))
    win_spans = np.array([[int(seg_abs[k][0]) + w0,
                           int(seg_abs[k][0]) + w1 - 1] for k in idx_keep],
                         dtype=np.int64)
    notes['n_beats_detected'] = int(raw_abs_all.size)

    # ── STAGE B: RR cleaning, pooled per meditation block ───────────────────
    # clean_rr must see block-level statistics, not per-segment ones. Step 1
    # derives min_rr/max_rr from the global median and MAD of the RR series,
    # and Step 4 from a ±7.5 s local median. Run per segment, those would come
    # from ~60-120 beats instead of the block's ~2000, and different beats would
    # be rejected.
    #
    # The npz stores only segments, so the beats are pooled into their block in
    # absolute samples and cleaned together. The RR intervals that straddle a
    # gap between two segments are many seconds long and Step 3 rejects them as
    # > max_rr, which is correct (they are not real RR intervals), while every
    # genuine within-segment interval is judged against block-level statistics.
    # Concatenate the inputs here, never the outputs.
    n_removed = 0
    if p['DO_CLEAN_RR']:
        blocks = np.atleast_2d(np.asarray(d['block_spans'])).astype(np.int64)
        all_abs = np.concatenate(seg_peaks_abs) if seg_peaks_abs else np.array([], np.int64)
        all_amp = (np.concatenate(seg_peaks_amp) if seg_peaks_amp
                   else np.array([], float))
        kept = np.zeros(all_abs.size, bool)
        for b0, b1 in blocks:
            m = (all_abs >= b0) & (all_abs < b1)
            if m.sum() < 3:
                kept |= m                  # too few to judge; keep rather than invent
                continue
            # Sort the amplitudes with the peaks. `all_abs` is the concatenation
            # of per-segment trains, so it is only sorted within a segment; the
            # amplitude of beat i must follow beat i through the sort or Step 2
            # judges the wrong beats.
            _o = np.argsort(all_abs[m])
            pk_b = all_abs[m][_o]
            amp_b = all_amp[m][_o] if all_amp.size == all_abs.size else None
            # Seconds: clean_rr takes RR in seconds, matching its default `rr_t`
            # axis (peaks - index_base)/fs. Milliseconds would make every
            # interval look 1000x too long and disable the 200 bpm hard rail.
            rr_b = np.diff(pk_b) / fs                   # SECONDS, per the contract
            # clean_rr requires len(peaks) == len(rr). n peaks give n-1
            # intervals, so the first peak is dropped and each RR is labelled by
            # the beat that ends it (as get_RR does). Keep pk_b[1:] and rr_b
            # aligned.
            try:
                # index_base=1: clean_rr is given 1-based peak indices.
                # peak_amp is aligned to the trailing peak of each interval,
                # like `peaks`, so it is amp_b[1:] and not amp_b.
                _rrc, pk_clean, _bad, _info = clean_rr(
                    rr_b, pk_b[1:] + 1, fs,
                    peak_amp=(None if amp_b is None else amp_b[1:]),
                    interpolate_missing=False,
                    index_base=1, verbose=False)
                if _info.get('amp_step', '').startswith('skipped'):
                    notes.setdefault('clean_rr_amp_skipped', []).append(
                        f'[{b0},{b1}): {_info["amp_step"]}')
                keep_abs = np.asarray(pk_clean, float)
                keep_abs = (keep_abs[np.isfinite(keep_abs)] - 1).astype(np.int64)
                # the dropped first peak of the block is never judged, so it is
                # kept explicitly rather than lost to the isin() below
                kept |= np.isin(all_abs, keep_abs) | (all_abs == pk_b[0])
            except Exception as exc:
                # Say which block failed and keep its beats, so an unclean block
                # can never masquerade as a clean one.
                print(f'  {tag}: clean_rr failed on block [{b0},{b1}) ({exc}) '
                      f'— beats KEPT UNCLEANED for this block')
                kept |= m
        # Beats outside every meditation block are dropped: the R-peak scope is
        # the meditation blocks, so a depth segment lying outside them (the pre-
        # or post-meditation reports) contributes no beats. This is a stated
        # decision, not a side effect of the loop above never setting their bit.
        in_block = np.zeros(all_abs.size, bool)
        for b0, b1 in blocks:
            in_block |= (all_abs >= b0) & (all_abs < b1)
        n_out = int((~in_block).sum())
        if n_out and verbose:
            print(f'  {tag}: {n_out} beat(s) lie outside every meditation block '
                  f'— dropped (the R-peak scope is the meditation blocks)')
        n_removed = int(in_block.sum() - (kept & in_block).sum())
        # split the surviving beats back out to their segments
        off, new_peaks = 0, []
        for pk_abs in seg_peaks_abs:
            new_peaks.append(pk_abs[kept[off:off + pk_abs.size]])
            off += pk_abs.size
        seg_peaks_abs = new_peaks
        if verbose:
            frac = n_removed / max(all_abs.size, 1)
            print(f'  {tag}: clean_rr removed {n_removed}/{all_abs.size} beats '
                  f'({100*frac:.1f}%)')

    clean_abs_all = (np.concatenate(seg_peaks_abs) if seg_peaks_abs
                     else np.array([], np.int64))
    notes['n_beats_after_cleanrr'] = int(clean_abs_all.size)

    # The QA train, captured before the MIN_IBI filter runs. Both subject-QA
    # metrics read the post-clean_rr, pre-MIN_IBI train. Reading the post-filter
    # train would make ibi_short_prop ~0 by construction: dropping a beat only
    # lengthens the surrounding interval, so no short interval survives the
    # filter that removed it (on one participant, ~45-50% of intervals are under
    # 600 ms before the filter and 0% after). `clean_ratio_segments` pairs the
    # detections with clean_rr's output, with no MIN_IBI step between them, for
    # the same reason.
    qa_abs_all = clean_abs_all.copy()

    # ── GATE 3: MIN_IBI_MS, a beat-level filter on the clean train ─────────
    #     rri_ms = [NaN; diff(peaks)] / fs * 1000
    #     keep   = isnan(rri_ms) | rri_ms >= MIN_IBI_MS
    # This runs after clean_rr, on the whole session's train in absolute
    # samples, and the first beat (whose interval is NaN) is kept.
    #
    # The .npz stores only the analysed segments, so the first beat of each
    # segment is measured against the last beat of the previous segment (a gap
    # of many seconds, which always passes), not against the beat immediately
    # before it in the continuous recording. A handful of segment-leading beats
    # with a short true interval therefore survive. The number dropped is
    # reported (`n_beats_min_ibi_dropped`).
    n_ibi_drop = 0
    if float(p.get('MIN_IBI_MS', 0)) > 0 and clean_abs_all.size:
        order = np.argsort(clean_abs_all, kind='stable')
        srt = clean_abs_all[order]
        rri_ms = np.full(srt.size, np.nan)
        if srt.size > 1:
            rri_ms[1:] = np.diff(srt) / fs * 1000.0
        ok_ibi_srt = ~np.isfinite(rri_ms) | (rri_ms >= float(p['MIN_IBI_MS']))
        ok_ibi = np.empty(srt.size, bool)
        ok_ibi[order] = ok_ibi_srt
        n_ibi_drop = int((~ok_ibi).sum())
        if n_ibi_drop:
            off, new_peaks = 0, []
            for pk_abs in seg_peaks_abs:
                new_peaks.append(pk_abs[ok_ibi[off:off + pk_abs.size]])
                off += pk_abs.size
            seg_peaks_abs = new_peaks
            clean_abs_all = (np.concatenate(seg_peaks_abs) if seg_peaks_abs
                             else np.array([], np.int64))
            if verbose:
                print(f'  {tag}: MIN_IBI_MS dropped {n_ibi_drop} beat(s) with a '
                      f'preceding interval < {float(p["MIN_IBI_MS"]):g} ms')
    notes['n_beats_min_ibi_dropped'] = n_ibi_drop
    notes['n_beats_after_min_ibi'] = int(clean_abs_all.size)

    # ── Per-session ECG quality, for the subject gate the caller applies ────
    # Reported, not gated here: the ECG criterion is a subject criterion,
    # averaged over that subject's observations. Gating the session on it would
    # exclude a different set.
    try:
        cr = clean_ratio_segments(raw_abs_all, qa_abs_all, win_spans)
        notes['ecg_clean_ratio'] = float(cr.ratio)
        notes['ecg_n_all'] = int(cr.n_all)
        notes['ecg_n_kept'] = int(cr.n_kept)
    except Exception as exc:                              # noqa: BLE001
        # NaN, not 0.0: "cannot be assessed" must fail the gate, never pass it.
        notes['ecg_clean_ratio'] = float('nan')
        notes['ecg_error'] = f'{type(exc).__name__}: {exc}'
    notes['ibi_short_prop'] = float(ibi_short_prop(
        qa_abs_all, fs, min_ibi_ms=float(p.get('MIN_IBI_MS', 600))))

    notes['n_beats_analysed'] = int(clean_abs_all.size)

    # ── STAGE C: epochable beats, ECG-quality gate, CWT, selection ──────────
    seg_peaks, seg_ok, seg_nbeat = [], [], []
    sel_chunks, ei_rows, offset = [], [], 0
    # The surrogate shift is applied here, and only here. `SURROGATE_SHIFT` is a
    # rigid offset in samples, applied to the cleaned peak train of every
    # segment of this session. Everything downstream (epoching, the CWT
    # indexing, HEP/HRSP/HEPC, the per-beat arrays) follows from `seg_peaks`,
    # so shifting at this one point re-epochs the whole session at the shifted
    # beats without touching the continuous signal.
    #
    # That is also why it is safe against the cache trap surrogate_rpeaks warns
    # about: this route holds no epoched CWT cache, and the transform in PASS 2
    # is taken from the continuous segment signal after this shift. A cached
    # epoched CWT would return the real epochs and make the surrogate null
    # identical to the data.
    #
    # A shift that pushes a beat's epoch outside its segment drops that beat at
    # the bounds filter below, which changes the beat count and therefore the
    # L1 weights. The caller must choose a shift that does not
    # (surrogate_rpeaks.make_surrogate_rpeaks_retry does this against a guard);
    # `notes['surrogate_beats_lost']` reports it.
    #
    # `SURROGATE_SHIFT` may be an int (one rigid shift everywhere) or a dict
    # {(subject, session): shift}. The dict form is the correct one: which
    # shifts are admissible depends on where a session's first and last beats
    # sit inside its analysed window, so one cohort-wide shift is admissible for
    # some sessions and out of bounds for others. A session missing from a
    # non-empty dict is an error, not a zero shift (it would put real peaks into
    # the null); a session the driver could not shift must be in SURROGATE_DROP.
    _ss = p.get('SURROGATE_SHIFT')
    if isinstance(_ss, dict):
        _k = (str(sid), int(ses))
        if _k in _ss:
            _surr_shift = int(_ss[_k] or 0)
        elif (sid, ses) in _ss:
            _surr_shift = int(_ss[(sid, ses)] or 0)
        elif _ss:
            raise ValueError(
                f'{tag}: SURROGATE_SHIFT is a dict of {len(_ss)} session(s) and '
                f'this one is not in it, nor in SURROGATE_DROP. Analysing it at '
                f'shift 0 would put REAL, UNSHIFTED epochs into the surrogate '
                f'null. Add it to SURROGATE_SHIFT or to SURROGATE_DROP.')
        else:
            _surr_shift = 0
    else:
        _surr_shift = int(_ss or 0)
    _surr_lost = 0
    for j, k in enumerate(idx_keep):
        a = int(seg_abs[k][0]) + w0
        pk = seg_peaks_abs[j] - a
        if _surr_shift:
            _n_before = int(((pk + lo >= 0) & (pk + hi < n)).sum())
            pk = pk + _surr_shift
            _surr_lost += _n_before - int(((pk + lo >= 0) & (pk + hi < n)).sum())
        # only beats whose FULL epoch fits inside the segment
        pk = pk[(pk + lo >= 0) & (pk + hi < n)]
        seg_peaks.append(pk); seg_nbeat.append(int(pk.size))
        # A segment with fewer than MIN_BEATS_UNIT epochable beats is not a
        # unit, for any measure.
        if pk.size < int(p.get('MIN_BEATS_UNIT', 1)):
            seg_ok.append(False)
            continue
        seg_ok.append(True)

        # The selection CWT is the dominant cost of this function (every
        # component at every selection frequency, for every segment), so it is
        # not built when the selection is pinned.
        if not _pinned:
            if _cwt_seg:
                eeg = np.asarray(seg_eeg[k, :, :int(seg_len[k])], float)
            else:
                eeg = np.asarray(seg_eeg[k, :, w0:w1], float)
            # icawinv carries the component SIGN CONVENTION (reconstruct_ics).
            cand = reconstruct_ics(eeg, icaw, icas, icach,
                                   icawinv=icawinv)           # [nIC x n]
            co = morlet_cwt(cand, freqs[f_sel], fs, width=p['HRSP_WAVE_WIDTH'])
            if _cwt_seg:
                co = co[..., w0:w1]
            sel_chunks.append(co.astype(np.complex64))
        # ei_rows / offset are built regardless. Only the Lee scorer reads them,
        # but `offset` is the running concatenation position, and skipping it
        # would break anything else that reads it.
        ei_rows.append(np.stack([np.arange(q + lo, q + hi + 1) + offset for q in pk]))
        offset += n

    seg_ok = np.asarray(seg_ok)
    notes['n_seg_with_beats'] = int(seg_ok.sum())
    notes['n_seg_hepc_ok'] = int((np.asarray(seg_nbeat)[seg_ok]
                                  >= int(p['MIN_BEATS_UNIT'])).sum())
    # Reported even when zero, so a surrogate run can assert it. A rigid shift
    # that loses beats changes the L1 weights, and a surrogate whose weights
    # differ from the real fit is not a null for that fit.
    if _surr_shift:
        notes['surrogate_shift'] = _surr_shift
        notes['surrogate_beats_lost'] = int(_surr_lost)
    if seg_ok.sum() == 0:
        notes['skip_reason'] = (
            f'no segment carried a single epochable beat (MIN_BEATS_UNIT '
            f'{int(p.get("MIN_BEATS_UNIT", 1))})')
        if verbose:
            print(f'  {tag}: no segment carried an epochable beat — skipped')
        return None

    # ── SELECTION: ONE OR MORE ICs PER SESSION ──────────────────────────────
    # Per SESSION, over that session's pooled beats, never per segment and never
    # per depth level: selecting inside a depth level would make the depth
    # contrast partly a selection artefact.
    if not _pinned:
        pooled = np.concatenate(sel_chunks, axis=2)               # [nIC x nFsel x N]
        ei = np.concatenate(ei_rows, axis=0)                      # [nBeats x nT]
        t_bl = (times >= p['BASELINE_WIN'][0]) & (times <= p['BASELINE_WIN'][1])
        t_sel = (times >= p['HRSP_SEL_WIN'][0]) & (times <= p['HRSP_SEL_WIN'][1])
        sel, _w, _pool = select_ics_lee(
            pooled, ei, t_bl, t_sel,
            criterion=p['HRSP_SEL_CRITERION'], norm_mode=p['IC_SEL_NORM'],
            ic_sel_thresh=p['IC_SEL_THRESH'], ic_pool=None,
            n_pts=offset,
            sham_opts=dict(fs=fs, n=p['IC_SEL_SHAM_N'], shift_ms=(-500, 500),
                           min_shift_ms=150, seed=20260728))
        sel_ics = np.atleast_1d(np.asarray(sel, int)).ravel()
        # Every component's score is kept, not only the winner's (nIC floats).
        notes['lee_scores'] = [float(v) for v in np.asarray(_pool, float).ravel()]
        del pooled, sel_chunks
    else:
        # Order is meaningful: element 0 is the top component and every other
        # selected component is sign-aligned to ITS topography below.
        sel_ics = presel
    ic_sel = int(sel_ics[0])
    notes['ic_selected'] = [int(v) for v in sel_ics]
    notes['n_ic_selected'] = int(sel_ics.size)

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 2: measures, on the selected IC only (61x less memory than pass 1)
    # ═══════════════════════════════════════════════════════════════════════
    nF = len(freqs)
    hep = np.full((len(idx_keep), nT), np.nan)
    hrsp = np.full((len(idx_keep), nF, nT), np.nan)
    hepc = np.full((len(idx_keep), nF, nT), np.nan)

    unmix = icaw if icas is None else icaw @ icas

    # ── The sign-corrected, unit-norm topographies ──────────────────────────
    # topos = icawinv ./ vecnorm(icawinv), then each column negated if its
    # peak-absolute weight is negative. The combination below aligns the
    # selected ICs by the dot product of these columns, so the column norm and
    # the sign convention both enter the alignment.
    _tn = np.linalg.norm(icawinv, axis=0)
    topos = icawinv / np.where(_tn > 0, _tn, 1.0)
    _pk = np.argmax(np.abs(topos), axis=0)
    _neg = topos[_pk, np.arange(topos.shape[1])] < 0
    topos[:, _neg] *= -1.0

    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-IC COMBINATION
    # ═══════════════════════════════════════════════════════════════════════
    # Under 'lee' at IC_SEL_THRESH=1 exactly one IC is selected, and every step
    # below reduces to using it: s = sign(ref . ref) = +1 and the average of one
    # element is itself. The general form serves selections of several ICs.
    #
    # The order of operations is not interchangeable: each IC is normalised by
    # its own SD and then the ICs are averaged; the average is not normalised.
    # Normalising afterwards would let the loudest component dominate the mean
    # and give a different scale for every session, and scale is exactly what
    # IC_UNIT_VAR_NORM exists to fix.
    #
    # The polarity alignment is not cosmetic. Two ICs with opposite topographic
    # sign carry the same field with opposite polarity; averaging them unaligned
    # cancels the HEP toward zero. `s` is the sign of the dot product with the
    # top IC's map, which is why the order of sel_ics matters.
    sign_ref = topos[:, ic_sel]
    _xs, _xsf, per_ic = {}, {}, []
    for rank, ic_k in enumerate(int(v) for v in sel_ics):
        # The sign convention must be applied here too. This pass builds each
        # IC straight from the unmixing row and never calls `reconstruct_ics`,
        # so the icawinv correction there does not reach the measures. Any IC
        # whose peak-absolute icawinv weight is negative is negated (a large
        # share of the selected ICs, whose HEP would otherwise be inverted).
        # HRSP (|CWT|^2) and HEPC (unit phasors) are sign-invariant.
        w_k = unmix[ic_k]
        pk_ch = int(np.argmax(np.abs(icawinv[:, ic_k])))
        flipped = bool(icawinv[pk_ch, ic_k] < 0)
        if flipped:
            w_k = -w_k
        xs_k, xsf_k = {}, {}
        for j, k in enumerate(idx_keep):
            if not seg_ok[j]:
                continue
            if _cwt_seg:
                eeg = np.asarray(seg_eeg[k, :, :int(seg_len[k])], float)
                xsf_k[j] = w_k @ (eeg if icach is None or icach.size == 0
                                  else eeg[icach])
                xs_k[j] = xsf_k[j][w0:w1]
            else:
                eeg = np.asarray(seg_eeg[k, :, w0:w1], float)
                xs_k[j] = w_k @ (eeg if icach is None or icach.size == 0
                                 else eeg[icach])

        # ── IC_UNIT_VAR_NORM, per IC ────────────────────────────────────────
        # Each selected IC is divided by its SD over the analysed segments,
        # once per observation, so a subject's two sessions enter the pooled L1
        # on the same scale.
        #
        # It is not a harmless global constant: it is per session, and most
        # subjects have two sessions, so it changes the pooled slope and the
        # Huber weights rather than cancelling. The depth slope is in signal
        # units: HEP scales by 1/sd and HRSP power by 1/sd^2; HEPC (phase) is
        # scale-invariant.
        #
        # ddof=1, the sample SD (numpy's default is ddof=0).
        #
        # The SD is taken over the analysed windows, already overlap-resolved
        # above, so no sample is counted twice.
        nf_k, sd_k = 1.0, np.nan
        if bool(p.get('IC_UNIT_VAR_NORM', True)) and xs_k:
            sd_k = float(np.std(np.concatenate([xs_k[j] for j in sorted(xs_k)]),
                                ddof=1))
            if sd_k > 0:
                nf_k = 1.0 / sd_k
            else:
                # A constant IC over every analysed segment is a broken
                # decomposition, not something to divide by. Named, not silently
                # left at 1.0.
                notes.setdefault('ic_unit_var_norm_skipped', []).append(
                    f'IC {ic_k}: SD over the analysed segments is 0 — constant '
                    'there; normalisation skipped')
                if verbose:
                    print(f'  {tag}: IC_UNIT_VAR_NORM skipped for IC {ic_k} '
                          f'— SD is 0')

        # s = sign(ref . topo_k), with 0 taken as +1.
        s_k = float(np.sign(float(sign_ref @ topos[:, ic_k]))) or 1.0
        per_ic.append(dict(ic=ic_k, rank=rank, sign=s_k, sd=sd_k,
                           winv_flipped=flipped))
        for j, v in xs_k.items():
            _xs[j] = _xs.get(j, 0.0) + s_k * nf_k * v
        # The whole segment, combined the same way (same SD, taken over the
        # windows), for the transform under WAVELET_ON='segment'.
        for j, v in xsf_k.items():
            _xsf[j] = _xsf.get(j, 0.0) + s_k * nf_k * v

    # The average, after the per-IC normalisation.
    n_sel = int(sel_ics.size)
    for j in _xs:
        _xs[j] = _xs[j] / n_sel
    for j in _xsf:
        _xsf[j] = _xsf[j] / n_sel

    notes['ic_combine'] = per_ic
    notes['ic_sign_flipped'] = bool(per_ic[0]['winv_flipped'])
    notes['ic_unit_var_norm_sd'] = float(per_ic[0]['sd'])

    # ═══════════════════════════════════════════════════════════════════════
    # CFA TOPOGRAPHY DIAGNOSTIC
    # ═══════════════════════════════════════════════════════════════════════
    # r2 between the analysed signal's scalp map and this session's empirical
    # cardiac field (the beat-averaged scalp topography at the R-peak).
    #
    # Diagnostic only: it gates nothing, feeds no model and changes no number.
    # A high r2 flags a signal carrying the cardiac field's spatial signature; a
    # low r2 does not clear one. Read it beside cluster latency.
    #
    # The template is topos @ mean(ic_act[:, R-peaks]). The mean and the
    # unmixing are both linear, so the mean scalp map at the R-peaks is
    # accumulated first and that single vector is pushed through
    # reconstruct_ics, which gives the same result without a per-segment IC
    # reconstruction. The candidate topography is the sign-aligned average of
    # the selected ICs' maps, with equal weights (exact for a single IC).
    try:
        _acc = np.zeros(int(seg_eeg.shape[1]), float)
        _cnt = 0
        for j, k in enumerate(idx_keep):
            if not seg_ok[j]:
                continue
            a0 = int(seg_abs[k][0]) + w0
            loc = np.asarray([int(pk) - a0 for pk in seg_peaks_abs[j]], int)
            loc = loc[(loc >= 0) & (loc < n)]
            if loc.size == 0:
                continue
            _acc += np.asarray(seg_eeg[k, :, w0:w1], float)[:, loc].sum(axis=1)
            _cnt += int(loc.size)
        if _cnt > 0:
            mean_scalp_at_R = _acc / float(_cnt)
            cfa_src = reconstruct_ics(
                mean_scalp_at_R[:, None], icaw, icas, icach,
                icawinv=icawinv)[:, 0]
            cfa_map = cfa_topo_template(cfa_src[:, None], topos,
                                        np.array([0], int))
            tv = np.zeros(int(topos.shape[0]), float)
            for _d in per_ic:
                tv += float(_d['sign']) * np.asarray(topos[:, int(_d['ic'])],
                                                     float)
            tv = tv / float(max(len(per_ic), 1))
            notes['cfa_topo_r2'] = cfa_topo_r2(tv - tv.mean(), cfa_map)
            notes['cfa_topo_nbeats'] = int(_cnt)
        else:
            notes['cfa_topo_r2'] = float('nan')
    except Exception as _e:
        # Never let a diagnostic kill a session. nan means NOT MEASURED.
        notes['cfa_topo_r2'] = float('nan')
        notes['cfa_topo_r2_error'] = str(_e)
    if verbose and np.isfinite(notes.get('cfa_topo_r2', np.nan)):
        _flag = ("   << shares the cardiac field's spatial signature"
                 if notes['cfa_topo_r2'] >= 0.5 else '')
        print(f"  {tag}: CFA topography r2 = {notes['cfa_topo_r2']:.3f}{_flag}")

    # ═══════════════════════════════════════════════════════════════════════
    # PER-SEGMENT SUFFICIENT STATISTICS: what the cell grain is built from
    # ═══════════════════════════════════════════════════════════════════════
    # The cell grain needs no per-beat arrays, exactly. A cell's HEP, HRSP and
    # HEPC are the mean voltage, the mean power and (|sum z|^2 - N)/(N(N-1))
    # over the beats of the cell, all functions of a sum and a count. Depth is
    # constant within a segment and every segment lies inside one session, so a
    # (session x depth) cell is exactly a set of whole segments: summing these
    # per-segment sums over a cell's segments reproduces the per-beat
    # computation.
    #
    # The counts are stored per cell of the map, not as one scalar per segment.
    # `ppc_map` drops non-finite phasors per (freq, time) cell and `hep_average`
    # is a nanmean, so a segment's valid N can differ across the map.
    #
    # Cost on this cohort's shapes: ~3 MB per session on top of the measures.
    # The per-beat grain is not served by these (it needs the beats
    # themselves); see `keep_perbeat`.
    nS = len(idx_keep)
    sum_hep = np.zeros((nS, nT))
    n_hep = np.zeros((nS, nT), dtype=np.int32)
    sum_pow = np.zeros((nS, nF, nT))
    n_pow = np.zeros((nS, nF, nT), dtype=np.int32)
    sum_z = np.zeros((nS, nF, nT), dtype=complex)
    n_z = np.zeros((nS, nF, nT), dtype=np.int32)
    # Opt-in, because it is ~110 MB per session at this cohort's shapes and the
    # segment/cell grains never touch it. HEPC is absent: PPC is a set-level
    # statistic with no per-beat form.
    keep_perbeat = bool(p.get('KEEP_PERBEAT', False))
    pb_hep, pb_pow, pb_seg = [], [], []

    for j in sorted(_xs):
        x = _xs[j]
        if _cwt_seg:
            co = morlet_cwt(_xsf[j], freqs, fs, width=p['HRSP_WAVE_WIDTH'])[:, w0:w1]
        else:
            co = morlet_cwt(x, freqs, fs, width=p['HRSP_WAVE_WIDTH'])
        pk = seg_peaks[j]
        idx = np.stack([np.arange(q + lo, q + hi + 1) for q in pk])   # [nBeats x nT]
        ep = x[idx]                                                   # [nBeats x nT]
        cb = co[:, idx].transpose(1, 0, 2)                            # [nBeats x nF x nT]
        hep[j] = hep_average(ep)
        hrsp[j] = np.mean(np.abs(cb) ** 2, axis=0)
        pw = np.abs(cb) ** 2
        z = cb / (np.abs(cb) + EPS)
        hepc[j] = ppc_map(z)

        # ── the sufficient statistics, with the SAME validity rule each
        #    reducer uses: nanmean for HEP/HRSP, isfinite for the phasors.
        ok_e = np.isfinite(ep)
        sum_hep[j] = np.where(ok_e, ep, 0.0).sum(axis=0)
        n_hep[j] = ok_e.sum(axis=0)
        ok_p = np.isfinite(pw)
        sum_pow[j] = np.where(ok_p, pw, 0.0).sum(axis=0)
        n_pow[j] = ok_p.sum(axis=0)
        ok_z = np.isfinite(z)
        sum_z[j] = np.where(ok_z, z, 0).sum(axis=0)
        n_z[j] = ok_z.sum(axis=0)

        if keep_perbeat:
            pb_hep.append(ep.astype(np.float32))
            pb_pow.append(pw.astype(np.float32))
            pb_seg.append(np.full(ep.shape[0], j, dtype=np.int32))

    good = seg_ok
    # `ic` stays the top IC (scalar); `ic_all` is the full selection (1 element
    # under 'lee' at IC_SEL_THRESH=1).
    # ── the cell-grain sufficient statistics, gated to the same segments ────
    # Masked with `good` like every other returned array. A sums array that
    # kept the refused segments while `hep`/`depth`/`cond` dropped them would
    # misalign the cell grain against the segment grain.
    cellstats = dict(sum_hep=sum_hep[good], n_hep=n_hep[good],
                     sum_pow=sum_pow[good], n_pow=n_pow[good],
                     sum_z=sum_z[good], n_z=n_z[good])
    perbeat = None
    if keep_perbeat and pb_hep:
        # The segment id is remapped to the retained segment index, so a beat
        # can be traced to the row of `depth`/`cond` it belongs to. Without the
        # remap it would point into the pre-gate segment list.
        _remap = {int(j): i for i, j in enumerate(np.flatnonzero(good))}
        keepb = [np.array([_remap.get(int(v), -1) for v in s]) for s in pb_seg]
        sel_b = [k >= 0 for k in keepb]
        perbeat = dict(
            hep=np.concatenate([a[m] for a, m in zip(pb_hep, sel_b) if m.any()]),
            pow=np.concatenate([a[m] for a, m in zip(pb_pow, sel_b) if m.any()]),
            seg=np.concatenate([k[m] for k, m in zip(keepb, sel_b) if m.any()]))
    # ═══════════════════════════════════════════════════════════════════════
    # PER-SEGMENT COVARIATES (p['KEEP_SEGCOV'])
    # ═══════════════════════════════════════════════════════════════════════
    # Computed here because the beats are here: `seg_peaks` is the cleaned,
    # IBI-filtered, epochable train in window coordinates, and re-deriving it
    # downstream would mean a second R-peak detection that could drift from
    # this one.
    #
    # Per segment, not per beat: the per-beat arrays are ~1 GB across a cohort
    # and the primary grain is the segment. At the per-beat grain a covariate
    # is therefore broadcast to its segment's beats, so an adjusted model there
    # removes between-segment covariate variance only.
    segcov = segment_covariates(
        p, d, idx_keep, good, seg_peaks, seg_ecg, seg_eeg, seg_len, times, fs,
        w0, w1, lo, hi, _xs, notes)

    # ── Opt-in: the exact beats each retained segment was epoched on ────────
    # For analyses that must see the same heartbeats as this run on other
    # signals (for example sensor data projected to cortex). `seg` is the npz
    # segment index, `peaks` are in analysed-window coordinates (relative to
    # w0), ordered like `depth`/`hrsp`. Off by default; nothing in the run
    # reads it.
    beat_samples = None
    if bool(p.get('KEEP_BEAT_SAMPLES', False)):
        _kept = np.flatnonzero(good)
        beat_samples = dict(seg=[int(idx_keep[j]) for j in _kept],
                            peaks=[np.asarray(seg_peaks[j], np.int64) for j in _kept],
                            w0=int(w0), w1=int(w1), lo=int(lo), hi=int(hi), fs=float(fs))

    return dict(subject=sid, session=ses, ic=ic_sel, beat_samples=beat_samples,
                ic_all=[int(v) for v in sel_ics], ic_method=method, fs=fs,
                cellstats=cellstats, perbeat=perbeat, segcov=segcov,
                depth=depth[idx_keep][good], cond=cond[idx_keep][good],
                hep=hep[good], hrsp=hrsp[good], hepc=hepc[good],
                n_beats=np.asarray(seg_nbeat)[good],
                # The cleaned peak train, for the surrogate control.
                # min_abs_shift sets its rejection band from the median IBI, so
                # a shift drawn without the real train uses the wrong band, and
                # at meditator heart rates that band decides whether any shift
                # is admissible (see surrogate_rpeaks). Peaks are relative to
                # the segment's analysed window. One segment's train (the one
                # with the most beats), not a concatenation: peaks are relative
                # to their own segment, so a concatenation's diffs are
                # meaningless at every boundary. All segments share one window
                # length, so any one gives the right bound; the longest gives
                # the best IBI estimate. seg_len_samp is the bound a shift must
                # respect.
                peaks_rel=(max((np.asarray(v, np.int64) for v in seg_peaks),
                               key=lambda a: a.size)
                           if seg_peaks else np.array([], np.int64)),
                seg_len_samp=int(n),
                times=times, freqs=freqs)


# ═══════════════════════════════════════════════════════════════════════════
# L1: per-subject depth slopes
# ═══════════════════════════════════════════════════════════════════════════
def _cells_of_session(s, measure, p):
    """Aggregate one session's segments into (session x depth) cells.

    Exact, not an approximation, which is why the cell grain costs almost
    nothing here. A cell's HEP, HRSP and HEPC are the per-beat mean, the mean
    power and (|sum z|^2 - N)/(N(N-1)), each a function of a sum and a count.
    Depth is constant within a segment and every segment lies inside one
    session, so a cell is exactly a set of whole segments, and summing
    `session_measures`' per-segment sums over them reproduces the per-beat
    computation (checked on one session: max|d| = 0.0 per segment for HEP,
    HRSP and HEPC, and ~5e-9 for a depth cell against a float32 per-beat
    reference).

    The returned beat count is per cell, so the HEPC beat floor in
    subject_slopes is applied to a cell's pooled beats.

    Returns (values [nCells x ...], depth [nCells], n_beats [nCells]).
    """
    cs = s.get('cellstats')
    if cs is None:
        raise KeyError(
            f"{s.get('subject')} ses{s.get('session')}: this session dict has no "
            "`cellstats`, so the CELL grain cannot be built. It is produced by "
            "heart_depth_pipeline.session_measures; a session loaded from an "
            "older cached result predates it and must be recomputed.")
    dep = np.asarray(s['depth']).astype(int)
    lev = np.unique(dep)
    meas = measure.upper()
    vals, nbs = [], []
    for lv in lev:
        m = dep == lv
        if meas == 'HEP':
            num, den = cs['sum_hep'][m].sum(0), cs['n_hep'][m].sum(0)
        elif meas == 'HRSP':
            num, den = cs['sum_pow'][m].sum(0), cs['n_pow'][m].sum(0)
        elif meas == 'HEPC':
            num, den = None, cs['n_z'][m].sum(0)
        else:
            raise ValueError(f'unknown measure {measure!r}')
        if meas == 'HEPC':
            N = den.astype(float)
            with np.errstate(invalid='ignore', divide='ignore'):
                v = (np.abs(cs['sum_z'][m].sum(0)) ** 2 - N) / (N * (N - 1))
            v = np.where(N < 2, np.nan, v)
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                v = num / np.where(den > 0, den, np.nan)
        vals.append(v)
        # The cell's beat count is the count the measure actually used, taken
        # at the map's own resolution and reduced by max: `n_beats` per segment
        # is a scalar, but a cell's usable N can vary across the map when a
        # beat carried a non-finite value.
        nbs.append(int(np.max(den)) if np.size(den) else 0)
    if not vals:
        return np.zeros((0,)), np.zeros(0), np.zeros(0, dtype=np.int64)
    return (np.stack(vals, axis=0), lev.astype(float),
            np.asarray(nbs, dtype=np.int64))


def _beats_of_session(s, measure, p):
    """One session's per-beat units.

    The data come from `session_measures`, not from a second epoching pass.
    Under KEEP_PERBEAT it keeps `perbeat['hep']` [nBeats x nT],
    `perbeat['pow']` [nBeats x nF x nT] and `perbeat['seg']`, the retained
    segment index of each beat: the same epochs the segment averages were built
    from, so the two grains cannot drift apart by an epoching difference.

    HEPC has no per-beat form and is refused here. PPC is defined over a set
    of phasors; for one beat (|sum z|^2 - N) / (N(N-1)) has N = 1 and is 0/0.

    Returns (values, depth, seg_id) with seg_id the CRVE cluster.
    """
    meas = measure.upper()
    if meas == 'HEPC':
        raise ValueError(
            'HEPC has no per-beat form: PPC is a set-level statistic and is 0/0 '
            'over a single beat. Exclude it from MEASURES at the per-beat grain '
            '(heart_depth_run.GRAIN_MEASURES does) rather than asking for it.')
    pb = s.get('perbeat')
    if pb is None:
        raise KeyError(
            f"{s.get('subject')} ses{s.get('session')}: this session dict has no "
            "`perbeat`, so the PER-BEAT grain cannot be built. It is produced by "
            "heart_depth_pipeline.session_measures ONLY when p['KEEP_PERBEAT'] "
            "is true - set it before building the cohort; it cannot be "
            "recovered from the segment averages.")
    seg = np.asarray(pb['seg']).astype(int)
    v = np.asarray(pb['hep' if meas == 'HEP' else 'pow'])
    dep = np.asarray(s['depth']).astype(int)[seg]
    return v, dep, seg


# ═══════════════════════════════════════════════════════════════════════════
# COVARIATE MODELS: which design column(s) each one contributes
# ═══════════════════════════════════════════════════════════════════════════
# Maps a model name to the segcov key(s) that supply its column(s).
#
# A covariate model carries no session dummies, by design: 'rr' is
# [intercept, depth, rr], not [intercept, depth, session..., rr]. The
# session+<cov> models below add them.
#
# respPhase is two columns. Phase is circular, so it enters as sin and cos of
# the respiration phase at the R-peak. It therefore spends 2 df where every
# other covariate spends 1, so at the margin a subject can be fit for 'rr' and
# refused for 'respPhase' in the same run (the same df trap the Alday baseline
# column creates for HEP vs HEPC).
#
# 'cond' does not come from segcov. The SAPO indicator is a property of the
# depth report and is already carried on the session dict as `cond`
# (0 = nosapo, 1 = sapo), so it needs no KEEP_SEGCOV pass. It is meaningful
# only when INCLUDE_SAPO is on; otherwise it is a constant column, which
# subject_slopes drops as degenerate.
COVARIATE_MODELS = {
    'theta':        ('theta',),
    'alpha':        ('alpha',),
    'beta':         ('beta',),
    'rr':           ('rr',),
    'ecgAmp':       ('ecgAmp',),
    'ecgAmpT':      ('ecgAmpT',),
    'respAmp':      ('respAmp',),
    'respPhase':    ('respPhase_sin', 'respPhase_cos'),
    'reportmethod': ('reportmethod',),
    'cond':         ('cond',),
}
# ═══════════════════════════════════════════════════════════════════════════
# MODERATION MODELS: does the depth slope itself change with the covariate?
# ═══════════════════════════════════════════════════════════════════════════
# A covariate model asks 'is the depth effect still there once X is adjusted
# for' (confounding). A moderation model asks 'is the depth effect stronger at
# some values of X than others'. They are not interchangeable and neither
# implies the other.
#
# Design: [intercept, depth, X, depth*X]. Both main effects are always
# present, by construction: an interaction without its main effects is
# uninterpretable.
#
# The coefficient that goes to L2 is the interaction, not the depth slope
# (cell_ols's `target_col`). A cluster in one of these maps means 'the depth
# slope varies with X here'; it is not evidence of a depth effect and must
# never be reported as one.
#
# Centring does not change the interaction coefficient. Replacing d by (d - a)
# changes the product column by -a*X, a multiple of the X column already in
# the design; the span is unchanged and only X's own coefficient absorbs the
# shift. Both are still centred here for conditioning.
#
# respPhase is excluded: it is two columns (sin, cos), so its moderation would
# be two interaction coefficients with no single signed map for L2 to test,
# and testing the sine alone would be an arbitrary choice of phase origin.
#
# The df cost is real at the cell grain: these designs need 4 parameters
# against ~4-10 cells per subject, where unadj needs 2. More subjects are
# refused, so a cell-grain moderation map can rest on far fewer subjects than
# its unadj sibling. l1_fit_summary.csv records the refusals per model; read
# it before comparing a moderation map with a main-effect one.
INTERACTION_SUFFIX = 'Xdepth'
INTERACTION_MODELS = {f'{c}{INTERACTION_SUFFIX}': c
                      for c in COVARIATE_MODELS if c != 'respPhase'}

# ── SESSION + COVARIATE ────────────────────────────────────────────────────
# `session+<cov>` fits [intercept, depth, sesD..., X]: the primary design plus
# one covariate column. The plain covariate models above carry no session
# dummies, so they are read against the unadj map; these are the like-for-like
# confound tests for the primary model.
# 'cond' has no session+ form: under INCLUDE_SAPO=False it is constant, and
# under INCLUDE_SAPO=True a sapo block sits inside one session.
SESSION_COV_PREFIX = 'session+'
SESSION_COV_MODELS = {f'{SESSION_COV_PREFIX}{c}': c
                      for c in COVARIATE_MODELS if c != 'cond'}

# Everything subject_slopes accepts as `model`.
ALL_MODELS = (('unadj', 'session') + tuple(COVARIATE_MODELS)
              + tuple(INTERACTION_MODELS) + tuple(SESSION_COV_MODELS))
# Which models need the KEEP_SEGCOV pass in session_measures. 'cond' does not:
# it rides on the session dict. A moderation model needs whatever its base
# covariate needs, so it is derived, never restated: adding a covariate to
# COVARIATE_MODELS automatically gives its moderation model the same gate.
def _base_of(model):
    """The covariate a model rests on: itself, or the base of an interaction."""
    return INTERACTION_MODELS.get(model, SESSION_COV_MODELS.get(model, model))


SEGCOV_MODELS = tuple(sorted(
    {_base_of(m) for m in tuple(COVARIATE_MODELS) + tuple(INTERACTION_MODELS)}
    - {'cond'}))


def _stack_cov(cov_cols, bl_col):
    """Model covariate(s) first, Alday baseline last.

    The coefficient row indices callers read are counted from that order.
    Appending the baseline first would not error; it would return a different
    coefficient under the covariate's name.
    """
    if cov_cols is None:
        return bl_col
    if bl_col is None:
        return cov_cols
    return np.hstack([cov_cols, bl_col])


def _covariate_columns(s, model, grain, p):
    """One session's covariate column(s) for `model`, expressed at `grain`.

    Returns [nUnits x k] with nUnits matching what the grain helper returns for
    this same session, or raises with a named reason. It never returns a
    silently wrong-length column: a length mismatch would misalign a covariate
    against the DV and produce a plausible, wrong adjusted map.

    The covariates exist at the segment only: session_measures computes one
    value per depth-report segment (the KEEP_SEGCOV block) because the per-beat
    arrays are ~1 GB across a cohort. The other two grains are derived here:

      segment  used as-is.

      cell     beat-weighted mean over the segments of that (session x depth)
               cell, in the same depth order _cells_of_session emits
               (np.unique(dep), ascending); the two must agree row for row. A
               segment value is already that segment's beat mean, so weighting
               by the segment's beat count reconstructs the cell's beat mean.
               Exact for ecgAmp / ecgAmpT / respAmp / respPhase (per-beat
               quantities averaged under the same weights) and for
               reportmethod / cond (constant within a segment). Approximate for
               'rr' by one beat per segment: rr_seg is a mean over the
               segment's pk.size-1 intervals while the weight is pk.size beats.
               The error is O(1/nBeats) per segment and does not bias the depth
               contrast.

      perbeat  the segment's value broadcast to each of its beats. This is a
               weaker adjustment than a genuine per-beat column: every beat in a
               segment carries the same value, so the model removes only
               between-segment covariate variance. A per-beat covariate model
               that survives here has cleared a lower bar; say so when
               reporting a per-beat survival.
    """
    if model not in COVARIATE_MODELS:
        raise ValueError(f'{model!r} is not a covariate model; expected one of '
                         f'{sorted(COVARIATE_MODELS)}')
    keys = COVARIATE_MODELS[model]
    dep = np.asarray(s['depth']).astype(int)
    tag = f"{s.get('subject')} ses{s.get('session')}"

    if model == 'cond':
        raw = [np.asarray(s['cond'], float)]
    else:
        sc = s.get('segcov')
        if sc is None:
            raise KeyError(
                f"{tag}: model {model!r} needs per-segment covariates but this "
                "session dict has no `segcov`. session_measures writes it ONLY "
                "when p['KEEP_SEGCOV'] is true — set it before building the "
                "cohort; it cannot be recovered from the segment averages.")
        absent = [k for k in keys if k not in sc]
        if absent:
            # A NAMED refusal, not an all-NaN column. respAmp / respPhase are
            # absent whenever the .npz carried no respiration; NaNs would read
            # downstream as "measured and uninformative" rather than "never
            # measured", and the model would silently degrade to unadj.
            raise KeyError(
                f"{tag}: model {model!r} needs segcov key(s) {absent}, which "
                f"this session has not got (it has {sorted(sc)}). For respAmp / "
                "respPhase this means the .npz carried no respiration channel.")
        raw = [np.asarray(sc[k], float) for k in keys]

    cols = []
    for v in raw:
        if v.shape[0] != dep.shape[0]:
            raise ValueError(
                f'{tag}: covariate {model!r} has {v.shape[0]} value(s) but the '
                f'session has {dep.shape[0]} retained segment(s). These must be '
                'the same list, in the same order.')
        if grain == 'segment':
            cols.append(v)
        elif grain == 'cell':
            nb = np.asarray(s['n_beats']).astype(float)
            lev = np.unique(dep)               # SAME order as _cells_of_session
            out = np.full(lev.size, np.nan)
            for i, lv in enumerate(lev):
                m = (dep == lv) & np.isfinite(v)
                if not m.any():
                    continue                   # stays NaN -> zeroed after centring
                w = nb[m]
                out[i] = (float(np.sum(v[m] * w) / np.sum(w)) if np.sum(w) > 0
                          else float(np.mean(v[m])))
            cols.append(out)
        elif grain == 'perbeat':
            pb = s.get('perbeat')
            if pb is None:
                raise KeyError(
                    f"{tag}: the per-beat grain needs `perbeat`, which "
                    "session_measures writes only under p['KEEP_PERBEAT'].")
            seg = np.asarray(pb['seg']).astype(int)
            cols.append(v[seg])
        else:
            raise ValueError(f'unknown grain {grain!r}')
    return np.column_stack(cols)


def subject_slopes(sessions, measure, p, model='session', grain='segment',
                   diag=None):
    """Depth slope map for one subject, pooling that subject's sessions.

    sessions : list of session_measures() dicts for this subject
    measure  : 'hep' | 'hrsp' | 'hepc'
    model    : one of ALL_MODELS; 'session' is the primary model
    grain    : 'segment' (default) | 'cell' | 'perbeat'
    diag     : optional dict, filled in place with fit diagnostics:
               'n_bl_cols'     baseline columns built (0, 1, or nFreq)
               'n_bl_dead'     of those, how many were dead and were dropped
               'n_cov_dropped' degenerate model-covariate columns dropped
               An out-parameter rather than a fourth return value, so that the
               many callers that unpack a 3-tuple are untouched.

    At the segment grain a unit is a depth-report segment.

    The session (day) effect is a design column here, unlike the FMT arm:
    `session` adds centred blocking dummies via cell_ols's sesD. It is not
    removed by z-scoring within session (the FMT convention, used there
    because that arm has no session term): z-scoring rescales every unit by
    that session's SD, changing the weighting of sessions against each other,
    whereas a blocking dummy only removes the mean. STANDARDIZE_WITHIN_SESSION
    therefore defaults to False.

    'unadj' is [intercept, depth] only, with no session dummies.

    The DV is trimmed to TLIMS before the fit, and the baseline is not. The fit
    sees [-100 600] ms while the baseline regressor is computed from the full
    epoch. The trim sets the multiple-comparison space the L2 corrects over,
    and keeping the baseline on the full axis is what lets
    BASELINE_WIN = [-150 -50] straddle the trim edge. Under L1_METHOD='WLS' the
    trim also moves the slopes, because the Huber weight of a unit is computed
    from its residual norm across the fitted columns (l1_slopes.cell_ols).

    HEPC carries a per-unit `ok` mask, nbeat >= MIN_BEATS_UNIT; HEP and HRSP
    use every unit. `n_units` comes back per measure for the same reason: the
    L2's precision_nseg scheme weights by it, so returning the HEP count for
    HEPC would weight a subject by units its HEPC map never saw.

    Returns (beta_map, svar_map, n_units) with beta_map shaped like one unit's
    measure trimmed to TLIMS, or (None, None, 0) if the subject cannot be fit.
    """
    if model not in ALL_MODELS:
        raise ValueError(
            f'model must be one of {ALL_MODELS}, got {model!r}. A typo is '
            'refused by name rather than silently treated as unadjusted.')
    if grain not in ('segment', 'cell', 'perbeat'):
        raise ValueError(
            f"grain must be 'segment', 'cell' or 'perbeat', got {grain!r}. A "
            "typo is refused by name rather than silently treated as 'segment'.")
    # minSeg is a gate on segments, at every grain. Counting units instead
    # would, at the cell grain, test ~5-10 cells against a floor written for
    # ~30 segments.
    n_seg_total = int(sum(len(np.asarray(s['depth'])) for s in sessions))
    # Standardise in the natural shape, then flatten, and build the baseline
    # regressor from the same standardised array, so that the DV and its own
    # baseline column are on the same scale. The z-score is per column
    # (axis 0 = units), so doing it before or after the reshape is identical;
    # only the baseline consistency depends on the order.
    Y, dep, Zstd, ses_id, nbeat, seg_id = [], [], [], [], [], []
    covrows = []          # per-session covariate columns, same row order as Y
    seg_base = 0            # so segment 0 of session 2 is not segment 0 of session 1
    for s in sessions:
        # The only thing the grain changes is what a unit is. Everything after
        # this loop (the session dummies, the Alday baseline column, the TLIMS
        # trim, the per-frequency fit) is identical across grains and is shared,
        # not copied.
        n_seg_here = int(np.asarray(s['depth']).size)
        if grain == 'cell':
            v, d_, nb_ = _cells_of_session(s, measure, p)
            # The CRVE cluster is only defined at the per-beat grain. At the
            # cell and segment grains a unit is its own cluster, so clustering
            # would be the identity and `sg_` stays None.
            sg_ = None
        elif grain == 'perbeat':
            v, d_, sg_ = _beats_of_session(s, measure, p)
            sg_ = sg_ + seg_base
            nb_ = np.ones(v.shape[0], np.int64)
        else:
            v = np.asarray(s[measure])
            d_ = np.asarray(s['depth'])
            # `n_beats` is per RETAINED segment, same order as the measure rows.
            nb_ = np.asarray(s['n_beats']).astype(np.int64)
            sg_ = None
        seg_base += n_seg_here
        if v.size == 0:
            continue
        if p['STANDARDIZE_WITHIN_SESSION']:
            mu = np.nanmean(v, axis=0, keepdims=True)
            sd = np.nanstd(v, axis=0, keepdims=True)
            v = (v - mu) / (sd + EPS)
        Zstd.append(v)
        Y.append(v.reshape(v.shape[0], -1).astype(float)); dep.append(d_)
        ses_id.append(np.full(v.shape[0], int(s['session'])))
        nbeat.append(nb_)
        # Appended inside the same guard as Y: the `if v.size == 0: continue`
        # above skips a session that contributed no unit, so building the
        # covariate anywhere else would leave it one session longer than the DV
        # and shift every value by one session's worth of rows.
        if (model in COVARIATE_MODELS or model in INTERACTION_MODELS
                or model in SESSION_COV_MODELS):
            # A moderation model takes its BASE covariate's columns here; the
            # depth*X product is formed after pooling, where the pooled depth
            # vector exists.
            cvs = _covariate_columns(s, _base_of(model), grain, p)
            if cvs.shape[0] != v.shape[0]:
                raise ValueError(
                    f"{s.get('subject')} ses{s.get('session')}: covariate "
                    f'{model!r} produced {cvs.shape[0]} row(s) at grain '
                    f'{grain!r} but the measure has {v.shape[0]}.')
            covrows.append(cvs)
        seg_id.append(np.full(v.shape[0], -1, np.int64) if sg_ is None
                      else np.asarray(sg_, np.int64))
    if not Y:
        return None, None, 0
    Y3 = np.concatenate(Zstd, axis=0).astype(float)   # [nUnits x ...] natural shape
    dep = np.concatenate(dep).astype(float)
    ses_id = np.concatenate(ses_id)
    nbeat = np.concatenate(nbeat)
    seg_id = np.concatenate(seg_id) if seg_id else None
    # ── The covariate column(s), pooled over the subject's sessions ─────────
    # Centre, then zero the non-finite, in that order, over the pooled vector.
    # Zeroing before centring would drag the mean toward 0 by however many
    # units were missing, which is a different regressor.
    #
    # Degeneracy is dropped, not refused: a covariate column that cannot inform
    # the fit is removed, so for that subject the model becomes unadjusted
    # rather than failing. reportmethod and cond need >1 distinct finite value
    # (they are indicators, and a constant column is rank-deficient); ecgAmp and
    # ecgAmpT need at least one finite value; rr, respAmp and respPhase are
    # taken as given. A subject counted under 'cond' may therefore have been fit
    # without the cond column, so 'survived cond' is not guaranteed per subject;
    # `n_cov_dropped` records how many columns were dropped.
    cov_cols, n_cov_dropped = None, 0
    # The interaction and session+ models come through here too.
    if (model in COVARIATE_MODELS or model in INTERACTION_MODELS
            or model in SESSION_COV_MODELS) and covrows:
        cov_cols = np.concatenate(covrows, axis=0).astype(float)
        fin = np.isfinite(cov_cols)
        # Degeneracy is a property of the BASE covariate, so an interaction
        # model is tested on the same rule as the covariate it rests on.
        if _base_of(model) in ('reportmethod', 'cond'):
            keep_col = np.array([np.unique(cov_cols[fin[:, j], j]).size > 1
                                 for j in range(cov_cols.shape[1])])
        elif _base_of(model) in ('ecgAmp', 'ecgAmpT'):
            keep_col = fin.any(axis=0)
        else:
            keep_col = np.ones(cov_cols.shape[1], bool)
        if not keep_col.all():
            n_cov_dropped = int((~keep_col).sum())
            cov_cols = cov_cols[:, keep_col]
        if cov_cols.shape[1] == 0:
            cov_cols = None
        else:
            mu = np.nanmean(cov_cols, axis=0, keepdims=True)
            cov_cols = cov_cols - mu
            cov_cols[~np.isfinite(cov_cols)] = 0.0
    # ── MODERATION: append the depth x covariate product ───────────────────
    # target_col is the index of the coefficient cell_ols must return. The
    # design it builds is [intercept, depth, sesD..., covC...] and a moderation
    # model has no session dummies, so with k covariate columns the product
    # columns start at 2 + k. The Alday baseline is appended AFTER these by
    # _stack_cov, so it never shifts the index.
    target_col = 1                       # 1 = the depth slope, every other model
    if model in INTERACTION_MODELS:
        if cov_cols is None:
            # Refuse, do not degrade to unadj. A covariate model whose column
            # was dropped still answers its own question (nothing to adjust for,
            # so the unadjusted slope is the adjusted slope). A moderation model
            # does not: with no covariate there is no interaction, and returning
            # the plain depth slope under a name like 'rrXdepth' would put a
            # main effect into a moderation map where every reader would take it
            # for evidence of moderation.
            return None, None, 0
        k = cov_cols.shape[1]
        dep_ctr = dep - np.nanmean(dep)
        inter = cov_cols * dep_ctr[:, None]
        inter = inter - np.nanmean(inter, axis=0, keepdims=True)
        inter[~np.isfinite(inter)] = 0.0
        cov_cols = np.hstack([cov_cols, inter])
        target_col = 2 + k
        # A product column with no variance carries no moderation to estimate,
        # and would make the design singular. Say so rather than return a
        # pinv-split coefficient that looks like an answer.
        if not np.any(np.std(inter, axis=0) > 0):
            return None, None, 0
    # minSeg is the subject gate on all units, not per measure, so it is tested
    # before the HEPC beat mask below.
    if n_seg_total < p['minSeg'] or len(set(dep.tolist())) < 2:
        return None, None, 0

    # ── The per-measure unit mask ───────────────────────────────────────────
    if measure.upper() == 'HEPC':
        ok = nbeat >= int(p['MIN_BEATS_UNIT'])
    else:
        ok = np.ones(Y3.shape[0], bool)
    # cell_ols refuses <3 units or <2 distinct depths anyway, but it does so by
    # returning an all-NaN map, which reads downstream as "fit, every cell
    # refused". Say no here instead, so the caller can print a reason.
    if int(ok.sum()) < 3 or len(set(dep[ok].tolist())) < 2:
        return None, None, 0

    # ── SESSION BLOCKING DUMMIES (PRIMARY model only) ───────────────────────
    # k-1 CENTRED indicator columns for k sessions; cell_ols expects them
    # already centred. With one session there is nothing to block on and sesD
    # stays None, which makes 'session' and 'unadj' identical for that subject —
    # correct, and worth knowing when reading a single-session subject's maps.
    sesD = None
    if model == 'session' or model in SESSION_COV_MODELS:
        levels = np.unique(ses_id)
        if levels.size >= 2:
            D = np.stack([(ses_id == lv).astype(float) for lv in levels[1:]], axis=1)
            sesD = D - D.mean(axis=0, keepdims=True)

    # ── ALDAY (2019) GLM BASELINE ───────────────────────────────────────────
    # The baseline enters as a regressor column, never as a subtraction
    # (subtract / dB / percent assume the baseline instead of estimating it,
    # and the two ratio transforms are undefined on a signed HEP voltage).
    #
    # The df trap: the baseline column costs one degree of freedom that HEPC
    # never spends (HEPC takes no baseline: PPC is computed over unit-length
    # phasors, so there is no scale to adjust), so at the margin a subject can
    # be fit for HEPC and refused for HEP/HRSP in the same run. See
    # glm_baseline's docstring. Do not "fix" it by dropping the column for HEP.
    covC = None
    if p['DO_BASELINE_CORRECT'] and measure.upper() != 'HEPC':
        bl = baseline_mask(sessions[0]['times'], tuple(p['BASELINE_WIN']))
        # real_is_power=True for HRSP is a declaration: session_measures stores
        # hrsp as mean(|CWT|^2) over beats, already power. glm_baseline refuses
        # a bare real HRSP array because power and amplitude give different
        # baselines and it cannot tell them apart. Taking the baseline after
        # averaging over beats is exact: both steps are means, so they commute.
        blocks = [glm_baseline_regressors(v, bl, measure=measure.upper(),
                                          strict=False, real_is_power=True)
                  for v in Zstd]
        covC = np.concatenate(blocks, axis=0) if blocks else None

    # A dead baseline column is dropped (not the live one). The baseline's one
    # df is a deliberate cost for a column that estimates something. A column
    # that is entirely zero after centring estimates nothing: it is collinear
    # with the intercept, the ridge gives it beta = 0, and it still shrinks the
    # residual df, inflating every SE and deflating every t in that map. Two
    # causes: an all-non-finite baseline window, and a baseline that is finite
    # but identical across units.
    #
    # glm_baseline_regressors is called with strict=False, so nothing upstream
    # refuses such a column. Dropping it changes no coefficient (a zero column
    # contributes nothing to beta and nothing to trace(X'X), so the ridge is
    # identical too); it only stops the fit paying a df for it.
    bl_dead = set()
    if covC is not None:
        bl_dead = {int(f) for f in dead_columns(covC)}
    if diag is not None:
        diag['n_bl_cols'] = 0 if covC is None else int(covC.shape[1])
        diag['n_bl_dead'] = len(bl_dead)
        diag['n_cov_dropped'] = int(n_cov_dropped)

    # ── THE FIT ─────────────────────────────────────────────────────────────
    # A time x frequency map is fit one frequency at a time, each with only
    # that frequency's baseline column: cell_ols(Y[:, f, :], dep, sesD,
    # [covC, bl[:, f]]). Handing the solver all nFreq baseline columns at once
    # is a different model: it costs nFreq df instead of 1, and at this
    # cohort's unit counts it refuses the fit outright (17 units against
    # 2 + 27 baseline columns gives nUnits < nP and an all-NaN HRSP map).
    #
    # HEP is 2-D so it takes one baseline column and one fit; HEPC takes none.
    # ── THE TLIMS TRIM: after the baseline columns, before the fit ─────────
    # covC above was built from the full epoch. Trimming first would shorten
    # BASELINE_WIN [-150 -50] to [-100 -50] and drop its cells into the tested
    # map.
    t_keep = tlims_mask(sessions[0]['times'], p['TLIMS'])
    Y3 = Y3[:, t_keep] if Y3.ndim == 2 else Y3[:, :, t_keep]

    dep_c = dep - np.nanmean(dep)
    # Cluster-robust variance at the per-beat grain only, and it is not
    # optional there. Beats inside one depth-report segment share that
    # segment's state, artefacts and depth label, so an OLS variance over ~1200
    # correlated beats understates the SE by roughly sqrt(beats per segment).
    # At the cell and segment grains a unit is its own cluster, so clustering
    # would be the identity and `clus` stays None.
    clus = seg_id if grain == 'perbeat' else None
    m3 = Y3.ndim == 3
    if not m3:
        bl_col = None if (covC is None or 0 in bl_dead) else covC[:, [0]]
        bl_col = _stack_cov(cov_cols, bl_col)
        beta, svar, _B, _fit = cell_ols(Y3, dep_c, sesD=sesD, covC=bl_col,
                                        ok=ok, method=p['L1_METHOD'],
                                        cluster=clus, target_col=target_col)
        beta = np.asarray(beta).ravel(); svar = np.asarray(svar).ravel()
    else:
        nU, nFq, nTm = Y3.shape
        beta = np.full((nFq, nTm), np.nan); svar = np.full((nFq, nTm), np.nan)
        for f in range(nFq):
            bl_col = None if (covC is None or f in bl_dead) else covC[:, [f]]
            bl_col = _stack_cov(cov_cols, bl_col)
            bf, vf, _B, _fit = cell_ols(Y3[:, f, :], dep_c, sesD=sesD, covC=bl_col,
                                        ok=ok, method=p['L1_METHOD'],
                                        cluster=clus, target_col=target_col)
            beta[f] = np.asarray(bf).ravel(); svar[f] = np.asarray(vf).ravel()

    # `n_units` is the segment count at every grain, not the unit count. It
    # feeds the L2's `precision_nseg` weight, which is the number of distinct
    # depth-report segments a subject contributed, at every grain. Returning
    # cells would weight by a different quantity under the same scheme name,
    # and returning beats would weight by something ~60x larger and far more
    # variable, so a subject with a fast heart rate would dominate. It stays
    # per measure: the HEPC unit mask can refuse units the HEP map keeps.
    if grain == 'segment':
        n_seg_used = int(ok.sum())                     # a unit IS a segment
    elif grain == 'perbeat':
        n_seg_used = int(np.unique(seg_id[ok]).size)
    else:                                              # cell
        # A cell is exactly a set of whole segments, so the segments backing the
        # KEPT cells are the segments of those (session, depth) pairs.
        keep = {(int(a), int(b)) for a, b in zip(ses_id[ok], dep[ok])}
        n_seg_used = 0
        for s in sessions:
            sv = int(s['session'])
            d_ = np.asarray(s['depth']).astype(int)
            for dv in np.unique(d_):
                if (sv, int(dv)) in keep:
                    n_seg_used += int((d_ == dv).sum())

    shape = Y3.shape[1:]
    return (np.asarray(beta).reshape(shape),
            np.asarray(svar).reshape(shape),
            n_seg_used)
