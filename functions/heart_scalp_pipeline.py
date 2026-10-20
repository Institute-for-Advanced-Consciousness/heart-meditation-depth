# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Mass-univariate scalp analysis of the heartbeat depth effect.

  segment files (data/segments, config.DATA_DIR)
      -> per-channel HEP / HRSP / HEPC (no ICA, no selection)
      -> per-subject, per-channel L1 depth slopes (segment grain)
      -> 3-D spatio-temporal cluster permutation (chan x freq x time,
         MNE channel adjacency, one FWER family over the whole map)

HEPC, as the code names it, is the paper's HRPC (heartbeat-related phase consistency).

The sensor-space counterpart of heart_depth_pipeline (the selected-IC route). Instead of
selecting one ICA component per subject, it analyses every scalp electrode at once and
corrects across the full nCh x freq x time space with one spatio-temporal cluster
permutation test: no selection, no ICA, no anatomical prior.

What it leaves out relative to the selected-IC analysis:
  * IC selection, IC clustering, anatomical ROI and equivalent-dipole fitting. The only
    channel-level operation is removing ECG/RESP so that every remaining axis is a scalp
    electrode, and that happens upstream (the .npz stores only scalp axes).
  * The covariate models. Only 'session' and the depth-only 'unadj' are fitted. The
    state band powers (alpha/beta/theta) and the beat-level covariates (rr, respAmp,
    respPhase, ecgAmp, ecgAmpT) are computed per selected IC in the main analysis and
    would have to be refit per channel here.

Cardiac field artifact. A cluster whose time support overlaps CFA_WINDOW_MS = [-150, +150]
ms sits on the QRS complex and is the cardiac field until shown otherwise; the literature
window is 200-600 ms precisely to clear the QRS and T wave. A negative Start_ms means the
cluster begins before the trigger and overlaps the baseline window, so the baseline was
fit on the effect. A 15-25 Hz peak is QRS spectral content. Every overlapping cluster is
flagged in CFA_flag; such a row reads "heartbeat-locked at peri-QRS latency, origin
undetermined", not "depth-linked brain activity". Only latency discriminates (see what
survives away from the QRS, in the `lit` window). Surviving the ecgAmp covariate cannot
rescue a peri-QRS cluster: ecgAmp adjusts for how large the cardiac field is, never for
its latency.

Memory: the per-session cost is the full 64-channel CWT over the retained segments, about
1.1 GB complex64 per session at EPOCH_DUR = 60. The permutation null is the other large
array, 64 ch x 27 f x 176 t x nPerm. Unlike the IC route, whose [nCell x nPerm] null fits
in RAM, the scalp null t-maps are computed in chunks of cells (see cluster_perm_3d).
"""
from __future__ import annotations

import csv
import os

import numpy as np

# The selected-IC pipeline's core is reused unchanged: window arithmetic, R-peak
# detection, RR cleaning and IBI gating (channel-invariant: the train is built once from
# seg_ecg), ECG-quality reporting, and every cohort gate that does not touch the IC.
# Only what the measures are computed on changes: every scalp channel instead of
# `w_ic @ eeg` for one selected IC.
from heart_depth_pipeline import (DEFAULT_PARAMS, window_indices,
                                  tlims_mask)
from heart_core import morlet_cwt, ppc_map, ecg_polarity, detect_rpeaks as _detect_rpeaks_raw
from get_rr import get_rr, resolve_polarity, GetRRError
from rr_cleaning import clean_rr, qrs_bandpass
from ecg_quality import (clean_ratio_segments, ibi_short_prop,
                             subject_ecg_clean_ratio, ecg_ratio_passes_gate)
from l1_slopes import cell_ols

EPS = np.finfo(float).eps
SEED = 20260728
# Imported so that the CFA flag means exactly what it means on the IC route.
from l2_run import CFA_WINDOW_MS   # noqa: E402

# Largest [nCell_chunk x nPerm] float64 block the null matmul may hold at once. The IC
# route caps the whole map at 4 GiB (`_check_memory`); the scalp map is too big to hold
# whole in float64, so the null t-maps are computed chunk by chunk over cells. At the
# default nPerm = 500, 0.5 GiB is ~134k cells per chunk, three chunks for the full
# 64 x 27 x 176 map, few enough that the loop cost is invisible against the matmul.
MAX_NULL_CHUNK_GB = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_SCALP_PARAMS = dict(DEFAULT_PARAMS)
# What the run is called. It selects nothing (every channel is analysed), but the tag
# uses the same folder machinery as the IC methods, so it takes a value there.
DEFAULT_SCALP_PARAMS['IC_SELECT_METHOD'] = 'scalp'
# The scalp analysis fits the same two models as the IC route: 'session' (the primary
# model, centred blocking dummies) and 'unadj' ([intercept, depth] only). No covariate
# models.
DEFAULT_SCALP_PARAMS['MODELS'] = ('session', 'unadj')
DEFAULT_SCALP_PARAMS['PRIMARY_MODEL'] = 'session'
DEFAULT_SCALP_PARAMS['nPerm'] = 500       # memory-friendly; 10000 needs far more RAM
DEFAULT_SCALP_PARAMS['ALPHA'] = 0.05

# A speed setting only; it must never change an answer. The per-session L1 is
# embarrassingly parallel (`session_measures_scalp` takes a path and the params and
# shares no state), and it is where this analysis spends its time: the 64-channel
# Morlet CWT takes ~16 s per session, ~20 min over the cohort, against ~40 s for the
# entire L1 solve and seconds for the permutation null. Everything is NumPy, so more
# CPU helps and a GPU does not.
#
# Default 1: each worker holds the full ~1.1 GiB complex64 CWT for its own session, so
# N workers add N x 1.1 GiB of transient peak memory on top of the retained cohort.
#
# Results are collected in input order, so the cohort is assembled identically at any
# N_JOBS. Run both ways, every array of run_results.npz came out bit-identical, NaN
# positions included (on 3 subjects / 6 sessions: 0.000e+00 over 38 float arrays, 191 s
# at 1 worker, 91 s at 8).
# N_JOBS is in `_COLLISION_IGNORE` for the same reason: re-running one analysis at a
# different worker count is a repeat, not a collision.
#
# Windows spawns rather than forks, so a caller using N_JOBS > 1 must be importable:
# run it from a .py file or behind `if __name__ == '__main__'`. Driving it from
# `python -` or a REPL raises BrokenProcessPool, because the worker cannot re-import
# `<stdin>` as __main__.
DEFAULT_SCALP_PARAMS['N_JOBS'] = 1


def _session_job(args):
    """One session, for the worker pool -> (sid, ses, ms, notes).

    Module-level and picklable on purpose: Windows spawns rather than forks, so a
    closure or a lambda here would fail to pickle at submit time.

    Exceptions are not caught. `Executor.map` re-raises them on iteration, as the
    sequential path does; catching them here would let the parallel path drop sessions
    the sequential one refuses, and the two would stop being the same run.

    `notes` is returned, not mutated in place: across a process boundary the caller's
    dict would not see the mutation, so the drop reason travels in the return value.
    """
    sid, ses, npz_path, p = args
    notes = {}
    ms = session_measures_scalp(npz_path, p, verbose=False, notes=notes)
    return sid, ses, ms, notes


# One row of cohort_sessions.csv, and one row of cohort_subjects.csv.
SCALP_SESSION_FIELDS = ('subject', 'session', 'dur_min_lowerbound',
                        'n_seg_stored', 'n_seg_cond', 'n_seg_window_ok',
                        'n_window_overlaps', 'n_seg_used', 'n_beats_detected',
                        'n_beats_analysed', 'n_getrr_failed_windows',
                        'ecg_clean_ratio', 'ibi_short_prop', 'status', 'reason')
SCALP_SUBJECT_FIELDS = ('subject', 'n_sessions_found', 'n_sessions_kept',
                        'n_segments', 'n_depths_pooled', 'ecg_clean_ratio',
                        'status', 'reason')


def _scalp_session_record(sid, ses, ms, notes):
    """(sid, ses, session_measures_scalp output, its notes) -> one CSV row."""
    rec = {k: notes.get(k, '') for k in SCALP_SESSION_FIELDS}
    rec.update(subject=str(sid), session=int(ses),
               n_seg_used=0 if ms is None else int(len(ms['depth'])),
               n_getrr_failed_windows=len(notes.get('getrr_failed_windows', [])),
               ecg_clean_ratio=float(notes.get('ecg_clean_ratio', np.nan)),
               ibi_short_prop=float(notes.get('ibi_short_prop', np.nan)),
               status='dropped' if ms is None else 'kept',
               reason=notes.get('skip_reason', '') if ms is None else '')
    return rec


# ═══════════════════════════════════════════════════════════════════════════
# PER-SESSION PROCESSING: every scalp channel, not one selected IC
# ═══════════════════════════════════════════════════════════════════════════
def scalp_session_beats(npz_path, p, verbose=True, notes=None):
    """One session -> every session gate and the channel-invariant beat train.

    The front half of `session_measures_scalp`.

    Returns None when a gate drops the session (the reason is in
    notes['skip_reason']), else a dict holding the loaded npz `d`, the ids,
    `idx_keep` (retained segment indices), `seg_peaks_abs` (one array of
    absolute beat samples per retained segment, after RR cleaning and
    MIN_IBI_MS), the analysis-window offsets `w0`/`w1`/`n`, the epoch offsets
    `lo`/`hi` in samples, and the `times`/`freqs` axes.
    """
    if notes is None:
        notes = {}
    notes.setdefault('skip_reason', '')
    d = np.load(npz_path, allow_pickle=True)
    sid = str(d['subject_id']); ses = int(d['session']); fs = float(d['fs'])
    tag = f'{sid} ses{ses}'

    seg_eeg = d['seg_eeg']                       # [nSeg x nChan x nPts]
    seg_ecg = d['seg_ecg']                       # [nSeg x nPts]
    depth = np.asarray(d['seg_depth']).astype(int)
    cond = np.asarray(d['seg_cond']).astype(int)
    seg_len = np.asarray(d['seg_len']).astype(int)
    ch_names = [str(c) for c in np.asarray(d['ch_names'])]

    nC = int(seg_eeg.shape[1])
    notes['n_chan_stored'] = nC
    if nC != len(ch_names):
        raise ValueError(
            f'{tag}: seg_eeg has {nC} channels but ch_names has '
            f'{len(ch_names)} — cannot name the scalp axes. Refuse rather than '
            f'analyse channels anonymously.')
    # FCz/AFz were the recording reference and are never restored; every stored
    # axis is a scalp electrode, so no per-session channel removal is needed.
    notes['n_chan_scalp'] = nC
    notes['n_seg_stored'] = int(len(depth))

    # ── GATE 0: MIN_SESSION_MIN, on the lower-bound duration proxy ──────────
    # The same proxy and the same order as heart_depth_run._minimal_session_npz:
    # before any other gate.
    from session_length import session_duration_lowerbound_min
    dur = session_duration_lowerbound_min(d['block_spans'], d['seg_span'], fs)
    notes['dur_min_lowerbound'] = round(float(dur), 1)
    if p.get('MIN_SESSION_MIN') and np.isfinite(dur) \
            and dur < float(p['MIN_SESSION_MIN']):
        notes['skip_reason'] = (f'MIN_SESSION_MIN: duration lower bound '
                                f'{dur:.1f} min < {float(p["MIN_SESSION_MIN"]):g} min')
        if verbose:
            print(f'  {tag}: {notes["skip_reason"]} — session skipped')
        return None

    # ── GATE 1: condition ────────────────────────────────────────────────────
    # (0, 1) under INCLUDE_SAPO, not every condition. The pre-/post-meditation
    # reports (cond 2, 3) lie outside every meditation block, so their beats never
    # reach a map, but they would count towards MIN_DEPTHS and their detected beats
    # would inflate ecg_clean_ratio. The IC route gates (0, 1) the same way
    # (heart_depth_run._minimal_session_npz).
    keep = np.isin(cond, (0, 1) if p['INCLUDE_SAPO'] else (0,))
    keep &= np.isin(depth, p['DEPTH_LEVELS'])
    notes['n_seg_cond'] = int(keep.sum())
    if keep.sum() == 0:
        notes['skip_reason'] = 'no segment in the analysed condition/depth set'
        if verbose:
            print(f'  {tag}: no segments in DEPTH_LEVELS — skipped')
        return None

    # ── GATE 1b: the requested window is actually stored for this segment ────
    w0, w1, w_ok = window_indices(seg_len, fs, p['EPOCH_DUR'], p['PRE_END'],
                                  p['EXTRACT_EPOCH_DUR'], p['EXTRACT_PRE_END'])
    notes['n_seg_window_refused'] = int((keep & ~w_ok).sum())
    if notes['n_seg_window_refused']:
        keep &= w_ok
    notes['n_seg_window_ok'] = int(keep.sum())
    _won = str(p.get('WAVELET_ON', 'window'))
    if _won not in ('window', 'segment'):
        raise ValueError(f"WAVELET_ON must be 'window' or 'segment', got {_won!r}")
    _cwt_seg = _won == 'segment'

    # ── GATE 1c: resolve overlaps at the ANALYSIS duration ──────────────────
    ov_key = 'overlap_resolved' in d.files
    already = bool(d['overlap_resolved']) if ov_key else True
    notes['overlap_resolved_at_extraction'] = already
    n_ov = 0
    if not already and keep.sum() > 1:
        span_all = np.asarray(d['seg_span'])
        kept_ix = np.where(keep)[0]
        kept_ix = kept_ix[np.argsort(span_all[kept_ix, 0])]
        starts = span_all[kept_ix, 0] + w0
        ends = span_all[kept_ix, 0] + w1
        drop = []
        for a_ in range(len(kept_ix) - 1):
            if ends[a_] > starts[a_ + 1]:
                drop.append(kept_ix[a_]); n_ov += 1
        if drop:
            keep[np.asarray(drop)] = False
    notes['n_window_overlaps'] = n_ov
    if keep.sum() == 0:
        notes['skip_reason'] = 'no segment holds the window'
        if verbose:
            print(f'  {tag}: no segment holds the window — skipped')
        return None

    # ── GATE 2: MIN_DEPTHS, per session ──────────────────────────────────────
    if len(set(depth[keep].tolist())) < p['MIN_DEPTHS']:
        notes['skip_reason'] = (
            f'MIN_DEPTHS: {len(set(depth[keep].tolist()))} distinct depth(s) '
            f'< {int(p["MIN_DEPTHS"])}')
        if verbose:
            print(f'  {tag}: {notes["skip_reason"]} — session skipped')
        return None

    idx_keep = np.where(keep)[0]
    freqs = np.asarray(p['HRSP_FREQS'], float)
    win = p['HEP_WIN']
    nT = int(round((win[1] - win[0]) / 1000.0 * fs)) + 1
    times = win[0] + np.arange(nT) / fs * 1000.0

    # ── THE BEAT TRAIN: once per session, on the ECG (channel-invariant) ────

    lo = int(round(win[0] / 1000.0 * fs)); hi = int(round(win[1] / 1000.0 * fs))
    seg_abs = np.asarray(d['seg_span'])
    n = w1 - w0

    # PEAK_SOURCE is honoured as on the IC route (heart_depth_pipeline.session_measures,
    # polarity + stage A). Polarity is resolved once per session by the detector's own
    # front end, then forced to +1 per window; a window get_rr refuses yields an empty
    # train for the beat floors to drop, never a lost session.
    pol = 1
    _cat = np.concatenate([np.asarray(seg_ecg[k, :int(seg_len[k])], float)
                           for k in idx_keep[:6]])
    if p['PEAK_SOURCE'] == 'brainbeats':
        pol = resolve_polarity(_cat, fs)
    else:
        pol = ecg_polarity(_cat, fs)
    if verbose:
        print(f'  {tag}: ECG polarity '
              f'{"upright" if pol > 0 else "INVERTED"}')

    seg_peaks_abs, seg_peaks_amp = [], []
    for k in idx_keep:
        a = int(seg_abs[k][0]) + w0
        _win = pol * np.asarray(seg_ecg[k, w0:w1], float)
        if p['PEAK_SOURCE'] == 'brainbeats':
            try:
                pk, _ = get_rr(_win, fs, polarity=1, drop_first=False)
            except GetRRError as exc:
                pk = np.array([], np.int64)
                notes.setdefault('getrr_failed_windows', []).append(
                    f'seg {int(k)}: {exc}')
        else:
            pk = _detect_rpeaks_raw(_win, fs)
        seg_peaks_abs.append(pk.astype(np.int64) + a)
        _bp = qrs_bandpass(_win, fs)
        seg_peaks_amp.append(_bp[pk.astype(np.int64)]
                             if pk.size else np.array([], float))

    raw_abs_all = (np.concatenate(seg_peaks_abs) if seg_peaks_abs
                   else np.array([], np.int64))
    win_spans = np.array([[int(seg_abs[k][0]) + w0,
                           int(seg_abs[k][0]) + w1 - 1] for k in idx_keep],
                         dtype=np.int64)
    notes['n_beats_detected'] = int(raw_abs_all.size)

    # ── RR CLEANING, POOLED PER MEDITATION BLOCK (as on the IC route) ──────
    n_removed = 0
    if p['DO_CLEAN_RR']:
        blocks = np.atleast_2d(np.asarray(d['block_spans'])).astype(np.int64)
        all_abs = (np.concatenate(seg_peaks_abs) if seg_peaks_abs
                   else np.array([], np.int64))
        all_amp = (np.concatenate(seg_peaks_amp) if seg_peaks_amp
                   else np.array([], float))
        kept = np.zeros(all_abs.size, bool)
        for b0, b1 in blocks:
            m = (all_abs >= b0) & (all_abs < b1)
            if m.sum() < 3:
                kept |= m
                continue
            _o = np.argsort(all_abs[m])
            pk_b = all_abs[m][_o]
            amp_b = all_amp[m][_o] if all_amp.size == all_abs.size else None
            rr_b = np.diff(pk_b) / fs                     # seconds, as clean_rr expects
            try:
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
                kept |= np.isin(all_abs, keep_abs) | (all_abs == pk_b[0])
            except Exception as exc:
                print(f'  {tag}: clean_rr failed on block [{b0},{b1}) ({exc}) '
                      f'— beats KEPT UNCLEANED for this block')
                kept |= m
        in_block = np.zeros(all_abs.size, bool)
        for b0, b1 in blocks:
            in_block |= (all_abs >= b0) & (all_abs < b1)
        n_out = int((~in_block).sum())
        if n_out and verbose:
            print(f'  {tag}: {n_out} beat(s) outside every meditation block '
                  f'— dropped (outside the R-peak scope)')
        n_removed = int(in_block.sum() - (kept & in_block).sum())
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

    # QA train captured before MIN_IBI (see heart_depth_pipeline.session_measures).
    # The two subject-QA metrics read RPEAKS_STORE_RAW, the post-clean_rr, pre-MIN_IBI
    # train; measuring the short-IBI fraction after the short-IBI filter would make it
    # ~0 by construction.
    qa_abs_all = clean_abs_all.copy()

    # ── MIN_IBI_MS, beat-level, on the clean train (as on the IC route) ─────
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
            clean_abs_all = (np.concatenate(seg_peaks_abs)
                             if seg_peaks_abs else np.array([], np.int64))
    notes['n_beats_min_ibi_dropped'] = n_ibi_drop
    notes['n_beats_after_min_ibi'] = int(clean_abs_all.size)

    try:
        cr = clean_ratio_segments(raw_abs_all, qa_abs_all, win_spans)
        notes['ecg_clean_ratio'] = float(cr.ratio)
        notes['ecg_n_all'] = int(cr.n_all)
        notes['ecg_n_kept'] = int(cr.n_kept)
    except Exception as exc:
        notes['ecg_clean_ratio'] = float('nan')
        notes['ecg_error'] = f'{type(exc).__name__}: {exc}'
    notes['ibi_short_prop'] = float(ibi_short_prop(
        qa_abs_all, fs, min_ibi_ms=float(p.get('MIN_IBI_MS', 600))))

    notes['n_beats_analysed'] = int(clean_abs_all.size)

    return dict(d=d, sid=sid, ses=ses, fs=fs, tag=tag, seg_eeg=seg_eeg,
                seg_ecg=seg_ecg, depth=depth, cond=cond, ch_names=ch_names,
                nC=nC, idx_keep=idx_keep, seg_peaks_abs=seg_peaks_abs,
                seg_abs=seg_abs, w0=w0, w1=w1, n=n, lo=lo, hi=hi,
                times=times, freqs=freqs, nT=nT, seg_len=seg_len, cwt_seg=_cwt_seg)


def session_measures_scalp(npz_path, p, verbose=True, notes=None):
    """One session -> per-segment HEP / HRSP / HEPC for EVERY scalp channel.

    Returns None when the session fails a gate, else a dict with:
        depth    [nSeg]                    depth per retained segment
        cond     [nSeg]                    condition code
        hep      [nSeg x nChan x nT]       float32
        hrsp     [nSeg x nChan x nF x nT]  float32
        hepc     [nSeg x nChan x nF x nT]  float32
        ch_names list[str]                 the nChan scalp electrodes
        times    [nT]                      ms, the FULL HEP_WIN axis
        freqs    [nF]                      Hz
        n_beats  [nSeg]                    epochable beats per retained segment

    The beat train is channel-invariant and is built once. R-peak detection, RR
    cleaning, the MIN_IBI filter and the gates all run on seg_ecg, as on the IC route,
    because a heartbeat is where it is whichever electrode the EEG is read from. Only
    what the train is applied to is per channel. The gates and the train live in
    `scalp_session_beats`.
    """
    if notes is None:
        notes = {}
    ctx = scalp_session_beats(npz_path, p, verbose=verbose, notes=notes)
    if ctx is None:
        return None
    sid, ses, fs, tag = ctx['sid'], ctx['ses'], ctx['fs'], ctx['tag']
    seg_eeg, depth, cond = ctx['seg_eeg'], ctx['depth'], ctx['cond']
    ch_names, nC, idx_keep = ctx['ch_names'], ctx['nC'], ctx['idx_keep']
    seg_peaks_abs, seg_abs = ctx['seg_peaks_abs'], ctx['seg_abs']
    w0, w1, n, lo, hi = ctx['w0'], ctx['w1'], ctx['n'], ctx['lo'], ctx['hi']
    times, freqs, nT = ctx['times'], ctx['freqs'], ctx['nT']
    seg_len, _cwt_seg = ctx['seg_len'], ctx['cwt_seg']

    # ── PER SEGMENT: epochable beats, then one full-channel CWT per segment ──
    nF = len(freqs)
    nS = len(idx_keep)
    hep = np.full((nS, nC, nT), np.nan, dtype=np.float32)
    hrsp = np.full((nS, nC, nF, nT), np.nan, dtype=np.float32)
    hepc = np.full((nS, nC, nF, nT), np.nan, dtype=np.float32)

    seg_ok, seg_nbeat = [], []
    for j, k in enumerate(idx_keep):
        a = int(seg_abs[k][0]) + w0
        pk = seg_peaks_abs[j] - a
        # only beats whose full epoch fits inside the segment
        pk = pk[(pk + lo >= 0) & (pk + hi < n)]
        seg_nbeat.append(int(pk.size))
        if pk.size < int(p.get('MIN_BEATS_UNIT', 1)):
            seg_ok.append(False)
            continue
        seg_ok.append(True)

        # [nChan x n]: every scalp electrode's continuous trace over the window
        eeg = np.asarray(seg_eeg[k, :, w0:w1], float)
        # One morlet_cwt call on the [nChan x n] block returns every channel's
        # transform, with no per-channel re-transform. Under
        # WAVELET_ON='segment' the whole stored segment is transformed and the
        # window cut from it (see heart_depth_pipeline.DEFAULT_PARAMS).
        if _cwt_seg:
            co = morlet_cwt(np.asarray(seg_eeg[k, :, :int(seg_len[k])], float), freqs, fs,
                            width=p['HRSP_WAVE_WIDTH'])[:, :, w0:w1]
        else:
            co = morlet_cwt(eeg, freqs, fs, width=p['HRSP_WAVE_WIDTH'])   # [nChan x nF x n]
        idx = np.stack([np.arange(q + lo, q + hi + 1) for q in pk])  # [nBeats x nT]
        # Vectorized per-channel map assembly: same algebra as the IC route's
        # per-segment loop, with the channel axis batched into the indexing.
        ep = eeg[:, idx]                                   # [nChan x nBeats x nT]
        cb = co[:, :, idx].transpose(0, 2, 1, 3)           # [nChan x nBeats x nF x nT]
        hep[j] = np.nanmean(ep, axis=1).astype(np.float32)  # [nChan x nT]
        pw = np.abs(cb) ** 2
        hrsp[j] = np.nanmean(pw, axis=1).astype(np.float32)  # [nChan x nF x nT]
        z = cb / (np.abs(cb) + EPS)
        hepc[j] = np.stack([ppc_map(z[c_i]) for c_i in range(nC)],
                           axis=0).astype(np.float32)        # [nChan x nF x nT]

    seg_ok = np.asarray(seg_ok)
    notes['n_seg_with_beats'] = int(seg_ok.sum())
    notes['n_seg_hepc_ok'] = int((np.asarray(seg_nbeat)[seg_ok]
                                  >= int(p['MIN_BEATS_UNIT'])).sum())
    if seg_ok.sum() == 0:
        notes['skip_reason'] = (
            f'no segment carried a single epochable beat (MIN_BEATS_UNIT '
            f'{int(p.get("MIN_BEATS_UNIT", 1))})')
        if verbose:
            print(f'  {tag}: no segment carried an epochable beat — skipped')
        return None

    return dict(subject=sid, session=ses, fs=fs, ch_names=ch_names,
                depth=depth[idx_keep][seg_ok], cond=cond[idx_keep][seg_ok],
                hep=hep[seg_ok], hrsp=hrsp[seg_ok], hepc=hepc[seg_ok],
                n_beats=np.asarray(seg_nbeat)[seg_ok],
                times=times, freqs=freqs)


# ═══════════════════════════════════════════════════════════════════════════
# L1: per-subject, per-channel depth slopes
# ═══════════════════════════════════════════════════════════════════════════
def subject_slopes_scalp(sessions, measure, p, model='session',
                         baseline_measure=None):
    """Depth slope map for one subject, at every channel, pooling sessions.

    baseline_measure : which glm_baseline branch builds the baseline column.
        None = `measure` itself (every scalp call). A caller fitting another
        real-valued [nC x nF x nT] map can pass 'HRSP', whose baseline is the
        per-frequency mean over BASELINE_WIN; glm_baseline knows no other
        time-frequency name.

    The same model as heart_depth_pipeline.subject_slopes (centred depth, centred
    session dummies for the primary 'session' model, one GLM baseline column per
    frequency for HEP/HRSP and none for HEPC, the DV trimmed to TLIMS), with the
    channels as the leading map axis. Returns one [nChan x ...] slope map per subject
    instead of one [...] map: (beta, svar, n_units, depth_mean, depth_n, depth_levels).

    The channel axis is folded into the fit the way time already is: cell_ols is
    called per channel for HEP and per (channel, freq) for the time-frequency measures,
    so the per-frequency baseline rule generalises to per-(channel, freq) unchanged and
    still costs exactly 1 df.
    """
    if model not in ('session', 'unadj'):
        raise ValueError(f"model must be 'session' or 'unadj', got {model!r}")

    from glm_baseline import baseline_mask, glm_baseline_regressors

    meas = measure.upper()
    Y, dep, Zstd, ses_id, nbeat = [], [], [], [], []
    for s in sessions:
        v = np.asarray(s[measure])                # [nSeg x nChan x ...]
        if v.size == 0:
            continue
        if p['STANDARDIZE_WITHIN_SESSION']:
            mu = np.nanmean(v, axis=0, keepdims=True)
            sd = np.nanstd(v, axis=0, keepdims=True)
            v = (v - mu) / (sd + EPS)
        Zstd.append(v)
        dep.append(s['depth'])
        ses_id.append(np.full(v.shape[0], int(s['session'])))
        nbeat.append(np.asarray(s['n_beats']).astype(np.int64))
    if not Zstd:
        return None, None, 0, None, None, None
    Y3 = np.concatenate(Zstd, axis=0).astype(float)  # [nUnits x nChan x ...]
    dep = np.concatenate(dep).astype(float)
    ses_id = np.concatenate(ses_id)
    nbeat = np.concatenate(nbeat)
    nC = Y3.shape[1]

    if Y3.shape[0] < p['minSeg'] or len(set(dep.tolist())) < 2:
        return None, None, 0, None, None, None

    if meas == 'HEPC':
        ok = nbeat >= int(p['MIN_BEATS_UNIT'])
    else:
        ok = np.ones(Y3.shape[0], bool)
    if int(ok.sum()) < 3 or len(set(dep[ok].tolist())) < 2:
        return None, None, 0, None, None, None

    # Per-depth means, stored in run_results.npz (depth__<meas>__<model>__<subject>) so
    # the scalp results can be drawn per depth level. Computed on the same standardised
    # units and the same `ok` mask as the L1 fit, so the curves describe the data the
    # slope was estimated from.
    depth_levels = np.arange(1, 6, dtype=float)
    dmean = np.full((len(depth_levels),) + Y3.shape[1:], np.nan)
    dn = np.zeros(len(depth_levels), dtype=np.int64)
    for _i, _L in enumerate(depth_levels):
        _m = ok & (dep == _L)
        if _m.sum() >= 1:
            dmean[_i] = np.nanmean(Y3[_m], axis=0)
            dn[_i] = int(_m.sum())

    sesD = None
    if model == 'session':
        levels = np.unique(ses_id)
        if levels.size >= 2:
            D = np.stack([(ses_id == lv).astype(float) for lv in levels[1:]],
                         axis=1)
            sesD = D - D.mean(axis=0, keepdims=True)

    covC = None
    if p['DO_BASELINE_CORRECT'] and meas != 'HEPC':
        bl = baseline_mask(sessions[0]['times'], tuple(p['BASELINE_WIN']))
        # glm_baseline_regressors validates its DV shape and does not take a channel
        # axis, so the baseline is built once per channel and the per-channel columns
        # are then lifted to the full map. The channel set is the retained segments'
        # shared one, so a single per-channel loop is exact. The per-frequency rule
        # then applies to HRSP at each channel separately: column f of that channel's
        # block feeds only that (channel, f) slice's fit, and the model stays at exactly
        # 1 baseline df per fit.
        nC = Zstd[0].shape[1]
        if meas == 'HEP':
            # Per-channel baseline blocks, each [nU x 1] -> [nU x nC] on the
            # channel axis (concatenate, not stack, so no trailing axis).
            covC = np.concatenate(
                [np.concatenate(
                    [glm_baseline_regressors(v[:, c, :], bl, measure=meas,
                                             strict=False, real_is_power=True)
                     for v in Zstd], axis=0)
                 for c in range(nC)], axis=1)              # [nU x nC]
        else:
            # Per-channel [nU x nF] blocks -> [nU x nC x nF]
            covC = np.stack(
                [np.concatenate(
                    [glm_baseline_regressors(v[:, c, :, :], bl,
                                             measure=baseline_measure or meas,
                                             strict=False, real_is_power=True)
                     for v in Zstd], axis=0)
                 for c in range(nC)], axis=1)              # [nU x nC x nF]

    t_keep = tlims_mask(sessions[0]['times'], p['TLIMS'])
    if meas == 'HEP':
        Y3 = Y3[:, :, t_keep]                        # [nU x nC x nT_fit]
    else:
        Y3 = Y3[:, :, :, t_keep]                     # [nU x nC x nF x nT_fit]

    dep_c = dep - np.nanmean(dep)
    L1 = p['L1_METHOD']

    # ── THE FIT, ONE cell_ols CALL PER (CHANNEL, FREQ) ──────────────────────
    # The baseline enters the design as one column per fit: the (channel, freq) slice
    # is fit against [1, depth, sesD, that slice's baseline], never against every
    # slice's baseline at once, which would cost one df per (channel, freq) instead of
    # one per fit and, at this cohort's unit counts, be under-determined. The channel
    # axis adds an outer loop on top of the IC route's per-frequency one.

    def _bl_col(cc, ff=None):
        """This slice's baseline regressor column, or None.

        HEP  : covC holds one column per channel ([nU x nC]); this channel's fit gets
               that one column, [nU x 1], as on the IC route.
        HRSP : covC holds one column per (channel, freq) ([nU x nC x nF]), and the
               per-frequency rule hands that (channel, freq)'s column to its slice's
               fit. The df cost is the same 1 per fit as on the IC route, and the
               condition at which a fit runs out of df (`nUnits < nP`) is unchanged."""
        if covC is None:
            return None
        if meas == 'HEP':
            return covC[:, [cc]]                        # [nU x 1]
        return covC[:, cc, [ff]]                        # [nU x 1]

    if meas == 'HEP':
        nT_fit = Y3.shape[-1]
        beta = np.full((nC, nT_fit), np.nan)
        svar = np.full((nC, nT_fit), np.nan)
        for c in range(nC):
            bc, vc, _B, _fit = cell_ols(Y3[:, c, :], dep_c, sesD=sesD,
                                        covC=_bl_col(c), ok=ok, method=L1)
            beta[c] = np.asarray(bc).ravel()
            svar[c] = np.asarray(vc).ravel()
        shape = (nC, nT_fit)
    else:
        nF, nT_fit = Y3.shape[2], Y3.shape[3]
        beta = np.full((nC, nF, nT_fit), np.nan)
        svar = np.full((nC, nF, nT_fit), np.nan)
        for c in range(nC):
            for f in range(nF):
                bf, vf, _B, _fit = cell_ols(Y3[:, c, f, :], dep_c, sesD=sesD,
                                            covC=_bl_col(c, f), ok=ok,
                                            method=L1)
                beta[c, f] = np.asarray(bf).ravel()
                svar[c, f] = np.asarray(vf).ravel()
        shape = (nC, nF, nT_fit)

    return (np.asarray(beta).reshape(shape),
            np.asarray(svar).reshape(shape), int(ok.sum()),
            dmean, dn, depth_levels)


# ═══════════════════════════════════════════════════════════════════════════
# L2: 3-D spatio-temporal cluster permutation with channel adjacency
# ═══════════════════════════════════════════════════════════════════════════
def build_channel_adjacency(ch_names):
    """MNE channel adjacency on the standard 10-20/10-10 montage (Delaunay
    triangulation of the 2-D scalp projection).

    The .npz stores labels only, not positions; the montage supplies them.

    Returns (adjacency scipy CSR [nC x nC], names). On any montage failure this raises
    (`on_missing='raise'`), and the cluster test must not run: a missing adjacency
    replaced by all-to-all would merge the whole scalp into one cluster and read as a
    spatial claim it is not.
    """
    import mne
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(list(ch_names), sfreq=250.0, ch_types='eeg')
    info.set_montage(montage, on_missing='raise')
    adj, names = mne.channels.find_ch_adjacency(info, ch_type='eeg')
    return adj, names


def _null_t(X2d, flips):
    """Sign-flip one-sample t-maps, one matmul. X2d [nCell x nSub], flips
    [nSub x P] of +-1. Reduces to the unweighted member of l2_heart's
    `_null_t_weighted` at uniform weights; kept separate so this module needs no
    import of the weighting machinery it does not use (the scalp test has no
    precision weighting).
    """
    n = X2d.shape[1]
    m = (X2d @ flips) / n                      # [nCell x P]
    SS = (X2d ** 2).sum(axis=1, keepdims=True) - n * m ** 2
    var = SS / (n - 1.0)
    np.clip(var, 1e-30, None, out=var)
    return m / np.sqrt(var / n)


def build_cluster_labeller(chan_adj, nF, nT):
    """A `_label(mask_flat) -> [n_cell] int labels, -1 background` closure.

    Separate from cluster_perm_3d so that other code can reproduce its clusters
    exactly: the CSV carries only a cluster's bounding box (Channels / Start_ms /
    Start_Hz), and a cluster is a connected set whose box holds cells the test never
    selected. tfce_scalp.check_against_cluster calls this to re-derive the cluster
    test, and cluster_perm_3d calls it for the test.

    The graph is channel adjacency x 4-connectivity in (freq, time), built once; only
    the mask changes between calls.
    """
    from scipy import sparse as _sp
    # chan_adj is [nC x nC]
    nC = int(chan_adj.shape[0])
    FT = nF * nT
    from scipy.ndimage import generate_binary_structure
    ft_struct = generate_binary_structure(2, 1)
    ft_off = np.argwhere(ft_struct) - 1
    ft_off = ft_off[np.any(ft_off != 0, axis=1)]
    ft_off_flat = ft_off[:, 0] * nT + ft_off[:, 1]
    edges_r, edges_c = [], []
    for o in ft_off_flat:
        src = np.arange(FT, dtype=np.int64); dst = src + o
        ok = (dst >= 0) & (dst < FT)
        for c in range(nC):
            edges_r.append(c * FT + src[ok]); edges_c.append(c * FT + dst[ok])
    coo = _sp.coo_matrix(chan_adj)
    ch_r = coo.row.astype(np.int64); ch_c = coo.col.astype(np.int64)
    for ft in range(FT):
        edges_r.append(ch_r * FT + ft); edges_c.append(ch_c * FT + ft)
    row = np.concatenate(edges_r); col = np.concatenate(edges_c)
    keep0 = np.ones(len(row), bool)
    # `_label` keeps only edges with both endpoints supra-threshold before calling
    # connected_components; otherwise the graph would span the whole map and return
    # one huge component per null draw.
    def _label(mask_flat, _row=row, _col=col):
        keep = mask_flat[_row] & mask_flat[_col]
        rr = _row[keep]; cc = _col[keep]
        if not rr.size:
            return np.full(mask_flat.size, -1, dtype=np.int64)
        nodes = np.unique(np.concatenate([rr, cc]))
        remap = np.empty(mask_flat.size, dtype=np.int64)
        remap[nodes] = np.arange(nodes.size)
        g = _sp.coo_matrix((np.ones(rr.size), (remap[rr], remap[cc])),
                           shape=(nodes.size, nodes.size))
        lab_sub = _sp.csgraph.connected_components(g, directed=False)[1]
        out = np.full(mask_flat.size, -1, dtype=np.int64)
        out[nodes] = lab_sub
        return out

    return _label


def label_signed(t_flat, cft, label):
    """Components of t >= +cft and of t <= -cft, labelled separately.

    `label` is a `build_cluster_labeller` closure; `t_flat` is the signed flat t map.
    Positive components keep their labels, negative ones are offset past them, and the
    background stays -1, so a caller's mass/rank arithmetic is unchanged.

    Per sign, as l2_run._clusters and FieldTrip do. Labelling |t| >= cft in one pass
    lets a positive and a negative patch that touch anywhere on the 64-channel
    adjacency graph fuse into one component, whose span, peak, polarity and effect
    size then describe no single effect. On the IC analysis's 27 x 176 maps per-sign
    labelling gives the same clusters, cells and p on hep/hrsp/hepc, because
    opposite-sign patches do not touch there at these thresholds.
    """
    lp = label(t_flat >= cft)
    ln = label(t_flat <= -cft)
    out = lp.copy()
    kp = int(lp.max()) + 1 if (lp >= 0).any() else 0
    m = ln >= 0
    out[m] = ln[m] + kp
    return out


def cluster_perm_3d(X, chan_adj, n_perm=1000, seed=SEED, alpha=0.05,
                    max_chunk_gb=MAX_NULL_CHUNK_GB, verbose=True):
    """Spatio-temporal cluster permutation over a [nC x (nF) x nT] map.

    X : [nC x nT x nSub] (HEP) or [nC x nF x nT x nSub] (HRSP/HEPC), already
    NaN-free in the subject axis (the caller applies listwise deletion to the
    whole cell family at once, exactly like `_listwise_keep`).

    chan_adj : scipy CSR [nC x nC] from build_channel_adjacency.

    The null t-maps are computed in chunks of cells, so the float64 intermediate never
    exceeds max_chunk_gb. The IC route's `[nCell x nPerm]` matmul holds 27f x 176t x
    10k = 380 MB; at 64 channels that would be ~24 GB. The float32 t array for every
    cell and permutation is still held, because the cluster null needs every column.

    Returns (t_obs, cft, clusters, n_sub) in the same contract as the 2-D
    route, with channels replacing the first map axis:
        t_obs    [nC x nT] or [nC x nF x nT]
        cft      cluster-forming threshold, the (1-alpha) percentile of |t|
        clusters list of dicts sorted by mass desc, each carrying
                 {p, mass, ch0..ch1, f0..f1, t0..t1, p_ch, pf, pt, n_cells,
                  cells [n_cells x ndim], g}
        n_sub    subjects entering the test
    """
    X = np.asarray(X, float)
    if X.ndim == 3:                            # HEP: [nC x nT x nSub]
        X3 = X[:, None, :, :]                  # -> [nC x 1 x nT x nSub]
        squeeze_f = True
    elif X.ndim == 4:
        X3 = X
        squeeze_f = False
    else:
        raise ValueError(f'cluster_perm_3d needs [nC x nT x nSub] or '
                         f'[nC x nF x nT x nSub], got {X.shape}')
    nC, nF, nT, nSub = X3.shape
    Xmap = X3.reshape(nC, nF, nT, nSub)

    if not np.isfinite(Xmap).all():
        raise ValueError('cluster_perm_3d got non-finite cells — apply '
                         'listwise deletion to X before calling.')
    rng = np.random.default_rng(seed)
    flips = rng.choice(np.array([-1.0, 1.0]), size=(nSub, n_perm))
    flips[:, 0] = 1.0                          # identity == observed, never in the null

    # ── CHUNK THE NULL ACROSS CELLS ──────────────────────────────────────────
    # Each chunk reshapes to [nChunk x nSub], runs the one-matmul t and stores its own
    # [nChunk x nPerm] block of the null in float32. The float32 copy halves the memory
    # of the big array and costs nothing the cluster test can resolve (a cell is
    # supra-threshold or not at 7 significant digits).
    # max_chunk_gb caps the chunk, not the whole null: the full map at
    # 64ch x 27f x 176t x 10k permutations is ~12 GB even in float32.
    n_cell = nC * nF * nT
    cells_per_chunk = max(1, int(max_chunk_gb * (1 << 30) / (n_perm * 8)))
    X2d = Xmap.reshape(n_cell, nSub)
    t_flat32 = np.empty((n_cell, n_perm), dtype=np.float32)
    for s in range(0, n_cell, cells_per_chunk):
        e = min(s + cells_per_chunk, n_cell)
        tc = _null_t(X2d[s:e], flips)          # [nChunk x nPerm]
        t_flat32[s:e] = tc.astype(np.float32)
    t_obs = t_flat32[:, 0].astype(float).reshape(nC, nF, nT)

    # Cluster-forming threshold: (1-alpha) percentile of |t| under H0, pooling
    # all columns (the identity included), as l2_heart.cluster_perm_weighted does.
    cft = float(np.percentile(np.abs(t_flat32), 100.0 * (1.0 - alpha)))

    if verbose:
        print(f'  [scalp L2] map {nC}ch x {nF}f x {nT}t, n={nSub}, '
              f'n_perm={n_perm}, cft={cft:.3f} '
              f'(chunked over {int(np.ceil(n_cell / cells_per_chunk))} cell '
              f'blocks of {cells_per_chunk})')

    # ── OBSERVED CLUSTERS + MAX-MASS NULL ────────────────────────────────────
    # The null holds one distribution of the max absolute cluster mass, not two
    # one-sided ones, and column 0 (the identity) is excluded from it, so the
    # smallest attainable p is 1/n_perm. As in l2_heart.cluster_perm_weighted.
    from scipy import sparse as _sp

    _label = build_cluster_labeller(chan_adj, nF, nT)

    def _label_signed(tf, thr=cft, label=_label):
        return label_signed(tf, thr, label)

    # Observed map: one labelling, vectorised masses.
    t_obs_flat = np.abs(t_obs).astype(float).ravel()
    obs_lab = _label_signed(t_obs.astype(float).ravel())     # [n_cell] int, -1 bg
    obs_pos = obs_lab >= 0
    n_obs_cluster = int(obs_lab.max()) + 1 if obs_pos.any() else 0
    obs_masses = (np.bincount(obs_lab[obs_pos],
                              weights=t_obs_flat[obs_pos])
                  if n_obs_cluster else np.array([]))

    # Null of the max cluster mass, from the same matmul result (no recomputation).
    # A cluster's cells share one label, so the argmax of the flat |t| map under that
    # label is its peak, and no per-cluster loop is needed.
    null_max = np.empty(n_perm - 1)
    for p in range(1, n_perm):
        tp = t_flat32[:, p]                          # already computed
        lab_p = _label_signed(tp.astype(float))
        pos = lab_p >= 0
        if pos.any():
            masses_p = np.bincount(lab_p[pos], weights=np.abs(tp)[pos])
            null_max[p - 1] = float(masses_p.max())
        else:
            null_max[p - 1] = 0.0

    # Assemble, sorted by mass. The peak is the argmax of the flat |t| map under each
    # cluster label.
    out = []
    n_eff = float(nSub)
    for ci in np.argsort(obs_masses)[::-1]:
        mass = float(obs_masses[ci])
        this = obs_lab == ci
        pk_flat = int(np.argmax(np.where(this, t_obs_flat, -np.inf)))
        pk = np.unravel_index(pk_flat, (nC, nF, nT))    # (ch, f, t)
        cells = np.argwhere(this.reshape(nC, nF, nT))   # [n x 3] ch, f, t
        # Signed, from the signed t at the peak cell (the peak cell is chosen by |t|),
        # so that g and Polarity carry the cluster's sign.
        t_signed_peak = float(t_obs.ravel()[pk_flat])
        d = t_signed_peak / np.sqrt(n_eff)
        out.append(dict(
            p=(1 + int((null_max >= mass).sum())) / (1 + (n_perm - 1)),
            mass=mass, peak_t=t_signed_peak,
            # the cluster's channels, all of them, not only the peak channel
            chans=sorted({int(c) for c in cells[:, 0]}),
            ch0=int(cells[:, 0].min()), ch1=int(cells[:, 0].max()),
            f0=int(cells[:, 1].min()), f1=int(cells[:, 1].max()),
            t0=int(cells[:, 2].min()), t1=int(cells[:, 2].max()),
            p_ch=int(pk[0]), pf=int(pk[1]), pt=int(pk[2]),
            n_cells=int(this.sum()), cells=cells.astype(np.int32),
            g=float(d * (1 - 3 / (4 * n_eff - 9)) if n_eff > 2 else np.nan)))
    if squeeze_f:
        t_obs = t_obs[:, 0, :]                        # back to [nC x nT]
    return t_obs, cft, out, nSub


# ═══════════════════════════════════════════════════════════════════════════
# THE TIDY TABLE: channels on the map axis, one row per cluster
# ═══════════════════════════════════════════════════════════════════════════
SCALP_FIELDS = (
    'Measure', 'Model', 'Status', 'N_subjects', 'N_dropped_nonfinite',
    'CFT', 'N_supra', 'N_clusters_total', 'Best_p',
    'Cluster', 'Polarity', 'Mixed_polarity',
    'Start_ms', 'End_ms', 'Peak_ms', 'Start_Hz', 'End_Hz', 'Peak_Hz',
    'Channels', 'N_channels', 'Peak_channel',
    'Peak_t', 'SumAbsT', 'perm_p', 'N_cells', 'Survived',
    'CohenD', 'HedgesG', 'CFA_flag', 'Note',
)


def _cfa_overlap_ms(start_ms, end_ms):
    """True where the [start_ms, end_ms] interval overlaps the cardiac-field
    window, the same rule l2_heart applies to its clusters. A True flag means
    "this sits on the QRS and is cardiac until shown otherwise", which is the
    only thing a latency alone can say."""
    return not (end_ms < CFA_WINDOW_MS[0] or start_ms > CFA_WINDOW_MS[1])


def write_scalp_table(rows, path):
    """The tidy per-cluster CSV. Status separates a tested-but-null cell from
    one that was never run; never read a blank perm_p as a null without
    checking Status first. Empty rows still write the header, so `exists` is
    never confounded with "detected"."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(SCALP_FIELDS))
        wr.writeheader()
        wr.writerows(rows)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# SUBJECT STACKING + THE COHORT RUNNER
# ═══════════════════════════════════════════════════════════════════════════
MIN_SUB_L2 = 5           # fewest subjects for a cluster test (FMT's value)


def _scalp_ch_names(subjects):
    """The channel labels, from the first subject that carries them. All are
    the same montage; where they are not the run refuses rather than silently
    aligning two different caps."""
    ch = None
    for s in subjects:
        for sess in s['sessions']:
            if ch is None:
                ch = sess['ch_names']
            elif sess['ch_names'] != ch:
                raise ValueError(
                    f"channel sets disagree: {sess['ch_names']!r} vs {ch!r}")
    return ch


def _stack_scalp(subjects, meas, model):
    """[nChan x (nF) x nT x nUse] with listwise-complete subjects only, and the
    channel labels. A subject whose map is non-finite anywhere is dropped from
    the whole stack, the same listwise rule `_listwise_keep` applies on the IC
    route."""
    maps = []
    for s in subjects:
        per = s.get('l1', {}).get((meas, model))
        if per is None or per.get('beta') is None:
            continue
        maps.append(per['beta'])
    if not maps:
        return np.empty((0, 0, 0, 0)), _scalp_ch_names(subjects)
    X = np.stack(maps, axis=-1)                      # [nC x (nF) x nT x nSub]
    finite = np.isfinite(X).all(axis=tuple(range(X.ndim - 1)))
    X = X[..., finite]
    return X, _scalp_ch_names(subjects)


def _blank_scalp_row(meas, model, status, n_sub, n_drop, note=''):
    r = {k: '' for k in SCALP_FIELDS}
    r.update(Measure=meas, Model=model, Status=status, N_subjects=n_sub,
             N_dropped_nonfinite=n_drop, Cluster=0, Survived=False,
             Mixed_polarity=False, CFA_flag=False, Note=note)
    return r


def _fmt_ms(t):
    return '' if t is None else f'{float(t):.0f}'


def _scalp_cluster_row(meas, model, cl, rank, t_obs, times, freqs, ch_names,
                       n_sub, alpha, cft=None, n_supra=None, n_cl_total=None,
                       best_p=None):
    """One tidy row per cluster, channels on the map axis.

    `Channels` is the row's spatial extent ("Fz+FC1+FC2"), not a single peak
    electrode. The peak is one cell of a cluster that may span dozens of electrodes,
    and a dipolar field's peak electrode depends on the reference as much as on the
    source, so the extent is what carries the claim.
    """
    t0 = float(times[cl['t0']]); t1 = float(times[cl['t1']])
    pt = float(times[cl['pt']])
    if freqs is not None:
        f0 = float(freqs[cl['f0']]); f1 = float(freqs[cl['f1']])
        pf = float(freqs[cl['pf']])
    else:
        f0 = f1 = pf = None
    # the cluster's own channels (`chans`); the ch0..ch1 range is the fallback for a
    # cluster dict without them
    chans = sorted({ch_names[c] for c in cl.get(
        'chans', range(cl['ch0'], cl['ch1'] + 1))})
    r = {k: '' for k in SCALP_FIELDS}
    r.update(
        Measure=meas, Model=model, Status='ok',
        N_subjects=n_sub, N_dropped_nonfinite=0,
        # Map-level fields. Without the CFT a reader cannot say what threshold
        # formed the clusters, and the cluster mask (not only its bounding box)
        # could not be rebuilt; tfce_scalp checks its re-derived CFT against it.
        CFT=('' if cft is None else round(float(cft), 4)),
        N_supra=('' if n_supra is None else int(n_supra)),
        N_clusters_total=('' if n_cl_total is None else int(n_cl_total)),
        Best_p=('' if best_p is None else round(float(best_p), 4)),
        Cluster=rank + 1,
        # From the signed t at the peak cell (`mass` is sum|t| and never negative).
        # Mixed_polarity is False because clusters are formed per polarity.
        Polarity=('pos' if float(cl.get('peak_t', cl['mass'])) >= 0 else 'neg'),
        Mixed_polarity=False,
        Start_ms=_fmt_ms(t0), End_ms=_fmt_ms(t1), Peak_ms=_fmt_ms(pt),
        Start_Hz=('' if f0 is None else f'{f0:.0f}'),
        End_Hz=('' if f1 is None else f'{f1:.0f}'),
        Peak_Hz=('' if pf is None else f'{pf:.0f}'),
        Channels='+'.join(chans) if len(chans) <= 8
                 else f"{chans[0]}+...+{chans[-1]} ({len(chans)}ch)",
        N_channels=len(chans),
        Peak_channel=ch_names[cl['p_ch']],
        # The signed t at the peak cell (the mean |t| is the fallback for a cluster
        # dict without `peak_t`).
        Peak_t=round(float(cl.get('peak_t', cl['mass'] / max(cl['n_cells'], 1))), 3),
        SumAbsT=round(float(cl['mass']), 3),
        perm_p=round(float(cl['p']), 4),
        N_cells=int(cl['n_cells']),
        Survived=bool(cl['p'] < alpha),
        HedgesG=round(float(cl['g']), 3),
        CFA_flag=_cfa_overlap_ms(t0, t1),
        Note=('touches the QRS window — cardiac field until a post-QRS '
              'refit survives' if _cfa_overlap_ms(t0, t1) else ''),
    )
    return r


def run_scalp_l2(subjects, times_hep, times_tf, freqs, n_perm, alpha=0.05,
                 models=('session', 'unadj'), verbose=True,
                 measures=('hep', 'hrsp', 'hepc'), adjacency=None):
    """The whole L2 battery over the cohort's L1 maps.

    subjects : list of per-subject dicts, each with a `'sessions'` list and an
               `'l1'` dict mapping (meas, model) -> {'beta': [nC x ...], ...}.
    measures : the L1 keys to test. Any name other than 'hep' is read as a
               [nC x nF x nT] time-frequency map.
    adjacency : the [nC x nC] sparse graph clusters may grow along. None (the
               default) builds the montage's Delaunay graph from the channel
               labels, which is what a scalp map needs. A caller whose nodes
               should never merge (for instance source ROIs) can pass an
               identity graph.
    Returns (rows, detections): the tidy rows for the CSV, and a dict of the
    surviving clusters keyed (meas, model) -> [clusters].
    """
    rows, detections = [], {}
    for model in models:
        for meas in measures:
            X, ch_names = _stack_scalp(subjects, meas, model)
            n_sub = X.shape[-1] if X.size else 0
            if n_sub < MIN_SUB_L2:
                rows.append(_blank_scalp_row(
                    meas, model, f'skipped:n<{MIN_SUB_L2}', n_sub, 0,
                    note=f'fewer than {MIN_SUB_L2} subjects with a finite map'))
                continue
            if adjacency is None:
                chan_adj, _names = build_channel_adjacency(ch_names)
            else:
                chan_adj = adjacency
                if chan_adj.shape != (len(ch_names), len(ch_names)):
                    raise ValueError(
                        f'adjacency is {chan_adj.shape} but there are '
                        f'{len(ch_names)} nodes')
            t_obs, cft, clusters, n_used = cluster_perm_3d(
                X, chan_adj, n_perm=n_perm, alpha=alpha, verbose=verbose)
            times = times_hep if meas == 'hep' else times_tf
            n_supra = int(np.sum(np.abs(t_obs) >= cft))
            best_p = min((c['p'] for c in clusters), default=None)
            for rank, cl in enumerate(clusters):
                rows.append(_scalp_cluster_row(
                    meas, model, cl, rank, t_obs, times,
                    (freqs if meas != 'hep' else None), ch_names, n_used, alpha,
                    cft=cft, n_supra=n_supra, n_cl_total=len(clusters),
                    best_p=best_p))
                if cl['p'] < alpha:
                    detections.setdefault((meas, model), []).append(cl)
            if not clusters:
                r0 = _blank_scalp_row(
                    meas, model, 'no supra-threshold cells', n_sub, 0,
                    note=f'|t| < cft={cft:.3f} at every cell')
                r0.update(CFT=round(float(cft), 4), N_supra=0,
                          N_clusters_total=0, Best_p='')
                rows.append(r0)
            if verbose:
                n_surv = sum(1 for cl in clusters if cl['p'] < alpha)
                print(f'  [{meas}/{model}] n={n_used} cft={cft:.3f} '
                      f'clusters={len(clusters)} survived={n_surv}')
    return rows, detections


def apply_scalp_subject_gates(session_records, subj, p, verbose=True):
    """Participant gates over the per-session records, in the IC route's order
    (heart_depth_run.build_cohort). Drops failing subjects from `subj` in place
    and returns one record per subject, kept or not.

    minSeg and depth diversity come first, so a participant with no usable data is
    not also reported as an ECG failure; the ECG mean is over kept sessions.
    """
    subject_records = []
    for sid in sorted({r['subject'] for r in session_records}):
        _qa = [r for r in session_records
               if r['subject'] == sid and r['status'] == 'kept']
        n_seg = int(sum(r['n_seg_used'] for r in _qa))
        n_dep = len({int(v) for s in subj.get(sid, {}).get('sessions', [])
                     for v in s['depth']})
        _ecg_mean, _ = subject_ecg_clean_ratio(
            [r['ecg_clean_ratio'] for r in _qa])
        srec = dict(subject=sid,
                    n_sessions_found=sum(1 for r in session_records
                                         if r['subject'] == sid),
                    n_sessions_kept=len(_qa), n_segments=n_seg,
                    n_depths_pooled=n_dep,
                    ecg_clean_ratio=('' if not np.isfinite(_ecg_mean)
                                     else round(float(_ecg_mean), 4)),
                    status='kept', reason='')
        if not _qa:
            srec['status'] = 'dropped'
            srec['reason'] = 'no session survived the session-level gates'
        elif n_seg < int(p['minSeg']):
            srec['status'] = 'dropped'
            srec['reason'] = f'minSeg: {n_seg} pooled segment(s) < {int(p["minSeg"])}'
        elif n_dep < 2:
            srec['status'] = 'dropped'
            srec['reason'] = f'{n_dep} distinct depth level(s) pooled'
        elif not ecg_ratio_passes_gate(_ecg_mean, float(p['ECG_MAX'])):
            srec['status'] = 'dropped'
            srec['reason'] = 'subject ECG QA: ' + (
                'ECG unassessable' if not np.isfinite(_ecg_mean)
                else f'ECG {_ecg_mean:.2f}>{float(p["ECG_MAX"]):.2f}')
        if srec['status'] == 'dropped':
            subj.pop(sid, None)
        subject_records.append(srec)
    if verbose:
        _kept = [r for r in session_records if r['status'] == 'kept'
                 and r['subject'] in subj]
        print(f'\n  cohort: {len(subj)} subject(s), {len(_kept)} session(s), '
              f'{sum(r["n_seg_used"] for r in _kept)} segment(s)')
    return subject_records


# ═══════════════════════════════════════════════════════════════════════════
# THE COHORT RUNNER: sessions -> per-subject per-channel slopes -> L2
# ═══════════════════════════════════════════════════════════════════════════
def run_scalp(data_dir, results_root, p=None, limit=None, verbose=True):
    """The full mass-univariate scalp run.

    `p` is DEFAULT_SCALP_PARAMS with any overrides. Returns a small dict of output
    paths. The result tree mirrors the selected-IC route, so the two can be read side
    by side.

    The write order is fixed: run_results.npz is written with every L1 map before any
    L2 is attempted, so an L2 failure cannot make a finished L1 unrecoverable.
    """
    from heart_depth_pipeline import list_sessions
    from heart_depth_run import _write_csv
    from result_path import (result_dir, heart_depth_tag,
                                 assert_no_param_collision)

    if p is None:
        p = dict(DEFAULT_SCALP_PARAMS)

    # ── SESSION LIST (the same discovery the IC route uses, without its
    #    IC-selection pre-pass: this analysis selects nothing) ──────────────
    all_sessions = list_sessions(data_dir)
    if not all_sessions:
        raise FileNotFoundError(
            f'no segment files under {data_dir!r} (expected '
            f'<data_dir>/<subject>/ses<N>.npz, written by extract_npz.py)')
    if limit is not None:
        keep_sids = sorted({sid for sid, _s, _f in all_sessions})[:int(limit)]
        all_sessions = [s for s in all_sessions if s[0] in set(keep_sids)]
        if verbose:
            print(f'  --limit {limit}: {len(keep_sids)} subject(s), '
                  f'{len(all_sessions)} session(s) — SMOKE TEST, not a cohort.')

    # ── L1: per-session measures, then per-subject per-channel slopes ───────
    subj = {}                                   # sid -> {'sessions': [..], 'l1': {}}
    session_records = []                        # one row per session, kept or not
    n_jobs = max(1, int(p.get('N_JOBS', 1) or 1))
    if n_jobs > 1:
        # Order is the correctness condition here. `Executor.map` yields in input
        # order regardless of which worker finishes first, so sessions are appended
        # to each subject in exactly the sequence the sequential path uses.
        # as_completed() would return them in completion order, reorder each
        # subject's session list and silently change which session is session 1 for
        # the blocking dummies.
        from concurrent.futures import ProcessPoolExecutor
        if verbose:
            print('  L1 sessions: %d worker process(es), ~1.1 GiB transient each'
                  % n_jobs)
        jobs = [(a, b, c, p) for a, b, c in all_sessions]
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            done = ex.map(_session_job, jobs)
            for k, (sid, ses, ms, notes) in enumerate(done, 1):
                if verbose:
                    print('\n[%d/%d] %s ses%s' % (k, len(all_sessions), sid, ses))
                session_records.append(_scalp_session_record(sid, ses, ms, notes))
                if ms is None:
                    if verbose:
                        print('  DROPPED — %s' % notes.get('skip_reason', '?'))
                    continue
                subj.setdefault(sid, {'sessions': [], 'l1': {}})
                subj[sid]['sessions'].append(ms)
                if verbose:
                    print('  KEPT — %d segment(s), %d channels'
                          % (len(ms['depth']), ms['hep'].shape[1]))
    else:
        for k, (sid, ses, npz_path) in enumerate(all_sessions, 1):
            if verbose:
                print(f'\n[{k}/{len(all_sessions)}] {sid} ses{ses}')
            notes = {}
            ms = session_measures_scalp(npz_path, p, verbose=verbose, notes=notes)
            session_records.append(_scalp_session_record(sid, ses, ms, notes))
            if ms is None:
                if verbose:
                    print(f'  DROPPED — {notes.get("skip_reason", "?")}')
                continue
            subj.setdefault(sid, {'sessions': [], 'l1': {}})
            subj[sid]['sessions'].append(ms)
            if verbose:
                print(f'  KEPT — {len(ms["depth"])} segment(s), '
                      f'{ms["hep"].shape[1]} channels')

    # ── SUBJECT GATES (apply_scalp_subject_gates) ───────────────────────────
    subject_records = apply_scalp_subject_gates(session_records, subj, p,
                                                verbose=verbose)

    subjects = []
    for sid, rec in sorted(subj.items()):
        if verbose:
            print(f'  L1 {sid}: ', end='', flush=True)
        for model in p['MODELS']:
            for meas in ('hep', 'hrsp', 'hepc'):
                beta, svar, n_units, dmean, dn, dlv = subject_slopes_scalp(
                    rec['sessions'], meas, p, model=model)
                rec['l1'][(meas, model)] = {'beta': beta, 'svar': svar,
                                            'n_units': n_units,
                                            'depth_mean': dmean,
                                            'depth_n': dn,
                                            'depth_levels': dlv}
        subjects.append(rec)
        if verbose:
            print(f'{len(rec["sessions"])} session(s), '
                  f'{sum(v["n_units"] for v in rec["l1"].values())} units')

    if not subjects:
        # Still create the folder and write an empty CSV, so the run is
        # traceable rather than a silent no-show.
        main_dir = result_dir(results_root, 'HEART_DEPTH_SCALP',
                              tag=heart_depth_tag(p), section='main', create=True)
        assert_no_param_collision(os.path.dirname(main_dir), p)
        _write_csv(os.path.join(main_dir, 'cohort_sessions.csv'),
                   session_records, SCALP_SESSION_FIELDS)
        _write_csv(os.path.join(main_dir, 'cohort_subjects.csv'),
                   subject_records, SCALP_SUBJECT_FIELDS)
        csv_path = os.path.join(main_dir, 'l2_scalp_summary.csv')
        write_scalp_table([], csv_path)
        if verbose:
            print('  no subjects survived the cohort gates — empty CSV written')
        return dict(main=main_dir, npz=None, csv=csv_path,
                    detections={}, l2_errors={})

    # ── WRITE run_results.npz FIRST, before any L2 ──────────────────────────
    ch_names = _scalp_ch_names(subjects)
    freqs = subjects[0]['sessions'][0]['freqs']
    # The trimmed axis, because the maps are trimmed: subject_slopes_scalp applies
    # `tlims_mask(sessions[0]['times'], p['TLIMS'])` to Y3, so every L1 beta is on the
    # TLIMS axis (176 points, -100..600 ms) while the session's epoch axis is 251
    # points (-300..700 ms). Labelling clusters against the untrimmed axis would make
    # every latency 200 ms early. heart_depth_run trims `times` the same way.
    times_full = np.asarray(subjects[0]['sessions'][0]['times'], float)
    times = times_full[tlims_mask(times_full, p['TLIMS'])]
    times_h = times_hep = times                    # same axis for HEP and TF here

    # And refuse to label against an axis that does not fit (run_l2_heart raises on
    # this too).
    _nT = next((np.asarray(v['beta']).shape[-1]
                for s in subj.values() for v in s['l1'].values()
                if v.get('beta') is not None), None)
    if _nT is not None and _nT != times.size:
        raise ValueError(
            f'scalp L1 maps have {_nT} time points but the TLIMS axis '
            f'{tuple(p["TLIMS"])} has {times.size} (full epoch axis is '
            f'{times_full.size}). Refusing to label clusters in ms against the '
            f'wrong axis.')
    # HEART_DEPTH_SCALP is this analysis's slot in result_path.ARMS; the tag is the
    # same parameter hash as the selected-IC route, so a scalp run and a Lee run at
    # the same parameters share a parent and differ only at the arm level: they are
    # one analysis family read two ways.
    main_dir = result_dir(results_root, 'HEART_DEPTH_SCALP',
                          tag=heart_depth_tag(p), section='main', create=True)

    # Stamp the parameters into the folder. `assert_no_param_collision` writes the
    # stamp and refuses to reuse a folder for a different parameter set, which the
    # shortened tag does not prevent on its own.
    _param_changes = assert_no_param_collision(os.path.dirname(main_dir), p)
    if _param_changes and verbose:
        print('  NOTE reusing this folder; parameters that changed: %s'
              % ', '.join(_param_changes))
    _write_csv(os.path.join(main_dir, 'cohort_sessions.csv'),
               session_records, SCALP_SESSION_FIELDS)
    _write_csv(os.path.join(main_dir, 'cohort_subjects.csv'),
               subject_records, SCALP_SUBJECT_FIELDS)
    npz_out = os.path.join(main_dir, 'run_results.npz')
    np.savez_compressed(
        npz_out,
        **{f'l1__{meas}__{model}__{sid}': subj[sid]['l1'][(meas, model)]['beta']
           for sid in subj
           for model in p['MODELS'] for meas in ('hep', 'hrsp', 'hepc')
           if subj[sid]['l1'].get((meas, model), {}).get('beta') is not None},
        **{f'depth__{meas}__{model}__{sid}':
           subj[sid]['l1'][(meas, model)]['depth_mean']
           for sid in subj
           for model in p['MODELS'] for meas in ('hep', 'hrsp', 'hepc')
           if subj[sid]['l1'].get((meas, model), {}).get('depth_mean')
           is not None},
        **{f'depthn__{meas}__{model}__{sid}':
           subj[sid]['l1'][(meas, model)]['depth_n']
           for sid in subj
           for model in p['MODELS'] for meas in ('hep', 'hrsp', 'hepc')
           if subj[sid]['l1'].get((meas, model), {}).get('depth_n')
           is not None},
        # The participant's mean L1 slope variance over the whole map, the one
        # number ivw_own needs (tfce_scalp --scheme ivw_own). Readers that select
        # on the l1__/depth__ prefixes ignore it.
        **{f'svarmean__{meas}__{model}__{sid}':
           np.asarray(np.nanmean(subj[sid]['l1'][(meas, model)]['svar']))
           for sid in subj
           for model in p['MODELS'] for meas in ('hep', 'hrsp', 'hepc')
           if subj[sid]['l1'].get((meas, model), {}).get('beta') is not None
           and subj[sid]['l1'][(meas, model)].get('svar') is not None},
        depth_levels=np.arange(1, 6),
        ch_names=np.asarray(ch_names), times=times, freqs=freqs,
        n_sub=len(subjects))
    if verbose:
        print(f'  wrote {npz_out}')

    # ── L2 ──────────────────────────────────────────────────────────────────
    rows, detections = run_scalp_l2(subjects, times_h, times, freqs,
                                    p['nPerm'], alpha=p['ALPHA'],
                                    verbose=verbose)
    csv_path = os.path.join(main_dir, 'l2_scalp_summary.csv')
    write_scalp_table(rows, csv_path)

    return dict(main=main_dir, npz=npz_out, csv=csv_path,
                detections=detections, l2_errors={})
