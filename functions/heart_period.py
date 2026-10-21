# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
HEP / HRSP / HEPC contrasted between task periods. HEPC, as the code names it, is the paper's
HRPC (heartbeat-related phase consistency).

run_period_contrasts.py calls `run(data_dir, results_root, params, limit=...)` with
config.PERIOD_HEART; `params` updates CONFIG below.

Uses `heart_core` (reconstruct_ics, epoch_beats, morlet_cwt, ppc_map, hep_average), `get_rr`
(the BrainBeats detector), `lee_selection` for the component choice and
`l2_heart.run_l2_heart` for the group stage, and reads the periods from the segment files
(data/segments/<subject>/ses<N>.npz, config.DATA_DIR).

===========================================================================
FOUR PERIODS, SIX CONTRASTS
===========================================================================
    baseline · pre-meditation · post-meditation · active-oddball

`delta = mean(B) - mean(A)`, A the reference. Six contrasts over four means carry three df
between them, and nothing corrects across them.

===========================================================================
HOW THIS DIFFERS FROM A DEPTH RUN
===========================================================================
1. The unit is a beat, not a segment. A period is one continuous block, so there is no
   segment/cell/perbeat grain to choose: one map per (period, session), pooled over sessions,
   then differenced.

2. Selection must not see the contrast. `lee` is functional (it ranks ICs by heartbeat-band
   response), so selecting on all four periods and then contrasting them would ask whether a
   component chosen for having a large heartbeat response has a large heartbeat response. The
   component is therefore selected per session on the reference periods only (`SELECT_ON`),
   and the run parameters record which. `lee` is the only selection method this module
   accepts.

3. The .npz gives 4 s windows, not a recording. `extract_period_windows` steps
   `pos += win_samp` with no gap, so consecutive windows are contiguous samples and this module
   re-joins them into one block before detecting R-peaks or epoching. A period made of more
   than one span is spliced at the joins, and the .npz does not record which window came from
   which span. `DROP_EDGE_BEATS` drops beats whose epoch would run off the block, and the count
   is reported, but a beat sitting exactly on an internal splice cannot be detected.
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

from heart_core import reconstruct_ics, epoch_beats, hep_average, morlet_cwt, ppc_map
from get_rr import get_rr, resolve_polarity, GetRRError             # noqa: E402
from fmt_period import period_block, contrasts_of  # noqa: E402

CONFIG = dict(
    IC_SELECT_METHOD='lee',            # functional selection; the only method
    SELECT_ON=['baseline', 'pre-meditation'],   # periods selection runs on; never the contrast
    # -- lee selection knobs, passed to select_ics_lee ----------------------
    HRSP_SEL_CRITERION='hrsp',         # 'hrsp' (power) | 'ppc' (phase)
    IC_SEL_NORM='raw',                 # 'raw'|'ztrial'|'sham'|'sham_raw'|'blratio'
    IC_SEL_THRESH=1.0,                 # 1 = single best IC; <1 averages more
    IC_SEL_SHAM_N=50,                  # only read by the sham norms
    HRSP_SEL_BANDS=[4, 20],            # the SCORE's band (not HRSP_FREQS)
    HRSP_SEL_WIN=[300, 550],           # ms, the score's window
    BASELINE_WIN=[-150, -50],          # ms, the score's baseline

    PERIODS=['baseline', 'pre-meditation', 'post-meditation', 'active-oddball'],
    CONTRASTS='all',

    TLIMS=[-100, 600],                 # ms, the analysed epoch window
    EPOCH_MS=[-300, 700],              # ms, extracted (wider, for the CWT edges)
    HRSP_FREQS=list(range(2, 21)),     # Hz
    HRSP_WAVE_WIDTH=5.0,
    MEASURES=['hep', 'hrsp', 'hepc'],

    MIN_IBI_MS=600,
    MIN_BEATS_PER_PERIOD=50,           # minimum epochable beats per period
    DROP_EDGE_BEATS=True,
    # What the Morlet transform runs on. 'epoch': each -300 to 700 ms epoch on its own,
    # so at low frequencies most of the epoch is within the wavelet's reach of a
    # zero-padded edge (3 SD: 0.6 s at 4 Hz, 1.2 s at 2 Hz). 'block': each period's
    # continuous block, then epoched, with beats whose epoch comes within
    # EDGE_GUARD_S of a block end dropped; in the selection, each SELECT_ON period is
    # transformed on its own, so no transform runs across the join between two
    # periods. Folder token '_cwtblock'.
    WAVELET_ON='epoch',
    EDGE_GUARD_S=2.0,

    # 'none' | 'glm' | 'subtract'. 'glm' is the period form of the depth analysis's baseline
    # correction (each beat's own pre-R power, BASELINE_WIN, as a regressor): per session and
    # per (freq, time) cell, HRSP power is regressed on the beat's pre-R power at that
    # frequency with the slope pooled within periods, and every period mean is moved to the
    # session's mean pre-R power. HRSP only; HEP and HEPC are untouched. Folder tag '_bl'.
    # 'subtract': each beat's own mean pre-R power (BASELINE_WIN) is subtracted from that beat
    # at every frequency before averaging, so the map is the change from the pre-R level.
    # Folder tag '_blsub'.
    BASELINE_METHOD='none',

    N_PERM=1000,
    ALPHA=0.05,
    SCHEMES=['unweighted', 'precision_nseg'],
)


def _period_signals(z, per, p):
    """One period re-joined -> (eeg [nCh x N], ecg [N], n_win) or (None, ...)."""
    eeg, n_win, win = period_block(z, per, 'eeg')
    if eeg is None:
        return None, None, 0
    ecg, _n2, _w2 = period_block(z, per, 'ecg')
    return eeg, (None if ecg is None else ecg.ravel()), n_win


def _wavelet_on(p):
    w = str(p.get('WAVELET_ON', 'epoch'))
    if w not in ('epoch', 'block'):
        raise ValueError("WAVELET_ON must be 'epoch' or 'block', got %r" % w)
    return w


def period_measures(eeg_ic, ecg, fs, p, pol=1, notes=None):
    """HEP / HRSP / HEPC for one continuous period block of one component.

    -> dict(measure -> (map, var_map, n_beats)), where `map` is the mean over
    beats and `var_map` the per-beat variance / n (the standard error squared),
    which is what `run_l2_heart` wants as `svar`.
    """
    notes = {} if notes is None else notes
    n = int(eeg_ic.size)
    # -- R-peaks on the re-joined block -------------------------------------
    try:
        pk, _ = get_rr(pol * np.asarray(ecg, float), fs, polarity=1,
                       drop_first=False)
    except GetRRError as exc:
        notes['rpeak_error'] = str(exc)
        return {}
    if pk.size < 3:
        notes['rpeak_error'] = 'fewer than 3 R-peaks in the block'
        return {}
    # Same cleaning as the depth analysis: short intervals removed beat by beat.
    rr = np.diff(pk) / float(fs)
    keep = np.ones(pk.size, bool)
    keep[1:] = rr >= (float(p['MIN_IBI_MS']) / 1000.0)
    notes['n_short_ibi'] = int((~keep).sum())
    pk = pk[keep]

    lo_ms, hi_ms = p['EPOCH_MS']
    lo_s = int(round(lo_ms * fs / 1000.0))
    hi_s = int(round(hi_ms * fs / 1000.0))
    cwt_block = _wavelet_on(p) == 'block'
    if cwt_block:
        # Real data on both sides of every epoch, beyond the wavelet's reach.
        g = int(round(float(p['EDGE_GUARD_S']) * fs))
        ok = (pk + lo_s - g >= 0) & (pk + hi_s + g < n)
        notes['n_edge_dropped'] = int((~ok).sum())
        pk = pk[ok]
    elif p['DROP_EDGE_BEATS']:
        ok = (pk + lo_s >= 0) & (pk + hi_s < n)
        notes['n_edge_dropped'] = int((~ok).sum())
        pk = pk[ok]
    if pk.size < int(p['MIN_BEATS_PER_PERIOD']):
        notes['skip_reason'] = ('%d epochable beat(s) < MIN_BEATS_PER_PERIOD %d'
                                % (pk.size, int(p['MIN_BEATS_PER_PERIOD'])))
        return {}

    # `epoch_beats` returns a pair (epochs, times_ms), and its axis includes the end sample
    # (`i1 + 1`), so rebuilding the axis here from lo_ms/fs would be one sample short and
    # would misalign the window mask.
    ep, t_full = epoch_beats(np.asarray(eeg_ic, float), pk, (lo_ms, hi_ms), fs)
    if ep is None or ep.shape[0] == 0:
        notes['skip_reason'] = 'epoch_beats returned nothing'
        return {}
    n_beats = int(ep.shape[0])
    notes['n_beats'] = n_beats
    tsel = (t_full >= p['TLIMS'][0]) & (t_full <= p['TLIMS'][1])

    out = {}
    if 'hep' in p['MEASURES']:
        m = hep_average(ep)[tsel]
        v = np.var(ep[:, tsel], axis=0, ddof=1) / n_beats
        out['hep'] = (m, v, n_beats)

    if 'hrsp' in p['MEASURES'] or 'hepc' in p['MEASURES']:
        freqs = np.asarray(p['HRSP_FREQS'], float)
        if cwt_block:
            # The block's transform indexed at the epochs, with the sample convention of
            # `epoch_beats` (i0..i1 inclusive); every beat here fits, after the guard.
            Wc = morlet_cwt(np.asarray(eeg_ic, float), freqs, fs,
                            width=float(p['HRSP_WAVE_WIDTH']))          # [nF x n]
            i0 = int(round(lo_ms * fs / 1000.0))
            i1 = int(round(hi_ms * fs / 1000.0))
            idx = np.stack([np.arange(q + i0, q + i1 + 1) for q in pk])
            W = Wc[:, idx].transpose(1, 0, 2)                           # [nBeat x nF x nT]
        else:
            W = morlet_cwt(ep, freqs, fs, width=float(p['HRSP_WAVE_WIDTH']))
        if 'hrsp' in p['MEASURES']:
            pw = np.abs(W) ** 2
            if str(p.get('BASELINE_METHOD', 'none')).lower() == 'subtract':
                bl0, bl1 = p['BASELINE_WIN']
                pw = pw - pw[..., (t_full >= bl0) & (t_full <= bl1)].mean(
                    axis=-1, keepdims=True)
            if str(p.get('BASELINE_METHOD', 'none')).lower() == 'glm':
                # sufficient statistics for the pooled-within-period slope,
                # combined over the session's periods in `baseline_adjust`
                bl0, bl1 = p['BASELINE_WIN']
                x = pw[..., (t_full >= bl0) & (t_full <= bl1)].mean(axis=-1)
                xc = x - x.mean(axis=0)                       # [nBeat x nF]
                yc = pw[..., tsel] - pw[..., tsel].mean(axis=0)
                out['_bl'] = dict(xbar=x.mean(axis=0), sxx=(xc ** 2).sum(axis=0),
                                  sxy=np.einsum('bf,bft->ft', xc, yc),
                                  syy=(yc ** 2).sum(axis=0), n=n_beats)
            pw = pw[..., tsel]
            m = np.nanmean(pw, axis=0)
            v = np.nanvar(pw, axis=0, ddof=1) / n_beats
            out['hrsp'] = (m, v, n_beats)
        if 'hepc' in p['MEASURES']:
            # PPC has no per-beat value, so its variance cannot be a variance over beats.
            # Vinck's unbiased estimator has sampling variance ~ (1 - ppc^2) / n; that is
            # what is supplied, as an approximation used only as an IVW weight, never quoted.
            z = W / (np.abs(W) + np.finfo(float).eps)
            ppc = ppc_map(z)[..., tsel]
            v = np.clip(1.0 - ppc ** 2, 1e-6, None) / max(n_beats - 1, 1)
            out['hepc'] = (ppc, v, n_beats)
    return out


def baseline_adjust(out, notes=None):
    """BASELINE_METHOD='glm': adjust a session's per-period HRSP means in place.

    b[f, t] = sum_p Sxy_p / sum_p Sxx_p (slope of power on the beat's pre-R
    power, pooled within periods, so a between-period difference in pre-R power
    does not enter it); mean_p -> mean_p - b * (xbar_p - xbar_session), with
    xbar_session the beat-weighted mean over the session's periods. The variance
    map becomes the residual variance of that fit / n. A no-op when no period
    carries the `_bl` statistics (BASELINE_METHOD='none').
    """
    pers = [q for q in out if '_bl' in out[q] and 'hrsp' in out[q]]
    if not pers:
        return out
    sxx = sum(out[q]['_bl']['sxx'] for q in pers)
    sxy = sum(out[q]['_bl']['sxy'] for q in pers)
    ntot = float(sum(out[q]['_bl']['n'] for q in pers))
    xses = sum(out[q]['_bl']['n'] * out[q]['_bl']['xbar'] for q in pers) / ntot
    b = sxy / np.where(sxx > 0, sxx, np.nan)[:, None]
    b = np.where(np.isfinite(b), b, 0.0)
    for q in pers:
        st = out[q].pop('_bl')
        m, _v, n = out[q]['hrsp']
        m_adj = m - b * (st['xbar'] - xses)[:, None]
        res = st['syy'] - 2.0 * b * st['sxy'] + b ** 2 * st['sxx'][:, None]
        v_adj = np.clip(res, 0.0, None) / max(n - 2, 1) / n
        out[q]['hrsp'] = (m_adj, v_adj, n)
    if notes is not None:
        notes['baseline_slope_median'] = float(np.median(b))
    return out


def lee_select_for_session(z, p, fs, pol, verbose=False):
    """Lee IC selection for one session, on the SELECT_ON periods only.

    -> (list of 0-based IC indices, notes)

    Selection must not see the contrast: the selection signal is built from the SELECT_ON
    periods only, concatenated, and a session with none of them is dropped with a named reason.
    The lee knobs in CONFIG are passed through to select_ics_lee, not defaulted.
    """
    from lee_selection import select_ics_lee
    notes = {}
    sel_periods = [q for q in (p.get('SELECT_ON') or []) if q in p['PERIODS']]
    if not sel_periods:
        notes['skip_reason'] = ('SELECT_ON names no period in PERIODS, so there '
                                'is nothing to select on')
        return [], notes

    sigs, ecgs = [], []
    for per in sel_periods:
        eeg, ecg, _n = _period_signals(z, per, p)
        if eeg is None or ecg is None:
            continue
        comp = reconstruct_ics(eeg, np.asarray(z['icaweights'], float),
                               np.asarray(z['icasphere'], float),
                               np.asarray(z['icachansind']).ravel())
        sigs.append(comp)
        ecgs.append(np.asarray(ecg, float))
    if not sigs:
        notes['skip_reason'] = ('none of the SELECT_ON periods (%s) carried both '
                                'EEG and ECG' % ','.join(sel_periods))
        return [], notes
    comp = np.concatenate(sigs, axis=1)
    ecg = np.concatenate(ecgs)

    try:
        pk, _ = get_rr(pol * ecg, fs, polarity=1, drop_first=False)
    except GetRRError as exc:
        notes['skip_reason'] = 'R-peak detection failed on the selection block: %s' % exc
        return [], notes
    rr = np.diff(pk) / float(fs)
    keep = np.ones(pk.size, bool)
    keep[1:] = rr >= (float(p['MIN_IBI_MS']) / 1000.0)
    pk = pk[keep]

    lo_ms, hi_ms = p['EPOCH_MS']
    lo_s = int(round(lo_ms * fs / 1000.0))
    hi_s = int(round(hi_ms * fs / 1000.0))
    n = comp.shape[1]
    pk = pk[(pk + lo_s >= 0) & (pk + hi_s < n)]
    cwt_block = _wavelet_on(p) == 'block'
    if cwt_block:
        # Drop beats whose epoch comes within EDGE_GUARD_S of either end of any of the
        # concatenated periods: each period is transformed on its own below.
        g = int(round(float(p['EDGE_GUARD_S']) * fs))
        ends = np.cumsum([s.shape[1] for s in sigs])
        starts = np.concatenate([[0], ends[:-1]])
        ok = np.zeros(pk.size, bool)
        for s0, s1 in zip(starts, ends):
            ok |= (pk + lo_s - g >= s0) & (pk + hi_s + g < s1)
        notes['n_edge_dropped_selection'] = int((~ok).sum())
        pk = pk[ok]
    if pk.size < int(p['MIN_BEATS_PER_PERIOD']):
        notes['skip_reason'] = ('%d epochable beat(s) in the selection block < '
                                'MIN_BEATS_PER_PERIOD %d'
                                % (pk.size, int(p['MIN_BEATS_PER_PERIOD'])))
        return [], notes
    notes['n_beats_selection'] = int(pk.size)

    # The selection band, not HRSP_FREQS: HRSP_SEL_BANDS is the band the Lee score is computed
    # over ([4, 20] by default), HRSP_FREQS the band of the measures. Using the latter would
    # score components on a different band than the depth analysis.
    b0, b1 = p['HRSP_SEL_BANDS']
    sel_freqs = np.arange(float(b0), float(b1) + 0.5, 1.0)
    ei = np.stack([np.arange(q + lo_s, q + hi_s + 1) for q in pk])
    t_axis = (np.arange(lo_s, hi_s + 1) / float(fs)) * 1000.0
    w0, w1 = p['HRSP_SEL_WIN']
    t_sel = (t_axis >= w0) & (t_axis <= w1)
    bl0, bl1 = p['BASELINE_WIN']
    t_bl = (t_axis >= bl0) & (t_axis <= bl1)

    if cwt_block:
        ic_cwt = np.concatenate([morlet_cwt(s, sel_freqs, fs, width=float(p['HRSP_WAVE_WIDTH']))
                                 for s in sigs], axis=-1)
    else:
        ic_cwt = morlet_cwt(comp, sel_freqs, fs, width=float(p['HRSP_WAVE_WIDTH']))
    sel, _w, _pool = select_ics_lee(
        ic_cwt, ei, t_bl, t_sel,
        criterion=p['HRSP_SEL_CRITERION'], norm_mode=p['IC_SEL_NORM'],
        ic_sel_thresh=float(p['IC_SEL_THRESH']), n_pts=n,
        sham_opts=dict(fs=fs, n=int(p.get('IC_SEL_SHAM_N', 50)),
                       shift_ms=(-500, 500), min_shift_ms=150, seed=20260728))
    ics = [int(v) for v in np.atleast_1d(np.asarray(sel, int)).ravel()]
    notes['ics'] = ics
    notes['select_on'] = sel_periods
    if verbose:
        print('    lee selected IC(s) %s on %s'
              % (ics, '+'.join(sel_periods)))
    return ics, notes


def session_periods(npz_path, p, verbose=True):
    """Per-period measure maps for one session. -> (dict, notes)."""
    z = np.load(npz_path, allow_pickle=True)
    fs = int(z['fs'])
    sid, ses = str(z['subject_id']), int(z['session'])
    notes = {'subject': sid, 'session': ses}

    # Polarity once per session, on the longest period: re-voting per period would let one
    # recording disagree with itself between two blocks of one session. Resolved before
    # selection, because lee needs the polarity to detect R-peaks on its selection block.
    longest, best = None, -1
    for per in p['PERIODS']:
        _e, ecg, nw = _period_signals(z, per, p)
        if ecg is not None and nw > best:
            longest, best = ecg, nw
    if longest is None:
        notes['skip_reason'] = 'no period carried an ECG channel'
        return None, notes
    pol_early = int(resolve_polarity(np.asarray(longest, float), fs))

    # Selection happens here, per session, on the SELECT_ON periods only, so it never sees the
    # tested contrast.
    ics, lnotes = lee_select_for_session(z, p, fs, pol_early, verbose=verbose)
    notes.update({k: v for k, v in lnotes.items() if k != 'skip_reason'})
    if not ics:
        notes['skip_reason'] = lnotes.get(
            'skip_reason', 'lee selected no component for this session')
        return None, notes
    notes['ics'] = [int(i) for i in ics]
    # get_rr's own polarity vote, not `ecg_polarity`: R-peaks are detected with get_rr, so
    # polarity is resolved with get_rr's front end (`ecg_polarity` votes with two neurokit2
    # passes). The two agree on every analysed session of the depth cohort; see
    # `get_rr.resolve_polarity`.
    pol = pol_early          # already resolved above, before selection
    notes['ecg_polarity'] = pol

    out = {}
    for per in p['PERIODS']:
        eeg, ecg, n_win = _period_signals(z, per, p)
        if eeg is None or ecg is None:
            notes.setdefault('absent_periods', []).append(per)
            continue
        comp = reconstruct_ics(eeg, np.asarray(z['icaweights'], float),
                               np.asarray(z['icasphere'], float),
                               np.asarray(z['icachansind']).ravel())
        # Polarity-aligned mean, not a plain one: averaging components of opposite
        # topographic sign cancels the HEP toward zero (see the multi-IC combination in
        # heart_depth_pipeline.session_measures), and IC_SEL_THRESH < 1 can select several
        # components. The sign is the dot product of each IC's topography with the top IC's,
        # as there; icawinv is the mixing matrix (columns = maps).
        _ics = np.asarray(ics, int)
        _sig = np.atleast_2d(comp[_ics, :])
        if _ics.size > 1:
            _winv = (np.asarray(z['icawinv'], float) if 'icawinv' in z.files
                     else np.linalg.pinv(np.asarray(z['icaweights'], float)
                                         @ np.asarray(z['icasphere'], float)))
            _tp = _winv[:, _ics]
            _tp = _tp / np.where(np.linalg.norm(_tp, axis=0) > 0,
                                 np.linalg.norm(_tp, axis=0), 1.0)
            _sgn = np.sign(_tp.T @ _tp[:, 0])
            _sgn[_sgn == 0] = 1.0
            _sig = _sig * _sgn[:, None]
            notes.setdefault('polarity_flips', {})[per] = int((_sgn < 0).sum())
        sig = np.mean(_sig, axis=0)
        pn = {}
        res = period_measures(sig, ecg, fs, p, pol=pol, notes=pn)
        notes.setdefault('per_period', {})[per] = pn
        if res:
            out[per] = res
            out[per]['_nwin'] = n_win
    if not out:
        notes['skip_reason'] = 'no period produced a measure'
        return None, notes
    baseline_adjust(out, notes)
    return out, notes


def pool_periods(sessions, p):
    """A subject's sessions pooled per period. -> {(period, meas): (m, v, n)}.

    Kept separate from `subject_deltas` so that the per-period means, which a per-period
    profile figure is drawn from, are available without recomputing the cohort;
    `subject_deltas` calls it and differences the result.
    """
    per = {}
    for s in sessions:
        for q, byq in s.items():
            for meas, val in byq.items():
                if meas in ('_nwin', '_bl'):
                    continue
                per.setdefault((q, meas), []).append(val)
    pooled = {}
    for (q, meas), lst in per.items():
        w = np.asarray([v[2] for v in lst], float)
        w = w / np.sum(w)
        m = np.sum([wi * v[0] for wi, v in zip(w, lst)], axis=0)
        # variance of a weighted mean of independent session means
        v = np.sum([wi ** 2 * v[1] for wi, v in zip(w, lst)], axis=0)
        pooled[(q, meas)] = (m, v, int(np.sum([v2[2] for v2 in lst])))
    return pooled


def subject_deltas(sessions, p):
    """Pool a subject's sessions, then difference. -> {(contrast, meas): tuple}.

    Pooled before differenced: differencing per session and then averaging would weight a
    session with few beats as heavily as a long one; pooling first lets the beat counts speak.
    """
    pooled = pool_periods(sessions, p)
    out = {}
    for tag, a, b, _label in contrasts_of(p):
        for meas in p['MEASURES']:
            va, vb = pooled.get((a, meas)), pooled.get((b, meas))
            if va is None or vb is None:
                continue
            out[(tag, meas)] = (vb[0] - va[0], va[1] + vb[1],
                                int(min(va[2], vb[2])))
    return out


def _save_period_maps(path, subjects, times, p, ctag, label):
    """The per-subject delta maps for one contrast, as an .npz.

    These make the results drawable (a t-map, a per-period profile, a check of a cluster by eye)
    without re-running the cohort. Keys mirror the depth analysis's `run_results.npz` so one
    reader serves both: `subject_ids`, `times`, `freqs`, and `delta_<meas>` / `svar_<meas>`
    [nSub x (nFreq x) nTime], plus `nbeats_<meas>`.

    HEP is 2-D and HRSP/HEPC are 3-D: HEP is a time course, the other two are time-frequency. A
    reader must branch on `ndim`; the shapes are not padded to a common rank.
    """
    out = {'subject_ids': np.asarray([str(s['subject']) for s in subjects]),
           'times': np.asarray(times, float),
           'freqs': np.asarray(p['HRSP_FREQS'], float),
           'contrast': np.asarray(str(ctag)), 'label': np.asarray(str(label))}
    for meas in p['MEASURES']:
        got = [s.get(meas) for s in subjects]
        if not any(g is not None for g in got):
            continue
        ref = next(g for g in got if g is not None)
        shp = np.shape(ref[0])
        d = np.full((len(got),) + tuple(shp), np.nan, float)
        v = np.full((len(got),) + tuple(shp), np.nan, float)
        nb = np.zeros(len(got), float)
        for i, g in enumerate(got):
            if g is None:
                continue
            d[i] = g[0]
            v[i] = g[1]
            nb[i] = g[2]
        out['delta_%s' % meas] = d
        out['svar_%s' % meas] = v
        out['nbeats_%s' % meas] = nb
    np.savez_compressed(path, **out)
    return path


def _save_period_means(path, per_period, times, p):
    """Cohort per-period means, [nSub x nPeriod x (nFreq x) nTime], as an .npz.

    A subject with no data for a period gets NaN in that slice rather than
    being dropped, so the subject axis is the same in every measure and a
    reader can count coverage instead of inferring it.
    """
    sids = sorted(per_period)
    periods = list(p['PERIODS'])
    out = {'subject_ids': np.asarray([str(x) for x in sids]),
           'periods': np.asarray(periods),
           'times': np.asarray(times, float),
           'freqs': np.asarray(p['HRSP_FREQS'], float)}
    for meas in p['MEASURES']:
        ref = None
        for sid in sids:
            for q in periods:
                v = per_period[sid].get((q, meas))
                if v is not None:
                    ref = v
                    break
            if ref is not None:
                break
        if ref is None:
            continue
        shp = tuple(np.shape(ref[0]))
        M = np.full((len(sids), len(periods)) + shp, np.nan, float)
        N = np.zeros((len(sids), len(periods)), float)
        for i, sid in enumerate(sids):
            for j, q in enumerate(periods):
                v = per_period[sid].get((q, meas))
                if v is None:
                    continue
                M[i, j] = v[0]
                N[i, j] = v[2]
        out['mean_%s' % meas] = M
        out['nbeats_%s' % meas] = N
    np.savez_compressed(path, **out)
    return path


def run(data_dir, results_root, p=None, limit=None, verbose=True):
    p = dict(CONFIG, **(p or {}))
    if str(p['IC_SELECT_METHOD']) != 'lee':
        raise ValueError("IC_SELECT_METHOD must be 'lee', got %r"
                         % (p['IC_SELECT_METHOD'],))

    from l2_heart import run_l2_heart

    if verbose:
        print('=' * 74)
        print('HEART PERIOD CONTRASTS')
        print('  periods    %s' % ', '.join(p['PERIODS']))
        print('  contrasts  %d   measures %s'
              % (len(contrasts_of(p)), ','.join(p['MEASURES'])))
        print('  selection  %s' % p['IC_SELECT_METHOD'])
        print('=' * 74)
    subs = sorted(d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and (d.startswith('sub-') or d.isdigit()))
    if limit:
        subs = subs[:int(limit)]

    per_sub, per_period, records = {}, {}, []
    for sid in subs:
        sdir = os.path.join(data_dir, sid)
        got = []
        for f in sorted(os.listdir(sdir)):
            if not f.endswith('.npz'):
                continue
            res, notes = session_periods(os.path.join(sdir, f), p,
                                         verbose=verbose)
            rec = dict(subject=sid, session=f,
                       status='kept' if res else 'dropped',
                       reason=notes.get('skip_reason', ''),
                       ics=','.join(map(str, notes.get('ics', []))),
                       periods=','.join(
                           '%s:%d' % (q, res[q][p['MEASURES'][0]][2])
                           for q in res) if res else '')
            records.append(rec)
            if verbose:
                print('  %s %s %s %s' % (sid, f, rec['status'],
                                         rec['reason'] or rec['periods']))
            if res:
                got.append(res)
        if got:
            per_sub[sid] = subject_deltas(got, p)
            # The per-period means, which differencing throws away. They are what a
            # "measure by period" profile is drawn from, and the pooling is a weighted sum
            # over maps that already exist, so no extra sessions are read for it.
            per_period[sid] = pool_periods(got, p)

    # -- L2: the same cluster-permutation stage as the depth analysis --------
    # The folder tag carries the lee knobs, or two runs that differ only in them (e.g. 1IC
    # against 0.8IC) would share one folder and the second would overwrite the first.
    _seltag = 'lee-%s-%s-%gIC' % (p['HRSP_SEL_CRITERION'], p['IC_SEL_NORM'],
                                  float(p['IC_SEL_THRESH']))
    tag = 'periods%d_%s_t%g-%g' % (len(p['PERIODS']), _seltag,
                                   p['TLIMS'][0], p['TLIMS'][1])
    if str(p.get('BASELINE_METHOD', 'none')).lower() == 'glm':
        tag += '_bl'
    elif str(p.get('BASELINE_METHOD', 'none')).lower() == 'subtract':
        tag += '_blsub'
    elif str(p.get('BASELINE_METHOD', 'none')).lower() != 'none':
        raise ValueError('unknown BASELINE_METHOD %r' % p['BASELINE_METHOD'])
    if _wavelet_on(p) == 'block':
        tag += '_cwtblock'
    out_dir = os.path.join(results_root, 'HEART_PERIOD', tag)
    all_rows = []
    times = None
    for ctag, _a, _b, label in contrasts_of(p):
        subjects = []
        for sid, d in per_sub.items():
            entry = {'subject': sid}
            for meas in p['MEASURES']:
                entry[meas] = d.get((ctag, meas))
            if any(entry.get(m) is not None for m in p['MEASURES']):
                subjects.append(entry)
        if len(subjects) < 3:
            if verbose:
                print('  %-24s SKIPPED - %d subject(s) have the pair'
                      % (ctag, len(subjects)))
            continue
        n_t = next(len(s[m][0]) if np.ndim(s[m][0]) == 1 else s[m][0].shape[-1]
                   for s in subjects for m in p['MEASURES']
                   if s.get(m) is not None)
        times = np.linspace(p['TLIMS'][0], p['TLIMS'][1], n_t)
        sub_dir = os.path.join(out_dir, ctag, 'main')
        os.makedirs(sub_dir, exist_ok=True)
        if verbose:
            print('\n  -- %s (n=%d) --' % (label, len(subjects)))
        # run_l2_heart returns (rows, detected), not rows.
        rows, detected = run_l2_heart(
            subjects, times_hep=times, times_tf=times,
            freqs=np.asarray(p['HRSP_FREQS'], float),
            measures=tuple(p['MEASURES']), schemes=tuple(p['SCHEMES']),
            n_perm=int(p['N_PERM']), alpha=float(p['ALPHA']),
            out_csv=os.path.join(sub_dir, 'l2_period_summary.csv'),
            verbose=verbose)
        # Persist the per-subject delta maps, so the run can be drawn (t-map, per-period
        # profile, a check of a cluster by eye) without re-running the cohort.
        _save_period_maps(os.path.join(sub_dir, 'period_maps.npz'),
                          subjects, times, p, ctag, label)
        for r in (rows or []):
            r = dict(r)
            r['Contrast'] = ctag
            r['Label'] = label
            all_rows.append(r)
        if verbose:
            for m, d_ in (detected or {}).items():
                if isinstance(d_, dict) and d_.get('sig'):
                    print('     DETECTED  %-5s under %s' % (m, d_.get('scheme')))

    os.makedirs(out_dir, exist_ok=True)
    # The period means, cohort-wide, in one file. Written even when every contrast is null:
    # "what does HEP look like in each period" is a question about the measure, not about a
    # surviving effect.
    if per_period and times is not None:
        _save_period_means(os.path.join(out_dir, 'period_means.npz'),
                           per_period, times, p)
    import csv
    if all_rows:
        fields = ['Contrast', 'Label'] + [k for k in all_rows[0]
                                          if k not in ('Contrast', 'Label')]
        with io.open(os.path.join(out_dir, 'period_L2_all.csv'), 'w',
                     newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
    with io.open(os.path.join(out_dir, 'cohort_sessions.csv'), 'w',
                 newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['subject', 'session', 'status',
                                           'reason', 'ics', 'periods'],
                           extrasaction='ignore')
        w.writeheader()
        for r in records:
            w.writerow(r)
    with io.open(os.path.join(out_dir, 'run_params.json'), 'w',
                 encoding='utf-8') as fh:
        json.dump({k: (list(v) if isinstance(v, (list, tuple)) else v)
                   for k, v in p.items()}, fh, indent=1, sort_keys=True)
    if verbose:
        kept = sum(1 for r in records if r['status'] == 'kept')
        print('\n  %d subject(s), %d session(s) kept, %d dropped'
              % (len(per_sub), kept, len(records) - kept))
        print('  %d contrasts, UNCORRECTED across them (3 df between six)'
              % len(contrasts_of(p)))
        print('  -> %s' % out_dir)
    return {'rows': all_rows, 'records': records, 'out_dir': out_dir}
