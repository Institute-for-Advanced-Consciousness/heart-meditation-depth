# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Autonomic variables as a function of meditation depth.

The covariate models in `heart_depth_pipeline` ask a different question from this one.
There, heart period / ECG amplitude / respiration amplitude enter the level-1 design and
Frisch-Waugh-Lovell removes the part of the depth column each explains within a
recording; the reported quantity is still the neural depth slope. Here the autonomic
variable is the outcome: does heart rate, heart rate variability, respiratory rate or
respiratory amplitude itself move with reported depth?

A covariate can abolish a neural effect without itself tracking depth at the group level,
because level 1 operates within recording, and a within-recording association whose sign
differs between recordings averages to nothing across participants. So both numbers are
emitted: the group slope, and the distribution of within-recording correlations that the
group slope cannot see.

Design, identical to the neural arms so the numbers are comparable:

  level 1   OLS slope of the variable on mean-centred depth, within recording
  level 2   participant mean over their recordings, then the weighted sign-flip
            permutation test (`fmt_depth_pipeline.signflip_ttest`), under the same
            two weighting schemes the paper reports for the neural measures:
            segment-count and unweighted

Nothing is re-extracted. It reads the same `<data>/<sid>/ses<n>.npz` the neural arms
read, applies the same condition / depth / window gates, and takes its R-peaks from the
same detector (`get_rr`, which follows BrainBeats' get_RR).

The gates are the pipeline's, not a local copy. `window_indices` decides which stored
segments actually contain the requested analysis window, and it is imported rather than
reimplemented: a segment can be long enough and still not reach the window, because the
stored segment is anchored at its start (see its docstring).

The 300-2000 ms IBI filter here is not the pipeline's beat gate. The neural arms drop a
beat whose preceding interval is under 600 ms and keep the rest; that rule exists to limit
ectopic contamination of an evoked response. RMSSD is a statistic of the interval series,
so a single implausible interval distorts it directly rather than costing one epoch, and
it is computed here on intervals inside a physiological range instead.

`main(argv)` is called by run_autonomic.py with `--data <segments folder>` and
`--out <output folder>`; `--set KEY=VAL` overrides one pipeline parameter.

Writes into `--out`:
    autonomic_depth_group.csv       one row per variable x scheme
    autonomic_depth_within.csv      one row per recording x variable
    autonomic_depth_segments.csv    one row per analysed segment
"""
import argparse
import ast
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heart_depth_pipeline as HDP                               # noqa: E402
from get_rr import get_rr, resolve_polarity, GetRRError      # noqa: E402
from fmt_depth_pipeline import signflip_ttest                    # noqa: E402

# The four variables, in the order the paper reports them. `resp_amp` is the
# same quantity as the `respAmp` covariate (mean of the 0.1-1 Hz Hilbert
# envelope over the analysis window), so the covariate that abolishes the
# neural effect is tested here on its own terms.
VARIABLES = ('hr_ms', 'rmssd', 'resp_bpm', 'resp_amp')
VAR_LABEL = {
    'hr_ms':    'mean inter-beat interval (ms)',
    'rmssd':    'RMSSD (ms)',
    'resp_bpm': 'respiratory rate (breaths/min)',
    'resp_amp': 'respiration amplitude (a.u.)',
}
# SDNN is not computed. At a 60 s window it is dominated by respiratory sinus
# arrhythmia, the same variance RMSSD indexes (on this dataset the two were highly
# correlated across segments and across recording-level depth slopes), and RMSSD is
# reliable in far shorter windows, while SDNN's agreement with a five-minute reference
# degrades below 60 s. `np.std(ibi, ddof=1)` in `segment_features` would add it.
SCHEMES = ('precision_nseg', 'unweighted')

# Physiological bounds on a single inter-beat interval, in ms. 300 ms is 200
# bpm and 2000 ms is 30 bpm; anything outside is a detection failure, not a
# heartbeat. See the module header for why this differs from the neural arms.
IBI_MIN_MS, IBI_MAX_MS = 300.0, 2000.0
MIN_IBI_PER_SEG = 8          # below this, RMSSD is not estimable
MIN_SEG_PER_REC = 3          # below this, a within-recording slope is noise
N_PERM = 10000


def segment_features(ecg_win, resp_seg, fs, resp_band=(0.1, 1.0),
                     win=(None, None)):
    """Autonomic features for one analysed segment.

    `ecg_win` is the analysis window already sliced and already polarity-
    corrected. `resp_seg` is the full stored respiration segment, because
    `_resp_env_phase` filters at 0.1 Hz and needs the samples either side of the
    window; `win` gives the (w0, w1) slice to read back out of it.

    Returns a dict of the four variables, with NaN where a variable could not be
    computed. Never raises: a bad window costs its segment, not the recording.
    """
    out = dict((v, np.nan) for v in VARIABLES)
    try:
        pk, _ = get_rr(np.asarray(ecg_win, float), fs, polarity=1,
                       drop_first=False)
    except (GetRRError, ValueError):
        pk = np.array([], np.int64)
    if pk.size >= 2:
        ibi = np.diff(np.asarray(pk, float)) / fs * 1000.0
        ibi = ibi[(ibi >= IBI_MIN_MS) & (ibi <= IBI_MAX_MS)]
        if ibi.size >= MIN_IBI_PER_SEG:
            out['hr_ms'] = float(np.mean(ibi))
            out['rmssd'] = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    if resp_seg is not None:
        w0, w1 = win
        # `_resp_env_phase` is the pipeline's own respiration envelope and phase
        # (reflect-padded, single-pass). Imported rather than copied so respAmp
        # here is the same number the covariate uses.
        env, ph = HDP._resp_env_phase(np.asarray(resp_seg, float), fs,
                                      resp_band)
        if env is not None:
            # Respiratory rate as the median instantaneous frequency of the
            # unwrapped Hilbert phase, not from peak counting and not from the
            # phase advanced end to end. Peak counting misses a shallow breath;
            # the end-to-end difference is contaminated by the filter's edge
            # behaviour, which at 0.15 Hz (9 breaths/min, near the 0.1 Hz
            # passband edge) measured 9.9 against a synthetic 9.0. The median of
            # the per-sample phase increments is insensitive to both.
            phw = np.unwrap(np.asarray(ph, float)[w0:w1])
            if phw.size > 1:
                dphi = np.diff(phw)
                dphi = dphi[np.isfinite(dphi)]
                if dphi.size:
                    out['resp_bpm'] = float(np.median(dphi) * fs
                                            / (2 * np.pi) * 60.0)
            out['resp_amp'] = float(np.nanmean(np.asarray(env, float)[w0:w1]))
    return out


def recording_table(npz_path, p):
    """Per-segment features for one recording, under the pipeline's own gates.

    Returns (depth[nSeg], {var: values[nSeg]}) or None when the recording has
    too few usable segments or spans fewer than two depth levels.
    """
    d = np.load(npz_path, allow_pickle=True)
    try:
        fs = float(d['fs'])
        depth = np.asarray(d['seg_depth']).astype(int)
        cond = np.asarray(d['seg_cond']).astype(int)
        seg_len = np.asarray(d['seg_len']).astype(int)
        seg_ecg = d['seg_ecg']
        has_resp = 'seg_resp' in d.files
        seg_resp = d['seg_resp'] if has_resp else None

        # GATE 1: condition and depth set, exactly session_measures' gate 1.
        keep = np.ones(depth.size, bool)
        if not p['INCLUDE_SAPO']:
            keep &= (cond == 0)
        keep &= np.isin(depth, p['DEPTH_LEVELS'])
        # GATE 1b: is the requested window actually stored for this segment?
        w0, w1, w_ok = HDP.window_indices(
            seg_len, fs, p['EPOCH_DUR'], p['PRE_END'],
            p['EXTRACT_EPOCH_DUR'], p['EXTRACT_PRE_END'])
        keep &= w_ok
        idx = np.flatnonzero(keep)
        if idx.size < MIN_SEG_PER_REC or np.unique(depth[idx]).size < 2:
            return None

        # Polarity once per recording, on the concatenated analysed windows. get_rr
        # would otherwise re-vote per 60 s window on ~65 beats and could disagree
        # with itself between two windows of one recording; heart_depth_pipeline
        # does the same.
        cat = np.concatenate([np.asarray(seg_ecg[k, w0:w1], float)
                              for k in idx])
        try:
            pol = int(resolve_polarity(cat, fs))
        except Exception:
            pol = 1

        cols = dict((v, []) for v in VARIABLES)
        dep = []
        for k in idx:
            rs = (np.asarray(seg_resp[k, :int(seg_len[k])], float)
                  if has_resp else None)
            f = segment_features(pol * np.asarray(seg_ecg[k, w0:w1], float),
                                 rs, fs, p.get('RESP_BAND', (0.1, 1.0)),
                                 win=(w0, w1))
            if not np.isfinite(f['hr_ms']):
                continue                      # no usable beat train: no segment
            for v in VARIABLES:
                cols[v].append(f[v])
            dep.append(float(depth[k]))
        if len(dep) < MIN_SEG_PER_REC or np.unique(dep).size < 2:
            return None
        return (np.asarray(dep, float),
                dict((v, np.asarray(cols[v], float)) for v in VARIABLES))
    finally:
        d.close()


def level1_slope(depth, y):
    """OLS slope of `y` on mean-centred depth. NaN when not estimable."""
    ok = np.isfinite(depth) & np.isfinite(y)
    if ok.sum() < MIN_SEG_PER_REC:
        return np.nan
    dc = depth[ok] - depth[ok].mean()
    if np.std(dc) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.polyfit(dc, y[ok], 1)[0])


def within_corr(depth, y):
    """Pearson r between depth and `y` within one recording. NaN if degenerate."""
    ok = np.isfinite(depth) & np.isfinite(y)
    if ok.sum() < 4 or np.std(depth[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(depth[ok], y[ok])[0, 1])


def analyse(data_dir, p, verbose=True):
    """Full arm. Returns (group_rows, within_rows, segment_rows)."""
    per_rec, seg_rows, within_rows = {}, [], []
    for sid, ses, path in HDP.list_sessions(data_dir):
        try:
            r = recording_table(path, p)
        except Exception as exc:                      # noqa: BLE001
            if verbose:
                print('  skip %s ses%s: %s' % (sid, ses, exc))
            continue
        if r is None:
            continue
        depth, cols = r
        per_rec[(sid, int(ses))] = r
        for j, dv in enumerate(depth):
            row = dict(Subject=sid, Session=int(ses), Depth=int(dv))
            row.update((v, cols[v][j]) for v in VARIABLES)
            seg_rows.append(row)
        for v in VARIABLES:
            within_rows.append(dict(
                Subject=sid, Session=int(ses), Variable=v,
                N_segments=int(depth.size),
                Slope=level1_slope(depth, cols[v]),
                r=within_corr(depth, cols[v])))
    if verbose:
        print('recordings analysed: %d   segments: %d'
              % (len(per_rec), len(seg_rows)))

    # ---- level 2: participant mean slope, then the project's sign-flip test --
    subj = {}
    for (sid, _ses), (depth, cols) in per_rec.items():
        for v in VARIABLES:
            b = level1_slope(depth, cols[v])
            if np.isfinite(b):
                subj.setdefault(v, {}).setdefault(sid, []).append(
                    (b, int(depth.size)))
    group_rows = []
    for v in VARIABLES:
        if v not in subj:
            continue
        sids = sorted(subj[v])
        betas = np.array([np.mean([b for b, _n in subj[v][s]]) for s in sids])
        nseg = np.array([np.sum([n for _b, n in subj[v][s]]) for s in sids],
                        float)
        for scheme in SCHEMES:
            w = nseg.copy() if scheme == 'precision_nseg' else np.ones_like(nseg)
            t, pval = signflip_ttest(betas, w, n_perm=N_PERM)
            # The reported mean is the one the test used: signflip_ttest takes the
            # weighted mean, so printing the unweighted mean beside a weighted t
            # would give two rows with an identical effect and different statistics.
            wm = float(np.sum(betas * w) / np.sum(w))
            sd = float(np.sqrt(np.sum(w * (betas - wm) ** 2) / np.sum(w)))
            group_rows.append(dict(
                Variable=v, Label=VAR_LABEL[v], Scheme=scheme,
                N_subjects=len(sids), Mean_slope=wm, SD_slope=sd,
                t=float(t), p_perm=float(pval),
                Cohens_d=float(wm / sd) if sd > 0 else np.nan))
    return group_rows, within_rows, seg_rows


def _write(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    return path


def _summarise(group_rows, within_rows):
    print('\n== group: autonomic variable ~ depth ==')
    print('%-9s %-15s %4s %12s %7s %8s %7s'
          % ('var', 'scheme', 'n', 'mean slope', 't', 'p_perm', 'd'))
    for r in group_rows:
        print('%-9s %-15s %4d %12.4f %7.2f %8.4f %7.2f'
              % (r['Variable'], r['Scheme'], r['N_subjects'], r['Mean_slope'],
                 r['t'], r['p_perm'], r['Cohens_d']))
    print('\n== within-recording coupling (what a level-1 covariate removes) ==')
    print('%-9s %5s %8s %9s %8s %8s'
          % ('var', 'nRec', 'mean r', 'med |r|', '|r|>.3', '|r|>.5'))
    for v in VARIABLES:
        rs = np.array([r['r'] for r in within_rows
                       if r['Variable'] == v and np.isfinite(r['r'])])
        if rs.size == 0:
            continue
        a = np.abs(rs)
        print('%-9s %5d %8.3f %9.3f %7.0f%% %7.0f%%'
              % (v, rs.size, rs.mean(), np.median(a),
                 100 * np.mean(a > .3), 100 * np.mean(a > .5)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--data', default=os.environ.get(
        'HMD_DATA', os.path.join(here, '..', 'data', 'segments')))
    ap.add_argument('--out', default=os.path.join(here, '..', 'results',
                                                  'AUTONOMIC'))
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VAL',
                    help='override a pipeline parameter, e.g. EPOCH_DUR=45')
    a = ap.parse_args(argv)

    p = dict(HDP.DEFAULT_PARAMS)
    # The paper's primary configuration. Stated here rather than inherited so a
    # change to DEFAULT_PARAMS cannot silently move this arm off it.
    p.update(EPOCH_DUR=60, INCLUDE_SAPO=False)
    for kv in a.set:
        k, _, val = kv.partition('=')
        if k not in p:
            ap.error('unknown parameter %r' % k)
        try:                      # 'False' must become False, '[4, 20]' a list
            p[k] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            p[k] = val

    os.makedirs(a.out, exist_ok=True)
    grp, wit, seg = analyse(os.path.abspath(a.data), p)
    _write(os.path.join(a.out, 'autonomic_depth_group.csv'), grp,
           ['Variable', 'Label', 'Scheme', 'N_subjects', 'Mean_slope',
            'SD_slope', 't', 'p_perm', 'Cohens_d'])
    _write(os.path.join(a.out, 'autonomic_depth_within.csv'), wit,
           ['Subject', 'Session', 'Variable', 'N_segments', 'Slope', 'r'])
    _write(os.path.join(a.out, 'autonomic_depth_segments.csv'), seg,
           ['Subject', 'Session', 'Depth'] + list(VARIABLES))
    _summarise(grp, wit)
    print('\nwrote 3 CSVs to %s' % os.path.abspath(a.out))
    return 0
