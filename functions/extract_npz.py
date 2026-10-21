# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Extract data segments from the preprocessed EEGLAB .set files into one compressed .npz per
session.

Reads each session's .set file and extracts:
  - depth-report segments (up to 90 s, ending PRE_END = 2 s before each report, confidence
    >= 4) with scalp EEG, ECG and respiration;
  - period windows (baseline, pre-meditation, post-meditation, active-oddball) with scalp EEG,
    ECG and respiration;
  - the ICA decomposition (icaweights, icasphere, icawinv, icachansind), so that IC activations
    can be reconstructed from the scalp data by matrix multiplication;
  - channel positions, segment metadata, and the meditation-block spans that scope R-peak
    detection;
  - the ICLabel class probabilities of the components, all recordings in one file.

Usage:
    python extract_npz.py [--data <folder of .set files>] [--out <output folder>] [--force]
    python extract_npz.py [--data ...] [--out ...] --iclabels-only

Output:
    <out>/<subject>/ses<N>.npz  (about 125 MB each)
    <out>/ic_labels.npz         ('<subject>_ses<N>|iclabel' -> [nIC x 7])
"""
import os
import re
import sys
import time
import numpy as np
import warnings
from pymatreader import read_mat
import mne

warnings.filterwarnings('ignore')

# =====================================================================
# PARAMETERS
# =====================================================================
# The preprocessed recordings (preprocessing/process_raw_data.m) and the segment files;
# override with --data / --out, or with HMD_PREPROCESSED / HMD_DATA.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get('HMD_PREPROCESSED', os.path.join(_ROOT, 'data', 'preprocessed'))
OUTPUT_DIR = os.environ.get('HMD_DATA', os.path.join(_ROOT, 'data', 'segments'))

# Epoch parameters (extract at 90s so downstream can sub-select 30/45/60s)
EPOCH_DUR = 90       # seconds, maximum epoch length to extract
PRE_END = 2          # seconds excluded before depth report
MIN_CONF = 4         # minimum confidence rating
DUR_TOL_SEC = 3      # tolerance for short epochs

# Overlap resolution is left to the analysis (see the note in extract_epochs): resolving at the
# extraction duration (90 s) would discard data that a 60 s analysis needs, so the analysis
# resolves at its own EPOCH_DUR + PRE_END. True trims each segment here so that it does not
# overlap the next one at the extraction duration.
RESOLVE_OVERLAPS = False

# Period parameters
PERIOD_TRIM_EDGE_S = 1   # trim from each period span edge
PERIOD_WIN_DUR = 4        # non-overlapping window duration for period analysis

# Which periods to extract
PERIOD_NAMES = ['baseline', 'pre-meditation', 'post-meditation', 'active-oddball']

# The meditation blocks are exported as continuous spans (not fixed windows), so a consumer can
# restrict R-peak detection to them instead of running over the entire ~104-min file (rest,
# warm-up, oddball and cool-down included, of which the analysed segments are only ~16%). A
# block is ~35 min: ample RR context for clean_rr while excluding the noisy non-analysed
# stretches.
#
# The list of block bases is derived inside _rpeak_scope_windows from `include_sapo`, so that a
# list and a flag cannot disagree about which blocks are in scope. Only two of the four bases
# exist in any one recording (the noSAPO stretch is entirely one task block and the SAPO
# stretch the other), and almost every participant flips the pairing across their two
# days.


# =====================================================================
# HELPERS
# =====================================================================

def _parse_depth_value(desc):
    m = re.search(r'depth-(\d+)', desc)
    return int(m.group(1)) if m else None


def find_session_set(data_path, subject_id, session_num):
    fname = f'{subject_id}_ses{session_num}.set'
    p = os.path.join(data_path, fname)
    if os.path.exists(p):
        return data_path, fname
    ses_path = os.path.join(data_path, subject_id, str(session_num))
    p = os.path.join(ses_path, fname)
    if os.path.exists(p):
        return ses_path, fname
    return None, None


def get_subject_list(data_path):
    subjects = []
    for d in sorted(os.listdir(data_path)):
        full = os.path.join(data_path, d)
        if os.path.isdir(full) and re.match(r'^(\d+|sub-[A-Za-z0-9]+)$', d):
            subjects.append(d)
    return subjects


def _get_block_windows(annotations, sfreq, block_base):
    """Find [start_sample, end_sample] windows for block markers."""
    descs = [str(d).lower().strip() for d in annotations.description]
    samples = np.round(annotations.onset * sfreq).astype(int)

    beg_labels = [f'{block_base}-begin', f'block-{block_base}-begin']
    end_labels = [f'{block_base}-end', f'block-{block_base}-end']

    beg_idx, end_idx = [], []
    for bl, el in zip(beg_labels, end_labels):
        bi = [i for i, d in enumerate(descs) if d == bl]
        ei = [i for i, d in enumerate(descs) if d == el]
        if bi and ei:
            beg_idx, end_idx = bi, ei
            break

    n_pairs = min(len(beg_idx), len(end_idx))
    windows = np.zeros((n_pairs, 2), dtype=int)
    for k in range(n_pairs):
        windows[k, 0] = samples[beg_idx[k]]
        windows[k, 1] = samples[end_idx[k]]
    return windows


def _block_marker_latencies(descs, samples, block_base):
    """Every begin/end sample for a base, under either naming form, matched or not.

    _get_block_windows above deliberately returns nothing for an unmatched pair (a lone marker
    is not a window). This is how the surviving half is recovered.
    """
    beg, end = [], []
    for pref in ('', 'block-'):
        beg += [int(samples[i]) for i, d in enumerate(descs) if d == f'{pref}{block_base}-begin']
        end += [int(samples[i]) for i, d in enumerate(descs) if d == f'{pref}{block_base}-end']
    return sorted(set(beg)), sorted(set(end))


def _rpeak_scope_windows(annotations, sfreq, n_pts, include_sapo=True,
                         include_oddball=False, epoch_dur=EPOCH_DUR,
                         pre_end=PRE_END, guard_s=5.0):
    """Sample spans that R-peak detection should run inside.

    Per condition base with at least one depth report in this recording:

        lo = min( block-begin marker ,  first depth report - (epoch_dur+pre_end) )
        hi = max( block-end   marker ,  last  depth report                       )

    then padded by guard_s each side and clipped to [0, n_pts-1].

    The depth-report term does two jobs. Coverage: a segment reaches epoch_dur+pre_end back
    from its report, and the first report of a block can arrive seconds after block-begin; with
    the block markers alone, a sizeable share of the analysed segments lie partly outside the
    detection window. Recovery: many sessions lack a matched begin/end pair for one or both
    blocks, and the reports still give a span where a marker is missing.

    Do not reconstruct a missing edge from block-<base>-duration-<sec>: on the blocks that carry
    begin, end and duration markers it disagrees with the measured span by seconds to minutes.

    Returns a list of dicts: {'span': (s0, s1), 'base': str, 'source': str}, where source is
    'markers' | 'halfmarker' | 'reports'. Spans are per base and not merged, so they stay
    aligned with their names; overlap is checked and warned about instead (it does not occur on
    this cohort at epoch_dur=60 or 90).
    """
    descs = [str(d).lower().strip() for d in annotations.description]
    samples = np.round(np.asarray(annotations.onset) * sfreq).astype(np.int64)

    reach = int(round((epoch_dur + pre_end) * sfreq))
    guard = int(round(guard_s * sfreq))

    bases = ['nosapo-emerge', 'nosapo-probe']
    if include_sapo:
        bases += ['sapo-emerge', 'sapo-probe']

    out = []
    for base in bases:
        # Every in-meditation depth marker carries its base in the label once the older
        # format has been normalised (see normalise_rating_labels).
        rat = [int(samples[i]) for i, d in enumerate(descs)
               if d.startswith(f'rating-{base}-') and 'meditation_depth-' in d]
        if not rat:
            continue                     # base not run in this recording

        lo, hi = min(rat) - reach, max(rat)
        beg, end = _block_marker_latencies(descs, samples, base)
        if beg and end:
            lo, hi, source = min(lo, beg[0]), max(hi, end[-1]), 'markers'
        elif beg or end:
            if beg:
                lo = min(lo, beg[0])
            if end:
                hi = max(hi, end[-1])
            source = 'halfmarker'
        else:
            source = 'reports'

        s0 = max(0, lo - guard)
        s1 = min(int(n_pts) - 1, hi + guard)
        if s1 > s0:
            out.append({'span': (s0, s1), 'base': base, 'source': source})

    if include_oddball:
        b0 = [int(samples[i]) for i, d in enumerate(descs) if d == 'active-oddball-begin']
        e0 = [int(samples[i]) for i, d in enumerate(descs) if d == 'active-oddball-end']
        if b0 and e0:
            s0, s1 = max(0, b0[0] - guard), min(int(n_pts) - 1, e0[0] + guard)
            if s1 > s0:
                out.append({'span': (s0, s1), 'base': 'active-oddball',
                            'source': 'markers'})

    out.sort(key=lambda r: r['span'][0])
    for a, b in zip(out, out[1:]):
        if b['span'][0] <= a['span'][1]:
            print(f"    WARNING: scope spans overlap ({a['base']} / {b['base']}); "
                  f"a consumer that detects per span will count those beats twice")
    return out


# -- OLDER RATING MARKER FORMAT (two participants) ----------------------------
# Two participants were recorded with a task build that wrote depth reports without the block
# name in the label:
#
#     rating-depth-N              depth, inside an EMERGE report window
#     rating-meditation_depth-N   depth, inside a PROBE prompt window
#     rating-confidence_depth-N   the paired confidence report, in either
#
# Every other recording writes rating-<base>-meditation_depth-N, and this file selects on that
# base (_rpeak_scope_windows keys on `rating-{base}-`, and extract_epochs reads the condition
# out of the label). Without normalisation these recordings produce no R-peak scope spans and
# hence no segments.
#
# The block markers themselves are fine: the inner report/prompt windows are named
# emerge-<base>-begin / probe-<base>-begin with the base present, as everywhere else. A
# base-less rating therefore belongs to the block of the most recent such begin marker, and
# normalise_rating_labels rewrites it into the canonical form so every selector below sees one
# vocabulary. Checked against the `block` column of the BIDS events.tsv, which the BIDS
# conversion derived independently: 222/222 base-less markers in the 4 recordings get the
# block the conversion assigned, across all seven bases (nosapo/sapo x emerge/probe, oddball,
# pre-meditation, post-meditation).
#
# No-op for every other recording: the function returns the object it was handed unless a
# base-less label is present, which happens in 4 of the 78 recordings (the two participants
# above).
_BASELESS_RATING = re.compile(
    r'^rating-(depth|meditation_depth|confidence_depth)-(\d+)$')
_WINDOW_BEGIN = re.compile(r'^(?:emerge|probe)-(.+)-begin$')


def normalise_rating_labels(annotations):
    """Give base-less rating markers the base of the window they sit in."""
    descs = [str(d).strip().lower() for d in annotations.description]
    if not any(_BASELESS_RATING.match(d) for d in descs):
        return annotations                      # every other recording

    order = np.argsort(np.asarray(annotations.onset), kind='stable')
    new_desc = list(annotations.description)
    base = None
    n_fixed, n_orphan = 0, 0
    for i in order:
        d = descs[i]
        m = _WINDOW_BEGIN.match(d)
        if m:
            base = m.group(1)
            continue
        m = _BASELESS_RATING.match(d)
        if not m:
            continue
        if base is None:
            n_orphan += 1                       # no window has opened yet
            continue
        kind = 'confidence_depth' if m.group(1) == 'confidence_depth' \
            else 'meditation_depth'
        new_desc[i] = f'rating-{base}-{kind}-{m.group(2)}'
        n_fixed += 1

    print(f'    Older rating format: {n_fixed} marker(s) given their block name'
          + (f', {n_orphan} before any window marker left as-is' if n_orphan else ''))
    return mne.Annotations(onset=annotations.onset.copy(),
                           duration=annotations.duration.copy(),
                           description=new_desc)


def filter_depth_ratings(annotations, sfreq, conf_min=4, max_pair_lag=9):
    """Confidence filtering of the depth reports.

    Drops depth-0 reports, caps depth and confidence values at 5, and keeps a depth report only
    when the next confidence report follows within max_pair_lag events and is >= conf_min. The
    confidence markers themselves are removed.
    """
    descs = [str(d).lower().strip() for d in annotations.description]
    onsets = annotations.onset.copy()
    durations = annotations.duration.copy()
    n = len(descs)

    is_depth0 = np.array(['meditation_depth-0' in d or 'rating-depth-0' in d for d in descs])
    keep = ~is_depth0

    for i in np.where(keep)[0]:
        for pat in [r'(.+meditation_depth-)(\d+)$', r'(.+confidence_depth-)(\d+)$']:
            m = re.match(pat, descs[i])
            if m and int(m.group(2)) > 5:
                descs[i] = m.group(1) + '5'

    is_depth = np.array([('meditation_depth-' in d or 'rating-depth-' in d) and keep[i]
                         for i, d in enumerate(descs)])
    is_conf = np.array(['confidence_depth-' in d and keep[i]
                        for i, d in enumerate(descs)])

    depth_indices = np.where(is_depth)[0]
    conf_indices = np.where(is_conf)[0]

    if len(depth_indices) == 0 or len(conf_indices) == 0:
        keep_mask = keep.copy()
    else:
        conf_vals = np.full(len(conf_indices), np.nan)
        for j, ci in enumerate(conf_indices):
            v = _parse_depth_value(descs[ci])
            if v is not None:
                conf_vals[j] = v

        to_remove = np.zeros(n, dtype=bool)
        for di in depth_indices:
            after = conf_indices[conf_indices > di]
            if len(after) == 0:
                to_remove[di] = True
                continue
            next_ci = after[0]
            if (next_ci - di) > max_pair_lag:
                to_remove[di] = True
                continue
            pos = np.where(conf_indices == next_ci)[0][0]
            if np.isnan(conf_vals[pos]) or conf_vals[pos] < conf_min:
                to_remove[di] = True
        to_remove[conf_indices] = True
        keep_mask = keep & ~to_remove

    kept_idx = np.where(keep_mask)[0]
    return mne.Annotations(
        onset=onsets[kept_idx],
        duration=durations[kept_idx],
        description=[descs[i] for i in kept_idx],
    )


def extract_epochs(annotations, sfreq, n_total_samples, condition,
                   pre_end=2, epoch_dur=90, dur_tol_sec=3,
                   resolve_overlaps=RESOLVE_OVERLAPS):
    """Extract fixed-duration epochs ending pre_end sec before each depth report."""
    descs = [str(d).lower().strip() for d in annotations.description]
    samples = np.round(annotations.onset * sfreq).astype(int)

    is_depth = np.array([('meditation_depth-' in d or 'rating-depth-' in d) for d in descs])

    if condition in ('nosapo', 'nosapo-emerge', 'nosapo-probe',
                     'sapo', 'sapo-emerge', 'sapo-probe'):
        is_depth = is_depth & \
            ~np.array(['pre-meditation' in d for d in descs]) & \
            ~np.array(['post-meditation' in d for d in descs]) & \
            ~np.array(['oddball' in d for d in descs])
    elif condition == 'pre-meditation':
        is_depth = is_depth & \
            ~np.array(['post-meditation' in d for d in descs]) & \
            ~np.array(['oddball' in d for d in descs])
    elif condition == 'post-meditation':
        is_depth = is_depth & \
            ~np.array(['pre-meditation' in d for d in descs]) & \
            ~np.array(['oddball' in d for d in descs])

    ix_all = np.where(is_depth)[0]
    if len(ix_all) == 0:
        return {'epochs_samp': np.empty((0, 2), dtype=int),
                'depth': np.empty(0), 'subcond': [], 'n': 0}

    # Build block windows for classification
    windows = {}
    for wb in ['nosapo-emerge', 'nosapo-probe', 'sapo-emerge', 'sapo-probe',
               'pre-meditation', 'post-meditation']:
        windows[wb.replace('-', '_')] = _get_block_windows(annotations, sfreq, wb)

    cond_of = [''] * len(ix_all)
    subcond_of = [''] * len(ix_all)

    for i, ev_idx in enumerate(ix_all):
        lab = descs[ev_idx]
        lat = samples[ev_idx]
        if 'nosapo' in lab: cond_of[i] = 'nosapo'
        elif 'sapo' in lab: cond_of[i] = 'sapo'
        elif 'pre-meditation' in lab: cond_of[i] = 'pre-meditation'
        elif 'post-meditation' in lab: cond_of[i] = 'post-meditation'
        if 'emerge' in lab: subcond_of[i] = 'emerge'
        elif 'probe' in lab: subcond_of[i] = 'probe'
        if cond_of[i] == '' or subcond_of[i] == '':
            for wb_key, c, sc in [
                ('nosapo_emerge', 'nosapo', 'emerge'),
                ('nosapo_probe', 'nosapo', 'probe'),
                ('sapo_emerge', 'sapo', 'emerge'),
                ('sapo_probe', 'sapo', 'probe'),
                ('pre_meditation', 'pre-meditation', 'continuous'),
                ('post_meditation', 'post-meditation', 'continuous'),
            ]:
                w = windows.get(wb_key, np.empty((0, 2)))
                if len(w) > 0 and np.any((lat >= w[:, 0]) & (lat <= w[:, 1])):
                    if cond_of[i] == '': cond_of[i] = c
                    if subcond_of[i] == '': subcond_of[i] = sc
                    break
        if cond_of[i] == '': cond_of[i] = 'unknown'
        if subcond_of[i] == '': subcond_of[i] = 'unknown'

    keep_mask = np.zeros(len(ix_all), dtype=bool)
    for i in range(len(ix_all)):
        c, sc = cond_of[i], subcond_of[i]
        if condition == 'nosapo':
            keep_mask[i] = (c == 'nosapo') and sc in ('emerge', 'probe')
        elif condition == 'sapo':
            keep_mask[i] = (c == 'sapo') and sc in ('emerge', 'probe')
        elif condition == 'pre-meditation':
            keep_mask[i] = (c == 'pre-meditation')
        elif condition == 'post-meditation':
            keep_mask[i] = (c == 'post-meditation')

    ix = ix_all[keep_mask]
    subconds = [subcond_of[i] for i in np.where(keep_mask)[0]]
    if len(ix) == 0:
        return {'epochs_samp': np.empty((0, 2), dtype=int),
                'depth': np.empty(0), 'subcond': [], 'n': 0}

    depth = np.array([_parse_depth_value(descs[ev]) or np.nan for ev in ix])

    epochs_sec = np.full((len(ix), 2), np.nan)
    for i in range(len(ix)):
        report_sec = samples[ix[i]] / sfreq
        epoch_end = report_sec - pre_end
        epoch_start = max(0, epoch_end - epoch_dur)
        epochs_sec[i, :] = [epoch_start, epoch_end]

    epochs_samp = np.round(epochs_sec * sfreq).astype(int)
    epochs_samp[:, 0] = np.maximum(0, epochs_samp[:, 0])
    epochs_samp[:, 1] = np.minimum(n_total_samples - 1, epochs_samp[:, 1])

    bad = np.isnan(depth) | np.any(np.isnan(epochs_samp), axis=1) | \
          (epochs_samp[:, 1] <= epochs_samp[:, 0])
    good = ~bad
    epochs_samp = epochs_samp[good]
    depth = depth[good]
    subconds = [s for s, g in zip(subconds, good) if g]

    # -- Overlap resolution, off by default ----------------------------------
    # Resolving here would throw away data the analysis needs. This function cuts EPOCH_DUR
    # (90 s) before each report, but the analysis uses only its own EPOCH_DUR + PRE_END. Two
    # reports 70 s apart overlap at 90 s but not at 60 s, so trimming here would discard
    # seconds that a 60 s analysis is entitled to.
    #
    # The stored segments may therefore overlap, and the analysis must resolve them at its own
    # duration. `overlap_resolved` is written into every .npz so a consumer can tell which
    # convention it is holding: a reader that does not resolve must refuse a file marked False
    # rather than double-count samples shared by two adjacent segments.
    if resolve_overlaps:
        for i in range(len(epochs_samp) - 1):
            if epochs_samp[i, 1] >= epochs_samp[i + 1, 0]:
                epochs_samp[i, 1] = epochs_samp[i + 1, 0] - 1

    # Drop short epochs (keep more lenient since we want 90s max but 30s min)
    dur_sec = (epochs_samp[:, 1] - epochs_samp[:, 0] + 1) / sfreq
    short = dur_sec < 30 - dur_tol_sec  # keep anything >= 27s (usable for 30s analysis)
    good = ~short
    epochs_samp = epochs_samp[good]
    depth = depth[good]
    subconds = [s for s, g in zip(subconds, good) if g]

    return {'epochs_samp': epochs_samp, 'depth': depth,
            'subcond': subconds, 'n': len(epochs_samp)}


def extract_period_windows(annotations, sfreq, n_total_samples, period_name,
                           win_dur=4, trim_edge_s=1):
    """Extract non-overlapping windows from a period span (baseline, pre-med, etc.).
    Handles periods where only the end marker exists (e.g. baseline has no 'begin')."""
    descs = [str(d).lower().strip() for d in annotations.description]
    samples = np.round(annotations.onset * sfreq).astype(int)

    # Try multiple begin/end patterns
    beg_labels = [f'{period_name}-begin', f'block-{period_name}-begin']
    end_labels = [f'{period_name}-end', f'block-{period_name}-end']

    beg_samples, end_samples = [], []
    for bl in beg_labels:
        bi = [samples[i] for i, d in enumerate(descs) if d == bl]
        if bi:
            beg_samples = bi
            break
    for el in end_labels:
        ei = [samples[i] for i, d in enumerate(descs) if d == el]
        if ei:
            end_samples = ei
            break

    # Fallback for baseline (no begin marker: starts at task-begin or recording start)
    if not beg_samples and period_name == 'baseline':
        task_beg = [samples[i] for i, d in enumerate(descs) if d == 'task-begin']
        if task_beg:
            beg_samples = task_beg
        else:
            beg_samples = [0]  # recording start

    if not beg_samples or not end_samples:
        return np.empty((0, 2), dtype=int)

    n_pairs = min(len(beg_samples), len(end_samples))
    windows = np.zeros((n_pairs, 2), dtype=int)
    for k in range(n_pairs):
        windows[k, 0] = beg_samples[k]
        windows[k, 1] = end_samples[k]

    trim = int(trim_edge_s * sfreq)
    win_samp = int(win_dur * sfreq)

    all_windows = []
    for w in windows:
        s0 = max(0, w[0] + trim)
        s1 = min(n_total_samples - 1, w[1] - trim)
        pos = s0
        while pos + win_samp <= s1 + 1:
            all_windows.append([pos, pos + win_samp - 1])
            pos += win_samp

    return np.array(all_windows, dtype=int) if all_windows else np.empty((0, 2), dtype=int)


# =====================================================================
# MAIN EXTRACTION
# =====================================================================

def extract_session(data_path, output_dir, subject_id, session_num):
    """Extract one session's data to a compressed .npz file."""
    ses_path, fname = find_session_set(data_path, subject_id, session_num)
    if ses_path is None:
        return False

    filepath = os.path.join(ses_path, fname)
    t0 = time.time()

    # --- Load EEG via MNE ---
    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose='ERROR')
    to_drop = [ch for ch in ['ECG', 'RESP', 'EKG'] if ch in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)

    fs = int(raw.info['sfreq'])
    n_pts = raw.n_times
    ch_names = raw.ch_names
    n_chan = len(ch_names)

    # Get channel positions
    montage = raw.get_montage()
    if montage is not None:
        pos_dict = montage.get_positions()['ch_pos']
        ch_positions = np.array([pos_dict[ch] for ch in ch_names])
    else:
        ch_positions = np.array([ch['loc'][:3] for ch in raw.info['chs']])

    # --- Load ECG/RESP via pymatreader (MNE doesn't read these) ---
    try:
        mat = read_mat(filepath, variable_names=['ECG', 'RESP'])
        ecg_data = np.array(mat['ECG']['data']).ravel() if 'ECG' in mat else np.full(n_pts, np.nan)
        resp_data = np.array(mat['RESP']['data']).ravel() if 'RESP' in mat else np.full(n_pts, np.nan)
    except Exception as e:
        print(f'    WARNING: Could not read ECG/RESP ({e})')
        ecg_data = np.full(n_pts, np.nan)
        resp_data = np.full(n_pts, np.nan)

    # --- Joins left by the removal of ECG-contaminated samples ---
    # EEGLAB marks each join with a 'boundary' event at the half sample between the two
    # pieces; the index stored is the first sample after the join (0-based). The data on
    # either side were not adjacent in time, so filters, wavelets and inter-beat intervals
    # must not be taken across one.
    try:
        ev = read_mat(filepath, variable_names=['event'])['event']
        lat = np.asarray(ev['latency'], float).ravel()
        typ = np.asarray(ev['type'], dtype=object).ravel()
        boundaries = np.unique(np.round(lat[typ == 'boundary'] - 0.5).astype(np.int64))
        boundaries = boundaries[(boundaries > 0) & (boundaries < n_pts)]
    except Exception as e:
        print(f'    WARNING: Could not read boundary events ({e})')
        boundaries = np.array([], np.int64)

    # --- Load ICA weights via pymatreader ---
    # icasphere is required, not optional. EEGLAB reconstructs activations as
    #     ic_act = icaweights * icasphere * data(icachansind,:)
    # runica spheres by default, so icasphere is the whitening matrix, not the identity.
    # Dropping it yields wrong IC activations, hence wrong topographies and a wrong selection.
    try:
        mat_ica = read_mat(filepath, variable_names=['icaweights', 'icasphere',
                                                     'icawinv', 'icachansind'])
        icaweights = np.array(mat_ica['icaweights'])  # [nIC x nChan]
        icasphere = np.array(mat_ica['icasphere'])    # [nChan x nChan] whitening
        icawinv = np.array(mat_ica['icawinv'])        # [nChan x nIC] (mixing matrix)
        icachansind = np.array(mat_ica.get('icachansind', list(range(n_chan)))).ravel().astype(int) - 1
        # Guard the alignment assumption: icachansind indexes the original channel
        # order, and ECG/RESP/EKG were dropped above. If those were real channels the
        # indices have shifted and every reconstruction would be wrong.
        if to_drop and icachansind.max() >= n_chan:
            raise ValueError(
                f'icachansind max {icachansind.max()} >= n_chan {n_chan} after '
                f'dropping {to_drop} - channel indices have shifted. Do not trust '
                f'this export; fix the drop order before re-running.')
    except Exception as e:
        print(f'    WARNING: Could not read ICA weights ({e})')
        icaweights = None
        icasphere = None
        icawinv = None
        icachansind = None

    # --- Filter events ---
    # normalise first: a base-less rating label (two participants only) is invisible
    # to every selector below until it carries its block name. No-op elsewhere.
    annots = normalise_rating_labels(raw.annotations)
    filtered_annots = filter_depth_ratings(annots, fs, conf_min=MIN_CONF)
    raw.set_annotations(filtered_annots)

    # --- Extract depth-report segments (nosapo + sapo) ---
    eeg_data = raw.get_data()  # [nChan x nPts]

    # nosapo epochs
    out_nosapo = extract_epochs(raw.annotations, fs, n_pts, 'nosapo',
                                pre_end=PRE_END, epoch_dur=EPOCH_DUR, dur_tol_sec=DUR_TOL_SEC)
    samp = out_nosapo['epochs_samp']
    dep = out_nosapo['depth']
    sub_c = out_nosapo['subcond']
    cond = np.zeros(len(dep), dtype=int)  # 0 = nosapo

    # sapo epochs
    out_sapo = extract_epochs(raw.annotations, fs, n_pts, 'sapo',
                              pre_end=PRE_END, epoch_dur=EPOCH_DUR, dur_tol_sec=DUR_TOL_SEC)
    if out_sapo['n'] > 0:
        samp = np.vstack([samp, out_sapo['epochs_samp']])
        dep = np.concatenate([dep, out_sapo['depth']])
        sub_c = sub_c + out_sapo['subcond']
        cond = np.concatenate([cond, np.ones(len(out_sapo['depth']), dtype=int)])

    # pre-meditation epochs
    out_pre = extract_epochs(raw.annotations, fs, n_pts, 'pre-meditation',
                             pre_end=PRE_END, epoch_dur=EPOCH_DUR, dur_tol_sec=DUR_TOL_SEC)
    if out_pre['n'] > 0:
        samp = np.vstack([samp, out_pre['epochs_samp']])
        dep = np.concatenate([dep, out_pre['depth']])
        sub_c = sub_c + out_pre['subcond']
        cond = np.concatenate([cond, np.full(len(out_pre['depth']), 2, dtype=int)])  # 2 = pre-med

    # post-meditation epochs
    out_post = extract_epochs(raw.annotations, fs, n_pts, 'post-meditation',
                              pre_end=PRE_END, epoch_dur=EPOCH_DUR, dur_tol_sec=DUR_TOL_SEC)
    if out_post['n'] > 0:
        samp = np.vstack([samp, out_post['epochs_samp']])
        dep = np.concatenate([dep, out_post['depth']])
        sub_c = sub_c + out_post['subcond']
        cond = np.concatenate([cond, np.full(len(out_post['depth']), 3, dtype=int)])  # 3 = post-med

    n_seg = len(samp)
    print(f'    Segments: {n_seg} (nosapo={np.sum(cond==0)}, sapo={np.sum(cond==1)}, '
          f'pre-med={np.sum(cond==2)}, post-med={np.sum(cond==3)})')

    if n_seg == 0:
        print(f'    No segments - skipping session')
        return False

    # Extract EEG data for each segment (variable length -> store as array of arrays)
    seg_eeg = []
    seg_ecg = []
    seg_resp = []
    seg_depth = dep.astype(int)
    seg_cond = cond
    seg_subcond = np.array([1 if 'probe' in s else 0 if 'emerge' in s else -1
                            for s in sub_c], dtype=int)
    seg_dur_sec = (samp[:, 1] - samp[:, 0] + 1) / fs

    for i in range(n_seg):
        s0, s1 = samp[i, 0], samp[i, 1] + 1
        seg_eeg.append(eeg_data[:, s0:s1].astype(np.float32))      # [nChan x nSegSamples]
        seg_ecg.append(ecg_data[s0:s1].astype(np.float32))          # [nSegSamples]
        seg_resp.append(resp_data[s0:s1].astype(np.float32))        # [nSegSamples]

    # --- Extract period windows (baseline, pre-med, post-med, oddball) ---
    period_data = {}
    for pname in PERIOD_NAMES:
        pwins = extract_period_windows(raw.annotations, fs, n_pts, pname,
                                       win_dur=PERIOD_WIN_DUR, trim_edge_s=PERIOD_TRIM_EDGE_S)
        if len(pwins) == 0:
            print(f'    Period {pname}: no windows found')
            period_data[pname] = {'eeg': np.empty((0, n_chan, PERIOD_WIN_DUR * fs), dtype=np.float32),
                                  'ecg': np.empty((0, PERIOD_WIN_DUR * fs), dtype=np.float32),
                                  'resp': np.empty((0, PERIOD_WIN_DUR * fs), dtype=np.float32),
                                  'n_windows': 0}
        else:
            # All windows are fixed length -> stack into 3D array
            win_samp = PERIOD_WIN_DUR * fs
            per_eeg = np.zeros((len(pwins), n_chan, win_samp), dtype=np.float32)
            per_ecg = np.zeros((len(pwins), win_samp), dtype=np.float32)
            per_resp = np.zeros((len(pwins), win_samp), dtype=np.float32)
            for j, (s0, s1) in enumerate(pwins):
                per_eeg[j] = eeg_data[:, s0:s1 + 1].astype(np.float32)
                per_ecg[j] = ecg_data[s0:s1 + 1].astype(np.float32)
                per_resp[j] = resp_data[s0:s1 + 1].astype(np.float32)
            period_data[pname] = {'eeg': per_eeg, 'ecg': per_ecg, 'resp': per_resp,
                                  'n_windows': len(pwins), 'spans': np.asarray(pwins, np.int64)}
            print(f'    Period {pname}: {len(pwins)} windows ({PERIOD_WIN_DUR}s each)')

    # --- Meditation block spans (continuous, for block-restricted detection) ---
    # Built by _rpeak_scope_windows. include_sapo=True on purpose: this is an export, not an
    # analysis. The analysis drops the SAPO block when INCLUDE_SAPO=false, but an .npz written
    # that way could never serve a SAPO run afterwards. block_names/block_sources let a
    # consumer subset instead: filter on the name, do not re-derive spans.
    scope = _rpeak_scope_windows(raw.annotations, fs, n_pts, include_sapo=True,
                                 include_oddball=False, epoch_dur=EPOCH_DUR,
                                 pre_end=PRE_END)
    block_spans = (np.array([r['span'] for r in scope], dtype=np.int64)
                   if scope else np.zeros((0, 2), dtype=np.int64))
    block_names = [r['base'] for r in scope]
    block_sources = [r['source'] for r in scope]
    if len(block_spans):
        tot = (block_spans[:, 1] - block_spans[:, 0]).sum() / fs / 60
        detail = ', '.join(f"{r['base']}[{r['source']}]" for r in scope)
        print(f'    Meditation blocks: {len(block_spans)} span(s), {tot:.1f} min total '
              f'({tot/(n_pts/fs/60)*100:.0f}% of the recording) - {detail}')
    else:
        # Only reachable if the recording holds neither a block marker nor a single depth
        # report, which is an acquisition failure, not a marker-naming one.
        print('    Meditation blocks: NONE FOUND (no block markers AND no depth reports)')

    # --- Save ---
    out_dir = os.path.join(output_dir, subject_id)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'ses{session_num}.npz')

    # Save with compression
    # Variable-length segments: pad to max length and store indices
    # (more efficient than object arrays and works with np.savez_compressed)
    max_seg_len = max(s.shape[1] for s in seg_eeg) if seg_eeg else 0
    n_chan_seg = seg_eeg[0].shape[0] if seg_eeg else n_chan

    # Pad all segments to max_seg_len, store actual lengths
    seg_eeg_padded = np.zeros((n_seg, n_chan_seg, max_seg_len), dtype=np.float32)
    seg_ecg_padded = np.zeros((n_seg, max_seg_len), dtype=np.float32)
    seg_resp_padded = np.zeros((n_seg, max_seg_len), dtype=np.float32)
    seg_len = np.zeros(n_seg, dtype=int)
    for i in range(n_seg):
        sl = seg_eeg[i].shape[1]
        seg_eeg_padded[i, :, :sl] = seg_eeg[i]
        seg_ecg_padded[i, :sl] = seg_ecg[i]
        seg_resp_padded[i, :sl] = seg_resp[i]
        seg_len[i] = sl

    save_dict = {
        'subject_id': subject_id,
        'session': session_num,
        'fs': fs,
        # -- Provenance: the geometry of what is in this file ------------------
        # Written so a consumer never has to assume the extraction settings.
        # `overlap_resolved` matters most: when False the stored segments may overlap and the
        # analysis must resolve them at its own EPOCH_DUR + PRE_END. A reader that cannot
        # resolve must refuse such a file rather than double-count the samples two adjacent
        # segments share.
        'extract_epoch_dur': float(EPOCH_DUR),
        'extract_pre_end': float(PRE_END),
        'overlap_resolved': bool(RESOLVE_OVERLAPS),
        'ch_names': np.array(ch_names, dtype=object),
        'ch_positions': ch_positions.astype(np.float32),
        'n_chan': n_chan,
        # ICA (small). Reconstruct activations as
        #     ic_act = icaweights @ icasphere @ seg_eeg[icachansind]
        # not icaweights @ eeg: icasphere is the whitening matrix and omitting it gives wrong
        # activations. See the read block above.
        'icaweights': icaweights.astype(np.float32) if icaweights is not None else np.array([]),
        'icasphere': icasphere.astype(np.float32) if icasphere is not None else np.array([]),
        'icawinv': icawinv.astype(np.float32) if icawinv is not None else np.array([]),
        'icachansind': icachansind if icachansind is not None else np.array([]),
        # Segments (padded to max length)
        'seg_eeg': seg_eeg_padded,
        'seg_ecg': seg_ecg_padded,
        'seg_resp': seg_resp_padded,
        'seg_len': seg_len,
        'seg_depth': seg_depth,
        'seg_cond': seg_cond,
        'seg_subcond': seg_subcond,
        'seg_dur_sec': seg_dur_sec.astype(np.float32),
        # Absolute [start, end] sample index of each segment in the full recording, needed to
        # map whole-recording R-peak trains onto these segments.
        'seg_span': samp[:n_seg].astype(np.int64),
        'n_seg': n_seg,
        # Continuous meditation-block spans, absolute sample indices: the scope
        # R-peak detection should run inside (see _rpeak_scope_windows). Row i of
        # block_spans belongs to block_names[i]; block_sources[i] records how the
        # span was derived ('markers' | 'halfmarker' | 'reports'), since some
        # sessions have an incomplete begin/end pair somewhere.
        'block_spans': block_spans,
        'block_names': np.array(block_names, dtype=object),
        'block_sources': np.array(block_sources, dtype=object),
        # Joins left by the ECG-contaminated samples removed in preprocessing: index of the
        # first sample after each join, in the same coordinates as seg_span.
        'boundaries': boundaries,
        # Periods (fixed-length windows -> 3D arrays)
    }

    for pname in PERIOD_NAMES:
        pd = period_data[pname]
        save_dict[f'period_{pname}_eeg'] = pd['eeg']
        save_dict[f'period_{pname}_ecg'] = pd['ecg']
        save_dict[f'period_{pname}_resp'] = pd['resp']
        save_dict[f'period_{pname}_nwin'] = pd['n_windows']
        # [start, end] sample of each 4 s window, in the coordinates of `boundaries`
        save_dict[f'period_{pname}_spans'] = pd.get('spans', np.empty((0, 2), np.int64))

    np.savez_compressed(out_file, **save_dict)

    file_mb = os.path.getsize(out_file) / 1e6
    elapsed = time.time() - t0
    print(f'    Saved: {out_file} ({file_mb:.1f} MB) in {elapsed:.0f}s')
    return True


ICLABEL_FILE = 'ic_labels.npz'
ICLABEL_CLASSES = ('Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Channel Noise', 'Other')


def read_iclabel(filepath):
    """ICLabel class probabilities of one preprocessed recording, [nIC x 7], or None.

    The preprocessing (preprocessing/process_raw_data.m) runs ICLabel and removes the muscle
    and eye components; EEGLAB keeps EEG.etc.ic_classification.ICLabel.classifications
    aligned with the components that remain. The rows must match icaweights, one per
    component, or the labels would describe other components.
    """
    mat = read_mat(filepath, variable_names=['etc', 'icaweights'])
    try:
        lab = np.asarray(mat['etc']['ic_classification']['ICLabel']['classifications'], float)
    except (KeyError, TypeError):
        return None
    n_ic = np.atleast_2d(np.asarray(mat['icaweights'])).shape[0]
    return lab if lab.shape == (n_ic, len(ICLABEL_CLASSES)) else None


def write_iclabels(data_path, output_dir, subjects):
    """One file, <output_dir>/ic_labels.npz, keyed '<subject>_ses<N>|iclabel'."""
    out, missing = {}, []
    for subj_id in subjects:
        for ses in [1, 2]:
            ses_path, fname = find_session_set(data_path, subj_id, ses)
            if ses_path is None:
                continue
            lab = read_iclabel(os.path.join(ses_path, fname))
            if lab is None:
                missing.append(f'{subj_id}_ses{ses}')
            else:
                out[f'{subj_id}_ses{ses}|iclabel'] = lab
    np.savez_compressed(os.path.join(output_dir, ICLABEL_FILE),
                        classes=np.array(ICLABEL_CLASSES), **out)
    print(f'ICLabel probabilities: {len(out)} recordings -> {ICLABEL_FILE}'
          + (f'; none readable for {", ".join(missing)}' if missing else ''))


def main():
    # --data and --out override DATA_PATH and OUTPUT_DIR, so the extraction can be run on
    # another copy of the preprocessed data without editing the constants.
    global DATA_PATH, OUTPUT_DIR
    if '--data' in sys.argv:
        DATA_PATH = sys.argv[sys.argv.index('--data') + 1]
    if '--out' in sys.argv:
        OUTPUT_DIR = sys.argv[sys.argv.index('--out') + 1]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subjects = get_subject_list(DATA_PATH)
    n_sub = len(subjects)
    if '--iclabels-only' in sys.argv:
        write_iclabels(DATA_PATH, OUTPUT_DIR, subjects)
        return
    print(f'Extracting data for {n_sub} subjects...')
    print(f'  Data path: {DATA_PATH}')
    print(f'  Output: {OUTPUT_DIR}')
    print(f'  Epoch: {EPOCH_DUR}s, pre_end={PRE_END}s, conf>={MIN_CONF}')
    print()

    t_total = time.time()
    n_done = 0
    n_skipped = 0

    # Resume support: a full pass is ~78 sessions x ~900 MB reads, so an interrupted run must
    # not start over. Existing outputs are skipped unless --force is passed.
    force = '--force' in sys.argv
    n_resumed = 0

    for i_sub, subj_id in enumerate(subjects):
        for ses in [1, 2]:
            out_file = os.path.join(OUTPUT_DIR, subj_id, f'ses{ses}.npz')
            if os.path.exists(out_file) and not force:
                print(f'[{i_sub+1}/{n_sub}] {subj_id} ses{ses}: already extracted '
                      f'({os.path.getsize(out_file)/1e6:.0f} MB) - skip')
                n_resumed += 1
                continue
            print(f'[{i_sub+1}/{n_sub}] {subj_id} ses{ses}:')
            ok = extract_session(DATA_PATH, OUTPUT_DIR, subj_id, ses)
            if ok:
                n_done += 1
            else:
                n_skipped += 1

    elapsed = time.time() - t_total
    total_mb = sum(
        os.path.getsize(os.path.join(root, f))
        for root, dirs, files in os.walk(OUTPUT_DIR)
        for f in files if f.endswith('.npz')
    ) / 1e6

    print(f'\nDone: {n_done} extracted, {n_resumed} already present, '
          f'{n_skipped} unavailable, in {elapsed/60:.1f} min')
    print(f'Total output: {total_mb/1e3:.1f} GB')
    write_iclabels(DATA_PATH, OUTPUT_DIR, subjects)
    print(f'\nOutput folder: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()