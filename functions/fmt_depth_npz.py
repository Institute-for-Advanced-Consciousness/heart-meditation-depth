# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""FMT depth pipeline, npz route: frontal-midline theta as a function of reported depth.

Reads the pre-extracted .npz written by `extract_npz.py` and runs the FMT depth analysis.
Section 3 holds the parameters; Section 9 says why each setting is what it is. The file
is a script and runs when executed (run_fmt_depth.py launches it once per setting).

This file carries its own copy of the pipeline. It imports a few helpers from
`fmt_depth_pipeline`, but the npz loading, the GED build and the L2 chain are implemented
here. The GED loop duplicates `fmt_depth_pipeline.build_ged`; a change to one does not
reach the other, so keep them in step by hand (the duplicated spots say so inline).

Usage:
    python functions/fmt_depth_npz.py
    python functions/fmt_depth_npz.py --set INCLUDE_SAPO=True --set EPOCH_DUR=60
    python functions/fmt_depth_npz.py --data-dir data/segments --results-root results
"""

# Dependencies: mne must be importable (the theta filter is mne.filter.filter_data).

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

"""
=============================================================================
SECTION 2: Set paths + load pipeline module
=============================================================================
"""

# -- COMMAND LINE --------------------------------------------------------
# Defaults reproduce the paper's runs. --set overrides any Section 3 parameter.
import os
import argparse as _argparse

# The repository-local extraction, written by extract_npz.py.
_DEFAULT_DATA_DIR = os.environ.get('HMD_DATA', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'segments'))
_DEFAULT_RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

_ap = _argparse.ArgumentParser(description='FMT depth pipeline, npz route')
_ap.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                 help='folder of <subject>/ses*.npz (default: %(default)s)')
_ap.add_argument('--results-root', default=_DEFAULT_RESULTS_ROOT,
                 help='results ROOT; the run folder is named from PARAMS')
_ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                 dest='overrides',
                 help='override a Section 3 parameter, repeatable')
# The folder tag is built from PARAMS only and does not encode --data-dir, so two runs of
# identical parameters over different segment sets collide and the second overwrites the
# first. `--arm` must be one of result_path.ARMS (anything else is rejected there); to
# separate a run over non-standard segments, use a separate --results-root.
_ap.add_argument('--arm', default='FMT_DEPTH',
                 help='results sub-tree; must be a known arm (default: '
                      '%(default)s). To isolate a non-standard --data-dir use '
                      '--results-root, not this.')
_args = _ap.parse_args()

import matplotlib as _mpl
_mpl.use('Agg')          # no display: this script writes PNGs, never shows them

DATA_DIR = _args.data_dir

# Where to save results. This is the root only: the run's own folder is built from its
# parameters below, so two runs differing in any swept setting cannot overwrite each
# other.
RESULTS_ROOT = _args.results_root

# Pipeline module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmt_depth_pipeline import (zscore_omitnan, fit_glm_wls, signflip_ttest, shrunk_ivw,
                                make_frontal_template, plot_depth_profile)

import os
from result_path import fmt_depth_tag, result_dir, write_readme
from session_length import session_duration_lowerbound_min
print(f'Data: {DATA_DIR}')
print(f'Results root: {RESULTS_ROOT}  (the run folder is named from PARAMS in Section 3)')

# -- FIXED: COHORT + EXTRACTION. Not settings to change between runs. --------
# These describe who is in the analysis and what is on disk. Sweeping them here would
# create a differently named folder holding the same data cut a way it was never cut.
# Section 3 merges them into PARAMS so they still reach the folder tag.
EXTRACT_EPOCH_DUR = 90   # seconds actually written by extract_npz.py
EXTRACT_PRE_END = 2      # seconds before the report the .npz segment ends
EXTRACT_MIN_CONF = 4     # the confidence gate the .npz were cut at        [R1]

COHORT = dict(
    # INCLUDE_SAPO is set in Section 3 (PARAMS), not here: it is the setting switched
    # most often, and it does not belong among the extraction gates that must match the
    # .npz.
    MIN_CONF=EXTRACT_MIN_CONF,   # must equal EXTRACT_MIN_CONF             [R1]
    MIN_DEPTHS=2,
    minSeg=5,
    MIN_SESSION_MIN=45,          # on a seg_span lower bound               [R2]
    DEPTH_LEVELS=list(range(1, 6)),
    LOW_DEPTHS=[1, 2],
    HIGH_DEPTHS=[4, 5],
    DEPTH_MODE='ordinal',
)

if int(COHORT['MIN_CONF']) != int(EXTRACT_MIN_CONF):
    raise ValueError(
        'MIN_CONF=%s but the .npz were extracted at MIN_CONF=%s, and they carry '
        'no per-segment confidence, so this run cannot apply the gate it would be '
        'named after. Re-extract, or put MIN_CONF back to %s.'
        % (COHORT['MIN_CONF'], EXTRACT_MIN_CONF, EXTRACT_MIN_CONF))

"""
=============================================================================
SECTION 3: Parameters

Two GED template modes:

    GED_TEMPLATE_ELEC=None      ROI template: Gaussians on all FMT_ELECS
    GED_TEMPLATE_ELEC=['fz']    single Fz: one Gaussian (Zuure et al. style)
    GED_TEMPLATE_ELEC=['cz']    single Cz

Each writes its own folder (tmpl token in the tag), so the three do not overwrite each
other and can be compared side by side. --set overrides any parameter here.
=============================================================================
"""

PARAMS = dict(
    **COHORT,                # cohort + extraction gates, fixed in Section 2

    # == CONDITION ==========================================================
    # False = the silent meditation blocks only (primary). True pools them with the
    # attention-probe blocks, roughly doubling the segments at the cost of auditory
    # stimulation. At False the `cond` model is not identified and is dropped; the
    # folder tag switches nosapo / nosapo+sapo, so the two do not overwrite each other.
    INCLUDE_SAPO=False,

    # == WINDOW =============================================================
    EPOCH_DUR=60,            # 30 / 45 / 60, sub-select from the 90 s        [R3]
    PRE_END=3,               # s excluded before the report; must be >= 2    [R3]

    # == MEASURE ============================================================
    THETA_BP=[4, 8],
    # Scalp ROI (recorded channels only).
    FMT_ELECS=['fz', 'f1', 'f2', 'f3', 'f4',
               'fc1', 'fc2', 'fc3', 'fc4', 'cz'],
    # GED template centres. None = FMT_ELECS (ROI). ['fz'] / ['cz'] = single. [R5]
    GED_TEMPLATE_ELEC=None,
    GED_SHRINKAGE=0.01,
    GED_R2_THRESH=0.30,
    GED_TOP_N=3,             # how many leading components are SCREENED
    # How many qualifying components are combined (not how many are screened). 1 uses
    # the first component that passes the R2 screen; 3 averages the power time courses
    # of the first three that pass, the GED analogue of averaging several ICs. The
    # folder tag gains `_gedComp3` so the two cannot collide.
    GED_MAX_COMP=1,
    STANDARDIZE_WITHIN_SESSION=True,

    # == OVERLAPS AND JOINS (see heart_depth_pipeline.DEFAULT_PARAMS) =======
    OVERLAP_RULE='drop',     # 'drop' | 'trim' (earlier window kept to the next one's start)
    OVERLAP_MIN_S=30,
    BOUNDARY_GUARD_S=0,      # s of samples excluded each side of a join (0 = none)

    # == STATISTICS =========================================================
    L1_METHOD='WLS',
    nPerm=1000,
    alpha=0.05,
    schemes=['unweighted', 'precision_nseg', 'ivw_shrunk'],
    USE_IVW_SHRUNK=True,
    SIG_PRIORITY=['precision_nseg', 'ivw_shrunk', 'unweighted'],  # ORDER=RULE [R6]
)

# -- --set OVERRIDES, APPLIED BEFORE THE FOLDER TAG ----------------------
# These must be applied before fmt_depth_tag(PARAMS) below, or the run would be
# computed at one setting and named after another.
import ast as _ast
for _kv in _args.overrides:
    _k, _eq, _v = _kv.partition('=')
    if not _eq:
        raise SystemExit('--set needs KEY=VALUE, got %r' % _kv)
    _k = _k.strip()
    if _k not in PARAMS:
        raise SystemExit('--set %s: not a parameter. Known: %s'
                         % (_k, ', '.join(sorted(PARAMS))))
    try:
        _val = _ast.literal_eval(_v)
    except (ValueError, SyntaxError):
        _val = _v
    print('  --set %s: %r -> %r' % (_k, PARAMS[_k], _val))
    PARAMS[_k] = _val

# -- THE RUN'S OWN FOLDER, NAMED FROM ITS PARAMETERS -----------------------
# Every swept setting must reach this name, or the next run overwrites this one. Add a
# swept parameter to result_path.fmt_depth_tag in the same edit. `P = PARAMS` is not
# bound until later, so this block uses PARAMS.                                  [R7]
RUN_TAG = fmt_depth_tag(PARAMS)
ARM = _args.arm
RUN_ROOT = os.path.join(RESULTS_ROOT, ARM, RUN_TAG)
RESULTS_DIR = result_dir(RESULTS_ROOT, ARM, RUN_TAG, 'main')
write_readme(RUN_ROOT, ARM,
             'FMT depth (scalp + GED), npz route', PARAMS)
print('')
print(f'Run folder: FMT_DEPTH/{RUN_TAG}')
print(f'  main -> {RESULTS_DIR}')

print('Parameters:')
for k, v in PARAMS.items():
    print(f'  {k} = {v}')

"""
=============================================================================
SECTION 4: Load all subject data from .npz files
=============================================================================
"""

import os
import numpy as np
import mne
# The theta filter is mne.filter.filter_data (see theta_filter).

def load_npz_session(npz_path):
    """Load a pre-extracted session .npz file."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        'subject_id': str(d['subject_id']),
        'session': int(d['session']),
        'fs': int(d['fs']),
        'ch_names': [str(c) for c in d['ch_names']],
        'ch_positions': d['ch_positions'],
        'n_chan': int(d['n_chan']),
        'icaweights': d['icaweights'],
        'icawinv': d['icawinv'],
        'icachansind': d['icachansind'],
        # Segments (padded arrays)
        'seg_eeg': d['seg_eeg'],      # [nSeg, nChan, maxLen]
        'seg_ecg': d['seg_ecg'],      # [nSeg, maxLen]
        'seg_resp': d['seg_resp'],    # [nSeg, maxLen]
        'seg_len': d['seg_len'],      # [nSeg] actual lengths
        # seg_span is required for overlap resolution: two depth reports closer
        # together than EPOCH_DUR + PRE_END give analysis windows that share samples,
        # and without the absolute bounds there is no way to see it.
        'seg_span': d['seg_span'],    # [nSeg x 2] absolute [start, end] samples
        # the meditation-block spans, for the recording-length lower bound
        'block_spans': (d['block_spans'] if 'block_spans' in d.files
                        else np.empty((0, 2))),
        # joins left by the removed ECG-contaminated samples (None in older files)
        'boundaries': d['boundaries'] if 'boundaries' in d.files else None,
        # Extraction geometry written by the extractor, so a consumer never has to
        # assume it. `None` marks a file without the keys; the caller announces the
        # fallback rather than guessing.
        'extract_epoch_dur': (float(d['extract_epoch_dur'])
                              if 'extract_epoch_dur' in d.files else None),
        'extract_pre_end': (float(d['extract_pre_end'])
                            if 'extract_pre_end' in d.files else None),
        'overlap_resolved': (bool(d['overlap_resolved'])
                             if 'overlap_resolved' in d.files else None),
        'seg_depth': d['seg_depth'],
        'seg_cond': d['seg_cond'],
        'seg_subcond': d['seg_subcond'],
        'seg_dur_sec': d['seg_dur_sec'],
        'n_seg': int(d['n_seg']),
        # Periods
        'period_baseline_eeg': d['period_baseline_eeg'],
        'period_pre_meditation_eeg': d['period_pre-meditation_eeg'],
        'period_post_meditation_eeg': d['period_post-meditation_eeg'],
        'period_active_oddball_eeg': d['period_active-oddball_eeg'],
    }

def list_participants(data_dir):
    """Participant folders of the segment data (BIDS labels, sub-<label>)."""
    subjects = []
    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if os.path.isdir(full) and (d.startswith('sub-') or d.isdigit()):
            subjects.append(d)
    return subjects

# The floor added before log10, in double precision. The .npz store raw.get_data(), in
# volts, and theta power in this cohort is ~3.5e-12 V^2, so a float32 eps (1.1921e-07)
# would be ~24,000x the signal and log10(power + eps) would be log10(eps) almost
# everywhere. `_assert_floor_negligible` below makes that mistake loud (it fires on
# the float32 value).
EPS = np.finfo(float).eps

# The guard tests what matters, whether the floor moves the reported value, rather than
# a percentile ratio standing in for it (on this data a percentile proxy fails about a
# quarter of segments whose reported value the floor moves by at most 1e-4 log10
# units). It recomputes the same statistic with a floor 1000x smaller: if EPS does
# nothing, the two agree; if the measure has collapsed onto log10(EPS), they cannot.
# With float32 eps on volt-scaled data it raises by ~4.7 log10 units.
FLOOR_TOL = 1e-3        # log10 units the floor is allowed to move the segment mean


def _assert_floor_negligible(pow_ts, where):
    """Raise if EPS measurably changes mean(log10(power + EPS))."""
    p = np.asarray(pow_ts, float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return
    got = float(np.mean(np.log10(p + EPS)))
    ref = float(np.mean(np.log10(p + EPS * 1e-3)))
    if not np.isfinite(got - ref) or abs(got - ref) > FLOOR_TOL:
        raise ValueError(
            'log-power floor is NOT negligible in %s: the floor moves the '
            'segment mean by %.4f log10 units (limit %g). floor=%.3e, '
            'median power=%.3e. Either the data are in the wrong units or the '
            'floor is too large, so the measure would be log10(eps) instead of '
            'log10(power).'
            % (where, got - ref, FLOOR_TOL, EPS, float(np.median(p))))


def theta_filter(data, fs, l_freq=5, h_freq=8):
    """Zero-phase FIR bandpass, one pass, MNE / EEGLAB pop_eegfiltnew convention.

    Works on 2D [nChan x nSamples] or 3D [nSeg x nChan x nSamples].

    A firwin design run through `filtfilt` would apply the filter twice and square the
    magnitude response (half-power band 5.21-7.79 Hz against 4.25-8.75 Hz for
    mne/EEGLAB [5, 8]), so this calls the MNE filter, as fmt_depth_pipeline does,
    rather than reimplementing one.
    """
    x = np.asarray(data, dtype=np.float64)
    out = mne.filter.filter_data(
        x, sfreq=float(fs), l_freq=float(l_freq), h_freq=float(h_freq),
        method='fir', phase='zero', fir_design='firwin',
        fir_window='hamming', pad='reflect_limited', verbose='ERROR')
    return out.astype(np.float32)


# ---- Index the subjects (paths only) ----
#
# Sessions are not preloaded: holding them all at once takes over 11 GB. Nothing
# needs two recordings resident at once, because Section 1 reduces each session to a
# handful of per-segment scalars (theta power, depth, block, condition) and never looks
# at the raw EEG again. So the index below holds paths, and the Section 1 loop loads one
# session, reduces it and frees it (peak ~250 MB for the largest session). If a session
# is needed twice, load it twice: decompressing one .npz takes a second or two.
subjects = list_participants(DATA_DIR)
print(f'Subjects found: {len(subjects)}')

all_sessions = {}  # {subject_id: [path_to_ses1, path_to_ses2]}
for subj in subjects:
    paths = [os.path.join(DATA_DIR, subj, f'ses{ses}.npz') for ses in (1, 2)]
    paths = [p for p in paths if os.path.exists(p)]
    if paths:
        all_sessions[subj] = paths

n_npz = sum(len(v) for v in all_sessions.values())
print(f'Indexed {len(all_sessions)} subjects, {n_npz} sessions '
      f'(loaded one at a time in Section 1)')
if not all_sessions:
    raise FileNotFoundError(
        f'No <subject>/ses<N>.npz under {DATA_DIR}. Expected the layout '
        f'extract_npz.py writes: <data dir>/<subject_id>/ses<N>.npz '
        f'(data/segments by default, config.DATA_DIR)')

"""
=============================================================================
SECTION 5: Run the FMT depth pipeline
=============================================================================
"""

import gc
import time
import pandas as pd
from scipy.stats import spearmanr

# One seed for every resampling step below, so a rerun is byte-identical.
SEED_L2 = 20260817

P = PARAMS
# A default, not an assumption: load_npz_session reads the file's own `fs`, and every
# session is checked against this value below; a mismatch raises.
fs = 250  # expected sampling rate; verified per session against the .npz

# ---- Storage ----
SEG_DEPTH, SEG_BLOCK, SEG_COND, SEG_SES = [], [], [], []
SEG_SCALP, SEG_GED, SEG_GED_TOPO = [], [], []
# Per-subject GED selection quality, kept index-aligned with SEG_SUBJ.
SEG_GED_R2, SEG_GED_R2_BEST, SEG_GED_NSES = [], [], []
SEG_NSEG = []
SEG_SUBJ = []
sensor_ch_names = None

# Run-level provenance for the things this route used to do silently. Every one
# of these reaches the README and fmt_results.json, because a gate nobody can
# see afterwards is indistinguishable from a gate that was never applied.
n_drop_short = 0            # sessions dropped by MIN_SESSION_MIN
n_ged_linalg = 0            # sessions whose GED eigenproblem failed
n_overlap_dropped = 0       # analysis windows dropped as overlapping
overlap_secs = []           # the overlap of each dropped pair, seconds
elecs_used = set()          # FMT_ELECS labels that actually matched a channel
elecs_missing = set()       # FMT_ELECS labels present in no recording
warned_elecs = False        # print the requested-vs-used warning once
floor_checked = False       # _assert_floor_negligible has run once

print(f'\n{"="*72}')
print('  SECTION 1: Per-session FMT theta power')
print(f'{"="*72}')

t0 = time.time()
for subj_id, session_paths in all_sessions.items():
    dep_a, blk_a, cnd_a, ses_a, sc_a, ge_a = [], [], [], [], [], []
    ged_topo_acc = None
    ged_topo_n = 0
    ged_r2_acc, ged_r2_best = [], []

    for i_ses, npz_path in enumerate(session_paths, 1):
        # One recording resident at a time. `sess` is rebound on the next
        # iteration and dropped at the end of this one, so peak RSS is set by the
        # largest single session (~250 MB), not by the cohort (11.87 GB).
        sess = load_npz_session(npz_path)
        ch_names = sess['ch_names']
        ch_pos = sess['ch_positions']
        n_chan = sess['n_chan']
        n_seg = sess['n_seg']

        if sensor_ch_names is None:
            sensor_ch_names = ch_names
        elif n_chan != len(sensor_ch_names):
            print(f'  {subj_id} ses{i_ses}: {n_chan} ch != ref {len(sensor_ch_names)} - skip')
            continue

        if n_seg == 0:
            continue

        # --- Sub-select the analysis window out of the 90 s extracted segment ---
        #
        # extract_npz writes each segment as
        #     [report - EXTRACT_PRE_END - EXTRACT_EPOCH_DUR, report - EXTRACT_PRE_END]
        # i.e. it ends just before the depth report. The analysis window is the last
        # EPOCH_DUR seconds before report - PRE_END, derived from PRE_END and the
        # geometry the .npz records (not the first EPOCH_DUR seconds of the segment).
        # EXTRACT_PRE_END (2) is a different quantity: what the extractor cut, not what
        # the analysis asks for. PRE_END >= EXTRACT_PRE_END is required and is checked
        # below.
        # -- GEOMETRY IS READ FROM THE FILE, NOT ASSUMED ------------------
        # A re-extraction at different values would otherwise cut a different window
        # out of every segment, with nothing in the run to show it. The module
        # constants are only the fallback for files without the keys, and taking that
        # fallback is announced.
        ex_dur = sess.get('extract_epoch_dur')
        ex_pre = sess.get('extract_pre_end')
        if ex_dur is None or ex_pre is None:
            ex_dur = EXTRACT_EPOCH_DUR if ex_dur is None else ex_dur
            ex_pre = EXTRACT_PRE_END if ex_pre is None else ex_pre
            print(f'  {subj_id} ses{i_ses}: .npz predates the extraction-geometry '
                  f'keys - ASSUMING {ex_dur:g} s / pre_end {ex_pre:g} s. '
                  f'Re-extract to remove the assumption.')

        # The file's own rate, not the module default (which is only a default).
        fs_file = int(sess.get('fs', fs))
        if fs_file != int(fs):
            raise ValueError(
                '%s ses%d is sampled at %d Hz but this run assumes %d Hz. Every '
                'window length, filter design and trim below is in samples, so a '
                'silent rate change moves all of them.'
                % (subj_id, i_ses, fs_file, int(fs)))

        epoch_samples = int(round(P['EPOCH_DUR'] * fs))
        seg_len = np.asarray(sess['seg_len']).astype(int)

        # How much later than the extraction's window this run's window must end.
        # PRE_END is measured back from the report, so a LARGER PRE_END means the
        # window ends EARLIER and we trim that much off the segment's tail.
        trim = int(round((P['PRE_END'] - ex_pre) * fs))
        if trim < 0:
            raise ValueError(
                'PRE_END=%g is smaller than the %g used at extraction: the data '
                'between them was never written to the .npz, so this window '
                'cannot be reconstructed. Re-run extract_npz.py with a '
                'smaller PRE_END.' % (P['PRE_END'], ex_pre))
        if float(P['EPOCH_DUR']) > float(ex_dur):
            raise ValueError(
                'EPOCH_DUR=%g > %g s stored per segment by extract_npz.py; '
                'that data is not in the .npz.' % (P['EPOCH_DUR'], ex_dur))

        # Start-anchored and inclusive at both ends. The stored segment is anchored at
        # its start (extract_npz computes `start` from the report and only ever moves
        # the end), so the analysis window sits at a fixed offset from the start and is
        # the same slice for every segment; an end-anchored window would read the wrong
        # place on any end-truncated segment. Same arithmetic as
        # heart_depth_pipeline.window_indices, including the +1 that makes the window
        # `epoch_dur*fs + 1` samples.
        hi_incl = int(round(float(ex_dur) * fs)) - trim
        lo = hi_incl - epoch_samples
        if lo < 0:
            raise ValueError(
                'EPOCH_DUR=%g with PRE_END=%g needs %g s before the report, but '
                'extract_npz.py stored only %g s. Re-extract, do not pad.'
                % (P['EPOCH_DUR'], P['PRE_END'],
                   float(P['EPOCH_DUR']) + float(P['PRE_END']) - float(ex_pre),
                   float(ex_dur) + float(ex_pre)))
        hi = hi_incl + 1

        # A segment either contains the whole requested window or it is dropped, so
        # zero padding is never read as signal.
        usable = seg_len >= hi
        if not np.any(usable):
            continue
        idx = np.where(usable)[0]

        # -- RESOLVE OVERLAPS AT THE *ANALYSIS* DURATION ------------------
        # extract_npz writes `overlap_resolved=False`: it does not trim at its own 90 s,
        # which would discard seconds a 60 s analysis is entitled to, so the consumer
        # resolves overlaps. Two analysis windows that overlap share samples, and
        # counting those samples in two "independent" segments inflates n. This happens
        # on the shipped segment files at 60 s / PRE_END 3.
        #
        # The earlier window of each pair is dropped, as in heart_depth_pipeline. (An
        # alternative rule trims the earlier window's end and drops it only if what
        # remains is under EPOCH_DUR - 3 s; the two agree whenever the overlap exceeds
        # 3 s.) Dropping is chosen because a variable window would make `lo`/`hi`
        # per-segment, and every slice below assumes they are constant.
        #
        # An unmarked file is assumed already resolved (at 90 s), and says so.
        ov_flag = sess.get('overlap_resolved')
        if ov_flag is None:
            ov_flag = True
            print(f'  {subj_id} ses{i_ses}: .npz predates the overlap-provenance '
                  f'key - ASSUMING it was resolved at extraction.')
        n_overlap = 0
        seg_lim = {}   # {segment index: usable window length} under OVERLAP_RULE='trim'
        if not ov_flag and idx.size > 1:
            span_all = np.asarray(sess['seg_span'])
            order = idx[np.argsort(span_all[idx, 0])]
            starts_abs = span_all[order, 0] + lo
            ends_abs = span_all[order, 0] + hi            # exclusive
            drop = []
            for a in range(len(order) - 1):
                if ends_abs[a] > starts_abs[a + 1]:
                    _lim = int(starts_abs[a + 1] - starts_abs[a])
                    if (P.get('OVERLAP_RULE', 'drop') == 'trim'
                            and _lim >= float(P.get('OVERLAP_MIN_S', 30)) * fs):
                        seg_lim[int(order[a])] = _lim
                    else:
                        drop.append(int(order[a]))
            n_overlap = len(drop)
            # Record each dropped pair's overlap, so that the pairs the trim rule above
            # would have kept (overlap of 3 s or less) can be counted; the run prints the
            # count and writes it as n_overlap_3s_or_less.
            for _a in range(len(order) - 1):
                if ends_abs[_a] > starts_abs[_a + 1]:
                    overlap_secs.append(
                        float(ends_abs[_a] - starts_abs[_a + 1]) / float(fs))
            n_overlap_dropped += n_overlap
            if drop:
                idx = np.asarray(sorted(set(int(v) for v in idx) - set(drop)),
                                 dtype=int)
                print(f'  {subj_id} ses{i_ses}: {n_overlap} analysis window(s) '
                      f'OVERLAP at {P["EPOCH_DUR"]:g} s - the earlier of each pair '
                      f'is dropped')
                if idx.size == 0:
                    continue

        # -- GATE: MIN_SESSION_MIN ---------------------------------------
        # The segment files carry no recording duration, so the gate is applied to a lower
        # bound, computed exactly as in the heartbeat analyses (session_length.py): a
        # session is dropped only when even that lower bound is under the threshold.
        if P.get('MIN_SESSION_MIN'):
            _dur_lb_min = session_duration_lowerbound_min(sess['block_spans'],
                                                          sess['seg_span'], fs)
            if _dur_lb_min < float(P['MIN_SESSION_MIN']):
                print(f'    {subj_id} ses{i_ses}: session duration lower bound '
                      f'{_dur_lb_min:.2f} min < MIN_SESSION_MIN='
                      f'{P["MIN_SESSION_MIN"]:g} - session DROPPED')
                n_drop_short += 1
                continue

        # Full stored segments are kept BY REFERENCE (no copy) so the theta
        # filter can run on the whole 90 s and the analysis window be cut out
        # afterwards. Filtering the 60 s window itself puts the filter's edge
        # transient inside the data the trim is supposed to be protecting.
        eeg_full = [sess['seg_eeg'][i] for i in idx]
        eeg_segs = np.stack([sess['seg_eeg'][i][:, lo:hi] for i in idx])
        ecg_segs = np.stack([sess['seg_ecg'][i][lo:hi] for i in idx])
        resp_segs = np.stack([sess['seg_resp'][i][lo:hi] for i in idx])
        depth = sess['seg_depth'][idx]
        cond = sess['seg_cond'][idx]
        subcond = sess['seg_subcond'][idx]
        n_use = len(idx)

        # Filter conditions: nosapo (0) and sapo (1) for FMT depth
        # pre-med (2) and post-med (3) are excluded from depth analysis
        if P['INCLUDE_SAPO']:
            keep_cond = np.isin(cond, [0, 1])
        else:
            keep_cond = cond == 0
        if np.sum(keep_cond) == 0:
            continue

        eeg_segs = eeg_segs[keep_cond]
        eeg_full = [eeg_full[k] for k in np.flatnonzero(keep_cond)]
        idx_f = np.asarray(idx)[keep_cond]
        depth = depth[keep_cond]
        cond = cond[keep_cond]
        subcond = subcond[keep_cond]
        n_seg_f = len(depth)

        # Gate: MIN_DEPTHS per session
        if len(np.unique(depth)) < P['MIN_DEPTHS']:
            print(f'    {subj_id} ses{i_ses}: {len(np.unique(depth))} depths < {P["MIN_DEPTHS"]} - skip')
            continue

        # --- Theta filter ---
        # Filter the full stored segment, then cut the analysis window out of the
        # result: filtering the 60 s window itself would put the filter's edge transient
        # inside the window. This is not the same as filtering the continuous recording;
        # it is the most context the .npz can supply.
        theta_segs = np.empty_like(eeg_segs)
        for i in range(n_seg_f):
            theta_segs[i] = theta_filter(
                eeg_full[i], fs, P['THETA_BP'][0], P['THETA_BP'][1])[:, lo:hi]
        del eeg_full
        broadband_segs = eeg_segs

        # Samples each window's power is averaged over: 2 s in from both ends, up to the
        # next window's start for a trimmed window, and none within BOUNDARY_GUARD_S of a
        # join.
        trim = int(2.0 * fs)  # 2s trim
        _g = int(round(float(P.get('BOUNDARY_GUARD_S') or 0) * fs))
        if _g and sess.get('boundaries') is None:
            raise SystemExit('BOUNDARY_GUARD_S is set but %s ses%d has no `boundaries`; '
                             'extract the segment files again' % (subj_id, i_ses))
        _W = theta_segs.shape[2]
        # own[i]: samples window i owns (up to the next window's start when trimmed) and
        # that are not within the guard of a join
        own = np.ones((n_seg_f, _W), bool)
        for i in range(n_seg_f):
            own[i, seg_lim.get(int(idx_f[i]), _W):] = False
            if _g:
                _a = int(np.asarray(sess['seg_span'])[idx_f[i], 0]) + lo
                for _j in np.asarray(sess['boundaries'], np.int64) - _a:
                    if -_g < _j < _W + _g:
                        own[i, max(0, _j - _g):max(0, min(_W, _j + _g))] = False
        # power: 2 s in from both window ends, owned samples only
        vmask = np.zeros((n_seg_f, _W), bool)
        vmask[:, trim:epoch_samples - trim + 1] = True
        vmask &= own
        # GED covariance: every owned sample (it never had the 2 s edge trim)
        cov_mask = own

        # --- Scalp Brandmeyer power ---
        # A label matching no channel is reported; matching NONE raises, because
        # a run that retains 0 subjects still writes a full-shaped L2 table of
        # zeros, and a well-formed empty result reads as a measured null.
        chan_lc = [c.lower() for c in ch_names]
        brand_ix = [i for i, c in enumerate(chan_lc) if c in P['FMT_ELECS']]
        _missing = [c for c in P['FMT_ELECS'] if c not in chan_lc]
        if not brand_ix:
            raise ValueError(
                'FMT_ELECS matches NO channel in %s ses%d (requested %s; the '
                'recording has %d channels). A run cannot be named after an '
                'electrode set it never used.'
                % (subj_id, i_ses, sorted(P['FMT_ELECS']), len(ch_names)))
        elecs_used.update(chan_lc[i] for i in brand_ix)
        if _missing:
            elecs_missing.update(_missing)
            if not warned_elecs:
                print(f'    WARNING FMT_ELECS: {len(P["FMT_ELECS"])} requested, '
                      f'{len(brand_ix)} present - missing '
                      f'{", ".join(sorted(_missing))}.')
                warned_elecs = True

        # NaN, not 0: a segment whose window is too short must be missing, not a
        # log-power of exactly zero sitting in the middle of the distribution.
        scalp_pow = np.full(n_seg_f, np.nan)
        for i in range(n_seg_f):
            pow_ts = np.mean(theta_segs[i, brand_ix, :] ** 2, axis=0)
            if vmask[i].sum() >= fs:
                if not floor_checked:
                    _assert_floor_negligible(pow_ts[vmask[i]],
                                             '%s ses%d scalp theta' % (subj_id, i_ses))
                    floor_checked = True
                scalp_pow[i] = np.mean(np.log10(pow_ts[vmask[i]] + EPS))

        # --- GED trained on THIS session's segments ---
        # Build segment mask in the concatenated data space
        # S = cov(theta segments), R = cov(broadband segments) + shrinkage
        X_theta = np.concatenate([theta_segs[i][:, cov_mask[i]] for i in range(n_seg_f)], axis=1)  # [nChan x totalSamples]
        X_broad = np.concatenate([broadband_segs[i][:, cov_mask[i]] for i in range(n_seg_f)], axis=1)
        n_t = X_theta.shape[1]

        ged_ok = False
        ged_pow = np.full(n_seg_f, np.nan)
        ged_topo = None
        if n_t > n_chan * 2:
            S = np.cov(X_theta)
            R = np.cov(X_broad)
            R = (1 - P['GED_SHRINKAGE']) * R + \
                P['GED_SHRINKAGE'] * np.trace(R) / n_chan * np.eye(n_chan)
            from scipy.linalg import eigh as scipy_eigh
            try:
                evals, evecs = scipy_eigh(S, R)
                idx = np.argsort(evals)[::-1]
                evals, evecs = np.real(evals[idx]), np.real(evecs[:, idx])
                Ag = S @ evecs
                # Template centres are their own setting (default: FMT_ELECS).
                tmpl = make_frontal_template(ch_names, ch_pos,
                                             P.get('GED_TEMPLATE_ELEC') or P['FMT_ELECS'])
                n_cand = min(P['GED_TOP_N'], evecs.shape[1])
                # Keep every candidate's score. `r2` is the squared correlation
                # between the candidate's forward topography and the template, the
                # only direct answer to "is this component frontocentral".
                # This loop duplicates fmt_depth_pipeline.build_ged; keep the two in
                # step.
                _all_r2, _qual = [], []
                for k in range(n_cand):
                    tn = Ag[:, k] / (np.linalg.norm(Ag[:, k]) + np.finfo(float).eps)
                    r2 = np.corrcoef(tn, tmpl)[0, 1] ** 2
                    _all_r2.append(float(r2))
                    if r2 >= P['GED_R2_THRESH']:
                        _qual.append(k)
                _use = _qual[:max(1, int(P.get('GED_MAX_COMP', 1)))]
                if _use:
                    # Power is averaged across components, then logged. The mean of
                    # the logs would be a geometric mean, a different quantity, and
                    # it is what averaging `ged_pow` instead of the linear power gives.
                    for i in range(n_seg_f):
                        if vmask[i].sum() >= fs:
                            acc = None
                            for k in _use:
                                ws = np.sign(np.corrcoef(Ag[:, k], tmpl)[0, 1]) or 1
                                c = (ws * evecs[:, k]) @ theta_segs[i]
                                acc = c[vmask[i]] ** 2 if acc is None else acc + c[vmask[i]] ** 2
                            ged_pow[i] = np.mean(np.log10(acc / len(_use) + EPS))
                    # `topo` and `r2` describe the FIRST qualifying component
                    # only: a mean forward pattern over different components has
                    # no source to localize.
                    k0 = _use[0]
                    wsign = np.sign(np.corrcoef(Ag[:, k0], tmpl)[0, 1])
                    if wsign == 0: wsign = 1
                    ged_topo = wsign * Ag[:, k0]
                    ged_ok = True
                    if ged_topo_acc is None:
                        ged_topo_acc = np.zeros_like(ged_topo)
                    ged_topo_acc += ged_topo
                    ged_topo_n += 1
                    ged_r2_acc.append(float(_all_r2[k0]))
                if _all_r2:
                    # Recorded even on rejection: a session that peaked at .29
                    # against a .30 gate is a different fact from one that peaked at
                    # .04, and only the second says the filter found nothing
                    # frontocentral.
                    ged_r2_best.append(max(_all_r2))
            except np.linalg.LinAlgError as _e:
                # Only the generalised-eigenproblem failure is caught, as in
                # fmt_depth_pipeline. A broader except would also swallow
                # make_frontal_template's deliberate raise on a missing label, which
                # exists because a flat template makes every R2 NaN and silently
                # empties the run; `ged ok=0` is also a legitimate outcome on real data
                # (some sessions accept no component), so the two must not be confused.
                n_ged_linalg += 1
                print(f'    {subj_id} ses{i_ses}: GED eigenproblem failed '
                      f'({_e}) - no component this session')

        # --- Block labels ---
        blk = np.where(subcond == 1, 0.5, np.where(subcond == 0, -0.5, 0))  # probe=+0.5, emerge=-0.5

        dep_a.append(depth)
        blk_a.append(blk)
        cnd_a.append(cond.astype(float))
        ses_a.append(np.full(n_seg_f, i_ses, dtype=float))
        sc_a.append(scalp_pow)
        ge_a.append(ged_pow)

        # Drop the raw arrays now rather than at the next rebind: the filtered
        # copies inside this body are the peak, and holding this session while
        # the next one loads would double it.
        del sess, eeg_segs, ecg_segs, resp_segs
        gc.collect()

    if not dep_a:
        continue
    depth = np.concatenate(dep_a)
    block = np.concatenate(blk_a)
    cond = np.concatenate(cnd_a)
    ses = np.concatenate(ses_a)
    scalp = np.concatenate(sc_a)
    ged = np.concatenate(ge_a)

    # Standardize within session
    if P['STANDARDIZE_WITHIN_SESSION']:
        for d_val in np.unique(ses):
            m = ses == d_val
            scalp[m] = zscore_omitnan(scalp[m])
            ged[m] = zscore_omitnan(ged[m])

    if np.sum(np.isfinite(scalp)) < P['minSeg']:
        print(f'  {subj_id}: {np.sum(np.isfinite(scalp))} segs < minSeg - skip')
        continue

    SEG_DEPTH.append(depth)
    SEG_BLOCK.append(block)
    SEG_COND.append(cond)
    SEG_SES.append(ses)
    SEG_SCALP.append(scalp)
    SEG_GED.append(ged)
    if ged_topo_n > 0:
        SEG_GED_TOPO.append(ged_topo_acc / ged_topo_n)
    else:
        SEG_GED_TOPO.append(None)
    SEG_GED_R2.append(float(np.mean(ged_r2_acc)) if ged_r2_acc else np.nan)
    SEG_GED_R2_BEST.append(float(np.max(ged_r2_best)) if ged_r2_best else np.nan)
    SEG_GED_NSES.append(int(ged_topo_n))
    SEG_NSEG.append(len(depth))
    SEG_SUBJ.append(subj_id)
    print(f'  {subj_id}: {len(depth)} segs (scalp ok={np.sum(np.isfinite(scalp))}, '
          f'ged ok={np.sum(np.isfinite(ged))})')

n_sub_fmt = len(SEG_DEPTH)
print(f'\nSubjects retained: {n_sub_fmt}')
print(f'  FMT_ELECS requested {len(PARAMS["FMT_ELECS"])}, used '
      f'{len(elecs_used)}' +
      (f' (missing: {", ".join(sorted(elecs_missing))})' if elecs_missing else ''))
if n_drop_short:
    print(f'  sessions dropped by MIN_SESSION_MIN={PARAMS["MIN_SESSION_MIN"]:g}: '
          f'{n_drop_short}')
if n_overlap_dropped:
    _u3 = sum(1 for x in overlap_secs if x <= 3.0)
    print(f'  overlapping analysis windows dropped: {n_overlap_dropped} '
          f'({_u3} of them overlapping by 3 s or less, i.e. {_u3} '
          f'segment(s) a trim-instead-of-drop rule would have kept)')
if n_ged_linalg:
    print(f'  sessions whose GED eigenproblem failed: {n_ged_linalg}')

# A run with no subjects must not produce a well-formed result: an 18-row table of
# zeros with a README listing its parameters would read as a measured null.
if n_sub_fmt < 2:
    raise RuntimeError(
        'Only %d subject(s) survived Section 1, so there is no group to test. '
        'Refusing to write L2 tables: an empty result with the right shape '
        'reads as a measured null. Check the gates printed above -- '
        'FMT_ELECS matched %d channel(s), %d session(s) were dropped as short, '
        'and %d had no GED component.'
        % (n_sub_fmt, len(elecs_used), n_drop_short, n_ged_linalg))
print(f'Section 1 completed in {time.time()-t0:.0f}s')

# =====================================================================
# SECTION 2: L1 GLM
# =====================================================================
print(f'\n{"="*72}')
print('  SECTION 2: L1 per-subject depth GLM')
print(f'{"="*72}')

models = ['unadj', 'reportmethod', 'cond']
n_models = len(models)

# Is the `cond` regressor identified at all? With INCLUDE_SAPO=False every segment is
# nosapo, so the condition column of the L1 design is all zeros. fit_glm_wls
# ridge-regularises (X'X + 1e-6*tr*I) rather than refusing, so it returns the
# unadjusted depth beta and only burns one df. The `cond` rows of the L2 table would
# then reproduce the `unadj` rows exactly and read as "adjusting for condition changed
# nothing", when the model was never estimable. Those rows are not written (Section 3).
COND_IDENTIFIED = any(
    np.unique(np.asarray(SEG_COND[i], float)[
        np.isfinite(np.asarray(SEG_COND[i], float))]).size > 1
    for i in range(n_sub_fmt))
if not COND_IDENTIFIED:
    print("  [cond] the condition regressor has NO variance in any subject "
          "(INCLUDE_SAPO=%s) - the `cond` model is not identified and its L2 "
          "rows will NOT be written. It would reproduce `unadj` exactly."
          % P['INCLUDE_SAPO'])
SCALP_BETA = np.full((n_models, n_sub_fmt), np.nan)
SCALP_RV = np.full((n_models, n_sub_fmt), np.nan)
GED_BETA = np.full((n_models, n_sub_fmt), np.nan)
GED_RV = np.full((n_models, n_sub_fmt), np.nan)

for i_s in range(n_sub_fmt):
    depth = SEG_DEPTH[i_s]
    block = SEG_BLOCK[i_s]
    cond = SEG_COND[i_s]

    if P['DEPTH_MODE'] == 'binary':
        dep_pred = np.full(len(depth), np.nan)
        dep_pred[np.isin(depth, P['LOW_DEPTHS'])] = 0
        dep_pred[np.isin(depth, P['HIGH_DEPTHS'])] = 1
        if not np.any(dep_pred == 0) or not np.any(dep_pred == 1):
            continue
        dep_ctr = dep_pred - np.nanmean(dep_pred)
    else:
        dep_ctr = depth.astype(float) - np.nanmean(depth)

    blk_eff = block.copy()
    cnd_eff = cond - np.nanmean(cond)
    cnd_eff[np.isnan(cnd_eff)] = 0

    for i_meas, y in enumerate([SEG_SCALP[i_s], SEG_GED[i_s]]):
        ok = np.isfinite(y) & np.isfinite(dep_ctr)
        if np.sum(ok) < P['minSeg'] or len(np.unique(dep_ctr[ok])) < 2:
            continue
        for i_mod, model in enumerate(models):
            if model == 'unadj':
                X = np.column_stack([np.ones(np.sum(ok)), dep_ctr[ok]])
            elif model == 'reportmethod':
                X = np.column_stack([np.ones(np.sum(ok)), dep_ctr[ok], blk_eff[ok]])
            elif model == 'cond':
                # Drop a dead column rather than ridge through it. A subject with
                # only one condition (common even at INCLUDE_SAPO=True: not every
                # session has SAPO blocks) contributes a constant column here.
                # Fitting it anyway gives the same depth beta but divides the
                # residual variance by n-3 instead of n-2, inflating `rv` by
                # (n-2)/(n-3), and `rv` is what the ivw_shrunk weights are built
                # from. The reduced design gives the same estimate with the right df.
                if np.ptp(cnd_eff[ok]) == 0:
                    X = np.column_stack([np.ones(np.sum(ok)), dep_ctr[ok]])
                else:
                    X = np.column_stack([np.ones(np.sum(ok)), dep_ctr[ok],
                                         cnd_eff[ok]])
            B, rv, _ = fit_glm_wls(X, y[ok], method=P['L1_METHOD'])
            if i_meas == 0:
                SCALP_BETA[i_mod, i_s] = B[1]
                SCALP_RV[i_mod, i_s] = rv
            else:
                GED_BETA[i_mod, i_s] = B[1]
                GED_RV[i_mod, i_s] = rv

print(f'  L1 GLM completed for {n_sub_fmt} subjects')

"""
=============================================================================
SECTION 5b: the rest of L2, and the post-hocs
=============================================================================

FMT's L2 quantity is a scalar per subject (one theta-power depth slope), so there is
no map to cluster over. What applies is implemented below: the 20% trimmed mean with a
percentile bootstrap, and the weighted group CI.
"""

import json

TRIM = 0.20
N_BOOT = 10000


def robust_one_sample_vec(x, trim=TRIM, n_boot=N_BOOT, seed=SEED_L2):
    """20% trimmed mean + percentile bootstrap CI (Wilcox; Rousselet).

    Robust to the few extreme per-subject slopes that a one-sample t over a few dozen
    subjects is sensitive to. Reports the CI, not only a p: a trimmed mean whose CI
    excludes 0 is the claim, and the width is the uncertainty."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return dict(n=int(x.size), tm=np.nan, lo=np.nan, hi=np.nan, excludes_zero=False)
    rng = np.random.default_rng(seed)
    tm = float(trim_mean_1d(x, trim))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = trim_mean_1d(x[rng.integers(0, x.size, x.size)], trim)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=int(x.size), tm=tm, lo=float(lo), hi=float(hi),
                excludes_zero=bool(lo > 0 or hi < 0))


def trim_mean_1d(x, prop):
    x = np.sort(np.asarray(x, float))
    k = int(np.floor(len(x) * prop))
    return np.mean(x[k:len(x) - k]) if len(x) - 2 * k > 0 else np.mean(x)


def weighted_ci_vec(b, w, alpha=0.05):
    """Weighted group mean with a 95% CI, on the same construction as the t.

    It must reproduce the reported t exactly, or the interval and the p describe
    different quantities. The L2 t is wm / sqrt(SS / ((n_eff-1)*Ws)); the CI is
    wm +/- t_crit(n_eff-1) * that same standard error.

    Used in place of the Bayesian HDI+ROPE reading: the heartbeat analyses report a
    frequentist CI on the group coefficient, so reporting one here keeps the three
    analyses consistent.
    """
    from scipy.stats import t as _t
    b = np.asarray(b, float)
    w = np.asarray(w, float)
    ok = np.isfinite(b) & np.isfinite(w)
    b, w = b[ok], w[ok]
    if b.size < 2:
        return {'n': int(b.size), 'mean': np.nan, 'lo': np.nan, 'hi': np.nan,
                'n_eff': np.nan, 'excludes_zero': False}
    Ws = w.sum()
    wm = float((w * b).sum() / Ws)
    n_eff = float(Ws ** 2 / (w ** 2).sum())
    ss = float((w * (b - wm) ** 2).sum())
    se = np.sqrt(ss / (max(n_eff - 1.0, 1.0) * Ws))
    crit = float(_t.ppf(1 - alpha / 2, max(n_eff - 1.0, 1.0)))
    lo, hi = wm - crit * se, wm + crit * se
    return {'n': int(b.size), 'mean': wm, 'lo': float(lo), 'hi': float(hi),
            'n_eff': n_eff, 'excludes_zero': bool(lo > 0 or hi < 0)}


def monotonicity_check(seg_depth, seg_val, depth_levels):
    """Per-subject Spearman rho over depth levels, and how many are monotone.

    The diagnostic that decides whether a slope is usable as a neurofeedback target. A
    linear slope can be strongly significant through a V-shape (p = 5e-51 with 0% of
    subjects monotone on a synthetic V). For real-time feedback a non-monotonic measure
    is worse than useless, because the participant is rewarded for moving in the wrong
    direction over part of the range.

    Do not replace this with an omnibus ANOVA: it fires on non-monotonic patterns too,
    and it needs complete cases (few subjects have all five depth levels)."""
    rhos, mono, per_level = [], [], {d: [] for d in depth_levels}
    for dep, val in zip(seg_depth, seg_val):
        dep = np.asarray(dep, float)
        val = np.asarray(val, float)
        ok = np.isfinite(dep) & np.isfinite(val)
        if ok.sum() < 3:
            continue
        levels = sorted(set(dep[ok].astype(int)))
        if len(levels) < 3:
            continue
        means = [float(np.mean(val[ok][dep[ok] == L])) for L in levels]
        for L, mn in zip(levels, means):
            if L in per_level:
                per_level[L].append(mn)
        r = float(spearmanr(levels, means).statistic)
        rhos.append(r)
        d = np.diff(means)
        mono.append(bool(np.all(d > 0) or np.all(d < 0)))
    rhos = np.asarray(rhos, float)
    return dict(n_subjects=int(rhos.size),
                rho_mean=float(np.nanmean(rhos)) if rhos.size else np.nan,
                rho_median=float(np.nanmedian(rhos)) if rhos.size else np.nan,
                frac_monotone=float(np.mean(mono)) if mono else np.nan,
                n_monotone=int(np.sum(mono)),
                level_means={int(k): (float(np.mean(v)) if v else None)
                             for k, v in per_level.items()})


def loso_influence(betas, weights, n_perm=1000, seed=SEED_L2):
    """Jackknife: drop each subject in turn, does the group effect survive?

    Not cross-validation: this asks whether the effect is carried by one or
    two subjects. Reports the worst-case p across all drops and which subject
    produces it. An effect whose p crosses alpha when a single subject is
    removed is a different claim from one that does not."""
    betas = np.asarray(betas, float)
    ok = np.isfinite(betas)
    idx = np.where(ok)[0]
    if idx.size < 6:
        return dict(n=int(idx.size), worst_p=np.nan, worst_i=None, all_survive=False)
    ps = []
    for drop in range(idx.size):
        keep = np.delete(idx, drop)
        w = np.asarray(weights, float)[keep]
        _, p = signflip_ttest(betas[keep], w, n_perm)
        ps.append(p)
    ps = np.asarray(ps, float)
    wi = int(np.argmax(ps))
    return dict(n=int(idx.size), worst_p=float(ps[wi]),
                worst_subject_index=int(idx[wi]),
                median_p=float(np.median(ps)),
                all_survive=bool(np.all(ps < P['alpha'])))


# =====================================================================
# SECTION 3: L2 GROUP TESTS
# =====================================================================
print(f'\n{"="*72}')
print('  SECTION 3: L2 group tests')
print(f'{"="*72}')

MIN_SUB_L2 = 5
schemes = list(P['schemes'])

W_IVW = None
if P['USE_IVW_SHRUNK']:
    W_IVW, ivw_info = shrunk_ivw(SCALP_RV[0, :], 'median')
    print(f'  [ivw_shrunk] {ivw_info["nUsed"]} subjects | c={ivw_info["c"]:.3g} | '
          f'eff_n={ivw_info["eff_n"]:.1f}/{n_sub_fmt} | '
          f'wmax={ivw_info["wmax_ratio"]:.1f}x')
    # Weights this concentrated mean the "group" result rests on a subset, and
    # precision_nseg should be read as primary instead.
    if ivw_info['eff_n'] < 0.5 * ivw_info['nUsed']:
        print('  [ivw_shrunk] WARNING effective n is under half the real n - this '
              'result rests on a subset of subjects. Read precision_nseg as primary.')

rows = []
for mname, B in [('FMT_scalp', SCALP_BETA), ('FMT_GED', GED_BETA)]:
    for i_mod, model in enumerate(models):
        # Not written when the model is not identified (see COND_IDENTIFIED in
        # Section 2). The npz still carries the (empty) row so that model indexing
        # stays positional; only the L2 table, which is what gets read as a result,
        # loses the row.
        if model == 'cond' and not COND_IDENTIFIED:
            continue
        betas = B[i_mod, :]
        for scheme in schemes:
            ok = np.isfinite(betas)
            n = int(np.sum(ok))
            t_val, p_val, d_val, neff = np.nan, np.nan, np.nan, np.nan
            if n >= MIN_SUB_L2:
                if scheme == 'precision_nseg':
                    w = np.array(SEG_NSEG)[ok].astype(float)
                elif scheme == 'ivw_shrunk' and W_IVW is not None:
                    w = W_IVW[ok]
                else:
                    w = np.ones(n)
                if np.sum(w) > 0:
                    t_val, p_val = signflip_ttest(betas[ok], w, P['nPerm'])
                    # CohenD is unweighted in all three rows by construction
                    # (mean/sd of the same slopes), so `n` and `CohenD` are
                    # identical down a measure x model block and only t/p move.
                    # N_eff_kish is written so the difference is visible: it is n
                    # for unweighted and strictly below n otherwise.
                    d_val = np.nanmean(betas[ok]) / (np.nanstd(betas[ok]) + np.finfo(float).eps)
                    _wn = np.asarray(w, float) / np.mean(w)
                    neff = float(np.sum(_wn) ** 2 / np.sum(_wn ** 2))
            sig = '***' if p_val < P['alpha'] else ''
            print(f'  {mname:9s} {model:6s} {scheme:15s}: n={n} n_eff={neff:.1f} '
                  f't={t_val:.3f} p={p_val:.4f} d={d_val:.3f} {sig}')
            rows.append({'Measure': mname, 'Model': model, 'Scheme': scheme,
                         'n': n, 'N_eff_kish': neff,
                         't': t_val, 'p': p_val, 'CohenD': d_val,
                         'Sig': float(p_val < P['alpha']) if np.isfinite(p_val) else 0})

l2_df = pd.DataFrame(rows)
l2_csv = os.path.join(RESULTS_DIR, 'fmt_depth_L2.csv')
l2_df.to_csv(l2_csv, index=False)
print(f'\nL2 summary: {l2_csv}')

# The robust checks are post-hocs, so they run in Section 4, only on an effect that
# survived the main model.

# =====================================================================
# SECTION 3b: DEPTH PROFILE PLOT
# =====================================================================
# This is a between-subject mean; the L2 test above is a within-subject slope. The two
# can disagree and neither is wrong: each subject contributes only the depth levels
# they reported, and few report all five, so the mean at each level is taken over a
# different set of subjects. A flat or non-monotonic profile next to a significant
# slope is the expected shape of that mismatch, not evidence against the slope.
#
# The monotonicity diagnostic is in the post-hoc battery below (`monotonicity_check`:
# per-subject Spearman rho plus the fraction whose ordering is monotone). It is gated
# on detection, so a null run has only this profile to show the depth shape, and a
# slope can be null through an inverted U just as it can be significant through a V.
# Do not use an omnibus ANOVA instead: it fires on non-monotonic patterns too, and it
# needs complete cases.
n_dl = len(P['DEPTH_LEVELS'])
prof_mean = {'scalp': np.full(n_dl, np.nan), 'ged': np.full(n_dl, np.nan)}
prof_sem = {'scalp': np.full(n_dl, np.nan), 'ged': np.full(n_dl, np.nan)}
# Per-subject means kept, not only their summary (as in fmt_depth_pipeline.py).
prof_sub = {'scalp': np.full((n_dl, n_sub_fmt), np.nan),
            'ged': np.full((n_dl, n_sub_fmt), np.nan)}

for di, dl in enumerate(P['DEPTH_LEVELS']):
    sub_scalp = np.full(n_sub_fmt, np.nan)
    sub_ged = np.full(n_sub_fmt, np.nan)
    for i_s in range(n_sub_fmt):
        m = SEG_DEPTH[i_s] == dl
        v = SEG_SCALP[i_s][m]; v = v[np.isfinite(v)]
        if len(v) > 0: sub_scalp[i_s] = np.mean(v)
        v = SEG_GED[i_s][m]; v = v[np.isfinite(v)]
        if len(v) > 0: sub_ged[i_s] = np.mean(v)
    prof_sub['scalp'][di] = sub_scalp
    prof_sub['ged'][di] = sub_ged
    # Sample sd (ddof=1).
    ok = np.isfinite(sub_scalp)
    if np.sum(ok) >= 2:
        prof_mean['scalp'][di] = np.mean(sub_scalp[ok])
        prof_sem['scalp'][di] = np.std(sub_scalp[ok], ddof=1) / np.sqrt(np.sum(ok))
    ok = np.isfinite(sub_ged)
    if np.sum(ok) >= 2:
        prof_mean['ged'][di] = np.mean(sub_ged[ok])
        prof_sem['ged'][di] = np.std(sub_ged[ok], ddof=1) / np.sqrt(np.sum(ok))

prof_png = os.path.join(RESULTS_DIR, 'fmt_depth_profile.png')
try:
    cond_tag = 'nosapo+sapo' if P['INCLUDE_SAPO'] else 'nosapo'
    plot_depth_profile(P['DEPTH_LEVELS'], prof_mean, prof_sem, l2_df,
                        cond_tag, '', prof_png)
except Exception as e:
    print(f'  Plot failed ({e})')

# =====================================================================
# SAVE THE PRIMARY RESULTS, BEFORE ANY POST-HOC RUNS
# =====================================================================
# The primary result is complete once Section 3 has run, so it is saved here: a
# post-hoc must not be able to destroy it.
# SUBJ_IDS is saved as a real string dtype so that it loads with allow_pickle=False.
# Selection quality is part of the result. GED_TOPO / GED_TEMPLATE_R2 answer "is the
# selected component frontocentral", which an anatomical label cannot, because a single
# dipole fitted to a broad topography lands at the head centre and gets a subcortical
# label whatever the true generator. SCALP_RV / GED_RV are L1 slope variances, not
# dipole residual variance.
_nch_g = next((len(t) for t in SEG_GED_TOPO if t is not None), 0)
GED_TOPO = np.full((_nch_g, len(SEG_SUBJ)), np.nan)
for _i_g, _t_g in enumerate(SEG_GED_TOPO):
    if _t_g is not None and _nch_g:
        GED_TOPO[:, _i_g] = _t_g
np.savez(os.path.join(RESULTS_DIR, 'run_results.npz'),
         SCALP_BETA=SCALP_BETA, GED_BETA=GED_BETA,
         SCALP_RV=SCALP_RV, GED_RV=GED_RV,
         NSEG_SUB=np.array(SEG_NSEG),
         SUBJ_IDS=np.array([str(x) for x in SEG_SUBJ], dtype='U32'),
         GED_TOPO=GED_TOPO,
         GED_TEMPLATE_R2=np.asarray(SEG_GED_R2, float),
         GED_TEMPLATE_R2_BEST=np.asarray(SEG_GED_R2_BEST, float),
         GED_N_SESSIONS=np.asarray(SEG_GED_NSES, int),
         GED_CH_NAMES=np.array(list(sensor_ch_names), dtype=object),
         DEPTH_LEVELS=np.asarray(P['DEPTH_LEVELS'], int),
         PROFILE_MEAN_SCALP=prof_mean['scalp'],
         PROFILE_MEAN_GED=prof_mean['ged'],
         PROFILE_SEM_SCALP=prof_sem['scalp'],
         PROFILE_SEM_GED=prof_sem['ged'],
         PROFILE_SUB_SCALP=prof_sub['scalp'],
         PROFILE_SUB_GED=prof_sub['ged'])
print(f'  primary results saved: {os.path.join(RESULTS_DIR, "run_results.npz")}')

print(f'\n{"="*72}')
print('  SECTION 4: Post-hoc (gated on significance)')
print(f'{"="*72}')

meas_tags = ['FMT_scalp', 'FMT_GED']
sig_meas = [False, False]
sig_sch = ['', '']
for mi, mname in enumerate(meas_tags):
    for sch in P['SIG_PRIORITY']:
        mask = (l2_df['Measure'] == mname) & (l2_df['Model'] == 'unadj') & (l2_df['Scheme'] == sch)
        if mask.sum() > 0 and l2_df.loc[mask, 'Sig'].values[0] == 1:
            sig_meas[mi] = True
            sig_sch[mi] = sch
            break

sig_scalp, sig_ged = sig_meas
print(f'  DETECTED: scalp={"YES ("+sig_sch[0]+")" if sig_scalp else "none"}, '
      f'GED={"YES ("+sig_sch[1]+")" if sig_ged else "none"}')

# ==========================================================================
# THE POST-HOC BATTERY: GATED, AND SCOPED TO THE DETECTED EFFECT
# ==========================================================================
# One post-hoc set per detected measure. Within a measure the scope is fixed:
#
#   model  = 'unadj' only. The covariate models (reportmethod, cond) are adjustments
#            of the same slopes, not separate effects.
#   scheme = the one scheme that detected it (sig_sch[mi]), not all of them. The
#            weighting schemes are re-readings of one set of per-subject slopes;
#            running the battery over each would manufacture agreement, not test
#            anything.
#
# So: 4 checks x (however many measures survived).
#
# The robust and monotonicity checks do not take weights (they operate on the raw
# per-subject quantities), so the scheme only decides whether they run and how the
# output is labelled. LOSO uses the detecting scheme's weights; the group CI uses the
# precision_nseg weights when that scheme detected the effect and equal weights
# otherwise.
#
# Monotonicity is gated too: a non-monotone measure with a significant slope only
# exists once there is a significant slope, and on a null measure the monotone
# fraction is a property of noise.
print(f'\n{"-"*72}')
print('  POST-HOC BATTERY (gated on detection; unadj model, detecting scheme only)')
print(f'{"-"*72}')

rob_rows, mono_out, loso_out = [], {}, {}

if not any(sig_meas):
    print('  Skipped entirely - no detected unadj effect in either measure.')
else:
    for mi, (mname, B, vals) in enumerate(
            [('FMT_scalp', SCALP_BETA, SEG_SCALP), ('FMT_GED', GED_BETA, SEG_GED)]):
        if not sig_meas[mi]:
            print(f'\n  {mname}: skipped - no detected unadj effect.')
            continue
        b = B[0, :]                      # unadj slopes, and only those
        print(f'\n  {mname} (unadj, {sig_sch[mi]} scheme)')

        # 1. The group coefficient and its 95% CI. With precision_nseg the weights are
        # the detecting scheme's, so the interval and the reported p describe the same
        # quantity. This is what the paper reports beside the permutation p, as for the
        # heartbeat analyses.
        w_det = (np.asarray(SEG_NSEG, float)
                 if sig_sch[mi] == 'precision_nseg'
                 else np.ones(len(SEG_NSEG), float))
        ci = weighted_ci_vec(b, w_det)
        # 2. Robust. The trimmed mean answers "is this carried by the bulk of
        # subjects or by the tails", which the CI above cannot. (No Bayesian
        # HDI+ROPE reading is computed.)
        rb = robust_one_sample_vec(b)
        rob_rows.append({'Measure': mname, 'Model': 'unadj', 'Scheme': sig_sch[mi],
                         'n': rb['n'],
                         'GroupMean': ci['mean'], 'GroupCI_lo': ci['lo'],
                         'GroupCI_hi': ci['hi'], 'N_eff': ci['n_eff'],
                         'TrimmedMean': rb['tm'], 'CI_lo': rb['lo'], 'CI_hi': rb['hi'],
                         'RobustSig': rb['excludes_zero']})
        print(f'    group    b {ci["mean"]:+.4f}  '
              f'95% CI [{ci["lo"]:+.4f}, {ci["hi"]:+.4f}]  '
              f'(n = {ci["n"]}, n_eff = {ci["n_eff"]:.1f})')
        print(f'    robust   20% trimmed mean {rb["tm"]:+.4f}  '
              f'CI [{rb["lo"]:+.4f}, {rb["hi"]:+.4f}]  '
              f'-> {"excludes 0" if rb["excludes_zero"] else "INCLUDES 0"}')

        # 3. Monotonicity: the property that decides whether this is usable as a
        # neurofeedback target. A slope can be significant through a V-shape.
        mono = monotonicity_check(SEG_DEPTH, vals, P['DEPTH_LEVELS'])
        mono_out[mname] = mono
        print(f'    monotone rho median {mono["rho_median"]:+.3f}  '
              f'{mono["n_monotone"]}/{mono["n_subjects"]} subjects '
              f'({100*mono["frac_monotone"]:.0f}%) monotone')

        # 4. LOSO: is the group effect carried by one or two subjects?
        w = np.array(SEG_NSEG, float) if sig_sch[mi] == 'precision_nseg' else \
            (W_IVW if sig_sch[mi] == 'ivw_shrunk' and W_IVW is not None
             else np.ones(n_sub_fmt))
        lo = loso_influence(b, w)
        loso_out[mname] = lo
        verdict = 'survives every drop' if lo['all_survive'] else \
                  f'FAILS when subject index {lo["worst_subject_index"]} is dropped'
        print(f'    LOSO     worst p={lo["worst_p"]:.4f} '
              f'(median {lo["median_p"]:.4f}) - {verdict}')

    print('\n    A significant slope with few monotone subjects is not a usable')
    print('    neurofeedback target - see the note in Section 3b.')

robust_df = pd.DataFrame(rob_rows)


# =====================================================================
# ONE COMPACT RESULTS FILE
# =====================================================================
# Everything a report needs is a few kB of numbers, so it goes in one JSON.
# run_results.npz stays because it holds the per-subject betas any later post-hoc has
# to re-read (~5 kB), and the profile PNG stays because it is the figure.
_summary = {
    'params': {k: (list(v) if isinstance(v, (list, tuple)) else v)
               for k, v in P.items() if not callable(v)},
    # Which segment set the run read. Check this field before quoting a number.
    'data_dir': DATA_DIR,
    'arm': ARM,
    'extraction': {'EXTRACT_EPOCH_DUR': EXTRACT_EPOCH_DUR,
                   'EXTRACT_PRE_END': EXTRACT_PRE_END,
                   'EXTRACT_MIN_CONF': EXTRACT_MIN_CONF},
    # What the run actually DID, as opposed to what it was asked to do. The
    # folder tag records the REQUEST; every entry here is a gate or a divergence
    # that would otherwise leave no trace after the fact.
    'as_run': {
        'FMT_ELECS_requested': sorted(PARAMS['FMT_ELECS']),
        'FMT_ELECS_used': sorted(elecs_used),
        'FMT_ELECS_missing': sorted(elecs_missing),
        'n_sessions_dropped_short': int(n_drop_short),
        'n_ged_eigenproblem_failed': int(n_ged_linalg),
        'n_overlapping_windows_dropped': int(n_overlap_dropped),
        'overlap_seconds_dropped': [round(float(x), 3) for x in overlap_secs],
        'n_overlap_3s_or_less': int(sum(1 for x in overlap_secs if x <= 3.0)),
    },
    'cohort': {'n_subjects_retained': int(n_sub_fmt),
               'subject_ids': [str(x) for x in SEG_SUBJ],
               'n_segments_per_subject': [int(x) for x in SEG_NSEG]},
    'l2': l2_df.to_dict(orient='records'),
    'l2_robust_bayesian': robust_df.to_dict(orient='records'),
    'monotonicity': mono_out,
    'loso': loso_out,
    'detected': {'FMT_scalp': {'sig': bool(sig_scalp), 'scheme': sig_sch[0]},
                 'FMT_GED':   {'sig': bool(sig_ged),   'scheme': sig_sch[1]}},
}
if W_IVW is not None:
    _summary['ivw'] = {k: float(v) for k, v in ivw_info.items()}


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _sanitise(o):
    """Recursively replace non-finite floats with None.

    `default=` alone is not enough: json calls `default` only for objects it cannot
    serialise itself, so np.float64 NaN goes through _jsonable and becomes null, but a
    plain Python float('nan') (which `{k: float(v) for ...}` and the robust rows
    produce) is serialisable, and with allow_nan=True json.dump writes the bare literal
    NaN, which is not JSON. allow_nan=False below turns any future miss into a raise
    instead of a file nobody can read (as heart_depth_run does).
    """
    if isinstance(o, dict):
        return {k: _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _sanitise(o.tolist())
    return o


_json_path = os.path.join(RESULTS_DIR, 'fmt_results.json')
with open(_json_path, 'w', encoding='utf-8') as _fh:
    json.dump(_sanitise(_summary), _fh, indent=1, default=_jsonable,
              allow_nan=False)
# Read it straight back with the strictest parser available, so an unreadable
# summary fails HERE rather than in whatever reads it next week.
with open(_json_path, encoding='utf-8') as _fh:
    json.load(_fh, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError('fmt_results.json contains the bare literal %r - it is not '
                   'valid JSON' % c)))
print('  compact summary: %s (%.1f kB)'
      % (_json_path, os.path.getsize(_json_path) / 1024))


print(f'\n{"="*72}')
print(f'Done. Results: {RESULTS_DIR}')
print(f'  - fmt_depth_L2.csv')
print(f'  - fmt_depth_profile.png')
print(f'  - run_results.npz')
print(f'{"="*72}')

"""
=============================================================================
SECTION 6: Display results
=============================================================================
"""

print('\n' + '='*72)
print('  L2 GROUP TEST RESULTS')
print('='*72)
print(l2_df.to_string(index=False))
print('\n  n and CohenD are IDENTICAL down each measure x model block: d is the')
print('  unweighted mean/sd of the same slopes, so only t, p and N_eff_kish move')
print('  with the scheme. Equal n/d is the schemes working, not the schemes off.')

print('\n' + '='*72)
print('  DEPTH PROFILE  (BETWEEN-subject means at each level)')
print('  The L2 test above is a WITHIN-subject slope. These can disagree without')
print('  either being wrong: each level averages over whichever subjects reported')
print('  it, and few report all five. A flat or non-monotonic profile here is NOT')
print('  evidence against a significant slope. See the note in Section 3b.')
print('='*72)
for meas in ['scalp', 'ged']:
    print(f'\n  {meas.upper()}:')
    for di, dl in enumerate(range(1, 6)):
        m = prof_mean[meas][di]
        s = prof_sem[meas][di]
        if np.isfinite(m):
            print(f'    Depth {dl}: {m:.4f} +/- {s:.4f}')

print('\n' + '='*72)
print(f'  DETECTED: scalp={"YES" if sig_scalp else "no"}, GED={"YES" if sig_ged else "no"}')
print('='*72)

"""
=============================================================================
SECTION 7: Show depth profile plot
=============================================================================
"""

plot_path = os.path.join(RESULTS_DIR, 'fmt_depth_profile.png')
if os.path.exists(plot_path):
    print(f'Depth profile written: {plot_path}')

"""
=============================================================================
SECTION 8: HOW TO SWEEP

Change a Section 3 parameter (or pass --set) and run again. Each setting writes its
own folder, so nothing overwrites anything. No re-extraction is needed (the .npz hold
90 s segments).

    GED template     GED_TEMPLATE_ELEC = None  /  ['fz']  /  ['cz']
    window           EPOCH_DUR = 30 / 45 / 60
    R2 screen        GED_R2_THRESH = 0.30 / 0.50 / 0.80
=============================================================================
"""

"""
=============================================================================
SECTION 9: REFERENCE (read-only; nothing here runs)

Why the non-obvious settings in Section 3 are what they are. Tags match the [Rn]
markers there.

-----------------------------------------------------------------------------
[R1] MIN_CONF IS NOT A SETTING HERE AND CANNOT BE
  The .npz carry no per-segment confidence. The gate was applied once, at extraction
  (extract_npz.py). It still names the folder (result_path), so asking for a value the
  data was not cut at would create a conf3-labelled duplicate of conf4 data.
  EXTRACT_MIN_CONF is the only record of what the files hold, and the guard under the
  COHORT block raises on a mismatch.

-----------------------------------------------------------------------------
[R2] MIN_SESSION_MIN IS APPLIED HERE, NOT AT EXTRACTION
  extract_npz.py has no duration gate and the .npz carries no duration. The gate is
  enforced against a lower bound read from seg_span (Section 1); the run prints
  each session it drops.

-----------------------------------------------------------------------------
[R3] THE WINDOW IS CUT FROM THE END, NOT THE START
  The extracted segment is
      [report - EXTRACT_PRE_END - EXTRACT_EPOCH_DUR, report - EXTRACT_PRE_END]
  so any window ending at or before report - EXTRACT_PRE_END and no longer than
  EXTRACT_EPOCH_DUR can be cut from it. Set EPOCH_DUR to 30/45/60 and PRE_END to >= 2;
  the loader takes the window ending PRE_END before the report, not the first N seconds
  of the segment. PRE_END must be >= EXTRACT_PRE_END (2): the loader trims the segment
  tail to reach it and cannot extend the segment. PRE_END = 3 is the default here and
  in fmt_depth_pipeline.

-----------------------------------------------------------------------------
[R4] FMT_ELECS IS THE SCALP ROI: 10 RECORDED CHANNELS
  The channels averaged for the FMT_scalp measure. All 10 exist in this dataset.
  Different from the GED template [R5].

-----------------------------------------------------------------------------
[R5] GED_TEMPLATE_ELEC vs FMT_ELECS: DO NOT CONFUSE THEM
  FMT_ELECS = the scalp ROI that FMT_scalp averages.
  GED_TEMPLATE_ELEC = the Gaussian centres of the R2 screen that picks the GED
  component. None means "reuse FMT_ELECS"; a list means those centres instead.

      None      ROI template, 10 Gaussians    -> tag has no tmpl token
      ['fz']    single Fz                     -> tag ...tmpl1x<hash>
      ['cz']    single Cz                     -> tag ...tmpl1x<hash>

  The two template shapes are not on the same R2 scale. A sum of Gaussians over 10
  sites is broader and flatter than one bump, so the same topography scores higher
  against the ROI template. GED_R2_THRESH=0.50 is Zuure et al.'s cutoff, calibrated to
  a single centre; check what it retains before carrying it across (the "ged ok="
  counts in the per-subject lines).

-----------------------------------------------------------------------------
[R6] SIG_PRIORITY ORDER IS THE DETECTION RULE, NOT COSMETICS
  The first scheme in SIG_PRIORITY whose unadj test is significant is the detected
  scheme, which the post-hoc battery uses. With all three schemes listed (in `schemes`
  and in SIG_PRIORITY), a measure significant only under ivw_shrunk is detected too.

-----------------------------------------------------------------------------
[R7] Every swept setting must reach the folder tag
  A setting that is swept but missing from fmt_depth_tag means the next run overwrites
  this one and the difference is invisible afterwards. If you add a parameter and sweep
  it, add it to result_path.fmt_depth_tag in the same edit. `P = PARAMS` is not bound
  until later in this script, so the RUN_TAG block uses PARAMS directly; using P there
  would raise NameError.
=============================================================================
"""
