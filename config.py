# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parameter values of the analyses reported in the paper.

Every run script at the top level of this repository reads its settings from here, so the
values the paper used are written in one place. A setting that is not listed keeps the
default of the function that reads it (functions/).

Paths. HMD_BIDS is the BIDS dataset (default ./data/bids); the questionnaires are read from
its phenotype/ folder and the demographics from its participants.tsv (HMD_BEHAVIOUR overrides
the phenotype folder). HMD_DATA is the folder of per-recording segment files
(<participant>/ses<N>.npz, written by functions/extract_npz.py; default ./data/segments) and
HMD_RESULTS the folder that receives the results (default ./results). The cohort numbers
(analysis/cohort_numbers.py) also read the preprocessing summary sInfo.mat written by
preprocessing/process_raw_data.m (HMD_PREPROCESSING_SUMMARY, several files separated by the
path separator; default ./data/preprocessed/sInfo.mat).
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('HMD_DATA', os.path.join(ROOT, 'data', 'segments'))
RESULTS_DIR = os.environ.get('HMD_RESULTS', os.path.join(ROOT, 'results'))
BIDS_DIR = os.environ.get('HMD_BIDS', os.path.join(ROOT, 'data', 'bids'))
BEHAVIOUR_DIR = os.environ.get('HMD_BEHAVIOUR', os.path.join(BIDS_DIR, 'phenotype'))
# ICLabel probabilities of every recording, written by functions/extract_npz.py
ICLABELS = os.environ.get('HMD_ICLABELS', os.path.join(DATA_DIR, 'ic_labels.npz'))
PREPROCESSING_SUMMARY = os.environ.get(
    'HMD_PREPROCESSING_SUMMARY', os.path.join(ROOT, 'data', 'preprocessed', 'sInfo.mat'))
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# ============================================================================
# Analysis 1: scalp (HEP, HRSP, HRPC at all 64 electrodes, no component selection)
# ============================================================================
# The segment is the unit of analysis; recording session is in the participant-level model.
SCALP = dict(
    IC_SELECT_METHOD='scalp',       # names the arm in the folder tag; nothing is selected
    PRE_END=3,
    WAVELET_ON='segment',           # wavelet on the stored 90 s segment, window cut from it
    PRIMARY_MODEL='session',
    MODELS=('session',),
    BASELINE_METHOD='glm',
    BASELINE_WIN=[-150, -50],
    MIN_SESSION_MIN=45,
    MIN_DEPTHS=2,
    minSeg=5,
    ALPHA=0.05,
    nPerm=500,                      # the pipeline's own cluster test, not reported
)
# The eight settings: window length x baseline correction x block set.
SCALP_SETTINGS = {
    'dur%d_%s_%s' % (dur, 'bl' if bl else 'nobl', 'pooled' if sapo else 'silent'):
        dict(SCALP, EPOCH_DUR=dur, DO_BASELINE_CORRECT=bl, INCLUDE_SAPO=sapo)
    for sapo in (False, True) for dur in (60, 45) for bl in (True, False)
}
SCALP_ROOT = os.path.join(RESULTS_DIR, 'scalp')
# Group test: weighted TFCE over channel x frequency x time. 5,000 permutations for the
# 60 s silent settings and the 45 s silent setting with baseline correction, 1,000 for the
# other five.
SCALP_TFCE = dict(
    scheme='ivw_own',
    measures=('hrsp', 'hep', 'hepc'),
    models=('session',),
    nperm={name: (5000 if name in ('dur60_bl_silent', 'dur60_nobl_silent', 'dur45_bl_silent')
                  else 1000) for name in SCALP_SETTINGS},
)

# ============================================================================
# Analysis 2: component domain (HEP, HRSP, HRPC on one component per recording)
# ============================================================================
# The primary setting: 60 s windows ending 3 s before the depth report, silent blocks,
# baseline correction by regression on the -150 to -50 ms pre-R-peak window, recording
# session in the participant-level model, the heartbeat as the unit of analysis.
COMPONENT = dict(
    # one component per recording, selected by its heartbeat-locked power (Lee et al., 2024)
    IC_SELECT_METHOD='lee',
    HRSP_SEL_CRITERION='hrsp',      # mean squared wavelet magnitude ...
    HRSP_SEL_WIN=[300, 550],        # ... over 300 to 550 ms after the R-peak
    HRSP_SEL_BANDS=[4, 20],         # ... and 4 to 20 Hz
    IC_SEL_NORM='raw',              # absolute power, no per-trial normalisation
    IC_SEL_THRESH=1.0,              # the single highest-scoring component
    # segmentation
    EPOCH_DUR=60,                   # s
    PRE_END=3,                      # s between the end of the window and the report
    WAVELET_ON='segment',           # wavelet on the stored 90 s segment, window cut from it
    INCLUDE_SAPO=False,             # silent blocks only
    # baseline correction (regression form)
    DO_BASELINE_CORRECT=True,
    BASELINE_METHOD='glm',
    BASELINE_WIN=[-150, -50],       # ms
    # participant-level model
    PRIMARY_MODEL='session',
    MODELS=('session',),
    GRAINS=('segment', 'perbeat'),  # heartbeat as the unit (primary); segment (HRPC, sensitivity)
    # the runner's own cluster test is not reported: the group test is the TFCE step
    L2_WINDOWS=('full',),
    POSTHOC=False,
    nPerm=1000,
)

# The eight settings: window length x baseline correction x block set. The first is primary.
COMPONENT_SETTINGS = {
    'dur60_bl_silent': dict(COMPONENT),
    'dur60_nobl_silent': dict(COMPONENT, DO_BASELINE_CORRECT=False),
    'dur45_bl_silent': dict(COMPONENT, EPOCH_DUR=45),
    'dur45_nobl_silent': dict(COMPONENT, EPOCH_DUR=45, DO_BASELINE_CORRECT=False),
    'dur60_bl_pooled': dict(COMPONENT, INCLUDE_SAPO=True),
    'dur60_nobl_pooled': dict(COMPONENT, INCLUDE_SAPO=True, DO_BASELINE_CORRECT=False),
    'dur45_bl_pooled': dict(COMPONENT, EPOCH_DUR=45, INCLUDE_SAPO=True),
    'dur45_nobl_pooled': dict(COMPONENT, EPOCH_DUR=45, INCLUDE_SAPO=True,
                              DO_BASELINE_CORRECT=False),
}
PRIMARY = 'dur60_bl_silent'
COMPONENT_ROOT = os.path.join(RESULTS_DIR, 'component')

# Confound models: one regressor added to the primary model at a time. Ongoing band power
# (theta 5-8, alpha 8-13, beta 13-30 Hz) is computed on the analysed component or averaged
# over all channels, which needs one run each.
CONFOUND_SETTINGS = {
    'component': dict(
        COMPONENT, GRAINS=('perbeat',), STATE_BAND_SOURCE='component',
        MODELS=('session', 'session+rr', 'session+ecgAmp', 'session+ecgAmpT',
                'session+respAmp', 'session+respPhase',
                'session+theta', 'session+alpha', 'session+beta')),
    'channels': dict(
        COMPONENT, GRAINS=('perbeat',), STATE_BAND_SOURCE='channels',
        MODELS=('session', 'session+theta', 'session+alpha', 'session+beta')),
}
CONFOUND_ROOTS = {k: os.path.join(RESULTS_DIR, 'confounds_' + k) for k in CONFOUND_SETTINGS}

# Group test: weighted threshold-free cluster enhancement on the participants' slope maps.
# E = 0.5, H = 2, step 0.1 and the seed are the defaults of functions/l2_tfce.py.
TFCE = dict(
    nperm=5000,
    schemes=('ivw_own', 'unweighted'),   # precision weights (primary), unweighted (alongside)
    measures=('hrsp', 'hep', 'hepc'),
    grains=('perbeat', 'segment'),
    models=('session',),
)

# Surrogate heartbeat control of the primary effect.
SURROGATE = dict(
    n_surrogate=50,
    max_unshiftable=0.25,           # largest share of recordings a draw may leave out
    stat='both',                    # the paper quotes the TFCE statistic
    tfce_nperm=5000,
    grain='perbeat', measure='hrsp', model='session',
    schemes=('unweighted', 'ivw_own'),
)

# Leave one participant out, on the primary effect.
LOSO = dict(grain='perbeat', model='session', measure='hrsp', schemes=('ivw_own',), nperm=5000)
# Where the selected components sit (eLORETA) and the ongoing band power of the selected
# component against depth, both on the primary setting. The forward model is cached.
SOURCES = dict(cache_dir=os.path.join(RESULTS_DIR, 'head_models'))
ONGOING = dict(band='beta', n_perm=10000)

# ============================================================================
# Analysis 3: frontal-midline theta as a function of depth
# ============================================================================
# Two measures per segment: theta power averaged over ten frontocentral electrodes, and a
# GED-filtered component (Zuure et al., 2020). Values are z-scored within recording.
FMT_DEPTH = dict(
    THETA_BP=[5, 8],                # Hz
    EPOCH_DUR=60,
    PRE_END=3,
    FMT_ELECS=['fz', 'f1', 'f2', 'f3', 'f4', 'fc1', 'fc2', 'fc3', 'fc4', 'cz'],
    GED_TEMPLATE_ELEC=None,         # template centred on FMT_ELECS
    GED_SHRINKAGE=0.01,
    GED_R2_THRESH=0.15,             # template match needed to accept a component
    GED_TOP_N=2,                    # leading components screened
    GED_MAX_COMP=1,
    STANDARDIZE_WITHIN_SESSION=True,
    L1_METHOD='WLS',
    nPerm=1000,
)
FMT_DEPTH_SETTINGS = {
    'silent': dict(FMT_DEPTH, INCLUDE_SAPO=False),
    'pooled': dict(FMT_DEPTH, INCLUDE_SAPO=True),
}
FMT_ROOT = os.path.join(RESULTS_DIR, 'fmt')
# Self-report associations of the theta depth slope, with the method of ASSOCIATIONS below.
# The predictor is the slope of the theta measure that tracked depth: the GED component in
# the pooled blocks.
FMT_ASSOCIATIONS = dict(setting='pooled', measure='GED', n_boot=10000)

# ============================================================================
# Analysis 4: contrasts between recording periods
# ============================================================================
PERIODS = ['baseline', 'pre-meditation', 'post-meditation', 'active-oddball']
PERIOD_ROOT = os.path.join(RESULTS_DIR, 'periods')

# Heartbeat-related measures in one component per recording, selected on the baseline and
# pre-meditation periods only, so that the selection never sees the tested contrast.
PERIOD_HEART = dict(
    IC_SELECT_METHOD='lee',
    SELECT_ON=['baseline', 'pre-meditation'],
    HRSP_SEL_CRITERION='hrsp',
    HRSP_SEL_WIN=[300, 550],
    HRSP_SEL_BANDS=[4, 20],
    IC_SEL_NORM='raw',
    IC_SEL_THRESH=1.0,
    PERIODS=PERIODS,
    CONTRASTS='all',                # the six pairwise contrasts, uncorrected across them
    EPOCH_MS=[-300, 700],
    TLIMS=[-100, 600],
    HRSP_FREQS=list(range(4, 31)),  # 4 to 30 Hz
    HRSP_WAVE_WIDTH=5.0,
    MEASURES=['hep', 'hrsp', 'hepc'],
    BASELINE_METHOD='glm',          # baseline correction by regression
    BASELINE_WIN=[-150, -50],
    MIN_IBI_MS=600,
    MIN_BEATS_PER_PERIOD=50,
    DROP_EDGE_BEATS=True,
    WAVELET_ON='block',             # wavelet on each period's continuous block, then epochs
    EDGE_GUARD_S=2.0,               # s of real data required beyond each epoch
)
PERIOD_HEART_TFCE = dict(schemes=('ivw_own', 'unweighted'), measures=('hrsp', 'hep', 'hepc'),
                         nperm=5000)

# Frontal-midline theta in the same four periods (4 s windows; filter estimated on the
# baseline and pre-meditation periods only).
PERIOD_FMT = dict(
    THETA_BP=[5, 8],
    # FCz (the online reference) and AFz (the ground) are not recorded in this dataset and are
    # skipped, so the period ROI is the other six electrodes.
    FMT_ELECS=['Fz', 'FCz', 'Cz', 'AFz', 'F1', 'F2', 'FC1', 'FC2'],
    GED_TEMPLATE_ELEC='ROI',
    GED_R2_THRESH=0.30,
    GED_TOP_N=7,
    GED_SHRINKAGE=0.01,
    GED_TRAIN_ON=['baseline', 'pre-meditation'],
    PERIODS=PERIODS,
    CONTRASTS='all',
    MIN_WINDOWS=10,
    STANDARDIZE_WITHIN_SESSION=True,
    N_PERM=10000,
    SCHEMES=['unweighted', 'precision_nwin'],
)

# ============================================================================
# Autonomic variables as a function of depth
# ============================================================================
# Mean inter-beat interval, RMSSD, respiratory rate and respiration amplitude on the 60 s
# windows of the silent blocks (both pinned inside functions/autonomic_depth.py).
AUTONOMIC_DIR = os.path.join(RESULTS_DIR, 'autonomic')

# ============================================================================
# Self-report and demographic associations of the primary effect
# ============================================================================
# Predictor: each participant's depth slope averaged over the significant region of the
# primary test (largest TFCE component), z-scored. Skipped correlations with 10,000 bootstrap
# resamples for the 34 self-report measures; Spearman on all participants for the six
# demographic variables.
ASSOCIATIONS = dict(grain='perbeat', model='session', measure='hrsp', scheme='ivw_own',
                    n_boot=10000)
ASSOCIATIONS_DIR = os.path.join(RESULTS_DIR, 'associations')

# The primary test in subsets of participants defined by the experimenters' rating of EEG
# viability (1 ideal, 2 permissible, 3 to be corroborated with quality checks, 4 missing but
# usable), shipped with this repository: metadata/eeg_viability_ratings.tsv (participant_id,
# eeg_viability; levels in the .json beside it).
VIABILITY_RATINGS = os.environ.get(
    'HMD_VIABILITY', os.path.join(ROOT, 'metadata', 'eeg_viability_ratings.tsv'))
