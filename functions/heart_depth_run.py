# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
heart_depth_run: the end-to-end HEART depth runner (Lee selection path).

`heart_depth_pipeline` is a library: it turns one .npz into per-segment measures
and one subject's sessions into a slope map, and stops there. This module is the
driver that goes

    segment files (data/segments, config.DATA_DIR) -> cohort gates
        -> per-session measures -> per-subject L1 -> L2
        -> a parameter-tagged result folder

in one call. Every statistic it reports is computed by a helper module:

    heart_depth_pipeline.session_measures   per-session HEP / HRSP / HEPC
    heart_depth_pipeline.subject_slopes     L1 depth slopes (cell_ols inside)
    l2_heart.run_l2_heart                   L2 cluster permutation x 3 schemes
                                            (calls decide_detected itself)
    result_path.result_dir/write_readme     where the answer is allowed to land

HEPC, as the code names it, is the paper's HRPC (heartbeat-related phase
consistency).

What is new here is the cohort accounting, the EPOCH_DUR / PRE_END window
bookkeeping, and the write order that keeps a finished L1/L2 alive when
something after it throws.

Parameters that reach the folder name
-------------------------------------
`result_path.heart_depth_tag` reads EPOCH_DUR, PRE_END, PRIMARY_MODEL and nPerm
with `.get(key, default)`. A params dict that lacks them still produces a
plausible folder name, which would be wrong for any run not at exactly those
values, so two different analyses would land in one folder and the second
would replace the first. `DEFAULT_RUN_PARAMS` therefore carries all four
explicitly, and `_check_tag_params` refuses to build a path if any is missing.

MIN_SESSION_MIN: the recording length is not in the .npz
--------------------------------------------------------
The gate is on the recording length ("a recording this short is an acquisition
failure, not a short session"), which extract_npz.py does not write into the
.npz. A lower bound is used instead, and said so:

    dur_lb = max(block_spans[:,1].max(), seg_span[:,1].max()) / fs / 60

i.e. the last sample the extractor referred to. The true recording is at least
this long (the post-meditation block, cool-down and any trailing rest come
after it). Consequences, both reported:

  * A session whose lower bound clears the gate certainly clears the real gate.
  * A session below the bound might be above the gate in reality. It is
    dropped, and `_report_duration_proxy` names it and says the drop rests on a
    lower bound. `BORDERLINE_MARGIN_MIN` also warns for any kept session within
    15 min of the gate, where the proxy is too weak to decide either way.

On the shipped segment files (data/segments, config.DATA_DIR) the lower
bounds lie far from the 45 min gate on either side, so the gate is unambiguous
on this cohort; `_report_duration_proxy` prints their range at every run.

The epoch window belongs to session_measures
--------------------------------------------
`extract_npz.py` stores each segment as

    [report - EXTRACT_PRE_END - EXTRACT_EPOCH_DUR,  report - EXTRACT_PRE_END]
    = [report - 2 - 90,                             report - 2]

so the segment ends just before the report and the analysed window is its
tail; `[..., :epoch_samples]` would take [report-92, report-32] at EPOCH_DUR=60,
a 30 s shift onto a different stretch of the meditation. The cut is
session_measures' job: this module passes EPOCH_DUR and PRE_END through in `p`
and does not touch the samples (cutting here as well would shift every segment
a second time). `window_indices` is imported from the library for the
accounting and the preflight only.

A segment that cannot hold the window is refused, not clamped: the window is
anchored to the segment start (heart_depth_pipeline, GATE 1b), and a segment
that does not reach `w1` is dropped with a reason, because the missing samples
are the ones nearest the report. `n_seg_short` counts these refused segments,
and the refusal is about anchoring rather than raw length.

Matching the window does not guarantee the full segment set: extract_npz
applied its own ±3 s duration tolerance at 90 s, so a segment that would pass
at 60 s may never have been written to the .npz.

Why this module rebuilds the .npz in memory
-------------------------------------------
For the condition filter (next section). `session_measures` takes a path and
does its own `np.load`, so the only way to hand it a different segment set is
to hand it a different file. `_minimal_session_npz` writes the filtered arrays
into an in-memory `io.BytesIO` npz. The samples are copied verbatim at full
stored length, so the library still sees exactly the segments extract_npz
wrote and still makes its own cut. If the filter removes nothing, the original
path is passed straight through and no copy is made.

Cost when a copy is made: a 33-segment 90 s session is ~190 MB of float32 EEG,
so the transient is ~3x that before session_measures allocates its own. Peak is
set by the largest session, because one session is resident at a time, and the
dominant allocation of the run is elsewhere: the Lee selector's pooled CWT,
~4 GB at EPOCH_DUR=60 on this cohort's largest session.

Condition filter
----------------
`session_measures` filters conditions only when INCLUDE_SAPO is False
(`keep &= (cond == 0)`). With INCLUDE_SAPO True it would also keep cond 2 and
3, the pre- and post-meditation depth reports, which no depth model reads. This
runner keeps cond in {0, 1} (or {0} when INCLUDE_SAPO is False) and says so.

Numerically this changes almost nothing: R-peaks are kept only inside the
meditation blocks, so a pre-/post-meditation segment has no beats and would be
dropped at MIN_BEATS_UNIT anyway. It changes the accounting: a cond-2 segment
reported as "dropped: too few clean beats" is a misleading reason, and it would
let a pre-meditation depth level count towards the per-session MIN_DEPTHS gate.

Write order: a post-hoc must never destroy a finished L1/L2
-----------------------------------------------------------
The order is fixed and every write is atomic (`tmp` + `os.replace`):

    1. cohort_sessions.csv / cohort_subjects.csv   (as soon as the cohort exists)
    2. run_results.npz with the L1 maps            before any L2 is attempted
    3. l2_heart_summary_<model>[_<window>].csv     per (model, window), each
                                                   inside its own try/except
    4. posthoc_summary.csv                         always, even when empty
    5. run_results.npz rewritten with the L2 rows  (atomic; a crash mid-write
                                                   leaves step 2's file intact)
    6. README.txt + run_summary.txt                the human record, last

An L2 failure for one (model, window) is recorded in `l2_errors` and every other
(model, window) still runs. Nothing after step 2 can leave the L1 maps
unrecoverable.

Scope and conventions
---------------------
  * Models: MODELS lists the L1 designs to fit ('session' is the primary; the
    covariate, moderation and session+covariate models come from
    heart_depth_pipeline). Every model writes to main/, with the model in the
    file name.
  * Grains: `cell`, `segment` and `perbeat` are produced (p['GRAINS'], default
    all three), each into its own `<grain>_wise/` tree with its own L1, L2,
    post-hocs, npz and summary, off one cohort load. Not every measure exists at
    every grain (HEPC has no per-beat form) and ivw_shrunk is segment-only; see
    GRAIN_MEASURES / GRAIN_SCHEMES.
  * The per-beat variance is the cluster-robust (CRVE) estimator, clustered on
    segment. It does not reach L2 (GRAIN_SCHEMES gives the per-beat grain only
    unweighted and precision_nseg, neither of which reads svar); it is stored
    in run_results.npz.
  * IC_SELECT_METHOD: only 'lee' (functional single-component selection) is
    implemented; any other value is refused.
  * Post-hocs: LOSO and the depth-shape tests run here
    (posthoc_heart.run_posthocs, gated on an effect surviving the main model,
    controlled by POSTHOC, default True) and land in main/posthoc_summary.csv.
    The surrogate R-peak control, source localization, mediation and outcomes
    are not run by this module, and neither are the figures (the scripts in
    figures/ read run_results.npz and the group tables). run_summary.txt derives
    its own ran / not-run lists from the parameters in force.
  * Permutations use numpy's PCG64 (see l2_heart).
  * MIN_CONF is not enforced here: `extract_npz.py` applied conf >= 4 when the
    .npz files were written, so this module cannot re-apply or relax it. It is
    carried in the params (and the folder tag) as a record of what the data was
    extracted at, and `_check_extraction_params` raises for any other value.

run_component_analysis.py calls `run(data_dir, results_root, params, limit=...)` once
per setting, with `params` = DEFAULT_RUN_PARAMS updated by that setting.
"""
from __future__ import annotations

import os
import io
import sys
import csv
import json
import time
import traceback          # a failed grain writes its frames, not just its message
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from heart_depth_pipeline import (DEFAULT_PARAMS, session_measures,
                                  subject_slopes, list_sessions,
                                  tlims_mask,
                                  # The covariate-model registry.
                                  # Imported rather than restated so a
                                  # model added to the pipeline cannot be
                                  # silently unknown to the runner.
                                  ALL_MODELS, SEGCOV_MODELS, _base_of)
# Park 2018 / Coll 2020 / Lee 2024 reporting window. Inclusive at both ends,
# in milliseconds.
# This is the default, not the value in force: read it through `lit_ms(p)`,
# never directly, since a run may set p['LIT_MS'] and a hardcoded read would
# then test one window while the summary and the folder claim another.
LIT_MS = (200, 600)


def lit_ms(p):
    """The literature window this run is using -> (lo_ms, hi_ms).

    Validated, because a reversed or single-point window silently produces an
    L2 over nothing and the 'lit' table then reads as a clean null.
    """
    v = p.get('LIT_MS') if isinstance(p, dict) else None
    if v is None:
        return tuple(LIT_MS)
    try:
        lo, hi = (float(v[0]), float(v[1]))
    except (TypeError, ValueError, IndexError):
        raise ValueError(
            f'LIT_MS must be a 2-element (lo_ms, hi_ms), got {v!r}. It sets the '
            f'window the literature test runs over — a malformed one would '
            f'produce an empty L2 that reads as a null.')
    if not (hi > lo):
        raise ValueError(f'LIT_MS {v!r}: hi must be greater than lo.')
    return (lo, hi)

from ecg_quality import subject_ecg_clean_ratio, ecg_ratio_passes_gate
from posthoc_heart import run_posthocs, POSTHOC_FIELDS
from l2_heart import run_l2_heart, WEIGHT_SCHEMES, MIN_SUB_L2
# Imported, not reimplemented: `_emit` is print() reduced to whatever the
# current stdout can encode (a bare print of a warning glyph raises
# UnicodeEncodeError on a cp1252 console). A local copy of a shared helper would
# silently shadow it, so this is an import even though the name is private to
# l2_heart.
from l2_heart import _emit
from session_length import session_duration_lowerbound_min  # noqa: E402,F401  (re-export)
from result_path import (result_dir, write_readme, heart_depth_tag, heart_method_tag,
                         heart_run_tag, assert_no_param_collision)

__all__ = [
    'DEFAULT_RUN_PARAMS', 'EXTRACT_EPOCH_DUR', 'EXTRACT_PRE_END',
    'window_indices', 'session_duration_lowerbound_min', 'build_cohort',
    'fit_l1', 'run', 'run_paths',
    'load_l2_cells', 'models_after_cond_rule',
]

# ═══════════════════════════════════════════════════════════════════════════
# WHAT THE .npz FILES WERE EXTRACTED AT
# ═══════════════════════════════════════════════════════════════════════════
# These are facts about the data on disk (as written by extract_npz.py), not
# knobs, and are defined below the cond rule. Every run's window is cut out of
# that segment, so a run can ask for a shorter epoch or a larger PRE_END and
# nothing else. Re-extract if you need more; do not edit these to make an
# impossible request look possible.
# ── THE cond RULE, IN ONE PLACE ─────────────────────────────────────────────
# `run()` drops cond/condXdepth when INCLUDE_SAPO is False and stamps the
# reduced model list into .run_params.json, since it cannot fit them. A caller
# deciding whether an arm is already done must compare against that reduced
# list, not against its own grid (which still has cond), or every finished arm
# reads as a collision. So callers ask this function what run() will fit.
COND_MODELS = frozenset({'cond', 'condXdepth'})


def models_after_cond_rule(models, include_sapo):
    """MODELS as `run` will actually fit them, given INCLUDE_SAPO.

    With INCLUDE_SAPO False every retained segment has cond==0, so `cond` is a
    constant column and `condXdepth` a product with a constant. Both are
    undefined and `run` removes them; this reports that without running.
    """
    models = tuple(models)
    if include_sapo:
        return models
    return tuple(m for m in models if m not in COND_MODELS)


EXTRACT_EPOCH_DUR = 90.0        # s of data stored before the report
EXTRACT_PRE_END = 2.0           # s excluded before the report at extraction
EXTRACT_MIN_CONF = 4            # confidence floor applied at extraction

# A KEPT session whose duration lower bound sits within this many minutes of
# MIN_SESSION_MIN is one the proxy is too weak to adjudicate: the true duration
# is above the bound, but so is the bound's own uncertainty. Warned, never
# silently kept. Never fires on the shipped cohort, whose kept bounds lie well
# above the 45 min gate.
BORDERLINE_MARGIN_MIN = 15.0

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_RUN_PARAMS = dict(DEFAULT_PARAMS)
DEFAULT_RUN_PARAMS.update(
    # ── The four that heart_depth_tag reads through a silent .get() ─────────
    # Present here explicitly; see the module docstring.
    EPOCH_DUR=60,
    PRE_END=3,               # not EXTRACT_PRE_END, the geometry of the segments
                             # stored in the .npz; that is separate.
    PRIMARY_MODEL='session',
    # nPerm sets the p-value resolution and is in the folder tag, but p-values
    # at 1000 and 10000 permutations are not comparable. Raise to 10000 for a
    # final number.
    nPerm=1000,

    # ── which L1 designs to fit ────────────────────────────────────────────
    # `subject_slopes` builds each covariate column from the per-segment
    # covariates session_measures computes (heart_depth_pipeline.COVARIATE_MODELS).
    #
    # Every model writes to main/, with the model in the file name
    # (`l2_heart_summary_<model>.csv`), so nothing collides; readers glob
    # `l2_heart_summary_*.csv` and read the model off the stem.
    #
    # 'cond' is dropped automatically when INCLUDE_SAPO is False: a run with no
    # sapo segments has a constant cond column, which is rank deficient.
    #
    # ── moderation ─────────────────────────────────────────────────────────
    # `<cov>Xdepth` fits [intercept, depth, X, depth*X] and sends the
    # interaction coefficient to L2. It answers a different question from the
    # covariate model beside it: not 'is the depth effect explained by X'
    # (confounding) but 'is the depth effect stronger at some values of X'
    # (moderation). A cluster in one of these is not a depth effect; it says the
    # depth slope varies with X. respPhase has no moderation model (two columns,
    # sin and cos, so there is no single signed map to test; see
    # INTERACTION_MODELS). These designs cost 4 parameters instead of 2, against
    # ~4-10 cells per subject at the cell grain, so more subjects are refused
    # there and a moderation map can rest on many fewer subjects than its unadj
    # sibling; l1_fit_summary.csv records that per model.
    MODELS=('session', 'unadj', 'reportmethod', 'rr', 'respAmp', 'respPhase',
            'ecgAmp', 'ecgAmpT', 'cond'),
    # The moderation models are implemented and tested but off by default.
    # Add any of these to MODELS to fit them:
    #   'rrXdepth' 'ecgAmpXdepth' 'ecgAmpTXdepth' 'respAmpXdepth'
    #   'reportmethodXdepth' 'condXdepth'
    # Off by default because moderation is a follow-up on an effect already
    # found, and because they refuse more subjects than unadj does (see above).
    # 'condXdepth' is the most fragile (cond is binary, so it is near-degenerate
    # at the cell grain, with every subject exactly identified); prefer the
    # segment grain for it.
    MEASURES=('hep', 'hrsp', 'hepc'),

    # ── L2 ──────────────────────────────────────────────────────────────────
    ALPHA=0.05,              # also sets the cluster-forming threshold
    L2_SCHEMES=WEIGHT_SCHEMES,   # all three: unweighted / precision_nseg / ivw_shrunk
    L2_IVW_MEASURE='hep',    # HEP's slope variance weights every measure
    L2_MAX_CLUSTERS=5,
    # The second L2 window. The primary analysis runs over the whole TLIMS axis
    # (-100..600 ms). 'lit' also runs L2 restricted to LIT_MS, the window the
    # HEP literature reports in (Park 2018, Coll 2020, Lee 2024), so a cluster
    # that exists only because the test ranged over 700 ms of pre- and peri-QRS
    # signal is separable from one in the window those papers looked at. This is
    # a second test of the same data, not a replacement, and it is not corrected
    # against the full-window test. Read it as a targeted confirmation, never as
    # "the effect is significant in one of two windows".
    L2_WINDOWS=('full', 'lit'),
    # LIT_MS is a knob but is deliberately not listed here; read it with
    # `lit_ms(p)` (absent, it is the module default (200, 600)). `_comparable`
    # compares the recorded parameter stamp key for key, so a key added to the
    # defaults would make every folder written before it read as a parameter
    # collision. Set p['LIT_MS'] only to move the window; it then shows up as a
    # difference and needs its own RUN_LABEL.
    # Post-hocs are gated: the battery runs only on an effect that survived the
    # main model, and only on that effect (a forced-fallback post-hoc is how a
    # null becomes a finding). See posthoc_heart's docstring for what is not in
    # it and why.
    POSTHOC=True,
    POSTHOC_NPERM=1000,
    L2_MAX_GB=4.0,
    L2_POSTHOC_SIGNFLIP=True,

    # Every grain in this tuple gets its own <grain>_wise/ tree (its own L1, L2,
    # post-hocs, npz and summary) off one cohort load. The cohort does not
    # depend on the grain (minSeg gates segment counts at every grain), so a
    # second grain costs one extra L1+L2, not a second run. The grain is not a
    # folder-tag axis: it is already a path level (<grain>_wise/), and putting
    # it in the tag too would give one analysis two names.
    GRAINS=('cell', 'segment', 'perbeat'),

    # ── recorded, not enforced (see the module docstring) ───────────────────
    MIN_CONF=EXTRACT_MIN_CONF,
    SESSION_DUR_SOURCE='npz_span_lowerbound',
)

# The four the tag reads through .get(). Missing = a wrong folder name, silently.
_TAG_REQUIRED = ('EPOCH_DUR', 'PRE_END', 'PRIMARY_MODEL', 'nPerm')

ARM = 'HEART_DEPTH'

# GRAIN is the default, GRAINS_IMPLEMENTED the capability, and p['GRAINS'] what
# this run will produce. A grain that is not implemented must be refused by
# name, never silently swapped for `segment`: a run that analysed segments and
# wrote them into `perbeat_wise/` would be worse than one that never ran.
GRAIN = 'segment'
GRAINS_IMPLEMENTED = ('cell', 'segment', 'perbeat')

# The per-beat grain needs the beats themselves, and session_measures keeps
# them only under p['KEEP_PERBEAT'], because the arrays are large (the per-beat
# HRSP power is roughly 2-3 GB across the cohort, on top of a 0.8 GB run).
# run() sets the flag from `GRAINS` and prints that it did.
# Not every measure and not every weighting scheme exists at every grain:
#     cell     unweighted, precision_nseg   HEP HEPC HRSP
#     segment  + ivw_shrunk                 HEP HEPC HRSP
#     perbeat  unweighted, precision_nseg   HEP HRSP
#
#   * HEPC has no per-beat form. PPC is defined over a set of phasors and is
#     meaningless for one beat, so running it would produce a number with no
#     estimand.
#   * IVW needs a usable per-subject variance, and at the cell grain a subject
#     with as many cells as predictors is exactly identified: cell_ols returns
#     svar = Inf, so it enters `unweighted` at full weight and IVW at zero. An
#     IVW cell-grain estimate would then be a weighted mean over a silently
#     different subset of the cohort than the unweighted one beside it.
# A grain absent from a map keeps the full list, so adding a grain does not
# inherit a restriction meant for another one.
GRAIN_MEASURES = {'perbeat': ('hep', 'hrsp')}
GRAIN_SCHEMES = {'cell': ('unweighted', 'precision_nseg'),
                 'perbeat': ('unweighted', 'precision_nseg')}


def grain_measures(p, grain):
    """The measures this grain fits, after the grain map and the HEPC rule.

    HEPC is skipped whenever DO_BASELINE_CORRECT is on. HEPC takes no baseline
    column at all (`subject_slopes` branches on `measure.upper() != 'HEPC'`, and
    `glm_baseline` raises if asked for one), so a `bl` run and a `nobl` run
    produce a byte-identical HEPC result: same maps, same fixed-seed sign-flip
    null, same p. Computing it under both switches would give the same number
    twice under two folder names, and reading the two as agreeing would be
    reading one result as a replication. HEPC therefore comes from the `nobl`
    arm. Set `HEPC_UNDER_BASELINE=True` to override; the only use is to
    demonstrate the invariance.
    """
    keep = GRAIN_MEASURES.get(grain)
    ms = tuple(p['MEASURES'])
    if keep is not None:
        ms = tuple(m for m in ms if m in keep)
    if (p.get('DO_BASELINE_CORRECT') and not p.get('HEPC_UNDER_BASELINE')
            and 'hepc' in ms):
        ms = tuple(m for m in ms if m != 'hepc')
    return ms


def grain_schemes(p, grain):
    keep = GRAIN_SCHEMES.get(grain)
    sc = tuple(p['L2_SCHEMES'])
    return sc if keep is None else tuple(x for x in sc if x in keep)


# Empty: all three grains are implemented. check_grains still consults it, so
# that a future grain can be refused by name with a reason rather than silently
# reduced to segments.
GRAINS_NOT_PORTED = {}


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER VALIDATION: refusals, never repairs
# ═══════════════════════════════════════════════════════════════════════════
def _check_tag_params(p):
    """Refuse to build a result path from a params dict the tag cannot describe.

    `heart_depth_tag` would return `..._dur60_pre2_..._prim-session` for a dict
    containing none of those keys, and that folder is wrong for every run not
    at those exact values, which is how two analyses end up in one folder with
    the second overwriting the first. A missing key is an error here, not a
    default.
    """
    missing = [k for k in _TAG_REQUIRED if k not in p or p[k] is None]
    if missing:
        raise ValueError(
            'params is missing ' + ', '.join(missing) + ' — result_path.'
            'heart_depth_tag reads these through .get() with a silent default, '
            'so the folder name would describe a run that did not happen. Pass '
            'them explicitly (DEFAULT_RUN_PARAMS does).')
    # Validated against the models subject_slopes implements, over the whole of
    # MODELS and not just PRIMARY_MODEL, so a typo in a non-primary entry is
    # caught before the cohort load rather than mid-run in fit_l1.
    _bad = [m for m in tuple(p['MODELS']) if m not in ALL_MODELS]
    if _bad:
        raise ValueError(
            f'MODELS contains {_bad!r}, which subject_slopes does not '
            f'implement. Valid models are {ALL_MODELS}.')
    if p['PRIMARY_MODEL'] not in ALL_MODELS:
        raise ValueError(
            f"PRIMARY_MODEL={p['PRIMARY_MODEL']!r} is not a model "
            f'subject_slopes implements. Valid models are {ALL_MODELS}.')
    if p['PRIMARY_MODEL'] not in tuple(p['MODELS']):
        raise ValueError(
            f"PRIMARY_MODEL={p['PRIMARY_MODEL']!r} is not in MODELS="
            f"{tuple(p['MODELS'])!r}: the folder tag would name a model the run "
            f'never fits.')
    # BASELINE_METHOD names the folder but nothing in the analysis reads it:
    # glm_baseline uses its own constant, 'glm'. result_path puts whatever is
    # set into the tag, so BASELINE_METHOD='none' would name a folder
    # `bl-150--50none` for a run that did a GLM baseline. To turn the baseline
    # off, set DO_BASELINE_CORRECT=False (tag: `nobl`).
    if p.get('DO_BASELINE_CORRECT') and str(p.get('BASELINE_METHOD', 'glm')) != 'glm':
        raise ValueError(
            f"BASELINE_METHOD={p.get('BASELINE_METHOD')!r} is not implemented — "
            f"glm_baseline only does 'glm' and does not read this key, so the "
            f"run would apply a GLM baseline while the folder tag claimed "
            f"{p.get('BASELINE_METHOD')!r}. To run WITHOUT a baseline set "
            "DO_BASELINE_CORRECT=False, which tags the folder 'nobl'.")

    if str(p.get('IC_SELECT_METHOD', 'lee')) != 'lee':
        raise ValueError(
            f"IC_SELECT_METHOD must be 'lee', got {p.get('IC_SELECT_METHOD')!r}.")

    # ── PPC + a sham norm: inert, but the folder name says otherwise ─────────
    # `use_sham` is gated on both knobs (in lee_selection:
    #     use_sham = norm_mode in ('sham', 'sham_raw') and criterion == 'hrsp'
    # ), so under 'ppc' no sham is drawn and the scores are byte-identical to
    # IC_SEL_NORM='raw'. But `_selection_token` appends the norm
    # unconditionally, so the run lands in a folder called `lee-ppc-shamraw_...`
    # and README.txt lists IC_SEL_NORM=sham_raw / IC_SEL_SHAM_N=50 with no
    # caveat, while the equivalent `lee-ppc-raw_...` run would produce identical
    # numbers under a different name (two folders, one result, which
    # assert_no_param_collision cannot catch).
    # Not an error (the setting is harmless), but it must not be silent.
    if ('sham' in str(p.get('IC_SEL_NORM', ''))
            and str(p.get('HRSP_SEL_CRITERION', 'hrsp')) != 'hrsp'):
        import warnings
        warnings.warn(
            "IC_SEL_NORM=%r with HRSP_SEL_CRITERION=%r: the sham branch is "
            "gated on criterion=='hrsp', so NO sham is drawn and selection is "
            "identical to IC_SEL_NORM='raw'. The folder name will still read "
            "'%s' and README.txt will still list IC_SEL_SHAM_N — neither "
            "reflects what ran. Set IC_SEL_NORM='raw' for a truthful name."
            % (p.get('IC_SEL_NORM'), p.get('HRSP_SEL_CRITERION'),
               str(p.get('IC_SEL_NORM', '')).replace('_', '')),
            UserWarning, stacklevel=2)


def check_grains(p):
    """-> the tuple of grains this run will produce. Refuses an unknown one.

    A grain this route cannot fit must never fall back to `segment`: that would
    write segment results into <grain>_wise/ where nothing downstream could
    tell them apart from the real thing. GRAINS_NOT_PORTED carries a reason per
    grain so the error says what is missing; it is empty (all three grains are
    implemented).
    """
    g = tuple(p.get('GRAINS') or (GRAIN,))
    if not g:
        raise ValueError('GRAINS is empty - a run must produce at least one grain.')
    for x in g:
        if x in GRAINS_IMPLEMENTED:
            continue
        if x in GRAINS_NOT_PORTED:
            raise NotImplementedError(
                f'GRAINS includes {x!r}, which is not implemented here: '
                f'{GRAINS_NOT_PORTED[x]} Remove it from GRAINS.')
        raise ValueError(
            f'unknown grain {x!r}; implemented here: {GRAINS_IMPLEMENTED}, '
            f'known-but-not-ported: {tuple(GRAINS_NOT_PORTED)}')
    if len(set(g)) != len(g):
        raise ValueError(f'GRAINS lists a grain twice: {g}. The second pass '
                         f'would overwrite the first in the same folder.')
    # Order is not cosmetic: 'perbeat' last, always. It is the only grain that
    # can exhaust memory (it holds every beat of every session), and a run that
    # dies there must leave the cell and segment trees already complete on disk
    # rather than taking them down with it. The caller's order is otherwise
    # preserved.
    return tuple(sorted(g, key=lambda x: x == 'perbeat'))


def perbeat_bytes_estimate(sessions_by_subject, p):
    """(bytes, note): the per-beat arrays already held, measured not guessed.

    Called AFTER build_cohort, so this reports what is actually in memory rather
    than predicting it. `session_measures` stores `perbeat['pow']` as float32
    [nBeats x nFreq x nTime] on the FULL epoch axis, which is by far the largest
    thing this route ever holds.
    """
    tot = n_b = 0
    for sess in sessions_by_subject.values():
        for s in sess:
            pb = s.get('perbeat')
            if not pb:
                continue
            n_b += int(np.asarray(pb['seg']).size)
            for k in ('hep', 'pow'):
                a = pb.get(k)
                if a is not None:
                    tot += int(np.asarray(a).nbytes)
    return tot, f'{n_b} beat(s) held across the cohort'


def primary_grain(p):
    """The grain whose results this run reports at the top of its return dict.

    `segment` when it is being produced, since every reader of this tree was
    written against it. Otherwise the first grain requested, so a cell-only run
    still reports one.
    """
    g = check_grains(p)
    return GRAIN if GRAIN in g else g[0]


def _check_extraction_params(p):
    """Refuse a window or a MIN_CONF the .npz files cannot honour.

    Every one of these is a case where the run would produce a plausible-looking
    result whose folder name is a lie about what was analysed.
    """
    # EXTRACT_* are read from `p` because session_measures reads them from `p`.
    # They describe the data on disk, so a value differing from what
    # extract_npz.py wrote would make the library cut the wrong window. Pinned
    # to the module constants so a stray override cannot pass silently.
    for key, truth in (('EXTRACT_EPOCH_DUR', EXTRACT_EPOCH_DUR),
                       ('EXTRACT_PRE_END', EXTRACT_PRE_END)):
        if key not in p:
            raise ValueError(
                f'params is missing {key}: heart_depth_pipeline.session_measures '
                f'reads it to cut the analysis window and would raise a KeyError '
                f'halfway through the cohort.')
        if float(p[key]) != float(truth):
            raise ValueError(
                f'{key}={p[key]} but extract_npz.py wrote the .npz files at '
                f'{truth:g} (EPOCH_DUR / PRE_END of extract_npz.py). session_measures uses '
                f'this to locate the analysis window inside each stored segment, '
                f'so a wrong value cuts the wrong seconds out of every segment. '
                f'Re-extract if the data really changed.')
    if float(p['EPOCH_DUR']) > float(p['EXTRACT_EPOCH_DUR']):
        raise ValueError(
            f"EPOCH_DUR={p['EPOCH_DUR']} exceeds the {EXTRACT_EPOCH_DUR:g} s "
            f'stored per segment by extract_npz.py. The extra data was '
            f'never written to the .npz. Re-extract, do not pad.')
    if float(p['PRE_END']) < float(p['EXTRACT_PRE_END']):
        raise ValueError(
            f"PRE_END={p['PRE_END']} is smaller than the {EXTRACT_PRE_END:g} s "
            f'used at extraction: the samples between them are not in the .npz, '
            f'so this window cannot be reconstructed. Re-run '
            f'extract_npz.py with a smaller PRE_END.')
    if int(p['MIN_CONF']) != EXTRACT_MIN_CONF:
        raise ValueError(
            f"MIN_CONF={p['MIN_CONF']} but the .npz files were written at "
            f'conf >= {EXTRACT_MIN_CONF} (MIN_CONF of extract_npz.py) and this '
            f'module cannot re-apply the gate: the rating is not stored per '
            f'segment. The folder tag would claim a cohort the run did not use.')


def _check_path_length(main, verbose=True,
                       longest_name='l2_heart_summary_session.csv'):
    """Warn before the compute if the run would write past Windows' MAX_PATH.

    Long tags and a deep results root can exceed Windows' 260-character MAX_PATH
    (result_path's docstring flags the budget). A run that computes for an hour
    and then cannot write its CSV has lost the hour, so this is checked first,
    not at the first open().
    """
    n = len(os.path.abspath(os.path.join(main, longest_name)))
    if n > 250 and verbose:
        _emit(f'\n  ⚠️ LONGEST OUTPUT PATH WOULD BE {n} CHARACTERS; Windows '
              f'MAX_PATH is 260.\n     Shorten the results root (HMD_RESULTS, see '
              f'config.py) or enable long paths BEFORE starting the compute.')
    return n


def run_paths(results_root, p, create=True, grain=None):
    """(run_root, main_dir, method_tag, param_tag) for this parameter set.

        <results_root>/HEART_DEPTH/<heart_run_tag>/<grain>_wise/main

    One folder under the arm. `heart_run_tag` carries only the swept axes
    (selection, number of ICs, duration, baseline, condition) plus an optional
    RUN_LABEL. Two runs differing only in another parameter map to the same
    folder, so run() calls `assert_no_param_collision` before it writes.
    README.txt at the run root and <grain>_wise/main/run_summary.txt carry the
    full parameter set.

    `method_tag` and `param_tag` are not path components; they are returned
    because they are the human-readable record of the selection parameters
    outside the README.

    `grain` defaults to `primary_grain(p)`.
    """
    _check_tag_params(p)
    # Still dispatched on IC_SELECT_METHOD although it no longer names a folder:
    # heart_method_tag refuses an unimplemented method, and that refusal is what
    # stops a typo'd IC_SELECT_METHOD from selecting ICs some other way.
    method = heart_method_tag(p)
    tag = heart_depth_tag(p)
    short = heart_run_tag(p)
    run_root = os.path.join(results_root, ARM, short)
    main = result_dir(results_root, ARM, short, 'main', method=None,
                      create=create, grain=(grain or primary_grain(p)))
    return run_root, main, method, tag


# ═══════════════════════════════════════════════════════════════════════════
# THE EPOCH WINDOW
# ═══════════════════════════════════════════════════════════════════════════
# window_indices is imported from heart_depth_pipeline, not mirrored: two
# implementations of one window would be the defect. It is re-exported so the
# accounting code here describes the window that was actually cut.
from heart_depth_pipeline import window_indices  # noqa: E402  (re-export)




# ═══════════════════════════════════════════════════════════════════════════
# PER-SESSION PREPARATION (gates + window + the in-memory npz)
# ═══════════════════════════════════════════════════════════════════════════
def _minimal_session_npz(npz_path, p):
    """Apply the pre-measure gates; return (what to hand session_measures, record).

    The first element is the original path when nothing was filtered out, an
    in-memory BytesIO npz when segments were removed, or None when the session
    is dropped. `np.load` accepts both of the first two.

    Segment selection only: this never cuts the analysis window (see the module
    docstring).

    The record is one row of `cohort_sessions.csv` and carries a named reason
    for every drop.
    """
    rec = dict(subject='', session=-1, npz=npz_path, dur_min_lowerbound=np.nan,
               n_seg_raw=0, n_seg_cond=0, n_seg_full_window=0, n_seg_short=0,
               shortest_window_s='', n_depths=0, n_seg_used=0, ic=-1,
               status='', reason='',
               # The CFA topography diagnostic (heart_depth_pipeline,
               # notes['cfa_topo_r2']), computed for every kept session. Empty
               # (not 0.0) on a dropped session: not measured is not r2 = 0.
               cfa_topo_r2='', cfa_topo_nbeats='')
    d = np.load(npz_path, allow_pickle=True)
    try:
        fs = float(d['fs'])
        rec['subject'] = str(d['subject_id'])
        rec['session'] = int(d['session'])
        seg_len = np.asarray(d['seg_len']).astype(np.int64)
        seg_span = np.atleast_2d(np.asarray(d['seg_span'])).astype(np.int64)
        block_spans = np.atleast_2d(np.asarray(d['block_spans'])).astype(np.int64)
        cond = np.asarray(d['seg_cond']).astype(int)
        depth = np.asarray(d['seg_depth']).astype(int)
        rec['n_seg_raw'] = int(seg_len.size)

        # ── GATE: MIN_SESSION_MIN, on the LOWER-BOUND proxy ─────────────────
        dur = session_duration_lowerbound_min(block_spans, seg_span, fs)
        rec['dur_min_lowerbound'] = round(float(dur), 1)
        if p.get('MIN_SESSION_MIN') and np.isfinite(dur) \
                and dur < float(p['MIN_SESSION_MIN']):
            rec['status'] = 'dropped'
            rec['reason'] = (f'MIN_SESSION_MIN: duration lower bound {dur:.1f} '
                             f'min < {float(p["MIN_SESSION_MIN"]):g} min '
                             f'(lower bound, not the .set header — see header)')
            return None, rec

        # ── GATE: CONDITION + DEPTH LEVELS ──────────────────────────────────
        # cond 0 = nosapo, 1 = sapo, 2 = pre-meditation, 3 = post-meditation
        # (as written by extract_npz.py). Only the meditation reports are
        # analysed; see the module docstring's condition-filter note.
        keep = np.isin(cond, (0, 1) if p['INCLUDE_SAPO'] else (0,))
        keep &= np.isin(depth, np.asarray(p['DEPTH_LEVELS']))
        rec['n_seg_cond'] = int(keep.sum())
        if rec['n_seg_cond'] == 0:
            rec['status'] = 'dropped'
            rec['reason'] = ('no segment in the analysed condition set '
                             + ('{nosapo, sapo}' if p['INCLUDE_SAPO'] else '{nosapo}')
                             + f' with depth in {list(p["DEPTH_LEVELS"])}')
            return None, rec

        # ── NOT A GATE: HOW LONG THE ANALYSED WINDOW WILL ACTUALLY BE ───────
        # Reporting only: the cut belongs to session_measures and is not made
        # here. window_indices returns scalar bounds plus a per-segment mask:
        # the window is anchored to the segment start, so [lo, hi) is the same
        # for every segment and what varies is only whether a given segment is
        # long enough to contain it. A segment that cannot hold the window is
        # refused by the library, not clamped; it is counted here so the
        # accounting names it.
        lo_ix, hi_ix, holds_full = window_indices(
            seg_len, fs, p['EPOCH_DUR'], p['PRE_END'],
            extract_epoch_dur=p['EXTRACT_EPOCH_DUR'],
            extract_pre_end=p['EXTRACT_PRE_END'])
        rec['n_seg_full_window'] = int((keep & holds_full).sum())
        rec['n_seg_short'] = int((keep & ~holds_full).sum())
        if rec['n_seg_short']:
            # how much of the window the SHORTEST refused segment could supply
            avail = np.asarray(seg_len)[keep & ~holds_full].min() - lo_ix
            rec['shortest_window_s'] = round(float(max(avail, 0) / fs), 1)

        # ── GATE: MIN_DEPTHS, PER SESSION ───────────────────────────────────
        # Per session and counting noSAPO + SAPO together. Applied here rather
        # than left to session_measures so the reason is nameable and so
        # pre-/post-meditation depths cannot count towards it.
        n_dep = len(set(depth[keep].tolist()))
        rec['n_depths'] = n_dep
        if n_dep < int(p['MIN_DEPTHS']):
            rec['status'] = 'dropped'
            rec['reason'] = (f'MIN_DEPTHS: {n_dep} distinct depth(s) < '
                             f'{int(p["MIN_DEPTHS"])}')
            return None, rec

        # ── GATE: MIN_SEG_SESSION (per-session floor; 0 = off) ──────────────
        n_keep = int(keep.sum())
        if int(p.get('MIN_SEG_SESSION', 0)) and n_keep < int(p['MIN_SEG_SESSION']):
            rec['status'] = 'dropped'
            rec['reason'] = (f'MIN_SEG_SESSION: {n_keep} usable segment(s) < '
                             f'{int(p["MIN_SEG_SESSION"])}')
            return None, rec

        # ── NOTHING FILTERED: HAND OVER THE ORIGINAL FILE ───────────────────
        # No copy, no buffer, and (more importantly) no opportunity for this
        # module to differ from the file on disk by a rounding error.
        if bool(keep.all()):
            rec['status'] = 'prepared'
            return npz_path, rec

        # ── BUILD THE FILTERED npz IN MEMORY ────────────────────────────────
        # The samples are copied verbatim, at full stored length. This selects
        # segments; it does not cut the analysis window. session_measures makes
        # that cut from whatever it is handed, and cutting here as well would
        # shift every window by (PRE_END - EXTRACT_PRE_END) seconds.
        idx = np.where(keep)[0]
        # Hoist the NpzFile lookups: `d['seg_eeg'][i]` inside a loop re-reads and
        # re-decompresses the whole ~190 MB array on every iteration
        # (NpzFile.__getitem__ is a file read, not an array view).
        seg_eeg_all = d['seg_eeg']
        seg_ecg_all = d['seg_ecg']
        eeg = seg_eeg_all[idx]
        ecg = seg_ecg_all[idx]
        del seg_eeg_all, seg_ecg_all
        # The covariate inputs must survive the copy too. `seg_resp` and
        # `seg_subcond` feed session_measures' KEEP_SEGCOV block (respAmp,
        # respPhase, reportmethod). A buffer that dropped them would leave those
        # covariates absent for exactly the sessions that had a segment filtered
        # out, because the `keep.all()` branch above hands over the original
        # path with every key intact. Guarded on presence so an older export
        # that lacks a key still loads.
        _extra = {}
        if 'seg_resp' in d.files:
            _extra['seg_resp'] = np.asarray(d['seg_resp'])[idx]
        if 'seg_subcond' in d.files:
            _extra['seg_subcond'] = np.asarray(d['seg_subcond'])[idx]

        # The extraction-provenance keys must survive the copy too.
        # session_measures decides whether to resolve overlaps from
        # `overlap_resolved`, and treats a file that lacks the key as already
        # resolved. A buffer that dropped the key would skip overlap resolution,
        # and only for the sessions that had something filtered.
        _prov = {k: np.asarray(d[k]) for k in
                 ('extract_epoch_dur', 'extract_pre_end', 'overlap_resolved')
                 if k in d.files}

        buf = io.BytesIO()
        # Uncompressed: this buffer is read back immediately and compression
        # would cost seconds per session for nothing.
        np.savez(buf,
                 subject_id=np.array(rec['subject']),
                 session=np.array(rec['session']),
                 fs=np.array(fs),
                 seg_eeg=eeg, seg_ecg=ecg,
                 seg_len=seg_len[idx],
                 seg_depth=depth[idx], seg_cond=cond[idx],
                 seg_span=seg_span[idx],
                 block_spans=block_spans,
                 icaweights=np.asarray(d['icaweights']),
                 icasphere=np.asarray(d['icasphere']),
                 icachansind=np.asarray(d['icachansind']),
                 # icawinv carries the IC sign convention; session_measures
                 # refuses a buffer without it.
                 icawinv=np.asarray(d['icawinv']),
                 # ch_names / ch_positions are copied as well (small arrays:
                 # 64 strings / 64x3 floats).
                 ch_names=np.asarray(d['ch_names'], dtype=object) if 'ch_names' in d.files else np.array([], dtype=object),
                 ch_positions=np.asarray(d['ch_positions']) if 'ch_positions' in d.files else np.array([]),
                 **_extra, **_prov)
        del eeg, ecg, _extra
        buf.seek(0)
        rec['status'] = 'prepared'
        return buf, rec
    finally:
        d.close()


# ═══════════════════════════════════════════════════════════════════════════
# COHORT
# ═══════════════════════════════════════════════════════════════════════════
def build_cohort(data_dir, p, limit=None, verbose=True):
    """Gates + per-session measures for the whole cohort.

    Returns (sessions_by_subject, session_records, subject_records).

    Every subject found on disk gets a row in `subject_records`, including the
    ones that end up with nothing. A dropped subject that simply vanishes from
    the output is how a cohort silently shrinks between two runs.
    """
    all_sessions = list_sessions(data_dir)
    if not all_sessions:
        raise FileNotFoundError(
            f'no segment files under {data_dir!r} (expected '
            f'<data_dir>/<subject>/ses<N>.npz, written by extract_npz.py)')
    subjects_all = sorted({sid for sid, _ses, _f in all_sessions})
    if limit is not None:
        subjects_all = subjects_all[:int(limit)]
        all_sessions = [s for s in all_sessions if s[0] in set(subjects_all)]
        if verbose:
            _emit(f'  --limit {limit}: {len(subjects_all)} subject(s), '
                  f'{len(all_sessions)} session(s) — SMOKE TEST, not a cohort.')

    sessions_by_subject, session_records = {}, []
    for k, (sid, ses, path) in enumerate(all_sessions, 1):
        if verbose:
            _emit(f'\n[{k}/{len(all_sessions)}] {sid} ses{ses}')
        buf, rec = _minimal_session_npz(path, p)
        if buf is None:
            if verbose:
                _emit(f'  DROPPED — {rec["reason"]}')
            session_records.append(rec)
            continue
        t0 = time.time()
        try:
            sm_notes = {}
            s = session_measures(buf, p, verbose=verbose,
                                 notes=sm_notes)
        except Exception as exc:                       # noqa: BLE001
            # An exception is a named drop, never a null.
            rec['status'] = 'dropped'
            rec['reason'] = f'session_measures raised {type(exc).__name__}: {exc}'
            session_records.append(rec)
            _emit(f'  DROPPED — {rec["reason"]}')
            continue
        finally:
            # `buf` is the original path when nothing was filtered out; only an
            # actual buffer has anything to close.
            if hasattr(buf, 'close'):
                buf.close()
        if s is None:
            # Do not guess the gate: `session_measures` records the gate it
            # refused on in notes['skip_reason']. An empty reason is a defect in
            # the library, not a cohort decision, and says so.
            rec['status'] = 'dropped'
            rec['reason'] = sm_notes.get('skip_reason') or (
                'session_measures returned no result and named no gate — this '
                'is a defect in heart_depth_pipeline.session_measures, not a '
                'cohort decision')
            session_records.append(rec)
            _emit(f'  DROPPED — {rec["reason"]}')
            continue
        rec['status'] = 'kept'
        rec['n_seg_used'] = int(len(s['depth']))
        rec['ic'] = int(s['ic'])
        # The whole selected set, not just the top IC. `ic` is sel_ics[0]; when
        # several ICs are selected the analysed signal is their polarity-aligned
        # average, and source localization must describe that signal, not the
        # top component. Order is meaningful: element 0 is the alignment
        # reference.
        rec['ic_all'] = [int(v) for v in (s.get('ic_all') or [s['ic']])]
        rec['reason'] = ''
        # Not written to cohort_sessions.csv (it is a list, and that file has one
        # row of scalars per session). It rides on the in-memory record so a
        # cohort-level pass can read every session's scores from one
        # build_cohort call.
        if 'lee_scores' in sm_notes:
            rec['_lee_scores'] = sm_notes['lee_scores']
        # The analysed signal's sign and scale. The IC is negated when its
        # peak-|icawinv| weight is negative and divided by its SD over the
        # analysed segments (IC_UNIT_VAR_NORM), so projecting a HEP effect back
        # to the scalp needs both: scalp = icawinv[:, ic] * (-1 if flipped) * sd
        # * x. Without them a back-projected map carries a random sign per
        # session (46.6% of selected ICs are flipped). Nothing in the run reads
        # them.
        for _k in ('ic_sign_flipped', 'ic_unit_var_norm_sd'):
            if _k in sm_notes:
                rec[_k] = sm_notes[_k]
        # cfa_topo_r2 (below) stays empty on NaN: session_measures writes
        # float('nan') when the diagnostic could not be computed, and 'nan' in a
        # CSV column of r2 values is one str() away from being read as a number.
        # The ECG gate inputs are carried so that the subject gate can be
        # applied. NaN is kept as NaN there (not ''), because the gate is written
        # so that an unassessable observation fails; blanking it would pass it.
        for _k in ('ecg_clean_ratio', 'ibi_short_prop'):
            _v = sm_notes.get(_k)
            rec[_k] = (float('nan') if _v is None else float(_v))
        _r2 = sm_notes.get('cfa_topo_r2')
        rec['cfa_topo_r2'] = ('' if _r2 is None or not np.isfinite(_r2)
                              else round(float(_r2), 6))
        rec['cfa_topo_nbeats'] = int(sm_notes.get('cfa_topo_nbeats', 0)) or ''
        if sm_notes.get('cfa_topo_r2_error'):
            rec['cfa_topo_nbeats'] = ''
        session_records.append(rec)
        sessions_by_subject.setdefault(sid, []).append(s)
        if verbose:
            _emit(f'  KEPT — {rec["n_seg_used"]} segment(s), IC {rec["ic"]}, '
                  f'{time.time() - t0:.1f} s')

    # ── SUBJECT-LEVEL GATE: minSeg, pooled over sessions ────────────────────
    # `subject_slopes` enforces the same floor internally, so applying it here
    # changes no number; it makes the drop reportable instead of showing up as
    # a subject that quietly has no map. The ECG half of the subject QA gate is
    # applied here too, in keep form, so an unassessable (NaN) subject fails
    # rather than passing on no evidence.
    #
    # Order is deliberate: minSeg and the depth-diversity rule run first, so a
    # subject already out for having no usable data is not also reported as an
    # ECG failure. A subject dropped here is named with its own numbers.
    #
    # Averaged over the kept sessions only (the observations that reached the
    # gate, not every recording the subject gave). A record is 'kept' exactly
    # when its measures went into sessions_by_subject, so filtering on status
    # is the same set.
    subject_records = []
    for sid in subjects_all:
        found = sum(1 for r in session_records if r['subject'] == sid)
        kept = [s for s in sessions_by_subject.get(sid, [])]
        n_seg = int(sum(len(s['depth']) for s in kept))
        n_dep = len({int(v) for s in kept for v in s['depth']})
        _qa = [r for r in session_records
               if r['subject'] == sid and r['status'] == 'kept']
        # The ECG half only. MIN_IBI_MS already removes short intervals beat by
        # beat and MIN_BEATS_UNIT then decides whether enough remain, so gating
        # the subject on a statistic of the discarded beats would test the same
        # defect twice. `ibi_short_prop` is still reported below, just not gated.
        _ecg_mean, _n_unassessable = subject_ecg_clean_ratio(
            [r.get('ecg_clean_ratio', float('nan')) for r in _qa])
        # Keep form, so an unassessable (NaN) subject fails rather than passing
        # on no evidence; ecg_ratio_passes_gate is written for exactly that.
        _ecg_ok = ecg_ratio_passes_gate(_ecg_mean, float(p['ECG_MAX']))
        _ibi_mean, _ = subject_ecg_clean_ratio(
            [r.get('ibi_short_prop', float('nan')) for r in _qa])
        srec = dict(subject=sid, n_sessions_found=found,
                    n_sessions_kept=len(kept), n_segments=n_seg,
                    n_depths_pooled=n_dep,
                    ecg_clean_ratio=('' if not np.isfinite(_ecg_mean)
                                     else round(float(_ecg_mean), 4)),
                    ibi_short_prop=('' if not np.isfinite(_ibi_mean)
                                    else round(float(_ibi_mean), 4)),
                    status='kept', reason='')
        if not kept:
            srec['status'] = 'dropped'
            srec['reason'] = 'no session survived the session-level gates'
        elif n_seg < int(p['minSeg']):
            srec['status'] = 'dropped'
            srec['reason'] = f'minSeg: {n_seg} pooled segment(s) < {int(p["minSeg"])}'
        elif n_dep < 2:
            srec['status'] = 'dropped'
            srec['reason'] = (f'{n_dep} distinct depth level(s) pooled — a slope '
                              f'needs at least 2')
        elif not _ecg_ok:
            srec['status'] = 'dropped'
            srec['reason'] = 'subject ECG QA: ' + (
                'ECG unassessable (no detected beat inside any analysed segment)'
                if not np.isfinite(_ecg_mean)
                else f'ECG {_ecg_mean:.2f}>{float(p["ECG_MAX"]):.2f}')
        if srec['status'] == 'dropped':
            sessions_by_subject.pop(sid, None)
        subject_records.append(srec)

    return sessions_by_subject, session_records, subject_records


def _report_duration_proxy(p, session_records, verbose=True):
    """Say out loud what MIN_SESSION_MIN was actually applied to, and how close it got."""
    gate = float(p.get('MIN_SESSION_MIN') or 0)
    if not verbose:
        return
    _emit('\n  MIN_SESSION_MIN = ' + (f'{gate:g} min' if gate else '0 (gate off)'))
    if not gate:
        return
    _emit('    ⚠️ THE .set DURATION IS NOT IN THE .npz. The gate is defined on '
          'hdr.pnts/hdr.srate/60')
    _emit('       (the .set header); '
          'extract_npz.py never wrote it.')
    _emit('       USED INSTEAD: max(block_spans[:,1], seg_span[:,1]) / fs / 60 — a '
          'LOWER BOUND on')
    _emit('       the recording length. Above the gate is certain; below it is a '
          'drop the proxy')
    _emit('       cannot fully justify, and each such session is named below.')
    dur = np.array([r['dur_min_lowerbound'] for r in session_records], float)
    dropped = [r for r in session_records
               if str(r['reason']).startswith('MIN_SESSION_MIN')]
    kept = dur[[i for i, r in enumerate(session_records)
                if not str(r['reason']).startswith('MIN_SESSION_MIN')]]
    kept = kept[np.isfinite(kept)]
    if kept.size:
        _emit(f'    admitted: {kept.size} session(s), lower bound '
              f'{kept.min():.1f}-{kept.max():.1f} min '
              f'(margin over the gate: {kept.min() - gate:+.1f} min)')
    for r in dropped:
        _emit(f'    ⚠️ DROPPED on the proxy: {r["subject"]} ses{r["session"]} '
              f'lower bound {r["dur_min_lowerbound"]:.1f} min — the true '
              f'recording is AT LEAST this long, so this drop is safe only '
              f'because the bound is far below the gate.')
    borderline = [r for r in session_records
                  if np.isfinite(r['dur_min_lowerbound'])
                  and not str(r['reason']).startswith('MIN_SESSION_MIN')
                  and r['dur_min_lowerbound'] < gate + BORDERLINE_MARGIN_MIN]
    for r in borderline:
        _emit(f'    ⚠️ BORDERLINE: {r["subject"]} ses{r["session"]} lower bound '
              f'{r["dur_min_lowerbound"]:.1f} min is within '
              f'{BORDERLINE_MARGIN_MIN:g} min of the gate — the proxy cannot '
              f'decide this one; check the .set header before quoting it.')


def print_accounting(session_records, subject_records, p, verbose=True):
    """The per-subject accounting table. A dropped subject is NEVER silently absent.

    Returns (n_subjects_kept, n_sessions_kept) whether or not it printed, so the
    caller's counts do not depend on the verbosity.
    """
    if not verbose:
        n_keep = sum(1 for r in subject_records if r['status'] == 'kept')
        return n_keep, sum(1 for r in session_records if r['status'] == 'kept')
    _emit('\n' + '=' * 92)
    _emit('  COHORT ACCOUNTING — sessions')
    _emit('=' * 92)
    # `full` = segments long enough to hold the whole EPOCH_DUR window;
    # `short` = segments the library refused because the stored segment does
    # not reach the window's end (anchoring, not length; see GATE 1b). They are
    # dropped, so `short` > 0 means data was excluded, not that the folder tag
    # over-states the window.
    _emit(f'  {"subject":9s} {"ses":>3s} {"dur_lb":>7s} {"raw":>4s} {"cond":>5s} '
          f'{"full":>5s} {"short":>6s} {"dep":>4s} {"used":>5s} {"IC":>4s}  '
          f'status / why')
    for r in sorted(session_records, key=lambda r: (r['subject'], r['session'])):
        _emit(f'  {r["subject"]:9s} {r["session"]:3d} '
              f'{r["dur_min_lowerbound"]:7.1f} {r["n_seg_raw"]:4d} '
              f'{r["n_seg_cond"]:5d} {r["n_seg_full_window"]:5d} '
              f'{r["n_seg_short"]:6d} {r["n_depths"]:4d} '
              f'{r["n_seg_used"]:5d} {r["ic"]:4d}  {r["status"]}'
              + (f' — {r["reason"]}' if r['reason'] else ''))
    short = [r for r in session_records if r['n_seg_short']]
    if short:
        _emit(f'\n  ⚠️ {sum(r["n_seg_short"] for r in short)} segment(s) in '
              f'{len(short)} session(s) do not reach the requested '
              f'{float(p["EPOCH_DUR"]):g} s window.')
        _emit('     heart_depth_pipeline GATE 1b DROPS these with a reason '
              'rather than clamping them, so')
        _emit('     they are EXCLUDED from the analysis - NOT analysed at a '
              'shorter length. The window is')
        _emit('     anchored to the segment START, so a segment can be long '
              'enough and still not hold')
        _emit('     the seconds it asks for. Named here so the drop can be '
              'checked:')
        for r in short:
            _emit(f'       {r["subject"]} ses{r["session"]}: {r["n_seg_short"]} '
                  f'segment(s), shortest window {r["shortest_window_s"]} s')
    _emit('\n' + '=' * 92)
    _emit('  COHORT ACCOUNTING — subjects')
    _emit('=' * 92)
    _emit(f'  {"subject":9s} {"found":>5s} {"kept":>4s} {"segs":>5s} {"deps":>4s}  '
          f'status / why')
    for r in subject_records:
        _emit(f'  {r["subject"]:9s} {r["n_sessions_found"]:5d} '
              f'{r["n_sessions_kept"]:4d} {r["n_segments"]:5d} '
              f'{r["n_depths_pooled"]:4d}  {r["status"]}'
              + (f' — {r["reason"]}' if r['reason'] else ''))
    n_keep = sum(1 for r in subject_records if r['status'] == 'kept')
    n_drop = len(subject_records) - n_keep
    n_ses = sum(1 for r in session_records if r['status'] == 'kept')
    _emit(f'\n  {n_keep} subject(s) enter L1, {n_drop} dropped; '
          f'{n_ses}/{len(session_records)} session(s) kept.')
    if n_keep < MIN_SUB_L2:
        _emit(f'  ⚠️ {n_keep} < MIN_SUB_L2 = {MIN_SUB_L2}: the L2 test will be '
              f'SKIPPED for every measure (l2_heart refuses a sign-flip null '
              f'this small). Rows will read Status="skipped:too_few_subjects" — '
              f'a SKIP, never a null.')
    return n_keep, n_ses


def _write_csv(path, rows, fields):
    """Atomic CSV write: a crash mid-write cannot leave a truncated table behind."""
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(fields))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, '') for k in fields})
    os.replace(tmp, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
def _combine_masks(outer, inner):
    """Compose a TLIMS mask with a within-TLIMS window mask.

    `posthoc_heart.unit_cluster_values` indexes the full HEP_WIN axis, but a
    windowed L2 ('lit') produced its cluster on the TLIMS axis restricted again.
    Applying the inner mask to the full axis directly would select the wrong
    samples: the two masks live on different rulers. This lifts `inner` back onto the full axis
    by placing it at the positions `outer` selected.
    """
    outer = np.asarray(outer, bool)
    inner = np.asarray(inner, bool)
    if int(outer.sum()) != inner.size:
        raise ValueError(
            f'_combine_masks: inner mask has {inner.size} element(s) but the '
            f'outer mask selects {int(outer.sum())}. These must match — the '
            f'inner mask is defined ON the outer selection.')
    out = np.zeros(outer.shape, bool)
    out[np.where(outer)[0][inner]] = True
    return out


def _restrict_time(subjects, measures, t_mask):
    """Copy the L2 subject list with every map trimmed along the TIME axis.

    Trims the map; it does not refit. Under OLS the L1 slope at a given
    (freq, time) cell does not depend on which other cells are in the map, so
    restricting the window afterwards gives exactly the slopes a windowed fit
    would give. Under L1_METHOD='WLS' the Huber weights are computed from each
    unit's residual norm across all fitted columns (l1_slopes.cell_ols), so a
    windowed refit could differ slightly. What changes either way is the
    permutation null and the cluster-forming threshold, which is the point of
    testing a narrower window.

    HEP maps are [nT] and HRSP/HEPC are [nF x nT], so the mask is applied to the
    LAST axis in both cases. `svar` is trimmed identically or the weighting would
    be computed from cells that are no longer in the test.
    """
    out = []
    for rec in subjects:
        r = {'subject': rec['subject']}
        for meas in measures:
            v = rec.get(meas)
            if v is None:
                r[meas] = None
                continue
            beta, svar, nu = v
            r[meas] = (np.asarray(beta)[..., t_mask],
                       np.asarray(svar)[..., t_mask], nu)
        out.append(r)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# L1
# ═══════════════════════════════════════════════════════════════════════════
def _l1_summary_columns():
    """Column order of l1_fit_summary.csv.

    N_bl_cols / N_bl_dead make the Alday baseline auditable per subject: a
    folder named `bl-...` is not evidence that the baseline applied. N_bl_cols
    is 1 for HEP, nFreq for HRSP, and 0 for HEPC or a `nobl` run; N_bl_dead is
    how many of those carried no information and were dropped without spending
    a degree of freedom.
    """
    return ('Model', 'Subject', 'Measure', 'Grain', 'N_sessions',
            'N_units', 'N_segments', 'N_depths', 'Fit', 'N_seg_weight',
            'N_bl_cols', 'N_bl_dead', 'N_cov_dropped',
            'Frac_finite', 'Reason')


def fit_l1(sessions_by_subject, p, model, verbose=True, grain=GRAIN):
    """Per-subject depth slope maps for one L1 design, at one GRAIN.

    Returns (subjects, l1_rows) where `subjects` is the list of dicts
    `run_l2_heart` expects, one per subject IN COHORT ORDER (a subject whose map
    was refused is present with None, so the L2 sees the same subject list for
    every measure).
    """
    # Validated against ALL_MODELS, imported from the pipeline, so a model added
    # there is known here. Keep it derived, never restated.
    if model not in ALL_MODELS:
        raise ValueError(f'model must be one of {ALL_MODELS}, got {model!r}')
    subjects, l1_rows = [], []
    for sid in sorted(sessions_by_subject):
        sess = sessions_by_subject[sid]
        n_units = int(sum(len(s['depth']) for s in sess))
        n_dep = len({int(v) for s in sess for v in s['depth']})
        n_ses = len(sess)
        rec = {'subject': sid}
        # Three different counts, which must not share a column:
        #   N_units       units at this grain (cells, segments or beats): the n
        #                 the L1 design actually has.
        #   N_segments    the subject's depth-report segment count, what
        #                 `minSeg` gates on at every grain.
        #   N_seg_weight  what subject_slopes returned: the segments backing the
        #                 units this measure kept. The L2's precision_nseg
        #                 scheme weights by it at every grain, so it must not
        #                 become a cell or beat count.
        # Reporting one under another's name is how a grain comes to look like
        # it analysed something it did not.
        if grain == 'cell':
            n_units_grain = int(sum(len({int(v) for v in s['depth']}) for s in sess))
        elif grain == 'perbeat':
            n_units_grain = int(sum(
                (0 if s.get('perbeat') is None
                 else int(np.asarray(s['perbeat']['seg']).size)) for s in sess))
        else:
            n_units_grain = n_units
        for meas in grain_measures(p, grain):
            # `diag` is an out-parameter, filled in place; it stays empty
            # when subject_slopes refuses before it builds the design.
            diag = {}
            beta, svar, nu = subject_slopes(sess, meas, p, model=model,
                                            grain=grain, diag=diag)
            row = dict(Model=model, Subject=sid, Measure=meas, Grain=grain,
                       N_sessions=n_ses, N_units=n_units_grain,
                       N_segments=n_units, N_depths=n_dep,
                       Fit=beta is not None, N_seg_weight=int(nu),
                       N_bl_cols=diag.get('n_bl_cols', ''),
                       N_bl_dead=diag.get('n_bl_dead', ''),
                       N_cov_dropped=diag.get('n_cov_dropped', ''),
                       Frac_finite='', Reason='')
            if beta is None:
                rec[meas] = None
                # The likely refusal reason, reconstructed from the counts in
                # the order subject_slopes tests them (no units, then minSeg,
                # then fewer than 2 depths). subject_slopes can also refuse for
                # reasons not listed here, e.g. a degenerate moderation column or
                # fewer than 3 usable HEPC units.
                row['Reason'] = (
                    f'subject_slopes refused: '
                    + ('no session carried this measure'
                       if n_units == 0 else
                       f'{n_units} segment(s) < minSeg {int(p["minSeg"])}'
                       if n_units < int(p['minSeg']) else
                       f'{n_dep} distinct depth(s) < 2'))
            else:
                rec[meas] = (beta, svar, int(nu))
                frac = float(np.isfinite(np.asarray(beta, float)).mean())
                row['Frac_finite'] = round(frac, 4)
                if frac < 1.0:
                    # Not a null. A map with any non-finite cell is deleted
                    # listwise at L2 (l2_heart._listwise_keep), so a subject
                    # that "was fit" can still contribute nothing. Said so here.
                    row['Reason'] = (
                        f'{100 * (1 - frac):.1f}% of cells non-finite — '
                        f'cell_ols refused those columns; l2_heart deletes '
                        f'this subject LISTWISE from this measure')
                # A dead baseline column is not a failure and not a null: the
                # map is fit, just without the Alday adjustment at those
                # frequencies, because the baseline window carried nothing to
                # adjust with (all non-finite, or identical across every unit).
                # Reported because the run folder is named `bl-...` and a reader
                # is entitled to know where the baseline did not apply. It costs
                # no degree of freedom: heart_depth_pipeline drops it rather
                # than fitting a zero column.
                nd = diag.get('n_bl_dead', 0) or 0
                if nd:
                    nc = diag.get('n_bl_cols', 0) or 0
                    msg = (f'{nd}/{nc} Alday baseline column(s) DEAD and '
                           f'dropped — no baseline adjustment at '
                           + ('this frequency' if nd == 1 and nc == 1
                              else f'{nd} frequenc' + ('y' if nd == 1 else 'ies'))
                           + ' (df not spent)')
                    row['Reason'] = (row['Reason'] + '; ' + msg
                                     if row['Reason'] else msg)
            l1_rows.append(row)
            if verbose:
                _emit(f'    {sid} {meas:5s} {model:8s} {grain:7s}: '
                      + ('REFUSED — ' + row['Reason'] if beta is None
                         else f'{nu} unit(s), {row["Frac_finite"]:.3f} finite'
                              + (' ⚠️ ' + row['Reason'] if row['Reason'] else '')))
        subjects.append(rec)
    return subjects, l1_rows


# ═══════════════════════════════════════════════════════════════════════════
# THE EXACT CLUSTER CELLS -> run_results.npz
# ═══════════════════════════════════════════════════════════════════════════
# A finished run persists the exact cluster cells, not only the bounding box
# that the `l2_rows_json` rows carry (Start_ms/End_ms/Start_Hz/End_Hz). A
# cluster is a connected set, so its box contains cells the test never
# selected (on one segment-grain unadj HRSP cluster, 1584 box cells against
# 1090 cluster cells), and a reader that averages over the box describes a
# different region.
#
# Absence is distinguishable: a run without these keys makes `load_l2_cells`
# return None rather than an empty dict, so "this run did not persist cells"
# and "this run had no clusters" stay two different answers.
#
# One concatenated array, not JSON. A single HRSP map is 27 x 176 = 4752 cells
# and a run has up to 9 models x 2 windows x 3 measures x 3 schemes; as JSON
# that would be tens of MB per run. int32 pairs in the npz compress to a
# fraction of it.
def _pack_l2_cells(cells_by_key, taxis_by_key):
    """Exact cluster cells for every (model@window, measure, scheme) -> npz keys.

    Returns three keys:
      l2_cells_xy          int32 [N x 2]: every cluster's cells, concatenated.
                           Column 0 is the FREQUENCY index, column 1 the TIME
                           index ON THAT WINDOW'S OWN AXIS.
      l2_cells_index_json  {l2key: {"<measure>|<scheme>": [ {rank, p, mass,
                           n_cells, start, stop}, ... ]}}; start/stop slice
                           l2_cells_xy.
      l2_cells_taxis_json  {l2key: [indices into the run's full `times`]}: how
                           to map a window time index back. Without it a 'lit'
                           cluster would read as sitting at the start of the
                           full axis, which is plausible and wrong.
    """
    blocks, index, taxis = [], {}, {}
    n = 0
    for key in sorted(cells_by_key):
        per_key = {}
        for (meas, sch), cl in sorted(cells_by_key[key].items(),
                                      key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            recs = []
            for rank, c in enumerate(cl, 1):
                xy = np.asarray(c.get('cells'), np.int32)
                if xy.ndim != 2 or xy.shape[1] != 2 or xy.size == 0:
                    continue
                blocks.append(xy)
                recs.append(dict(rank=rank, p=float(c['p']), mass=float(c['mass']),
                                 n_cells=int(xy.shape[0]), start=n,
                                 stop=n + int(xy.shape[0])))
                n += int(xy.shape[0])
            if recs:
                per_key[f'{meas}|{sch}'] = recs
        index[key] = per_key
        taxis[key] = [int(v) for v in np.asarray(taxis_by_key.get(key, []), int)]
    xy_all = (np.concatenate(blocks, axis=0).astype(np.int32, copy=False)
              if blocks else np.zeros((0, 2), np.int32))
    return {
        'l2_cells_xy': xy_all,
        'l2_cells_index_json': np.array(json.dumps(index, allow_nan=False)),
        'l2_cells_taxis_json': np.array(json.dumps(taxis, allow_nan=False)),
    }


def load_l2_cells(z):
    """run_results.npz (or a path) -> {l2key: {"meas|scheme": [cluster dicts]}}.

    Each cluster dict carries `cells` [n x 2] (freq index, time index on the
    WINDOW's axis), `t_index` (the same time indices mapped onto the run's FULL
    `times` array) and the cluster's own p / mass / n_cells / rank.

    Returns None, not {}, for a run that did not persist cells, so a caller can
    say "this run cannot supply exact masks, fall back to the bounding box and
    stamp Mask=bbox" without mistaking it for "this run had no clusters".
    """
    own = False
    if isinstance(z, str):
        z = np.load(z, allow_pickle=False)
        own = True
    try:
        if 'l2_cells_index_json' not in getattr(z, 'files', []):
            return None
        index = json.loads(str(z['l2_cells_index_json']))
        taxis = json.loads(str(z['l2_cells_taxis_json']))
        xy = np.asarray(z['l2_cells_xy'], np.int32)
    finally:
        if own:
            z.close()
    out = {}
    for key, per_key in index.items():
        ti = np.asarray(taxis.get(key, []), int)
        conv = {}
        for ms, recs in per_key.items():
            lst = []
            for r in recs:
                cells = xy[int(r['start']):int(r['stop'])]
                d = dict(r)
                d['cells'] = cells
                # An empty taxis means the window was the full axis (nothing was
                # trimmed) or the key was never recorded. Map only when we have
                # it; a wrong mapping is worse than an absent one.
                d['t_index'] = (ti[cells[:, 1]] if ti.size else cells[:, 1])
                lst.append(d)
            conv[ms] = lst
        out[key] = conv
    return out


def _pack_l1(subjects, p, model, grain=GRAIN):
    """Per-subject slope maps as plain arrays for run_results.npz.

    A subject with no map for a measure gets an all-NaN slice and n_units 0, so
    the arrays are rectangular and the subject axis is the same for every
    measure. `<key>_hasmap` records which is which: an all-NaN slice must never
    be readable as "fit, and all cells refused".
    """
    out = {}
    sids = [s['subject'] for s in subjects]
    # The measures of this grain, not of the run. Packing an all-NaN slice for a
    # measure this grain never fits (HEPC at the per-beat grain) would write a
    # key that reads as "attempted, every cell refused", which is a result, and
    # the wrong one. An absent key is the honest record of "not defined here".
    for meas in grain_measures(p, grain):
        shape = None
        for s in subjects:
            e = s.get(meas)
            if e is not None and e[0] is not None:
                shape = np.asarray(e[0]).shape
                break
        if shape is None:
            out[f'L1_{model}_{meas}_beta'] = np.zeros((len(sids), 0))
            out[f'L1_{model}_{meas}_svar'] = np.zeros((len(sids), 0))
            out[f'L1_{model}_{meas}_nunits'] = np.zeros(len(sids), np.int64)
            out[f'L1_{model}_{meas}_hasmap'] = np.zeros(len(sids), bool)
            continue
        beta = np.full((len(sids),) + shape, np.nan)
        svar = np.full((len(sids),) + shape, np.nan)
        nun = np.zeros(len(sids), np.int64)
        has = np.zeros(len(sids), bool)
        for i, s in enumerate(subjects):
            e = s.get(meas)
            if e is None or e[0] is None:
                continue
            beta[i] = np.asarray(e[0], float)
            svar[i] = np.asarray(e[1], float)
            nun[i] = int(e[2])
            has[i] = True
        out[f'L1_{model}_{meas}_beta'] = beta
        out[f'L1_{model}_{meas}_svar'] = svar
        out[f'L1_{model}_{meas}_nunits'] = nun
        out[f'L1_{model}_{meas}_hasmap'] = has
    return out


def _save_npz_atomic(path, payload):
    """np.savez_compressed to `path` via a temp file + os.replace.

    run_results.npz is written twice: once with the L1 maps before any L2 runs,
    once again with the L2 rows appended. If the second write died halfway
    through onto the same filename, a finished L1 would be gone. os.replace is
    atomic on both POSIX and Windows.
    """
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return path


def _l2_key(model, wname):
    """The one definition of the key an (L1 model, L2 window) result is filed under.

    Used by both the L2 loop and the summary writer, so every 'lit' result (the
    200-600 ms literature window) is named in run_summary.txt and a 'lit' L2
    that threw is reported as failed, not as never attempted.

    'full' keeps the bare model name, so readers of l2_heart_summary_<model>.csv
    and of the npz keys find the primary result under its plain name.
    """
    return str(model) if wname == 'full' else f'{model}@{wname}'


def _split_l2_key(key):
    """`_l2_key` inverted: 'unadj@lit' -> ('unadj', 'lit'), 'unadj' -> ('unadj', 'full')."""
    model, _, wname = str(key).partition('@')
    return model, (wname or 'full')


def _posthoc_status_rows(p, posthoc_errors, n_posthoc_rows=0):
    """The rows that stop posthoc_summary.csv from lying by omission.

    An empty posthoc_summary.csv would read as a result, so each non-result
    state gets an explicit row: a disabled battery (POSTHOC=False), a battery
    that threw, and a battery that ran but was gated out because no effect
    survived the main model. Without the last one, "tested and came back null"
    and "never tested" would both be a header-only file. `n_posthoc_rows` is how
    many real rows the battery produced, so a zero here is a positive
    statement, not an absence.
    """
    rows = []
    if not p.get('POSTHOC', True):
        r = dict.fromkeys(POSTHOC_FIELDS, '')
        r['Note'] = ('POSTHOC=False — the battery was DISABLED for this run. '
                     'This table is empty because it DID NOT RUN, not because '
                     'nothing was detected.')
        rows.append(r)
    elif not n_posthoc_rows and not posthoc_errors:
        r = dict.fromkeys(POSTHOC_FIELDS, '')
        r['Note'] = ('POST-HOC BATTERY RAN AND WAS GATED OUT: no effect survived '
                     'the main model in this grain, so there was nothing to run '
                     'a post-hoc ON. This is a MEASURED null, not a skipped '
                     'check — the battery is deliberately never run on an effect '
                     'that did not survive.')
        rows.append(r)
    for key in sorted(posthoc_errors):
        model, wname = _split_l2_key(key)
        r = dict.fromkeys(POSTHOC_FIELDS, '')
        r.update(Model=model, Window=wname,
                 Note=f'POST-HOC BATTERY FAILED: {posthoc_errors[key]} — this '
                      f'is a FAILURE, not a null result.')
        rows.append(r)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# THE RUN
# ═══════════════════════════════════════════════════════════════════════════
def _write_grain_failure(main, grain, p, err, tb):
    """RUN_FAILED.txt in the grain's main/: what died, and what is NOT here.

    The filename is deliberately not run_summary.txt: readers treat
    run_summary.txt as the record of a grain that ran, and writing the failure
    under that name would make a dead grain look like a finished one with bad
    news inside. Its absence next to a missing run_results.npz means
    "interrupted", which is a different fact again.
    """
    os.makedirs(main, exist_ok=True)
    L = [f'GRAIN {grain} FAILED', '=' * 78, '',
         f'error          {err}', '',
         'WHAT IS NOT IN THIS FOLDER', '-' * 78,
         '  run_results.npz      - no L1 map was saved for this grain',
         '  l1_fit_summary.csv   - the L1 did not complete',
         '  l2_heart_summary_*   - nothing was tested',
         '  run_summary.txt      - written only by a grain that finished', '',
         '⚠️ THIS IS A FAILURE, NOT A NULL. Nothing was tested at this grain, so',
         '   an absent result here must never be read as "ran and found nothing".',
         '   The other grains of this run are independent and may be complete -',
         '   check for run_results.npz in each <grain>_wise/main/.', '',
         f'grains requested   {tuple(p.get("GRAINS") or (grain,))}',
         f'KEEP_PERBEAT       {bool(p.get("KEEP_PERBEAT"))}', '',
         'TRACEBACK', '-' * 78, tb]
    with io.open(os.path.join(main, 'RUN_FAILED.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def _run_one_grain(grain, sess_by_sub, p, data_dir, run_root, main,
                   session_records, subject_records, limit=None, verbose=True):
    """L1 -> save -> L2 -> post-hocs -> summary, for one grain, into one main/.

    The ordering matters: the L1 maps are on disk before any L2 runs, so a
    post-hoc failure cannot destroy a finished L1.

    The cohort is not something this function may change. Every grain of a run
    sees the same `sess_by_sub`, because the gates that build it are decided on
    segment counts at every grain. A grain that re-gated would report a
    different n for the same run with nothing saying why.

    Returns the per-grain result dict.
    """
    t0 = time.time()
    # ── L1, every model ─────────────────────────────────────────────────────
    # The L2 axis must be the trimmed one. session_measures returns the full
    # HEP_WIN axis (-300..700 ms, 251 points), but subject_slopes fits over
    # TLIMS only (-100..600 ms, 176 points). Handing L2 the full axis would put
    # every cluster's reported latency on the wrong ruler; l2_heart refuses the
    # mismatch ("time axis has 251 points but the slope map has 176"). Do not
    # "fix" that by widening the guard.
    times_full = np.asarray(next(iter(sess_by_sub.values()))[0]['times'], float)
    t_keep = tlims_mask(times_full, p['TLIMS'])
    times = times_full[t_keep]
    freqs = np.asarray(next(iter(sess_by_sub.values()))[0]['freqs'], float)
    subjects_by_model, l1_rows_all = {}, []
    for model in tuple(p['MODELS']):
        if verbose:
            _emit(f'\n{"=" * 92}\n  L1 — {grain} grain, model {model!r}'
                  + ('   (PRIMARY)' if model == p['PRIMARY_MODEL'] else '')
                  + f'\n{"=" * 92}')
        subs, rows = fit_l1(sess_by_sub, p, model, verbose=verbose,
                            grain=grain)
        subjects_by_model[model] = subs
        l1_rows_all.extend(rows)
    _write_csv(os.path.join(main, 'l1_fit_summary.csv'), l1_rows_all,
               _l1_summary_columns())

    # ── SAVE THE L1 MAPS *BEFORE* ANY L2 ────────────────────────────────────
    # This ordering is the requirement: a post-hoc failure must not be able to
    # destroy a finished L1.
    payload = dict(
        subject_ids=np.array([s['subject'] for s in
                              subjects_by_model[tuple(p['MODELS'])[0]]]),
        times=times, freqs=freqs, grain=np.array(grain),
        # allow_nan=False on every dump in this module. `_jsonable` already
        # turns a non-finite float into its string ('nan'/'inf'); if anything
        # slips past it, Python's default would emit the bare token NaN, which
        # is not JSON (Python's json.loads accepts it, other readers reject the
        # file). With allow_nan=False a miss raises here instead.
        params_json=np.array(json.dumps(_jsonable(_params_for_record(p)), sort_keys=True,
                                        allow_nan=False)),
        models=np.array(list(map(str, p['MODELS']))),
        measures=np.array(list(map(str, grain_measures(p, grain)))),
        schemes=np.array(list(map(str, grain_schemes(p, grain)))),
        session_records_json=np.array(json.dumps(_jsonable(session_records),
                                                 allow_nan=False)),
        subject_records_json=np.array(json.dumps(_jsonable(subject_records),
                                                 allow_nan=False)),
        l1_rows_json=np.array(json.dumps(_jsonable(l1_rows_all),
                                         allow_nan=False)),
        # Present from the FIRST write so a reader of a run that died between
        # the two saves finds the keys and an explicit "no L2 yet", never a
        # KeyError that has to be interpreted.
        l2_rows_json=np.array('{}'),
        l2_errors_json=np.array('{}'),
        l2_skipped_json=np.array('{}'),
        l2_complete=np.array(False),
    )
    for model in tuple(p['MODELS']):
        payload.update(_pack_l1(subjects_by_model[model], p, model, grain=grain))
    npz_path = os.path.join(main, 'run_results.npz')
    _save_npz_atomic(npz_path, payload)
    if verbose:
        _emit(f'\n  L1 SAVED (before any L2): {npz_path}')

    # ── L2, PER MODEL, EACH IN ITS OWN try ──────────────────────────────────
    l2_by_model, l2_errors, detected_by_model = {}, {}, {}
    l2_skipped = {}
    posthoc_rows, posthoc_errors = [], {}
    # The exact cluster cells, kept so they can be persisted (_pack_l2_cells).
    # `cl_out` is rebuilt inside the window loop; without this a reader of a
    # finished run would have only the bounding box (Start/End_ms x
    # Start/End_Hz), which holds cells the test never selected (31% of the
    # rectangle on one surviving segment-grain cluster).
    l2_cells_by_key, l2_taxis_by_key = {}, {}
    for model in tuple(p['MODELS']):
        if verbose:
            _emit(f'\n{"=" * 92}\n  L2 — {grain} grain, L1 model {model!r}'
                  + ('   (PRIMARY)' if model == p['PRIMARY_MODEL'] else '')
                  + f'\n{"=" * 92}')
        for wname in tuple(p.get('L2_WINDOWS', ('full',))):
            # The 'full' result keeps its plain filename and key, which readers
            # of this run tree look for (l2_heart_summary_<model>.csv); a second
            # window must not rename the primary result.
            key = _l2_key(model, wname)
            if wname == 'full':
                t_mask = np.ones(times.size, bool)
                subs_w = subjects_by_model[model]
                out_csv = os.path.join(main, f'l2_heart_summary_{model}.csv')
            elif wname == 'lit':
                _lit = lit_ms(p)
                t_mask = (times >= _lit[0]) & (times <= _lit[1])
                if int(t_mask.sum()) < 2:
                    l2_skipped[key] = (
                        f'window {wname!r} SKIPPED: LIT_MS {_lit} selects '
                        f'{int(t_mask.sum())} of {times.size} point(s) on the '
                        f'TLIMS axis {tuple(p["TLIMS"])}')
                    _emit(f'  ⚠️ {l2_skipped[key]}. Widen TLIMS or drop '
                          f'{wname!r} from L2_WINDOWS — not skipped silently.')
                    continue
                subs_w = _restrict_time(subjects_by_model[model],
                                        grain_measures(p, grain), t_mask)
                out_csv = os.path.join(main, f'l2_heart_summary_{model}_lit.csv')
            else:
                raise ValueError(
                    f'unknown L2 window {wname!r}; known: full, lit. A typo here '
                    f'would silently drop a test rather than run it.')
            t_ax = times[t_mask]
            if verbose and wname != 'full':
                _emit(f'\n  — window {wname!r}: {t_ax[0]:.0f}..{t_ax[-1]:.0f} ms '
                      f'({t_ax.size} of {times.size} points)')
            cl_out = {}
            try:
                rows, detected = run_l2_heart(
                    subs_w,
                    times_hep=t_ax, times_tf=t_ax, freqs=freqs,
                    measures=grain_measures(p, grain),
                    schemes=grain_schemes(p, grain),
                    n_perm=int(p['nPerm']), alpha=float(p['ALPHA']),
                    ivw_measure=p['L2_IVW_MEASURE'], max_gb=float(p['L2_MAX_GB']),
                    out_csv=out_csv, verbose=verbose,
                    posthoc_signflip=bool(p['L2_POSTHOC_SIGNFLIP']),
                    max_clusters=p['L2_MAX_CLUSTERS'],
                    clusters_out=cl_out)
                l2_by_model[key] = rows
                detected_by_model[key] = detected
                # The time axis goes with them. A 'lit' cluster's time indices
                # are on the trimmed axis, so a reader handed cells alone would
                # place a 200-600 ms cluster at -100 ms. The mask's own indices
                # into `times` are stored beside every key.
                l2_cells_by_key[key] = dict(cl_out)
                l2_taxis_by_key[key] = np.flatnonzero(t_mask)

                # ── POST-HOC BATTERY, GATED ON DETECTION ────────────────
                # Inside the L2 try/except on purpose. A post-hoc failure must
                # not cost the L2 that already succeeded, and it must not read as
                # "nothing survived"; it is recorded like any other failure.
                if p.get('POSTHOC', True):
                    try:
                        if verbose:
                            _emit(f'\n  — post-hoc battery ({key})')
                        # Every surviving (Measure, Scheme), not `detected`.
                        # `decide_detected` returns one scheme per measure and
                        # strips ivw_shrunk, which is right for deciding what
                        # the run claims but would leave survivors under a
                        # second scheme without LOSO or depth-shape rows. The
                        # battery runs on all of them; ivw_shrunk still cannot
                        # declare an effect, it is only described.
                        _batt = {}
                        for _r in rows:
                            if (_r.get('Status') == 'ok' and _r.get('Survived')
                                    and _r.get('Scheme')):
                                _batt.setdefault(_r['Measure'], [])
                                if _r['Scheme'] not in _batt[_r['Measure']]:
                                    _batt[_r['Measure']].append(_r['Scheme'])
                        ph = run_posthocs(
                            sess_by_sub, subs_w, rows, _batt, p,
                            t_keep=(t_keep if wname == 'full'
                                    else _combine_masks(t_keep, t_mask)),
                            times=t_ax, clusters_by_key=cl_out,
                            n_perm=int(p['POSTHOC_NPERM']),
                            grain=grain, verbose=verbose, emit=_emit)
                        for r in ph:
                            r['Model'], r['Window'] = model, wname
                        posthoc_rows.extend(ph)
                    except Exception as exc:            # noqa: BLE001
                        posthoc_errors[key] = f'{type(exc).__name__}: {exc}'
                        _emit(f'  ⚠️ POST-HOC FAILED for {key!r}: '
                              f'{posthoc_errors[key]}  (the L2 above is '
                              f'unaffected and still valid)')
            except Exception as exc:                    # noqa: BLE001
                # Caught on purpose, and recorded. The L1 maps are already on
                # disk; the other model/window still gets its chance. An L2 that
                # failed must never look like an L2 that found nothing, so the
                # error text goes into the return value, the summary and
                # run_results.npz.
                l2_errors[key] = f'{type(exc).__name__}: {exc}'
                _emit(f'\n  ⚠️ L2 FAILED for {key!r}: {l2_errors[key]}')
                _emit('     The L1 maps are already saved in run_results.npz and '
                      'are unaffected.')

    # Always written, even when empty. An absent file cannot be told apart from
    # a battery that never ran; a present file says what happened (see
    # _posthoc_status_rows).
    _write_csv(os.path.join(main, 'posthoc_summary.csv'),
               posthoc_rows + _posthoc_status_rows(p, posthoc_errors,
                                                   len(posthoc_rows)),
               POSTHOC_FIELDS)

    # ── REWRITE run_results.npz WITH THE L2 ROWS (atomic) ───────────────────
    payload['l2_rows_json'] = np.array(json.dumps(
        {m: _jsonable(r) for m, r in l2_by_model.items()}, allow_nan=False))
    payload['l2_errors_json'] = np.array(json.dumps(l2_errors, allow_nan=False))
    payload['l2_skipped_json'] = np.array(json.dumps(l2_skipped, allow_nan=False))
    payload['posthoc_rows_json'] = np.array(json.dumps(_jsonable(posthoc_rows),
                                                       allow_nan=False))
    payload['posthoc_errors_json'] = np.array(json.dumps(posthoc_errors,
                                                         allow_nan=False))
    # A skipped window is not a complete L2 either. l2_complete must mean
    # "every model x window in L2_WINDOWS was actually tested", or a reader
    # takes an untested window for a null one.
    payload['l2_complete'] = np.array(
        len(l2_errors) == 0 and not l2_skipped and bool(l2_by_model))
    payload.update(_pack_l2_cells(l2_cells_by_key, l2_taxis_by_key))
    _save_npz_atomic(npz_path, payload)

    # ── README + human summary ──────────────────────────────────────────────
    _write_summary(os.path.join(main, 'run_summary.txt'), p, data_dir,
                   run_root, main, session_records, subject_records,
                   l2_by_model, l2_errors, detected_by_model,
                   l2_skipped, posthoc_errors, time.time() - t0,
                   limit=limit, grain=grain)
    if verbose:
        _emit(f'\n  {grain} grain DONE in {time.time() - t0:.1f} s -> {main}')
    return dict(main=main, grain=grain, l1_rows=l1_rows_all, l2=l2_by_model,
                l2_errors=l2_errors, l2_skipped=l2_skipped,
                posthoc_errors=posthoc_errors, detected=detected_by_model,
                times=times, freqs=freqs, npz=npz_path,
                seconds=time.time() - t0)


def run(data_dir, results_root, p=None, limit=None, verbose=True):
    """Segment files -> gates -> L1 (every model) -> L2 (3 schemes) -> results.

    Returns a dict with the paths, the accounting, the L1 rows, the L2 rows per
    (model, window) key, any L2 error and any L2 window that was skipped before
    it could be tested. Never raises for a statistical outcome; raises only
    for a parameter or data problem that would make the output misleading.
    """
    t_start = time.time()
    p = dict(DEFAULT_RUN_PARAMS if p is None else p)
    _check_tag_params(p)
    _check_extraction_params(p)
    grains = check_grains(p)
    pgrain = primary_grain(p)
    # ── Covariate models need the per-segment covariates ───────────────────
    # Same discipline as KEEP_PERBEAT below: set it, and say so. Refusing would
    # make the covariate models impossible to ask for in one line; flipping it
    # silently would let a run's memory and content change without a record.
    # params_json stores what actually ran.
    _need_segcov = [m for m in tuple(p['MODELS'])
                    if m in SEGCOV_MODELS or _base_of(m) in SEGCOV_MODELS]
    if _need_segcov and not p.get('KEEP_SEGCOV'):
        p['KEEP_SEGCOV'] = True
        if verbose:
            _emit('  KEEP_SEGCOV set True because MODELS includes '
                  f'{_need_segcov} — session_measures must compute the '
                  'per-segment covariates. Cheap: a handful of floats per '
                  'segment, not per beat.')
    # 'cond' with no sapo segments is a constant column. INCLUDE_SAPO=False
    # keeps only cond==0 segments, so the model would be rank deficient for
    # every subject and would silently degrade to unadj (subject_slopes drops a
    # degenerate column). Dropping it here means the run does not advertise a
    # model it cannot fit. condXdepth goes with it: it rests on the same
    # column, so with no sapo segments the interaction is a product with a
    # constant and is just as undefined.
    _cond_models = COND_MODELS
    if _cond_models & set(tuple(p['MODELS'])) and not p.get('INCLUDE_SAPO'):
        p['MODELS'] = models_after_cond_rule(p['MODELS'], p.get('INCLUDE_SAPO'))
        if verbose:
            _emit("  'cond' and 'condXdepth' dropped from MODELS: INCLUDE_SAPO is "
                  'False, so every retained segment has cond==0 and the column '
                  'carries no contrast. Set INCLUDE_SAPO=True to fit them.')
        if p['PRIMARY_MODEL'] in _cond_models:
            raise ValueError(
                "PRIMARY_MODEL='cond' with INCLUDE_SAPO=False: the primary model "
                'would have no contrast to fit. Set INCLUDE_SAPO=True or choose '
                'another PRIMARY_MODEL.')
    if 'perbeat' in grains and not p.get('KEEP_PERBEAT'):
        # Set, and said. Silently flipping a memory-relevant flag is how a run
        # comes to use several GB more than expected; refusing instead would
        # make the grain impossible to ask for in one line. So: set it, print
        # it, and let params_json record it as run.
        p['KEEP_PERBEAT'] = True
        if verbose:
            _emit("  KEEP_PERBEAT set True because GRAINS includes 'perbeat' — "
                  'session_measures must keep the per-beat epochs. This costs '
                  'roughly 2-3 GB of RAM on the full cohort.')

    run_root, main, method, tag = run_paths(results_root, p, create=False,
                                            grain=pgrain)
    # Every grain's path, not only the primary one. The grain is a path level,
    # so the longest grain name decides whether this run fits inside the
    # Windows limit. Checking `segment` alone would clear a run that then dies
    # partway through, after the first grain had already been written.
    for _g in grains:
        _check_path_length(run_paths(results_root, p, create=False, grain=_g)[1],
                           verbose)
    if verbose:
        _emit('=' * 92)
        _emit(f'  HEART DEPTH RUNNER — '
              f'{p.get("IC_SELECT_METHOD", "lee")} selection, '
              f'{"/".join(grains)} grain' + ('s' if len(grains) > 1 else ''))
        _emit('=' * 92)
        _emit(f'  data          {data_dir}')
        _emit(f'  results root  {results_root}')
        _emit(f'  method        {method}')
        _emit(f'  param tag     {tag}')
        for _g in grains:
            _emit(f'  -> {os.path.join(run_root, _g + "_wise", "main")}'
                  + ('   (primary)' if _g == pgrain else ''))
        _emit(f'\n  window        {float(p["EPOCH_DUR"]):g} s ending '
              f'{float(p["PRE_END"]):g} s before the report, cut from the END of '
              f'the {EXTRACT_EPOCH_DUR:g} s stored segment')
        _emit(f'  L1 models     {tuple(p["MODELS"])}  (PRIMARY = '
              f'{p["PRIMARY_MODEL"]})')
        _emit(f'  measures      {tuple(p["MEASURES"])}')
        _emit(f'  L2            nPerm={int(p["nPerm"])}, alpha={float(p["ALPHA"]):g}, '
              f'schemes={tuple(p["L2_SCHEMES"])}')
        _emit(f'  gates         MIN_DEPTHS={int(p["MIN_DEPTHS"])} (per session), '
              f'minSeg={int(p["minSeg"])} (pooled per subject), '
              f'MIN_BEATS_UNIT={int(p["MIN_BEATS_UNIT"])}')
        _emit(f'  MIN_CONF      {int(p["MIN_CONF"])} — applied at EXTRACTION, not '
              f'here; recorded so the tag is honest')

    # ── COHORT ──────────────────────────────────────────────────────────────
    sess_by_sub, session_records, subject_records = build_cohort(
        data_dir, p, limit=limit, verbose=verbose)
    _report_duration_proxy(p, session_records, verbose)
    n_sub, n_ses = print_accounting(session_records, subject_records, p, verbose)

    # ── The per-beat memory, measured the moment it exists ──────────────────
    # Print the measured number, name the knob, and say what to drop: a number
    # in the log is what makes an out-of-memory death diagnosable afterwards
    # instead of a folder of cohort CSVs and no explanation.
    if 'perbeat' in grains:
        pb_bytes, pb_note = perbeat_bytes_estimate(sess_by_sub, p)
        if verbose:
            _emit(f'\n  PER-BEAT ARRAYS HELD: {pb_bytes / 1e9:.2f} GB  ({pb_note})')
            _emit('    This is the cost of GRAINS including \'perbeat\' '
                  '(KEEP_PERBEAT). Drop it from GRAINS to run cell+segment only;'
                  ' those two need none of it.')
            if pb_bytes > 4e9:
                _emit(f'    ⚠️ {pb_bytes / 1e9:.1f} GB is a large fraction of a '
                      f'machine with ~12 GB of RAM. If the kernel dies '
                      f'during L1 with no error, this is the first thing to '
                      f'suspect - the cell and segment grains run FIRST and '
                      f'their trees will already be complete.')

    # main/ is created here, after the cohort exists, so an empty main/ cannot
    # be left behind by a run that never got past loading.
    # One main/ per grain, and the cohort tables go in every one: the cohort is
    # identical across grains, but a <grain>_wise/main/ without its own
    # cohort_*.csv would force a reader of that folder to look in a sibling.
    # The paths come from run_paths, the one source of the layout.
    # The collision check runs before the first write. The folder name carries
    # only the swept axes, so two runs differing only in an unnamed knob
    # (wavelet width, HRSP_FREQS, nPerm, PRIMARY_MODEL, a gate) land in the same
    # folder; this refuses that by name instead of letting the second silently
    # overwrite the first. Re-running at the same parameters is a repeat, not a
    # collision, and is allowed.
    assert_no_param_collision(run_root, _params_for_record(p))

    mains = {g: run_paths(results_root, p, create=True, grain=g)[1]
             for g in grains}
    main = mains[pgrain]
    for _m in mains.values():
        _write_csv(os.path.join(_m, 'cohort_sessions.csv'), session_records,
                   ('subject', 'session', 'dur_min_lowerbound', 'n_seg_raw',
                    'n_seg_cond', 'n_seg_full_window', 'n_seg_short',
                    'shortest_window_s', 'n_depths', 'n_seg_used', 'ic',
                    'ecg_clean_ratio', 'ibi_short_prop',
                    'cfa_topo_r2', 'cfa_topo_nbeats',
                    'status', 'reason', 'npz'))
        _write_csv(os.path.join(_m, 'cohort_subjects.csv'), subject_records,
                   ('subject', 'n_sessions_found', 'n_sessions_kept', 'n_segments',
                    'n_depths_pooled', 'ecg_clean_ratio', 'ibi_short_prop',
                    'status', 'reason'))

    if not sess_by_sub:
        # Still a complete, readable run folder: the accounting says why.
        write_readme(run_root, ARM, _describe(p), _params_for_record(p))
        _emit('\n  NO SUBJECT SURVIVED THE COHORT GATES — L1/L2 not attempted. '
              'See cohort_subjects.csv.')
        return dict(run_root=run_root, main=main, method=method, tag=tag,
                    params=p, session_records=session_records,
                    subject_records=subject_records, l1_rows=[], l2={},
                    l2_errors={}, l2_skipped={}, detected={},
                    seconds=time.time() - t_start)

    # ── ONE COMPLETE ANALYSIS PER GRAIN ────────────────────────────────
    # A grain that throws must not take the others with it. Each grain is an
    # independent analysis writing an independent folder; losing one grain to a
    # bug must not discard another that already completed, and a grain that
    # failed must never be readable as a grain that found nothing, so the
    # failure is recorded, printed, and returned.
    by_grain, grain_errors = {}, {}
    for g in grains:
        try:
            by_grain[g] = _run_one_grain(
                g, sess_by_sub, p, data_dir, run_root, mains[g],
                session_records, subject_records, limit=limit, verbose=verbose)
            # A successful rerun retires an old failure marker: nothing else
            # removes RUN_FAILED.txt, so a stale crash report would otherwise
            # sit next to complete results. Renamed, not deleted.
            _stale = os.path.join(mains[g], 'RUN_FAILED.txt')
            if os.path.exists(_stale):
                os.replace(_stale, os.path.join(
                    mains[g], 'RUN_FAILED_superseded_by_later_success.txt'))
        except Exception as exc:                     # noqa: BLE001
            grain_errors[g] = f'{type(exc).__name__}: {exc}'
            _emit(f'\n  ⚠️ GRAIN {g!r} FAILED: {grain_errors[g]}')
            _emit('     The other grains are unaffected. This is a FAILURE, not '
                  'a null - nothing was tested at this grain.')
            # The failure also goes on disk, in that grain's own folder. Console
            # output is lost when the session ends, and a <grain>_wise/main/
            # holding only the two cohort CSVs is indistinguishable from an
            # interrupted run or one that never started this grain. The
            # traceback is written, not just the message, because
            # `KeyError: 'perbeat'` without a frame says nothing.
            try:
                _write_grain_failure(mains[g], g, p, grain_errors[g],
                                     traceback.format_exc())
            except Exception as exc2:                # noqa: BLE001
                _emit(f'     (could not write the failure record: {exc2})')

    # The README is written once, at the run root, because it describes the
    # parameters, which every grain shares. The per-grain record is
    # <grain>_wise/main/run_summary.txt.
    write_readme(run_root, ARM, _describe(p), _params_for_record(p))

    # The top-level keys name the primary grain; `by_grain` is what a
    # multi-grain caller reads.
    head = by_grain.get(pgrain) or (next(iter(by_grain.values()))
                                    if by_grain else {})
    if verbose:
        _emit(f'\n  ALL GRAINS DONE in {time.time() - t_start:.1f} s '
              f'({len(by_grain)} of {len(grains)} produced a result)')
    return dict(run_root=run_root, main=head.get('main', main), method=method,
                tag=tag, params=p,
                session_records=session_records, subject_records=subject_records,
                n_subjects=n_sub, n_sessions=n_ses,
                grains=list(grains), primary_grain=pgrain, by_grain=by_grain,
                grain_errors=grain_errors, mains=mains,
                l1_rows=head.get('l1_rows', []), l2=head.get('l2', {}),
                l2_errors=head.get('l2_errors', {}),
                l2_skipped=head.get('l2_skipped', {}),
                posthoc_errors=head.get('posthoc_errors', {}),
                detected=head.get('detected', {}),
                times=head.get('times'), freqs=head.get('freqs'),
                npz=head.get('npz'), seconds=time.time() - t_start)


def _describe(p):
    sel = 'Lee (functional) IC selection'
    return (f"{sel}, {float(p['EPOCH_DUR']):g}s segments ending "
            f"{float(p['PRE_END']):g}s before the depth report, "
            f"{'/'.join(tuple(p.get('GRAINS') or (GRAIN,)))} grain(s), "
            f"L1 models {tuple(p['MODELS'])} (PRIMARY {p['PRIMARY_MODEL']}), "
            f"L2 cluster permutation nPerm={int(p['nPerm'])} over "
            f"{tuple(p['L2_SCHEMES'])}. Python route "
            f"(heart_depth_run.py).")


def _params_for_record(p):
    """`p` as it should be WRITTEN DOWN (README, params_json, run_summary).

    A pinned-selection table {(subject, session): [ICs]} is replaced by a
    one-line summary: inlined, it puts an unreadable wall into README.txt and
    turns its tuple keys into strings no reader can index back. Replacing
    rather than dropping it keeps the record honest that a selection WAS pinned.
    """
    q = dict(p)
    t = q.get('IC_PRESELECT')
    if isinstance(t, dict):
        n_ic = sum(1 for v in t.values() if len(v))
        q['IC_PRESELECT'] = (f'<pinned selection: {len(t)} session(s), '
                             f'{n_ic} with >=1 IC>')
    return q


def _jsonable(obj):
    """Make numpy/tuple content JSON-serialisable without losing values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else str(v)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if callable(obj):
        return f'<callable {getattr(obj, "__name__", "?")}>'
    return obj


def _write_summary(path, p, data_dir, run_root, main, session_records,
                   subject_records, l2_by_model, l2_errors, detected,
                   l2_skipped=None, posthoc_errors=None, seconds=float('nan'),
                   limit=None, grain=GRAIN):
    """A plain-text summary that answers 'what ran, on whom, and what came out'.

    Everything below is derived from the parameters and results in force.
    Nothing here may be a hardcoded sentence about what this route does or does
    not do: this file is read as the human record of the run, and a literal
    goes stale the moment a parameter is added or flipped.
    """
    l2_skipped = dict(l2_skipped or {})
    posthoc_errors = dict(posthoc_errors or {})
    L = ['HEART DEPTH RUN — Python route', '=' * 78, '',
         f'data           {data_dir}',
         f'run root       {run_root}',
         f'main           {main}',
         # This file describes one grain. A run may produce several, each into
         # its own <grain>_wise/main/ with its own copy of this summary; the
         # numbers below are this grain's and no other's.
         f'grain          {grain}   (of {tuple(p.get("GRAINS") or (grain,))} '
         f'produced by this run)',
         # A measure or scheme this grain does not define is named here, or its
         # absent table would read as a step that was skipped. See
         # GRAIN_MEASURES / GRAIN_SCHEMES for why each restriction exists.
         f'measures       {grain_measures(p, grain)}'
         + (f'   ⚠️ NOT DEFINED at this grain: '
            f'{tuple(m for m in p["MEASURES"] if m not in grain_measures(p, grain))}'
            if len(grain_measures(p, grain)) != len(tuple(p['MEASURES'])) else ''),
         f'schemes        {grain_schemes(p, grain)}'
         + (f'   ⚠️ NOT RUN at this grain: '
            f'{tuple(x for x in p["L2_SCHEMES"] if x not in grain_schemes(p, grain))}'
            if len(grain_schemes(p, grain)) != len(tuple(p['L2_SCHEMES'])) else ''),
         f'wall clock     {seconds:.1f} s', '',
         'COHORT', '-' * 78,
         f'  subjects found   {len(subject_records)}',
         f'  subjects kept    {sum(1 for r in subject_records if r["status"] == "kept")}',
         f'  sessions found   {len(session_records)}',
         f'  sessions kept    {sum(1 for r in session_records if r["status"] == "kept")}',
         '']
    # "subjects found" is only what this run looked at. Under --limit the rest
    # of the cohort was never opened, and the folder tag does not carry the
    # limit, so a smoke test lands in the same folder as a full run.
    if limit is not None:
        L += [f'  ⚠️ --limit {limit}: ONLY THE FIRST {limit} SUBJECT(S) ON DISK '
              f'WERE EXAMINED.',
              '     The rest were never opened — they are absent from this run, '
              'not dropped',
              '     by a gate, and they are NOT in cohort_subjects.csv. SMOKE '
              'TEST, not a cohort.',
              '     ⚠️ The folder tag does not carry --limit, so this run '
              'sits where a full',
              '     run would sit.', '']
    L += ['  ⚠️ MIN_SESSION_MIN was applied to a LOWER BOUND on the recording '
         'length', '     (max sample referenced by block_spans / seg_span), NOT '
         'to the .set header', '     duration — extract_npz.py '
         'never wrote it to the .npz.', '',
         '  DROPPED SESSIONS', '  ' + '-' * 76]
    dropped = [r for r in session_records if r['status'] != 'kept']
    L += [f'    {r["subject"]} ses{r["session"]}: {r["reason"]}' for r in dropped] \
        or ['    (none)']
    L += ['', '  DROPPED SUBJECTS', '  ' + '-' * 76]
    dsub = [r for r in subject_records if r['status'] != 'kept']
    L += [f'    {r["subject"]}: {r["reason"]}' for r in dsub] or ['    (none)']
    # Segments that do not hold the requested window (refused by the library,
    # not clamped). The console says this too, but a console is not a record.
    short = [r for r in session_records if r.get('n_seg_short')]
    L += ['', f'  SEGMENTS SHORTER THAN THE REQUESTED '
              f'{float(p["EPOCH_DUR"]):g} s WINDOW', '  ' + '-' * 76]
    if short:
        L += ['    The library DROPS these rather than clamping them, so '
              'they were EXCLUDED from the',
              f'    analysis - NOT analysed at less than '
              f'{float(p["EPOCH_DUR"]):g} s. The window is anchored to the '
              'segment START, so a segment',
              '    can be long enough and still not hold the seconds it '
              'asks for:']
        L += [f'      {r["subject"]} ses{r["session"]}: {r["n_seg_short"]} of '
              f'{r["n_seg_cond"]} segment(s), shortest window '
              f'{r["shortest_window_s"]} s' for r in short]
    else:
        L += ['    (none — every analysed segment held the whole window)']
    L += ['', 'L2 DETECTION', '-' * 78]
    # One block per (L1 model x L2 window), keyed the way the L2 loop keys
    # them, so the 'lit' tables are named here and a 'lit' L2 that threw shows
    # as failed rather than as never attempted.
    expected = [_l2_key(m, w) for m in tuple(p['MODELS'])
                for w in tuple(p.get('L2_WINDOWS', ('full',)))]
    extra = [k for k in list(l2_by_model) + list(l2_errors) + list(l2_skipped)
             if k not in expected]
    seen = set()
    for key in expected + extra:
        if key in seen:
            continue
        seen.add(key)
        model, wname = _split_l2_key(key)
        prim = '  (PRIMARY)' if model == p['PRIMARY_MODEL'] else ''
        L.append(f'  model {model} / window {wname}{prim}')
        if key in l2_errors:
            L.append(f'    ⚠️ L2 FAILED: {l2_errors[key]}')
            L.append('       The L1 maps in run_results.npz are unaffected.')
            continue
        if key in l2_skipped:
            L.append(f'    ⚠️ NOT RUN: {l2_skipped[key]}')
            L.append('       This is a SKIP, not a null — nothing was tested.')
            continue
        if key not in l2_by_model:
            L.append('    ⚠️ NOT RUN: no result and no error was recorded for '
                     'this model x window.')
            L.append('       This is a SKIP, not a null — nothing was tested.')
            continue
        det = detected.get(key, {})
        for meas, d in det.items():
            if d.get('sig'):
                best = min(c['perm_p'] for c in d['clusters'])
                cfa = ' ⚠️ CFA WINDOW' if any(c['CFA_flag'] for c in d['clusters']) else ''
                L.append(f'    {meas.upper():5s} DETECTED ({d["scheme"]}, '
                         f'{len(d["clusters"])} cluster(s), best p={best:.4f}){cfa}')
            else:
                rows = [r for r in l2_by_model.get(key, [])
                        if r['Measure'] == meas]
                st = {r['Status'] for r in rows}
                L.append(f'    {meas.upper():5s} none   (status: '
                         f'{", ".join(sorted(st)) or "?"})')

    # ── WHAT RAN BESIDES THE MAIN L2, AND WHAT DID NOT ──────────────────────
    # Both lists are derived from the parameters in force, so the summary cannot
    # contradict the run it summarises: it is the human record.
    posthoc_on = bool(p.get('POSTHOC', True))
    signflip_on = bool(p.get('L2_POSTHOC_SIGNFLIP', False))
    windows = tuple(p.get('L2_WINDOWS', ('full',)))
    ran, notrun = [], []
    (ran if posthoc_on else notrun).append(
        'LOSO and depth-shape (per-subject monotonicity) post-hocs '
        '(POSTHOC=%s) -> posthoc_summary.csv. GATED: they run only on an '
        'effect that survived the main model.' % posthoc_on)
    (ran if signflip_on else notrun).append(
        'sign-flip robustness p at L2 (L2_POSTHOC_SIGNFLIP=%s) -> the '
        'Signflip_p column of the l2_heart_summary_* tables.' % signflip_on)
    (ran if 'lit' in windows else notrun).append(
        'the %g-%g ms literature-window L2 -> l2_heart_summary_*_lit.csv '
        '(L2_WINDOWS=%s).' % (lit_ms(p)[0], lit_ms(p)[1], windows))
    notrun += [
        'L1 designs beyond %s — subject_slopes implements no others.'
        % (tuple(p['MODELS']),),
        # Derived from p['GRAINS'], never a literal, so the sentence stays true
        # as grains are added.
        'grains other than %s — %s.'
        % (', '.join(map(repr, tuple(p.get('GRAINS') or (grain,)))),
           ('; '.join(f'{g}: {why}' for g, why in GRAINS_NOT_PORTED.items())
            or 'no other grain exists')),
        # The three below are separate scripts, run after this one.
        'the surrogate R-peak control: it REFITS L1 on shifted peak trains, so '
        'nothing in a finished run can produce it. Run '
        '`run_component_group_tests.py --steps surrogate`.',
        'outcome / demographic associations: run `run_associations.py`.',
        'figures: run the scripts in figures/.',
    ]
    if posthoc_errors:
        L += ['', 'POST-HOC FAILURES', '-' * 78]
        for key in sorted(posthoc_errors):
            m, w = _split_l2_key(key)
            L.append(f'  model {m} / window {w}: ⚠️ {posthoc_errors[key]}')
        L.append('  The L2 above is unaffected. A FAILED battery is not a null '
                 'one — posthoc_summary.csv carries a row saying so.')
    L += ['', 'ALSO RUN IN THIS ROUTE', '-' * 78]
    L += [f'  * {s}' for s in ran] or ['  (nothing beyond the main L2)']
    L += ['', 'NOT RUN IN THIS ROUTE', '-' * 78]
    L += [f'  * {s}' for s in notrun]
    L += ['  An ABSENT section folder means the step never ran — it is not a '
          'null.', '']
    # The file is opened as UTF-8, so every glyph survives here even when the
    # console cannot carry it; _emit's degradation is for stdout only.
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    os.replace(tmp, path)
    return path


def preselect_from_records(session_records):
    """`IC_PRESELECT` rebuilt from the components a run selected, {(subject, session): ICs}.

    With `PIN_SELECTION=True` a cohort rebuild then analyses exactly the components the run
    analysed, instead of selecting again. Built from the run's own per-session records, so it
    cannot drift from what was fit. The whole selected set (`ic_all`) is used; a record without
    it falls back to the top component, and the count is printed.
    """
    table, n_top_only = {}, 0
    for r in session_records:
        if (r.get('status') or '') != 'kept' or r.get('ic') is None:
            continue
        ics = r.get('ic_all')
        if not ics:
            ics, n_top_only = [r['ic']], n_top_only + 1
        table[(str(r['subject']), int(r['session']))] = np.asarray(
            [int(v) for v in ics], int)
    if n_top_only:
        print('  WARNING %d of %d kept session(s) carry no `ic_all`, so the rebuild is '
              'pinned to the top component only.' % (n_top_only, len(table)))
    return table


def params_of_run(main_dir):
    """(params, grain, session records, measures) of a run, read back from run_results.npz."""
    fp = os.path.join(main_dir, 'run_results.npz')
    if not os.path.isfile(fp):
        raise FileNotFoundError('no run_results.npz in %s' % main_dir)
    with np.load(fp, allow_pickle=True) as z:
        if 'params_json' not in z.files:
            raise KeyError('%s carries no params_json, so the run cannot be rebuilt' % fp)
        p = json.loads(str(z['params_json']))
        grain = str(z['grain']) if 'grain' in z.files else 'segment'
        recs = (json.loads(str(z['session_records_json']))
                if 'session_records_json' in z.files else [])
        meas = ([str(m) for m in z['measures']] if 'measures' in z.files
                else ['hep', 'hrsp', 'hepc'])
    return p, grain, recs, meas
