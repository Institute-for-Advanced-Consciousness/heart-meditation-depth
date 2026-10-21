# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
result_path: where a result goes.

    <results_root>/<ARM>/[<method>/]<param-tag>/[<grain>_wise/]<section>/<file>

    FMT_DEPTH/          dur60_nosapo_standardized_th5-8_tmplROI_gedR2-0.15_gedTop2/main/...
    HEART_DEPTH/        lee-hrsp-raw_1IC_dur60_bl_nosapo/segment_wise/main/...
    HEART_DEPTH_SCALP/  dur60_nosapo_bl-150--50glm_w5_f4to30_prim-session_nperm500/main/...

(fmt_depth_tag, heart_run_tag and heart_depth_tag at the paper's settings in config.py;
no current caller passes a method folder.)

The grain level is part of the path (see `result_dir`). A grained arm's three grains are
three analyses, not three views of one; '' means the arm is single-grain and gets no such
level at all.

The rule the tag has to satisfy: every varied knob must reach the folder name (or be
caught by the collision guard). A knob that varies but is absent from the tag means run
N+1 overwrites run N and the difference is invisible afterwards. If you add a parameter
to a pipeline and vary it, add it here.

Conversely a knob that is not varied should stay out of the tag: the path is ~180
characters against Windows' 260 limit, and the FMT electrode set alone is 60+ characters,
which is why long sets are hashed rather than spelled out.

Sections are fixed, and unknown ones are refused:
    main  covariates  surrogate_control  source_localization  outcomes_demographics

Folders are created on call, so an empty surrogate_control/ can never imply the
control ran.
"""
from __future__ import annotations
import io
import json
import os
# No numpy: this module decides paths and is imported by everything, including
# contexts that have no business pulling in an array library.
import re

SECTIONS = ('main', 'covariates', 'surrogate_control',
            'source_localization', 'outcomes_demographics')

ARMS = ('FMT_DEPTH', 'FMT_PERIOD', 'HEART_DEPTH', 'HEART_PERIOD',
        'HEART_DEPTH_SCALP')

# The grain level. A grained arm puts <grain>_wise between the run root and the section:
#     <ARM>/[<method>/]<tag>/<grain>_wise/<section>
# and '' means no grain folder at all (the single-grain arms: FMT and the period arms).
# Unknown grains are refused, for the same reason as SECTIONS: a mistyped grain would
# silently create another level, and a run would land where no coverage scan looks.
GRAINS = ('cell', 'segment', 'perbeat')


def _as_labels(v):
    """Normalise an electrode-set knob to a list of labels.

    A bare string is one label, not len(label) labels: `GED_TEMPLATE_ELEC='fz'` must tag
    as one electrode, as make_frontal_template treats it, not as the labels 'f' and 'z'.
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def _hash_labels(labels):
    """Order- and case-insensitive djb2 over an electrode set -> 6 hex chars.

    Sorted and lower-cased before hashing (mod 2^24), so a reordering of the set does
    not change the folder while a change of set does.
    """
    txt = ','.join(sorted(str(x).lower() for x in labels))
    h = 5381
    for ch in txt:
        h = (h * 33 + ord(ch)) % 16777216
    return format(h, '06x')


def _pretty_elec(label):
    """'fz' -> 'Fz', 'cz' -> 'Cz', 'fc1' -> 'FC1' (the 10-20 convention).

    Uppercase throughout, except a trailing 'z' (which means midline, not the
    letter), so the token reads as the electrode a person would say out loud.
    Alphanumeric only, so it is always a legal path segment.
    """
    up = str(label).upper()
    return up[:-1] + 'z' if up.endswith('Z') else up


def _template_token(tmpl, elecs):
    """Human-readable body for the `tmpl` folder token.

    In practice the template is a one-electrode or whole-ROI choice, so the token says
    which:

        the ROI (== FMT_ELECS, or None)  -> 'ROI'
        one centre                       -> 'Fz'    / 'Cz'
        two or three centres             -> 'Fz+Cz'
        larger, and not the ROI          -> '<n>x<hash>'  (unreadable, but it still
                                            separates the folders)

    'ROI' is reserved for the ROI itself: if any multi-electrode template were spelled
    'ROI', two different R2 screens would land in one folder and the second run would
    overwrite the first with nothing in the name to show it.
    """
    if sorted(x.lower() for x in tmpl) == sorted(x.lower() for x in elecs):
        return 'ROI'
    if len(tmpl) <= 3:
        return '+'.join(_pretty_elec(x) for x in tmpl)
    return f"{len(tmpl)}x{_hash_labels(tmpl)}"


# ===========================================================================
# TAG-SUPPRESSED DEFAULTS: values that get no folder token at their default
# ===========================================================================
# These are emitted only when they differ from the values below, so the settled case
# gets a short name and the folders still separate the moment one of them moves. This
# is not the same as dropping them: a knob that varies but is missing from the folder
# name lets the next run silently overwrite the previous one. Changing a reference value
# can map two different settings to the same folder name, so check the folders already
# on disk before changing one.
TAG_DEFAULTS = {
    'MIN_CONF':        4,
    'PRE_END':         3,
    'MIN_DEPTHS':      2,
    'minSeg':          5,
    'MIN_SESSION_MIN': 45,
    'nPerm':           1000,
    # FMT_ELECS is the scalp ROI and (when GED_TEMPLATE_ELEC is None) the GED template
    # centres, so changing it changes both measures; at any other set the `elec` token
    # is emitted. A set that names FCz or AFz differs from this one even though its
    # scalp measure uses the same 10 recorded channels: its ROI template places both as
    # virtual centres from standard_1005 neighbours, which is a different template and
    # can select a different GED component.
    'FMT_ELECS':       ('cz', 'f1', 'f2', 'f3', 'f4',
                        'fc1', 'fc2', 'fc3', 'fc4', 'fz'),
}


def _gate_tokens(p, t):
    """Append the gate tokens, each only if it differs from TAG_DEFAULTS."""
    if p.get('MIN_CONF', TAG_DEFAULTS['MIN_CONF']) != TAG_DEFAULTS['MIN_CONF']:
        t.append(f"conf{p['MIN_CONF']}")
    if p.get('PRE_END', TAG_DEFAULTS['PRE_END']) != TAG_DEFAULTS['PRE_END']:
        t.append(f"pre{p['PRE_END']}")
    if p.get('MIN_DEPTHS', TAG_DEFAULTS['MIN_DEPTHS']) != TAG_DEFAULTS['MIN_DEPTHS']:
        t.append(f"mindep{p['MIN_DEPTHS']}")
    if p.get('minSeg', TAG_DEFAULTS['minSeg']) != TAG_DEFAULTS['minSeg']:
        t.append(f"minseg{p['minSeg']}")
    # MIN_SESSION_MIN keeps its falsy check: 0/None means "no gate", which is a
    # real setting and is not the same as "at the default".
    ms = p.get('MIN_SESSION_MIN')
    if ms and ms != TAG_DEFAULTS['MIN_SESSION_MIN']:
        t.append(f"minses{ms:g}")
    return t


def _nperm_token(p, t):
    if p.get('nPerm') and p['nPerm'] != TAG_DEFAULTS['nPerm']:
        t.append(f"nperm{p['nPerm']}")
    return t


def fmt_depth_tag(p):
    """Parameter tag for an FMT depth run."""
    # EPOCH_DUR stays unconditional: it is swept constantly and is the one
    # window parameter a reader needs to see at a glance.
    t = [f"dur{p['EPOCH_DUR']}"]
    _gate_tokens(p, t)
    t.append('nosapo+sapo' if p.get('INCLUDE_SAPO', True) else 'nosapo')
    if p.get('STANDARDIZE_WITHIN_SESSION'):
        t.append('standardized')
    t.append(f"th{p['THETA_BP'][0]:g}-{p['THETA_BP'][1]:g}")
    elecs = _as_labels(p.get('FMT_ELECS'))
    if tuple(sorted(x.lower() for x in elecs)) != TAG_DEFAULTS['FMT_ELECS']:
        t.append(f"elec{len(elecs)}x{_hash_labels(elecs)}")
    # The template centre set changes which components pass the R2 screen and is not
    # implied by FMT_ELECS (a single-Fz, Zuure-style template and the broad ROI one
    # score the same topography differently), so this token is unconditional. `None`
    # means "reuse FMT_ELECS" and is spelled 'ROI' rather than left blank.
    tmpl = _as_labels(p.get('GED_TEMPLATE_ELEC')) or elecs
    t.append(f"tmpl{_template_token(tmpl, elecs)}")
    t.append(f"gedR2-{p['GED_R2_THRESH']:.2f}")
    t.append(f"gedTop{p['GED_TOP_N']}")
    # Emitted only when > 1, so the default keeps the shorter name, and a run that
    # combines components never shares a folder with one that does not.
    if int(p.get('GED_MAX_COMP', 1)) > 1:
        t.append(f"gedComp{int(p['GED_MAX_COMP'])}")
    if p.get('DEPTH_MODE', 'ordinal') != 'ordinal':
        t.append(p['DEPTH_MODE'])
    _nperm_token(p, t)
    return '_'.join(t)


def heart_depth_tag(p):
    """Parameter tag for a HEART depth run (the part after the method folder)."""
    t = [f"dur{p.get('EPOCH_DUR', 60)}"]
    _gate_tokens(p, t)
    t.append('nosapo+sapo' if p.get('INCLUDE_SAPO', True) else 'nosapo')
    if p.get('STANDARDIZE_WITHIN_SESSION'):
        t.append('standardized')
    # The baseline is the most result-changing switch in this arm and must be in the
    # tag: a baseline-on and a baseline-off run are different models (and for HEP/HRSP
    # a different df), not two readings of one.
    t.append(f"bl{p['BASELINE_WIN'][0]:g}--{abs(p['BASELINE_WIN'][1]):g}"
             f"{p.get('BASELINE_METHOD', 'glm')}" if p.get('DO_BASELINE_CORRECT')
             else 'nobl')
    t.append(f"w{p['HRSP_WAVE_WIDTH']:g}")
    if p.get('WAVELET_ON', 'window') != 'window':
        t.append('cwtseg')
    t.append(f"f{p['HRSP_FREQS'][0]:g}to{p['HRSP_FREQS'][-1]:g}")
    if p.get('PRIMARY_MODEL', 'session') != 'unadj':
        t.append(f"prim-{p.get('PRIMARY_MODEL', 'session')}")
    if not p.get('DO_CLEAN_RR', True):
        t.append('norrclean')
    _nperm_token(p, t)
    return '_'.join(t)


def lee_method_tag(p):
    """The selection-method folder for the HEART Lee path.

    Format:
        lee_<criterion>_<lo>-<hi>Hz_<t0>-<t1>ms_<thresh>IC
    plus the sham-draw count, which sets the SD every score is divided by and so
    is part of the statistic, not an implementation detail.
    """
    b, w = p['HRSP_SEL_BANDS'], p['HRSP_SEL_WIN']
    tag = (f"lee_{p['HRSP_SEL_CRITERION']}_{b[0]:g}-{b[1]:g}Hz_"
           f"{w[0]:g}-{w[1]:g}ms_{p['IC_SEL_THRESH']:g}IC_{p['IC_SEL_NORM']}")
    if 'sham' in str(p.get('IC_SEL_NORM', '')):
        tag += f"{p.get('IC_SEL_SHAM_N', 50)}"
    return tag


def heart_method_tag(p):
    """The method folder for a HEART depth run.

    Only the functional single-component selection ('lee') is implemented. An
    unknown method is refused, not defaulted to 'lee', for the same reason
    SECTIONS and GRAINS are refused above: a run must not land in a folder
    that describes selection parameters it never read.
    """
    method = str(p.get('IC_SELECT_METHOD', 'lee'))
    if method == 'lee':
        return lee_method_tag(p)
    raise ValueError(f"IC_SELECT_METHOD must be 'lee', got {method!r}")


# ===========================================================================
# THE SHORT RUN FOLDER: one level, the axes a reader needs
# ===========================================================================
# The folder name carries selection, number of components, window length,
# baseline correction and condition. Every other parameter is recorded in
# README.txt and run_summary.txt, and `assert_no_param_collision` refuses to
# reuse a folder for a run whose parameters differ from the recorded ones.


def _selection_token(p):
    """The short selection label: what was selected and on what criterion."""
    method = str(p.get('IC_SELECT_METHOD', 'lee'))
    if method != 'lee':
        raise ValueError(f"IC_SELECT_METHOD must be 'lee', got {method!r}")
    tok = f"lee-{p.get('HRSP_SEL_CRITERION', 'hrsp')}"
    norm = str(p.get('IC_SEL_NORM', ''))
    if norm:
        tok += f"-{norm.replace('_', '')}"
    return tok


def heart_run_tag(p):
    """The one folder a HEART depth run writes into, under the arm.

        <selection>_<n>IC_dur<D>_<bl|nobl>_<condition>[_<RUN_LABEL>]
        e.g. lee-hrsp-shamraw_1IC_dur60_bl_nosapo+sapo

    The name carries only the axes a reader needs; everything else lives in README.txt
    and run_summary.txt. Two runs differing only in an unnamed parameter (wavelet width,
    HRSP_FREQS, nPerm, PRIMARY_MODEL, a gate) map to the same folder, so run() calls
    `assert_no_param_collision` before it writes: the folder records its own parameters,
    and a run whose parameters differ from the recorded ones is refused by name rather
    than allowed to overwrite.

    `RUN_LABEL` gives such a run its own folder, which is what makes a sweep over an
    unnamed parameter possible inside one results_root: `p['RUN_LABEL'] = 'w7'` appends
    `_w7`. Empty or absent appends nothing. It is a name, not a parameter (it is in
    `_COLLISION_IGNORE`, so it never shows up as a difference in a collision report), and
    it is sanitised to [A-Za-z0-9.+-] so it cannot inject a path separator.
    """
    t = [_selection_token(p),
         f"{p.get('IC_SEL_THRESH', 1):g}IC",
         f"dur{p.get('EPOCH_DUR', 60):g}",
         'bl' if p.get('DO_BASELINE_CORRECT') else 'nobl',
         'nosapo+sapo' if p.get('INCLUDE_SAPO', True) else 'nosapo']
    # Window position and what the wavelet runs on, only when they differ from the
    # paper's first submission, so its folders keep their names.
    if p.get('PRE_END', TAG_DEFAULTS['PRE_END']) != TAG_DEFAULTS['PRE_END']:
        t.append(f"pre{p['PRE_END']:g}")
    if p.get('WAVELET_ON', 'window') != 'window':
        t.append('cwtseg')
    lbl = run_label_token(p)
    if lbl:
        t.append(lbl)
    return '_'.join(t)


def run_label_token(p):
    """`RUN_LABEL` reduced to a safe folder token, or '' when unset.

    Sanitised, not trusted: a label is typed by hand and goes straight into a path, so
    anything that is not [A-Za-z0-9.+-] becomes '-', and the token is cut to 40
    characters. A label of only illegal characters yields '' rather than a folder called
    '----', which would collide with every other such label.
    """
    raw = str(p.get('RUN_LABEL') or '').strip()
    if not raw:
        return ''
    tok = re.sub(r'[^A-Za-z0-9.+-]', '-', raw).strip('-')
    return tok[:40]


# -- the collision guard -----------------------------------------------------
# Parameters that legitimately differ between two runs sharing a folder, i.e.
# the ones that do not change the answer. Everything else, if it moves, means
# this is a different analysis wearing the same name.
_COLLISION_IGNORE = frozenset({
    'IC_PRESELECT', 'IC_PRESELECT_ROWS', 'IC_PRESELECT_INFO',
    'GRAINS', 'KEEP_PERBEAT',        # a grain is a path level, not an analysis
    'L2_MAX_GB', 'L2_MAX_CLUSTERS',  # memory ceilings
    'SESSION_DUR_SOURCE',
    'KEEP_SEGCOV',                   # set by run() from MODELS, not a knob
    # A name, not a parameter: RUN_LABEL is already in the folder name, so two runs
    # with different labels never share a folder.
    'RUN_LABEL',
    # A speed knob: N_JOBS only decides how many worker processes fan out over the
    # per-session loop. The per-session work is independent and the results are
    # collected in input order, so the answer is identical at any value (see the
    # N_JOBS note in heart_scalp_pipeline). Comparing it would refuse a legitimate
    # repeat at a different worker count.
    'N_JOBS',
})


def _comparable(p):
    """`p` reduced to the values that decide the answer, as plain strings."""
    def norm(v):
        if isinstance(v, (list, tuple)):
            return tuple(norm(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((str(k), norm(x)) for k, x in v.items()))
        if isinstance(v, float) and v == int(v):
            return int(v)
        return v
    return {str(k): repr(norm(v)) for k, v in p.items()
            if k not in _COLLISION_IGNORE}


def params_stamp_path(run_root):
    return os.path.join(run_root, '.run_params.json')


def assert_no_param_collision(run_root, p):
    """Refuse to reuse a run folder for a different analysis. -> list of changes.

    Writes a `.run_params.json` stamp on first use and compares against it after
    that. Returns the (possibly empty) list of differing keys; raises when the
    folder already belongs to a different parameter set.

    The folder name does not carry wavelet width, frequency range, nPerm or
    PRIMARY_MODEL, so without this check a re-run at different values would quietly
    replace the earlier result. Re-running at the same parameters is normal and stays
    allowed: that is a repeat, not a collision.
    """
    stamp = params_stamp_path(run_root)
    now = _comparable(p)
    if not os.path.isfile(stamp):
        os.makedirs(run_root, exist_ok=True)
        with io.open(stamp, 'w', encoding='utf-8') as fh:
            json.dump(now, fh, indent=1, sort_keys=True)
        return []
    try:
        with io.open(stamp, encoding='utf-8') as fh:
            was = json.load(fh)
    except (OSError, ValueError):
        # An unreadable stamp must not block a run; rewrite it and say nothing
        # was compared, rather than pretend the folder is clean.
        with io.open(stamp, 'w', encoding='utf-8') as fh:
            json.dump(now, fh, indent=1, sort_keys=True)
        return []
    diff = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
    if diff:
        lines = '\n'.join(
            f'    {k}: was {was.get(k, "<absent>")}  ->  now {now.get(k, "<absent>")}'
            for k in diff[:12])
        more = '' if len(diff) <= 12 else f'\n    ... and {len(diff) - 12} more'
        raise ValueError(
            f'RUN FOLDER COLLISION - this folder already holds a DIFFERENT '
            f'analysis:\n  {run_root}\n'
            f'  {len(diff)} parameter(s) differ from the run that created it:\n'
            f'{lines}{more}\n'
            f'  The folder name carries only selection / IC / duration / '
            f'baseline / condition, so these differences do not change it and '
            f'the new run would OVERWRITE the old one.\n'
            f"  FIX: set p['RUN_LABEL'] to something naming what you changed "
            f"(e.g. p['RUN_LABEL'] = '{diff[0].lower()}') - it is appended to "
            f'the folder name, so this run gets its own directory and the '
            f'earlier one is left intact. That is how a sweep over a parameter '
            f'the name does not carry is meant to be run.\n'
            f'  Or: delete/rename {run_root!r}, or change one of the parameters '
            f'the name already carries.')
    return diff


def result_dir(results_root, arm, tag, section='main', method=None, create=True,
               grain=''):
    """<results_root>/<arm>/[<method>/]<tag>/[<grain>_wise/]<section>, on call.

    Unknown arms, sections and grains are refused: a mistyped section silently creating
    a sixth folder is how a result tree stops being readable.

    `grain` '' (the default, or None) means no grain folder at all: the single-grain case
    (FMT and the period arms). The grain is part of the path because three grains of one
    variant are three different analyses of the same data; collapsing them into one
    folder would let a run at one grain answer for the other two, and a coverage scan
    could not tell "computed and found nothing" from "never computed". Callers that vary
    the grain pass it here rather than splicing the level in themselves.
    """
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm!r}; expected one of {ARMS}')
    if section not in SECTIONS:
        raise ValueError(f'unknown section {section!r}; expected one of {SECTIONS}')
    if grain and grain not in GRAINS:
        raise ValueError(f'unknown grain {grain!r}; expected one of {GRAINS} '
                         f"or '' for a single-grain arm")
    parts = ([results_root, arm] + ([method] if method else []) + [tag]
             + ([f'{grain}_wise'] if grain else []) + [section])
    d = os.path.join(*parts)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def write_readme(run_root, arm, description, params):
    """A README that spells out in full what the folder name only hashes or omits.

    The tag may carry `elec<n>x<hash>`, which distinguishes two electrode sets but cannot
    be read back into labels. This file is where a reader recovers what the run used.
    """
    os.makedirs(run_root, exist_ok=True)
    lines = [f'{arm} - {description}', '=' * 72, '',
             'Sections (present only if that step ran):',
             '  main/  covariates/  surrogate_control/  source_localization/  '
             'outcomes_demographics/', '',
             'FULL PARAMETERS FOR THIS RUN', '-' * 72]
    for k in sorted(params):
        v = params[k]
        if callable(v):
            continue
        lines.append(f'  {k:28s} {v}')
    with open(os.path.join(run_root, 'README.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
