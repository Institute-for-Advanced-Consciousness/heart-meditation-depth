# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
heart_surrogate: the shifted-R-peak control for the heartbeat depth effect.

`main(argv)` is called by run_component_group_tests.py (step surrogate) with `--data`,
`--results-root`, `--params` (the setting, as JSON), `--n-surrogate`, `--max-unshiftable`,
`--stat`, `--tfce-nperm`, `--cells` (the tested cells, as JSON) and `--out`.

What it answers: is the depth effect locked to the heartbeat, or is it ongoing activity that
co-varies with depth and would appear at any time point? A rigid shift of the R-peak train
preserves the IBI sequence, its variability and the beat count exactly, and destroys only
the phase relationship to the heart.

It does not separate a neural HEP from the cardiac field artifact: both are R-peak locked
and both vanish under shifting. The `ecgAmp` covariate model addresses that.

It refuses a sham-selected run. If IC selection ran with `IC_SEL_NORM in ('sham',
'sham_raw')` and `HRSP_SEL_CRITERION == 'hrsp'` under `lee`, the component was chosen for
having a large real-vs-shifted contrast, and testing that component with a shifted train is
biased in favour of the effect. Under `criterion='ppc'` the sham branch never executes (it
is gated on both settings), so a ppc run is not sham-selected whatever the folder tag says.
That case is allowed.

How the control is kept valid:
1. Selection is held fixed. The surrogate varies R-peak timing and nothing else, so the
   analysed component must not change. The real cohort's selection is captured and
   replayed through `PIN_SELECTION`, which makes even `lee` skip its selection pass.
   Otherwise `lee` would re-select on shifted peaks (a different question) and would also
   rebuild the pooled selection CWT once per surrogate, the dominant cost of a lee run.
2. Integrity check. A shift of zero through the surrogate path must reproduce the real
   measures. If caching or pinning ever returned the real epochs for a shifted train, the
   null would equal the data and would appear to confirm every effect. `--skip-assert` is
   for debugging only and prints a warning.
3. The beat count is preserved. A shift that pushed epochs out of a segment would drop
   beats and change the L1 weights, and a surrogate whose weights differ from the real fit
   is not a null for that fit. A session that admits no such shift is excluded from that
   draw (never analysed with its real peaks), and the draw is rejected when more than
   `--max-unshiftable` of the sessions are excluded.

p_surr = (1 + #{surr >= real}) / (1 + N_valid), so N floors the p at 1/(1+N): N = 50
cannot give p < .0196. The number of distinct surrogates is bounded by the admissible
integer shifts (~128 at fs = 250, +/-500 ms, 60 bpm), so asking for many more than ~100
adds no resolution.

--stat. The null is built on the strongest cluster's |SumAbsT| (`cluster`, the default),
on max |TFCE| of the observed weighted t-map (`tfce`: l2_tfce, Smith & Nichols, E .5, H 2,
dh .1, with the L2's own subjects and weights), or on both from the same draws (`both`).
A tfce cell is gated by the real TFCE L2 (p < ALPHA at --tfce-nperm) and also gets a
per-cell surrogate p-map in <out>_tfce_maps.npz.

Scheme ivw_own (l2_heart: shrunk inverse-variance weights from the tested measure's own L1
slope variance) is allowed at any grain. Its weights are re-estimated on every draw from
that draw's own L1 slope variance: the shifted peak trains give a different L1 fit, hence
a different svar per participant, a different median and different weights. This is
deliberate. The surrogate replays the whole procedure on shifted peaks, and under ivw_own
estimating the weights is part of the procedure; freezing the real fit's weights would give
the null a weighting tuned to the real data. Both statistics do it, `cluster` through
run_l2_heart and `tfce` through l2_tfce.prepare_measure, each given the draw's own subject
list.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from heart_depth_run import DEFAULT_RUN_PARAMS, build_cohort, run_paths, lit_ms, _restrict_time
from heart_depth_pipeline import subject_slopes, tlims_mask   # noqa: E402
from l2_heart import run_l2_heart                   # noqa: E402
from l2_heart import (                              # noqa: E402
    build_ivw_weights, _null_t_weighted, MIN_SUB_L2, SEED as L2_SEED)
from l2_tfce import (                               # noqa: E402
    tfce_scores, tfce_perm_weighted_multi, prepare_measure,
    DEFAULT_E as TFCE_E, DEFAULT_H as TFCE_H, DEFAULT_DH as TFCE_DH)
from surrogate_rpeaks import (                      # noqa: E402
    guard_from_windows, make_surrogate_rpeaks_retry, SURR_SHIFT_MS)

SURR_ASSERT_TOL = 1e-6          # tolerance of the zero-shift integrity check

# ── the statistic the null is built on ──────────────────────────────────────
# 'cluster' : the strongest cluster's |SumAbsT| from run_l2_heart (the default).
# 'tfce'    : max |TFCE| over the tested window of the observed weighted t-map,
#             Smith & Nichols TFCE (l2_tfce, convention 'sn', E .5, H 2, dh .1).
#             Subjects, keep rule and weights are the L2's own
#             (l2_tfce.prepare_measure -> _stack_measure, _listwise_keep,
#             build_weights; build_ivw_weights for ivw_shrunk).
# 'both'    : both, from the same cohort pass per draw, so the two nulls rest on
#             the same shifted peak trains.
STATS = ('cluster', 'tfce')
TFCE_CONVENTION = 'sn'


def _parse_stat(stat):
    s = str(stat or 'cluster').lower()
    if s == 'both':
        return STATS
    if s in STATS:
        return (s,)
    raise ValueError(f'stat must be one of cluster / tfce / both, got {stat!r}')


# ═══════════════════════════════════════════════════════════════════════════
# Refusal of sham-selected runs
# ═══════════════════════════════════════════════════════════════════════════

def sham_selected(p):
    """(is_rigged, why): is this configuration's selection sham-normalised?"""
    method = str(p.get('IC_SELECT_METHOD', '')).lower()
    norm = str(p.get('IC_SEL_NORM', '')).lower()
    crit = str(p.get('HRSP_SEL_CRITERION', '')).lower()
    if method != 'lee':
        return False, f'{method}: not the functional selection, no sham is scored'
    if norm not in ('sham', 'sham_raw'):
        return False, f"IC_SEL_NORM={norm!r}: no sham normalisation"
    if crit != 'hrsp':
        return False, (f"criterion={crit!r}: the sham branch is gated on "
                       f"criterion=='hrsp' and never executes here")
    return True, (
        f"IC_SEL_NORM={norm!r} with criterion='hrsp' under {method!r}: the "
        f"component was SELECTED for having a large real-vs-shifted contrast, "
        f"so a shifted-peak control on that same component is rigged in favour "
        f"of the effect. Run the "
        f"control on a selection that scores no sham, or accept that only a FAILURE "
        f"would be informative.")


# ═══════════════════════════════════════════════════════════════════════════
# The cohort, real and shifted
# ═══════════════════════════════════════════════════════════════════════════

def capture_selection(sessions_by_subject):
    """{(subject, session): [ic indices]} from a real cohort build.

    This pins the analysed component for every surrogate. A session the real build
    dropped is absent, and a surrogate build drops it for the same reason: the gates do
    not depend on peak timing.
    """
    table = {}
    for sid, sess in sessions_by_subject.items():
        for s in sess:
            ics = s.get('ic_all')
            if ics is None:
                ics = [s['ic']]
            table[(str(sid), int(s['session']))] = [int(v) for v in ics]
    return table


def cohort_L1(sessions_by_subject, p, model, grain, measures):
    """Per-subject L1 slope maps -> the list run_l2_heart consumes."""
    subs = []
    for sid, sess in sessions_by_subject.items():
        rec = {'subject': sid}
        for meas in measures:
            beta, svar, nu = subject_slopes(sess, meas, p, model=model,
                                            grain=grain)
            # A tuple, not a dict: l2_heart._entry unpacks `beta, svar, n_units = e`,
            # and a dict would unpack to its keys.
            rec[meas] = (beta, svar, nu)
        subs.append(rec)
    return subs


def _stat_of(rows, measure, scheme):
    """The statistic the null is built on: the strongest cluster's |SumAbsT|.

    The same quantity for real and surrogate, taken from the same column of the same
    table. A cell with no cluster contributes 0.0, a real "nothing here", which is what a
    surrogate should mostly produce.
    """
    best = 0.0
    for r in rows:
        if str(r.get('Measure')) != measure or str(r.get('Scheme')) != scheme:
            continue
        v = r.get('SumAbsT')
        try:
            v = abs(float(v))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            best = max(best, v)
    return best


def _tfce_of(subs, measure, scheme, ivw_all, keep_inputs=False):
    """The TFCE statistic of one cell from one L1 subject list, or None.

    Subjects and weights exactly as the L2 builds them (prepare_measure: the subjects with
    a map, the listwise finite-map keep rule, build_weights for the scheme), MIN_SUB_L2 as
    run_l2_heart applies it, then the observed weighted t-map (identity flip through
    l2_heart._null_t_weighted, the same arithmetic the permutation drivers use for
    column 0) and its Smith & Nichols TFCE.

    Under scheme 'ivw_own' prepare_measure estimates the weights from `subs` itself (the
    measure's own svar), so on a draw they are that draw's weights; `ivw_all` is not used.
    See the module docstring.

    -> dict(stat = max |TFCE| over the map, tfce, t, n_used, n_drop[, X, w]).
    None when the cell cannot be tested (no maps, too few subjects, or a weight vector
    build_weights refuses): the caller treats that as a failed fit, never as a statistic
    of 0.
    """
    try:
        prep = prepare_measure(subs, measure, scheme, ivw_all)
    except (ValueError, ZeroDivisionError):
        return None
    if prep is None:
        return None
    Xk, w, n_drop, _kept = prep
    n_sub = int(Xk.shape[-1])
    if n_sub < MIN_SUB_L2:
        return None
    X3 = Xk if Xk.ndim == 3 else Xk[None]
    nF, nT = X3.shape[0], X3.shape[1]
    t = _null_t_weighted(X3.reshape(nF * nT, n_sub), w,
                         np.ones((n_sub, 1)))[:, 0].reshape(nF, nT)
    tf = tfce_scores(t, TFCE_E, TFCE_H, TFCE_DH, TFCE_CONVENTION)
    out = dict(stat=float(np.max(np.abs(tf))), tfce=tf, t=t, n_used=n_sub,
               n_drop=int(n_drop))
    if keep_inputs:
        out.update(X=Xk, w=w)
    return out


def _tfce_group(subs, g, ivw_measure, keep_inputs=False):
    """{cell key: _tfce_of(...) or None} for every cell of one (grain, model,
    window) group, with the ivw weights built once over the group's subjects, as
    run_l2_heart builds them once per call."""
    ivw_all = None
    if 'ivw_shrunk' in g['schemes']:
        try:
            ivw_all, _ = build_ivw_weights(subs, ivw_measure, verbose=False)
        except ValueError:
            ivw_all = None          # ivw_shrunk cells then come back None
    return {_cell_key(c): _tfce_of(subs, str(c['measure']).lower(),
                                   str(c['scheme']), ivw_all, keep_inputs)
            for c in g['cells']}


# ═══════════════════════════════════════════════════════════════════════════
# The control
# ═══════════════════════════════════════════════════════════════════════════

def _cell_key(c):
    # The window is part of the cell's identity. A 'lit' cluster is a different test
    # from the full-window one (its own axis, its own CFT, its own null), so its
    # surrogate must be built on the lit axis.
    return (str(c['grain']), str(c['measure']).lower(), str(c['model']),
            str(c['scheme']), str(c.get('window') or 'full'))


def _group_cells(cells, ivw_measure='hep'):
    """{(grain, model, window): {'measures': [...], 'schemes': [...], 'cells': [...]}}

    Grouping makes a control over many cells affordable. `build_cohort` is the whole cost
    of a draw (5-8 min with the selection pinned, against seconds for cohort_L1 plus one
    L2); `cohort_L1` depends on (grain, model) and the L2 on the measures and schemes, so
    one cohort pass answers every cell of a run.
    """
    groups = {}
    for c in cells:
        k = (str(c['grain']), str(c['model']), str(c.get('window') or 'full'))
        g = groups.setdefault(k, {'measures': [], 'schemes': [], 'cells': []})
        m, sch = str(c['measure']).lower(), str(c['scheme'])
        if m not in g['measures']:
            g['measures'].append(m)
        if sch not in g['schemes']:
            g['schemes'].append(sch)
        g['cells'].append(c)
    # `ivw_shrunk` needs the IVW measure fitted even if no cell asked for it:
    # `build_ivw_weights` takes the weights from the unadjusted slope variance of
    # `L2_IVW_MEASURE` (default 'hep') for every measure, so a group whose only cells
    # are HEPC/ivw_shrunk would have no HEP map to weight from, and l2_heart raises
    # "no subject has a usable slope variance". It is added to `measures` only, never
    # to a cell: it is fitted so that the weights exist, and no result is reported for
    # it.
    for g in groups.values():
        if 'ivw_shrunk' in g['schemes'] and ivw_measure not in g['measures']:
            g['measures'].append(ivw_measure)
    return groups


def _window_axis(t_ax, f_ax, window, p):
    """(mask over t_ax, the window's own time axis) for 'full' or 'lit'.

    The surrogate must be built on the window the effect was found in. A 'lit' cluster
    came out of an L2 restricted to LIT_MS with its own CFT and its own null; comparing it
    against a null computed over the full axis would compare two different statistics.
    """
    if str(window) == 'full':
        return np.ones(t_ax.size, bool), t_ax
    lo, hi = lit_ms(p)
    m = (t_ax >= lo) & (t_ax <= hi)
    if int(m.sum()) < 2:
        raise ValueError(
            'window %r selects %d of %d point(s) on the TLIMS axis — the '
            'surrogate cannot be built on it.' % (window, int(m.sum()), t_ax.size))
    return m, t_ax[m]


def run_surrogate_multi(data_dir, p, cells, n_surrogate=50, limit=None,
                        skip_assert=False, verbose=True, allow_sham=False,
                        max_unshiftable=0.0, stat='cluster', tfce_nperm=None):
    """The shifted-R-peak control for many cells from one set of cohort passes.

    `cells` is [{grain, measure, model, scheme[, window]}, ...]. Returns one result dict
    per cell, in the order given, and per statistic when `stat` is 'both': every cell's
    cluster result, then every cell's tfce result (each dict carries 'stat').

    stat : 'cluster' (default), 'tfce' or 'both'. See STATS.
    tfce_nperm : permutations of the real TFCE L2 that gates a tfce cell
        (default p['nPerm'], as the cluster gate). The draws need none.

    The TFCE gate is the TFCE L2's own. A tfce cell is controlled only if the real
    weighted TFCE permutation test (l2_tfce.tfce_perm_weighted_multi, l2_heart's SEED,
    tfce_nperm) has a cell with p < ALPHA, the analogue of the cluster gate, never
    borrowed from it. Under 'both' a cell is drawn if either gate passes, and each
    statistic is reported as tested or not by its own gate.

    One cohort pass per draw, not per cell. Each draw rebuilds the cohort once and then
    answers every gated cell from it. The shifts are drawn once per draw and shared,
    which also makes the nulls of two cells of one run comparable: they rest on the same
    shifted peak trains rather than on two independent sets of draws.

    The gate is per cell. A cell whose real L2 found no surviving cluster is returned
    with tested=False and a reason, and costs no cohort work. If no cell is gated in, no
    draw is taken at all.
    """
    stats = _parse_stat(stat)
    rigged, why = sham_selected(p)
    if rigged and not allow_sham:
        raise ValueError(
            'REFUSING to run the surrogate control on a sham-selected run.\n  '
            + why + '\n  Pass allow_sham=True (or --allow-sham) only if you '
            'intend to report a FAILURE, which is still decisive; a PASS on '
            'this configuration is not evidence.')
    if not cells:
        raise ValueError('no cells given: the surrogate control is defined per '
                         '(grain, measure, model, scheme) and there is nothing '
                         'to test.')
    if verbose:
        print(f'surrogate control: {why}')
        print(f'  {len(cells)} cell(s) requested')

    fs = float(p.get('FS', 250.0))
    guard = guard_from_windows(p['HRSP_EPOCH_WIN'] if 'HRSP_EPOCH_WIN' in p
                               else p['HEP_WIN'], p['HEP_WIN'], fs)
    if verbose:
        print(f'  epoch guard {guard} samples at fs={fs:g}')

    # ── the real cohort, and the selection it made ──────────────────────────
    t0 = time.time()
    p_real = dict(p)
    p_real.pop('SURROGATE_SHIFT', None)
    # KEEP_PERBEAT must follow the requested cells. `session_measures` keeps the
    # per-beat epochs only when it is set, and they cannot be recovered from the
    # segment averages afterwards (`_beats_of_session` raises KeyError: "this
    # session dict has no `perbeat`"). The cell list decides which grains are needed.
    if any(str(c.get('grain')) == 'perbeat' for c in (cells or [])):
        p_real['KEEP_PERBEAT'] = True
        p = dict(p, KEEP_PERBEAT=True)
        if verbose:
            print('  KEEP_PERBEAT set: a per-beat cell was requested')
    sbs, _srec, _subrec = build_cohort(data_dir, p_real, limit=limit,
                                       verbose=False)
    presel = capture_selection(sbs)
    # A cohort that kept nothing is a configuration error, not an empty null.
    # Use the TLIMS-trimmed axis, not the full HEP_WIN one: `session_measures`
    # returns the full window (-300..700 ms, 251 points) and `subject_slopes` fits
    # over TLIMS only (176 points), and the L2 refuses an untrimmed axis.
    t_ax = f_ax = None
    for sess in sbs.values():
        if sess:
            _tfull = np.asarray(sess[0]['times'], float)
            t_ax = _tfull[tlims_mask(_tfull, p['TLIMS'])]
            f_ax = np.asarray(sess[0]['freqs'], float)
            break
    if t_ax is None:
        raise RuntimeError(
            'the REAL cohort kept no session, so there is nothing to control. '
            'Check the gates above.')

    # ── the real statistic and the gate, per cell, from the one cohort ──────
    ivw_measure = str(p.get('L2_IVW_MEASURE', 'hep'))
    groups = _group_cells(cells, ivw_measure)
    # keyed (cell key, stat) from here on; survived[k] = any stat's gate passed
    real_stat, gate, survived, tfce_real = {}, {}, {}, {}
    n_perm_tfce = int(p['nPerm'] if tfce_nperm is None else tfce_nperm)
    for (grain, model, window), g in groups.items():
        w_mask, w_ax = _window_axis(t_ax, f_ax, window, p)
        g['w_mask'], g['w_ax'] = w_mask, w_ax
        subs_full = cohort_L1(sbs, p_real, model, grain, g['measures'])
        subs_real = (subs_full if window == 'full'
                     else _restrict_time(subs_full, g['measures'], w_mask))
        g['subs_real'] = subs_full
        if 'cluster' in stats:
            rows_real, _det = run_l2_heart(
                subs_real, times_hep=w_ax, times_tf=w_ax, freqs=f_ax,
                measures=g['measures'], schemes=g['schemes'],
                n_perm=int(p['nPerm']), alpha=float(p['ALPHA']),
                max_gb=float(p['L2_MAX_GB']), out_csv=None, verbose=False)
            for c in g['cells']:
                k = _cell_key(c)
                meas, sch = k[1], k[3]
                real_stat[(k, 'cluster')] = _stat_of(rows_real, meas, sch)
                gate[(k, 'cluster')] = any(
                    r.get('Survived') for r in rows_real
                    if str(r.get('Measure')) == meas
                    and str(r.get('Scheme')) == sch)
        if 'tfce' in stats:
            tg = _tfce_group(subs_real, g, ivw_measure, keep_inputs=True)
            for c in g['cells']:
                k = _cell_key(c)
                r = tg[k]
                if r is None:
                    real_stat[(k, 'tfce')] = float('nan')
                    gate[(k, 'tfce')] = False
                    tfce_real[k] = None
                    continue
                # the real TFCE L2: the gate, and a runtime check that the
                # statistic here is l2_tfce's observed TFCE on the same map
                _t_o, res_l2, _n = tfce_perm_weighted_multi(
                    r['X'], r['w'], n_perm=n_perm_tfce, seed=L2_SEED,
                    E=TFCE_E, H=TFCE_H, dh=TFCE_DH,
                    conventions=(TFCE_CONVENTION,),
                    max_gb=float(p['L2_MAX_GB']))
                rl = res_l2[TFCE_CONVENTION]
                d = float(np.max(np.abs(np.asarray(rl['tfce_obs'])
                                        - r['tfce'])))
                if not d <= 1e-9 * max(1.0, r['stat']):
                    raise AssertionError(
                        f'TFCE statistic of {k} differs from l2_tfce\'s '
                        f'observed TFCE on the same map (max |diff| {d:.3g})')
                r['l2_p_map'] = np.asarray(rl['p_map'], float)
                r['l2_min_p'] = float(rl['p_map'].min())
                r['w_ax'] = w_ax
                r.pop('X', None)
                tfce_real[k] = r
                real_stat[(k, 'tfce')] = r['stat']
                gate[(k, 'tfce')] = r['l2_min_p'] < float(p['ALPHA'])
    for c in cells:
        k = _cell_key(c)
        survived[k] = any(gate[(k, st)] for st in stats)
    if verbose:
        print(f'  REAL cohort + L2 done in {time.time() - t0:.0f}s '
              f'({len(sbs)} subject(s))')
        for st in stats:
            for c in cells:
                k = _cell_key(c)
                if st == 'cluster':
                    msg = ('SURVIVED' if gate[(k, st)]
                           else 'no cluster — GATED OUT')
                else:
                    r = tfce_real[k]
                    msg = ('not testable (no maps / too few subjects / bad '
                           'weights) — GATED OUT' if r is None else
                           'TFCE L2 min p %.4g at nPerm %d: %s'
                           % (r['l2_min_p'], n_perm_tfce,
                              'SIGNIFICANT' if gate[(k, st)] else 'GATED OUT'))
                print('    %-8s %-5s %-13s %-15s %-5s %sstat=%-10.4g %s'
                      % (k[0], k[1], k[2], k[3], k[4],
                         '' if stats == ('cluster',) else st + ' ',
                         real_stat[(k, st)], msg))

    gated = [c for c in cells if survived[_cell_key(c)]]
    results = {}
    for st in stats:
        for c in cells:
            k = _cell_key(c)
            if not gate[(k, st)]:
                # The gate: a null is built only where the real L2 found a
                # surviving cluster; an ungated cell gets p_surr = NaN and
                # tested = False. Running it anyway would produce a p for a null
                # result. TFCE: the TFCE L2 had no cell with p < ALPHA.
                results[(k, st)] = {
                    'tested': False,
                    'reason': ('the real L2 found no surviving cluster for '
                               'this cell, so there is nothing to control'
                               if st == 'cluster' else
                               'the real TFCE L2 found no cell with p < ALPHA '
                               '(or the cell could not be tested), so there '
                               'is nothing to control'),
                    'real_stat': real_stat[(k, st)], 'p_surr': float('nan'),
                    'n_valid': 0,
                    'null': [], 'measure': k[1], 'grain': k[0], 'model': k[2],
                    'scheme': k[3], 'window': k[4], 'sham_selected': rigged,
                    'stat': st}
                if st == 'tfce' and tfce_real.get(k) is not None:
                    results[(k, st)]['tfce_l2_min_p'] = tfce_real[k]['l2_min_p']

    def _ordered():
        return [results[(_cell_key(c), st)] for st in stats for c in cells]

    if not gated:
        if verbose:
            print('  nothing survived the real L2 — no draw is taken.')
        return _ordered()

    gated_groups = _group_cells(gated)
    # Carry the window axis over: `_group_cells` rebuilds the groups from the gated
    # cells alone and knows nothing about the time axis, so the per-window mask
    # computed for the real L2 is copied across.
    for _k, _g in gated_groups.items():
        _g['w_mask'] = groups[_k]['w_mask']
        _g['w_ax'] = groups[_k]['w_ax']

    # ── integrity check: shift 0 must reproduce the real fit ────────────────
    # One cohort pass, checked against every gated (grain, model, measure). The
    # property is cohort-level, but the comparison is per L1 fit, and a fit that the
    # pinned path did not reproduce would make its null equal the data and appear to
    # confirm the effect.
    if not skip_assert:
        p0 = dict(p, SURROGATE_SHIFT=0, PIN_SELECTION=True, IC_PRESELECT=presel)
        sbs0, _, _ = build_cohort(data_dir, p0, limit=limit, verbose=False)
        worst = 0.0
        for (grain, model, window), g in gated_groups.items():
            subs0 = cohort_L1(sbs0, p0, model, grain, g['measures'])
            # Compare the untrimmed fits: the check is about the pinned path
            # reproducing the L1, which is window-independent; trimming both sides
            # first would only hide a mismatch outside the window.
            base = groups[(grain, model, window)]['subs_real']
            for meas in g['measures']:
                worst = max(worst, _max_beta_diff(base, subs0, meas))
        if not (worst <= SURR_ASSERT_TOL):
            raise AssertionError(
                f'SURROGATE INTEGRITY ASSERT FAILED: a shift of ZERO through '
                f'the surrogate path did not reproduce the real L1 (max |diff| '
                f'{worst:.3g} > {SURR_ASSERT_TOL:g}). Something in the pinned '
                f'path is not the analysis it claims to reproduce — do NOT '
                f'trust any surrogate null from this configuration.')
        if verbose:
            print(f'  integrity assert PASSED across {len(gated_groups)} '
                  f'(grain, model) fit(s) (max |diff| {worst:.2g} <= '
                  f'{SURR_ASSERT_TOL:g})')

    # ── the null ────────────────────────────────────────────────────────────
    # Every statistic of every gated cell is drawn, whichever gate let the cell in,
    # so under 'both' the two nulls rest on the same accepted draws; a statistic whose
    # own gate failed stays 'not tested' below.
    nulls = {(_cell_key(c), st): [] for c in gated for st in stats}
    n_lost_draws = 0
    n_dropped = []          # sessions excluded per accepted draw, for the CSV
    for k_draw in range(1, int(n_surrogate) + 1):
        shifts, drop, ok, n_bad = _draw_shifts(sbs, guard, fs, p, k_draw,
                                               max_unshiftable=max_unshiftable)
        if not ok:
            n_lost_draws += 1
            if verbose and k_draw == 1:
                print(f'  draw {k_draw}: REJECTED — {n_bad} session(s) admit no '
                      f'shift (limit {max_unshiftable:.0%}). The bound is the '
                      f'SEGMENT, not the recording; see _draw_shifts.')
            continue
        # One rigid shift per session; the pipeline applies it to every segment.
        # SURROGATE_DROP carries the sessions this draw could not shift, so they are
        # excluded rather than analysed at shift 0 with their real peaks.
        pk = dict(p, PIN_SELECTION=True, IC_PRESELECT=presel,
                  SURROGATE_SHIFT=dict(shifts), SURROGATE_DROP=set(drop))
        td = time.time()
        try:
            sbs_k, _, _ = build_cohort(data_dir, pk, limit=limit, verbose=False)
        except Exception as exc:                            # noqa: BLE001
            n_lost_draws += 1
            if verbose:
                print(f'  draw {k_draw}: cohort FAILED ({type(exc).__name__}: '
                      f'{exc}) — the draw is dropped, not counted as a null')
            continue
        got = {}
        for (grain, model, window), g in gated_groups.items():
            subs_kf = cohort_L1(sbs_k, pk, model, grain, g['measures'])
            subs_k = (subs_kf if window == 'full'
                      else _restrict_time(subs_kf, g['measures'], g['w_mask']))
            if 'cluster' in stats:
                rows_k, _d = run_l2_heart(
                    subs_k, times_hep=g['w_ax'], times_tf=g['w_ax'],
                    freqs=f_ax, measures=g['measures'], schemes=g['schemes'],
                    n_perm=int(pk['nPerm']), alpha=float(pk['ALPHA']),
                    max_gb=float(pk['L2_MAX_GB']), out_csv=None,
                    verbose=False)
                for c in g['cells']:
                    k = _cell_key(c)
                    got[(k, 'cluster')] = _stat_of(rows_k, k[1], k[3])
            if 'tfce' in stats:
                # same subjects, keep rule and weights as the real statistic,
                # rebuilt on this draw's L1 maps; no permutations
                tg = _tfce_group(subs_k, g, ivw_measure)
                for c in g['cells']:
                    k = _cell_key(c)
                    if tg[k] is not None:
                        got[(k, 'tfce')] = tg[k]['stat']
        # All or nothing per draw. A draw that produced a statistic for some cells
        # and not others would give those cells nulls of different N, built on
        # different subsets of draws. `_stat_of` returns 0.0 for "no cluster", a real
        # value, so a missing key here means the fit itself failed. A TFCE cell that
        # could not be built on this draw (too few subjects, bad weights) is missing
        # too, and drops the draw for every cell and statistic.
        if len(got) != len(nulls):
            n_lost_draws += 1
            if verbose:
                print(f'  draw {k_draw}: incomplete ({len(got)}/{len(nulls)} '
                      f'cells) — dropped so every cell keeps the same draws')
            continue
        for k, v in got.items():
            nulls[k].append(float(v))
        n_dropped.append(int(n_bad))
        if verbose:
            print('  surrogate %3d/%d: %s   (%.0fs)'
                  % (k_draw, n_surrogate,
                     '  '.join('%s/%s%s=%.4g'
                               % (kk[0][1], kk[0][3],
                                  '' if stats == ('cluster',) else ':' + kk[1],
                                  v)
                               for kk, v in sorted(got.items())),
                     time.time() - td))

    for st, c in [(st, c) for st in stats for c in gated]:
        k = _cell_key(c)
        if not gate[(k, st)]:
            continue            # drawn for the other stat; stays 'not tested'
        null = nulls[(k, st)]
        n_valid = len(null)
        if n_valid == 0:
            # Not an empty null but a failure: p_surr from zero draws would be 1.0,
            # which would read as a clear failure of the control.
            results[(k, st)] = {
                'tested': False,
                'reason': 'no surrogate draw produced a usable null — every '
                          'draw lost beats or failed. Widen SURR_SHIFT_MS, '
                          'raise --max-unshiftable, or check the guard.',
                'real_stat': real_stat[(k, st)], 'p_surr': float('nan'),
                'n_valid': 0,
                'n_failed_draws': n_lost_draws, 'null': [], 'measure': k[1],
                'grain': k[0], 'model': k[2], 'scheme': k[3], 'window': k[4],
                'sham_selected': rigged, 'stat': st}
            continue
        rs = real_stat[(k, st)]
        results[(k, st)] = {
            'tested': True, 'real_stat': rs,
            'p_surr': (1.0 + sum(1 for v in null if v >= rs)) / (1.0 + n_valid),
            'n_valid': n_valid, 'n_failed_draws': n_lost_draws,
            'null': [float(v) for v in null], 'measure': k[1], 'grain': k[0],
            'model': k[2], 'scheme': k[3], 'window': k[4],
            'sham_selected': rigged,
            'p_floor': 1.0 / (1.0 + n_valid),
            # How many sessions each accepted draw had to exclude. A null built on
            # a smaller cohort than the real fit is not quite a null for that fit;
            # this makes the size of that gap readable instead of absorbed into
            # p_surr.
            'n_sessions_dropped_max': (max(n_dropped) if n_dropped else 0),
            'n_sessions_dropped_mean': (float(np.mean(n_dropped))
                                        if n_dropped else 0.0),
            'stat': st}
        if st == 'tfce':
            # The per-cell surrogate p-map. The null is each draw's max |TFCE|
            # over the map (FWE over cells, as the TFCE L2's own null), so a cell's
            # p is (1 + #{draw max >= |TFCE_real(cell)|}) / (1 + N_valid) and shares
            # the 1/(1+N) floor. `n_cells_beat_all` counts the cells at that floor:
            # |TFCE_real| above every draw.
            r = tfce_real[k]
            srt = np.sort(np.asarray(null, float))
            a = np.abs(r['tfce']).ravel()
            n_ge = srt.size - np.searchsorted(srt, a, side='left')
            pmap = ((1.0 + n_ge) / (1.0 + n_valid)).reshape(r['tfce'].shape)
            results[(k, st)].update(
                tfce_l2_min_p=r['l2_min_p'],
                n_cells_beat_all=int(np.sum(n_ge == 0)),
                tfce_maps=dict(t_real=r['t'], tfce_real=r['tfce'],
                               p_surr_map=pmap, tfce_l2_p_map=r['l2_p_map'],
                               null_max=np.asarray(null, float),
                               times=np.asarray(r['w_ax'], float),
                               n_used=r['n_used']))
    return _ordered()


def _max_beta_diff(a, b, measure):
    """Largest |difference| between two L1 subject lists, for the assert."""
    ba = {r['subject']: r.get(measure) for r in a}
    bb = {r['subject']: r.get(measure) for r in b}
    worst = 0.0
    for sid in set(ba) | set(bb):
        x, y = ba.get(sid), bb.get(sid)
        x = None if (x is None or x[0] is None) else x
        y = None if (y is None or y[0] is None) else y
        if x is None and y is None:
            continue
        if x is None or y is None:
            return float('inf')        # a subject present in one fit only
        d = np.abs(np.asarray(x[0], float) - np.asarray(y[0], float))
        d = d[np.isfinite(d)]
        if d.size:
            worst = max(worst, float(d.max()))
    return worst


def _draw_shifts(sbs, guard, fs, p, k, max_unshiftable=0.0):
    """Per-session rigid shifts -> (shifts, drop, ok, n_unshiftable).

    The bound is the analysed segment, not the recording. With a 45-60 s analysed
    window, a session whose first or last beat sits within |shift| + guard of an edge
    admits no shift in the permitted band (about 1 session in 8 on this cohort).

    `max_unshiftable` is the fraction of sessions allowed to fail before the whole draw
    is rejected (0: any failure rejects the draw). A session that fails is excluded from
    that draw (returned in `drop`), and the count is returned so the caller can report
    it rather than discover a changed cohort from the p.
    """
    shifts, drop = {}, set()
    sess_all = [(j, sid, s) for j, (sid, ss) in enumerate(sorted(sbs.items()))
                for s in ss]
    for j, sid, s in sess_all:
        key = (str(sid), int(s['session']))
        rp = np.asarray(s.get('peaks_rel', []), np.int64)
        n_pts = int(s.get('seg_len_samp', 0))
        if rp.size < 2 or n_pts <= 0:
            drop.add(key)
            continue
        res = make_surrogate_rpeaks_retry(
            rp, n_pts=n_pts, shift_ms=SURR_SHIFT_MS, fs=fs, guard=guard,
            k=k, j=j + 1)
        if not getattr(res, 'ok', False):
            drop.add(key)
            continue
        shifts[key] = int(res.shift)
    n_bad = len(drop)
    frac = n_bad / max(1, len(sess_all))
    # The unshiftable sessions are returned so they can be dropped. Leaving them out
    # of `shifts` alone would give them shift 0 (`SURROGATE_SHIFT.get(key, 0)`), i.e.
    # analyse them with their real peaks inside the null, which pulls the null toward
    # the real statistic. session_measures refuses a session that is in neither set.
    return shifts, drop, (frac <= max_unshiftable and bool(shifts)), n_bad


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT (called by run_component_group_tests.py with an argv list)
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data', default=os.environ.get('HMD_DATA', os.path.join(os.path.dirname(_HERE), 'data', 'segments')))
    ap.add_argument('--results-root',
                    default=os.path.join(os.path.dirname(_HERE), 'results'))
    ap.add_argument('--params', default=None, help='JSON overrides')
    ap.add_argument('--n-surrogate', type=int, default=50)
    ap.add_argument('--measure', default='hrsp')
    ap.add_argument('--grain', default='segment')
    ap.add_argument('--model', default='unadj')
    ap.add_argument('--scheme', default='unweighted')
    # Many cells from one set of cohort passes: with --cells every gated cell of a
    # run shares the same cohort passes and the same shift draws, so the nulls are
    # comparable and the cost does not grow with the number of cells.
    ap.add_argument('--cells', default=None,
                    help='JSON list of {grain, measure, model, scheme} — one '
                         'control per entry, all sharing the same cohort passes '
                         'and the same shift draws. Overrides '
                         '--measure/--grain/--model/--scheme.')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--allow-sham', action='store_true')
    ap.add_argument('--skip-assert', action='store_true')
    ap.add_argument('--out', default=None, help='CSV to write')
    # With the default 0, any unshiftable session rejects the draw. On this route
    # the bound is a 45-60 s analysed window and about 1 session in 8 admits no
    # shift, so 0 rejects every draw; allowing a small fraction to be excluded from
    # a draw is the alternative, and the count is printed for every draw.
    ap.add_argument('--max-unshiftable', type=float, default=0.0,
                    help='fraction of sessions allowed to admit no shift before '
                         'the whole draw is rejected')
    # --stat: 'cluster' (default) writes the summary and nulls CSVs; 'tfce' / 'both'
    # add a Stat column (and the TFCE extras) to both CSVs and write
    # <out>_tfce_maps.npz with the per-cell p-maps.
    ap.add_argument('--stat', choices=('cluster', 'tfce', 'both'),
                    default='cluster',
                    help='statistic the null is built on: strongest cluster '
                         '|SumAbsT| (cluster), max |TFCE| of the observed '
                         'weighted t-map (tfce), or both from the same draws')
    ap.add_argument('--tfce-nperm', type=int, default=None,
                    help='permutations of the REAL TFCE L2 that gates a tfce '
                         'cell (default: nPerm from the params)')
    a = ap.parse_args(argv)

    p = dict(DEFAULT_RUN_PARAMS)
    if a.params:
        p.update(json.loads(a.params) if a.params.strip().startswith('{')
                 else json.load(open(a.params, encoding='utf-8')))
    if a.cells:
        cells = json.loads(a.cells)
        if not isinstance(cells, list) or not cells:
            ap.error('--cells must be a non-empty JSON list of '
                     '{grain, measure, model, scheme}')
        need = {'grain', 'measure', 'model', 'scheme'}
        for c in cells:
            if not isinstance(c, dict) or not need <= set(c):
                ap.error('each --cells entry needs %s; got %r'
                         % (sorted(need), c))
            # Default 'full'. A lit cell must say so, because its surrogate is
            # built on the lit axis with the lit CFT.
            c.setdefault('window', 'full')
    else:
        cells = [{'grain': a.grain, 'measure': a.measure, 'model': a.model,
                  'scheme': a.scheme}]

    out = run_surrogate_multi(
        a.data, p, cells, n_surrogate=a.n_surrogate, limit=a.limit,
        skip_assert=a.skip_assert, allow_sham=a.allow_sham,
        max_unshiftable=a.max_unshiftable, stat=a.stat,
        tfce_nperm=a.tfce_nperm)
    stats = _parse_stat(a.stat)

    _run_root, _, _, _ = run_paths(a.results_root, p, create=False)
    _run_tag = os.path.basename(os.path.normpath(_run_root))

    print(os.linesep + '=' * 70)
    for res in out:
        print('  %s / %s / %s / %s / %s window%s'
              % (res['grain'], res['measure'], res['model'], res['scheme'],
                 res.get('window', 'full'),
                 '' if stats == ('cluster',) else '  [%s]' % res['stat']))
        _report(res)
        print('-' * 70)

    # One CSV per run, inside the run folder, so the control is attached to the
    # run it belongs to by location; the `Run` column still names it, so a copied
    # file stays identifiable.
    out_csv = a.out or (os.path.join(_run_root, 'surrogate_control.csv')
                        if os.path.isdir(_run_root) else None)
    if out_csv:
        write_outputs(out, out_csv, _run_tag, stats)
    return 0


# the TFCE extras appended to the summary CSV under --stat tfce/both
TFCE_EXTRA_COLS = ['Stat', 'TFCE_L2_min_p', 'N_cells_beat_all',
                   'TFCE_convention', 'TFCE_E', 'TFCE_H', 'TFCE_dh']


def write_outputs(out, out_csv, run_tag, stats=('cluster',)):
    """The summary CSV, the per-draw nulls CSV and (tfce) the p-map npz.

    Under stats == ('cluster',) there is no Stat column, nothing appended and no npz.
    Any other set of statistics appends TFCE_EXTRA_COLS to the summary (blank TFCE
    fields on a cluster row) and a Stat column to the nulls, so a cluster row and a
    tfce row of the same cell can sit in one file.
    """
    import csv as _csv
    extra = tuple(stats) != ('cluster',)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = _csv.writer(fh)
        # The ratios and the verdict are columns, not something a reader derives.
        # At N = 20 the smallest attainable p_surr is .0476, so a bare
        # `p_surr = 0.0476` reads as "just under .05" when it means that none of 20
        # nulls reached the real statistic, the strongest outcome this N can
        # express. `N_null_ge_real` and the two ratios make that explicit.
        w.writerow(['Run', 'Measure', 'Grain', 'Model', 'Scheme', 'Tested',
                    'Window',
                    'Real_stat', 'p_surr', 'N_valid', 'p_floor',
                    'Null_median', 'Null_max',
                    'N_null_ge_real', 'Real_over_null_median',
                    'Real_over_null_max', 'At_floor', 'Verdict',
                    'N_failed_draws',
                    'N_sessions_dropped_max', 'N_sessions_dropped_mean',
                    'Sham_selected', 'Reason']
                   + (TFCE_EXTRA_COLS if extra else []))
        for res in out:
            nl = res.get('null') or []
            _med = float(np.median(nl)) if nl else None
            _mx = float(max(nl)) if nl else None
            _rs = res.get('real_stat')
            _ge = (sum(1 for v in nl if v >= _rs) if nl and _rs is not None
                   else '')
            _floor = res.get('p_floor')
            _atf = (int(abs(res['p_surr'] - _floor) < 1e-9)
                    if (_floor and res.get('p_surr') == res.get('p_surr'))
                    else '')
            # An untested cell gets no verdict: p_surr is NaN there and any
            # comparison with .05 is False, which would print as a failure of a
            # control that never ran.
            if not res.get('tested') or res.get('p_surr') != res.get('p_surr'):
                _verdict = 'not tested'
            elif res['p_surr'] >= 0.05:
                _verdict = 'FAIL: not distinguishable from ongoing activity'
            elif _atf:
                _verdict = ('PASS at the N=%d floor: no surrogate reached '
                            'the real statistic' % res['n_valid'])
            else:
                _verdict = 'PASS'
            _st = res.get('stat', 'cluster')
            _xt = ([] if not extra else
                   [_st, res.get('tfce_l2_min_p', ''),
                    res.get('n_cells_beat_all', '')]
                   + ([TFCE_CONVENTION, TFCE_E, TFCE_H, TFCE_DH]
                      if _st == 'tfce' else ['', '', '', '']))
            w.writerow([
                run_tag, res['measure'], res['grain'], res['model'],
                res['scheme'], int(res['tested']),
                res.get('window', 'full'), res['real_stat'],
                res['p_surr'], res['n_valid'], res.get('p_floor', ''),
                ('' if _med is None else _med),
                ('' if _mx is None else _mx),
                _ge,
                ('' if not _med else round(_rs / _med, 3)),
                ('' if not _mx else round(_rs / _mx, 3)),
                _atf, _verdict,
                res.get('n_failed_draws', ''),
                res.get('n_sessions_dropped_max', ''),
                res.get('n_sessions_dropped_mean', ''),
                int(res['sham_selected']), res.get('reason', '')] + _xt)
    print('wrote', out_csv)
    # ── the per-draw nulls, as their own file ──────────────────────────
    # `res['null']` holds every draw's statistic, but the summary CSV keeps
    # only median/max/count, so the null distribution is written here.
    # Row order is draw order. Rejected draws are skipped, so `Draw_index`
    # counts accepted draws and is not the shift seed.
    nulls_csv = os.path.splitext(out_csv)[0] + '_nulls.csv'
    with open(nulls_csv, 'w', newline='', encoding='utf-8') as fh:
        w = _csv.writer(fh)
        w.writerow(['Run', 'Measure', 'Grain', 'Model', 'Scheme', 'Window',
                    'Real_stat', 'Draw_index', 'Null_stat']
                   + (['Stat'] if extra else []))
        for res in out:
            for i, v in enumerate(res.get('null') or [], start=1):
                w.writerow([run_tag, res['measure'], res['grain'],
                            res['model'], res['scheme'],
                            res.get('window', 'full'), res['real_stat'],
                            i, float(v)]
                           + ([res.get('stat', 'cluster')] if extra else []))
    print('wrote', nulls_csv)
    # ── the per-cell TFCE p-maps ───────────────────────────────────────
    # One key block per tested tfce cell, tag = grain__measure__model__
    # scheme__window. p_surr_map is the per-cell surrogate p (see
    # run_surrogate_multi); tfce_l2_p_map is the real TFCE L2's own FWE p
    # (the gate); maps are [nF x nT] ([1 x nT] for HEP) on `times__<tag>`.
    tf = [r for r in out if r.get('stat') == 'tfce' and r.get('tfce_maps')]
    if tf:
        payload, index = {}, []
        for r in tf:
            tag = '__'.join(str(r[x]) for x in ('grain', 'measure', 'model',
                                                'scheme', 'window'))
            for kk, vv in r['tfce_maps'].items():
                payload[f'{kk}__{tag}'] = np.asarray(vv)
            index.append(dict(tag=tag, real_stat=r['real_stat'],
                              p_surr=r['p_surr'], n_valid=r['n_valid'],
                              n_cells_beat_all=r['n_cells_beat_all'],
                              tfce_l2_min_p=r['tfce_l2_min_p']))
        payload['index_json'] = np.array(json.dumps(dict(
            run=run_tag, cells=index, convention=TFCE_CONVENTION,
            E=TFCE_E, H=TFCE_H, dh=TFCE_DH,
            p_surr_map='(1 + #{draw max|TFCE| >= |TFCE_real(cell)|}) / '
                       '(1 + N_valid)')))
        npz_path = os.path.splitext(out_csv)[0] + '_tfce_maps.npz'
        np.savez_compressed(npz_path, **payload)
        print('wrote', npz_path)


def _report(res):
    """Print one cell's verdict.

    It never prints a verdict for an untested cell: `p_surr` is NaN there and any
    comparison against .05 is False, which would read as a failure of the control
    rather than as a cell that was never controlled.
    """
    if not res['tested']:
        print('    NOT TESTED — ' + res.get('reason', ''))
        return
    nl = res['null']
    med = float(np.median(nl))
    print('    real statistic   %.5g' % res['real_stat'])
    print('    surrogate null   n=%d  median=%.5g  max=%.5g'
          % (res['n_valid'], med, max(nl)))
    print('    p_surr           %.4f   (floor %.4f at N=%d)'
          % (res['p_surr'], res['p_floor'], res['n_valid']))
    if res.get('stat') == 'tfce':
        print('    TFCE L2 min p    %.4g;  %d cell(s) of the real map beat EVERY '
              'draw (per-cell p at the floor)'
              % (res['tfce_l2_min_p'], res['n_cells_beat_all']))
    _dmax = int(res.get('n_sessions_dropped_max', 0) or 0)
    if _dmax:
        # A null built on a smaller cohort is not quite a null for the real fit.
        print('    [!] up to %d session(s) were EXCLUDED from a draw (mean %.1f) '
              'because they admitted'
              % (_dmax, float(res.get('n_sessions_dropped_mean', 0) or 0)))
        print('       no shift in the permitted band. The null therefore rests '
              'on a slightly smaller')
        print('       cohort than the real statistic — read the separation with '
              'that in mind.')
    if res['p_surr'] < 0.05:
        print('    HEARTBEAT-LOCKED at this N — real is %s the median null'
              % ('%.2gx' % (res['real_stat'] / med) if med else 'above'))
        # At the floor the separation is the result, not the p. With N = 20 the
        # smallest attainable p_surr is .0476, so "p < .05" says only that no
        # surrogate reached the real statistic in 20 draws.
        if abs(res['p_surr'] - res['p_floor']) < 1e-9:
            print('    [!] p_surr IS THE N=%d FLOOR. Quote the SEPARATION, not '
                  'the p:' % res['n_valid'])
            print('       "no surrogate reached it in %d draws" is the '
                  'strongest statement this N permits.' % res['n_valid'])
    else:
        print('    NOT distinguishable from ongoing activity — %d of %d '
              'shifted nulls'
              % (sum(1 for v in nl if v >= res['real_stat']), len(nl)))
        print('       reached the real statistic. A FAILURE is decisive at '
              'any N.')
    if res['sham_selected']:
        print('    [!] SHAM-SELECTED RUN: a PASS here is not evidence; only a '
              'FAILURE would be.')
