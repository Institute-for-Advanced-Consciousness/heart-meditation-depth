# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Outcome, demographic and moderation associations on a finished heartbeat-locked run.

`main(argv)` takes the options below. run_associations.py calls it with `--run <primary run
folder> --beh <questionnaire tables> --tfce` and the primary model, grain and scheme.

    --run  <path to one run folder>
    --root <results root>          every run under it
    --models covariates
    --dry-run                      say what it would do
    --tfce                         on the TFCE-significant cells

Works from the run's `run_results.npz`, which holds the per-subject L1 maps
(`L1_<model>_<measure>_beta`, [nSub x nFreq x nTime]) of every model, so L1 is not refitted:
  * outcome associations          (Section A)
  * demographic associations      (Section B)
  * moderation by demographics    (Section C)

The surrogate R-peak control cannot be done here: it refits L1 on shifted peak trains, so it
must re-read the recordings. heart_surrogate runs it separately.

CLUSTER MASK. When the run persisted a cluster's own cells, those are averaged (`Mask=cells`).
Otherwise the mask is the bounding box of the cluster's Start/End in ms and Hz (`Mask=bbox`),
which on a ragged or L-shaped cluster includes cells the test did not select. Every row written
says which was used.

THE MODEL SWITCH
    --models cascade      (default)
        The first of main/full, main/lit, covariates/full, covariates/lit with a cluster that
        survived and clears the QRS edge (see `cascade`); only that level gets the battery.
    --models main
        `session` and `unadj`, but only where that model survived. Both significant -> both are
        analysed. One -> only that one. Neither -> the measure is skipped, and this is reported.
    --models covariates
        every covariate model that survived (reportmethod, rr, respAmp, respPhase, ecgAmp,
        ecgAmpT, cond, ...), same rule.
    --models both
        main, then covariates.

A post-hoc never runs on a non-surviving cell: it runs only if the effect survived, and only on
that effect.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Write UTF-8 to the console, so a non-ASCII character cannot crash a cp1252 terminal.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

from outcomes import (                                       # noqa: E402
    load_outcomes, load_demographics, spearman_ols_table, demographics_table,
    moderation_table,
    OUTCOME_SPEC)

# `cascade` is the single definition of which level of a run is the result and of the +50 ms
# QRS edge that decides it. It is imported, not restated, so the outcome tables and the
# surrogate describe the same cells of a run.
from cascade import (                                    # noqa: E402
    MAIN_MODELS, QRS_EDGE_MS, cascade_of_run, level_name)
GRAINS = ('cell', 'segment', 'perbeat')
ALPHA = 0.05
DEFAULT_BEH = os.environ.get('HMD_BEHAVIOUR',
                             os.path.join(os.path.dirname(_HERE), 'data', 'bids', 'phenotype'))


# ═══════════════════════════════════════════════════════════════════════════
def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x


def split_l2_key(key):
    """`l2_rows_json` key -> (model, window). 'session@lit' -> ('session', 'lit').

    `l2_rows_json` is keyed `<model>` for the full window and `<model>@lit` for the literature
    window. The window must come off before the model is classified, or 'session@lit' would not
    match MAIN_MODELS and the primary model's literature-window result would be treated as a
    covariate model.
    """
    k = str(key)
    model, sep, window = k.partition('@')
    return model, (window if sep else 'full')


def parse_qrs_edge(v):
    """'off' / 'none' / 'null' / None -> -inf (no gate); anything else a float.

    0 is not "off". `row_qualifies` tests `Start_ms >= qrs_edge` and the tested window starts at
    -100 ms, so an edge of 0 still demotes every pre-R-peak cluster. Turning the cardiac-field
    gate off means -inf, and this gives it a spelling that can be typed (`--qrs-edge off`).
    """
    if v is None:
        return float('-inf')
    s = str(v).strip().lower()
    if s in ('off', 'none', 'null', 'no', 'false', '-inf', '-infinity'):
        return float('-inf')
    return float(s)


def _clears_qrs(row, edge=QRS_EDGE_MS):
    """True when the cluster starts at or after QRS_EDGE_MS.

    A missing Start_ms is not a pass: a row with no bounds cannot be shown to clear the window.
    """
    t0 = _f(row.get('Start_ms'))
    return bool(np.isfinite(t0) and t0 >= edge)


def surviving_rows(l2, models_mode, qrs_gate_covariates=True,
                   only_window=None, qrs_edge=QRS_EDGE_MS):
    """[(model, window, row)] for every surviving cluster, filtered by the switch.

    `l2_rows_json` is {model[@window]: [row, ...]}, one row per cluster; a row for a measure that
    produced nothing carries Status='no_supra' with blank bounds. Survival is the row's own
    `Survived` flag, or perm_p < alpha when the row has no such column.

    Both windows are returned, with the window as the second element rather than glued into the
    model name: they are two tests over two separate correction spaces (see L2_WINDOWS in
    heart_depth_run), so a caller must be able to tell them apart.
    """
    # `only_window`, not `window`: the loop below binds `window` from the key, and a parameter
    # of that name would be overwritten on the first iteration.
    out = []
    for key, rows in (l2 or {}).items():
        model, window = split_l2_key(key)
        if only_window is not None and window != only_window:
            continue
        if models_mode == 'main' and model not in MAIN_MODELS:
            continue
        if models_mode == 'covariates' and model in MAIN_MODELS:
            continue
        _is_cov = model not in MAIN_MODELS
        for r in (rows or []):
            if not isinstance(r, dict):
                continue
            surv = r.get('Survived')
            if surv is None:
                surv = _f(r.get('perm_p')) < ALPHA
            if not surv:
                continue
            if not np.isfinite(_f(r.get('Start_ms'))):
                continue
            # Covariate survivors must clear the QRS window. Main models are not gated here:
            # they are the primary result and are reviewed directly, so dropping one would
            # remove it from the record rather than caveat it. Every row written carries
            # `Clears_QRS` either way.
            if (qrs_gate_covariates and _is_cov
                    and not _clears_qrs(r, float(qrs_edge))):
                continue
            out.append((model, window, r))
    return out


def exact_cells(cells_by_key, l2key, measure, scheme, cluster):
    """The persisted cell list for one cluster, on the full time axis, or None.

    None means "this run cannot supply it" (no persisted cells, or that cluster is not in the
    index); it never means "empty".

    Column 1 is `t_index`, not `cells`. `load_l2_cells` returns both: `cells` holds the time
    index on the window's own axis, `t_index` the same cells mapped onto the run's full `times`
    array. `cluster_predictor` reads the full-axis L1 maps in run_results.npz, so it needs the
    full-axis index. The two coincide for the full window and differ for '@lit', where the
    window index would read a 200-600 ms cluster off the -100 ms end of the map.
    """
    if not cells_by_key:
        return None
    recs = (cells_by_key.get(l2key) or {}).get('%s|%s' % (measure, scheme))
    if not recs:
        return None
    try:
        want = int(float(cluster))
    except (TypeError, ValueError):
        return None
    for r in recs:
        if int(r.get('rank', -1)) == want:
            xy = np.asarray(r['cells'], int)
            ti = np.asarray(r.get('t_index', xy[:, 1] if xy.size else []), int)
            if not xy.size or ti.size != xy.shape[0]:
                return None
            return np.column_stack([xy[:, 0], ti])
    return None


def cluster_predictor(z, model, measure, row, cells=None):
    """(per-subject value over the cluster, n_cells, mask_kind) or (None, 0, '').

    HEP is [nSub x nTime] and has no frequency axis; HRSP/HEPC are [nSub x nFreq x nTime]. A
    row's Hz bounds are ignored for HEP.

    `cells`, the exact cluster, is used when given. Without it this averages the bounding box of
    Start/End_ms x Start/End_Hz, which on a ragged cluster includes cells the test never selected
    (in one case 1584 box cells against 1090 cluster cells). The returned `mask_kind` says which
    was used and goes in the `Mask` column.

    The cell indices must be on the run's full time axis: for a 'lit' cluster pass `t_index`
    (see `exact_cells`), not the window's own `cells`.
    """
    key = 'L1_%s_%s_beta' % (model, measure)
    if key not in z.files:
        return None, 0, ''
    B = z[key]
    if cells is not None:
        xy = np.asarray(cells, int)
        if xy.ndim == 2 and xy.size:
            n_cells = int(xy.shape[0])
            with np.errstate(invalid='ignore'):
                if B.ndim == 2:                       # HEP: subjects x time
                    vals = np.nanmean(B[:, xy[:, 1]], axis=1)
                else:
                    vals = np.nanmean(B[:, xy[:, 0], xy[:, 1]], axis=1)
            return vals, n_cells, 'cells'
    times = np.asarray(z['times'], float)
    t0, t1 = _f(row.get('Start_ms')), _f(row.get('End_ms'))
    if not (np.isfinite(t0) and np.isfinite(t1)):
        return None, 0, ''
    tsel = (times >= t0) & (times <= t1)
    if not tsel.any():
        return None, 0, ''
    if B.ndim == 2:                                   # HEP: subjects x time
        sub = B[:, tsel]
        n_cells = int(tsel.sum())
    else:                                             # subjects x freq x time
        freqs = np.asarray(z['freqs'], float)
        f0, f1 = _f(row.get('Start_Hz')), _f(row.get('End_Hz'))
        fsel = ((freqs >= f0) & (freqs <= f1)
                if (np.isfinite(f0) and np.isfinite(f1))
                else np.ones(freqs.size, bool))
        if not fsel.any():
            return None, 0, ''
        sub = B[:, fsel][:, :, tsel]
        n_cells = int(fsel.sum() * tsel.sum())
    with np.errstate(invalid='ignore'):
        vals = np.nanmean(sub.reshape(sub.shape[0], -1), axis=1)
    return vals, n_cells, 'bbox'


def _zscore(v):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if ok.sum() >= 2 and np.nanstd(v[ok], ddof=1) > 0:
        out[ok] = (v[ok] - np.nanmean(v[ok])) / np.nanstd(v[ok], ddof=1)
    return out


INDEX_KEY = ('Model', 'Window', 'Measure', 'Scheme', 'Cluster')


def _merge_index(path, rows, fields):
    """Write posthoc_index.csv merged with what is already there.

    `--models both` calls posthoc_one_grain twice on the same folder; a plain write would make
    the second pass (covariates) delete the first pass's rows, and the index is read as the
    inventory of its directory. Merged on (Model, Window, Measure, Scheme, Cluster); a re-run of
    the same cell replaces its row rather than duplicating it.
    """
    old = {}
    if os.path.isfile(path):
        try:
            with io.open(path, encoding='utf-8-sig', newline='') as fh:
                for r in csv.DictReader(fh):
                    old[tuple(str(r.get(k, '')) for k in INDEX_KEY)] = r
        except (OSError, ValueError):
            old = {}
    for r in rows:
        old[tuple(str(r.get(k, '')) for k in INDEX_KEY)] = r
    _write_csv(path, [old[k] for k in sorted(old)], fields)


def _write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_outcomes_status(main_dir, grain, models_mode, status, n_cells=0):
    """Record in a file that this grain was looked at and produced nothing.

    Without it, a grain where the battery ran and was gated out (nothing survived, so there is
    nothing to associate with an outcome) would look on disk exactly like a grain that was never
    processed.
    """
    out = os.path.join(os.path.dirname(main_dir), 'outcomes_demographics',
                       grain)
    # Merged on (Grain, Models), not overwritten, for the same reason as `_merge_index`: under
    # `--models both` this is called twice on one folder, and a plain write would leave only
    # the covariates row, which reads as "the grain produced nothing" when the main pass
    # produced everything.
    path = os.path.join(out, 'outcomes_status.csv')
    fields = ['Grain', 'Models', 'Status', 'N_cells']
    keep = {}
    if os.path.isfile(path):
        try:
            with io.open(path, encoding='utf-8-sig', newline='') as fh:
                for r in csv.DictReader(fh):
                    keep[(str(r.get('Grain', '')), str(r.get('Models', '')))] = r
        except (OSError, ValueError):
            keep = {}
    keep[(grain, models_mode)] = dict(Grain=grain, Models=models_mode,
                                      Status=status, N_cells=n_cells)
    _write_csv(path, [keep[k] for k in sorted(keep)], fields)


# ═══════════════════════════════════════════════════════════════════════════
def posthoc_one_grain(main_dir, beh_path, models_mode, verbose=True,
                      dry_run=False, only_window=None,
                      qrs_edge=QRS_EDGE_MS):
    """Outcomes + demographics for one <grain>_wise/main. Returns a summary.

    `only_window` is forwarded to `surviving_rows`, not applied here: it is how the cascade
    restricts a run to the one window it stopped at.
    """
    grain = os.path.basename(os.path.dirname(main_dir)).replace('_wise', '')
    npz = os.path.join(main_dir, 'run_results.npz')
    if not os.path.isfile(npz):
        # No status file here on purpose: without run_results.npz this grain never produced an
        # L2, so there is no analysis to report the state of.
        return {'status': 'no npz', 'n_written': 0, 'cells': 0}
    z = np.load(npz, allow_pickle=True)
    try:
        l2 = json.loads(str(z['l2_rows_json']))
    except (KeyError, ValueError):
        if not dry_run:
            _write_outcomes_status(main_dir, grain, models_mode,
                                   'run_results.npz carries no L2 rows — the '
                                   'L2 never completed for this grain')
        return {'status': 'no l2 rows', 'n_written': 0, 'cells': 0}
    sids = [str(s) for s in z['subject_ids']]
    # None (not {}) when the run has no persisted cells (see heart_depth_run.load_l2_cells);
    # `exact_cells` then returns None for every cluster and each row is stamped Mask=bbox.
    try:
        from heart_depth_run import load_l2_cells
        cells_by_key = load_l2_cells(z)
    except Exception:                                       # noqa: BLE001
        cells_by_key = None
    # The covariate QRS gate is off only when qrs_edge is -inf: a covariate cluster that reaches
    # back into the cardiac-field window is then reported instead of dropped. A finite edge,
    # including 0, is applied (see parse_qrs_edge). Every row still carries `Clears_QRS`.
    hits = surviving_rows(l2, models_mode, only_window=only_window,
                          qrs_gate_covariates=np.isfinite(float(qrs_edge)),
                          qrs_edge=float(qrs_edge))
    if not hits:
        # Name the window in the status: under the cascade this gate is window-restricted, so
        # "no cluster survived" is a claim about one window, not about the whole grain.
        _wtxt = '' if only_window is None else ' at window %r' % only_window
        if not dry_run:
            _write_outcomes_status(
                main_dir, grain, models_mode,
                'RAN AND WAS GATED OUT: no cluster survived under --models %s%s'
                ', so there is nothing to associate with an outcome. A MEASURED '
                'null, not a skipped check.' % (models_mode, _wtxt))
        return {'status': 'no surviving cluster under --models %s%s'
                          % (models_mode, _wtxt),
                'n_written': 0, 'cells': 0}

    Y, floor, missing, gaps = load_outcomes(beh_path, sids, return_gaps=True)
    # An absent optional instrument is a data gap, not a null. Its rows are still written, with
    # n = 0 and NaN throughout, so it is reported here (see outcomes.OPTIONAL_TABS).
    if gaps and verbose:
        print('  WARNING optional questionnaire file(s) not on disk: %s -- '
              'their outcomes are written with n=0 and are NOT null results.'
              % ', '.join(gaps))
    if missing:
        if not dry_run:
            _write_outcomes_status(
                main_dir, grain, models_mode,
                'BEHAVIOURAL FILES MISSING (%s) — %d surviving cell(s) were '
                'never tested. This is a data gap, NOT a null association.'
                % (', '.join(missing), len(hits)), len(hits))
        return {'status': 'behavioural files missing: %s' % ', '.join(missing),
                'n_written': 0, 'cells': len(hits)}
    D, dlabels, dcov = load_demographics(beh_path, sids)

    out_dir = os.path.join(os.path.dirname(main_dir), 'outcomes_demographics')
    n_written = 0
    index = []
    for model, window, r in hits:
        meas = str(r.get('Measure') or '').lower()
        scheme = str(r.get('Scheme') or '')
        clu = r.get('Cluster')
        l2key = model if window == 'full' else '%s@%s' % (model, window)
        # `exact_cells` returns (freq_index, full-axis time index); see its docstring for why
        # the window index would be wrong for '@lit'.
        _rec = exact_cells(cells_by_key, l2key, meas, scheme, clu)
        vals, n_cells, mask_kind = cluster_predictor(z, model, meas, r,
                                                     cells=_rec)
        # The file name spells the lit window as `<model>-lit` (`session-lit`).
        mtag = model if window == 'full' else '%s-%s' % (model, window)
        if vals is None or not np.isfinite(vals).any():
            index.append(dict(Model=model, Window=window, Measure=meas,
                              Scheme=scheme, Cluster=clu,
                              Status='no L1 map for this cell',
                              N_subjects=0, N_cells_box=n_cells, Mask=mask_kind))
            continue
        zed = _zscore(vals)
        stem = '%s_%s_%s_c%s' % (meas.upper(), mtag, scheme, clu)
        if dry_run:
            n_written += 1
            index.append(dict(Model=model, Window=window, Measure=meas,
                              Scheme=scheme, Cluster=clu, Status='would write',
                              N_subjects=int(np.isfinite(zed).sum()),
                              N_cells_box=n_cells, Mask=mask_kind))
            continue

        rows = spearman_ols_table(zed, Y, floor)
        for row in rows:
            row['Model'] = model
            row['Window'] = window
            row['Scheme'] = scheme
            row['Grain'] = grain
            row['Cluster'] = clu
            # The mask actually used: 'cells' is the cluster's exact connected set; 'bbox' is
            # the Start/End_ms x Start/End_Hz rectangle, which on a ragged cluster includes
            # cells the test never selected.
            row['Mask'] = mask_kind
            # Reported against the standard +50 ms edge, not the run's own --qrs-edge: with the
            # gate off, this column is what still says which clusters sit on the cardiac field.
            row['Clears_QRS'] = int(_clears_qrs(r))
            row['QRS_edge_ms'] = QRS_EDGE_MS
            row['Window_ms'] = '%g..%g' % (_f(r.get('Start_ms')),
                                           _f(r.get('End_ms')))
            row['Window_Hz'] = ('' if str(meas) == 'hep' else
                                '%g..%g' % (_f(r.get('Start_Hz')),
                                            _f(r.get('End_Hz'))))
            row['N_cells_box'] = n_cells
        _write_csv(os.path.join(out_dir, grain, '%s_beta_results.csv' % stem),
                   rows,
                   ['Outcome', 'Questionnaire', 'n', 'floor_effect',
                    'N_outliers', 'Spearman_rho', 'p_Spearman', 'rho_clean',
                    'p_clean', 'p_fdr_intra', 'OLS_Beta_std',
                    'OLS_t', 'OLS_p', 'OLS_R2', 'OLS_reliable', 'Model',
                    'Scheme', 'Grain', 'Cluster', 'Mask', 'Window',
                    'Window_ms', 'Window_Hz', 'N_cells_box',
                    'Clears_QRS', 'QRS_edge_ms'])

        if D is not None:
            drows = demographics_table(zed, D, dlabels)
            for row in drows:
                row['Model'] = model
                row['Window'] = window
                row['Scheme'] = scheme
                row['Grain'] = grain
                row['Mask'] = mask_kind
                row['Clears_QRS'] = int(_clears_qrs(r))
                row['N_covered'] = dcov
            _write_csv(os.path.join(out_dir, grain,
                                    '%s_demographics.csv' % stem),
                       drows, ['Demographic', 'n', 'rho', 'p_raw', 'p_fdr',
                               'Model', 'Window', 'Scheme', 'Grain', 'Mask',
                               'Clears_QRS', 'N_covered'])
            # Section C, moderation: does a demographic change how strongly the effect maps
            # onto an outcome? This is the legitimate way for a between-subject variable to
            # enter a within-subject slope, which is why age is not a covariate in any depth
            # model.
            mrows = moderation_table(zed, Y, D, dlabels)
            if mrows:
                for row in mrows:
                    row.update(Model=model, Window=window, Scheme=scheme,
                               Grain=grain, Mask=mask_kind,
                               Clears_QRS=int(_clears_qrs(r)))
                _write_csv(os.path.join(out_dir, grain,
                                        '%s_moderation.csv' % stem),
                           mrows, ['Moderator', 'Outcome', 'n', 'beta_effect',
                                   'beta_demog', 'beta_interact', 't_interact',
                                   'p_interact', 'p_fdr', 'Model', 'Window',
                                   'Scheme', 'Grain', 'Mask', 'Clears_QRS'])

        n_hit = sum(1 for row in rows
                    # The intra-questionnaire FDR, as the column name says.
                    if np.isfinite(row['p_fdr_intra'])
                    and row['p_fdr_intra'] < ALPHA)
        n_raw = sum(1 for row in rows
                    if np.isfinite(row['p_Spearman'])
                    and row['p_Spearman'] < ALPHA)
        index.append(dict(Model=model, Window=window, Measure=meas,
                          Scheme=scheme, Cluster=clu, Status='ok', Mask=mask_kind,
                          Clears_QRS=int(_clears_qrs(r)),
                          N_subjects=int(np.isfinite(zed).sum()),
                          N_cells_box=n_cells,
                          N_outcomes_raw_p05=n_raw,
                          N_outcomes_fdr_intra_p05=n_hit))
        n_written += 1
        if verbose:
            print('    %-13s %-5s %-14s c%-2s  n=%-3d  mask=%-5s  '
                  'raw p<.05: %d/%d   intra-FDR p<.05: %d'
                  % (mtag, meas, scheme, clu, int(np.isfinite(zed).sum()),
                     mask_kind, n_raw, len(OUTCOME_SPEC), n_hit))
    if index and not dry_run:
        _merge_index(os.path.join(out_dir, grain, 'posthoc_index.csv'), index,
                   ['Model', 'Window', 'Measure', 'Scheme', 'Cluster', 'Mask',
                    'Clears_QRS', 'Status', 'N_subjects', 'N_cells_box',
                    'N_outcomes_raw_p05', 'N_outcomes_fdr_intra_p05'])
    return {'status': 'ok', 'n_written': n_written, 'cells': len(hits)}


def tfce_component_cells(tz, measure, scheme, convention='sn',
                         component='largest'):
    """Exact cells of the TFCE effect to average, or (None, info) when none.

    Reads l2_tfce_<model>.npz (written by tfce_run.py). `component`:
      'largest'  the largest connected component of p < alpha (component 1; components are
                 sorted by cell count). Default: the analogue of the one cluster the cluster
                 route passes to the battery.
      'sign'     every significant cell of the largest component's sign.
    Cells are (freq index, full-axis time index); HEP has freq index 0.
    """
    idx = json.loads(str(tz['sigcells_index_json']))
    recs = idx.get('%s|%s|%s' % (measure, scheme, convention)) or []
    if not recs:
        return None, dict(n_components=0)
    xy = np.asarray(tz['sigcells__%s__%s__%s' % (measure, scheme, convention)],
                    int)
    first = recs[0]
    if component == 'largest':
        sel = [first]
    elif component == 'sign':
        sel = [r for r in recs if r['polarity'] == first['polarity']]
    else:
        raise ValueError("component must be 'largest' or 'sign', got %r"
                         % (component,))
    cells = np.concatenate([xy[r['start']:r['stop']] for r in sel])
    return cells, dict(n_components=len(recs), polarity=first['polarity'],
                       n_cells_used=int(cells.shape[0]),
                       n_cells_sig_total=int(sum(r['n_cells'] for r in recs)))


TFCE_INDEX_KEY = ('Model', 'Measure', 'Scheme', 'Convention', 'Component_rule')


def tfce_pairs(trows, schemes=None):
    """(measure, scheme) pairs of a TFCE summary, in file order, tested ones
    only ('ok' / 'no_sig'). `schemes` None = every scheme in the file (which
    is how ivw_own gets its tables once tfce_run --schemes ivw_own has
    written its rows); otherwise only the named schemes."""
    pairs = []
    for r in trows:
        k = (r['Measure'], r['Scheme'])
        if schemes is not None and r['Scheme'] not in schemes:
            continue
        if k not in pairs and r.get('Status') in ('ok', 'no_sig'):
            pairs.append(k)
    return pairs


def _merge_tfce_index(path, rows, fields):
    """posthoc_index_<model>.csv with the rows of OTHER schemes kept: a
    --tfce-schemes call covers part of the file, and the index is read as the
    inventory of its folder. Keyed on TFCE_INDEX_KEY; file order preserved,
    a redone row replaced in place."""
    old = []
    if os.path.isfile(path):
        try:
            with io.open(path, encoding='utf-8-sig', newline='') as fh:
                old = list(csv.DictReader(fh))
        except (OSError, ValueError):
            old = []
    key = lambda r: tuple(str(r.get(k, '')) for k in TFCE_INDEX_KEY)   # noqa: E731
    new = {key(r): r for r in rows}
    out = [new.pop(key(r), r) for r in old]
    out.extend(r for r in rows if key(r) in new)
    _write_csv(path, out, fields)


def posthoc_one_grain_tfce(main_dir, beh_path, model='session',
                           convention='sn', component='largest', verbose=True,
                           schemes=None):
    """Outcomes + demographics + moderation on the TFCE-significant cells.

    The TFCE counterpart of posthoc_one_grain, for ONE model. Same predictor
    (per-participant mean L1 slope over an EXACT cell set, via
    cluster_predictor), same z-scoring and the same three tables; only the cell
    set differs: the `component` of p < alpha under the TFCE test instead of a
    surviving cluster (see tfce_component_cells). Written to
    <grain>_wise/outcomes_demographics_tfce/<grain>/, never into the cluster
    route's outcomes_demographics/.

    Gated like the cluster route: a measure x scheme with no significant TFCE cell gets no
    tables (it is listed in the index as gated), because a post-hoc runs only on an effect that
    survived.

    `schemes`: None = every scheme in the TFCE summary. A list restricts the battery to those
    schemes (e.g. ['ivw_own']) and the index is then merged, so the other schemes' rows stay
    listed. The scheme only decides which cells are significant; the predictor is each
    participant's unweighted mean L1 slope over those cells under every scheme.
    """
    grain = os.path.basename(os.path.dirname(main_dir)).replace('_wise', '')
    npz = os.path.join(main_dir, 'run_results.npz')
    tnpz = os.path.join(main_dir, 'l2_tfce_%s.npz' % model)
    tcsv = os.path.join(main_dir, 'l2_tfce_summary_%s.csv' % model)
    if not (os.path.isfile(npz) and os.path.isfile(tnpz) and os.path.isfile(tcsv)):
        return {'status': 'no TFCE result for model %s (run run_component_group_tests.py --steps tfce)'
                          % model, 'n_written': 0}
    z = np.load(npz, allow_pickle=False)
    tz = np.load(tnpz, allow_pickle=False)
    with io.open(tcsv, encoding='utf-8-sig', newline='') as fh:
        trows = [r for r in csv.DictReader(fh) if r.get('Convention') == convention]
    sids = [str(s) for s in z['subject_ids']]
    Y, floor, missing, _gaps = load_outcomes(beh_path, sids, return_gaps=True)
    if missing:
        return {'status': 'behavioural files missing: %s' % ', '.join(missing),
                'n_written': 0}
    D, dlabels, dcov = load_demographics(beh_path, sids)
    out_dir = os.path.join(os.path.dirname(main_dir), 'outcomes_demographics_tfce',
                           grain)
    index, n_written = [], 0
    pairs = tfce_pairs(trows, schemes)
    for meas, scheme in pairs:
        comp1 = [r for r in trows if r['Measure'] == meas and r['Scheme'] == scheme
                 and r['Status'] == 'ok' and str(r['Component']) == '1']
        cells, info = tfce_component_cells(tz, meas, scheme, convention,
                                           component)
        base = dict(Model=model, Measure=meas, Scheme=scheme,
                    Convention=convention, Component_rule=component,
                    N_components=info.get('n_components', 0))
        if cells is None or not comp1:
            index.append(dict(base, Status='gated: no significant TFCE cell'))
            continue
        c1 = comp1[0]
        vals, n_cells, mask_kind = cluster_predictor(z, model, meas, c1,
                                                     cells=cells)
        if vals is None or not np.isfinite(vals).any():
            index.append(dict(base, Status='no L1 map for this cell'))
            continue
        zed = _zscore(vals)
        clears = int(_clears_qrs(c1))
        wms = '%g..%g' % (_f(c1['Start_ms']), _f(c1['End_ms']))
        whz = ('' if meas == 'hep' else
               '%g..%g' % (_f(c1['Start_Hz']), _f(c1['End_Hz'])))
        stem = '%s_%s_%s_tfce-%s_%s' % (meas.upper(), model, scheme, convention,
                                        component)
        meta = dict(Model=model, Scheme=scheme, Grain=grain, Convention=convention,
                    Component_rule=component, Mask=mask_kind, Window_ms=wms,
                    Window_Hz=whz, N_cells=n_cells, Clears_QRS=clears,
                    QRS_edge_ms=QRS_EDGE_MS)
        rows = spearman_ols_table(zed, Y, floor)
        for row in rows:
            row.update(meta)
        _write_csv(os.path.join(out_dir, '%s_beta_results.csv' % stem), rows,
                   ['Outcome', 'Questionnaire', 'n', 'floor_effect',
                    'N_outliers', 'Spearman_rho', 'p_Spearman', 'rho_clean',
                    'p_clean', 'p_fdr_intra', 'OLS_Beta_std', 'OLS_t', 'OLS_p',
                    'OLS_R2', 'OLS_reliable'] + list(meta))
        if D is not None:
            drows = demographics_table(zed, D, dlabels)
            for row in drows:
                row.update(meta, N_covered=dcov)
            _write_csv(os.path.join(out_dir, '%s_demographics.csv' % stem),
                       drows, ['Demographic', 'n', 'rho', 'p_raw', 'p_fdr']
                       + list(meta) + ['N_covered'])
            mrows = moderation_table(zed, Y, D, dlabels)
            if mrows:
                for row in mrows:
                    row.update(meta)
                _write_csv(os.path.join(out_dir, '%s_moderation.csv' % stem),
                           mrows, ['Moderator', 'Outcome', 'n', 'beta_effect',
                                   'beta_demog', 'beta_interact', 't_interact',
                                   'p_interact', 'p_fdr'] + list(meta))
        n_hit = sum(1 for row in rows if np.isfinite(row['p_fdr_intra'])
                    and row['p_fdr_intra'] < ALPHA)
        n_raw = sum(1 for row in rows if np.isfinite(row['p_Spearman'])
                    and row['p_Spearman'] < ALPHA)
        index.append(dict(base, Status='ok', Mask=mask_kind, Window_ms=wms,
                          Window_Hz=whz, Clears_QRS=clears,
                          N_subjects=int(np.isfinite(zed).sum()),
                          N_cells=n_cells,
                          N_cells_sig_total=info.get('n_cells_sig_total'),
                          N_outcomes_raw_p05=n_raw,
                          N_outcomes_fdr_intra_p05=n_hit))
        n_written += 1
        if verbose:
            print('    %-8s %-5s %-14s %s  n=%-3d cells=%-4d  raw p<.05: %d/%d   '
                  'intra-FDR p<.05: %d'
                  % (model, meas, scheme, convention, int(np.isfinite(zed).sum()),
                     n_cells, n_raw, len(OUTCOME_SPEC), n_hit))
    if index:
        (_write_csv if schemes is None else _merge_tfce_index)(
            os.path.join(out_dir, 'posthoc_index_%s.csv' % model), index,
            ['Model', 'Measure', 'Scheme', 'Convention', 'Component_rule',
             'Status', 'N_components', 'Mask', 'Window_ms', 'Window_Hz',
             'Clears_QRS', 'N_subjects', 'N_cells', 'N_cells_sig_total',
             'N_outcomes_raw_p05', 'N_outcomes_fdr_intra_p05'])
    return {'status': 'ok', 'n_written': n_written}


def posthoc_run(run_dir, beh_path, models_mode, verbose=True,
                dry_run=False, qrs_edge=QRS_EDGE_MS):
    """Post-hocs for one run folder.

    `models_mode='cascade'` applies the rule in `cascade`: the run reports the first of
    main/full, main/lit, covariates/full, covariates/lit that has a cluster surviving and
    clearing +50 ms, and only that level gets the outcome battery. `main` / `covariates` /
    `both` are for a targeted re-run and do not consult the cascade.

    The cascade is computed once across all grains, then applied to each: the three grains are
    three views of one configuration, so the run reports one level.
    """
    name = os.path.basename(os.path.normpath(run_dir))
    if verbose:
        print('\n== %s' % name)
    # Only the full window is analysed outside the cascade, so a results tree that still holds
    # `<model>@lit` keys does not produce a parallel set of `-lit_` outcome tables.
    only_window = 'full'
    if models_mode == 'cascade':
        v = cascade_of_run(run_dir, GRAINS, qrs_edge=float(qrs_edge))
        if v['level'] == 0:
            if verbose:
                print('  cascade: ALL FOUR LEVELS EMPTY -- %d cluster(s) '
                      'survived but none clears +%g ms, so there is nothing '
                      'to run a post-hoc on. A MEASURED null.'
                      % (v['n_survived'], float(qrs_edge)))
                for _g, _m, _w, _r in v['demoted'][:6]:
                    print('    demoted: %-8s %-13s %-4s %s..%s ms'
                          % (_g, _m, _w, _r.get('Start_ms'), _r.get('End_ms')))
            return 0
        models_mode, only_window = v['models_mode'], v['window']
        if verbose:
            print('  cascade STOPS AT %s -- %d qualifying cluster(s)%s'
                  % (level_name(v['level']), len(v['hits']),
                     ('; %d QRS-window cluster(s) demoted' % len(v['demoted']))
                     if v['demoted'] else ''))
    total = 0
    for g in GRAINS:
        main_dir = os.path.join(run_dir, '%s_wise' % g, 'main')
        if not os.path.isdir(main_dir):
            continue
        res = posthoc_one_grain(main_dir, beh_path, models_mode,
                                verbose=verbose, dry_run=dry_run,
                                only_window=only_window,
                                qrs_edge=float(qrs_edge))
        total += res['n_written']
        if verbose and res['n_written'] == 0:
            print('  %-8s %s' % (g, res['status']))
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default=None, help='one run folder')
    ap.add_argument('--root', default=None,
                    help='a results root; every HEART_DEPTH run under it')
    ap.add_argument('--beh', default=DEFAULT_BEH)
    # 'main' analyses session/unadj where they survived; 'covariates' analyses every surviving
    # covariate model instead. 'both' runs the two in sequence (it is not a third mode), so one
    # call gives the outcome tables of the main and of the covariate models.
    ap.add_argument('--models',
                    choices=('cascade', 'main', 'covariates', 'both'),
                    default='cascade',
                    help="DEFAULT 'cascade': the run reports the FIRST of "
                         'main/full, main/lit, covariates/full, covariates/lit '
                         'with a cluster that SURVIVED and clears +50 ms, and '
                         'only that level gets the battery. The other three are '
                         'targeted re-runs and ignore the cascade.')
    # The cardiac-field (QRS) edge. Default +50 ms: a cluster reaching back before it sits on
    # the cardiac-field artefact and cannot stop the cascade (it is demoted, never deleted).
    # `--qrs-edge off` turns the gate off, so such a cluster can stop the cascade and gets the
    # full battery. Use it for sensitivity questions, and read `Clears_QRS` in the output.
    ap.add_argument('--qrs-edge', type=parse_qrs_edge, default=QRS_EDGE_MS,
                    metavar='MS',
                    help='cardiac-field edge in ms (default %g). '
                         "'off' disables the gate entirely; 0 does NOT -- it "
                         'is a bar at 0 ms, and the tested window starts at '
                         '-100 ms.' % QRS_EDGE_MS)
    ap.add_argument('--dry-run', action='store_true')
    # TFCE route, off by default. With --tfce only the TFCE battery runs (no cluster tables), on
    # l2_tfce_<model>.npz written by tfce_run.py, into <grain>_wise/outcomes_demographics_tfce/.
    ap.add_argument('--tfce', action='store_true',
                    help='outcome/demographic tables on the TFCE-significant '
                         'cells instead of the cluster cells')
    ap.add_argument('--tfce-model', default='session')
    ap.add_argument('--tfce-grains', nargs='+', default=list(GRAINS))
    ap.add_argument('--tfce-convention', default='sn', choices=('sn', 'mne'))
    ap.add_argument('--tfce-component', default='largest',
                    choices=('largest', 'sign'))
    ap.add_argument('--tfce-schemes', nargs='+', default=None,
                    help='restrict the TFCE battery to these schemes (default: '
                         'every scheme in l2_tfce_summary_<model>.csv, ivw_own '
                         'included when tfce_run wrote it)')
    a = ap.parse_args(argv)

    if not (a.run or a.root):
        ap.error('give --run or --root')
    if a.tfce:
        if not a.run:
            ap.error('--tfce takes one --run')
        n = 0
        for g in a.tfce_grains:
            main_dir = os.path.join(a.run, '%s_wise' % g, 'main')
            print('== TFCE post-hoc %s | %s | model %s | %s | %s'
                  % (os.path.basename(os.path.normpath(a.run)), g, a.tfce_model,
                     a.tfce_convention, a.tfce_component))
            res = posthoc_one_grain_tfce(main_dir, a.beh, a.tfce_model,
                                         a.tfce_convention, a.tfce_component,
                                         schemes=a.tfce_schemes)
            print('   %s, %d cell(s) written' % (res['status'], res['n_written']))
            n += res['n_written']
        return 0

    runs = []
    if a.run:
        runs = [a.run]
    else:
        arm = os.path.join(a.root, 'HEART_DEPTH')
        base = arm if os.path.isdir(arm) else a.root
        runs = [os.path.join(base, d) for d in sorted(os.listdir(base))
                if os.path.isdir(os.path.join(base, d))]

    print('=' * 78)
    print('HEART POST-HOC  (models=%s%s)' % (a.models,
                                             ', DRY RUN' if a.dry_run else ''))
    print('  behavioural  %s' % a.beh)
    print('  runs         %d' % len(runs))
    print('=' * 78)
    # 'cascade' decides its own mode per run inside posthoc_run and is not expanded here; 'both'
    # is the only mode that means two passes.
    modes = ('main', 'covariates') if a.models == 'both' else (a.models,)
    tot = 0
    for r in runs:
        for mi, mode in enumerate(modes):
            if len(modes) > 1:
                print('%s--- models=%s ---' % (os.linesep, mode))
            tot += posthoc_run(r, a.beh, mode,
                               dry_run=a.dry_run, qrs_edge=a.qrs_edge)
    print('\n%s\n%d cell(s) analysed across %d run(s)%s\n%s'
          % ('=' * 78, tot, len(runs),
             ' (dry run: nothing written)' if a.dry_run else '', '=' * 78))
    if tot == 0:
        print('⚠️  Nothing was analysed. Under --models %s that means no cell '
              'SURVIVED,' % a.models)
        print('    not that the association was null — the battery is GATED on '
              'a surviving cluster')
        print('    and a forced fallback is how a null becomes a finding.')
        if a.models != 'both':
            print('    Try --models %s, or --models both.'
                  % ('covariates' if a.models == 'main' else 'main'))
    return 0
