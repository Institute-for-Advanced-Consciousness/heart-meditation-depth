# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
fmt_period: frontal-midline theta contrasted between task periods.

run_period_contrasts.py calls `run(data_dir, results_root, params, limit=...)` with
config.PERIOD_FMT; `params` updates CONFIG below.

Reuses fmt_depth_pipeline (`build_ged`, which builds its template with
`make_frontal_template`, `signflip_ttest` and `shrunk_ivw`) and reads the periods from the
segment .npz files, which carry them.

===========================================================================
WHAT IT COMPARES
===========================================================================
Four periods per session:

    baseline          resting, eyes-open
    pre-meditation    the warm-up
    post-meditation   after the meditation
    active-oddball    the attention task

All six pairwise contrasts are fitted. They are not independent: six contrasts over four
means have only three degrees of freedom between them, and nothing here corrects across
them. Read the pre-registered ones (`premed_vs_baseline`, `oddball_vs_baseline`,
`postmed_vs_premed`) and treat the rest as description.

===========================================================================
FILTERING
===========================================================================
The .npz stores each period already cut into non-overlapping 4 s windows. Filtering each
window on its own would put the FIR transient inside every window, a large fraction of a
4 s window at 5-8 Hz. `extract_period_windows` steps `pos += win_samp` with no gap, so
consecutive windows are contiguous samples: this module concatenates a period's windows
back into one continuous block, filters that, and re-splits, which is close to filtering
the continuous recording.

A period made of more than one span is spliced at the span joins, because the .npz does
not record which span a window came from: at most (n_spans - 1) discontinuities per
period, each costing the filter about 1 s of transient out of several hundred seconds.
The window count of each period is recorded so a short period is visible.

===========================================================================
THE MEASURES
===========================================================================
  FMT_scalp   mean log10 theta power over FMT_ELECS (Brandmeyer & Delorme 2018)
  FMT_GED     GED spatial filter maximising theta / broadband variance, selected
              by template match, trained on the GED_TRAIN_ON periods only and
              applied to all four.

The GED filter must not be trained on the contrast it is tested on: a filter built to
maximise theta variance across the periods, then used to contrast them, is circular.
`GED_TRAIN_ON` defaults to the two reference periods (baseline and pre-meditation). The
selected component's template R² is kept (`ged_template_r2` in cohort_sessions.csv), so
whether it was frontocentral can be checked on a finished run.
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

from fmt_depth_pipeline import build_ged, signflip_ttest, shrunk_ivw

# ===========================================================================
#                              ↓↓  CONFIG  ↓↓
# ===========================================================================
CONFIG = dict(
    THETA_BP=[5, 8],              # Hz, Brandmeyer & Delorme's band
    FMT_ELECS=['Fz', 'FCz', 'Cz', 'AFz', 'F1', 'F2', 'FC1', 'FC2'],
    # FCz and AFz are not in this dataset. The scalp measure runs on whatever subset is
    # present, and the electrodes actually used are recorded per session.
    GED_TEMPLATE_ELEC='ROI',      # 'ROI' = the FMT_ELECS centroid | 'Fz' | 'Cz'
    GED_R2_THRESH=0.30,           # template match a component must reach
    GED_TOP_N=7,                  # candidates screened, in eigenvalue order
    GED_SHRINKAGE=0.01,
    GED_TRAIN_ON=['baseline', 'pre-meditation'],   # not the tested contrast
    PERIODS=['baseline', 'pre-meditation', 'post-meditation', 'active-oddball'],
    CONTRASTS='all',              # 'all' = the six pairs | a list of [A, B]
    MIN_WINDOWS=10,               # a period below this is dropped, and reported
    STANDARDIZE_WITHIN_SESSION=True,
    N_PERM=10000,
    ALPHA=0.05,
    SCHEMES=['unweighted', 'precision_nwin'],
)
# ===========================================================================

# Ordered (A = reference, B = test) so `delta = mean(B) - mean(A)` has a
# stable sign. Reversing a pair flips every number in its row.
_PAIRS = [
    ('pre-meditation',  'baseline'),        # warm-up vs rest
    ('active-oddball',  'baseline'),        # task vs rest
    ('post-meditation', 'baseline'),        # after meditation vs rest
    ('post-meditation', 'pre-meditation'),  # the meditation effect
    ('active-oddball',  'pre-meditation'),
    ('post-meditation', 'active-oddball'),
]


def contrasts_of(cfg):
    """[(tag, A, B, label)]: A is the reference, delta = mean(B) - mean(A)."""
    if cfg['CONTRASTS'] != 'all':
        pairs = [(a, b) for a, b in cfg['CONTRASTS']]
    else:
        pairs = [(a, b) for b, a in _PAIRS]      # _PAIRS is (test, ref)
    out = []
    for a, b in pairs:
        for p in (a, b):
            if p not in cfg['PERIODS']:
                raise ValueError(
                    'contrast names period %r, which is not in PERIODS %r. A '
                    'contrast against a period that was never loaded would be '
                    'all-NaN and would still print a row.' % (p, cfg['PERIODS']))
        out.append(('%s_vs_%s' % (_short(b), _short(a)), a, b,
                    '%s vs %s' % (b, a)))
    return out


def _short(p):
    return {'baseline': 'baseline', 'pre-meditation': 'premed',
            'post-meditation': 'postmed', 'active-oddball': 'oddball'}.get(p, p)


def theta_filter(x, fs, lo, hi):
    """Zero-phase FIR bandpass, MNE/EEGLAB `pop_eegfiltnew` convention.

    The same call as the depth route, so the FMT arms share one filter definition:
    `mne.filter.filter_data` with phase='zero', firwin, hamming.
    """
    import mne
    return mne.filter.filter_data(
        np.asarray(x, np.float64), sfreq=float(fs), l_freq=float(lo),
        h_freq=float(hi), method='fir', phase='zero', fir_design='firwin',
        fir_window='hamming', pad='reflect_limited',
        verbose='ERROR').astype(np.float32)


def period_block(z, period, key='eeg'):
    """A period's windows re-joined into one continuous [nChan x nSamp] block.

    Returns (block, n_windows, win_samp). `block` is None when the period is
    absent or empty, which is a fact about the session, not an error.
    """
    k = 'period_%s_%s' % (period, key)
    if k not in z.files:
        return None, 0, 0
    a = np.asarray(z[k])
    if a.size == 0 or a.shape[0] == 0:
        return None, 0, 0
    if a.ndim == 3:                       # [nwin, nChan, nSamp]
        n_win, n_ch, n_s = a.shape
        return a.transpose(1, 0, 2).reshape(n_ch, n_win * n_s), n_win, n_s
    n_win, n_s = a.shape                  # [nwin, nSamp] (ecg / resp)
    return a.reshape(1, n_win * n_s), n_win, n_s


def window_log_power(sig, n_win, win_samp):
    """Mean log10 power per window from a filtered [nSamp] or [nChan x nSamp].

    log10 of the mean square, with a double-precision floor. The .npz hold volts; theta
    power here is ~1e-12 V², so a float32 eps (1.19e-07) would be ~1e5 times the signal
    and log10(power + eps) would be log10(eps) almost everywhere.
    """
    x = np.asarray(sig, float)
    if x.ndim == 1:
        x = x[None, :]
    p = np.mean(x ** 2, axis=0)                       # over channels
    p = p[:n_win * win_samp].reshape(n_win, win_samp)
    return np.log10(np.mean(p, axis=1) + np.finfo(float).eps)


def session_periods(npz_path, p, verbose=True):
    """Per-window log theta power for every period of one session.

    -> dict(period -> {'scalp': [nwin], 'ged': [nwin], 'n': int}), plus meta.
    """
    z = np.load(npz_path, allow_pickle=True)
    fs = int(z['fs'])
    ch_names = [str(c) for c in np.asarray(z['ch_names']).ravel()]
    ch_pos = np.asarray(z['ch_positions'], float)
    notes = {'subject': str(z['subject_id']), 'session': int(z['session'])}

    # -- the scalp electrode set, and what is actually there ----------------
    want = [c.lower() for c in p['FMT_ELECS']]
    ix = [i for i, c in enumerate(ch_names) if c.lower() in want]
    missing = [c for c in p['FMT_ELECS'] if c.lower()
               not in {ch_names[i].lower() for i in ix}]
    notes['scalp_elecs_used'] = [ch_names[i] for i in ix]
    notes['scalp_elecs_missing'] = missing
    if not ix:
        notes['skip_reason'] = ('none of FMT_ELECS %r is a channel in this '
                                'recording' % (p['FMT_ELECS'],))
        return None, notes

    # -- filter each period once, on the re-joined block --------------------
    blocks, filt = {}, {}
    for per in p['PERIODS']:
        blk, n_win, win_samp = period_block(z, per)
        if blk is None or n_win < int(p['MIN_WINDOWS']):
            blocks[per] = None
            notes.setdefault('short_periods', []).append(
                '%s: %d window(s) < MIN_WINDOWS %d'
                % (per, n_win, int(p['MIN_WINDOWS'])))
            continue
        blocks[per] = (blk, n_win, win_samp)
        filt[per] = theta_filter(blk, fs, p['THETA_BP'][0], p['THETA_BP'][1])

    if not filt:
        notes['skip_reason'] = 'no period met MIN_WINDOWS'
        return None, notes

    # -- the GED filter: trained on the reference periods only --------------
    train = [q for q in p['GED_TRAIN_ON'] if q in filt]
    ged = {'ok': False}
    if train:
        bb = np.concatenate([blocks[q][0] for q in train], axis=1)
        th = np.concatenate([filt[q] for q in train], axis=1)
        tmpl_centre = (p['FMT_ELECS'] if str(p['GED_TEMPLATE_ELEC']).upper()
                       == 'ROI' else [p['GED_TEMPLATE_ELEC']])
        try:
            ged = build_ged(bb, th, np.ones(bb.shape[1], bool), ch_names,
                            ch_pos, shrinkage=p['GED_SHRINKAGE'],
                            template_elec=tmpl_centre,
                            r2_thresh=p['GED_R2_THRESH'],
                            top_n=p['GED_TOP_N'])
        except Exception as exc:                          # noqa: BLE001
            ged = {'ok': False, 'r2': None, 'best_possible_r2': None}
            notes['ged_error'] = '%s: %s' % (type(exc).__name__, exc)
    else:
        notes['ged_error'] = ('no GED_TRAIN_ON period survived MIN_WINDOWS, so '
                              'no filter could be trained')
    notes['ged_ok'] = bool(ged.get('ok'))
    notes['ged_template_r2'] = ged.get('r2')
    notes['ged_template_r2_best'] = ged.get('best_possible_r2')

    out = {}
    for per, blk in blocks.items():
        if blk is None:
            continue
        _b, n_win, win_samp = blk
        rec = {'n': int(n_win)}
        rec['scalp'] = window_log_power(filt[per][ix, :], n_win, win_samp)
        if ged.get('ok'):
            comp = ged['w'] @ filt[per]
            rec['ged'] = window_log_power(comp, n_win, win_samp)
        else:
            rec['ged'] = np.full(n_win, np.nan)
        out[per] = rec

    # -- z-score within session, across every period at once ----------------
    # Pooled across periods on purpose: standardising each period separately would
    # set every period's mean to 0 and delete the contrast. This removes the
    # per-session GED scale and the recording day.
    if p['STANDARDIZE_WITHIN_SESSION']:
        for meas in ('scalp', 'ged'):
            allv = np.concatenate([out[q][meas] for q in out])
            mu, sd = np.nanmean(allv), np.nanstd(allv)
            if np.isfinite(sd) and sd > 0:
                for q in out:
                    out[q][meas] = (out[q][meas] - mu) / sd
    return out, notes


def build_cohort(data_dir, p, limit=None, verbose=True):
    """Per-subject period means, pooled over that subject's sessions."""
    subs = sorted(d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and (d.startswith('sub-') or d.isdigit()))
    if limit:
        subs = subs[:int(limit)]
    per_sub, records = {}, []
    for sid in subs:
        sdir = os.path.join(data_dir, sid)
        acc = {}
        for f in sorted(os.listdir(sdir)):
            if not f.endswith('.npz'):
                continue
            res, notes = session_periods(os.path.join(sdir, f), p,
                                         verbose=verbose)
            rec = dict(subject=sid, session=f, status='kept', reason='',
                       ged_ok=notes.get('ged_ok'),
                       ged_template_r2=notes.get('ged_template_r2'),
                       n_scalp_elecs=len(notes.get('scalp_elecs_used') or []))
            if res is None:
                rec.update(status='dropped',
                           reason=notes.get('skip_reason', 'unknown'))
                records.append(rec)
                if verbose:
                    print('  %s %s DROPPED - %s' % (sid, f, rec['reason']))
                continue
            rec['periods'] = ','.join('%s:%d' % (q, res[q]['n']) for q in res)
            records.append(rec)
            for q, v in res.items():
                for meas in ('scalp', 'ged'):
                    acc.setdefault((q, meas), []).append(
                        (np.nanmean(v[meas]), v['n']))
            if verbose:
                print('  %s %s kept - %s%s' % (
                    sid, f, rec['periods'],
                    '' if notes.get('ged_ok') else '  (no GED component)'))
        if acc:
            # Mean over sessions, not over pooled windows: a subject whose two days
            # differ in length would otherwise be represented mostly by the longer
            # one. `nwin` is summed and kept for the weighting scheme.
            per_sub[sid] = {k: (float(np.nanmean([m for m, _ in v])),
                                int(np.nansum([n for _, n in v])))
                            for k, v in acc.items()}
    return per_sub, records


def run_l2(per_sub, p, verbose=True):
    """Sign-flip t-test on the per-subject contrast deltas. -> rows."""
    rows = []
    subs = sorted(per_sub)
    for tag, a, b, label in contrasts_of(p):
        for meas in ('scalp', 'ged'):
            d, w = [], []
            for s in subs:
                va = per_sub[s].get((a, meas))
                vb = per_sub[s].get((b, meas))
                if va is None or vb is None:
                    continue
                if not (np.isfinite(va[0]) and np.isfinite(vb[0])):
                    continue
                d.append(vb[0] - va[0])
                w.append(min(va[1], vb[1]))      # the limiting period's windows
            d = np.asarray(d, float)
            n = int(d.size)
            for scheme in p['SCHEMES']:
                if n < 3:
                    rows.append(dict(
                        Contrast=tag, Label=label, Measure='FMT_%s' % meas,
                        Scheme=scheme, n=n, t=np.nan, p=np.nan, CohenD=np.nan,
                        Delta=np.nan,
                        Status='fewer than 3 subjects have BOTH periods'))
                    continue
                # shrunk_ivw returns (w, info) and already normalises to sum 1. It
                # expects variances; a window count is a precision, so it is
                # inverted before shrinkage.
                if scheme == 'unweighted':
                    ww = np.ones(n) / n
                else:
                    with np.errstate(divide='ignore'):
                        ww, _info = shrunk_ivw(1.0 / np.maximum(
                            np.asarray(w, float), 1.0))
                    tot = float(np.sum(ww))
                    ww = (np.ones(n) / n) if not (tot > 0) else ww / tot
                t, pv = signflip_ttest(d, ww, n_perm=int(p['N_PERM']))
                mu = float(np.sum(ww * d))
                sd = float(np.sqrt(np.sum(ww * (d - mu) ** 2)))
                rows.append(dict(
                    Contrast=tag, Label=label, Measure='FMT_%s' % meas,
                    Scheme=scheme, n=n, t=float(t), p=float(pv),
                    CohenD=(mu / sd if sd > 0 else np.nan), Delta=mu,
                    Status='ok'))
                if verbose and pv < p['ALPHA']:
                    print('    %-26s %-10s %-15s n=%-3d t=%6.2f p=%.4f'
                          % (tag, meas, scheme, n, t, pv))
    return rows


def _write_csv(path, rows, fields):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_period_values(path, per_sub, p):
    """Long-format per-subject period means: subject, period, measure, value, n.

    Long, not wide: a subject missing one period contributes no row for it, so a reader
    counts coverage instead of having to tell a blank cell from a zero.
    """
    rows = []
    for sid in sorted(per_sub):
        for (q, meas), (val, nwin) in sorted(per_sub[sid].items()):
            rows.append(dict(subject=sid, period=q, measure=meas,
                             value=val, n_windows=nwin))
    _write_csv(path, rows, ['subject', 'period', 'measure', 'value',
                            'n_windows'])
    return path


def run(data_dir, results_root, p=None, limit=None, verbose=True):
    p = dict(CONFIG, **(p or {}))
    tag = 'th%g-%g_tmpl%s_gedR2-%.2f_train-%s_%dperiods' % (
        p['THETA_BP'][0], p['THETA_BP'][1], p['GED_TEMPLATE_ELEC'],
        p['GED_R2_THRESH'], '+'.join(_short(q) for q in p['GED_TRAIN_ON']),
        len(p['PERIODS']))
    out_dir = os.path.join(results_root, 'FMT_PERIOD', tag, 'main')
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print('=' * 74)
        print('FMT PERIOD CONTRASTS  ->  %s' % out_dir)
        print('  periods    %s' % ', '.join(p['PERIODS']))
        print('  contrasts  %d' % len(contrasts_of(p)))
        print('  theta      %g-%g Hz   GED template %s (R2>=%.2f), trained on %s'
              % (p['THETA_BP'][0], p['THETA_BP'][1], p['GED_TEMPLATE_ELEC'],
                 p['GED_R2_THRESH'], '+'.join(p['GED_TRAIN_ON'])))
        print('=' * 74)

    per_sub, records = build_cohort(data_dir, p, limit=limit, verbose=verbose)
    rows = run_l2(per_sub, p, verbose=verbose)

    _write_csv(os.path.join(out_dir, 'fmt_period_L2.csv'), rows,
               ['Contrast', 'Label', 'Measure', 'Scheme', 'n', 't', 'p',
                'CohenD', 'Delta', 'Status'])
    # The per-subject values, which only the t-test saw: without them the run's only
    # drawable object is the t column, and "FMT by period" could not be drawn without
    # re-running the cohort. Long format (see _write_period_values), with the window
    # counts that carry the weights.
    _write_period_values(os.path.join(out_dir, 'period_subject_values.csv'),
                         per_sub, p)
    _write_csv(os.path.join(out_dir, 'cohort_sessions.csv'), records,
               ['subject', 'session', 'status', 'reason', 'periods',
                'ged_ok', 'ged_template_r2', 'n_scalp_elecs'])
    with io.open(os.path.join(out_dir, 'run_params.json'), 'w',
                 encoding='utf-8') as fh:
        json.dump({k: (list(v) if isinstance(v, (list, tuple)) else v)
                   for k, v in p.items()}, fh, indent=1, sort_keys=True)

    if verbose:
        kept = [r for r in records if r['status'] == 'kept']
        sig = [r for r in rows if r['Status'] == 'ok' and r['p'] < p['ALPHA']]
        print('\n  %d subject(s), %d session(s) kept, %d dropped'
              % (len(per_sub), len(kept), len(records) - len(kept)))
        print('  %d of %d rows at p < %g  UNCORRECTED, and the six contrasts '
              'are not independent' % (len(sig), len(rows), p['ALPHA']))
        print('  -> %s' % os.path.join(out_dir, 'fmt_period_L2.csv'))
    return {'rows': rows, 'records': records, 'out_dir': out_dir}
