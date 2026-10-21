# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
cascade: which level of a run is the result.

    from cascade import cascade_of_run, LEVELS, QRS_EDGE_MS

One definition, imported by every caller. The cascade decides three things that must not
disagree with each other:
    * which level a run reports,
    * which cells get the post-hoc battery (`heart_posthoc`),
    * which cell the surrogate control tests.

===========================================================================
THE CASCADE
===========================================================================
Stop at the first level that finds something:

    1. main models      (session, unadj)   full window
    2. main models                         lit  window
    3. covariate models (rr, ecgAmp, ...)  full window
    4. covariate models                    lit  window

It is a reporting rule, not a compute order. The L2 fits every model at both windows off
one cohort load, and the cascade needs the lower levels to know they are empty: a level
that was never fitted is not an empty level.

===========================================================================
A cluster in the CFA window does not stop the cascade
===========================================================================
A cluster qualifies only if it survived and starts at or after `QRS_EDGE_MS` (+50 ms). A
cluster reaching back before that is sitting on the QRS complex, and nothing in this arm
clears it: `ecgAmp`/`ecgAmpT` are scalar per-beat R- and T-wave amplitudes, so they
remove variance in how big the cardiac field is, never the field itself.

Such clusters stay in the L2 tables with `CFA_flag` set and are returned here in
`demoted`, so a reader can see them. They cannot stop the cascade, claim the run's
result, or collect post-hocs.

`CFA_flag` (< 150 ms) and `QRS_EDGE_MS` (< 50 ms) are different bars, both deliberate:
the flag warns on the wider window, the edge gates on the narrower one. The cascade uses
the edge.

A missing `Start_ms` does not qualify: a row with no bounds cannot be shown to clear the
window.

===========================================================================
The cascade is per run, pooled across grains
===========================================================================
The three grains belong to one configuration, so the run reports one level. Judging each
grain separately would let the cell grain stop at level 1 while the segment grain
reported a level-4 covariate: two answers to one question, and the weaker one would be
read as independent support. There is no per-grain switch; for the per-grain picture,
call `cascade_of_l2` with a one-grain dict.
"""
from __future__ import annotations

import json
import os

import numpy as np

__all__ = ['QRS_EDGE_MS', 'MAIN_MODELS', 'LEVELS', 'split_l2_key',
           'row_qualifies', 'cascade_of_l2', 'cascade_of_run', 'level_name']

# The same number heart_posthoc gates covariate outcome tables on.
QRS_EDGE_MS = 50.0

MAIN_MODELS = ('session', 'unadj')

# (models_mode, window), in escalation order.
LEVELS = (('main', 'full'), ('main', 'lit'),
          ('covariates', 'full'), ('covariates', 'lit'))

ALPHA_FALLBACK = 0.05


def level_name(i):
    """1-based level -> 'L2 main/lit'. For logs and status columns."""
    if not (1 <= int(i) <= len(LEVELS)):
        return 'none'
    m, w = LEVELS[int(i) - 1]
    return 'L%d %s/%s' % (int(i), m, w)


def split_l2_key(key):
    """`l2_rows_json` key -> (model, window). 'session@lit' -> ('session','lit').

    The window must come off before the model is classified: testing the raw
    key against MAIN_MODELS would put 'session@lit' in the covariate family.
    """
    k = str(key)
    model, sep, window = k.partition('@')
    return model, (window if sep else 'full')


def _f(v, default=float('nan')):
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        x = float(s)
    except ValueError:
        return default
    return x if np.isfinite(x) else default


def row_qualifies(row, alpha=ALPHA_FALLBACK, qrs_edge=QRS_EDGE_MS):
    """(survived, clears_qrs) for one L2 row.

    `survived` falls back to perm_p < alpha only when the run wrote no
    `Survived` column. `alpha` is the run's, never a constant: it also sets the
    cluster-forming threshold, so a run at a different alpha has different
    clusters.
    """
    if not isinstance(row, dict):
        return False, False
    surv = row.get('Survived')
    if surv is None:
        surv = _f(row.get('perm_p'), 1.0) < alpha
    surv = bool(surv) if not isinstance(surv, str) else \
        str(surv).strip() in ('1', 'True', 'true')
    t0 = _f(row.get('Start_ms'))
    return surv, bool(np.isfinite(t0) and t0 >= qrs_edge)


def cascade_of_l2(l2_by_grain, alpha=ALPHA_FALLBACK, qrs_edge=QRS_EDGE_MS):
    """{grain: l2_rows_json dict} -> the cascade verdict for the run.

    Returns a dict:
        level        1..4, or 0 when every level is empty
        models_mode  'main' | 'covariates' | ''      (the stopping level's)
        window       'full' | 'lit' | ''
        hits         [(grain, model, window, row)]   qualifying, at that level
        demoted      [(grain, model, window, row)]   survived but inside the
                     QRS window, at any level: the run's artefact clusters
        n_survived   how many clusters survived in total, QRS or not

    `level=0` is a measured null, not a failure: every level was fitted and
    none produced a cluster clearing +50 ms.
    """
    surv_all, demoted = [], []
    by_level = {i: [] for i in range(1, len(LEVELS) + 1)}
    for grain, l2 in sorted((l2_by_grain or {}).items()):
        for key, rows in (l2 or {}).items():
            model, window = split_l2_key(key)
            fam = 'main' if model in MAIN_MODELS else 'covariates'
            for r in (rows or []):
                ok, clears = row_qualifies(r, alpha, qrs_edge)
                if not ok:
                    continue
                surv_all.append((grain, model, window, r))
                if not clears:
                    demoted.append((grain, model, window, r))
                    continue
                for i, (m, w) in enumerate(LEVELS, 1):
                    if fam == m and window == w:
                        by_level[i].append((grain, model, window, r))
                        break
    for i in range(1, len(LEVELS) + 1):
        if by_level[i]:
            m, w = LEVELS[i - 1]
            return {'level': i, 'models_mode': m, 'window': w,
                    'hits': by_level[i], 'demoted': demoted,
                    'n_survived': len(surv_all)}
    return {'level': 0, 'models_mode': '', 'window': '', 'hits': [],
            'demoted': demoted, 'n_survived': len(surv_all)}


def load_l2_by_grain(run_root, grains=('cell', 'segment', 'perbeat')):
    """{grain: l2_rows_json}, plus the run's own ALPHA. -> (dict, alpha).

    The alpha comes out of the run, not from a default. Reads each grain's
    `run_results.npz`; a grain with no npz is simply absent, which is what
    "that grain was not run" should look like.
    """
    out, alpha = {}, ALPHA_FALLBACK
    for g in tuple(grains):
        npz = os.path.join(run_root, '%s_wise' % g, 'main', 'run_results.npz')
        if not os.path.isfile(npz):
            continue
        try:
            z = np.load(npz, allow_pickle=False)
        except Exception:                                   # noqa: BLE001
            continue
        try:
            out[g] = json.loads(str(z['l2_rows_json']))
            alpha = float(json.loads(str(z['params_json'])).get('ALPHA', alpha))
        except Exception:                                   # noqa: BLE001
            pass
        finally:
            z.close()
    return out, alpha


def cascade_of_run(run_root, grains=('cell', 'segment', 'perbeat'),
                   qrs_edge=QRS_EDGE_MS):
    """The cascade verdict for a finished run folder. -> the cascade_of_l2 dict."""
    l2, alpha = load_l2_by_grain(run_root, grains)
    v = cascade_of_l2(l2, alpha=alpha, qrs_edge=qrs_edge)
    v['alpha'] = alpha
    v['grains_read'] = sorted(l2)
    return v
