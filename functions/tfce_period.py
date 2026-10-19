# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
tfce_period: weighted TFCE (l2_tfce, convention 'sn') as the group test of the period
arm (heart_period: HEP / HRSP / HEPC contrasted between task periods), in place of its
cluster-mass test. HEPC, as the code names it, is the paper's HRPC (heartbeat-related
phase consistency).

`main(argv)` is called by run_period_contrasts.py (step heart) with

    <run_dir> [<run_dir> ...]
        [--measures hrsp hep hepc] [--schemes ivw_own unweighted]
        [--nperm 5000] [--jobs 4]

At most 6 worker processes are used, whatever --jobs asks for.

<run_dir> is a heart_period output folder (holds one sub-folder per
contrast, each with main/period_maps.npz: the per-participant delta maps
delta_<meas> [nSub x (nF x) nT], their variance svar_<meas> and the beat
counts nbeats_<meas>).

Per contrast x measure x scheme: participants stacked through
l2_tfce.prepare_measure (listwise-finite; weights by l2_heart:
unweighted, precision_nseg on the beat count, ivw_own from the participant's
own delta variance, shrunk by the median), sign-flip max-|TFCE| null, per-cell
FWE p. The unweighted observed t-map is checked against a plain one-sample t.

Writes <run_dir>/period_tfce_all.csv (one row per significant component, or one
no_sig row per map) and <run_dir>/<contrast>/main/period_tfce.npz (t, tfce, p
per measure x scheme).

Six contrasts over four period means; nothing is corrected across them.
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse                                                     # noqa: E402
import csv                                                          # noqa: E402
import io                                                           # noqa: E402
import sys                                                          # noqa: E402
import time                                                         # noqa: E402

import numpy as np                                                  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from l2_heart import SEED                                       # noqa: E402
from l2_tfce import (tfce_perm_weighted, component_rows,        # noqa: E402
                         prepare_measure, TFCE_FIELDS)

FIELDS = ('Contrast', 'Label') + TFCE_FIELDS


def subjects_of(z, meas):
    """[{subject, meas: (delta, svar, nbeats)}] for participants with a map."""
    d, v, nb = z['delta_' + meas], z['svar_' + meas], z['nbeats_' + meas]
    out = []
    for i, sid in enumerate(z['subject_ids']):
        if not np.isfinite(d[i]).any() or nb[i] <= 0:
            continue
        out.append({'subject': str(sid), meas: (d[i], v[i], int(nb[i]))})
    return out


def run_task(task):
    t0 = time.time()
    z = np.load(task['npz'], allow_pickle=False)
    meas, scheme = task['measure'], task['scheme']
    if 'delta_' + meas not in z.files:
        return None
    times = np.asarray(z['times'], float)
    freqs = None if meas == 'hep' else np.asarray(z['freqs'], float)
    prep = prepare_measure(subjects_of(z, meas), meas, scheme, None)
    if prep is None:
        return None
    Xk, w, n_drop, _kept = prep
    X3 = Xk if Xk.ndim == 3 else Xk[None]                 # HEP as [1 x nT x nSub]
    # Units. The period arm does not normalise the component to unit variance, so
    # HRSP differences are ~1e-14 and their variance ~1e-27; the L2 code clips a
    # variance at 1e-30 (l2_heart._null_t_weighted), which cells of a
    # subtraction-baseline map can fall below, giving a wrong t there. The t-test is
    # scale-free, so the stack is divided by one scalar; the weights were already
    # built from the unscaled variances and are scale-free too (1 / (v + median v),
    # mean 1).
    X3 = X3 / float(np.nanmedian(np.abs(X3)))
    t, tf, p, _nm, n_used = tfce_perm_weighted(X3, w, task['nperm'], SEED)
    if scheme == 'unweighted':
        n = X3.shape[-1]
        t_hand = X3.mean(-1) / (X3.std(-1, ddof=1) / np.sqrt(n))
        # relative: a cell with a tiny variance carries a large t, and the two
        # arithmetic routes then differ in the last digits of a big number
        rel = np.abs(t_hand - np.atleast_2d(t)) / np.maximum(1.0, np.abs(t_hand))
        if np.max(rel) > 1e-6:
            raise RuntimeError('%s %s: unweighted t is not the plain one-sample t '
                               '(max relative diff %.2e)'
                               % (task['contrast'], meas, np.max(rel)))
    meta = dict(Model='period', Measure=meas, Scheme=scheme, Convention='sn',
                N_dropped_nonfinite=n_drop, nPerm=task['nperm'])
    rows, _c = component_rows(meta, t, tf, p, times, freqs, w, n_used)
    for r in rows:
        r.update(Contrast=task['contrast'], Label=str(z['label']))
    return dict(task=task, rows=rows, t=np.atleast_2d(t), tf=np.atleast_2d(tf),
                p=np.atleast_2d(p), elapsed=time.time() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description='weighted TFCE on a period run')
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--measures', nargs='+', default=['hrsp', 'hep', 'hepc'])
    ap.add_argument('--schemes', nargs='+', default=['ivw_own', 'unweighted'])
    ap.add_argument('--nperm', type=int, default=5000)
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args(argv)
    for rd in a.run_dirs:
        contrasts = sorted(c for c in os.listdir(rd)
                           if os.path.isfile(os.path.join(rd, c, 'main', 'period_maps.npz')))
        if not contrasts:
            raise SystemExit('no <contrast>/main/period_maps.npz under %s' % rd)
        tasks = [dict(npz=os.path.join(rd, c, 'main', 'period_maps.npz'), contrast=c,
                      measure=m, scheme=s, nperm=a.nperm)
                 for c in contrasts for m in a.measures for s in a.schemes]
        print('%s\n  %d contrast(s) x %s x %s, nPerm %d' % (rd, len(contrasts), a.measures,
                                                          a.schemes, a.nperm), flush=True)
        from concurrent.futures import ProcessPoolExecutor
        res = []
        with ProcessPoolExecutor(max_workers=min(a.jobs, 6)) as ex:
            for r in ex.map(run_task, tasks):
                if r is None:
                    continue
                res.append(r)
                r0, tk = r['rows'][0], r['task']
                print('  %-22s %-5s %-10s n %s (eff %s) min p %-7s sig %-5s %s..%s ms %s..%s Hz '
                      'preR %s g %s [%.0fs]'
                      % (tk['contrast'], tk['measure'], tk['scheme'], r0['N_subjects'],
                         r0['N_eff_kish'], r0['Best_p'], r0['N_sig_total'], r0['Start_ms'],
                         r0['End_ms'], r0['Start_Hz'], r0['End_Hz'], r0['N_cells_preR'],
                         r0['HedgesG'], r['elapsed']), flush=True)
        with io.open(os.path.join(rd, 'period_tfce_all.csv'), 'w', encoding='utf-8',
                     newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(FIELDS), extrasaction='ignore')
            w.writeheader()
            for r in res:
                w.writerows(r['rows'])
        for c in contrasts:
            store = {}
            for r in res:
                tk = r['task']
                if tk['contrast'] != c:
                    continue
                key = '%s__%s' % (tk['measure'], tk['scheme'])
                store['t__' + key] = r['t'].astype(np.float32)
                store['tfce__' + key] = r['tf'].astype(np.float32)
                store['p__' + key] = r['p'].astype(np.float32)
            np.savez_compressed(os.path.join(rd, c, 'main', 'period_tfce.npz'), **store)
        print('  wrote %s' % os.path.join(rd, 'period_tfce_all.csv'), flush=True)
    return 0
