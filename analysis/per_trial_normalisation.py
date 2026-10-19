# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quality of the component chosen with the per-trial normalisation of Lee et al. (2024)
against the paper's rule (absolute power).

The per-trial picks come from the cohort build of the primary run with only the selection
normalisation changed (IC_SEL_NORM='ztrial'; nothing is fitted); the paper's picks are read
from the primary run's cohort_sessions.csv. Per pick: ICLabel Brain probability (against the
pool of all components), whether Brain is the top class, percent variance accounted for, and
the variance percentile within the recording; per participant with two recordings, |r|
between the two picks' scalp maps. The ICLabel probabilities are read from config.ICLABELS
(written by functions/extract_npz.py). Prints only; the numbers are those of the Methods
paragraph on the per-trial normalisation.

    python analysis/per_trial_normalisation.py
"""
import csv
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                      # noqa: E402
import component_quality as Q                                     # noqa: E402
from heart_depth_run import build_cohort                       # noqa: E402
from run_component_group_tests import run_root                     # noqa: E402


def topo(sid, ses, ic):
    with np.load(os.path.join(config.DATA_DIR, str(sid), 'ses%d.npz' % int(ses)),
                 allow_pickle=True) as z:
        return np.asarray(z['icawinv'], float)[:, int(ic)]


def describe(label, picks, labels):
    """picks: {(participant, session): component}."""
    bp, top, pv, pct, pool = [], [], [], [], []
    for (sid, ses), ic in sorted(picks.items()):
        st = Q.session_ic_stats(config.DATA_DIR, sid, ses, labels)
        if st is None or ic >= st['n_ic']:
            continue
        bp.append(st['brain_p'][ic])
        top.append(st['argmax'][ic] == 0)
        pv.append(st['pvaf'][ic])
        pct.append(100.0 * (st['pvaf'] <= st['pvaf'][ic]).mean())
        pool += st['brain_p'].tolist()
    bp, pv = np.array(bp), np.array(pv)
    by = {}
    for (sid, ses), ic in picks.items():
        by.setdefault(sid, []).append((ses, ic))
    rr = np.array([abs(np.corrcoef(topo(sid, *v[0]), topo(sid, *v[1]))[0, 1])
                   for sid, v in by.items() if len(v) == 2])
    print('%-8s %d picks | Brain probability median %.2f (all components %.2f) | Brain top '
          'class %.0f%% | variance accounted for median %.2f%%, < 1%% in %.0f%% | variance '
          'percentile median %.0f | |r| between a participant\'s two picks median %.2f, > .7 in '
          '%.0f%% (%d pairs)' % (label, bp.size, np.median(bp), np.median(pool),
                                 100 * np.mean(top), np.median(pv), 100 * np.mean(pv < 1),
                                 np.median(pct), np.median(rr), 100 * np.mean(rr > .7), rr.size))


def main():
    main_dir = os.path.join(run_root(config.COMPONENT_ROOT,
                                     config.COMPONENT_SETTINGS[config.PRIMARY]),
                            'perbeat_wise', 'main')
    with np.load(os.path.join(main_dir, 'run_results.npz'), allow_pickle=False) as z:
        p = json.loads(str(z['params_json']))
    raw = {(r['subject'], int(r['session'])): int(r['ic'])
           for r in csv.DictReader(io.open(os.path.join(main_dir, 'cohort_sessions.csv'),
                                           encoding='utf-8-sig')) if r['status'] == 'kept'}
    q = dict(p, IC_SEL_NORM='ztrial', KEEP_PERBEAT=False, KEEP_SEGCOV=False)
    _sess, recs, _subs = build_cohort(config.DATA_DIR, q, limit=None, verbose=False)
    zt = {(str(r['subject']), int(r['session'])): int(r['ic'])
          for r in recs if r['status'] == 'kept'}
    if set(zt) != set(raw):
        print('kept recordings differ: per-trial rule %d, absolute-power rule %d'
              % (len(zt), len(raw)))
    print('same component under both rules in %d of %d recordings'
          % (sum(1 for k in raw if zt.get(k) == raw[k]), len(raw)))
    labels = np.load(config.ICLABELS)
    describe('per-trial', zt, labels)
    describe('absolute', raw, labels)


if __name__ == '__main__':
    main()
