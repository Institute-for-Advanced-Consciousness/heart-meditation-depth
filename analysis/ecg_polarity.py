# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""How many recordings have an inverted ECG trace.

Polarity is resolved per recording with the R-peak detector's own rule
(get_rr.resolve_polarity) on the concatenated depth-report segments of that recording, as
the analysis does. Prints the count over the recordings the primary analysis kept and over
all recordings with segments; writes nothing.

    python analysis/ecg_polarity.py
"""
import csv
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                      # noqa: E402
from get_rr import resolve_polarity                            # noqa: E402
from heart_depth_pipeline import list_sessions                     # noqa: E402
from run_component_group_tests import run_root                     # noqa: E402


def main():
    root = run_root(config.COMPONENT_ROOT, config.COMPONENT_SETTINGS[config.PRIMARY])
    cohort = os.path.join(root, 'perbeat_wise', 'main', 'cohort_sessions.csv')
    kept = {(r['subject'], int(r['session'])) for r in csv.DictReader(
        io.open(cohort, encoding='utf-8-sig')) if r['status'] == 'kept'}
    n = dict(kept=[0, 0], all=[0, 0])                    # [inverted, total]
    for sid, ses, fp in list_sessions(config.DATA_DIR):
        with np.load(fp, allow_pickle=True) as d:
            ecg, ln = d['seg_ecg'], np.atleast_1d(d['seg_len']).astype(int)
            fs = float(d['fs'])
        if ecg.shape[0] == 0:
            continue
        cat = np.concatenate([np.asarray(ecg[i, :ln[i] if ln.size > i else None], float)
                              for i in range(ecg.shape[0])])
        inv = int(resolve_polarity(cat[np.isfinite(cat)], fs) < 0)
        for key in ['all'] + (['kept'] if (str(sid), int(ses)) in kept else []):
            n[key][0] += inv
            n[key][1] += 1
    print('analysed recordings: %d of %d inverted' % tuple(n['kept']))
    print('all recordings with segments: %d of %d inverted' % tuple(n['all']))


if __name__ == '__main__':
    main()
