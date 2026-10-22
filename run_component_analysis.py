# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analysis 2, participant level: component selection, HEP / HRSP / HRPC, depth slopes.

For each of the nine settings of config.COMPONENT_SETTINGS (window length x baseline
correction x block set, and 30 s windows in the silent blocks with baseline correction) and
for the two confound runs, this selects one heartbeat-related component per recording,
computes the three measures and fits each participant's depth slope map. Results go to <results>/component/HEART_DEPTH/<setting folder>/ and
<results>/confounds_{component,channels}/HEART_DEPTH/<setting folder>/.

The group tests on these maps are run by run_component_group_tests.py.

    python run_component_analysis.py                      # everything
    python run_component_analysis.py --only dur60_bl_silent
    python run_component_analysis.py --limit 3            # first 3 participants: a smoke test
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                     # noqa: E402
from heart_depth_run import DEFAULT_RUN_PARAMS, run          # noqa: E402


def jobs():
    """(name, results root, parameters) of every run, the primary setting first."""
    out = [(name, config.COMPONENT_ROOT, p) for name, p in config.COMPONENT_SETTINGS.items()]
    out += [('confounds_' + name, config.CONFOUND_ROOTS[name], p)
            for name, p in config.CONFOUND_SETTINGS.items()]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', nargs='+', default=None, help='names of the runs to do')
    ap.add_argument('--limit', type=int, default=None,
                    help='first N participants only (a smoke test, not a result)')
    a = ap.parse_args(argv)
    todo = jobs()
    if a.only:
        unknown = sorted(set(a.only) - {n for n, _r, _p in todo})
        if unknown:
            ap.error('unknown run(s) %s; known: %s' % (unknown, [n for n, _r, _p in todo]))
        todo = [j for j in todo if j[0] in a.only]
    for name, root, over in todo:
        print('\n' + '=' * 78 + '\n%s\n' % name + '=' * 78, flush=True)
        run(config.DATA_DIR, root, dict(DEFAULT_RUN_PARAMS, **over), limit=a.limit)
    return 0


if __name__ == '__main__':
    sys.exit(main())
