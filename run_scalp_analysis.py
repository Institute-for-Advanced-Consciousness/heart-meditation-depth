# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analysis 1: HEP, HRSP and HRPC at all 64 electrodes, with no component selection.

  fit    participant-level depth slopes at every channel, for the eight settings of
         config.SCALP_SETTINGS (window length x baseline correction x block set)
  tfce   weighted threshold-free cluster enhancement over channel x frequency x time

    python run_scalp_analysis.py                              # both steps, eight settings
    python run_scalp_analysis.py --only dur60_bl_silent
    python run_scalp_analysis.py --limit 4 --nperm 50         # smoke test, not a result

Results go to <results>/scalp/HEART_DEPTH_SCALP/<setting folder>/main/; the group table is
l2_scalp_tfce_ivw_own_summary.csv.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                     # noqa: E402
import tfce_scalp                                             # noqa: E402
from heart_scalp_pipeline import DEFAULT_SCALP_PARAMS, run_scalp   # noqa: E402
from result_path import heart_depth_tag, result_dir          # noqa: E402

STEPS = ('fit', 'tfce')



def _check(rc):
    """Stop when a group test reports a failure (a non-zero return)."""
    if rc:
        raise SystemExit(rc)

def params_of(name, jobs=1):
    return dict(DEFAULT_SCALP_PARAMS, **config.SCALP_SETTINGS[name], N_JOBS=int(jobs))


def run_root(name):
    """The folder the fit step writes for this setting."""
    main_dir = result_dir(config.SCALP_ROOT, 'HEART_DEPTH_SCALP',
                          tag=heart_depth_tag(params_of(name)), section='main', create=False)
    return os.path.dirname(main_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--steps', nargs='+', default=list(STEPS), choices=STEPS)
    ap.add_argument('--only', nargs='+', default=None, help='settings (default: all eight)')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None,
                    help='first N participants only (a smoke test, not a result)')
    ap.add_argument('--nperm', type=int, default=None,
                    help='TFCE permutations, overriding config (smoke tests only)')
    a = ap.parse_args(argv)
    names = list(config.SCALP_SETTINGS)
    if a.only:
        unknown = sorted(set(a.only) - set(names))
        if unknown:
            ap.error('unknown setting(s) %s; known: %s' % (unknown, names))
        names = [n for n in names if n in a.only]
    t = config.SCALP_TFCE
    for name in names:
        if 'fit' in a.steps:
            print('\n' + '=' * 78 + '\nscalp fit: %s\n' % name + '=' * 78, flush=True)
            run_scalp(config.DATA_DIR, config.SCALP_ROOT, params_of(name, a.jobs), limit=a.limit)
        if 'tfce' in a.steps:
            print('\n== scalp TFCE: %s' % name, flush=True)
            _check(tfce_scalp.main([run_root(name), '--scheme', t['scheme'],
                                        '--measures', *t['measures'], '--models', *t['models'],
                                        '--nperm', str(a.nperm or t['nperm'][name]),
                                        '--jobs', str(a.jobs)]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
