# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analysis 4: contrasts between the four recording periods.

Baseline, pre-meditation, post-meditation and the active oddball task are compared pairwise
(six contrasts, uncorrected across them) for

  heart   HEP, HRSP and HRPC in one component per recording, selected on the baseline and
          pre-meditation periods only; then weighted TFCE on the period-difference maps
  fmt     frontal-midline theta (frontocentral scalp mean and GED component)

    python run_period_contrasts.py
    python run_period_contrasts.py --steps heart
    python run_period_contrasts.py --limit 6 --nperm 50       # smoke test, not a result

Results go to <results>/periods/HEART_PERIOD/<run folder>/ (period_tfce_all.csv) and
<results>/periods/FMT_PERIOD/<run folder>/main/ (fmt_period_L2.csv).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                     # noqa: E402
import fmt_period                                             # noqa: E402
import heart_period                                           # noqa: E402
import tfce_period                                            # noqa: E402

STEPS = ('heart', 'fmt')



def _check(rc):
    """Stop when a group test reports a failure (a non-zero return)."""
    if rc:
        raise SystemExit(rc)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--steps', nargs='+', default=list(STEPS), choices=STEPS)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None,
                    help='first N participants only (a smoke test, not a result)')
    ap.add_argument('--nperm', type=int, default=None,
                    help='TFCE permutations, overriding config (smoke tests only)')
    a = ap.parse_args(argv)
    if 'heart' in a.steps:
        print('\n' + '=' * 78 + '\nperiod contrasts: heartbeat-related measures\n' + '=' * 78,
              flush=True)
        out = heart_period.run(config.DATA_DIR, config.PERIOD_ROOT,
                                   dict(config.PERIOD_HEART), limit=a.limit)
        t = config.PERIOD_HEART_TFCE
        _check(tfce_period.main([out['out_dir'], '--measures', *t['measures'],
                                 '--schemes', *t['schemes'], '--nperm', str(a.nperm or t['nperm']),
                                 '--jobs', str(a.jobs)]))
    if 'fmt' in a.steps:
        print('\n' + '=' * 78 + '\nperiod contrasts: frontal-midline theta\n' + '=' * 78,
              flush=True)
        fmt_period.run(config.DATA_DIR, config.PERIOD_ROOT, dict(config.PERIOD_FMT),
                           limit=a.limit)
    return 0


if __name__ == '__main__':
    sys.exit(main())
