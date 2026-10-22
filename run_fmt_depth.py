# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analysis 3: frontal-midline theta as a function of reported depth.

  fit      theta power per segment (frontocentral scalp mean and GED component), participant
           slopes on depth and the sign-flip group test, for the silent blocks alone, for
           the silent and SAPO blocks pooled, and for the silent blocks with 30 s windows
  weights  the same group test with each participant weighted by the precision of their own
           slope, the weighting the paper reports
  associations  skipped correlations between each participant's theta depth slope and the
           34 self-report measures (config.FMT_ASSOCIATIONS; needs config.BEHAVIOUR_DIR)

    python run_fmt_depth.py
    python run_fmt_depth.py --steps weights associations

Results go to <results>/fmt/FMT_DEPTH/<run folder>/main/ (fmt_depth_L2.csv, run_results.npz),
<results>/fmt/fmt_depth_weighted.csv and <results>/fmt/fmt_outcome_skipped_correlations.csv.
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                     # noqa: E402
import fmt_weights                                            # noqa: E402
import outcome_associations as OA                             # noqa: E402

STEPS = ('fit', 'weights', 'associations')
# The FMT depth pipeline is a script (it runs when executed), so it is launched, not imported.
SCRIPT = os.path.join(ROOT, 'functions', 'fmt_depth_npz.py')


def fmt_runs():
    """Folders of the FMT depth runs on disk (the ones holding main/)."""
    return sorted(os.path.dirname(os.path.dirname(f)) for f in glob.glob(
        os.path.join(config.FMT_ROOT, 'FMT_DEPTH', '*', 'main', 'run_results.npz')))


def run_of_setting(name):
    """The run folder of one entry of config.FMT_DEPTH_SETTINGS."""
    want = config.FMT_DEPTH_SETTINGS[name]
    hits = []
    for run in fmt_runs():
        with io.open(os.path.join(run, 'main', 'fmt_results.json'), encoding='utf-8') as fh:
            params = json.load(fh)['params']
        if all(params.get(k) == v for k, v in want.items()):
            hits.append(run)
    if len(hits) != 1:
        raise SystemExit('%d FMT depth runs match the %r setting under %s'
                         % (len(hits), name, config.FMT_ROOT))
    return hits[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--steps', nargs='+', default=list(STEPS), choices=STEPS)
    ap.add_argument('--only', nargs='+', default=None, choices=list(config.FMT_DEPTH_SETTINGS))
    a = ap.parse_args(argv)
    if 'fit' in a.steps:
        for name, p in config.FMT_DEPTH_SETTINGS.items():
            if a.only and name not in a.only:
                continue
            print('\n' + '=' * 78 + '\nFMT depth: %s blocks\n' % name + '=' * 78, flush=True)
            cmd = [sys.executable, SCRIPT, '--data-dir', config.DATA_DIR,
                   '--results-root', config.FMT_ROOT]
            for k, v in p.items():
                cmd += ['--set', '%s=%r' % (k, v)]
            subprocess.run(cmd, check=True)
    if 'weights' in a.steps:
        runs = fmt_runs()
        if not runs:
            raise SystemExit('no FMT depth run under %s: run the fit step first' % config.FMT_ROOT)
        fmt_weights.write_table(runs, os.path.join(config.FMT_ROOT, 'fmt_depth_weighted.csv'))
    if 'associations' in a.steps:
        c = config.FMT_ASSOCIATIONS
        run = run_of_setting(c['setting'])
        sids, x = OA.theta_slope_predictor(os.path.join(run, 'main'), c['measure'])
        print('predictor: depth slope of the %s theta measure, %s blocks, %d participants'
              % (c['measure'], c['setting'], sum(v == v for v in x)))
        rows, _Y = OA.skipped_correlation_table(sids, x, config.BEHAVIOUR_DIR,
                                                n_boot=c['n_boot'])
        out = os.path.join(config.FMT_ROOT, 'fmt_outcome_skipped_correlations.csv')
        OA.write_table(rows, out)
        OA.summarise(rows)
        print('wrote', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
