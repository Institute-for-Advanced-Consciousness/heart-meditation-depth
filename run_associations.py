# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-report and demographic associations of the primary depth effect.

Needs the primary run with its TFCE tables (run_component_analysis.py, then
run_component_group_tests.py --steps tfce) and the questionnaire tables in config.BEHAVIOUR_DIR.

  questionnaires  skipped correlations between each participant's depth slope and the 34
                  self-report measures (one coefficient per measure, bootstrap CI and p,
                  false discovery rate within questionnaire)
  demographics    Spearman correlations with the six demographic variables

    python run_associations.py

Writes <results>/associations/outcome_skipped_correlations.csv; the demographic table is
written inside the primary run folder (perbeat_wise/outcomes_demographics_tfce/).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                     # noqa: E402
import heart_posthoc                                          # noqa: E402
import outcome_associations as OA                             # noqa: E402
from heart_depth_run import DEFAULT_RUN_PARAMS, run_paths    # noqa: E402

STEPS = ('questionnaires', 'demographics')


def primary_run_root():
    over = config.COMPONENT_SETTINGS[config.PRIMARY]
    return run_paths(config.COMPONENT_ROOT, dict(DEFAULT_RUN_PARAMS, **over), create=False)[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--steps', nargs='+', default=list(STEPS), choices=STEPS)
    a = ap.parse_args(argv)
    c = config.ASSOCIATIONS
    run_root = primary_run_root()
    if 'questionnaires' in a.steps:
        main_dir = os.path.join(run_root, '%s_wise' % c['grain'], 'main')
        sids, x, n_cells = OA.depth_slope_predictor(main_dir, c['model'], c['measure'],
                                                    c['scheme'])
        print('predictor: mean %s slope of the %s model over %d significant points, %d '
              'participants' % (c['measure'].upper(), c['model'], n_cells, len(sids)))
        rows, Y = OA.skipped_correlation_table(sids, x, config.BEHAVIOUR_DIR, n_boot=c['n_boot'])
        out = os.path.join(config.ASSOCIATIONS_DIR, 'outcome_skipped_correlations.csv')
        OA.write_table(rows, out)
        OA.summarise(rows)
        r, n = OA.scale_correlation(Y, 'MEDI_Relaxation', 'MEDI_Hindrances')
        print('MEDI Relaxation vs MEDI Hindrances across participants: r = %.2f (n = %d)' % (r, n))
        print('wrote', out)
    if 'demographics' in a.steps:
        heart_posthoc.main(['--run', run_root, '--beh', config.BEHAVIOUR_DIR, '--tfce',
                                '--tfce-model', c['model'], '--tfce-grains', c['grain'],
                                '--tfce-schemes', c['scheme']])
    return 0


if __name__ == '__main__':
    sys.exit(main())
