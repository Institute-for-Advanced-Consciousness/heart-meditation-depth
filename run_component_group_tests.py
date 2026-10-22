# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Analysis 2, group level: weighted TFCE, leave one participant out, surrogate heartbeats.

Runs on the participant-level maps written by run_component_analysis.py:

  tfce       weighted threshold-free cluster enhancement on each of the nine settings
             (precision weights and unweighted; HEP, HRSP, HRPC)
  confounds  the same test under each added confound regressor (primary setting)
  loso       the primary test repeated with each participant left out in turn
  surrogate  the primary analysis repeated on shifted R-peak trains
  subsets    the primary test in subsets of participants defined by the rating of EEG
             viability (needs config.VIABILITY_RATINGS and the tfce step)
  sources    eLORETA maps of the components the primary analysis selected, and their regions
  ongoing    the selected component's ongoing beta power against depth

    python run_component_group_tests.py                       # all seven steps
    python run_component_group_tests.py --steps tfce --only dur60_bl_silent
    python run_component_group_tests.py --nperm 200 --n-surrogate 3 --limit 5   # smoke test

Tables are written inside each run folder: <grain>_wise/main/l2_tfce_summary_<model>.csv,
l2_tfce_loso_<model>.csv, and at the run root surrogate_control_tfce.csv,
source_localization/ (component_source_maps.npz, source_localization.json) and
ongoing_power_<band>.json.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                     # noqa: E402
import heart_surrogate                                        # noqa: E402
import ongoing_power                                              # noqa: E402
import participant_subsets                                    # noqa: E402
import source_localization                                        # noqa: E402
import tfce_loso                                              # noqa: E402
import tfce_run                                               # noqa: E402
from heart_depth_run import DEFAULT_RUN_PARAMS, run_paths    # noqa: E402

STEPS = ('tfce', 'confounds', 'loso', 'surrogate', 'subsets', 'sources', 'ongoing')



def _check(rc):
    """Stop when a group test reports a failure (a non-zero return)."""
    if rc:
        raise SystemExit(rc)

def run_root(results_root, over):
    """The folder run_component_analysis.py wrote for these parameters."""
    return run_paths(results_root, dict(DEFAULT_RUN_PARAMS, **over), create=False)[0]


def step_tfce(a):
    t = config.TFCE
    for name, over in config.COMPONENT_SETTINGS.items():
        if a.only and name not in a.only:
            continue
        print('\n== TFCE: %s' % name, flush=True)
        _check(tfce_run.main([run_root(config.COMPONENT_ROOT, over),
                                  '--grains', *t['grains'], '--models', *t['models'],
                                  '--measures', *t['measures'], '--schemes', *t['schemes'],
                                  '--nperm', str(a.nperm or t['nperm']), '--jobs', str(a.jobs)]))


def step_confounds(a):
    t = config.TFCE
    for name, over in config.CONFOUND_SETTINGS.items():
        print('\n== TFCE under each confound regressor: %s' % name, flush=True)
        _check(tfce_run.main([run_root(config.CONFOUND_ROOTS[name], over),
                                  '--grains', 'perbeat', '--models', 'all', '--measures', 'hrsp',
                                  '--schemes', *t['schemes'],
                                  '--nperm', str(a.nperm or t['nperm']), '--jobs', str(a.jobs)]))


def step_loso(a):
    c = config.LOSO
    print('\n== leave one participant out', flush=True)
    root = run_root(config.COMPONENT_ROOT, config.COMPONENT_SETTINGS[config.PRIMARY])
    _check(tfce_loso.main([root, '--grain', c['grain'], '--model', c['model'],
                           '--measure', c['measure'], '--schemes', *c['schemes'],
                           '--nperm', str(a.nperm or c['nperm']), '--jobs', str(a.jobs)]))


def step_surrogate(a):
    c = config.SURROGATE
    over = dict(config.COMPONENT_SETTINGS[config.PRIMARY], KEEP_PERBEAT=True)
    cells = [dict(grain=c['grain'], measure=c['measure'], model=c['model'], scheme=s,
                  window='full') for s in c['schemes']]
    out = os.path.join(run_root(config.COMPONENT_ROOT, over), 'surrogate_control_tfce.csv')
    argv = ['--data', config.DATA_DIR, '--results-root', config.COMPONENT_ROOT,
            '--params', json.dumps(over),
            '--n-surrogate', str(a.n_surrogate or c['n_surrogate']),
            '--max-unshiftable', str(c['max_unshiftable']), '--stat', c['stat'],
            '--tfce-nperm', str(a.nperm or c['tfce_nperm']),
            '--cells', json.dumps(cells), '--out', out]
    if a.limit:
        argv += ['--limit', str(a.limit)]
    print('\n== surrogate heartbeat control', flush=True)
    _check(heart_surrogate.main(argv))


def step_subsets(a):
    c = config.LOSO
    root = run_root(config.COMPONENT_ROOT, config.COMPONENT_SETTINGS[config.PRIMARY])
    print('\n== subsets of participants by EEG-viability rating', flush=True)
    participant_subsets.viability_subsets(
        os.path.join(root, '%s_wise' % c['grain'], 'main'), config.VIABILITY_RATINGS,
        os.path.join(root, 'viability_subsets.csv'), model=c['model'], measure=c['measure'],
        n_perm=a.nperm or c['nperm'])


def primary_records():
    """Root of the primary run and the session records it wrote (one selected component per
    kept recording)."""
    root = run_root(config.COMPONENT_ROOT, config.COMPONENT_SETTINGS[config.PRIMARY])
    z = np.load(os.path.join(root, 'perbeat_wise', 'main', 'run_results.npz'), allow_pickle=True)
    return root, json.loads(str(z['session_records_json']))


def step_sources(a):
    root, records = primary_records()
    s = source_localization.localize(records, config.DATA_DIR,
                                     os.path.join(root, 'source_localization'),
                                     config.SOURCES['cache_dir'])
    print('mean map: peak at MNI %s (%s); most power per component: %s; label stable across '
          'regularisation for %.0f%%' % (
              [round(v) for v in s['mean_map_peak_mni']], s['mean_map_peak_region'],
              ', '.join('%s %d' % (k, v) for k, v in s['total_rule_counts'][:4]),
              100 * s['total_label_stable_share']))


def step_ongoing(a):
    root, records = primary_records()
    c = config.ONGOING
    params = dict(DEFAULT_RUN_PARAMS, **config.COMPONENT_SETTINGS[config.PRIMARY])
    slopes = ongoing_power.depth_slopes(config.DATA_DIR, params, records, band=c['band'])
    g = ongoing_power.group_test(list(slopes.values()), n_perm=c['n_perm'])
    with open(os.path.join(root, 'ongoing_power_%s.json' % c['band']), 'w',
              encoding='utf-8') as fh:
        json.dump(dict(band=c['band'], group=g, slopes=slopes), fh, indent=1)
    print('ongoing %s power against depth: n %d, t %.2f, p %.4f, sign-flip p %.4f, g %.2f' % (
        c['band'], g['n'], g['t'], g['p'], g['p_signflip'], g['hedges_g']))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--steps', nargs='+', default=list(STEPS), choices=STEPS)
    ap.add_argument('--only', nargs='+', default=None,
                    help='settings for the tfce step (default: all eight)')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--nperm', type=int, default=None,
                    help='permutations, overriding config (smoke tests only)')
    ap.add_argument('--n-surrogate', type=int, default=None,
                    help='surrogate draws, overriding config (smoke tests only)')
    ap.add_argument('--limit', type=int, default=None,
                    help='surrogate on the first N participants (smoke tests only)')
    a = ap.parse_args(argv)
    if a.only:
        unknown = sorted(set(a.only) - set(config.COMPONENT_SETTINGS))
        if unknown:
            ap.error('unknown setting(s) %s; known: %s'
                     % (unknown, list(config.COMPONENT_SETTINGS)))
    for step in STEPS:
        if step in a.steps:
            {'tfce': step_tfce, 'confounds': step_confounds, 'loso': step_loso,
             'surrogate': step_surrogate, 'subsets': step_subsets,
             'sources': step_sources, 'ongoing': step_ongoing}[step](a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
