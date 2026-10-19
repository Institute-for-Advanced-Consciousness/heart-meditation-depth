# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Supplementary Figure 3: depth slope against the eight self-report measures with the smallest p.

Each point is a participant. Red crosses are the bivariate outliers removed before the skipped
correlation; the black line is the least-squares fit to the remaining participants and the red
dashed line the fit to all participants. Reads the table written by run_associations.py and
the participant depth slopes of the primary run.

    python figures/supplementary_figure3_questionnaires.py
"""
import csv
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                      # noqa: E402
import outcome_associations as OA                              # noqa: E402
import outcomes as OC                                          # noqa: E402
import robust_corr as RC                                       # noqa: E402
from heart_depth_run import DEFAULT_RUN_PARAMS, run_paths     # noqa: E402

LABEL = {'Trait_FFMQ_Observing': 'FFMQ Observing',
         'Change_POMS_Depression': 'POMS Depression, post minus pre',
         'Task_TonesDistracting': 'Tones distracting (1-10)', 'MEDI_Hindrances': 'MEDI Hindrances',
         'MEDI_Relaxation': 'MEDI Relaxation', 'Task_GongReliability': 'Gong helped (1-10)',
         'Trait_FFMQ_Describing': 'FFMQ Describing', 'MEDI_PersonalSelf': 'MEDI Personal Self'}


def main():
    c = config.ASSOCIATIONS
    run_root = run_paths(config.COMPONENT_ROOT,
                         dict(DEFAULT_RUN_PARAMS, **config.COMPONENT_SETTINGS[config.PRIMARY]),
                         create=False)[0]
    sids, x, _n = OA.depth_slope_predictor(
        os.path.join(run_root, '%s_wise' % c['grain'], 'main'), c['model'], c['measure'],
        c['scheme'])
    Y, _f, _m, _g = OC.load_outcomes(config.BEHAVIOUR_DIR, sids, return_gaps=True)
    names = [s[0] for s in OC.OUTCOME_SPEC]
    table = os.path.join(config.ASSOCIATIONS_DIR, 'outcome_skipped_correlations.csv')
    tab = sorted(csv.DictReader(io.open(table, encoding='utf-8')),
                 key=lambda r: float(r['Rule_p']))[:8]
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8))
    for ax, t in zip(axes.ravel(), tab):
        j = names.index(t['Scale'])
        ok = np.isfinite(x) & np.isfinite(Y[:, j])
        xs, ys = x[ok], Y[ok, j]
        r = RC.skipped_correlation(xs, ys, n_boot=200, seed=j)
        out = r['outliers']
        assert int(out.sum()) == int(t['N_outliers'])
        ax.scatter(xs[~out], ys[~out], s=18, color='#1f5fa8', alpha=.85, edgecolor='none',
                   label='kept (%d)' % (~out).sum())
        ax.scatter(xs[out], ys[out], s=34, marker='x', color='#c0392b', lw=1.4,
                   label='removed (%d)' % out.sum())
        b = np.polyfit(xs[~out], ys[~out], 1)
        xx = np.array([xs[~out].min(), xs[~out].max()])
        ax.plot(xx, np.polyval(b, xx), color='k', lw=1)
        ba = np.polyfit(xs, ys, 1)
        xa = np.array([xs.min(), xs.max()])
        ax.plot(xa, np.polyval(ba, xa), color='#c0392b', lw=.8, ls='--')
        ax.set_title('%s\n%s = %+.2f [%+.2f, %+.2f], p = %.3f, q = %.3f'
                     % (LABEL.get(t['Scale'], t['Scale']), t['Rule_coef'], float(t['Rule_value']),
                        float(t['Rule_CI_lo']), float(t['Rule_CI_hi']), float(t['Rule_p']),
                        float(t['Rule_q'])), fontsize=7)
        ax.set_xlabel('Depth slope (z); negative = stronger decrease')
        ax.set_ylabel('Score')
        ax.legend(frameon=False, fontsize=6.5, loc='best')
    fig.suptitle('Black line: fit on kept points. Red dashed: fit on all participants.',
                 fontsize=8, y=.995)
    fig.tight_layout()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    out_png = os.path.join(config.FIGURES_DIR, 'Supplementary_Figure_3_questionnaires.png')
    fig.savefig(out_png, dpi=200)
    print('wrote', out_png)


if __name__ == '__main__':
    main()
