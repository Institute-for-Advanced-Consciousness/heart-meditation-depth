# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Supplementary Figure 2 and Supplementary Table 1: the HRSP depth effect with one regressor
added to the primary model at a time.

  A  smallest TFCE-corrected p under inverse-variance weights and unweighted
  B  time span of the largest significant region under inverse-variance weights, with its
     frequency range and Hedges g at the peak

Each model is the primary model with one regressor added, fitted to the same data; nothing is
corrected across them. Reads perbeat_wise/main/l2_tfce_summary_<model>.csv in the two confound
run folders of run_component_analysis.py (config.CONFOUND_ROOTS; band power of the analysed
component, and averaged over all channels), as written by run_component_group_tests.py
--steps confounds. The table is written beside the figure.

    python figures/supplementary_figure2_confounds.py
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
import figure_style as F                                          # noqa: E402
from run_component_group_tests import run_root                     # noqa: E402

STEM = 'Supplementary_Figure_2_confounds'
TABLE = 'Supplementary_Table_1_confounds.csv'
# (confound run, model, row label)
MODELS = [('component', 'session', 'Primary (depth + session)'),
          ('component', 'session+rr', '+ heart period'),
          ('component', 'session+ecgAmp', '+ R-wave amplitude'),
          ('component', 'session+ecgAmpT', '+ T-wave amplitude'),
          ('component', 'session+respAmp', '+ respiration amplitude'),
          ('component', 'session+respPhase', '+ respiration phase'),
          ('component', 'session+theta', '+ component theta power'),
          ('component', 'session+alpha', '+ component alpha power'),
          ('component', 'session+beta', '+ component beta power'),
          ('channels', 'session+theta', '+ scalp theta power'),
          ('channels', 'session+alpha', '+ scalp alpha power'),
          ('channels', 'session+beta', '+ scalp beta power')]
SCHEMES = [('ivw_own', 'inverse variance', '#B22222', 'o'),
           ('unweighted', 'unweighted', '0.45', 's')]
ALPHA = 0.05


def load():
    """{(run, model, scheme): the largest TFCE component of HRSP, or its no-effect row}."""
    out = {}
    for run, model, _lab in MODELS:
        main = os.path.join(run_root(config.CONFOUND_ROOTS[run], config.CONFOUND_SETTINGS[run]),
                            'perbeat_wise', 'main')
        rows = list(csv.DictReader(io.open(
            os.path.join(main, 'l2_tfce_summary_%s.csv' % model), encoding='utf-8-sig')))
        for sch, *_ in SCHEMES:
            r = [x for x in rows if x['Measure'] == 'hrsp' and x['Scheme'] == sch
                 and x['Convention'] == 'sn']
            out[(run, model, sch)] = r[0]
    return out


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    R = load()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    y = np.arange(len(MODELS))[::-1]
    with plt.rc_context(F.PAPER_RC):
        fig, (a, b) = plt.subplots(1, 2, figsize=(6.85, 3.3), dpi=600, sharey=True,
                                   gridspec_kw=dict(width_ratios=[1, 1.35]))
        for sch, lab, col, mk in SCHEMES:
            p = [float(R[(t, m, sch)]['Best_p']) for t, m, _l in MODELS]
            dy = 0.14 if sch == 'ivw_own' else -0.14      # the two schemes side by side
            a.scatter(p, y + dy, s=12, marker=mk, color=col, label=lab, zorder=3,
                      linewidths=0)
        a.axvline(ALPHA, color='k', lw=0.7, ls='--')
        a.set_xscale('log')
        a.set_xlim(1e-4, 1.0)
        a.set_xticks([1e-4, 1e-3, 1e-2, 0.05, 1.0])
        a.set_xticklabels(['.0001', '.001', '.01', '.05', '1'])
        a.set_xlabel('minimum TFCE-corrected p')
        a.set_yticks(y)
        a.set_yticklabels([lab for _t, _m, lab in MODELS])
        a.legend(frameon=False, fontsize=5, loc='lower right',
                 handletextpad=0.2, borderaxespad=0.2)
        a.set_title('A', loc='left', fontweight='bold')

        for i, (t, m, _l) in enumerate(MODELS):
            r = R[(t, m, 'ivw_own')]
            if r['Start_ms'] == '':
                b.text(0, y[i], 'no significant cell', va='center', fontsize=4.5,
                       color='0.4')
                continue
            s, e = float(r['Start_ms']), float(r['End_ms'])
            b.plot([s, e], [y[i], y[i]], color='#B22222', lw=3.2, solid_capstyle='butt')
            b.text(612, y[i], '%.0f-%.0f Hz, g = %.2f'
                   % (float(r['Start_Hz']), float(r['End_Hz']), float(r['HedgesG'])),
                   va='center', ha='left', fontsize=4.5, color='0.25')
        b.axvline(0, color='k', lw=0.7)
        b.set_xlim(-100, 600)
        b.set_xlabel('time from the R-peak (ms): largest significant region')
        b.set_title('B', loc='left', fontweight='bold')
        for ax in (a, b):
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.set_ylim(-0.6, len(MODELS) - 0.4)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            p = os.path.join(config.FIGURES_DIR, '%s.%s' % (STEM, ext))
            fig.savefig(p, dpi=600, bbox_inches='tight')
            print('wrote %s' % p)
        plt.close(fig)

    fields = ('Model', 'Scheme', 'N', 'N_eff', 'Min_p', 'N_sig_cells', 'Start_ms',
              'End_ms', 'Start_Hz', 'End_Hz', 'N_cells_before_R', 'Peak_ms',
              'Peak_Hz', 'Peak_t', 'HedgesG')
    tp = os.path.join(config.FIGURES_DIR, TABLE)
    with io.open(tp, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for t, m, lab in MODELS:
            for sch, slab, _c, _m in SCHEMES:
                r = R[(t, m, sch)]
                w.writerow([lab, slab, r['N_subjects'], r['N_eff_kish'], r['Best_p'],
                            r['N_sig_total'], r['Start_ms'], r['End_ms'],
                            r['Start_Hz'], r['End_Hz'], r['N_cells_preR'],
                            r['Peak_ms'], r['Peak_Hz'], r['Peak_t'], r['HedgesG']])
                print('%-28s %-17s p %-7s %6s to %-6s ms %4s to %-4s Hz, g %s'
                      % (lab, slab, r['Best_p'], r['Start_ms'], r['End_ms'],
                         r['Start_Hz'], r['End_Hz'], r['HedgesG']))
    print('wrote %s' % tp)


if __name__ == '__main__':
    main()
