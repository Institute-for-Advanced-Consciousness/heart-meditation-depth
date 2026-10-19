# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Supplementary Figure 4: HRSP differences between recording periods.

One panel per contrast (second period minus first): the group t-map of the participants'
difference in heartbeat-related spectral power, inverse-variance weighted, with baseline
correction by regression. Points with TFCE-corrected p >= .05 are grey; titles give the
smallest corrected p and the number of participants. The six contrasts are not corrected
across one another.

Reads the heartbeat period run folder of run_period_contrasts.py: period_tfce_all.csv and,
per contrast, main/period_tfce.npz and main/period_maps.npz.

    python figures/supplementary_figure4_periods.py
"""
import csv
import glob
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                      # noqa: E402
import figure_style as F                                          # noqa: E402

STEM = 'Supplementary_Figure_4_periods'
MEAS, SCHEME, ALPHA = 'hrsp', 'ivw_own', 0.05
CONTRASTS = [('premed_vs_baseline', 'Pre-meditation minus baseline'),
             ('oddball_vs_baseline', 'Oddball minus baseline'),
             ('postmed_vs_baseline', 'Post-meditation minus baseline'),
             ('oddball_vs_premed', 'Oddball minus pre-meditation'),
             ('postmed_vs_premed', 'Post- minus pre-meditation'),
             ('postmed_vs_oddball', 'Post-meditation minus oddball')]


def period_run():
    """The run folder whose recorded parameters are those of config.PERIOD_HEART."""
    hits = []
    for fp in glob.glob(os.path.join(config.PERIOD_ROOT, 'HEART_PERIOD', '*', 'run_params.json')):
        with io.open(fp, encoding='utf-8') as fh:
            p = json.load(fh)
        if all(p.get(k) == v for k, v in config.PERIOD_HEART.items()):
            hits.append(os.path.dirname(fp))
    if len(hits) != 1:
        raise SystemExit('%d heartbeat period runs match config.PERIOD_HEART under %s'
                         % (len(hits), config.PERIOD_ROOT))
    return hits[0]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run = period_run()
    rows = list(csv.DictReader(io.open(os.path.join(run, 'period_tfce_all.csv'),
                                       encoding='utf-8')))
    info = {r['Contrast']: r for r in rows
            if r['Measure'] == MEAS and r['Scheme'] == SCHEME and r['Component'] in ('0', '1')}
    data = {}
    for c, _l in CONTRASTS:
        z = np.load(os.path.join(run, c, 'main', 'period_tfce.npz'))
        m = np.load(os.path.join(run, c, 'main', 'period_maps.npz'))
        t, p = z['t__%s__%s' % (MEAS, SCHEME)], z['p__%s__%s' % (MEAS, SCHEME)]
        if abs(float(p.min()) - float(info[c]['Best_p'])) > 1e-6:
            raise SystemExit('%s: stored p map does not match period_tfce_all.csv' % c)
        data[c] = (t, p, np.asarray(m['times'], float), np.asarray(m['freqs'], float))
    vmax = max(float(np.abs(d[0]).max()) for d in data.values())

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    with plt.rc_context(F.PAPER_RC):
        fig, axes = plt.subplots(2, 3, figsize=(6.85, 3.9), dpi=600, sharex=True,
                                 sharey=True)
        im = None
        for ax, (c, lab) in zip(axes.ravel(), CONTRASTS):
            t, p, times, freqs = data[c]
            dt, df = times[1] - times[0], freqs[1] - freqs[0]
            ext = [times[0] - dt / 2, times[-1] + dt / 2,
                   freqs[0] - df / 2, freqs[-1] + df / 2]
            ax.imshow(np.zeros_like(t), origin='lower', aspect='auto', extent=ext,
                      cmap='Greys', vmin=-1, vmax=3)
            im = ax.imshow(np.ma.masked_where(p >= ALPHA, t), origin='lower',
                           aspect='auto', extent=ext, cmap='RdBu_r', vmin=-vmax,
                           vmax=vmax)
            ax.axvline(0, color='k', lw=0.6, ls='--')
            r = info[c]
            ax.set_title('%s\np = %s, n = %s' % (lab, ('%.4f' % float(r['Best_p'])).lstrip('0'),
                                                 r['N_subjects']), pad=3)
        for ax in axes[-1]:
            ax.set_xlabel('time from the R-peak (ms)')
        for ax in axes[:, 0]:
            ax.set_ylabel('frequency (Hz)')
        fig.tight_layout()
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
        cb.set_label('t (significant cells)')
        for e in ('pdf', 'png'):
            fp = os.path.join(config.FIGURES_DIR, '%s.%s' % (STEM, e))
            fig.savefig(fp, dpi=600, bbox_inches='tight')
            print('wrote %s' % fp)
        plt.close(fig)
    for c, lab in CONTRASTS:
        r = info[c]
        print('%-32s p %-7s %s points, %s to %s ms, %s to %s Hz, g %s'
              % (lab, r['Best_p'], r['N_sig_total'], r['Start_ms'], r['End_ms'],
                 r['Start_Hz'], r['End_Hz'], r['HedgesG']))


if __name__ == '__main__':
    main()
