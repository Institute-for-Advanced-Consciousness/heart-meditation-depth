# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure 7: frontal-midline theta across meditation depth, SAPO blocks pooled with silent.

  A  the frontocentral scalp measure (no spatial filter, no component selection)
  B  the GED-filtered component

Each violin is the distribution over participants of mean log theta power at that depth level
(z-scored within recording), relative to the participant's own mean over the levels they
reported; grey points are participants. The marker and bar beside each violin are the group
mean and its 95% confidence interval, and the heavier line joins the means. The kernel density
is drawn between the 2nd and 98th percentiles; every point is plotted. The panels share a
vertical axis. Titles give the inverse-variance weighted group depth coefficient, with weights
1 / (v + median v) from each participant's own slope variance v.

Reads main/run_results.npz of the pooled and silent run folders of run_fmt_depth.py. The
silent-block coefficients are printed, not drawn.

    python figures/figure7_theta.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                      # noqa: E402
import figure_style as F                                          # noqa: E402
from run_fmt_depth import run_of_setting                           # noqa: E402

STEM = 'Figure_7_theta'
C_GED = '#B22222'
C_SCALP = '#2C6E8F'
C_PT = '#3A4750'


def load(setting):
    with np.load(os.path.join(run_of_setting(setting), 'main', 'run_results.npz'),
                 allow_pickle=True) as z:
        # row 0 of *_BETA and *_RV is the model without covariates, the one reported
        return dict(beta=np.asarray(z['GED_BETA'], float)[0],
                    beta_scalp=np.asarray(z['SCALP_BETA'], float)[0],
                    w=ivw_own(np.asarray(z['GED_RV'], float)[0]),
                    w_scalp=ivw_own(np.asarray(z['SCALP_RV'], float)[0]),
                    prof=np.asarray(z['PROFILE_SUB_GED'], float),
                    prof_scalp=np.asarray(z['PROFILE_SUB_SCALP'], float),
                    levels=np.asarray(z['DEPTH_LEVELS'], int))


def ivw_own(v):
    """Shrunk inverse-variance weights, 1 / (v + median v)."""
    ok = np.isfinite(v) & (v > 0)
    w = np.full(v.shape, np.nan)
    w[ok] = 1.0 / (v[ok] + np.median(v[ok]))
    return w


def weighted_ci(b, w, alpha=0.05):
    """Weighted mean and confidence interval (Kish effective n)."""
    from scipy.stats import t as _t
    ok = np.isfinite(b) & np.isfinite(w)
    b, w = b[ok], w[ok]
    Ws = w.sum()
    wm = (w * b).sum() / Ws
    n_eff = Ws ** 2 / (w ** 2).sum()
    se = np.sqrt((w * (b - wm) ** 2).sum() / (max(n_eff - 1, 1) * Ws))
    crit = _t.ppf(1 - alpha / 2, max(n_eff - 1, 1))
    return wm, wm - crit * se, wm + crit * se, n_eff, int(ok.sum())


def violin(ax, vals, x, colour, width=0.20, seed=11):
    """Symmetric violin with the points drawn over it."""
    from scipy.stats import gaussian_kde
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return
    lo, hi = np.percentile(v, [2, 98])
    if hi <= lo:
        lo, hi = v.min(), v.max()
    grid = np.linspace(lo, hi, 200)
    dens = gaussian_kde(v)(grid)
    dens = dens / dens.max() * width
    ax.fill_betweenx(grid, x - dens, x + dens, facecolor=colour, alpha=0.22,
                     linewidth=0.5, edgecolor=colour, zorder=1)
    jit = np.random.default_rng(seed).uniform(-0.085, 0.085, v.size)
    ax.scatter(x + jit, v, s=3.4, color=C_PT, alpha=0.55, linewidths=0, zorder=3)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    S, P = load('silent'), load('pooled')
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    with plt.rc_context(F.PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=600)
        lv = P['levels']
        panels = [(axes[0], P['prof_scalp'], P['beta_scalp'], P['w_scalp'], C_SCALP,
                   'Scalp, frontal-midline', 'Theta power, within-participant centred'),
                  (axes[1], P['prof'], P['beta'], P['w'], C_GED, 'GED spatial filter', None)]
        profs = []
        for ax, raw, betas, wts, col, ttl, ylab in panels:
            prof = raw - np.nanmean(raw, axis=0, keepdims=True)    # [level, participant]
            profs.append(prof)
            for i in range(len(lv)):
                violin(ax, prof[i], i, col, width=0.24, seed=20 + i)
                v = prof[i][np.isfinite(prof[i])]
                if v.size >= 2:
                    se = v.std(ddof=1) / np.sqrt(v.size)
                    ax.errorbar(i + 0.30, v.mean(), yerr=1.96 * se, fmt='o', ms=3.8,
                                color=col, ecolor=col, elinewidth=1.5,
                                capsize=2.6, zorder=6, markeredgewidth=0.6,
                                markeredgecolor='w')
            means = [np.nanmean(prof[i]) for i in range(len(lv))]
            ax.plot(np.arange(len(lv)) + 0.30, means, color=col, lw=1.7,
                    alpha=0.95, zorder=5, solid_capstyle='round')
            ax.axhline(0, color='k', lw=0.6, zorder=0)
            ax.set_xticks(range(len(lv)))
            ax.set_xticklabels([str(x) for x in lv])
            ax.set_xlabel('Self-reported meditation depth')
            if ylab:
                ax.set_ylabel(ylab)
            wm, lo, hi, _n_eff, n = weighted_ci(betas, wts)
            ax.set_title('%s\nslope %+.3f [%+.3f, %+.3f], n = %d' % (ttl, wm, lo, hi, n), pad=4)
            ax.set_xlim(-0.55, len(lv) - 0.45)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        ylo = min(a.get_ylim()[0] for a in axes)
        yhi = max(a.get_ylim()[1] for a in axes)
        for ax, prof in zip(axes, profs):
            ax.set_ylim(ylo, yhi)
            for i in range(len(lv)):
                ax.text(i, ylo, 'n=%d' % int(np.isfinite(prof[i]).sum()),
                        ha='center', va='bottom', fontsize=4.2, color='0.45')
        axes[1].set_yticklabels([])
        fig.tight_layout()
        fig.canvas.draw()
        for i, ax in enumerate(axes):
            bb = ax.get_position()
            fig.text(max(bb.x0 - 0.058, 0.002), min(bb.y1 + 0.030, 0.998),
                     'AB'[i], fontsize=8, fontweight='bold', va='top')
        for ext in ('pdf', 'png'):
            p = os.path.join(config.FIGURES_DIR, '%s.%s' % (STEM, ext))
            fig.savefig(p, dpi=600, bbox_inches='tight')
            print('wrote %s' % p)
        plt.close(fig)

    for d, lab in ((P, 'pooled'), (S, 'silent')):
        for key, wk, meas in (('beta_scalp', 'w_scalp', 'scalp'), ('beta', 'w', 'GED')):
            wm, lo, hi, n_eff, n = weighted_ci(d[key], d[wk])
            print('%-6s blocks, %-5s slope %+.4f [%+.4f, %+.4f], n = %d (Kish n_eff %.1f)'
                  % (lab, meas, wm, lo, hi, n, n_eff))


if __name__ == '__main__':
    main()
