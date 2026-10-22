# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure 7: frontal-midline theta across meditation depth, silent blocks.

  A  the frontocentral scalp measure (no spatial filter, no component selection)
  B  the GED-filtered component

Drawn as panel D of Figure 5: at each depth level, the mean and standard error over
participants of mean log theta power (z-scored within recording), centred within participant
(relative to the participant's own mean over the levels they reported); points in the depth
colours, cold = shallow, warm = deep; the number of participants under each level. The panels
share a vertical axis. Titles give the inverse-variance weighted group depth coefficient and
its 95% interval, with weights 1 / (v + median v) from each participant's own slope variance v.

Reads main/run_results.npz of the silent and pooled run folders of run_fmt_depth.py. The
pooled-block coefficients are printed, not drawn.

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
FONT = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'pdf.fonttype': 42}
INK, MUTED = '#1a1a1a', '#6b6b6b'                 # as in Figure 5


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


def level_stats(prof):
    """[level, participant] values centred within participant -> mean, SE and n per level."""
    c = prof - np.nanmean(prof, axis=0, keepdims=True)
    mu, se, nn = [], [], []
    for row in c:
        v = row[np.isfinite(row)]
        nn.append(v.size)
        mu.append(v.mean() if v.size else np.nan)
        se.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan)
    return np.array(mu), np.array(se), np.array(nn)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    S, P = load('silent'), load('pooled')
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    W, H = 120.0, 56.0                                     # mm
    with plt.rc_context(dict(F.PAPER_RC, **FONT)):
        fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600, facecolor='white')

        def add(x, y, w, h):                               # mm from the top-left corner
            return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

        axes = [add(16.0, 12.0, 44.0, 30.0), add(72.0, 12.0, 44.0, 30.0)]
        lv = S['levels']
        cmap_d = F.depth_cmap()                            # cold = shallow, warm = deep
        dcol = [cmap_d(i / max(lv.size - 1, 1)) for i in range(lv.size)]
        panels = [(S['prof_scalp'], S['beta_scalp'], S['w_scalp'], 'Frontocentral scalp theta'),
                  (S['prof'], S['beta'], S['w'], 'GED theta component')]
        for k, (ax, (prof, betas, wts, ttl)) in enumerate(zip(axes, panels)):
            mu, se, nn = level_stats(prof)
            ax.axhline(0, color=MUTED, lw=0.5, ls='--')
            ax.errorbar(lv, mu, yerr=se, color=INK, lw=0.9, capsize=1.6, elinewidth=0.7,
                        zorder=2)
            ax.scatter(lv, mu, s=20, c=dcol, edgecolors=INK, linewidths=0.5, zorder=3)
            ax.set_xticks(lv)
            ax.set_xticklabels(['%d\nn = %d' % (a, b) for a, b in zip(lv, nn)])
            ax.set_xlim(lv[0] - 0.45, lv[-1] + 0.45)
            ax.set_xlabel('reported meditation depth')
            wm, lo, hi, _n_eff, n = weighted_ci(betas, wts)
            ax.set_title('%s\nslope %+.3f [%+.3f, %+.3f], n = %d' % (ttl, wm, lo, hi, n),
                         loc='left', pad=6)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            fig.text((ax.get_position().x0 * W - 13.0) / W, 1 - 4.0 / H, 'AB'[k], fontsize=8,
                     fontweight='bold', va='top', ha='left', color=INK)
        axes[0].set_ylabel('theta power, z\n(centred within participant)')
        ylo = min(a.get_ylim()[0] for a in axes)
        yhi = max(a.get_ylim()[1] for a in axes)
        for ax in axes:
            ax.set_ylim(ylo, yhi)
        for ext in ('pdf', 'png'):
            p = os.path.join(config.FIGURES_DIR, '%s.%s' % (STEM, ext))
            fig.savefig(p, dpi=600, facecolor='white')
            print('wrote %s' % p)
        plt.close(fig)

    for d, lab in ((S, 'silent'), (P, 'pooled')):
        for key, wk, meas, pk in (('beta_scalp', 'w_scalp', 'scalp', 'prof_scalp'),
                                  ('beta', 'w', 'GED', 'prof')):
            wm, lo, hi, n_eff, n = weighted_ci(d[key], d[wk])
            mu, se, nn = level_stats(d[pk])
            print('%-6s blocks, %-5s slope %+.4f [%+.4f, %+.4f], n = %d (Kish n_eff %.1f); '
                  'level means %s (n %s)' % (lab, meas, wm, lo, hi, n, n_eff,
                                             ', '.join('%+.2f' % x for x in mu), nn.tolist()))


if __name__ == '__main__':
    main()
