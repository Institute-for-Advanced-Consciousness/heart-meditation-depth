# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure 5: heartbeat-related spectral power decreases with meditative depth.

Primary analysis (HRSP, one component per recording, heartbeat as the unit, baseline
correction by regression, session in the model, inverse-variance weights, TFCE). Panels:

  A  group t-map of the depth slope; points with TFCE-corrected p >= .05 are grey
  B  model-adjusted power in the significant frequency band at each depth level, centred
     within participant (mean and standard error over the participants with that level)
  C  depth slope of that band over time (weighted mean, 95% confidence interval)
  D  adjusted power averaged over the significant points at each depth level
  E  largest |TFCE| of the real map against 50 surrogate datasets with shifted R-peaks
  F  smallest corrected p of each leave-one-participant-out refit

Reads, in the primary run folder of run_component_analysis.py: perbeat_wise/main/
l2_tfce_session.npz, run_results.npz and l2_tfce_loso_session.csv (run_component_group_
tests.py, steps tfce and loso), surrogate_control_tfce.csv and _nulls.csv (step surrogate),
and the model-adjusted level means of functions/adjusted_depth_means.py (built and cached on
the first run, about ten minutes).

    python figures/figure5_primary.py
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
import adjusted_depth_means as AD                                  # noqa: E402
import figure_style as F                                          # noqa: E402
from run_component_group_tests import run_root                     # noqa: E402

GRAIN, MODEL, MEAS, SCHEME, CONV = 'perbeat', 'session', 'hrsp', 'ivw_own', 'sn'
STEM = 'Figure_5_primary'
ALPHA = 0.05
SCALE_POW = 1e3          # power is drawn in units of 1e-3
FONT = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'pdf.fonttype': 42}

INK, MUTED, NULLC, REALC = '#1a1a1a', '#6b6b6b', '#b8b8b8', '#b2182b'
GREY_CELL = '#d9d9d9'    # A: non-significant points
SHADE = '#ebebeb'        # B and C: time points with a significant point


def primary_root():
    return run_root(config.COMPONENT_ROOT, config.COMPONENT_SETTINGS[config.PRIMARY])


def read_csv(fp):
    return list(csv.DictReader(io.open(fp, newline='', encoding='utf-8-sig')))


def runs_of(flags):
    """[(i0, i1)] inclusive index runs where `flags` is True."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]))


def load():
    root = primary_root()
    main = os.path.join(root, '%s_wise' % GRAIN, 'main')
    key = '%s__%s' % (MEAS, SCHEME)
    with np.load(os.path.join(main, 'l2_tfce_%s.npz' % MODEL), allow_pickle=True) as z:
        d = dict(times=np.asarray(z['times'], float), freqs=np.asarray(z['freqs'], float),
                 t=np.asarray(z['t_obs__' + key], float),
                 p=np.asarray(z['p__%s__%s' % (key, CONV)], float),
                 tfce=np.asarray(z['tfce__%s__%s' % (key, CONV)], float),
                 kept=np.asarray(z['kept__' + key], bool),
                 w=np.asarray(z['w__' + key], float),
                 sid=np.asarray(z['subject_ids']).astype(str))
        sc = np.asarray(z['sigcells__%s__%s' % (key, CONV)], int)
    d['n'] = int(d['kept'].sum())
    d['sig'] = d['p'] < ALPHA
    chk = np.zeros_like(d['sig'])
    chk[sc[:, 0], sc[:, 1]] = True
    assert (chk == d['sig']).all(), 'p map and stored significant points disagree'
    d['band'] = d['sig'].any(axis=1)                       # frequency rows with a sig point
    d['tsig'] = d['sig'].any(axis=0)                       # time points with a sig point
    assert np.all(np.diff(np.flatnonzero(d['band'])) == 1), 'significant band is not contiguous'
    fb = d['freqs'][d['band']]
    d['band_lo'], d['band_hi'] = float(fb.min()), float(fb.max())

    # participant depth slopes: the weighted t of A and the time course of C
    with np.load(os.path.join(main, 'run_results.npz'), allow_pickle=True) as r:
        beta = np.asarray(r['L1_%s_%s_beta' % (MODEL, MEAS)], float)       # [sub, f, t]
        has = np.asarray(r['L1_%s_%s_hasmap' % (MODEL, MEAS)], bool)
        assert np.allclose(r['times'], d['times']) and np.allclose(r['freqs'], d['freqs'])
    assert (has == d['kept']).all(), 'TFCE participants differ from the fitted ones'
    X, w = beta[d['kept']], d['w']
    assert w.shape[0] == X.shape[0] and np.isfinite(X).all()
    W = w.sum()
    n_eff = W ** 2 / (w ** 2).sum()
    d['n_eff'] = float(n_eff)
    # weighted t: m = sum(w x) / W, s_w^2 = sum(w (x - m)^2) n_eff / ((n_eff - 1) W)
    wm = np.tensordot(w, X, axes=(0, 0)) / W
    sw2 = np.tensordot(w, (X - wm) ** 2, axes=(0, 0)) * n_eff / ((n_eff - 1.0) * W)
    assert np.allclose(wm / np.sqrt(sw2 / n_eff), d['t'], atol=1e-9), 'stored t-map differs'
    from scipy import stats
    Xb = X[:, d['band'], :].mean(axis=1)                   # [sub, t]
    m, ci196 = F.weighted_mean_ci(Xb.T, w)
    d['C_mean'] = m
    d['C_ci'] = ci196 / F.CI_MULT * stats.t.ppf(0.975, n_eff - 1)
    d['C_n'] = int(X.shape[0])

    # surrogate heartbeats
    def surr(r):
        return (r['Stat'] == 'tfce' and r['Scheme'] == SCHEME and r['Grain'] == GRAIN
                and r['Model'] == MODEL and r['Measure'] == MEAS)
    null = [r for r in read_csv(os.path.join(root, 'surrogate_control_tfce_nulls.csv'))
            if surr(r)]
    summ = [r for r in read_csv(os.path.join(root, 'surrogate_control_tfce.csv')) if surr(r)]
    assert len(summ) == 1 and null
    d['null'] = np.array([float(r['Null_stat']) for r in null])
    d['real'] = float(summ[0]['Real_stat'])
    d['p_surr'] = float(summ[0]['p_surr'])
    assert all(abs(float(r['Real_stat']) - d['real']) < 1e-6 for r in null)
    p_re = (1 + np.sum(d['null'] >= d['real'])) / (1 + d['null'].size)
    assert abs(p_re - d['p_surr']) < 1e-12, (p_re, d['p_surr'])
    assert abs(d['real'] - np.abs(d['tfce']).max()) <= 1e-6 * d['real'], \
        'surrogate real statistic is not the largest |TFCE| of the primary map'

    # leave one participant out
    lo = [r for r in read_csv(os.path.join(main, 'l2_tfce_loso_%s.csv' % MODEL))
          if r['Scheme'] == SCHEME and r['Measure'] == MEAS]
    assert len(lo) == d['n'], '%d leave-one-out refits, expected %d' % (len(lo), d['n'])
    d['loso_p'] = np.array([float(r['Min_p']) for r in lo])
    assert abs(float(lo[0]['Full_min_p']) - d['p'].min()) < 1e-9

    # model-adjusted level means
    ad = AD.load(main, config.DATA_DIR, model=MODEL, measure=MEAS)
    assert list(ad['subject_ids'].astype(str)) == list(d['sid'])
    assert np.array_equal(ad['times'], d['times'])
    sc_b = float(np.nanmax(np.abs(beta)))
    assert float(np.nanmax(np.abs(ad['beta_fit'] - beta))) <= 1e-9 * sc_b, \
        'adjusted level means were built from another fit'
    assert float(np.max(ad['err_hw'])) <= 1e-5 * sc_b, \
        'Huber-weighted slope of the adjusted heartbeats differs from the depth coefficient'
    d['levels'] = np.asarray(ad['levels'], int)
    d['have'] = np.asarray(ad['cnt']) > 0                  # [sub, level]
    assert (d['have'].sum(1) >= 2).all(), 'a participant with one level cannot be centred'
    d['n_level'] = d['have'].sum(0)
    d['adj_hw'] = np.asarray(ad['adj_hw'], float)          # [sub, level, f, t], uncentred
    d['prof'] = AD.centre_levels(d['adj_hw'], d['have'])[:, :, d['sig']].mean(axis=2)
    assert (np.isfinite(d['prof']) == d['have']).all()
    return d


def outline(ax, mask, x, y, **kw):
    """Cell-edge outline of a boolean [y, x] mask on a regular grid."""
    dx, dy = np.diff(x).mean(), np.diff(y).mean()
    m = np.pad(mask, 1)
    segs = []
    ny, nx = mask.shape
    for i in range(ny + 1):              # horizontal edges
        for j in np.flatnonzero(m[i, 1:-1] != m[i + 1, 1:-1]):
            yy = y[0] - dy / 2 + i * dy
            segs.append([(x[j] - dx / 2, yy), (x[j] + dx / 2, yy)])
    for j in range(nx + 1):              # vertical edges
        for i in np.flatnonzero(m[1:-1, j] != m[1:-1, j + 1]):
            xx = x[0] - dx / 2 + j * dx
            segs.append([(xx, y[i] - dy / 2), (xx, y[i] + dy / 2)])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, **kw))


def clean(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def shade_sig(ax, t, tsig):
    dt = np.diff(t).mean()
    for i0, i1 in runs_of(tsig):
        ax.axvspan(t[i0] - dt / 2, t[i1] + dt / 2, color=SHADE, lw=0, zorder=0)


def time_axis(ax, t, label=True):
    dt = np.diff(t).mean()
    ax.set_xlim(t[0] - dt / 2, t[-1] + dt / 2)
    ax.set_xticks([-100, 0, 100, 200, 300, 400, 500, 600])
    if label:
        ax.set_xlabel('time after the heartbeat (ms)')


def draw(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    t, f = d['times'], d['freqs']
    W, H = 180.0, 148.0                                    # mm
    with plt.rc_context(dict(F.PAPER_RC, **FONT)):
        fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600, facecolor='white')

        def add(x, y, w, h):                               # mm from the top-left corner
            return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

        def letter(s, x, y):
            fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight='bold', va='top',
                     ha='left', color=INK)

        LX, LW = 13.0, 86.0                                # left column, shared time axis
        RX, RW = 133.0, 44.0                               # right column
        axA = add(LX, 8.0, LW, 36.0)
        axB = add(LX, 59.0, LW, 32.0)
        axC = add(LX, 106.0, LW, 30.0)
        axD = add(RX, 8.0, RW, 26.0)
        axE = add(RX, 60.0, RW, 24.0)
        axF = add(RX, 108.0, RW, 28.0)
        cmap_d = F.depth_cmap()                                # cold = shallow, warm = deep
        lv = d['levels']
        dcol = [cmap_d(i / max(lv.size - 1, 1)) for i in range(lv.size)]

        # A: t-map
        dt, df = np.diff(t).mean(), np.diff(f).mean()
        ext = (t[0] - dt / 2, t[-1] + dt / 2, f[0] - df / 2, f[-1] + df / 2)
        axA.set_facecolor(GREY_CELL)
        im = axA.imshow(np.ma.masked_where(~d['sig'], d['t']), origin='lower',
                        aspect='auto', cmap='RdBu_r', vmin=-F.T_CLIM, vmax=F.T_CLIM,
                        extent=ext, interpolation='nearest')
        outline(axA, d['sig'], t, f, colors=INK, linewidths=0.6)
        axA.axvline(0, color=INK, ls=':', lw=0.8)
        axA.text(0, ext[3] + 0.4, 'R-peak', ha='center', va='bottom', fontsize=5, color=MUTED)
        time_axis(axA, t, label=False)
        axA.set_ylim(ext[2], ext[3])
        axA.set_yticks([4, 10, 15, 20, 25, 30])
        axA.set_ylabel('frequency (Hz)')
        axA.set_title('Depth slope of power after each heartbeat', loc='left', pad=8)
        cax = add(LX + LW + 2.0, 8.0, 2.2, 36.0)
        cb = fig.colorbar(im, cax=cax, ticks=[-4, -2, 0, 2, 4])
        cb.set_label('depth slope (weighted t, n = %d)' % d['n'], labelpad=2)
        cb.outline.set_linewidth(0.5)
        axA.text(1.0, 1.02, 'grey: TFCE p >= .05', transform=axA.transAxes,
                 ha='right', va='bottom', fontsize=5, color=MUTED)

        # B: adjusted power at each depth level over the significant band
        Ab = AD.centre_levels(d['adj_hw'], d['have'])[:, :, d['band'], :].mean(axis=2)
        pmu, pse, pn = AD.level_stats(Ab, d['have'])       # [level, t]
        assert (pn == d['n_level']).all()
        shade_sig(axB, t, d['tsig'])
        axB.axhline(0, color=MUTED, lw=0.5, ls='--')
        axB.axvline(0, color=INK, ls=':', lw=0.8)
        for li, a in enumerate(lv):
            axB.fill_between(t, (pmu[li] - pse[li]) * SCALE_POW, (pmu[li] + pse[li]) * SCALE_POW,
                             color=dcol[li], alpha=0.10, lw=0, zorder=2)
        for li, a in enumerate(lv):
            axB.plot(t, pmu[li] * SCALE_POW, color=dcol[li], lw=1.0, solid_capstyle='round',
                     zorder=3, label='depth %d (n = %d)' % (a, pn[li]))
        time_axis(axB, t, label=False)
        axB.set_ylabel('adjusted %g-%g Hz power\n(centred, a.u. $\\times10^{-3}$)'
                       % (d['band_lo'], d['band_hi']))
        axB.set_title('Adjusted power after each heartbeat at each depth level '
                      '(mean, SEM)', loc='left', pad=8)
        axB.legend(loc='upper left', bbox_to_anchor=(1.01, 1.02), frameon=False,
                   fontsize=5, handlelength=1.6, borderaxespad=0, labelspacing=0.5)
        clean(axB)

        # C: depth slope over time
        shade_sig(axC, t, d['tsig'])
        m, ci = d['C_mean'] * SCALE_POW, d['C_ci'] * SCALE_POW
        axC.fill_between(t, m - ci, m + ci, color=INK, alpha=0.14, lw=0)
        axC.plot(t, m, color=INK, lw=1.0)
        axC.axhline(0, color=MUTED, lw=0.5, ls='--')
        axC.axvline(0, color=INK, ls=':', lw=0.8)
        time_axis(axC, t)
        axC.set_ylabel('depth slope, %g-%g Hz\n(a.u. $\\times10^{-3}$ per level)'
                       % (d['band_lo'], d['band_hi']))
        axC.set_title('Depth slope over time (weighted mean, 95%% CI, n = %d)' % d['C_n'],
                      loc='left', pad=8)
        clean(axC)

        # D: adjusted power over the significant points at each depth level
        P = d['prof']
        mu, se, nn = [], [], []
        for li in range(lv.size):
            v = P[:, li][np.isfinite(P[:, li])]
            nn.append(v.size)
            mu.append(v.mean() if v.size else np.nan)
            se.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan)
        mu, se, nn = np.array(mu), np.array(se), np.array(nn)
        d['D'] = (lv, mu, se, nn)
        axD.axhline(0, color=MUTED, lw=0.5, ls='--')
        axD.errorbar(lv, mu * SCALE_POW, yerr=se * SCALE_POW, color=INK, lw=0.9,
                     capsize=1.6, elinewidth=0.7, zorder=2)
        axD.scatter(lv, mu * SCALE_POW, s=20, c=dcol, edgecolors=INK, linewidths=0.5,
                    zorder=3)
        axD.set_xticks(lv)
        axD.set_xticklabels(['%d\nn = %d' % (a, b) for a, b in zip(lv, nn)])
        axD.set_xlim(lv[0] - 0.45, lv[-1] + 0.45)
        axD.set_xlabel('reported meditation depth')
        axD.set_ylabel('adjusted power in\nsignificant cells\n(centred, a.u. $\\times10^{-3}$)')
        axD.set_title('Power at each depth level', loc='left', pad=8)
        clean(axD)

        # E: surrogate heartbeats
        null, real = d['null'], d['real']
        bins = np.linspace(0, max(real, null.max()) * 1.08, 28)
        axE.hist(null, bins=bins, color=NULLC, edgecolor='white', linewidth=0.4)
        axE.axvline(real, color=REALC, lw=1.2)
        top = axE.get_ylim()[1] * 1.55
        axE.text(real - bins[-1] * 0.02, top * 0.98, 'real\nheartbeats', color=INK,
                 ha='right', va='top', fontsize=5)
        axE.text(np.median(null), top * 0.98, 'shifted\n(%d surrogates)' % null.size,
                 color=MUTED, ha='center', va='top', fontsize=5)
        axE.set_ylim(0, top)
        axE.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        axE.set_xlim(0, bins[-1])
        axE.set_xlabel('largest |TFCE| in the map')
        axE.set_ylabel('surrogates')
        axE.set_title('Real vs shifted heartbeats', loc='left', pad=8)
        clean(axE)

        # F: leave one participant out
        lp = np.sort(d['loso_p'])
        x = np.arange(1, lp.size + 1)
        logscale = lp.max() < 0.02
        axF.axhline(ALPHA, color=REALC, lw=0.8, ls='--')
        axF.scatter(x, lp, s=5, color=INK, lw=0, zorder=3)
        if logscale:
            # the lower limit sits just under the smallest attainable p (1 / permutations)
            axF.set_yscale('log')
            axF.set_ylim(min(1.5e-4, lp.min() * 0.6), 0.12)
            axF.set_yticks([0.001, 0.01, 0.05])
            axF.set_yticklabels(['.001', '.01', '.05'])
            axF.minorticks_off()
            axF.text(x[-1] + 0.5, ALPHA * 1.12, 'p = .05', color=INK, ha='right',
                     va='bottom', fontsize=5)
        else:
            ytop = max(0.062, lp.max() * 1.15)
            axF.set_ylim(0, ytop)
            yt = [0, 0.02, 0.04, 0.06] if ytop <= 0.07 else list(np.arange(0, ytop, 0.05))
            axF.set_yticks(yt)
            axF.set_yticklabels(['0'] + [('%.2f' % v)[1:] for v in yt[1:]])
            axF.text(x[-1] + 0.5, ALPHA + ytop * 0.025, 'p = .05', color=INK, ha='right',
                     va='bottom', fontsize=5)
        axF.set_xlim(0, x[-1] + 1)
        axF.set_xticks([1, 10, 20, 30, x[-1]])
        axF.set_xlabel('refit without one participant (sorted)')
        axF.set_ylabel('smallest TFCE p' + ('\n(log scale)' if logscale else ''))
        axF.set_title('Leaving out one participant at a time', loc='left', pad=8)
        clean(axF)

        for s, (xx, yy) in zip('ABCDEF', ((1.0, 2.0), (1.0, 53.0), (1.0, 100.0),
                                        (121.0, 2.0), (121.0, 54.0), (121.0, 102.0))):
            letter(s, xx, yy)

        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        paths = []
        for ext_ in ('png', 'pdf'):
            fp = os.path.join(config.FIGURES_DIR, '%s.%s' % (STEM, ext_))
            fig.savefig(fp, dpi=600, facecolor='white')
            paths.append(fp)
        plt.close(fig)
    return paths


def main():
    d = load()
    paths = draw(d)
    t, m = d['times'], d['C_mean']
    print('A  n = %d (Kish n_eff %.2f), %d significant points, smallest p %.4f'
          % (d['n'], d['n_eff'], int(d['sig'].sum()), d['p'].min()))
    ti, fi = np.where(d['sig'].T)[0], np.where(d['sig'])[0]
    print('   significant points span %.0f to %.0f ms, %.0f to %.0f Hz'
          % (t[ti].min(), t[ti].max(), d['freqs'][fi].min(), d['freqs'][fi].max()))
    print('C  band %g-%g Hz: most negative weighted slope %+.5f at %.0f ms'
          % (d['band_lo'], d['band_hi'], m.min(), t[int(m.argmin())]))
    lv, mu, se, nn = d['D']
    print('D  adjusted power, centred (a.u. x 1e-3): ' + ', '.join(
        'depth %d %+.1f (n = %d)' % (a, v * SCALE_POW, n) for a, v, n in zip(lv, mu, nn)))
    print('E  real largest |TFCE| %.2f; surrogates median %.2f, largest %.2f (%d); p %.4f'
          % (d['real'], np.median(d['null']), d['null'].max(), d['null'].size, d['p_surr']))
    print('F  %d refits; smallest p %.4f to %.4f; refits with p >= .05: %d'
          % (d['loso_p'].size, d['loso_p'].min(), d['loso_p'].max(),
             int(np.sum(d['loso_p'] >= ALPHA))))
    for p_ in paths:
        print('wrote', p_)
    return 0


if __name__ == '__main__':
    sys.exit(main())
