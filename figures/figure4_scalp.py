# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure 4 and Supplementary Figure 1: scalp analysis of HRSP at all 64 electrodes.

Figure 4 is the setting that matches the primary analysis (60 s windows, silent blocks,
baseline correction); Supplementary Figure 1 the same with 45 s windows. Both use the
inverse-variance weighted TFCE over channel x frequency x time. Panels:

  A  depth-slope t averaged over 11 posterior channels (chosen in advance, where the selected
     components' mean scalp map peaks); points not significant at any of them are grey
  B  the same map unthresholded; the dashed box is the time and frequency span of the
     component-domain effect in its unweighted form
  C  scalp topography of the depth-slope t averaged over that span
  D  largest |TFCE| of the real map against its sign-flip permutation distribution

A, B and C average t values over channels or points. Reads main/l2_scalp_tfce_ivw_own.npz and
its summary table in the setting folders of run_scalp_analysis.py, and the primary
component-domain summary (to confirm the span drawn in B and C).

    python figures/figure4_scalp.py
"""
import csv
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'functions'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                      # noqa: E402
import figure_style as F                                          # noqa: E402
from figure5_primary import INK, MUTED, NULLC, REALC, outline, clean, primary_root  # noqa: E402
from run_scalp_analysis import run_root                            # noqa: E402

FIGURES = (('dur60_bl_silent', 'Figure_4_scalp'),
           ('dur45_bl_silent', 'Supplementary_Figure_1_scalp_45s'))
MEAS, MODEL = 'hrsp', 'session'
ALPHA = 0.05
POSTERIOR = ('POz', 'Oz', 'O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8', 'Pz', 'P3', 'P4')
# The span of the component-domain effect (unweighted, largest TFCE component of the primary
# run), outlined on the scalp maps; read from the primary run by component_window().
WIN_MS = WIN_HZ = None


def read_csv(fp):
    return list(csv.DictReader(io.open(fp, newline='', encoding='utf-8-sig')))


def component_window():
    """(ms span, Hz span) of the component-domain effect of the primary run."""
    fp = os.path.join(primary_root(), 'perbeat_wise', 'main', 'l2_tfce_summary_session.csv')
    r = [r for r in read_csv(fp) if r['Measure'] == 'hrsp' and r['Scheme'] == 'unweighted'
         and r['Convention'] == 'sn' and r['Component'] == '1'][0]
    return ((float(r['Start_ms']), float(r['End_ms'])),
            (float(r['Start_Hz']), float(r['End_Hz'])))


def load(setting):
    main = os.path.join(run_root(setting), 'main')
    sfx = '_' + config.SCALP_TFCE['scheme']
    key = '%s__%s' % (MEAS, MODEL)
    with np.load(os.path.join(main, 'l2_scalp_tfce%s.npz' % sfx), allow_pickle=True) as z:
        d = dict(times=np.asarray(z['times'], float), freqs=np.asarray(z['freqs'], float),
                 ch=[str(c) for c in z['ch_names']],
                 t=np.asarray(z['t__' + key], float),          # [ch, f, t]
                 tfce=np.asarray(z['tfce__' + key], float),
                 p=np.asarray(z['p__' + key], float),
                 null=np.asarray(z['null_max__' + key], float),
                 params=json.loads(str(z['params'])),
                 check=json.loads(str(z['check__' + key])))
    d['n'] = int(d['check']['n_sub'])
    d['nperm'] = int(d['params']['nPerm'])
    assert d['null'].size == d['nperm'] - 1
    d['real'] = float(np.nanmax(np.abs(d['tfce'])))
    d['p_min'] = float(np.nanmin(d['p']))
    p_re = (1 + np.sum(d['null'] >= d['real'])) / d['nperm']
    assert abs(p_re - d['p_min']) < 1e-6, (p_re, d['p_min'])   # p is stored as float32
    summ = [r for r in read_csv(os.path.join(main, 'l2_scalp_tfce%s_summary.csv' % sfx))
            if r['Measure'] == MEAS and r['Model'] == MODEL]
    assert summ and abs(float(summ[0]['Best_p']) - d['p_min']) < 1e-3

    low = {c.lower(): i for i, c in enumerate(d['ch'])}
    d['post_idx'] = np.array([low[c.lower()] for c in POSTERIOR])
    d['post_t'] = d['t'][d['post_idx']].mean(axis=0)                  # [f, t]
    d['post_sig'] = (d['p'][d['post_idx']] < ALPHA).any(axis=0)       # [f, t]
    tm = (d['times'] >= WIN_MS[0]) & (d['times'] <= WIN_MS[1])
    fm = (d['freqs'] >= WIN_HZ[0]) & (d['freqs'] <= WIN_HZ[1])
    d['topo'] = d['t'][:, fm][:, :, tm].mean(axis=(1, 2))            # [ch]
    return d


def make_info(ch):
    import mne
    mont = mne.channels.make_standard_montage('standard_1005')
    std = {c.lower(): c for c in mont.ch_names}
    info = mne.create_info([std[c.lower()] for c in ch], 250.0, 'eeg')
    info.set_montage(mont)
    return info


def tf_axes(ax, ext):
    ax.axvline(0, color=INK, ls=':', lw=0.8)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([-100, 0, 100, 200, 300, 400, 500, 600])
    ax.set_yticks([4, 10, 15, 20, 25, 30])
    ax.set_xlabel('time after the heartbeat (ms)')


def panel_letter(fig, ax, s, dx=-0.055, dy=0.03):
    b = ax.get_position()
    fig.text(b.x0 + dx, b.y1 + dy, s, fontsize=8, fontweight='bold', va='top',
             ha='left', color=INK)


def draw(d, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mne
    from matplotlib.ticker import MaxNLocator

    t, f = d['times'], d['freqs']
    info = make_info(d['ch'])
    norm = matplotlib.colors.Normalize(-F.T_CLIM, F.T_CLIM)
    with plt.rc_context(F.PAPER_RC):
        fig = plt.figure(figsize=(180 / 25.4, 118 / 25.4), dpi=600, facecolor='white')
        gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.045),
                              height_ratios=(1.0, 0.95),
                              left=0.07, right=0.94, bottom=0.085, top=0.92,
                              wspace=0.30, hspace=0.62)
        axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])
        gsb = gs[1, :].subgridspec(1, 2, width_ratios=(1.0, 1.25), wspace=0.35)
        axC, axD = fig.add_subplot(gsb[0, 0]), fig.add_subplot(gsb[0, 1])

        dt, df = np.diff(t).mean(), np.diff(f).mean()
        ext = (t[0] - dt / 2, t[-1] + dt / 2, f[0] - df / 2, f[-1] + df / 2)
        npost = len(POSTERIOR)

        # A: thresholded
        axA.set_facecolor('#d9d9d9')
        im = axA.imshow(np.ma.masked_where(~d['post_sig'], d['post_t']), origin='lower',
                        aspect='auto', cmap='RdBu_r', norm=norm, extent=ext,
                        interpolation='nearest')
        if d['post_sig'].any():
            outline(axA, d['post_sig'], t, f, colors=INK, linewidths=0.7)
        else:
            axA.text(0.5, 0.5, 'no TFCE-significant cell\n(smallest p = %s)'
                     % ('%.3f' % d['p_min']).lstrip('0'), transform=axA.transAxes,
                     ha='center', va='center', fontsize=6, color=INK)
        tf_axes(axA, ext)
        axA.text(0, f[-1] + df / 2 + 0.4, 'R-peak', ha='center', va='bottom',
                 fontsize=5, color=MUTED)
        axA.set_ylabel('frequency (Hz)')
        axA.set_title('Depth slope, %d posterior channels, TFCE-thresholded' % npost,
                      loc='left', pad=9)
        axA.text(1.0, -0.20, 'grey: TFCE p >= .05 at every channel in the set',
                 transform=axA.transAxes, ha='right', va='top', fontsize=5, color=MUTED)

        # B: unthresholded
        axB.imshow(d['post_t'], origin='lower', aspect='auto', cmap='RdBu_r', norm=norm,
                   extent=ext, interpolation='nearest')
        x0, x1 = WIN_MS[0] - dt / 2, WIN_MS[1] + dt / 2
        y0, y1 = WIN_HZ[0] - df / 2, WIN_HZ[1] + df / 2
        axB.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=INK, lw=0.7, ls='--')
        tf_axes(axB, ext)
        axB.text(0, f[-1] + df / 2 + 0.4, 'R-peak', ha='center', va='bottom',
                 fontsize=5, color=MUTED)
        axB.set_title('Same map, unthresholded' + (' (not significant)'
                                                 if d['p_min'] >= 0.05 else ''),
                      loc='left', pad=9)
        axB.text(1.0, -0.20, 'dashed box: main-effect window (Figure 5)',
                 transform=axB.transAxes, ha='right', va='top', fontsize=5, color=MUTED)
        cb = fig.colorbar(im, cax=cax, ticks=[-4, -2, 0, 2, 4])
        cb.set_label('mean depth slope t over %d channels (n = %d)' % (npost, d['n']))
        cb.outline.set_linewidth(0.5)

        # C: topography over the component-domain span
        mask = np.zeros(len(d['ch']), bool)
        mask[d['post_idx']] = True
        imC, _ = mne.viz.plot_topomap(
            d['topo'], info, axes=axC, show=False, cmap='RdBu_r', contours=0,
            vlim=(-F.T_CLIM, F.T_CLIM), sensors=True, mask=mask,
            mask_params=dict(marker='o', markerfacecolor='white', markeredgecolor=INK,
                             linewidth=0, markersize=2.6, markeredgewidth=0.5))
        axC.set_title('Depth slope t, %d-%d ms, %d-%d Hz'
                      % (WIN_MS[0], WIN_MS[1], WIN_HZ[0], WIN_HZ[1]), loc='left', pad=9)
        axC.text(0.5, -0.08, 'white dots: posterior channels in A and B',
                 transform=axC.transAxes, ha='center', va='top', fontsize=5, color=MUTED)
        cbc = fig.colorbar(imC, ax=axC, fraction=0.05, pad=0.04, ticks=[-4, -2, 0, 2, 4])
        cbc.set_label('t (mean over window)')
        cbc.outline.set_linewidth(0.5)

        # D: permutation distribution
        null, real = d['null'], d['real']
        q95 = float(np.percentile(null, 95))
        d['q95'] = q95
        bins = np.linspace(0, max(real, null.max()) * 1.05, 40)
        axD.hist(null, bins=bins, color=NULLC, edgecolor='white', linewidth=0.3)
        top = axD.get_ylim()[1] * 1.3
        axD.axvline(q95, color=INK, lw=0.8, ls='--')
        axD.axvline(real, color=REALC, lw=1.4)
        # each label on the outer side of its own line
        s_real, s_q = (-1, 1) if real <= q95 else (1, -1)
        axD.text(real + s_real * bins[-1] * 0.012, top * 0.97, 'real\nmap', color=INK,
                 ha='right' if s_real < 0 else 'left', va='top', fontsize=5.5)
        axD.text(q95 + s_q * bins[-1] * 0.012, top * 0.97, '95th\npercentile', color=INK,
                 ha='right' if s_q < 0 else 'left', va='top', fontsize=5.5)
        axD.set_ylim(0, top)
        axD.set_xlim(0, bins[-1])
        axD.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        axD.set_xlabel('largest |TFCE| in the map (smallest p = %s)'
                       % ('%.3f' % d['p_min']).lstrip('0'))
        axD.set_ylabel('sign-flip permutations')
        axD.set_title('Real map vs %s sign-flip permutations' % format(null.size, ','),
                      loc='left', pad=9)
        clean(axD)

        fig.canvas.draw()
        for ax, s in zip((axA, axB, axD), 'ABD'):
            panel_letter(fig, ax, s)
        # the topography sits inset in its cell: C goes in A's column
        panel_letter(fig, axC, 'C', dx=axA.get_position().x0 - axC.get_position().x0 - 0.055)
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        paths = []
        for ext_ in ('png', 'pdf'):
            fp = os.path.join(config.FIGURES_DIR, '%s.%s' % (stem, ext_))
            fig.savefig(fp, dpi=600, facecolor='white')
            paths.append(fp)
        plt.close(fig)
    return paths


def main():
    global WIN_MS, WIN_HZ
    WIN_MS, WIN_HZ = component_window()
    print('component-domain span: %g to %g ms, %g to %g Hz' % (WIN_MS + WIN_HZ))
    for setting, stem in FIGURES:
        d = load(setting)
        paths = draw(d, stem)
        i = np.unravel_index(int(np.argmin(d['post_t'])), d['post_t'].shape)
        print('%s: n = %d, %d permutations, smallest p %.4f; posterior mean t %.2f to %.2f '
              '(most negative at %.0f ms, %.0f Hz); topography %.2f (%s) to %.2f (%s)'
              % (setting, d['n'], d['nperm'], d['p_min'], d['post_t'].min(), d['post_t'].max(),
                 d['times'][i[1]], d['freqs'][i[0]], d['topo'].min(),
                 d['ch'][int(np.argmin(d['topo']))], d['topo'].max(),
                 d['ch'][int(np.argmax(d['topo']))]))
        for p_ in paths:
            print('wrote', p_)
    return 0


if __name__ == '__main__':
    sys.exit(main())
