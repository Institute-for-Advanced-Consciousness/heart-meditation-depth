# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure 6: where the selected components sit.

Mean eLORETA source power of the components the primary analysis selected (each normalised
to sum to one), on axial, coronal and sagittal slices through the group peak, drawn on the
MNI152 T1 when it is found (functions/mni_slices.py), with the components' mean scalp map
on the right. Slices are in neurological convention (left on the left) and show the 5 mm grid
layer that contains the slice.

Reads source_localization/component_source_maps.npz and source_localization.json in the
primary run folder (run_component_group_tests.py --steps sources) and the run's
perbeat_wise/main/cohort_sessions.csv.

    python figures/figure6_sources.py
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

import config                                                      # noqa: E402
import figure_style as F                                          # noqa: E402
from run_component_group_tests import run_root                     # noqa: E402

STEM = 'Figure_6_sources'
INK = '#1a1a1a'
FONT = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'pdf.fonttype': 42}
GRID = 5.0                                            # mm, fsaverage volume grid
CROP = {0: (-78, 78), 1: (-112, 78), 2: (-58, 86)}    # MNI mm shown per axis
PLANES = (('z', 0, 1), ('y', 0, 2), ('x', 1, 2))      # sliced axis, horizontal, vertical
AX = {'x': 0, 'y': 1, 'z': 2}
SCALE = 0.258                                         # figure mm per MNI mm (all slices)
GAP, X0 = 2.5, 3.0                                    # mm between slices, left margin


def inputs():
    """The group map, its peak, the mean scalp map and the checks against the run."""
    from aal_atlas import AAL
    root = run_root(config.COMPONENT_ROOT, config.COMPONENT_SETTINGS[config.PRIMARY])
    src = os.path.join(root, 'source_localization')
    with np.load(os.path.join(src, 'component_source_maps.npz'), allow_pickle=True) as z:
        cm = {k: z[k] for k in z.files}
    with io.open(os.path.join(src, 'source_localization.json'), encoding='utf-8') as fh:
        rows = json.load(fh)['components']
    kept = {(r['subject'], int(r['session'])): int(r['ic']) for r in csv.DictReader(io.open(
        os.path.join(root, 'perbeat_wise', 'main', 'cohort_sessions.csv'), encoding='utf-8-sig'))
        if r['status'] == 'kept'}
    maps = {(str(s), int(ses)): int(ic) for s, ses, ic in zip(cm['subject'], cm['session'],
                                                               cm['ic'])}
    assert maps == kept, 'the source maps are not those of the components the run analysed'
    aal = AAL()
    key = str(float(cm['alpha']))
    assert len(rows) == len(cm['power'])
    for r, pw in zip(rows, cm['power']):
        assert aal.names[int(cm['codes'][int(np.argmax(pw))])] == r['labels'][key]['peak']
    grp = cm['power'].mean(0)
    ipk = int(np.argmax(grp))
    share = {}
    for c, v in zip(cm['codes'], grp):
        share[aal.names[int(c)]] = share.get(aal.names[int(c)], 0.0) + float(v)
    return dict(pos=cm['pos'], rel=grp / grp[ipk] * 100.0, peak=cm['pos'][ipk],
                label=aal.label(cm['pos'][ipk])[0], topo=aligned_mean_topo(cm['scalp']),
                ch_names=cm['ch_names'], n=int(len(cm['power'])),
                top=sorted(share.items(), key=lambda kv: -kv[1])[:5])


def aligned_mean_topo(scalp):
    """Mean of unit-norm maps, each sign-aligned to the first principal component."""
    M = scalp / np.linalg.norm(scalp, axis=1, keepdims=True)
    _u, _s, vt = np.linalg.svd(M, full_matrices=False)
    tmpl = vt[0] * (1.0 if (M @ vt[0]).sum() >= 0 else -1.0)
    return (M * np.sign(M @ tmpl)[:, None]).mean(0)


def layer_image(pos, val, plane, at):
    """The grid layer containing `at` as a 2-D image (NaN off-cortex) and its extent."""
    a, h, v = AX[plane], *[p for p in (0, 1, 2) if p != AX[plane]]
    sel = np.abs(pos[:, a] - at) <= GRID / 2
    hh = np.arange(CROP[h][0] // GRID * GRID, CROP[h][1] + GRID, GRID)
    vv = np.arange(CROP[v][0] // GRID * GRID, CROP[v][1] + GRID, GRID)
    img = np.full((vv.size, hh.size), np.nan)
    ih = np.round((pos[sel, h] - hh[0]) / GRID).astype(int)
    iv = np.round((pos[sel, v] - vv[0]) / GRID).astype(int)
    ok = (ih >= 0) & (ih < hh.size) & (iv >= 0) & (iv < vv.size)
    img[iv[ok], ih[ok]] = val[sel][ok]
    ext = (hh[0] - GRID / 2, hh[-1] + GRID / 2, vv[0] - GRID / 2, vv[-1] + GRID / 2)
    return img, ext


def draw_slices(axes, pos, val, peak, cmap, vmin, vmax, anat):
    from mni_slices import slice_at
    vol, mni_of_voxel = anat
    im = None
    for ax, (plane, h, v) in zip(axes, PLANES):
        at = peak[AX[plane]]
        if vol is not None:
            bg, ext = slice_at(vol, mni_of_voxel, plane, at)
            if bg is not None:
                ax.imshow(bg, cmap='gray', origin='lower', extent=ext, aspect='equal',
                          interpolation='bilinear')
        img, ext = layer_image(pos, val, plane, at)
        im = ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                       extent=ext, aspect='equal', interpolation='nearest', alpha=0.92)
        ax.plot(peak[h], peak[v], marker='+', color='k', ms=5, mew=0.8)
        ax.set_xlim(*CROP[h])
        ax.set_ylim(*CROP[v])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        tc = 'w' if vol is not None else 'k'              # the T1 is black at the edges
        ax.text(0.03, 0.97, f'{plane} = {at:+.0f}', transform=ax.transAxes, fontsize=5.5,
                va='top', ha='left', color=tc)
        if plane in ('z', 'y'):
            ax.text(0.03, 0.03, 'L', transform=ax.transAxes, fontsize=5.5, ha='left', va='bottom',
                    color=tc)
            ax.text(0.97, 0.03, 'R', transform=ax.transAxes, fontsize=5.5, ha='right', va='bottom',
                    color=tc)
    return im


def draw(G):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mne
    from mni_slices import load_mni_mri
    from source_localization import eeg_info

    W, H, ytop = 180.0, 58.0, 7.0                          # mm
    with plt.rc_context(dict(F.PAPER_RC, **FONT)):
        fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600, facecolor='white')
        anat = load_mni_mri()
        if anat[0] is None:
            print('MNI152 T1 not found: the slices are drawn without anatomy')

        def add(x, y, w, h):                               # mm from the top-left corner
            return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        # the three slices share one mm-per-MNI-mm scale
        sz = [(CROP[h][1] - CROP[h][0], CROP[v][1] - CROP[v][0]) for _p, h, v in PLANES]
        row_h = max(v for _h, v in sz) * SCALE
        cbx = X0 + sum(w for w, _v in sz) * SCALE + GAP * len(sz) + 2.0
        axs, x = [], X0
        for wv, hv in sz:
            axs.append(add(x, ytop + (row_h - hv * SCALE) / 2, wv * SCALE, hv * SCALE))
            x += wv * SCALE + GAP
        im = draw_slices(axs, G['pos'], G['rel'], G['peak'], 'YlOrRd', 0, 100, anat)
        cb = fig.colorbar(im, cax=add(cbx, ytop + 4, 2.4, row_h - 8))
        cb.set_label('eLORETA power (% of peak)', fontsize=6, labelpad=2)
        cb.set_ticks([0, 50, 100])
        cb.ax.tick_params(labelsize=5.5)
        tx = cbx + 17.0
        axT = add(tx, ytop + (row_h - 24) / 2 + 2, W - tx - 3, 24)
        vt = float(np.max(np.abs(G['topo'])))
        mne.viz.plot_topomap(G['topo'], eeg_info([str(c) for c in G['ch_names']]), axes=axT,
                             show=False, cmap='RdBu_r', vlim=(-vt, vt), contours=0,
                             sensors=True, extrapolate='head')
        axT.set_title('mean scalp map', fontsize=6, pad=2)
        fig.text(X0 / W + 5.0 / W, 1 - (ytop - 4.6) / H,
                 'Where the component is: mean eLORETA source power of the %d selected '
                 'components' % G['n'], fontsize=6, va='top', ha='left', color=INK)
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        paths = []
        for ext_ in ('png', 'pdf'):
            fp = os.path.join(config.FIGURES_DIR, '%s.%s' % (STEM, ext_))
            fig.savefig(fp, dpi=600, facecolor='white')
            paths.append(fp)
        plt.close(fig)
    return paths


def main():
    G = inputs()
    paths = draw(G)
    print('%d components; group peak at MNI %s (%s)'
          % (G['n'], np.round(G['peak']).astype(int).tolist(), G['label']))
    print('largest shares of the mean map: ' + ', '.join(
        '%s %.1f%%' % (k, 100 * v) for k, v in G['top']))
    for p_ in paths:
        print('wrote', p_)
    return 0


if __name__ == '__main__':
    sys.exit(main())
