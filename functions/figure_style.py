# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""figure_style: the shared look of the paper's figures (figures/*.py).

Type sizes and line widths for a journal page (PAPER_RC), the colour limit of the t-maps, the
depth colour ramp, and the weighted group mean with its 95% interval drawn in Figure 5.
matplotlib is imported only on call.
"""
from __future__ import annotations

import numpy as np

# The t-map colour scale is fixed at [-4 4], and the interval uses 1.96.
T_CLIM = 4.0
CI_MULT = 1.96

PAPER_RC = {
    'font.size': 6.0, 'axes.titlesize': 6.0, 'axes.labelsize': 6.0,
    'xtick.labelsize': 5.0, 'ytick.labelsize': 5.0, 'legend.fontsize': 4.5,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
}


def weighted_mean_ci(beta, w):
    """Weighted group mean and 1.96 * weighted SE, cell-wise.

    Ws = sum w; m = sum(w x)/Ws; SS = sum(w (x-m)^2); n_eff = Ws^2/sum(w^2);
    var_g = SS/((n_eff-1) Ws). A non-finite cell is dropped from both sums for that
    cell (its weight is zeroed there), so one bad cell cannot NaN the whole trace.
    """
    X = np.asarray(beta, float)                 # [nCell x nSub]
    w = np.asarray(w, float)[None, :]
    bad = ~np.isfinite(X)
    S = np.where(bad, 0.0, X)
    W = np.where(bad, 0.0, np.broadcast_to(w, X.shape))
    Ws = W.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.where(Ws > 0, (S * W).sum(axis=1) / np.where(Ws > 0, Ws, 1), np.nan)
        SS = (((S - m[:, None]) ** 2) * W).sum(axis=1)
        n_eff = Ws ** 2 / np.where((W ** 2).sum(axis=1) > 0,
                                   (W ** 2).sum(axis=1), np.nan)
        var_g = SS / ((n_eff - 1.0) * Ws)
        ci = CI_MULT * np.sqrt(np.where(var_g > 0, var_g, np.nan))
    return m, ci


def depth_cmap():
    """The depth colour ramp: purple (shallowest), blue, green, amber, red (deepest).

    Adjacent levels of a continuous map such as viridis cannot be told apart as thin lines,
    and a categorical palette gives no cue that depth is ordinal. Amber rather than yellow,
    which is nearly invisible as a thin line on white. `cmap(t)` with t in [0, 1].
    """
    from matplotlib.colors import LinearSegmentedColormap
    anchors = [(0.42, 0.24, 0.60),      # 1 purple (shallowest)
               (0.15, 0.45, 0.80),      # 2 blue
               (0.20, 0.65, 0.35),      # 3 green
               (0.95, 0.75, 0.10),      # 4 amber
               (0.84, 0.19, 0.15)]      # 5 red (deepest)
    return LinearSegmentedColormap.from_list('depth_palette', anchors, N=256)
