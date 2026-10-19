# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
l2_run: the cluster labelling and the window constants shared by the L2 code.

`_clusters` labels the supra-threshold cells of a frequency x time map, one polarity at a
time, with the 4-connectivity of CONN. The cluster permutation test that uses it is
l2_heart.cluster_perm_weighted; SEED is its default seed.

CFA_WINDOW_MS
    Any cluster whose span reaches into this window is flagged 'CFA window'. The window
    holds the QRS complex, so the cardiac field and a neural effect cannot be told apart
    by latency: a tonic power difference between conditions also gives a cluster that
    starts near 0 ms. The flag marks that ambiguity; it does not identify an artefact.
"""
from __future__ import annotations
import sys
import numpy as np

try:
    from scipy.ndimage import label as _cc_label
except ImportError:
    sys.exit('needs scipy:  pip install scipy')

SEED = 20260819
CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])     # 4-connectivity
CFA_WINDOW_MS = (-150, 150)      # anything reaching in here is QRS territory


def _clusters(t2d, thr):
    """Connected components of the supra-threshold map, one polarity each.

    A positive and a negative patch that touch would otherwise fuse into a single
    component whose span, peak and effect size describe no one effect (FieldTrip also
    clusters per polarity). Mass is sum|t| inside a component, which for a
    single-polarity component is |sum t|, the `SumAbsT` column.

    Labels run 1..k (positive components first, then negative), so the caller's
    `lab == ci + 1` convention holds.
    """
    labp, kp = _cc_label(t2d >= thr, structure=CONN)
    labn, kn = _cc_label(t2d <= -thr, structure=CONN)
    lab = np.zeros(labp.shape, dtype=int)
    lab[labp > 0] = labp[labp > 0]
    lab[labn > 0] = labn[labn > 0] + kp
    k = kp + kn
    if k == 0:
        return np.array([]), lab
    return np.array([np.abs(t2d[lab == i]).sum() for i in range(1, k + 1)]), lab
