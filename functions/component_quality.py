# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
component_quality: ICLabel Brain probability, top class and variance accounted for of every
component of one recording.

Inputs: the recording's segment file (`icawinv`, `icaweights`, `icasphere`, `seg_eeg`) and
the ICLabel probabilities written by extract_npz.py (`ic_labels.npz`,
'<subject>_ses<N>|iclabel' -> [nIC x 7], Brain in column 0).

ICLabel is a classifier, not ground truth, and its Brain probability is not calibrated for
this montage. Differences between selection rules on the same recordings are the
interpretable quantity; an absolute Brain probability is not.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ['session_ic_stats']


def session_ic_stats(data_dir, sid, ses, labels):
    """Per-component (brain_p, argmax, pvaf) for one recording, or None.

    labels : the loaded ic_labels.npz.
    """
    f = os.path.join(data_dir, str(sid), 'ses%d.npz' % int(ses))
    if not os.path.isfile(f):
        return None
    z = np.load(f, allow_pickle=True)
    winv = np.asarray(z['icawinv'], float)             # [nChan x nIC]

    # PVAF: back-projected variance of each component, as a share of the total.
    # winv column norm^2 is the spatial part; scale by the activation variance.
    w = np.asarray(z['icaweights'], float)
    sph = np.asarray(z['icasphere'], float) if 'icasphere' in z.files else None
    unmix = w if sph is None else w @ sph
    nIC = winv.shape[1]

    # Activation variance per component, from the first segments of the recording.
    seg = np.asarray(z['seg_eeg'], np.float32)          # [nSeg x nChan x nT]
    ch = (np.asarray(z['icachansind'], int).ravel() - 0
          if 'icachansind' in z.files else np.arange(seg.shape[1]))
    if ch.max() >= seg.shape[1]:
        ch = np.arange(min(unmix.shape[1], seg.shape[1]))
    take = seg[:min(4, seg.shape[0])][:, ch, :]         # a few segments is plenty
    X = take.transpose(1, 0, 2).reshape(len(ch), -1)
    act = unmix[:nIC, :len(ch)] @ X
    act_var = act.var(axis=1)
    pv = (winv[:, :nIC] ** 2).sum(axis=0) * act_var
    pvaf = 100.0 * pv / max(pv.sum(), 1e-30)

    key = '%s_ses%d|iclabel' % (sid, int(ses))
    if key not in labels.files:
        return None
    lab = np.asarray(labels[key], float)                # [nIC x 7]
    n = min(nIC, lab.shape[0], pvaf.size)
    return dict(brain_p=lab[:n, 0], argmax=lab[:n].argmax(axis=1),
                pvaf=pvaf[:n], n_ic=n)
