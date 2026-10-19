# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
mni_slices: an MNI152 T1 background for the source-map figure.

Dipoles are drawn on the MNI152 T1 that ships with EEGLAB's DIPFIT
(`plugins/dipfit/standard_BEM/standard_mri.mat`, the volume `dipplot` uses) rather
than on a bare outline, so a reader can place a coordinate on an anatomical slice.

`standard_mri.mat` is read directly, at its 1 mm grid. Nothing is resampled,
registered or re-oriented: the only transform applied is the volume's own
`transform`, which maps a 1-based voxel index to MNI millimetres.

The file is located on disk; it is not part of this repository and nothing is
downloaded. When it is absent every caller falls back to the outline plot and says
so, because a silently plain background would read as "we chose not to show
anatomy".
"""
from __future__ import annotations

import glob
import os

import numpy as np

__all__ = ['find_mni_mri', 'load_mni_mri', 'slice_at']

_CACHE = {}

# Where EEGLAB puts it. The env var wins so a machine with a non-standard
# layout can point at the file without editing code.
_ENV = 'HEP_MNI_MRI'
_GLOBS = (
    os.path.join('~', 'Documents', 'MATLAB', 'eeglab*', 'plugins', 'dipfit*',
                 'standard_BEM', 'standard_mri.mat'),
    os.path.join('~', 'Documents', 'eeglab*', 'plugins', 'dipfit*',
                 'standard_BEM', 'standard_mri.mat'),
    os.path.join('~', 'eeglab*', 'plugins', 'dipfit*', 'standard_BEM',
                 'standard_mri.mat'),
)


def find_mni_mri():
    """Absolute path to standard_mri.mat, or '' when it is not on this machine.

    Order: $HEP_MNI_MRI, then $EEGLAB_PATH, then the usual install locations.
    """
    env = os.environ.get(_ENV, '').strip()
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    root = os.environ.get('EEGLAB_PATH', '').strip()
    if root:
        hit = glob.glob(os.path.join(root, 'plugins', 'dipfit*', 'standard_BEM',
                                     'standard_mri.mat'))
        if hit:
            return os.path.abspath(sorted(hit)[-1])
    for pat in _GLOBS:
        hit = glob.glob(os.path.expanduser(pat))
        if hit:
            return os.path.abspath(sorted(hit)[-1])
    return ''


def load_mni_mri(path=''):
    """(anatomy, mni_of_voxel) for the MNI152 T1, or (None, None).

    `anatomy` is the raw uint8 volume; `mni_of_voxel` is a callable mapping a
    0-based voxel index array to MNI millimetres, built from the file's own 4x4.

    The 1-based offset matters: `transform` is a MATLAB matrix and maps a 1-based
    index, while numpy indexes from 0. Applying it to a 0-based index puts every
    slice one millimetre off, invisible on a picture and wrong. The +1 is applied
    here, once.
    """
    path = path or find_mni_mri()
    if not path or not os.path.isfile(path):
        return None, None
    if path in _CACHE:
        return _CACHE[path]
    try:
        import scipy.io as sio
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        mri = m['mri']
        vol = np.asarray(mri.anatomy)
        T = np.asarray(mri.transform, float)
    except Exception:                                    # noqa: BLE001
        # A corrupt or unexpected file is the same practical case as an absent
        # one: draw the outline and say the background is missing.
        _CACHE[path] = (None, None)
        return None, None

    def mni_of_voxel(ijk0):
        ijk = np.atleast_2d(np.asarray(ijk0, float)) + 1.0      # -> 1-based
        h = np.column_stack([ijk, np.ones(len(ijk))])
        return (h @ T.T)[:, :3]

    _CACHE[path] = (vol, mni_of_voxel)
    return vol, mni_of_voxel


def _axis_mm(vol, mni_of_voxel, axis):
    """The MNI millimetre value of every voxel index along one volume axis."""
    n = vol.shape[axis]
    ijk = np.zeros((n, 3))
    ijk[:, axis] = np.arange(n)
    return mni_of_voxel(ijk)[:, axis]


def slice_at(vol, mni_of_voxel, plane, mm):
    """One anatomical slice, ready for `imshow`.

    Returns (image, extent) where `extent` is in MNI millimetres and matches
    the (horizontal, vertical) axis pair of the plane: z -> (x, y), y -> (x, z),
    x -> (y, z). A requested `mm` outside the volume is clamped to the nearest slice.

    The image is transposed: `vol[i, j, k]` is (x, y, z), and matplotlib reads an
    image as (row, column) = (vertical, horizontal). Passing the array through
    unrotated draws every slice reflected across the diagonal, which is
    anatomically plausible and wrong.
    """
    ax = {'x': 0, 'y': 1, 'z': 2}[plane]
    mmv = _axis_mm(vol, mni_of_voxel, ax)
    k = int(np.argmin(np.abs(mmv - float(mm))))
    if not (0 <= k < vol.shape[ax]):
        return None, None
    img = [vol[k, :, :], vol[:, k, :], vol[:, :, k]][ax]
    # Remaining axes, in volume order, are the (horizontal, vertical) pair.
    rest = [a for a in (0, 1, 2) if a != ax]
    h_mm = _axis_mm(vol, mni_of_voxel, rest[0])
    v_mm = _axis_mm(vol, mni_of_voxel, rest[1])
    extent = (h_mm[0], h_mm[-1], v_mm[0], v_mm[-1])
    return np.asarray(img).T, extent
