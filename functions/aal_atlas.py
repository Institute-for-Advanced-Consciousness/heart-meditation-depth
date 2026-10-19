#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
aal_atlas: AAL region labels by containment, the one labeller every arm uses.

    from aal_atlas import AAL
    aal = AAL()
    name, dist_mm, how = aal.label([x, y, z])                       # MNI mm, cortical parcels
    name, dist_mm, how = aal.label(xyz, cortical_only=False)        # all 116 parcels

Why containment and not the nearest AAL centroid: the centre of a midline field lands
near the cingulate centroids whatever it is, so labelling by the nearest centroid
assigns a cingulate parcel to far more components than the parcels that contain them.

Rule: the AAL116 parcel containing the point; if the point is in no parcel
(white matter, CSF, outside the brain), the parcel of the nearest labelled voxel
and its distance, `how` = 'nearest'. `cortical_only` restricts both to the 78
cortical parcels (no cerebellum, vermis, basal ganglia, thalamus, hippocampus,
amygdala), as the source space of source_localization does.
Atlas: FieldTrip's ROI_MNI_V4 (2 mm, MNI152). Set HEP_AAL_DIR to the folder that
holds ROI_MNI_V4.nii and ROI_MNI_V4.txt.
"""
from __future__ import annotations

import glob
import os

import numpy as np

_CANDIDATES = (
    os.environ.get('HEP_AAL_DIR', ''),
    *sorted(glob.glob(os.path.join(os.path.expanduser('~'), 'Documents', 'MATLAB', 'eeglab*',
                                   'plugins', 'Fieldtrip-lite*', 'template', 'atlas', 'aal'))),
)
# Both spellings: AAL writes "Cerebelum", FieldTrip's ROI_MNI_V4.txt "Cerebellum".
# Matching only one would count the cerebellar parcels as cortex.
NONCORTICAL = ('Cerebelum', 'Cerebellum', 'Vermis', 'Caudate', 'Putamen', 'Pallidum',
               'Thalamus', 'Hippocampus', 'Amygdala')


def aal_dir():
    for d in _CANDIDATES:
        if d and os.path.isfile(os.path.join(d, 'ROI_MNI_V4.nii')):
            return d
    raise FileNotFoundError('AAL volume ROI_MNI_V4.nii not found; set HEP_AAL_DIR')


class AAL:
    """AAL116 volume (FieldTrip's ROI_MNI_V4, 2 mm, MNI152)."""

    def __init__(self, directory=None):
        import nibabel as nib
        from scipy.spatial import cKDTree
        d = directory or aal_dir()
        img = nib.load(os.path.join(d, 'ROI_MNI_V4.nii'))
        self.data = np.asarray(img.get_fdata(), int)
        self.aff = img.affine
        self.inv = np.linalg.inv(img.affine)
        self.names = {}
        for line in open(os.path.join(d, 'ROI_MNI_V4.txt'), encoding='utf-8'):
            p = line.split()
            if len(p) >= 3:
                # ROI_MNI_V4 spells Frontal_Sup_Medial_*; it is renamed to the shorter
                # Frontal_Sup_Med_* so that tallies and ROI names use one spelling.
                self.names[int(p[2])] = p[1].replace('Frontal_Sup_Medial_', 'Frontal_Sup_Med_')
        assert len(self.names) == 116, f'{len(self.names)} AAL labels, expected 116'
        self.cortical = {c for c, n in self.names.items()
                         if not any(k in n for k in NONCORTICAL)}
        assert len(self.cortical) == 78, (
            f'{len(self.cortical)} cortical AAL parcels, expected 78 (116 minus 26 cerebellar/'
            f'vermis, 8 basal ganglia/thalamus, 2 hippocampus, 2 amygdala)')
        self._trees = {}
        for key, codes in (('all', set(self.names)), ('cortical', self.cortical)):
            ijk = np.argwhere(np.isin(self.data, list(codes)))
            xyz = (self.aff @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
            self._trees[key] = (cKDTree(xyz), self.data[tuple(ijk.T)], xyz)

    def code_at(self, xyz):
        xyz = np.atleast_2d(np.asarray(xyz, float))
        ijk = np.rint((self.inv @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.data.shape)), axis=1)
        out = np.zeros(len(xyz), int)
        out[ok] = self.data[tuple(ijk[ok].T)]
        return out

    def label(self, xyz, cortical_only=True):
        """(parcel, distance mm, 'inside'|'nearest'|'none'). Containment first."""
        xyz = np.asarray(xyz, float).ravel()
        if xyz.size != 3 or not np.isfinite(xyz).all():
            return '', np.nan, 'none'
        c = int(self.code_at(xyz)[0])
        allowed = self.cortical if cortical_only else set(self.names)
        if c in allowed:
            return self.names[c], 0.0, 'inside'
        tree, codes, _xyz = self._trees['cortical' if cortical_only else 'all']
        dd, i = tree.query(xyz)
        return self.names[int(codes[i])], float(dd), 'nearest'

