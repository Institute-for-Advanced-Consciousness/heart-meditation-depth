# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the selected components sit: eLORETA source power of each component's scalp map.

Forward model in MNE-Python on the fsaverage template head: three-layer boundary element
model (ico 4, conductivities 0.3, 0.3 / 50, 0.3 S/m), standard 10-05 electrode positions,
average reference, 5 mm volume grid restricted to the 78 cortical regions of the AAL atlas.
The scalp map of each selected component is projected with exact low-resolution
electromagnetic tomography (eLORETA; Pascual-Marqui, 2007). Source power is normalised to
sum to one within each component and averaged across components. Each component is labelled
under two rules, the region containing its strongest point and the region holding most of its
power, at the reference regularisation and at two others; a label is stable when the three
agree.

This describes where the selected components are located. It is not a source reconstruction
of the depth effect.
"""
import copy
import json
import os
from collections import Counter

import numpy as np

from aal_atlas import AAL

BEM_RATIO = 50                       # brain-to-skull conductivity ratio
ALPHAS = (0.01, 0.05, 0.10)          # eLORETA regularisation, fraction of the mean eigenvalue
BASE_ALPHA = 0.05


def fsaverage():
    import mne
    d = mne.datasets.fetch_fsaverage(verbose='ERROR')
    return d, os.path.dirname(d)


def bem_solution(cache_dir, ratio=BEM_RATIO):
    import mne
    f = os.path.join(cache_dir, 'fsaverage-ico4-bem-ratio%d-sol.fif' % ratio)
    if os.path.isfile(f):
        return mne.read_bem_solution(f, verbose='ERROR')
    _fs, subjects_dir = fsaverage()
    model = mne.make_bem_model('fsaverage', ico=4, conductivity=(0.3, 0.3 / ratio, 0.3),
                               subjects_dir=subjects_dir, verbose='ERROR')
    sol = mne.make_bem_solution(model, verbose='ERROR')
    mne.write_bem_solution(f, sol, overwrite=True, verbose='ERROR')
    return sol


def eeg_info(names):
    import mne
    info = mne.create_info(list(names), 250.0, 'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1005'), match_case=False,
                     on_missing='raise')
    return info


def forward_gain(info, src, bem):
    """Average-referenced gain [channels x points x 3] and the point positions in mm.

    The positions come from the input source space, which is in the MRI frame (MNI305 on
    fsaverage); the forward solution returns its source space in head coordinates.
    """
    import mne
    assert all(int(s['coord_frame']) == 5 for s in src), 'source space must be in the MRI frame'
    fs, _sd = fsaverage()
    trans = mne.read_trans(os.path.join(fs, 'bem', 'fsaverage-trans.fif'))
    fwd = mne.make_forward_solution(info, trans, copy.deepcopy(src), bem, meg=False, eeg=True,
                                    mindist=0.0, verbose='ERROR')
    G = np.asarray(fwd['sol']['data'], float)
    G = G - G.mean(axis=0, keepdims=True)
    pos = np.concatenate([s0['rr'][sf['vertno']] for s0, sf in zip(src, fwd['src'])], 0) * 1000.0
    return G.reshape(G.shape[0], -1, 3), pos


def average_reference(m):
    m = np.asarray(m, float).ravel()
    return m - m.mean()


def eloreta(Gc, alpha, n_iter=200, tol=1e-7):
    """Vector eLORETA operator T [points, 3, channels] for an average-referenced gain.

    Weights W_i = (G_i' M G_i)^(1/2), M = (G W^-1 G' + lambda H)^+, iterated to a fixed point
    (Pascual-Marqui, 2007); lambda = alpha times the mean eigenvalue of G W^-1 G'.
    """
    nch, N, _ = Gc.shape
    H = np.eye(nch) - 1.0 / nch
    Winv = np.tile(np.eye(3), (N, 1, 1))
    prev = None
    for _ in range(n_iter):
        GW = np.einsum('cnk,nkl->cnl', Gc, Winv, optimize=True)
        C = np.einsum('cnl,dnl->cd', GW, Gc, optimize=True)
        lam = alpha * np.trace(C) / (nch - 1)
        M = np.linalg.pinv(H @ (C + lam * np.eye(nch)) @ H)
        MG = np.einsum('cd,dnl->cnl', M, Gc, optimize=True)
        B = np.einsum('cnk,cnl->nkl', Gc, MG, optimize=True)
        w, V = np.linalg.eigh(B)
        w = np.clip(w, 1e-30, None)
        Winv = np.einsum('nkj,nj,nlj->nkl', V, 1.0 / np.sqrt(w), V)
        cur = np.einsum('njj->nj', Winv).ravel()
        if prev is not None and np.max(np.abs(cur - prev)) / np.max(np.abs(prev)) < tol:
            break
        prev = cur
    GW = np.einsum('cnk,nkl->cnl', Gc, Winv, optimize=True)
    return np.einsum('cnl,cd->nld', GW, M, optimize=True)


def source_power(T, scalp_map):
    """Source power of one scalp map at every grid point, summing to one."""
    J = np.einsum('nkc,c->nk', T, average_reference(scalp_map))
    pw = (J ** 2).sum(axis=1)
    return pw / pw.sum()


def region_rules(power, codes, names):
    """(region of the strongest point, region holding the most power, its share)."""
    total = Counter()
    for c, v in zip(codes, power):
        total[int(c)] += float(v)
    best = max(total, key=total.get)
    return names[int(codes[int(np.argmax(power))])], names[best], float(total[best])


def selected_components(session_records, data_dir):
    """Scalp map of the component each kept recording analysed, with the analysis' sign rule
    (largest absolute weight positive), and the channel names."""
    comps, names = [], None
    for r in session_records:
        if r.get('status') != 'kept':
            continue
        with np.load(os.path.join(data_dir, str(r['subject']), 'ses%d.npz' % int(r['session'])),
                     allow_pickle=True) as d:
            a = np.asarray(d['icawinv'], float)[:, int(r['ic'])]
            nm = [str(x) for x in d['ch_names']]
        if names is None:
            names = nm
        if nm != names:
            a = a[[nm.index(c) for c in names]]
        if a[int(np.argmax(np.abs(a)))] < 0:
            a = -a
        comps.append(dict(subject=str(r['subject']), session=int(r['session']), ic=int(r['ic']),
                          map=a))
    return comps, names


def localize(session_records, data_dir, out_dir, cache_dir):
    """eLORETA maps of the selected components and their region labels.

    Writes <out_dir>/component_source_maps.npz (power per component at BASE_ALPHA, grid
    positions and region codes, scalp maps) and <out_dir>/source_localization.json (labels
    per component and regularisation, summary). Returns the summary.
    """
    import mne
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    comps, names = selected_components(session_records, data_dir)
    fs_dir, _sd = fsaverage()
    vsrc = mne.read_source_spaces(os.path.join(fs_dir, 'bem', 'fsaverage-vol-5-src.fif'),
                                  verbose='ERROR')
    G, pos = forward_gain(eeg_info(names), vsrc, bem_solution(cache_dir))
    aal = AAL()
    codes = aal.code_at(pos)
    ctx = np.isin(codes, list(aal.cortical))
    Gc, pc, cc = G[:, ctx, :], pos[ctx], codes[ctx]
    print('%d selected components, %d channels, %d cortical grid points' % (
        len(comps), len(names), int(ctx.sum())), flush=True)
    labels = {}
    base_power = None
    for alpha in ALPHAS:
        T = eloreta(Gc, alpha)
        P = np.array([source_power(T, c['map']) for c in comps])
        labels[alpha] = [region_rules(p, cc, aal.names) for p in P]
        if alpha == BASE_ALPHA:
            base_power = P
        print('  eLORETA, regularisation %g: done' % alpha, flush=True)
    rows = []
    for k, c in enumerate(comps):
        per = {str(a): dict(peak=labels[a][k][0], total=labels[a][k][1],
                            total_share=labels[a][k][2]) for a in ALPHAS}
        rows.append(dict(subject=c['subject'], session=c['session'], ic=c['ic'], labels=per,
                         peak_stable=len({v['peak'] for v in per.values()}) == 1,
                         total_stable=len({v['total'] for v in per.values()}) == 1))
    mean = base_power.mean(axis=0)
    i_peak = int(np.argmax(mean))
    share = Counter()
    for c, v in zip(cc, mean):
        share[aal.names[int(c)]] += float(v)
    base = str(BASE_ALPHA)
    summary = dict(
        n_components=len(rows), bem_ratio=BEM_RATIO, alphas=list(ALPHAS), base_alpha=BASE_ALPHA,
        mean_map_peak_mni=[float(v) for v in pc[i_peak]],
        mean_map_peak_region=aal.label(pc[i_peak])[0],
        mean_map_top_regions=[[k, v] for k, v in share.most_common(5)],
        total_rule_counts=Counter(r['labels'][base]['total'] for r in rows).most_common(),
        peak_rule_counts=Counter(r['labels'][base]['peak'] for r in rows).most_common(),
        total_label_stable_share=float(np.mean([r['total_stable'] for r in rows])),
        peak_label_stable_share=float(np.mean([r['peak_stable'] for r in rows])))
    np.savez_compressed(os.path.join(out_dir, 'component_source_maps.npz'), power=base_power,
                        pos=pc, codes=cc, scalp=np.array([c['map'] for c in comps]),
                        ch_names=np.array(names, dtype=object),
                        subject=np.array([c['subject'] for c in comps], dtype=object),
                        session=np.array([c['session'] for c in comps]),
                        ic=np.array([c['ic'] for c in comps]), alpha=np.float64(BASE_ALPHA))
    with open(os.path.join(out_dir, 'source_localization.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(summary=summary, components=rows), fh, indent=1)
    return summary
