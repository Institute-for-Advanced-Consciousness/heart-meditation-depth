# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""The primary group test in subsets of participants defined by a rating of EEG viability.

The experimenters rated each participant's EEG from the session notes (1 ideal, 2 permissible,
3 to be corroborated with quality checks, 4 missing but usable). The weighted TFCE of the
primary analysis is repeated on the stored participant slope maps of those rated 1, 1-2, 1-3
and of everyone, the precision weights being re-estimated in each subset. The full sample
must reproduce the stored significant region before any subset is tested.

The ratings file is a tab-separated table: participant id, rating (other columns ignored).
"""
import csv
import io
import os

import numpy as np

from l2_heart import IVW_OWN, SEED, kish_n_eff
from l2_tfce import prepare_measure, tfce_components, tfce_perm_weighted
from tfce_run import subjects_from_npz

SUBSETS = (('all rated 1-4', (1, 2, 3, 4)), ('rated 1-3', (1, 2, 3)),
           ('rated 1-2', (1, 2)), ('rated 1 (ideal)', (1,)))


def read_ratings(path):
    rating = {}
    with io.open(path, encoding='utf-8-sig') as fh:
        for r in csv.reader(fh, delimiter='\t'):
            if r and (r[0].strip().startswith('sub-') or r[0].strip().isdigit()):
                rating[r[0].strip()] = int(r[1])
    return rating


def viability_subsets(main_dir, ratings_path, out_csv, model='session', measure='hrsp',
                      n_perm=5000, verbose=True):
    """Rows of the subset tests; written to out_csv. main_dir is <run>/<grain>_wise/main."""
    rating = read_ratings(ratings_path)
    z = np.load(os.path.join(main_dir, 'run_results.npz'), allow_pickle=False)
    tz = np.load(os.path.join(main_dir, 'l2_tfce_%s.npz' % model), allow_pickle=False)
    times, freqs = np.asarray(z['times'], float), np.asarray(z['freqs'], float)
    subs = subjects_from_npz(z, model, [measure])
    miss = [s['subject'] for s in subs if str(s['subject']) not in rating]
    if miss:
        raise ValueError('no viability rating for %s' % miss)
    ref = np.asarray(tz['sigcells__%s__%s__sn' % (measure, IVW_OWN)], int)
    ref_mask = np.zeros((freqs.size, times.size), bool)
    ref_mask[ref[:, 0], ref[:, 1]] = True
    rows = []
    for lab, keep in SUBSETS:
        sub = [s for s in subs if rating[str(s['subject'])] in keep]
        Xk, w, _nd, _kg = prepare_measure(sub, measure, IVW_OWN, None)
        t, tf, p, _nm, n_used = tfce_perm_weighted(Xk, w, n_perm, SEED, convention='sn')
        sig = np.atleast_2d(p) < .05
        if lab.startswith('all') and not np.array_equal(sig, ref_mask):
            raise RuntimeError('full sample does not reproduce the stored region '
                               '(%d vs %d cells)' % (sig.sum(), ref_mask.sum()))
        comps, best_p, n_sig = tfce_components(t, tf, p, times, freqs, w, n_used, .05)
        t2 = np.atleast_2d(t)
        # the effect in the full-sample region, whatever this subset's own map says
        m = np.array([np.asarray(s[measure][0], float)[ref_mask].mean() for s in sub])
        ww = w / w.mean()
        mw = float((ww * m).sum() / ww.sum())
        ne = kish_n_eff(ww)
        sdw = np.sqrt(float((ww * (m - mw) ** 2).sum() / ww.sum()) * ne / (ne - 1))
        row = dict(Cohort=lab, N=n_used, N_eff_kish=round(ne, 1), Min_p=round(best_p, 4),
                   N_sig_cells=n_sig, N_sig_preR=int((sig & (times[None, :] < 0)).sum()),
                   Overlap_with_full_region=int((sig & ref_mask).sum()),
                   Mean_t_in_full_region=round(float(t2[ref_mask].mean()), 2),
                   dz_in_full_region=round(mw / sdw, 2))
        if comps:
            c = comps[0]
            row.update(Start_ms=c['start_ms'], End_ms=c['end_ms'], Start_Hz=c['start_hz'],
                       End_Hz=c['end_hz'], Peak_ms=c['peak_ms'], Peak_Hz=c['peak_hz'],
                       HedgesG=round(c['hedges_g_neff'], 2),
                       HedgesG_plain=round(c['hedges_g'], 2), Polarity=c['polarity'])
        rows.append(row)
        if verbose:
            print(row, flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with io.open(out_csv, 'w', encoding='utf-8', newline='') as fh:
        w_ = csv.DictWriter(fh, fieldnames=keys)
        w_.writeheader()
        w_.writerows([{k: r.get(k, '') for k in keys} for r in rows])
    if verbose:
        print('wrote', out_csv)
    return rows
