# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Associations between each participant's depth effect and the self-report measures.

The predictor is each participant's depth slope averaged over the significant region of the
group test (the largest TFCE component), standardised across participants. Each of the 34
self-report measures (outcomes.OUTCOME_SPEC) is related to it with a skipped correlation
(Pernet, Wilcox and Rousselet, 2013; robust_corr): bivariate outliers are removed by the
projection method, then one coefficient per measure is reported, Pearson's when the remaining
pairs pass the Henze-Zirkler test of bivariate normality (p > .05) and the scale takes at
least 10 distinct values, Spearman's otherwise. Confidence interval and p come from a
percentile bootstrap of the remaining pairs; q is Benjamini-Hochberg within questionnaire.
The bootstrap is conditional on the outlier removal.
"""
import csv
import io
import os

import numpy as np

import heart_posthoc as HP
import outcomes as OC
import robust_corr as RC


def depth_slope_predictor(main_dir, model='session', measure='hrsp', scheme='ivw_own',
                          convention='sn', component='largest'):
    """(participant ids, z-scored mean slope over the significant region, number of points).

    main_dir is <run>/<grain>_wise/main of a run that holds the participant maps
    (run_results.npz) and the TFCE tables of `model` (l2_tfce_<model>.npz / _summary_<model>.csv).
    """
    z = np.load(os.path.join(main_dir, 'run_results.npz'), allow_pickle=False)
    tz = np.load(os.path.join(main_dir, 'l2_tfce_%s.npz' % model), allow_pickle=False)
    with io.open(os.path.join(main_dir, 'l2_tfce_summary_%s.csv' % model),
                 encoding='utf-8-sig') as fh:
        c1 = [r for r in csv.DictReader(fh)
              if r['Convention'] == convention and r['Measure'] == measure
              and r['Scheme'] == scheme and str(r['Component']) == '1'][0]
    cells, _info = HP.tfce_component_cells(tz, measure, scheme, convention, component)
    vals, n_cells, _kind = HP.cluster_predictor(z, model, measure, c1, cells=cells)
    return [str(s) for s in z['subject_ids']], HP._zscore(vals), int(n_cells)


def theta_slope_predictor(main_dir, measure='GED'):
    """(participant ids, z-scored depth slope of one theta measure) of an FMT depth run.

    main_dir is <run>/main of an FMT depth run (run_results.npz); measure is 'GED' or
    'scalp'. The slope is that of the depth-only model. Participants without an accepted
    GED component have no slope and are left out of the correlations.
    """
    z = np.load(os.path.join(main_dir, 'run_results.npz'), allow_pickle=True)
    slope = np.asarray(z['%s_BETA' % measure.upper()], float)[0, :]
    return [str(s) for s in z['SUBJ_IDS']], HP._zscore(slope)


def skipped_correlation_table(sids, x, beh_dir, n_boot=10000):
    """One row per self-report measure; q values are added within questionnaire."""
    Y, floor, missing, _gaps = OC.load_outcomes(beh_dir, sids, return_gaps=True)
    if missing:
        raise FileNotFoundError('behavioural files missing: %s' % missing)
    rows = []
    for j, (name, quest, _c1, _c2) in enumerate(OC.OUTCOME_SPEC):
        ok = np.isfinite(x) & np.isfinite(Y[:, j])
        xs, ys = x[ok], Y[ok, j]
        r = RC.skipped_correlation(xs, ys, n_boot=n_boot, seed=j)
        keep = ~r['outliers']
        _hz, hz_p = RC.hz_test(xs[keep], ys[keep])
        n_dist = int(np.unique(ys[keep]).size)
        pear = bool(hz_p > .05 and n_dist >= 10)
        pe, sp = r['pearson'], r['spearman']
        ch = pe if pear else sp
        rows.append(dict(
            Questionnaire=quest, Scale=name, n=r['n'], N_outliers=r['n_out'],
            MCD_ok=int(r['mcd_ok']), Floor=int(bool(floor[j])), N_distinct=n_dist,
            HZ_p=hz_p, Rule_coef='r' if pear else 'rho',
            Rule_value=ch['r'], Rule_CI_lo=ch['ci'][0], Rule_CI_hi=ch['ci'][1],
            Rule_p=ch['p_boot'], Rule_h_wilcox=int(ch['h_wilcox']),
            Spearman_rho=sp['r'], Spearman_CI_lo=sp['ci'][0], Spearman_CI_hi=sp['ci'][1],
            Spearman_p=sp['p_boot'], Spearman_h_wilcox=int(sp['h_wilcox']),
            Pearson_r=pe['r'], Pearson_CI_lo=pe['ci'][0], Pearson_CI_hi=pe['ci'][1],
            Pearson_p=pe['p_boot'], Pearson_h_wilcox=int(pe['h_wilcox'])))
    for key in ('Rule_p', 'Spearman_p', 'Pearson_p'):
        for q in {r['Questionnaire'] for r in rows}:
            idx = [i for i, r in enumerate(rows) if r['Questionnaire'] == q]
            for i, v in zip(idx, OC.fdr_bh(np.array([rows[i][key] for i in idx]))):
                rows[i][key[:-1] + 'q'] = float(v)
    return rows, Y


def scale_correlation(Y, name_a, name_b):
    """Pearson r between two self-report measures across participants, and the n."""
    names = [sp[0] for sp in OC.OUTCOME_SPEC]
    a, b = Y[:, names.index(name_a)], Y[:, names.index(name_b)]
    both = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[both], b[both])[0, 1]), int(both.sum())


def write_table(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with io.open(out_csv, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise(rows):
    """Printed summary: which coefficient the rule chose, and the measures that pass."""
    print('Pearson chosen by the rule for %d of %d measures; MCD failed for %d; outliers '
          'removed: median %d, range %d-%d' % (
              sum(r['Rule_coef'] == 'r' for r in rows), len(rows),
              sum(not r['MCD_ok'] for r in rows), np.median([r['N_outliers'] for r in rows]),
              min(r['N_outliers'] for r in rows), max(r['N_outliers'] for r in rows)))
    print('CI excludes 0: %d | p < .05: %d | q < .05: %d' % (
        sum(r['Rule_CI_lo'] > 0 or r['Rule_CI_hi'] < 0 for r in rows),
        sum(r['Rule_p'] < .05 for r in rows), sum(r['Rule_q'] < .05 for r in rows)))
    for r in sorted(rows, key=lambda r: r['Rule_p'])[:9]:
        print('  %-28s n %d, %d removed | %s %+.2f [%+.2f, %+.2f] p %.4f q %.3f' % (
            r['Scale'], r['n'], r['N_outliers'], r['Rule_coef'], r['Rule_value'],
            r['Rule_CI_lo'], r['Rule_CI_hi'], r['Rule_p'], r['Rule_q']))
