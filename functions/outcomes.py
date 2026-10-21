# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
outcomes: associations of a participant-level effect with post-session outcomes and
demographics.

Each outcome is read per session from the questionnaire tables, averaged over the sessions a
participant completed, and correlated with a participant-level predictor (for example the
depth slope of an effect).

What the numbers mean, and what they do not:
  * Spearman is primary. OLS is secondary and is not computed for an outcome with a floor
    effect (>= 35% of participants at the minimum), where a linear model is not describing a
    continuous variable.
  * `rho_clean` repeats Spearman after removing bivariate outliers by the skipped-projection
    rule. At n < 40 a single influential pair can carry a rho, so a large gap between
    `Spearman_rho` and `rho_clean` is informative.
  * One FDR correction is reported: Benjamini-Hochberg within questionnaire (`p_fdr_intra`).
    The raw p is kept so the reader can see what was tested.
  * Nothing here is causal. These are cross-sectional correlations between a neural summary
    and a questionnaire.

The outcomes are the post-session POMS, MEDI and TMS scales; five post-session task ratings
(effort, motivation, gong reliability, tones distracting, effort maintained); the pre-session
trait mindfulness battery (MAAS, CAMS-R, FFMQ, FMI); and the POMS post-minus-pre change, the
study's only pre-post contrast. Because BH is applied within `Questionnaire`, one family
cannot move the q values of another.

Questionnaire scores are not L1 covariates. L1 is one regression per participant over that
participant's segments, with `session` entering as k-1 centred dummies. A questionnaire score
is constant within a session, so over segments it is an exact linear function of those dummies
for a two-session participant and a constant column for a one-session participant. Either way
the design is rank deficient and the column would be dropped, as `cond` is under
INCLUDE_SAPO=False. A participant-level moderator is the only form these variables can take,
and this module is where that lives.
"""
from __future__ import annotations

import csv
import io
import os

import numpy as np

__all__ = [
    'BEH_FILES', 'OUTCOME_SPEC', 'FLOOR_THRESH', 'DEMOG_SPEC',
    'fdr_bh', 'skipped_outliers', 'load_outcomes', 'load_outcomes_by_session',
    'load_demographics',
    'spearman_ols_table', 'demographics_table', 'participants_file',
]

# The questionnaire tables of the dataset's phenotype/ folder, one per instrument and session,
# keyed by participant_id.
BEH_FILES = {'%s_%d' % (q, ses): '%s_ses%d.tsv' % (table, ses)
             for q, table in (('POMS', 'poms_post'), ('MEDI', 'medi'), ('TMS', 'tms'),
                              ('TASK', 'session_ratings'), ('PREPOMS', 'poms_pre'),
                              ('MAAS', 'maas'), ('CAMS', 'camsr'), ('FFMQ', 'ffmq'),
                              ('FMI', 'fmi'))
             for ses in (1, 2)}

# Required and optional instruments. A missing required file is returned in `missing`, and
# heart_posthoc then writes no outcomes at all. A missing optional instrument costs only its
# own outcomes, which come back with n = 0 and NaN throughout, and the instrument is named
# in `gaps` so a data gap is not read as a null association.
REQUIRED_TABS = ('POMS', 'MEDI', 'TMS')
OPTIONAL_TABS = ('TASK', 'PREPOMS', 'MAAS', 'CAMS', 'FFMQ', 'FMI')


def _same(col):
    return col, col


def _change(col):
    """POMS post minus pre within one session."""
    return tuple(('diff', ('POMS_%d' % ses, col), ('PREPOMS_%d' % ses, col)) for ses in (1, 2))


# (outcome name, questionnaire, session-1 column, session-2 column)
OUTCOME_SPEC = [
    ('POMS_TotalMoodDisturbance', 'POMS', *_same('total_mood_disturbance')),
    ('POMS_Tension',              'POMS', *_same('tension')),
    ('POMS_Depression',           'POMS', *_same('depression')),
    ('POMS_Anger',                'POMS', *_same('anger')),
    ('POMS_Vigor',                'POMS', *_same('vigor')),
    ('POMS_Fatigue',              'POMS', *_same('fatigue')),
    ('POMS_Confusion',            'POMS', *_same('confusion')),
    ('MEDI_Hindrances',             'MEDI', *_same('hindrances')),
    ('MEDI_Relaxation',             'MEDI', *_same('relaxation')),
    ('MEDI_PersonalSelf',           'MEDI', *_same('personal_self')),
    ('MEDI_TranspersonalQualities', 'MEDI', *_same('transpersonal_qualities')),
    ('MEDI_TranspersonalSelf',      'MEDI', *_same('transpersonal_self')),
    ('TMS_Curiosity',   'TMS', *_same('curiosity')),
    ('TMS_Decentering', 'TMS', *_same('decentering')),

    # Further instrument families in the same analysis. Each is a per-session number
    # averaged over the sessions a participant completed, correlated with the participant-level
    # slope, and BH corrected within its own `Questionnaire` group by spearman_ols_table.
    # They cannot be L1 covariates (see the module docstring).

    # -- post-session task ratings, 1-10
    ('Task_GongReliability',  'TASK', *_same('gong_reliability')),
    ('Task_TonesDistracting', 'TASK', *_same('tones_distracting')),
    ('Task_Effort',           'TASK', *_same('effort')),
    ('Task_Motivation',       'TASK', *_same('motivation')),
    ('Task_MaintainEffort',   'TASK', *_same('maintain_effort')),

    # -- pre-session trait mindfulness battery
    ('Trait_MAAS',  'MAAS', *_same('score')),
    ('Trait_CAMSR', 'CAMS', *_same('score')),
    ('Trait_FFMQ_Observing',           'FFMQ', *_same('observing')),
    ('Trait_FFMQ_Describing',          'FFMQ', *_same('describing')),
    ('Trait_FFMQ_ActingWithAwareness', 'FFMQ', *_same('acting_with_awareness')),
    ('Trait_FFMQ_NonJudging',          'FFMQ', *_same('nonjudging')),
    ('Trait_FFMQ_NonReactivity',       'FFMQ', *_same('nonreactivity')),
    ('Trait_FMI',   'FMI',  *_same('score')),

    # -- within-session mood change, post minus pre. POMS is the only instrument
    #    administered both before and after a recording, so this is the study's
    #    only pre-post contrast. A ('diff', (table, col), (table, col)) entry is
    #    read as first minus second within one session; see load_outcomes.
    ('Change_POMS_TotalMoodDisturbance', 'CHANGE', *_change('total_mood_disturbance')),
    ('Change_POMS_Tension',              'CHANGE', *_change('tension')),
    ('Change_POMS_Depression',           'CHANGE', *_change('depression')),
    ('Change_POMS_Anger',                'CHANGE', *_change('anger')),
    ('Change_POMS_Vigor',                'CHANGE', *_change('vigor')),
    ('Change_POMS_Fatigue',              'CHANGE', *_change('fatigue')),
    ('Change_POMS_Confusion',            'CHANGE', *_change('confusion')),
]

FLOOR_THRESH = 0.35

# Demographics are read from the dataset's participants.tsv, beside the phenotype/ folder.
PARTICIPANTS_TSV = 'participants.tsv'
SEX_CODE = {'female': 0.0, 'male': 1.0}

# (column of participants.tsv, label)
DEMOG_SPEC = [
    ('age', 'Age'),
    ('sex', 'Sex'),
    ('meditation_years_experience', 'Years_Experience'),
    ('retreat_days', 'Log_Retreat_Days'),
    ('meditation_frequency', 'Meditation_Frequency'),
    ('meditation_length', 'Session_Length'),
]


def participants_file(beh_path):
    """participants.tsv of the dataset whose phenotype/ folder is `beh_path`."""
    return os.path.join(os.path.dirname(os.path.abspath(beh_path)), PARTICIPANTS_TSV)


# ===========================================================================
# STATISTICS
# ===========================================================================
def fdr_bh(p):
    """Benjamini-Hochberg adjusted p values. A NaN p stays NaN and is not counted in m."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def skipped_outliers(x, y):
    """Projection-based bivariate outlier detection (Wilcox 2012; Pernet 2012).

    Each point is projected onto its own direction from the bivariate median and compared
    to a MAD-scaled chi2 threshold. This is what makes `rho_clean` meaningful: a single
    influential pair can carry a Spearman rho at n < 40.
    """
    from scipy.stats import chi2
    X = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n = X.shape[0]
    ctr = np.median(X, axis=0)
    thr = np.sqrt(chi2.ppf(0.975, 2))
    is_out = np.zeros(n, bool)
    D = X - ctr
    for i in range(n):
        v = D[i]
        nv = np.linalg.norm(v)
        if nv < np.finfo(float).eps:
            continue
        proj = D @ (v / nv)
        sc = np.median(np.abs(proj - np.median(proj))) / 0.6745
        if sc < np.finfo(float).eps:
            continue
        if abs(proj[i]) / sc > thr:
            is_out[i] = True
    return is_out


def _read_rows(path):
    with io.open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def _record_col(fieldnames):
    """The participant column of a BIDS table."""
    return 'participant_id' if 'participant_id' in fieldnames else fieldnames[0]


def _num(v):
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return np.nan
    return f


# ===========================================================================
# LOADERS
# ===========================================================================
def load_outcomes_by_session(beh_path, record_ids):
    """([nSub x 2 x N] outcomes, one value per session, missing, gaps).

    The per-session numbers `load_outcomes` averages. `missing` / `gaps` as there:
    an absent required file returns (None, missing, gaps).
    """
    def _absent(keys):
        return [BEH_FILES[k] for k in BEH_FILES
                if k.rsplit('_', 1)[0] in keys
                and not os.path.exists(os.path.join(beh_path, BEH_FILES[k]))]

    missing = _absent(REQUIRED_TABS)
    gaps = sorted(_absent(OPTIONAL_TABS))
    if missing:
        return None, missing, gaps
    tabs = {}
    for k, v in BEH_FILES.items():
        path = os.path.join(beh_path, v)
        # An absent optional table is an empty row list, which `col` already
        # answers with NaN. No branch downstream needs to know it was optional.
        tabs[k] = _read_rows(path) if os.path.exists(path) else []

    def col(rows, name, rid):
        if not rows:
            return np.nan
        idc = _record_col(list(rows[0].keys()))
        vals = [_num(r.get(name)) for r in rows
                if str(r.get(idc, '')).strip() == str(rid).strip()]
        vals = [x for x in vals if np.isfinite(x)]
        return float(np.mean(vals)) if vals else np.nan

    def value(spec, q, ses, rid):
        """One session's value for one outcome.

        A plain string reads that column out of the instrument's own table for
        this session. A ('diff', (tab, col), (tab, col)) entry is first minus
        second within the same session, which is how the POMS pre-post change
        outcomes are built. The subtraction is NaN unless both halves are present: a
        change score computed against a missing pre-score would be the post-score
        wearing a change label.
        """
        if isinstance(spec, tuple) and spec and spec[0] == 'diff':
            (ta, ca), (tb, cb) = spec[1], spec[2]
            a, b = col(tabs[ta], ca, rid), col(tabs[tb], cb, rid)
            return a - b if (np.isfinite(a) and np.isfinite(b)) else np.nan
        return col(tabs[f'{q}_{ses}'], spec, rid)

    S = np.full((len(record_ids), 2, len(OUTCOME_SPEC)), np.nan)
    for j, (_nm, q, c1, c2) in enumerate(OUTCOME_SPEC):
        for i, rid in enumerate(record_ids):
            S[i, 0, j] = value(c1, q, 1, rid)
            S[i, 1, j] = value(c2, q, 2, rid)
    return S, [], gaps


def load_outcomes(beh_path, record_ids, return_gaps=False):
    """([nSub x N] outcomes averaged over the two sessions, floor mask, missing).

    A missing required file is returned in `missing` (with None for the arrays), not as
    zeros: an outcome table that silently came back empty would be reported as "no
    association". An absent optional instrument (see REQUIRED_TABS) costs only its own
    outcomes (n = 0, NaN throughout), so a folder with only the POMS, MEDI and TMS tables
    still produces those 14 outcomes.

    `return_gaps=True` adds a fourth element naming the optional instruments that were not
    on disk, so a caller can record it in its status file.
    """
    S, missing, gaps = load_outcomes_by_session(beh_path, record_ids)
    if missing:
        return (None, None, missing, gaps) if return_gaps else (None, None, missing)
    Y = np.full(S.shape[::2], np.nan)
    for j in range(S.shape[2]):
        for i in range(S.shape[0]):
            vals = [v for v in S[i, :, j] if np.isfinite(v)]
            Y[i, j] = np.mean(vals) if vals else np.nan

    # Floor screen: an outcome where >=35% of subjects sit at the minimum is not
    # a continuous variable any more, and its OLS is not reliable.
    floor = np.zeros(len(OUTCOME_SPEC), bool)
    for j in range(Y.shape[1]):
        v = Y[np.isfinite(Y[:, j]), j]
        if v.size and np.mean(v == v.min()) >= FLOOR_THRESH:
            floor[j] = True
    return (Y, floor, [], gaps) if return_gaps else (Y, floor, [])


def load_demographics(beh_path, record_ids):
    """([nSub x 6], labels, n_covered). Retreat days are log1p-transformed."""
    path = participants_file(beh_path)
    if not os.path.exists(path):
        return None, [l for _c, l in DEMOG_SPEC], 0
    rows = _read_rows(path)
    if not rows:
        return None, [l for _c, l in DEMOG_SPEC], 0
    idc = _record_col(list(rows[0].keys()))
    by_id = {}
    for r in rows:
        rid = str(r.get(idc, '')).strip()
        if rid:
            by_id.setdefault(rid, r)
    D = np.full((len(record_ids), len(DEMOG_SPEC)), np.nan)
    covered = 0
    for i, rid in enumerate(record_ids):
        r = by_id.get(str(rid).strip())
        if r is None:
            continue
        covered += 1
        for j, (cname, _lab) in enumerate(DEMOG_SPEC):
            v = (SEX_CODE.get(str(r.get(cname, '')).strip().lower(), np.nan)
                 if cname == 'sex' else _num(r.get(cname)))
            # log1p on retreat days: the raw variable is heavily right-skewed
            # (a handful of long retreats).
            if cname == 'retreat_days' and np.isfinite(v):
                v = float(np.log1p(max(v, 0.0)))
            D[i, j] = v
    return D, [l for _c, l in DEMOG_SPEC], covered


# ===========================================================================
# TABLES
# ===========================================================================
def spearman_ols_table(pred_z, Y, floor):
    """Per-outcome Spearman (primary) + OLS (secondary), with both FDRs."""
    from scipy.stats import spearmanr
    pred_z = np.asarray(pred_z, float)
    rows = []
    for j, (name, quest, _c1, _c2) in enumerate(OUTCOME_SPEC):
        y = Y[:, j]
        ok = np.isfinite(pred_z) & np.isfinite(y)
        n = int(ok.sum())
        r = dict(Outcome=name, Questionnaire=quest, n=n,
                 floor_effect=int(bool(floor[j])), N_outliers=0,
                 Spearman_rho=np.nan, p_Spearman=np.nan,
                 rho_clean=np.nan, p_clean=np.nan,
                 p_fdr_intra=np.nan,
                 OLS_Beta_std=np.nan, OLS_t=np.nan, OLS_p=np.nan, OLS_R2=np.nan,
                 OLS_reliable=int(not bool(floor[j])))
        if n >= 5:
            xz, yy = pred_z[ok], y[ok]
            sr = spearmanr(xz, yy)
            r['Spearman_rho'] = float(sr.statistic)
            r['p_Spearman'] = float(sr.pvalue)
            is_out = skipped_outliers(xz, yy)
            r['N_outliers'] = int(is_out.sum())
            xc_, yc_ = xz[~is_out], yy[~is_out]
            if xc_.size >= 5:
                sr2 = spearmanr(xc_, yc_)
                r['rho_clean'] = float(sr2.statistic)
                r['p_clean'] = float(sr2.pvalue)
                if not floor[j]:
                    xc = xc_ - xc_.mean()
                    Xd = np.column_stack([np.ones(xc.size), xc])
                    b, *_ = np.linalg.lstsq(Xd, yc_, rcond=None)
                    res = yc_ - Xd @ b
                    nn = xc.size
                    if nn > 2 and (xc @ xc) > 0:
                        s2 = res @ res / (nn - 2)
                        se = np.sqrt(s2 / (xc @ xc))
                        sy = yc_.std(ddof=1)
                        sx = xc.std(ddof=1)
                        r['OLS_t'] = float(b[1] / se) if se > 0 else np.nan
                        r['OLS_Beta_std'] = (float(b[1] * sx / sy)
                                             if sy > 0 else np.nan)
                        ss_tot = float(((yc_ - yc_.mean()) ** 2).sum())
                        r['OLS_R2'] = (1.0 - float(res @ res) / ss_tot
                                       if ss_tot > 0 else np.nan)
                        from scipy.stats import t as _t
                        if np.isfinite(r['OLS_t']):
                            r['OLS_p'] = float(2 * _t.sf(abs(r['OLS_t']), nn - 2))
        rows.append(r)

    # One correction: within questionnaire. The outcomes of one instrument are its
    # subscales, and BH across them matches how they are read. It is not a correction
    # across the cells of a run (measure x model x scheme x grain x window), which are
    # heavily correlated; a count of survivors across a run is not a count of
    # independent findings.
    praw = np.array([r['p_Spearman'] for r in rows], float)
    for q in sorted({r['Questionnaire'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['Questionnaire'] == q]
        adj = fdr_bh(praw[idx])
        for k, i in enumerate(idx):
            rows[i]['p_fdr_intra'] = (float(adj[k]) if np.isfinite(adj[k])
                                      else np.nan)
    return rows


def demographics_table(pred_z, D, labels):
    """Spearman of the predictor against each demographic, with BH-FDR."""
    from scipy.stats import spearmanr
    pred_z = np.asarray(pred_z, float)
    rows = []
    for j, lab in enumerate(labels):
        d = D[:, j]
        ok = np.isfinite(pred_z) & np.isfinite(d)
        n = int(ok.sum())
        r = dict(Demographic=lab, n=n, rho=np.nan, p_raw=np.nan, p_fdr=np.nan)
        if n >= 5 and np.unique(d[ok]).size > 1:
            sr = spearmanr(pred_z[ok], d[ok])
            r['rho'] = float(sr.statistic)
            r['p_raw'] = float(sr.pvalue)
        rows.append(r)
    adj = fdr_bh(np.array([r['p_raw'] for r in rows], float))
    for i, r in enumerate(rows):
        r['p_fdr'] = float(adj[i]) if np.isfinite(adj[i]) else np.nan
    return rows


MODERATION_OUTCOME = 'TMS_Decentering'


def outcome_column(name=MODERATION_OUTCOME):
    """Index of one outcome in OUTCOME_SPEC, or None."""
    for j, (nm, _q, _c1, _c2) in enumerate(OUTCOME_SPEC):
        if nm == name:
            return j
    return None


def moderation_table(pred_z, Y, D, labels, outcome=MODERATION_OUTCOME):
    """Does a demographic moderate the effect -> outcome link (by default Decentering)?

    This is the only place a between-participant variable can enter. The depth effect is a
    within-participant slope, so age or years of practice cannot confound it; they can only
    change how strongly it maps onto an outcome. That is the interaction fitted here, and it
    is why the demographics are not covariates in any depth model.

    OLS `y ~ 1 + effect + z(demographic) + effect x z(demographic)`, the interaction t
    reported with BH-FDR across the demographics. Returns [] when the outcome is not in
    OUTCOME_SPEC or `D` is None. A demographic with fewer than 8 usable participants, or no
    variance, gives a row of NaN.

    The n >= 8 floor is low: four parameters on eight points leave 4 df, and a
    "significant" interaction there is not a finding, which is why every row carries its
    own n.
    """
    from scipy.stats import t as t_dist
    j = outcome_column(outcome)
    if j is None or D is None:
        return []
    pred_z = np.asarray(pred_z, float)
    y_all = np.asarray(Y, float)[:, j]
    rows = []
    for k, lab in enumerate(labels):
        d = np.asarray(D, float)[:, k]
        ok = np.isfinite(pred_z) & np.isfinite(y_all) & np.isfinite(d)
        n = int(ok.sum())
        r = dict(Moderator=lab, Outcome=outcome, n=n, beta_effect=np.nan,
                 beta_demog=np.nan, beta_interact=np.nan, t_interact=np.nan,
                 p_interact=np.nan, p_fdr=np.nan)
        dv = d[ok]
        sd = np.std(dv)
        if n >= 8 and sd > 0:
            xp = pred_z[ok]
            xd = (dv - np.mean(dv)) / sd
            X = np.column_stack([np.ones(n), xp, xd, xp * xd])
            try:
                b, *_ = np.linalg.lstsq(X, y_all[ok], rcond=None)
                res = y_all[ok] - X @ b
                s2 = float(res @ res) / (n - 4)
                cov = s2 * np.linalg.inv(X.T @ X)
                se = np.sqrt(np.diag(cov))
                r['beta_effect'] = float(b[1])
                r['beta_demog'] = float(b[2])
                r['beta_interact'] = float(b[3])
                if se[3] > 0:
                    tt = float(b[3] / se[3])
                    r['t_interact'] = tt
                    r['p_interact'] = float(
                        2 * t_dist.cdf(-abs(tt), n - 4))
            except np.linalg.LinAlgError:
                pass
        rows.append(r)
    adj = fdr_bh(np.array([r['p_interact'] for r in rows], float))
    for i, r in enumerate(rows):
        r['p_fdr'] = float(adj[i]) if np.isfinite(adj[i]) else np.nan
    return rows
