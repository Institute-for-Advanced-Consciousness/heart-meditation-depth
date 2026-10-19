# -*- coding: utf-8 -*-
# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
posthoc_heart: the post-hoc battery for the HEART depth arm, gated on detection.

    from posthoc_heart import run_posthocs
    rows = run_posthocs(sessions_by_subject, subjects, l2_rows, detected, p, ...)

Every test here runs only on an effect that survived the main model, and only on that
effect. A post-hoc battery run over everything manufactures significant-looking rows out
of nulls, and a forced-fallback post-hoc (run on the best cluster even though nothing
survived) is how a null becomes a finding. `run_posthocs` returns an empty list when
nothing was detected, and says so.

What it does:

  monotonicity   per-subject Spearman rho of the cluster value against depth, and
                 the fraction of subjects whose ordering is actually monotone.
  LOSO           leave-one-subject-out: refit the group test n times and report
                 the worst p. An effect carried by one or two subjects is not a
                 neurofeedback target.

Why not an omnibus ANOVA: it fires on non-monotonic patterns and it needs complete cases
(few subjects have all five depth levels). A significant linear slope through a V-shape is
exactly the failure this battery exists to expose, and an ANOVA would call that V a
success.

The cluster is the exact cell set, not its bounding box. `cluster_perm_weighted` returns
`cells` ([n x 2] freq/time indices). A cluster is a connected set, so its bounding box can
contain cells the test never selected; summarising the box would describe a different
region than the one that was significant.

What is not here, so that an absent section is never read as a null result:
  * Surrogate R-peak control. It re-runs L1 with shuffled peak trains, a ~200x
    multiplier over the whole cohort, and belongs in its own run.
  * Source localization. It is ungated (where the selected components sit describes the
    measure, not a surviving effect), while everything here is gated on detection. It is
    written separately (source_localization.localize).
  * Outcome and demographic associations. heart_posthoc runs them from the per-subject L1
    maps in run_results.npz. They need the behavioural tables, which the runner does not
    read, and are recoverable from a finished run, so they belong in a post-hoc script
    rather than inside the L2 loop.
An absent `source_localization/` or `outcomes_demographics/` means the step was not run
for that run.
"""
from __future__ import annotations
import numpy as np

SEED_POSTHOC = 20260821
MIN_LEVELS_FOR_RHO = 3      # rho over 2 points is always +-1, so uninformative


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
def cluster_cell_mask(cells, n_freq, n_time):
    """[n x 2] (freq_idx, time_idx) -> [nF x nT] bool mask."""
    m = np.zeros((int(n_freq), int(n_time)), bool)
    c = np.asarray(cells, int)
    if c.size:
        m[c[:, 0], c[:, 1]] = True
    return m


def unit_cluster_values(sessions, measure, mask, t_keep, grain='segment', p=None):
    """Per-unit mean of `measure` over the cluster cells, at the fitted grain.

    Returns (values, depth, subject) as parallel 1-D arrays over every unit of
    every session of every subject.

    The unit must be the one the L1 fit used. Under the cell grain, monotonicity and
    LOSO computed over segments would use a different n, a different within-subject
    variance, and a per-subject rho that is not the one the cluster came from.

    The cell reduction is `_cells_of_session`, imported, not reimplemented: it is the
    same sufficient-statistic reduction `subject_slopes` uses (sums and counts, so a
    cell is exactly its segments pooled), and the HEPC beat floor applies per cell,
    which a mean of segment means gets wrong.

    `t_keep` is the TLIMS mask. `session_measures` returns the full HEP_WIN axis, while
    the L1 fit, and therefore the cluster, lives on the TLIMS-trimmed axis.
    """
    if grain not in ('segment', 'cell', 'perbeat'):
        raise ValueError(
            f"grain must be 'segment', 'cell' or 'perbeat', got {grain!r}. A "
            f'typo is refused by name rather than silently reduced to segments.')
    if grain == 'cell':
        from heart_depth_pipeline import _cells_of_session
    elif grain == 'perbeat':
        from heart_depth_pipeline import _beats_of_session
    vals, deps, subs = [], [], []
    for sid in sorted(sessions):
        for s in sessions[sid]:
            if grain == 'cell':
                V, D, _nb = _cells_of_session(s, measure, p or {})
                X = np.asarray(V, float)
                dep_of = np.asarray(D, int)
            elif grain == 'perbeat':
                V, D, _sg = _beats_of_session(s, measure, p or {})
                X = np.asarray(V, float)
                dep_of = np.asarray(D, int)
            else:
                X = np.asarray(s[measure], float)      # [nSeg x nT] or [nSeg x nF x nT]
                dep_of = np.asarray(s['depth'], int)
            if X.ndim == 2:                            # HEP: add a singleton freq
                X = X[:, None, :]
            X = X[..., t_keep]
            if X.shape[1:] != mask.shape:
                raise ValueError(
                    f'{sid}: {measure} trimmed map is {X.shape[1:]} but the '
                    f'cluster mask is {mask.shape}. These must be the same grid; '
                    f'do not broadcast them together.')
            for k in range(X.shape[0]):
                cell = X[k][mask]
                if cell.size and np.isfinite(cell).any():
                    vals.append(float(np.nanmean(cell)))
                    deps.append(int(dep_of[k]))
                    subs.append(sid)
    return np.asarray(vals, float), np.asarray(deps, int), np.asarray(subs, object)


# ═══════════════════════════════════════════════════════════════════════════
# MONOTONICITY
# ═══════════════════════════════════════════════════════════════════════════
def _spearman(x, y):
    """Spearman rho without scipy (ties averaged), NaN if undefined."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2:
        return np.nan

    def rank(v):
        order = np.argsort(v, kind='mergesort')
        r = np.empty(v.size, float)
        r[order] = np.arange(1, v.size + 1, dtype=float)
        # average ties
        uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        for j in np.where(cnt > 1)[0]:
            sel = inv == j
            r[sel] = r[sel].mean()
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def monotonicity_check(depth, values, subject):
    """Per-subject depth trend and whether it is actually monotone.

    A significant group slope says the linear component is nonzero. It does not
    say the profile increases at every step, and for a neurofeedback target the
    second property is the one that matters: the participant is being asked to
    move a number that may not move in one direction.

    Returns dict(rho_median, frac_monotone, n_monotone, n_subjects, n_skipped).
    """
    depth = np.asarray(depth, int)
    values = np.asarray(values, float)
    subject = np.asarray(subject, object)

    rhos, mono, skipped = [], [], 0
    for sid in sorted(set(subject.tolist())):
        m = subject == sid
        d, v = depth[m], values[m]
        ok = np.isfinite(v)
        d, v = d[ok], v[ok]
        levels = sorted(set(d.tolist()))
        if len(levels) < MIN_LEVELS_FOR_RHO:
            # Counted, not dropped silently: rho over two levels is always ±1, so
            # including these would inflate the monotone fraction by construction.
            skipped += 1
            continue
        means = np.array([np.mean(v[d == L]) for L in levels], float)
        r = _spearman(np.asarray(levels, float), means)
        if np.isfinite(r):
            rhos.append(r)
        diffs = np.diff(means)
        mono.append(bool(np.all(diffs > 0) or np.all(diffs < 0)))

    return dict(
        rho_median=float(np.median(rhos)) if rhos else np.nan,
        frac_monotone=float(np.mean(mono)) if mono else np.nan,
        n_monotone=int(np.sum(mono)) if mono else 0,
        n_subjects=len(mono),
        n_skipped=int(skipped))


# ═══════════════════════════════════════════════════════════════════════════
# LOSO
# ═══════════════════════════════════════════════════════════════════════════
def _signflip_p(v, w, n_perm, rng):
    """Weighted one-sample sign-flip p for a vector of per-subject values."""
    v = np.asarray(v, float); w = np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w)
    v, w = v[ok], w[ok]
    n = v.size
    if n < 3 or not np.any(w > 0):
        return np.nan, np.nan, n

    def wt(vv):
        mu = np.sum(w * vv) / np.sum(w)
        var = np.sum(w * (vv - mu) ** 2) / max(np.sum(w), 1e-30)
        se = np.sqrt(var / max(n, 1))
        return mu / se if se > 0 else 0.0

    t_obs = wt(v)
    flips = rng.integers(0, 2, size=(n_perm, n)) * 2 - 1
    null = np.array([wt(v * f) for f in flips])
    # add-one p over the random flips, matching l2_heart's convention
    p = (1 + int((np.abs(null) >= abs(t_obs)).sum())) / (1 + n_perm)
    return float(t_obs), float(p), n


def loso_influence(betas, weights=None, n_perm=1000, seed=SEED_POSTHOC):
    """Leave-one-subject-out stability of a group effect.

    Returns dict(p_full, p_worst, worst_subject_index, n_subjects, all_p).

    The worst p is the answer, not the mean. The question a LOSO answers is "does this
    survive without any single participant", so one subject taking it above alpha is
    the finding, and averaging hides exactly that.
    """
    betas = np.asarray(betas, float)
    w = np.ones_like(betas) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    _, p_full, n = _signflip_p(betas, w, n_perm, rng)

    all_p = []
    for i in range(betas.size):
        keep = np.ones(betas.size, bool)
        keep[i] = False
        _, pi, _ = _signflip_p(betas[keep], w[keep], n_perm,
                               np.random.default_rng(seed + 1 + i))
        all_p.append(pi)
    all_p = np.asarray(all_p, float)
    worst = int(np.nanargmax(all_p)) if np.isfinite(all_p).any() else -1
    return dict(p_full=p_full,
                p_worst=float(np.nanmax(all_p)) if np.isfinite(all_p).any() else np.nan,
                worst_subject_index=worst,
                n_subjects=int(n), all_p=all_p)


# ═══════════════════════════════════════════════════════════════════════════
# THE GATED BATTERY
# ═══════════════════════════════════════════════════════════════════════════
POSTHOC_FIELDS = ('Model', 'Window', 'Measure', 'Scheme', 'Cluster',
                  'N_cells', 'Start_ms', 'End_ms',
                  'Rho_median', 'N_monotone', 'N_subjects_mono', 'Frac_monotone',
                  'N_skipped_lt3_levels',
                  'LOSO_p_full', 'LOSO_p_worst', 'LOSO_worst_subject',
                  'Note')


def run_posthocs(sessions_by_subject, subjects, l2_rows, detected, p,
                 t_keep, times, clusters_by_key=None, n_perm=1000,
                 grain='segment', verbose=True, emit=print):
    """Run the battery on detected effects only. Returns a list of rows.

    `detected` is `decide_detected`'s mapping. Its real shape is
    ``{measure: {'sig': bool, 'scheme': str, 'clusters': [...]}}``; the flatter
    ``{measure: scheme_or_None}`` and ``{measure: [schemes]}`` (what heart_depth_run
    passes) are also accepted. A measure that was not detected is skipped, and the skip is
    reported, so an empty section reads as "nothing survived", never as
    "the post-hocs did not run".
    """
    def _schemes_of(v):
        """Normalise every accepted shape to a list of scheme strings.

        Three shapes are accepted: decide_detected's dict, a bare scheme string, and a
        list. A list lets the battery run on every scheme a cluster survived under, not
        only on the first hit in SIG_PRIORITY.
        """
        if isinstance(v, dict):
            if not v.get('sig'):
                return []
            sch = v.get('scheme', '')
            return [sch] if sch else []
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v if x]
        return [v] if v else []

    detected = {m: _schemes_of(v) for m, v in (detected or {}).items()}

    rows = []
    if not detected or not any(detected.values()):
        if verbose:
            emit('  no detected effect — the post-hoc battery is SKIPPED by '
                 'design (a post-hoc runs only on an effect that survived the '
                 'main model).')
        return rows

    # One row per (measure, scheme), not per measure. The schemes are different
    # estimators of the same effect, so a cluster surviving under two of them is two
    # pieces of evidence about stability and both are reported (`Scheme` is a column).
    _pairs = [(m, sch) for m, schs in sorted(detected.items()) for sch in schs]
    for meas in sorted(detected):
        if not detected[meas] and verbose:
            emit(f'  {meas}: not detected — skipped.')
    for meas, scheme in _pairs:
        key = (meas, scheme)
        cl = (clusters_by_key or {}).get(key)
        if not cl:
            rows.append(dict.fromkeys(POSTHOC_FIELDS, ''))
            rows[-1].update(Measure=meas, Scheme=scheme,
                            Note='DETECTED but no cluster cells were supplied — '
                                 'post-hocs could not be run on the exact '
                                 'cluster and were NOT approximated by its '
                                 'bounding box')
            if verbose:
                emit(f'  {meas}/{scheme}: detected but no cluster cells '
                     f'supplied — post-hocs NOT run (not approximated).')
            continue

        # Surviving clusters only. `clusters_by_key` is the raw cluster list: every
        # connected blob above the CFT, most of which are the small components any null
        # map produces. Running LOSO and the depth-shape test on all of them would write
        # rows that read like a result for clusters that did not survive.
        # `alpha` is the run's own, because it also sets the cluster-forming threshold.
        _alpha = float(p.get('ALPHA', 0.05))
        _surv = [(rank, c) for rank, c in enumerate(cl, 1)
                 if float(c.get('p', 1.0)) < _alpha]
        if not _surv:
            # The cell was detected (decide_detected picked this scheme), so
            # this is a scheme-priority disagreement, not a null. Say so rather
            # than emitting nothing.
            rows.append(dict.fromkeys(POSTHOC_FIELDS, ''))
            rows[-1].update(
                Measure=meas, Scheme=scheme,
                Note='DETECTED at this scheme but no cluster of it is below '
                     'alpha=%g, so there is no surviving cluster to run a '
                     'post-hoc on. NOT a null post-hoc — nothing was tested.'
                     % _alpha)
            if verbose:
                emit(f'  {meas}/{scheme}: no cluster below alpha — post-hocs '
                     f'not run on non-surviving blobs.')
            continue
        if verbose and len(_surv) < len(cl):
            emit(f'  {meas}/{scheme}: {len(_surv)} surviving cluster(s) of '
                 f'{len(cl)}; the rest are not tested.')
        for rank, c in _surv:
            n_f = 1 if meas == 'hep' else len(np.asarray(p['HRSP_FREQS']))
            mask = cluster_cell_mask(c['cells'], n_f, int(np.sum(t_keep)))
            # At the grain the effect was fit at; see unit_cluster_values.
            vals, deps, subs = unit_cluster_values(
                sessions_by_subject, meas, mask, t_keep, grain=grain, p=p)
            mono = monotonicity_check(deps, vals, subs)

            # per-subject cluster-mean slope for LOSO, in cohort order
            beta = []
            for rec in subjects:
                v = rec.get(meas)
                if v is None:
                    beta.append(np.nan); continue
                b = np.asarray(v[0], float)
                if b.ndim == 1:
                    b = b[None, :]
                beta.append(float(np.nanmean(b[mask])) if b.shape == mask.shape
                            else np.nan)
            loso = loso_influence(np.asarray(beta, float), None, n_perm=n_perm)

            tms = np.asarray(times, float)
            row = dict.fromkeys(POSTHOC_FIELDS, '')
            row.update(
                Measure=meas, Scheme=scheme, Cluster=rank,
                N_cells=int(c.get('n_cells', len(np.asarray(c['cells'])))),
                Start_ms=round(float(tms[c['t0']]), 1),
                End_ms=round(float(tms[c['t1']]), 1),
                Rho_median=round(mono['rho_median'], 4)
                if np.isfinite(mono['rho_median']) else '',
                N_monotone=mono['n_monotone'],
                N_subjects_mono=mono['n_subjects'],
                Frac_monotone=round(mono['frac_monotone'], 4)
                if np.isfinite(mono['frac_monotone']) else '',
                N_skipped_lt3_levels=mono['n_skipped'],
                LOSO_p_full=round(loso['p_full'], 5)
                if np.isfinite(loso['p_full']) else '',
                LOSO_p_worst=round(loso['p_worst'], 5)
                if np.isfinite(loso['p_worst']) else '',
                LOSO_worst_subject=(subjects[loso['worst_subject_index']]['subject']
                                    if 0 <= loso['worst_subject_index'] < len(subjects)
                                    else ''))
            if np.isfinite(loso['p_worst']) and loso['p_worst'] >= float(p['ALPHA']):
                row['Note'] = (f'LOSO FAILS: dropping one subject takes p to '
                               f'{loso["p_worst"]:.4f} >= alpha {p["ALPHA"]}')
            rows.append(row)

            if verbose:
                emit(f'    {meas}/{scheme} cl{rank}: monotone '
                     f'{mono["n_monotone"]}/{mono["n_subjects"]} subjects '
                     f'(rho median {mono["rho_median"]:+.3f}), '
                     f'LOSO p_full {loso["p_full"]:.4f} -> worst '
                     f'{loso["p_worst"]:.4f}'
                     + ('   ' + row['Note'] if row['Note'] else ''))
    return rows
