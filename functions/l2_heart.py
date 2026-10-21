# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
l2_heart: L2 group stage for the HEART depth pipeline.

Takes the per-subject depth-slope maps produced by
``heart_depth_pipeline.subject_slopes`` (one ``(beta_map, svar_map, n_units)``
triple per subject per measure) and produces the group-level answer. The
measures are HEP, HRSP and HEPC; HEPC, as the code names it, is the paper's
HRPC (heartbeat-related phase consistency).

    per-subject slope maps
      -> listwise deletion of subjects with a non-finite map
      -> weighting schemes                        (unweighted / precision_nseg /
                                                   ivw_shrunk, optionally ivw_own)
      -> calibrated cluster permutation           (cluster_perm_weighted)
      -> tidy table, one row per measure x scheme x cluster
      -> `detected` decision by SIG_PRIORITY
      -> automatic CFA-window flagging

The cluster labelling is ``l2_run._clusters``, the shrunk inverse-variance
weights come from ``fmt_depth_pipeline.shrunk_ivw`` and the within-cluster
weighted readout from ``fmt_depth_pipeline.signflip_ttest``.

Calibration
-----------
  1. The cluster-forming threshold (CFT) is the (1-alpha) percentile of |t|
     under H0: the 95th at alpha = .05, not the 97.5th, which would leave only
     2.5% of null cells supra-threshold under a "p < .05" label.
  2. One null of the max absolute cluster mass over both polarities, not two
     one-sided nulls each at alpha: the union of two alpha-level events has FWER
     between alpha and 2*alpha (measured at .100 against a nominal .05).
Both live in ``cluster_perm_weighted``, which computes a weighted t (Kish
effective-n weighted t) and runs every scheme, so one table has one p
convention.

The observed statistic is not a member of its own null. Column 0 of the flip
matrix is the identity and supplies t_obs; ``cluster_perm_weighted`` excludes it
from the null, leaving n_perm - 1 random flips and a floor of 1/n_perm.

One matmul, no permutation loop over the map
--------------------------------------------
Under sign-flipping ``sum(w * X^2)`` is invariant and only the weighted mean
moves:

    SSw = sum_j w_j x_j^2                       (invariant)
    m   = ((X * w) @ flips) / Ws                (one matmul, all P flips at once)
    SS  = SSw - Ws * m^2                        (exact, see _null_t_weighted)

The only per-permutation loop left is connected-component labelling, which has
no closed form. ``_check_memory`` raises before the [nCell x nPerm] t-array can
exhaust memory.

CFA window (-150..150 ms), flagged automatically
------------------------------------------------
Any cluster whose time span reaches into -150..150 ms is flagged. That window
holds the QRS complex, so there the cardiac field and a heartbeat-evoked
response cannot be separated by latency. Surviving an ecgAmp / ecgAmpT
covariate does not clear the flag: those are scalar per-beat amplitudes, so
adjusting for them removes variance in how big the field is and never removes
the field itself.

Weighting schemes
-----------------
  unweighted      w = 1. The primary scheme.
  precision_nseg  w = the subject's segment count (subject_slopes' `n_units`).
                  The slope is fit over segments, so its precision depends on
                  how many units a subject contributed, not on how many beats
                  fell inside them.
  ivw_shrunk      w = 1/(v + median v), v = the subject's mean slope variance of
                  `ivw_measure` (default 'hep'). The same vector is used for
                  every measure, so the weighting cannot change underneath a
                  HEP-vs-HRSP-vs-HEPC comparison; changing `ivw_measure` breaks
                  that comparison, not just the numbers. Shrunk because v is
                  estimated: unshrunk, one subject whose variance came out small
                  by luck takes almost the whole weight.
  ivw_own         Not in WEIGHT_SCHEMES, so it runs only when named. The same
                  arithmetic with v from the measure being tested, in the same
                  model, over the same subject list (build_own_ivw_weights).
                  The weights are re-estimated on whatever list is passed (a LOSO
                  fold, a surrogate draw), shrinkage included, because weight
                  estimation is part of the procedure being resampled.

  ivw_shrunk is not in SIG_PRIORITY. It is a check on the headline, never a
  third chance to declare a detection; `decide_detected` removes it from any
  priority list it is given.

  Exact identification: cell_ols returns a finite slope with svar = Inf when
  nUnits == nP. Such a subject enters the unweighted mean but carries zero
  weight under ivw_shrunk and ivw_own, so the schemes in one table can rest on
  different effective samples. `N_subjects` and `N_eff_kish` are written on
  every row for that reason.

Conventions to know
-------------------
  * Clusters use 4-connectivity (``l2_run.CONN``): diagonal neighbours do not
    join a cluster.
  * Clusters are formed one polarity at a time (t >= thr and t <= -thr are
    labelled separately by ``l2_run._clusters``) and the mass is sum|t| (column
    `SumAbsT`). A cluster therefore never contains both polarities and
    `Mixed_polarity` is False on every row. The checks on it (no cluster-mean
    sign-flip readout on a mixed row, a warning naming any mixed survivor) are
    guards in case the labeller changes.
  * `Survived` is perm_p < alpha; no critical mass is returned. `Best_p` is the
    minimum p over all clusters of a measure x scheme.
  * A subject is dropped listwise from a measure if any cell of its map is
    non-finite (NaN or Inf): cluster_perm_weighted computes the null as one
    matmul over the whole map and cannot carry a per-cell weight mask. n_eff is
    therefore a scalar per measure x scheme, and the count is reported as
    `N_dropped_nonfinite`.
  * Permutations use numpy's default_rng (PCG64) seeded with SEED.
  * MIN_SUB_L2 = 5. Below it a measure is not tested and is recorded as
    Status = 'skipped:too_few_subjects' with n in the Note. A sign-flip null on
    4 subjects has 2^4 = 16 distinct flips, so the smallest attainable p is far
    from any usable alpha. The row is a skip, never a null; read `Status`.
  * Effect sizes: `CohenD`/`HedgesG` on N_subjects and `CohenD_neff`/
    `HedgesG_neff` on the Kish effective n, side by side; they are equal under
    `unweighted`. Hedges' J = 1 - 3/(4*(n-1) - 1), the one-sample correction on
    df = n-1, the same for every scheme.
  * Rows: when any cluster of a measure x scheme survives, only the survivors
    are written. Otherwise the top `max_clusters` (default 5) clusters by mass
    are written, because at a 5% cluster-forming threshold a 27 x 176 null map
    yields dozens of supra-threshold blobs by construction. `N_clusters_total`
    carries the full count on every row; `max_clusters=None` keeps everything.
  * This module is the group test only. Covariate models, post-hocs, source
    localization and plotting are elsewhere.

Usage:
    from heart_depth_pipeline import subject_slopes
    from l2_heart import run_l2_heart

    # one dict per subject:
    #  {'subject': sid, 'hep': subject_slopes(sessions, 'hep', p), ...}
    rows, detected = run_l2_heart(
        subjects, times_hep=t_hep, times_tf=t_tf, freqs=freqs,
        n_perm=1000, out_csv='l2_heart_summary.csv')
"""
from __future__ import annotations

import os
import sys
import csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# -- Shared parts, imported from l2_run --------------------------------------
# `_clusters` is the cluster-labelling step shared with the other L2 code.
# CFA_WINDOW_MS and SEED are imported so that each has one definition.
from l2_run import _clusters, CFA_WINDOW_MS, SEED  # noqa: E402

# -- The weighting and the weighted readout, imported from the FMT arm --------
from fmt_depth_pipeline import shrunk_ivw, signflip_ttest                 # noqa: E402

__all__ = [
    'WEIGHT_SCHEMES', 'SIG_PRIORITY', 'CFA_WINDOW_MS',
    'cfa_overlap', 'build_ivw_weights', 'build_own_ivw_weights',
    'build_weights', 'cluster_perm_weighted', 'run_l2_heart',
    'decide_detected', 'write_table', 'FIELDS', 'IVW_OWN', 'KNOWN_SCHEMES',
]

# ===========================================================================
# CONSTANTS
# ===========================================================================
WEIGHT_SCHEMES = ('unweighted', 'precision_nseg', 'ivw_shrunk')
# ivw_own is accepted everywhere a scheme is named but is not a default: the
# runner's L2_SCHEMES defaults to WEIGHT_SCHEMES. Ask for it by name.
IVW_OWN = 'ivw_own'
KNOWN_SCHEMES = WEIGHT_SCHEMES + (IVW_OWN,)

# Order matters, and ivw_shrunk is absent on purpose: the first scheme in this
# order with a surviving cluster sets the detection and its cluster table.
# ivw_shrunk is a robustness reading; putting it here would turn it into a third
# chance to declare an effect.
SIG_PRIORITY = ('precision_nseg', 'unweighted')

# Minimum subjects the group test will run on. Below this the t is not a t: a
# sign-flip null over 4 subjects has 2^4 = 16 distinct flips, so the attainable
# p is quantized far above any usable alpha. A measure below this is recorded as
# Status='skipped:too_few_subjects' with n in the Note; it is a skip, never a
# null.
MIN_SUB_L2 = 5

# Default ceiling on the [nCell x nPerm] null t-array, in GiB. See _check_memory.
MAX_NULL_GB = 4.0

# Which axes each measure lives on. HEP is 1-D over time; HRSP/HEPC are freq x time.
MEASURE_IS_TF = {'hep': False, 'hrsp': True, 'hepc': True}


# ===========================================================================
# CONSOLE OUTPUT THAT CANNOT KILL A RUN
# ===========================================================================
# Every warning in this module is printed after the compute and before
# write_table, and print() of a warning glyph raises UnicodeEncodeError on a
# non-UTF-8 stdout. On Windows a redirect, pipe, log file or subprocess capture
# is typically cp1252, which cannot encode U+26A0, so a run that found a
# CFA-window survivor could lose its CSV. All console output therefore goes
# through _emit: the words always survive; only the glyph degrades, and only on
# a console that cannot carry it.
_GLYPH_FALLBACK = (
    ('\u26a0\ufe0f', '[!]'), ('\u26a0', '[!]'),      # warning sign (+ VS16)
    ('\u2550', '='), ('\u2014', '-'), ('\u2013', '-'),
    ('\u2192', '->'), ('\u2265', '>='), ('\u2264', '<='),
    ('\u00d7', 'x'), ('\u2026', '...'), ('\u2019', "'"),
)


def _console_safe(msg):
    """`msg`, reduced to something the current sys.stdout can encode.

    sys.stdout.encoding is read at call time, never cached at import: a caller
    may redirect stdout after importing. A stream with no `.encoding` attribute (io.StringIO)
    accepts any str, so it is left untouched.
    """
    msg = str(msg)
    enc = getattr(sys.stdout, 'encoding', None)
    if not enc:
        return msg                      # StringIO and friends: no encoding step
    try:
        msg.encode(enc)
        return msg                      # UTF-8 console: keep every glyph
    except UnicodeEncodeError:
        pass
    except LookupError:                 # an encoding name Python does not know
        enc = 'ascii'
    for bad, good in _GLYPH_FALLBACK:
        msg = msg.replace(bad, good)
    # Backstop for any glyph not in the table. '?' is ugly; an exception is fatal.
    return msg.encode(enc, 'replace').decode(enc, 'replace')


def _emit(msg=''):
    """print() that cannot abort a run on a non-UTF-8 console. See _console_safe."""
    print(_console_safe(msg))


# ===========================================================================
# CFA WINDOW
# ===========================================================================
def cfa_overlap(start_ms, end_ms, window=CFA_WINDOW_MS):
    """True when [start_ms, end_ms] reaches into the CFA window.

    Overlap, not containment: a cluster running 100..400 ms has its peak well
    clear of the QRS and is still flagged, because its early edge is inside the
    complex and cluster mass is not attributable to a sub-range.

    Surviving an ecgAmp / ecgAmpT covariate does not clear this flag. Those
    covariates are scalar per-beat amplitudes; they remove variance in how big
    the cardiac field is, never the field.
    """
    return bool(start_ms <= window[1] and end_ms >= window[0])


# ===========================================================================
# INPUT MARSHALLING
# ===========================================================================
def _entry(subject, measure):
    """Normalise one subject's measure entry to (beta, svar, n_units) or None.

    ``heart_depth_pipeline.subject_slopes`` returns ``(None, None, 0)`` for a
    subject it refuses to fit, so both that and a plain ``None`` mean "no map".
    """
    e = subject.get(measure)
    if e is None:
        return None
    beta, svar, n_units = e
    if beta is None:
        return None
    return np.asarray(beta, float), np.asarray(svar, float), int(n_units)


def _stack_measure(subjects, measure):
    """Stack one measure across subjects.

    Returns
    -------
    X        [.. x nSubWithMap]  slope maps, trailing axis = subject
    svar     [nSubWithMap]       scalar slope variance per subject (nanmean of map)
    nseg     [nSubWithMap]       unit (segment) count per subject
    gidx     [nSubWithMap]       index back into `subjects`
    """
    betas, svars, nseg, gidx = [], [], [], []
    shape = None
    for i, s in enumerate(subjects):
        e = _entry(s, measure)
        if e is None:
            continue
        b, v, n = e
        if shape is None:
            shape = b.shape
        elif b.shape != shape:
            # A mismatched map means two different axes got mixed (e.g. one
            # subject fit on a trimmed time window). Reshaping or cropping here
            # would produce a plausible-looking group map over incomparable cells.
            raise ValueError(
                f'{measure}: subject {s.get("subject", i)} has slope map shape '
                f'{b.shape} but the cohort shape is {shape} - refusing to stack '
                f'maps on different axes.')
        betas.append(b)
        # nanmean over the whole map. Inf survives nanmean as Inf, which is what
        # we want: shrunk_ivw then gives that subject w = 0.
        with np.errstate(invalid='ignore'):
            svars.append(np.nanmean(v) if v is not None and v.size else np.nan)
        nseg.append(n)
        gidx.append(i)
    if not betas:
        return None, None, None, None
    X = np.stack(betas, axis=-1)
    return X, np.asarray(svars, float), np.asarray(nseg, float), np.asarray(gidx, int)


def _listwise_keep(X):
    """Subjects with a fully finite map.

    Listwise, not per cell: a subject with one non-finite cell (NaN or Inf) is
    dropped from the whole measure. cell_ols blanks a whole column to NaN when
    its input had a non-finite cell, and cluster_perm_weighted computes the null
    as one matmul over the map, so it cannot carry a per-cell weight mask and
    refuses a non-finite map outright. The drop is counted in
    `N_dropped_nonfinite` (not "N_dropped_nan", because Inf drops a subject too).
    """
    ax = tuple(range(X.ndim - 1))
    return np.isfinite(X).all(axis=ax)


# ===========================================================================
# WEIGHTS
# ===========================================================================
def build_ivw_weights(subjects, ivw_measure='hep', verbose=True,
                      label='ivw_shrunk'):
    """Shrunk inverse-variance weight per subject, over the full subject list.

    One weight vector for every measure, taken from `ivw_measure` (default
    'hep'), so that the weighting does not change underneath the comparison
    between measures. Per-measure weights would make a HEP-vs-HEPC difference
    partly a difference in who was weighted. (ivw_own does exactly that, on
    purpose, through build_own_ivw_weights; `label` only names the scheme in
    the messages.)

    Returns (w_all [nSubjects], info dict from shrunk_ivw).
    """
    sv = np.full(len(subjects), np.nan)
    for i, s in enumerate(subjects):
        e = _entry(s, ivw_measure)
        if e is None:
            continue
        v = e[1]
        if v is None or not np.size(v):
            continue
        with np.errstate(invalid='ignore'):
            sv[i] = np.nanmean(v)
    w_all, info = shrunk_ivw(sv, 'median')
    if info['nUsed'] == 0:
        # All-zero weights would make every ivw_shrunk cell come back "skipped",
        # which reads like a null. The likely cause is a configuration error:
        # ivw_measure names a measure this run does not carry (it defaults to
        # 'hep').
        if label != 'ivw_shrunk':
            raise ValueError(
                f"{label}: no subject has a usable slope variance for "
                f"measure {ivw_measure!r} (all NaN, Inf or <= 0), so no "
                f"inverse-variance weight can be formed.")
        raise ValueError(
            f"ivw_shrunk: no subject has a usable slope variance for "
            f"ivw_measure={ivw_measure!r} (all NaN, Inf or <= 0). These weights "
            f"come from the HEP slope variance for every measure, which is why "
            f"'hep' is the default. If "
            f"this run has no HEP maps, set ivw_measure explicitly - and note "
            f"that a per-measure ivw weighting makes the between-measure "
            f"comparison rest on different weights.")
    if verbose:
        n_zero = int(np.sum(w_all <= 0))
        _emit(f'  [{label}] {info["nUsed"]} of {len(subjects)} subjects weighted | '
              f'shrinkage c = {info["c"]:.3g} | effective n = {info["eff_n"]:.1f} | '
              f'heaviest = {info["wmax_ratio"]:.1f}x median')
        if n_zero:
            # Exact identification: cell_ols returns svar = Inf at nUnits == nP,
            # so the subject is fit, enters `unweighted`, and gets zero IVW
            # weight. Reported because it is otherwise silent.
            _emit(f'  [{label}] {n_zero} subject(s) carry ZERO weight '
                  f'(svar non-finite or <= 0 - usually cell_ols exact identification, '
                  f'svar = Inf). They still enter the UNWEIGHTED map.')
        if info['eff_n'] < 0.5 * max(info['nUsed'], 1):
            _emit(f'  [{label}] effective n is under half the real n - this '
                  'result rests on a subset of subjects. Read unweighted as primary.')
    return w_all, info


def build_own_ivw_weights(subjects, measure, verbose=False):
    """ivw_own: shrunk inverse-variance weights from `measure`'s own slope variance.

    The same arithmetic as build_ivw_weights (per-subject nanmean of the svar map
    over the whole map, shrunk_ivw with c = median of the usable values, a
    subject with no map / non-finite / <= 0 svar gets 0) with ivw_measure =
    the measure being tested, over the same subject list. Returns (w_all
    [nSubjects], info). Raises ValueError when no subject has a usable
    variance for that measure.

    The weights are a function of the list given. The shrinkage c is the median
    over these subjects, so a reduced list (a LOSO fold, a surrogate draw's own
    L1) re-estimates the whole weight vector rather than inheriting the full
    sample's. Callers that want the procedure resampled must pass the reduced
    list; l2_tfce.prepare_measure always does.
    """
    return build_ivw_weights(subjects, measure, verbose, label=IVW_OWN)


def build_weights(scheme, nseg, ivw_w=None):
    """Per-subject weight vector for one scheme, already subset to the kept subjects.

    Weights are rescaled to mean 1. The weighted t is scale-invariant, but the
    rescale makes Ws == n under uniform weights, which is what makes the weighted
    t reduce exactly to the unweighted t (see _null_t_weighted).

    Raises rather than substituting a default: a broken weight vector must not be
    able to look like a null result.
    """
    scheme = scheme.lower()
    if scheme == 'unweighted':
        w = np.ones(len(nseg), float)
    elif scheme == 'precision_nseg':
        w = np.asarray(nseg, float).copy()
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError(
                'precision_nseg: segment counts must be finite and > 0, got '
                f'{w[~(np.isfinite(w) & (w > 0))]} - subject_slopes returns '
                'n_units = 0 for a subject it refused; that subject should never '
                'have reached the stack.')
    elif scheme == 'ivw_shrunk':
        if ivw_w is None:
            raise ValueError("ivw_shrunk requested but no ivw weights supplied "
                             "(build_ivw_weights was not run).")
        w = np.asarray(ivw_w, float).copy()
        w[~np.isfinite(w)] = 0.0
        w[w < 0] = 0.0
    elif scheme == IVW_OWN:
        # same handling as ivw_shrunk; only where the vector came from differs
        # (build_own_ivw_weights over the TESTED measure, passed in as ivw_w)
        if ivw_w is None:
            raise ValueError("ivw_own requested but no own-measure weights "
                             "supplied (build_own_ivw_weights was not run).")
        w = np.asarray(ivw_w, float).copy()
        w[~np.isfinite(w)] = 0.0
        w[w < 0] = 0.0
    else:
        raise ValueError(f'Unknown weighting scheme: {scheme!r}. '
                         f'Known: {KNOWN_SCHEMES}')
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        raise ZeroDivisionError(
            f'{scheme}: all weights are zero - nothing to test. Under ivw_shrunk '
            'this means every subject had a non-finite slope variance.')
    return w * (len(w) / tot)          # mean 1


def kish_n_eff(w):
    """Kish effective sample size, (sum w)^2 / sum(w^2). Equals n exactly when w
    is uniform."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


# ===========================================================================
# WEIGHTED CLUSTER PERMUTATION
# ===========================================================================
def _check_memory(n_cell, n_perm, max_gb=MAX_NULL_GB):
    """Refuse to allocate a null t-array bigger than max_gb.

    The null is one matmul, but its result is [nCell x nPerm] float64 and that is
    what gets big: 27 freqs x 176 times x 10k perms = 380 MB, a 64-channel map
    would be 9 GB. Raise with the number rather than exhaust memory.
    """
    gb = n_cell * n_perm * 8 / (1 << 30)
    if gb > max_gb:
        raise MemoryError(
            f'null t-array would be {gb:.2f} GiB ({n_cell} cells x {n_perm} '
            f'permutations x 8 B) which exceeds max_gb={max_gb}. Reduce n_perm, '
            f'restrict the window, or raise max_gb deliberately - do NOT replace '
            f'the matmul with a permutation loop.')
    return gb


def _null_t_weighted(X2d, w, flips):
    """Weighted sign-flip t-maps for every flip at once, in one matmul.

    X2d [nCell x nSub], w [nSub] (mean 1), flips [nSub x P] of +-1.

    The weighted group t:

        Ws    = sum(w)                    m     = sum(w*x*s)/Ws
        SS    = sum(w*(x*s - m)^2)        n_eff = Ws^2/sum(w^2)
        var_g = SS/((n_eff-1)*Ws)         t     = m/sqrt(var_g)

    and the identity that makes it one matmul:

        SS = sum(w*x^2) - 2*m*sum(w*x*s) + m^2*Ws
           = SSw        - 2*m*(m*Ws)     + m^2*Ws
           = SSw - Ws*m^2                          <- SSw invariant under s

    It reduces to the unweighted t of heart_scalp_pipeline._null_t at w == 1.
    With w uniform: Ws = n, n_eff = n, SSw = sum(x^2), so var_w = SS/(n-1) and
    t = m/sqrt(var_w/n), the same two lines. The clip is on var_w (the
    variance), as in that function's ``np.clip(var, 1e-30, ...)``.
    """
    Ws = float(w.sum())
    n_eff = Ws ** 2 / float(np.sum(w ** 2))
    SSw = (X2d ** 2) @ w                       # [nCell]   invariant under flips
    m = ((X2d * w) @ flips) / Ws               # [nCell x P]  <- THE matmul
    SS = SSw[:, None] - Ws * m ** 2
    var_w = SS * n_eff / ((n_eff - 1.0) * Ws)  # weighted unbiased variance s_w^2
    np.clip(var_w, 1e-30, None, out=var_w)
    return m / np.sqrt(var_w / n_eff)


def _make_flips(n_sub, n_perm, seed):
    """Sign-flip matrix [nSub x nPerm], column 0 = identity (the observed map).

    Column 0 is the observed map, not a null draw. It exists so that t_obs comes
    out of the same matmul as the null, on the same arithmetic path. It must not
    enter the null distribution of cluster masses: cluster_perm_weighted
    excludes it, so the null there has n_perm - 1 members.
    """
    rng = np.random.default_rng(seed)
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_sub, n_perm))
    flips[:, 0] = 1.0
    return flips


def cluster_perm_weighted(X, w, n_perm=1000, seed=SEED, max_gb=MAX_NULL_GB,
                          alpha=0.05):
    """Calibrated cluster permutation with per-subject weights.

    Contract and return shape:
        X : [nF x nT x nSub] (nF may be 1) or [nT x nSub]
        -> (t_obs, cft, clusters sorted by mass desc, n_used)

    X must already be NaN-free in the subject axis (call _listwise_keep first);
    this does not drop subjects itself, because it cannot drop the matching
    weights.

    alpha : sets both the cluster-forming threshold, the (1-alpha) percentile of
        |t| under H0, and, through the caller, which clusters count as survivors.
        It is not only a label: a threshold fixed at the 95th percentile with
        alpha flipping only `Survived` would report "alpha = .01" over a .05
        threshold.

    The null excludes the observed map. Column 0 of the flip matrix is the
    identity and supplies t_obs, so the mass null runs over columns 1.. only and
    has n_perm - 1 members; the smallest attainable p is 1/n_perm.
    """
    X = np.asarray(X, float)
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f'alpha must be in (0, 1), got {alpha!r}. It sets the cluster-forming '
            f'threshold, not just a label on the output.')
    if n_perm < 2:
        # With n_perm = 1 the null would be empty and every cluster would come
        # back p = 1/1 = 1.0, a clean-looking null.
        raise ValueError(
            f'n_perm = {n_perm}: column 0 is the identity (it supplies t_obs), so '
            f'the null has n_perm - 1 members - n_perm must be at least 2, and '
            f'anything below a few hundred cannot resolve a usable p.')
    if X.ndim == 2:                              # HEP: [nT x nSub] -> [1 x nT x nSub]
        X = X[None]
    nF, nT, nSub = X.shape
    if nSub != len(w):
        raise ValueError(f'{nSub} subjects in X but {len(w)} weights.')
    if not np.isfinite(X).all():
        raise ValueError('cluster_perm_weighted got non-finite cells - apply '
                         '_listwise_keep to X and w together before calling.')
    _check_memory(nF * nT, n_perm, max_gb)

    X2d = X.reshape(nF * nT, nSub)
    flips = _make_flips(nSub, n_perm, seed)
    T = _null_t_weighted(X2d, np.asarray(w, float), flips)

    t_obs = T[:, 0].reshape(nF, nT)            # column 0 IS the identity flip
    # (1-alpha) percentile of |t| under H0: the 95th at the default alpha, not
    # the 97.5th. The percentile pools all n_perm columns, the identity included.
    # This is a threshold estimate, not an inference, and one column in n_perm
    # cannot move it materially. The p-value below is where including the
    # observed map would matter.
    cft = float(np.percentile(np.abs(T), 100.0 * (1.0 - alpha)))

    # One null, of max absolute cluster mass, not two one-sided nulls.
    # _clusters (imported) labels t >= cft and t <= -cft separately and returns
    # sum|t| per component.
    # The observed map is not in it: the loop starts at 1. Including column 0
    # would make the observed statistic a member of its own null, and the add-one
    # below would count it a second time (floor 2/(n_perm+1) instead of 1/n_perm).
    n_null = n_perm - 1
    null_max = np.empty(n_null)
    for p in range(1, n_perm):
        m, _ = _clusters(T[:, p].reshape(nF, nT), cft)
        null_max[p - 1] = m.max() if m.size else 0.0

    obs, lab = _clusters(t_obs, cft)
    out = []
    n_eff = kish_n_eff(w)
    for ci in np.argsort(obs)[::-1]:
        mass = obs[ci]
        cells = np.argwhere(lab == ci + 1)
        pk = cells[np.argmax(np.abs(t_obs[cells[:, 0], cells[:, 1]]))]
        out.append(dict(
            # add-one permutation p (Phipson & Smyth 2010) over the n_perm - 1
            # random flips; floor 1/n_perm.
            p=(1 + int((null_max >= mass).sum())) / (1 + n_null), mass=float(mass),
            f0=int(cells[:, 0].min()), f1=int(cells[:, 0].max()),
            t0=int(cells[:, 1].min()), t1=int(cells[:, 1].max()),
            pf=int(pk[0]), pt=int(pk[1]), n_cells=len(cells),
            # The exact cell list, not just the bounding box. A cluster is a
            # connected set; (f0..f1, t0..t1) is its bounding box and can contain
            # cells that are not in it, so a post-hoc that summarises "the
            # cluster" from the box would average cells the test never selected.
            # [n_cells x 2] int, columns (freq_index, time_index).
            cells=cells.astype(np.int32, copy=False),
            # `g` here is on the Kish effective n with J = 1 - 3/(4*(n-1)-1).
            # The table recomputes both readings (raw n and Kish n) for every
            # scheme.
            g=float(t_obs[pk[0], pk[1]] / np.sqrt(n_eff) * _hedges_J(n_eff))))
    return t_obs, cft, out, nSub


def _hedges_J(n_use):
    """Hedges' small-sample correction on df = n-1.

    J = 1 - 3/(4*(n_use-1) - 1). n_use may be non-integer (a Kish effective n);
    J is continuous in df, which is fine.
    """
    if not np.isfinite(n_use) or n_use <= 2:
        return np.nan
    return 1.0 - 3.0 / (4.0 * (n_use - 1.0) - 1.0)


# ===========================================================================
# THE TIDY TABLE
# ===========================================================================
FIELDS = (
    # N_eff_kish (always the Kish effective n), N_dropped_nonfinite (NaN or Inf
    # drops a subject) and SumAbsT (sum|t|, not a signed sum) are named for
    # exactly what they hold.
    'Measure', 'Scheme', 'Status', 'N_subjects', 'N_dropped_nonfinite', 'N_eff_kish',
    'CFT', 'N_supra', 'N_map', 'N_clusters_total', 'Best_p',
    'Cluster', 'Polarity', 'Mixed_polarity',
    'Start_ms', 'End_ms', 'Peak_ms', 'Start_Hz', 'End_Hz', 'Peak_Hz',
    'Peak_t', 'SumAbsT', 'perm_p', 'N_cells', 'Survived',
    'CohenD', 'HedgesG', 'CohenD_neff', 'HedgesG_neff',
    'Signflip_t', 'Signflip_p',
    'CFA_flag', 'Note',
)


def _blank_row(measure, scheme, status, n_sub, n_drop, n_eff, note=''):
    """A non-cluster row: a tested-but-null cell, or a cell that was not tested.

    `Status` is what separates the two, and it is the only thing that does. Never
    read a blank `perm_p` as evidence of a null; check `Status` first.
    """
    r = {k: '' for k in FIELDS}
    r.update(Measure=measure, Scheme=scheme, Status=status, N_subjects=n_sub,
             N_dropped_nonfinite=n_drop, Cluster=0, Survived=False,
             Mixed_polarity=False, CFA_flag=False, Note=note,
             N_eff_kish=round(n_eff, 3) if np.isfinite(n_eff) else '')
    return r


def write_table(rows, path):
    """Write the tidy table. Every row carries `Status`, so a null and a skip are
    distinguishable: a null row must never be mistaken for a tested one."""
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(FIELDS))
        wr.writeheader()
        wr.writerows(rows)
    return path


# ===========================================================================
# MAIN DRIVER
# ===========================================================================
def run_l2_heart(subjects, times_hep=None, times_tf=None, freqs=None,
                 measures=('hep', 'hrsp', 'hepc'), schemes=WEIGHT_SCHEMES,
                 n_perm=1000, alpha=0.05, seed=SEED, ivw_measure='hep',
                 max_gb=MAX_NULL_GB, out_csv=None, verbose=True,
                 posthoc_signflip=True, max_clusters=5, clusters_out=None):
    """L2 group stage: cluster permutation x three weighting schemes.

    Parameters
    ----------
    subjects : list of dict
        One dict per subject::

            {'subject': 'sub-01',
             'hep':  (beta_map, svar_map, n_units),   # subject_slopes' return
             'hrsp': (beta_map, svar_map, n_units),
             'hepc': (beta_map, svar_map, n_units)}

        A measure may be absent or ``None`` or the literal ``(None, None, 0)``
        that ``subject_slopes`` returns for a subject it refused; all three mean
        "this subject has no map for this measure" and it is excluded from that
        measure only.
    times_hep : [nT] ms axis for HEP.  times_tf, freqs : axes for HRSP / HEPC.
    schemes   : subset of KNOWN_SCHEMES (WEIGHT_SCHEMES + 'ivw_own'), in the
                order to run them. The default stays WEIGHT_SCHEMES.
    n_perm : sign-flip permutations. Column 0 of the flip matrix is the identity
        and supplies t_obs, so the mass null has n_perm - 1 members and the
        smallest attainable p is 1/n_perm (0.001 at the default n_perm = 1000):
        the observed map is not a member of its own null.
    clusters_out : optional dict. If given, it is filled with
        ``{(measure, scheme): [cluster dicts]}``, the raw clusters, each
        carrying its exact ``cells`` array. Use it when a post-hoc must operate
        on the cluster itself rather than on its bounding box.
    alpha : cluster-level alpha. It sets the cluster-forming threshold too (the
        (1-alpha) percentile of |t| under H0), not just which clusters come back
        `Survived`. A run at alpha=0.01 therefore returns a different CFT, a
        different N_supra and different clusters, not the same table with a
        stricter flag.
    max_gb : ceiling on the [nCell x nPerm] null t-array, enforced for EVERY
        scheme before anything is allocated. Raises MemoryError with the number.
    ivw_measure : which measure's slope variance defines the ivw weights; 'hep'
        by default, for every measure. Change it only if you intend the
        weighting to differ between measures.
    posthoc_signflip : run the weighted sign-flip readout on SURVIVING clusters
        only. See the Signflip_t note below.
    max_clusters : when no cluster of a measure x scheme survives, how many
        clusters to keep, ranked by mass. When any cluster survives, only the
        survivors are kept, all of them. A 27x176 noise map throws ~40
        supra-threshold blobs at a 5% CFT by construction, and a table full of
        p=1.000 rows buries the one row anyone reads. The truncation is
        recorded: `N_clusters_total` carries the full count on every row and
        `Best_p` is the minimum over all clusters, truncated ones included. Set
        None to keep everything.

    Returns
    -------
    (rows, detected): the tidy table as a list of dicts, and the per-measure
    detection decision from `decide_detected`.
    """
    schemes = tuple(s.lower() for s in schemes)
    for s in schemes:
        if s not in KNOWN_SCHEMES:
            raise ValueError(f'Unknown scheme {s!r}; known: {KNOWN_SCHEMES}')
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f'alpha must be in (0, 1), got {alpha!r}. It sets the cluster-forming '
            f'threshold AND the survival rule - see the alpha note below.')

    axes = {}
    for m in measures:
        if MEASURE_IS_TF.get(m, True):
            axes[m] = (np.asarray(times_tf, float) if times_tf is not None else None,
                       np.asarray(freqs, float) if freqs is not None else None)
        else:
            axes[m] = (np.asarray(times_hep, float) if times_hep is not None else None,
                       None)

    ivw_all = None
    if 'ivw_shrunk' in schemes:
        ivw_all, _ivw_info = build_ivw_weights(subjects, ivw_measure, verbose)

    rows = []
    for measure in measures:
        X, svar, nseg, gidx = _stack_measure(subjects, measure)
        tms, frs = axes[measure]
        if X is None:
            if verbose:
                _emit(f'\n== {measure.upper()} == no subject supplied a map - skipped')
            for sch in schemes:
                rows.append(_blank_row(measure, sch, 'skipped:no_maps', 0, 0, np.nan,
                                       'no subject supplied a map for this measure'))
            continue

        keep = _listwise_keep(X)
        n_drop = int((~keep).sum())
        Xk = X[..., keep]
        nsegk, gidxk = nseg[keep], gidx[keep]
        n_sub = int(keep.sum())
        # ivw_own: THIS measure's own slope variance, over the full subject
        # list (as build_ivw_weights is for ivw_shrunk), subset to the kept
        # subjects below. A measure with no usable variance makes the ivw_own
        # row 'skipped:bad_weights' through build_weights, never a crash.
        own_all, own_err = None, None
        if IVW_OWN in schemes:
            try:
                own_all, _own_info = build_own_ivw_weights(subjects, measure,
                                                           verbose)
            except ValueError as exc:
                own_err = exc

        # axis sanity: a wrong time axis silently relabels every cluster
        n_t = Xk.shape[-2] if Xk.ndim == 3 else Xk.shape[0]
        if tms is None or len(tms) != n_t:
            raise ValueError(
                f'{measure}: time axis has {0 if tms is None else len(tms)} points '
                f'but the slope map has {n_t}. Refusing to label clusters in ms '
                f'against the wrong axis.')
        if Xk.ndim == 3:
            if frs is None or len(frs) != Xk.shape[0]:
                raise ValueError(
                    f'{measure}: freq axis has {0 if frs is None else len(frs)} bins '
                    f'but the slope map has {Xk.shape[0]}.')

        if verbose:
            _emit(f'\n{"=" * 78}\n  L2 - {measure.upper()}   '
                  f'n = {n_sub} subjects'
                  + (f'  ({n_drop} dropped: a non-finite cell somewhere in the '
                     f'map)' if n_drop else '')
                  + f'\n{"=" * 78}')

        if n_sub < MIN_SUB_L2:
            if verbose:
                _emit(f'  n = {n_sub} < MIN_SUB_L2 = {MIN_SUB_L2} - not tested')
            for sch in schemes:
                rows.append(_blank_row(measure, sch, 'skipped:too_few_subjects',
                                       n_sub, n_drop, np.nan,
                                       f'n={n_sub} below MIN_SUB_L2={MIN_SUB_L2}'))
            continue

        for sch in schemes:
            try:
                if sch == IVW_OWN:
                    if own_err is not None:
                        raise own_err
                    w = build_weights(sch, nsegk, own_all[gidxk])
                else:
                    w = build_weights(sch, nsegk,
                                      None if ivw_all is None else ivw_all[gidxk])
            except (ValueError, ZeroDivisionError) as exc:
                # A scheme that cannot be built is recorded as SKIPPED with the
                # reason, never as a null.
                if verbose:
                    _emit(f'  {sch:15s}: SKIPPED - {exc}')
                rows.append(_blank_row(measure, sch, 'skipped:bad_weights',
                                       n_sub, n_drop, np.nan, str(exc)))
                continue

            n_eff = kish_n_eff(w)
            # All schemes go through one driver, 'unweighted' included, so that
            # max_gb guards every scheme, alpha reaches the cluster-forming
            # threshold of every scheme, and one table carries one p convention.
            t_obs, cft, cl, n_used = cluster_perm_weighted(
                Xk, w, n_perm, seed, max_gb, alpha)

            n_supra = int(np.sum(np.abs(t_obs) >= cft))   # >= : the cluster rule
            n_map = int(t_obs.size)
            best_p = float(min(c['p'] for c in cl)) if cl else np.nan

            if not cl:
                if verbose:
                    _emit(f'  {sch:15s}: no supra-threshold cluster '
                          f'(CFT={cft:.3f}, {n_supra}/{n_map} supra)')
                r = _blank_row(measure, sch, 'no_supra', n_used, n_drop, n_eff,
                               'no cell crossed the cluster-forming threshold')
                r.update(CFT=round(cft, 4), N_supra=n_supra, N_map=n_map,
                         N_clusters_total=0, Best_p='')
                rows.append(r)
                continue

            _, lab = _clusters(t_obs, cft)
            X3 = Xk[None] if Xk.ndim == 2 else Xk
            n_cl_total = len(cl)
            # Hand the raw clusters to the caller if asked. The CSV rows carry
            # only the bounding box (Start_ms/End_ms/Start_Hz/End_Hz), and a
            # cluster is a connected set whose box can contain cells the test
            # never selected. `clusters_out` is an opt-in dict; nothing changes
            # for callers that do not pass one.
            if clusters_out is not None:
                clusters_out[(measure, sch)] = cl
            # When anything survives, only survivors are emitted: non-surviving
            # rows next to a real one are indistinguishable at a glance, and each
            # would pick up its own LOSO and depth-shape row in the post-hoc
            # summary. When nothing survives the top-N are kept, which is the
            # only way to see what the map nearly did.
            _any_surv = any(c['p'] < alpha for c in cl)
            new_rows = []
            for rank, c in enumerate(cl, 1):
                # Keep every SURVIVOR, plus the top `max_clusters` by mass. See
                # the max_clusters docstring: the count survives in
                # N_clusters_total and Best_p is over the untruncated set.
                if _any_surv and not (c['p'] < alpha):
                    continue
                truncate = (max_clusters is not None and rank > max_clusters)
                if truncate and not (c['p'] < alpha):
                    continue
                new_rows.append(_cluster_row(
                    measure, sch, c, rank, t_obs, lab, X3, w, tms, frs,
                    n_used, n_drop, n_eff, cft, n_supra, n_map, n_cl_total,
                    best_p, alpha, posthoc_signflip, n_perm))
            rows.extend(new_rows)

            if verbose:
                # Print the top cluster and every survivor; the rest are the
                # ~5%-of-the-map blobs the CFT produces on any null map.
                shown = [r for r in new_rows if r['Cluster'] == 1 or r['Survived']]
                for r in shown:
                    star = '*' if r['Survived'] else ' '
                    hz = ('' if r['Start_Hz'] == '' else
                          f' {r["Start_Hz"]:.0f}-{r["End_Hz"]:.0f} Hz')
                    _emit(f'  {sch:15s}: cl{r["Cluster"]} p={r["perm_p"]:.4f}{star} '
                          f'g={r["HedgesG"]:+.3f} '
                          f'{r["Start_ms"]:.0f}..{r["End_ms"]:.0f} ms{hz}'
                          f'{"   CFA WINDOW" if r["CFA_flag"] else ""}')
                if n_cl_total > len(shown):
                    _emit(f'  {"":15s}  (+{n_cl_total - len(shown)} further '
                          f'supra-threshold cluster(s), best p={best_p:.4f}; '
                          f'{n_supra}/{n_map} cells supra at CFT={cft:.3f})')

    detected = decide_detected(rows, verbose=verbose)

    # Mixed-polarity survivors. A labeller that clusters on |t| can fuse a
    # positive and a negative blob into one component, whose extent and effect
    # size then do not describe one effect. l2_run._clusters labels each polarity
    # separately, so with it this warning does not fire; it is a guard.
    mixed_surv = [r for r in rows if r['Status'] == 'ok' and r['Survived']
                  and r['Mixed_polarity']]
    if mixed_surv:
        _emit('\n MIXED-POLARITY SURVIVING CLUSTER(S): '
              + ', '.join(f'{r["Measure"]}/{r["Scheme"]} cl{r["Cluster"]}'
                          for r in mixed_surv))
        _emit('    the shared labeller clusters on |t|, so opposite-signed blobs')
        _emit('    can fuse into one component.')
        _emit('    The p is valid; the span, Peak_t and effect size')
        _emit('    are NOT a single effect. The cluster-mean sign-flip readout is')
        _emit('    suppressed on these rows (the mean over opposite signs cancels).')

    # The console warning fires only for surviving flagged clusters. The column
    # is set on every cluster row, but a sub-threshold blob near 0 ms is not a
    # finding and warning about it would train the reader to ignore the warning.
    if any(r['CFA_flag'] and r['Survived'] for r in rows):
        _emit('\n[!]  CFA-WINDOW CLUSTER(S) FLAGGED. That window (-150..150 ms) holds')
        _emit('    the QRS complex, so the cardiac field and a neural effect are NOT')
        _emit('    separable by latency here. A tonic (sustained) power difference')
        _emit('    between conditions also gives a broad cluster starting near 0 ms,')
        _emit('    because every epoch inherits the ongoing difference. ecgAmp and')
        _emit('    ecgAmpT do not settle it either: they are scalar per-beat')
        _emit('    amplitudes and remove variance in how big the field is, never the')
        _emit('    field. Separate the two with evidence that is not latency: the CFA')
        _emit('    topography correlation, the same test on epochs cut at times')
        _emit('    unrelated to the R-peak, and the surrogate (shifted R-peaks).')

    if out_csv:
        write_table(rows, out_csv)
        if verbose:
            _emit(f'\n{len(rows)} rows -> {out_csv}')
    return rows, detected


def _cluster_row(measure, scheme, c, rank, t_obs, lab, X3, w, tms, frs,
                 n_used, n_drop, n_eff, cft, n_supra, n_map, n_cl_total,
                 best_p, alpha, posthoc_signflip, n_perm):
    """Build one tidy row for one cluster."""
    cid = int(lab[c['pf'], c['pt']])
    cells = (lab == cid)
    t_in = t_obs[cells]
    peak_t = float(t_obs[c['pf'], c['pt']])

    # Always False with l2_run._clusters, which labels each polarity separately;
    # kept as a guard.
    mixed = bool((t_in >= cft).any() and (t_in <= -cft).any())

    start_ms, end_ms = float(tms[c['t0']]), float(tms[c['t1']])
    peak_ms = float(tms[c['pt']])
    if frs is not None:
        s_hz, e_hz, p_hz = float(frs[c['f0']]), float(frs[c['f1']]), float(frs[c['pf']])
    else:
        s_hz = e_hz = p_hz = ''

    survived = bool(c['p'] < alpha)

    # Effect sizes at the PEAK cell, BOTH readings (see module header).
    d_obs = peak_t / np.sqrt(n_used)
    d_eff = peak_t / np.sqrt(n_eff)

    sf_t = sf_p = ''
    if posthoc_signflip:
        # Gated on survival, and circular by construction. A post-hoc runs only
        # if the effect survived the main model, and only on that effect. This is
        # a weighted readout of the same cells the cluster test selected, so it
        # is a descriptive effect statement, never independent confirmation.
        if survived and not mixed:
            beta_sub = X3[cells].mean(axis=0)          # [nSub] cluster-mean slope
            ok = np.isfinite(beta_sub)
            if ok.sum() >= MIN_SUB_L2 and np.sum(w[ok]) > 0:
                t_v, p_v = signflip_ttest(beta_sub[ok], w[ok], n_perm)
                sf_t = '' if not np.isfinite(t_v) else round(float(t_v), 4)
                sf_p = '' if not np.isfinite(p_v) else round(float(p_v), 5)
        # mixed-polarity clusters get NO sign-flip: the mean over cells of
        # opposite sign cancels and the number would be meaningless. Left blank
        # with the reason in Note, never silently zeroed.

    note = ''
    if mixed:
        note = ('mixed-polarity cluster (|t| clustering can join tails) - '
                'cluster-mean sign-flip suppressed')

    return {
        'Measure': measure, 'Scheme': scheme, 'Status': 'ok',
        'N_subjects': n_used, 'N_dropped_nonfinite': n_drop,
        'N_eff_kish': round(n_eff, 3),
        'CFT': round(cft, 4), 'N_supra': n_supra, 'N_map': n_map,
        'N_clusters_total': n_cl_total, 'Best_p': round(best_p, 5),
        'Cluster': rank,
        'Polarity': 'pos' if peak_t > 0 else 'neg',
        'Mixed_polarity': mixed,
        'Start_ms': start_ms, 'End_ms': end_ms, 'Peak_ms': peak_ms,
        'Start_Hz': s_hz, 'End_Hz': e_hz, 'Peak_Hz': p_hz,
        'Peak_t': round(peak_t, 4),
        'SumAbsT': round(float(c['mass']), 4),    # sum|t|, not a signed sum
        'perm_p': round(float(c['p']), 5),
        'N_cells': int(c['n_cells']),
        'Survived': survived,
        'CohenD': round(float(d_obs), 4),
        'HedgesG': round(float(d_obs * _hedges_J(n_used)), 4),
        'CohenD_neff': round(float(d_eff), 4),
        'HedgesG_neff': round(float(d_eff * _hedges_J(n_eff)), 4),
        'Signflip_t': sf_t, 'Signflip_p': sf_p,
        'CFA_flag': cfa_overlap(start_ms, end_ms),
        'Note': note,
    }


# ===========================================================================
# DETECTION
# ===========================================================================
def decide_detected(rows, sig_priority=SIG_PRIORITY, verbose=True):
    """Which measures count as detected, and under which scheme.

    The schemes are tried in priority order and the first one with a surviving
    cluster sets `.scheme` and supplies the cluster table. Every scheme is still
    reported; this only decides which one the post-hocs run on.

    ivw_shrunk cannot set a detection, even if it is passed in sig_priority. It
    is a robustness reading (a check on the headline), not a third chance to
    declare an effect. Passing it in is treated as a mistake and warned about.
    """
    prio = [s for s in sig_priority if s != 'ivw_shrunk']
    if len(prio) != len(sig_priority):
        _emit('  ivw_shrunk was passed in sig_priority and has been REMOVED: '
              'it is a robustness reading, never a detection scheme.')

    measures = []
    for r in rows:
        if r['Measure'] not in measures:
            measures.append(r['Measure'])

    detected = {}
    for meas in measures:
        d = {'sig': False, 'scheme': '', 'clusters': []}
        for sch in prio:
            hits = [r for r in rows
                    if r['Measure'] == meas and r['Scheme'] == sch
                    and r['Status'] == 'ok' and r['Survived']]
            if hits:
                d = {'sig': True, 'scheme': sch, 'clusters': hits}
                break
        detected[meas] = d

    if verbose:
        _emit(f'\n  DETECTED (priority {" > ".join(prio)}):')
        for meas, d in detected.items():
            if d['sig']:
                flag = ' CFA WINDOW' if any(c['CFA_flag'] for c in d['clusters']) else ''
                _emit(f'    {meas.upper():5s} : YES ({d["scheme"]}, '
                      f'{len(d["clusters"])} cluster(s), '
                      f'best p={min(c["perm_p"] for c in d["clusters"]):.4f}){flag}')
            else:
                _emit(f'    {meas.upper():5s} : none')
    return detected
