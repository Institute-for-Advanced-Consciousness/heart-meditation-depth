# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Shifted-R-peak surrogates for the heartbeat-locking control.

Builds a rigidly time-shifted copy of a session's R-peak train. A rigid shift preserves the IBI
sequence and its variability exactly, and preserves the beat count: the properties Steinfath et
al. (2026, Box 1 step 6A) require of a valid surrogate. Per-beat jitter would destroy the IBI
variability structure and is not equivalent. Follows Park et al. (2018), who shifted by
-500..+500 ms while preserving "the same interbeat interval (IBI) and variability"; Lee et al.
(2024) use the same +/-500 ms range. `heart_surrogate.py` uses these trains to re-epoch, refit
L1 and build the L2 null.

WHAT THIS CONTROL DOES AND DOES NOT TEST
  Does:     whether an effect is locked to the heartbeat at all, as opposed to ongoing activity
            that co-varies with depth and would show up at any arbitrary time point.
  Does not: separate neural HEP from cardiac field artifact. Both are R-peak locked, so both
            vanish under shifting. Use the 'ecgAmp' covariate model for that attribution.

NOT THE SAME AS `lee_selection.make_sham_shifts`
Confusing the two destroys the independence of the surrogate control. They are different
objects at different stages of the analysis:

                    make_sham_shifts                 make_surrogate_rpeaks
                    (lee_selection)                  (this module)
  purpose           normalises the IC-selection      controls the depth effect:
                    score: z the box statistic       is the L1/L2 result locked
                    against a sham distribution      to the heartbeat at all?
  stage             inside IC selection, i.e.        after L1 has been fit, per
                    before L1 exists                 subject, then refit
  returns           n_sh shift values (samples),     one shifted R-peak train
                    applied to an epoch-index        per call; the caller
                    matrix, ~200 of them at once     re-epochs and refits
  rejection band    fixed min_shift_ms=150           data-driven: 0.25 * median
                                                     IBI, capped at half the
                                                     largest available shift
  bounds            none per draw; out-of-range      per-draw rejection against
                    beats are dropped wholesale      an epoch guard, up to 100
                    via keep_b                       attempts, and it can fail
  seeding           one fixed seed for the whole     one seed per (surrogate k,
                    set (20260728)                   session j, retry t)
  used by           lee_selection (IC selection)     heart_surrogate.py

The two are also not independent of each other. If IC selection ran with a sham norm mode
(IC_SEL_NORM = 'sham' or 'sham_raw'), the component was chosen for having a large
real-vs-shifted contrast. Testing that same component with this surrogate then asks whether the
thing selection optimised for is present, and the answer is rigged; heart_surrogate.py refuses
such a run. (Under criterion='ppc' the norm mode has no effect, so a ppc run is not
sham-selected whatever the folder tag says; see lee_selection.)

Cached transforms
Any CWT computed for surrogate peaks must come from a continuous transform held in memory (or an
explicitly cache-disabled call), never from an on-disk epoched CWT cache. Such a cache is keyed
by subject/session/IC/epoch-window/freqs/width and not by R-peak positions, and it stores the
already-epoched CWT, so a surrogate call would hit it and return the real epochs. No size check
catches this (the beat count is unchanged by construction), and the null would then equal the
real data and appear to confirm every effect. The defence is an integrity check: the real train
re-epoched through the surrogate path must reproduce the main-loop arrays before any surrogate
is trusted. This module never sees the EEG, so the check belongs to the caller
(heart_surrogate.py runs it).

GRANULARITY (it bounds the smallest p)
The number of distinct surrogates is bounded by the number of admissible integer sample shifts,
not by how many draws are requested. At fs=250 with shift_ms=(-500, 500) and a 60 bpm IBI the
admissible band is |shift| in [62, 125] samples, i.e. 128 distinct values, so asking for many
more than ~100 surrogates buys resolution that does not exist and the null will contain repeats.
Widen shift_ms (respecting the IBI) if a finer null is needed. The L2 p is
p_surr = (1 + #{surr >= real}) / (1 + N_valid), so N_SURROGATE also floors p at 1/(1+N).

RANDOM NUMBERS
Each call uses a local `np.random.default_rng(seed)` and touches no global random state. `seed`
defaults to DEFAULT_SEED, so an unseeded call is still reproducible.
"""
from __future__ import annotations

import warnings
from typing import NamedTuple, Optional, Sequence, Union

import numpy as np

__all__ = [
    'SurrogateResult', 'make_surrogate_rpeaks', 'make_surrogate_rpeaks_retry',
    'min_abs_shift', 'surrogate_seed', 'guard_from_windows',
    'DEFAULT_SEED', 'MAX_ATTEMPTS', 'SURR_SHIFT_MS', 'SURR_RETRIES',
]

# -- Constants ----------------------------------------------------------------
DEFAULT_SEED = 0        # any fixed integer would do; 0 means a caller who forgets to
                        # seed still gets a reproducible surrogate.
MAX_ATTEMPTS = 100      # draws per call before giving up
SURR_SHIFT_MS = (-500.0, 500.0)   # shift range in ms (Park et al. 2018; Lee et al. 2024)
SURR_RETRIES = 5        # fresh seeds tried by make_surrogate_rpeaks_retry

# Reasons returned on failure, named so that a caller that gets ok=False can log why.
REASON_OK = ''
REASON_EMPTY = 'empty_rpeak_train'
REASON_NO_SHIFT = 'no_admissible_shift'
REASON_IDENTITY = 'zero_shift_identity_surrogate'   # ok=True, but see the warning


class SurrogateResult(NamedTuple):
    """What one surrogate draw produced.

    A 4-field NamedTuple, so `peaks, ok = make_surrogate_rpeaks(...)` raises ValueError rather
    than binding the wrong things. `reason` carries the named failure cause; `shift` is the
    accepted shift in samples (the only way to check afterwards that a null was built from
    distinct shifts).
    """
    peaks: Optional[np.ndarray]   # [nBeats] int64 shifted peaks, or None if not ok
    ok: bool                      # True iff a shift satisfying every bound was found
    reason: str                   # '' when ok, a named cause otherwise
    shift: Optional[int]          # accepted shift in samples (signed), or None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _mround(x):
    """MATLAB's round(): half away from zero.

    np.round() rounds half to even and disagrees on exactly the .5 cases this module hits. At an
    IBI of 202 samples (about 124 bpm at 250 Hz), 0.25*202 = 50.5 gives a minimum shift of 51
    here and 50 under np.round, and the cap (62) does not bind: np.round would admit a shift of
    exactly 50, a quarter-IBI from the real train, which is what min_abs exists to forbid. At
    60 bpm (IBI 250 samples, +/-500 ms) the two disagree on 62.5, but min_abs is then capped at
    floor(0.5*125) = 62 and the difference is hidden.
    """
    x = np.asarray(x, dtype=float)
    out = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return out if out.ndim else float(out)


def _as_peak_array(rp) -> np.ndarray:
    """Flatten and integerise the peak train."""
    a = np.asarray(rp).ravel()
    if a.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(a.dtype, np.integer):
        af = np.asarray(a, dtype=float)
        if not np.all(np.isfinite(af)):
            raise ValueError('rp contains non-finite sample indices')
        # Refuse fractional peaks rather than rounding them: a caller passing
        # seconds instead of samples must find out here, not from a silently
        # meaningless surrogate.
        if not np.array_equal(af, np.floor(af)):
            raise ValueError('rp must be integer SAMPLE indices, not times')
        a = af.astype(np.int64)
    return a.astype(np.int64, copy=False)


def surrogate_seed(k: int, j: int, tryit: int) -> int:
    """Seed of retry `tryit` of surrogate `k` for session `j`.

        seed = 100000*k + 1000*j + tryit

    with k = surrogate index (1..N_SURROGATE), j = session index within the subject (1..nSes),
    tryit = retry index (1..5), all 1-based. Seeds do not collide for j <= 99 and tryit <= 999.

    The seed has no subject index: for surrogate draw k, every subject's session j is seeded
    identically, so every subject's first candidate shift `d` is the same number. Subjects only
    diverge when their own median IBI or epoch guard rejects that first draw. The L2 null builds
    a group map per draw k, so shifts that are correlated across subjects within a draw make
    that null something other than an i.i.d. null.
    """
    return int(100000 * int(k) + 1000 * int(j) + int(tryit))


def guard_from_windows(hrsp_win_ms: Sequence[float], hep_win_ms: Sequence[float],
                       fs: float, fs_hep: Optional[float] = None) -> tuple:
    """Build `guard`, the epoch headroom in samples, from the HRSP and HEP epoch windows.

        ep_pre   = round(|hrsp_win[0]| / 1000 * fs)
        ep_post  = round(hrsp_win[1] / 1000 * fs)
        hep_pre  = round(|hep_win[0]| / 1000 * fs_hep)
        hep_post = round(hep_win[1] / 1000 * fs_hep)
        guard    = (max(ep_pre, hep_pre), max(ep_post, hep_post))

    i.e. the widest epoch either measure needs, so a shift that is legal here is legal for every
    measure at once.

    `fs` sizes the HRSP terms and `fs_hep` the HEP terms. In this study every recording is
    downsampled to 250 Hz, so they are equal; on a mixed-rate cohort pass the reference rate as
    `fs` and the session's own rate as `fs_hep`. `fs_hep=None` (the default) means the same rate
    everywhere.

    This is the epoch guard only. It is not the segment-CWT guard, which adds the shift range on
    top so that shifted epochs stay inside the transformed cores. Do not reuse one for the
    other.
    """
    if fs_hep is None:
        fs_hep = fs
    # Validate: a guard built from a bad rate does not guard, and nothing downstream would
    # notice.
    for nm, v in (('fs', fs), ('fs_hep', fs_hep)):
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f'{nm} must be positive and finite, got {v!r}')
    for nm, w in (('hrsp_win_ms', hrsp_win_ms), ('hep_win_ms', hep_win_ms)):
        if len(w) != 2:
            raise ValueError(f'{nm} must be (pre_ms, post_ms), got {w!r}')
        if not np.all(np.isfinite(np.asarray(w, dtype=float))):
            raise ValueError(f'{nm} must be finite, got {w!r}')

    ep_pre = int(_mround(abs(hrsp_win_ms[0]) / 1000.0 * fs))
    ep_post = int(_mround(hrsp_win_ms[1] / 1000.0 * fs))
    hep_pre = int(_mround(abs(hep_win_ms[0]) / 1000.0 * fs_hep))
    hep_post = int(_mround(hep_win_ms[1] / 1000.0 * fs_hep))
    return (max(ep_pre, hep_pre), max(ep_post, hep_post))


def min_abs_shift(rp, shift_ms: Sequence[float], fs: float) -> int:
    """The lower edge of the rejection band, in samples.

    Near-zero shifts are rejected because they would reproduce the real epochs. The threshold is
    a quarter of the median IBI, so the surrogate always lands at a materially different cardiac
    phase.

    `min_abs` is capped at half the largest available shift so it can never swallow the whole
    range. Without the cap the function would return ok=False for every draw whenever the IBI is
    large relative to shift_ms, which is the slow-heart-rate case common in this study
    (meditators at rest). The cap engages once median IBI > 2*max|shift|; the uncapped band
    would be empty once median IBI > 4*max|shift|.

    This helper is public, so it validates its own inputs: a reversed range or an empty train
    raises instead of returning a number.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'fs must be positive and finite, got {fs!r}')
    if len(shift_ms) != 2:
        raise ValueError(f'shift_ms must be (lo, hi), got {shift_ms!r}')
    if not np.all(np.isfinite(np.asarray(shift_ms, dtype=float))):
        raise ValueError(f'shift_ms must be finite, got {shift_ms!r}')
    if shift_ms[1] < shift_ms[0]:
        raise ValueError(f'shift_ms must be (lo, hi) with hi >= lo, got {shift_ms!r}')

    rp = _as_peak_array(rp)
    if rp.size == 0:
        # Undefined for an empty train (see the docstring).
        raise ValueError('min_abs_shift is undefined for an empty R-peak train')
    lo = int(_mround(shift_ms[0] / 1000.0 * fs))
    hi = int(_mround(shift_ms[1] / 1000.0 * fs))

    if rp.size > 1:
        # np.median over integer diffs can be a .5 value -> _mround, not np.round.
        m = int(_mround(0.25 * float(np.median(np.diff(rp)))))
    else:
        # Single beat (or none): no IBI to measure, fall back to 100 ms.
        m = int(_mround(0.10 * fs))

    max_avail = max(abs(lo), abs(hi))
    return int(min(m, int(np.floor(0.5 * max_avail))))


# -----------------------------------------------------------------------------
# The surrogate
# -----------------------------------------------------------------------------
def make_surrogate_rpeaks(rp,
                          n_pts: int,
                          shift_ms: Sequence[float] = SURR_SHIFT_MS,
                          fs: float = 250.0,
                          guard: Union[int, float, Sequence[float], None] = None,
                          seed: int = DEFAULT_SEED,
                          max_attempts: int = MAX_ATTEMPTS,
                          index_base: int = 0) -> SurrogateResult:
    """Rigidly shift an R-peak train.

    Parameters
    ----------
    rp : [nBeats] integer R-peak sample indices.
    n_pts : total samples in the recording.
    shift_ms : (lo, hi) shift range in ms, e.g. (-500, 500).
    fs : sampling rate (Hz).
    guard : required, deliberately without a working default. Samples of headroom needed at
        each edge so that every shifted beat still has a full epoch in bounds. A scalar
        broadcasts to (pre, post). Build it with guard_from_windows(); hep_win=[-300 700] at
        fs=250 gives (75, 175).

        Why there is no default: `shift_ms` and `fs` default to the values the analyses use, so
        a default guard would read as "the analyses' guard". A guard of 0 makes the guarantee
        above false while still returning ok=True: rp=[100,350,600], n_pts=800, seed=4 with
        guard=0 returns shift=+111 and peaks=[211,461,711], an epoch end of 711+175 = 886 on an
        800-sample recording, whereas the (75,175) guard returns ok=False/'no_admissible_shift'.
        The harm would be silent: a negative epoch start (`peak - hep_pre`) does not raise in
        numpy, it wraps to the far end of the recording. Pass guard=0 explicitly to disable it.
    seed : explicit integer seed. The default exists only so an unseeded call is still
        reproducible.
    max_attempts : draws before giving up (default 100).
    index_base : 0 for 0-based peaks (the convention of the Python analyses), 1 for 1-based
        peaks (bounds `cand - pre >= 1`, `cand + post <= nPts`).
    Returns
    -------
    SurrogateResult(peaks, ok, reason, shift).

    Notes
    -----
    The shift is drawn uniformly over the integer sample range [lo, hi] and
    rejected if (a) its magnitude is under min_abs_shift(), or (b) it would push
    any beat's epoch out of bounds. Up to `max_attempts` draws are tried before
    giving up; failure returns ok=False and peaks=None, never a fallback train.
    """
    # -- Validation: a malformed input raises instead of producing a wrong surrogate.
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'fs must be positive and finite, got {fs!r}')
    if int(n_pts) <= 0:
        raise ValueError(f'n_pts must be positive, got {n_pts!r}')
    if index_base not in (0, 1):
        raise ValueError(f'index_base must be 0 or 1, got {index_base!r}')
    if len(shift_ms) != 2:
        raise ValueError(f'shift_ms must be (lo, hi), got {shift_ms!r}')
    if shift_ms[1] < shift_ms[0]:
        # A negative width would give shifts outside the requested range.
        raise ValueError(f'shift_ms must be (lo, hi) with hi >= lo, got {shift_ms!r}')
    if max_attempts < 1:
        raise ValueError(f'max_attempts must be >= 1, got {max_attempts!r}')
    if guard is None:
        # Not defaultable (see the `guard` entry in the docstring). Checked before the
        # empty-train short-circuit below, so a missing guard is not masked by an empty train.
        raise ValueError(
            'guard is REQUIRED - it is the headroom '
            'that keeps every shifted epoch in bounds, and a guard of 0 '
            'returns ok=True for out-of-bounds surrogates. '
            'Build it with guard_from_windows(hrsp_win, hep_win, fs); the '
            'pipeline uses (75, 175) at fs=250 / hep_win=[-300 700]. Pass '
            'guard=0 explicitly if you really want no guard.')

    n_pts = int(n_pts)
    rp = _as_peak_array(rp)

    # Empty train: ok=False with a named reason.
    if rp.size == 0:
        return SurrogateResult(None, False, REASON_EMPTY, None)

    # A scalar guard broadcasts to (pre, post). The guard is kept as a float, not int(): int()
    # truncates toward zero and would loosen the bound by up to one sample in the unsafe
    # direction (guard=75.9 must reject a beat landing at sample 76). A length-1 sequence
    # raises: `(75,)` more likely means `(75, 175)` with a dropped term than "same both ends",
    # and the scalar form `75` is unambiguous.
    if np.isscalar(guard) or (isinstance(guard, np.ndarray) and guard.ndim == 0):
        pre_guard = post_guard = float(guard)
    else:
        g = np.asarray(guard, dtype=float).ravel()
        if g.size != 2:
            raise ValueError(f'guard must be a scalar or (pre, post), got {guard!r}')
        pre_guard, post_guard = float(g[0]), float(g[1])
    if not (np.isfinite(pre_guard) and np.isfinite(post_guard)):
        raise ValueError(f'guard must be finite, got ({pre_guard}, {post_guard})')
    if pre_guard < 0 or post_guard < 0:
        raise ValueError(f'guard must be non-negative, got ({pre_guard:g}, {post_guard:g})')

    lo = int(_mround(shift_ms[0] / 1000.0 * fs))
    hi = int(_mround(shift_ms[1] / 1000.0 * fs))
    m_abs = min_abs_shift(rp, shift_ms, fs)

    # Bounds, expressed once so index_base cannot drift between the two sides.
    # index_base=1 gives lo_bound=1, hi_bound=nPts.
    lo_bound = index_base
    hi_bound = n_pts - 1 + index_base

    # Sorted or not, all(cand >= x) <=> min(cand) >= x, so only the extreme beats are tested.
    p_min = int(rp.min())
    p_max = int(rp.max())

    rng = np.random.default_rng(seed)   # explicit seed, always
    width = hi - lo + 1                 # inclusive range [lo, hi]

    for _attempt in range(max_attempts):
        # d = lo + floor(rand * (hi - lo + 1)): uniform integer on [lo, hi], written out
        # rather than rng.integers() so the inclusive endpoints are visible.
        d = lo + int(np.floor(rng.random() * width))
        if abs(d) < m_abs:
            continue
        if (p_min + d - pre_guard >= lo_bound) and (p_max + d + post_guard <= hi_bound):
            reason = REASON_OK
            if d == 0:
                # d == 0 is reachable only when m_abs == 0, which needs
                # floor(0.5*max(|lo|,|hi|)) == 0, i.e. a shift range under two samples
                # (e.g. shift_ms=(0,0)). The "surrogate" is then the real train with ok=True:
                # the beat count and IBIs match, and every effect would appear to survive its
                # own control. The result is still returned (a degenerate configuration is the
                # caller's to fix), but it is named in `reason` and warned about. Under
                # `warnings.simplefilter('error')` this path raises; check
                # `reason == REASON_IDENTITY` if you need the value.
                reason = REASON_IDENTITY
                warnings.warn(
                    'make_surrogate_rpeaks accepted a ZERO shift: the "surrogate" '
                    'is the real R-peak train. This only happens when shift_ms is '
                    f'degenerate (got {tuple(shift_ms)} -> [{lo}, {hi}] samples, '
                    f'min_abs={m_abs}). Any null built from this CONFIRMS every '
                    'effect by construction.', RuntimeWarning, stacklevel=2)
            return SurrogateResult(rp + d, True, reason, d)

    # Exhausted. No fallback: the caller must leave this draw NaN, never substitute an
    # unshifted or partially shifted train.
    return SurrogateResult(
        None, False,
        f'{REASON_NO_SHIFT}: {max_attempts} draws on [{lo}, {hi}] samples all '
        f'rejected (|d| >= {m_abs} required; beats span [{p_min}, {p_max}] and '
        f'guard is ({pre_guard:g}, {post_guard:g}) in [{lo_bound}, {hi_bound}])',
        None)


def make_surrogate_rpeaks_retry(rp,
                                n_pts: int,
                                shift_ms: Sequence[float],
                                fs: float,
                                guard: Union[int, Sequence[int]],
                                k: int,
                                j: int,
                                n_tries: int = SURR_RETRIES,
                                max_attempts: int = MAX_ATTEMPTS,
                                index_base: int = 0) -> SurrogateResult:
    """Retry make_surrogate_rpeaks with fresh seeds until one draw succeeds.

        for tryit = 1..n_tries:
            seed = surrogate_seed(k, j, tryit); stop at the first ok draw

    Each retry is a fresh 100-draw stream under a new seed, so 5 tries is up to 500 draws, but
    the admissible set is unchanged by reseeding. If a session's bounds admit no shift, all
    tries fail and the caller must drop the whole draw (leaving that surrogate NaN for every
    session, not just the failing one).

    k and j are 1-based loop counters; see surrogate_seed() about the missing subject term.
    """
    res = SurrogateResult(None, False, 'no_tries_run', None)
    for tryit in range(1, int(n_tries) + 1):
        res = make_surrogate_rpeaks(rp, n_pts, shift_ms, fs, guard,
                                    seed=surrogate_seed(k, j, tryit),
                                    max_attempts=max_attempts,
                                    index_base=index_base)
        if res.ok:
            return res
    return res
