# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Functions of the frontal-midline theta analysis (Analysis 3), used by fmt_depth_npz.py.

  make_frontal_template   Gaussian template centred on the frontocentral electrodes
  build_ged               theta-against-broadband generalized eigendecomposition per session,
                          with shrinkage, and the template screen of the leading components
  zscore_omitnan          z score ignoring undefined values
  fit_glm_wls             participant-level regression (Huber-weighted least squares)
  signflip_ttest          weighted one-sample sign-flip test of the participant slopes
  shrunk_ivw              inverse-variance weights, shrunk towards the median
  plot_depth_profile      theta power at each depth level

GED: Zuure et al. (2020). Scalp measure: Brandmeyer and Delorme (2018).
"""
import warnings
import numpy as np
from scipy.linalg import eigh as scipy_eigh
import mne

warnings.filterwarnings('ignore', category=FutureWarning)

EPS = np.finfo(float).eps

# =====================================================================
# SECTION 1: GED & POWER COMPUTATION
# =====================================================================


def _virtual_centre_xy(label, ch_names, xs, ys, k=4):
    """
    2-D position for a template centre that is not one of the data channels.

    In this dataset FCz and AFz are not among the 64 channels (they were the recording
    reference and were never restored), yet 'fcz' is the natural single centre for
    Zuure et al. (2020)'s template.

    The label is located in MNE's standard_1005 montage, its k nearest neighbours among
    the channels that do exist are found there, and the centre is placed at their
    inverse-distance-weighted centroid in the data's own 2-D frame. Working through
    neighbours rather than transplanting the coordinate means the two montages never have
    to share a scale or an origin. FCz is very nearly the centroid of Fz/Cz/FC1/FC2, so
    this is accurate to a few mm.

    Returns (x, y) or None if the label is unknown to the standard montage or too
    few of its neighbours are present.
    """
    try:
        import mne
        std = mne.channels.make_standard_montage('standard_1005')
    except Exception:
        return None
    pos = std.get_positions().get('ch_pos', {})
    std_lower = {name.lower(): np.asarray(p, float) for name, p in pos.items()}
    if label.lower() not in std_lower:
        return None
    target = std_lower[label.lower()]

    # Neighbours: data channels that the standard montage also knows.
    cand = [(j, std_lower[nm.lower()]) for j, nm in enumerate(ch_names)
            if nm.lower() in std_lower]
    if len(cand) < k:
        return None
    d = np.array([np.linalg.norm(p - target) for _, p in cand])
    order = np.argsort(d)[:k]
    w = 1.0 / (d[order] + 1e-9)
    w = w / w.sum()
    idx = [cand[o][0] for o in order]
    return float(np.sum(w * xs[idx])), float(np.sum(w * ys[idx]))


def _eeglab_topo_xy(ch_positions):
    """
    EEGLAB `topoplot` 2-D coordinates for a set of 3-D sensor positions.

    This is the azimuthal-equidistant projection EEGLAB uses (convertlocs 'cart2topo',
    then sph2topo):

        angle  = -sph_theta                 (azimuth, degrees)
        radius = 0.5 - sph_phi / 180        (sph_phi = elevation, degrees)

    not an orthographic one (dropping z), which compresses the lower half of the head
    because a sensor's distance from the vertex there varies as cos(elevation) instead of
    linearly with elevation. sigma in make_frontal_template is a distance in this plane,
    so the projection decides how many channels fall under the same Gaussian, and with
    it which GED component passes the template screen.

    Convention. `ch_positions` is in MNE's head frame (x to the right, y towards the
    nasion, z up), which is what `raw.get_montage().get_positions()['ch_pos']` returns.
    EEGLAB's own frame is X = nasion, Y = left, Z = up, so X = y, Y = -x, Z = z, and
    therefore

        elevation = atan2(z, hypot(x, y))
        topo angle = -atan2(Y, X) = atan2(x, y)

    which makes xs point right and ys point anterior.

    Returns (xs, ys), each [nChan], scaled by the largest radius so the outermost sensor
    sits at 1.0 (EEGLAB's rmax normalisation), which makes `sigma` readable as a fraction
    of head radius.
    """
    pos = np.asarray(ch_positions, dtype=float)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    rho = np.hypot(x, y)
    radius = 0.5 - np.degrees(np.arctan2(z, rho)) / 180.0
    # At the vertex rho == 0 and the azimuth is undefined; radius is 0 there too,
    # so any direction gives the same point. Pick (0, 0) rather than divide by 0.
    safe = rho > 0
    ux = np.where(safe, x / np.where(safe, rho, 1.0), 0.0)
    uy = np.where(safe, y / np.where(safe, rho, 1.0), 0.0)
    xs, ys = radius * ux, radius * uy
    # np.nanmax, not np.max: a channel with no digitised position would otherwise NaN
    # the whole montage.
    rmax = np.nanmax(np.hypot(xs, ys)) if xs.size else 0.0
    if np.isfinite(rmax) and rmax > 0:
        xs, ys = xs / rmax, ys / rmax
    return xs, ys


def make_frontal_template(ch_names, ch_positions, center_labels, sigma=0.25):
    """
    Gaussian scalp template for the GED R² screen.

    Sum of Gaussians centred at each requested electrode in the 2-D scalp projection
    (_eeglab_topo_xy). One centre reproduces Zuure et al. (2020)'s method (a single
    Gaussian on FCz); several centres give a broader frontocentral template. A centre that
    is not a data channel (FCz in this dataset) is placed from its standard_1005
    neighbours (_virtual_centre_xy).

    The two are not on the same R² scale. A sum of Gaussians over many frontocentral
    sites is a broader, flatter target than one bump on FCz, so the same topography
    scores a higher R² against it. Zuure's published cutoff (R² > 0.5) is calibrated to
    the single-FCz template; do not carry a threshold across the two without checking
    what it retains.

    No fallback: if none of the requested centres can be placed, this raises. A flat
    template would make every R² NaN, every component would fail the screen, and every
    session would drop out as "no component qualified".
    """
    if center_labels is None:
        center_labels = ['fcz']
    if isinstance(center_labels, str):
        center_labels = [center_labels]

    n_chan = len(ch_names)
    labels_lower = [c.lower() for c in ch_names]

    # 2-D scalp projection in EEGLAB topoplot coordinates (azimuthal-equidistant,
    # outermost sensor at radius 1); see _eeglab_topo_xy.
    xs, ys = _eeglab_topo_xy(ch_positions)

    template = np.zeros(n_chan)
    n_found, virtual, unusable = 0, [], []
    for cl in center_labels:
        idx = labels_lower.index(cl.lower()) if cl.lower() in labels_lower else None
        if idx is not None:
            cx, cy = xs[idx], ys[idx]
        else:
            vc = _virtual_centre_xy(cl, ch_names, xs, ys)
            if vc is None:
                unusable.append(cl)
                continue
            cx, cy = vc
            virtual.append(cl)
        template += np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma**2))
        n_found += 1

    if virtual:
        print(f'    [template] placed {len(virtual)} centre(s) not present as data '
              f'channels, via standard_1005 neighbours: {", ".join(virtual)}')
    if unusable:
        print(f'    [template] WARNING ignored unknown centre(s): {", ".join(unusable)}')
    if n_found == 0:
        raise ValueError(
            'make_frontal_template: none of the requested template centres (%s) '
            'could be placed - not data channels and not in standard_1005. A flat '
            'template would make every R2 NaN and silently empty the whole run, so '
            'this raises instead.' % ', '.join(center_labels))

    return template / (np.linalg.norm(template) + EPS)


def build_ged(data_broadband, data_theta, segment_mask, ch_names, ch_positions,
              shrinkage=0.01, template_elec=None, r2_thresh=0.50, top_n=7,
              max_comp=1):
    """
    Generalized eigendecomposition of theta against broadband covariance for one session.
    S = cov(theta segments), R = cov(broadband segments) + shrinkage.
    Solve eig(S, R), select component by template topography R² match.

    Returns dict with keys:
      ok : bool, whether a component passed the R² screen
      w  : [nChan] spatial filter (sign-aligned) of the first qualifying component
      pow_ts : [nPts] power timecourse, or None
      topo : [nChan] forward topography (for source localization), or None
      r2 / rank : template R² and eigenvalue rank of the first qualifying component
      n_comp : how many components were combined into pow_ts
      ranks / r2s : the eigenvalue rank and template R² of each of them
      all_r2 : template R² of every screened candidate; best_possible_r2 : its maximum

    `max_comp` > 1 averages power, not filters. Averaging the spatial filters would be
    meaningless: GED components are not commensurate in scale and can be near-orthogonal,
    so their mean is not a filter for anything. Averaging the power time courses is well
    defined, and unlike the Lee IC average it cannot suffer polarity or timing
    cancellation, because power is non-negative. `w`, `topo`, `r2` and `rank` describe
    the first qualifying component only: a mean forward pattern over different components
    has no source to localize, so source localization uses one component.
    """
    n_chan, n_pts = data_broadband.shape

    if template_elec is None:
        template_elec = ch_names

    # Extract segment data
    Xb = data_broadband[:, segment_mask]   # broadband, segments only
    Xt = data_theta[:, segment_mask]       # theta, segments only
    n_t = Xb.shape[1]

    if n_t <= n_chan * 2:
        return {'ok': False, 'w': None, 'pow_ts': None, 'topo': None,
                'r2': None, 'rank': None, 'all_r2': [],
                'best_possible_r2': None}

    # Covariances
    S = np.cov(Xt)
    R = np.cov(Xb)

    # Ridge shrinkage: R = (1-α)*R + α*mean(eig(R))*I
    # mean(eig(R)) = trace(R) / n_chan
    R = (1 - shrinkage) * R + shrinkage * np.trace(R) / n_chan * np.eye(n_chan)

    # Generalized eigendecomposition: S @ w = λ @ R @ w
    # scipy.linalg.eigh solves this (both S and R are symmetric)
    try:
        eigenvalues, eigenvectors = scipy_eigh(S, R)
    except np.linalg.LinAlgError:
        return {'ok': False, 'w': None, 'pow_ts': None, 'topo': None,
                'r2': None, 'rank': None, 'all_r2': [],
                'best_possible_r2': None}

    # Sort descending (eigh returns ascending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Activation patterns: Ag = S @ Wg (equal to (Wg' S)' for symmetric S)
    Ag = S @ eigenvectors

    # Template
    tmpl = make_frontal_template(ch_names, ch_positions, template_elec)

    # Component selection by template R² match
    n_cand = min(top_n, eigenvectors.shape[1])
    best = None
    # Keep the scores, not only the gate. `r2` is the squared correlation between the
    # candidate's forward topography and the template; `all_r2` holds every candidate's
    # score, so a near-miss is distinguishable from a rout.
    all_r2 = []
    qualifying = []
    for k in range(n_cand):
        tn = Ag[:, k] / (np.linalg.norm(Ag[:, k]) + EPS)
        r = np.corrcoef(tn, tmpl)[0, 1]
        all_r2.append(float(r**2))
        if r**2 >= r2_thresh:
            qualifying.append(k)
            if best is None:
                best = k

    if best is None:
        return {'ok': False, 'w': None, 'pow_ts': None, 'topo': None,
                'r2': None, 'rank': None, 'n_comp': 0, 'ranks': [], 'r2s': [],
                'all_r2': all_r2, 'best_possible_r2': (max(all_r2) if all_r2
                                                       else None)}

    # Order is eigenvalue order, not R² order: `qualifying` is built in the loop above,
    # so its first max_comp entries are the qualifying components that explain the most
    # theta variance.
    use = qualifying[:max(1, int(max_comp))]

    # Sign-fix: align each component to the template. The sign is irrelevant to
    # the power average and is applied so that `w` and `topo` stay comparable
    # across sessions for the source-localization step.
    pows = []
    for k in use:
        s_k = np.sign(np.corrcoef(Ag[:, k], tmpl)[0, 1]) or 1
        pows.append((s_k * eigenvectors[:, k] @ data_theta) ** 2)
    pow_ts = np.mean(pows, axis=0)

    wsign = np.sign(np.corrcoef(Ag[:, best], tmpl)[0, 1])
    if wsign == 0:
        wsign = 1
    w = wsign * eigenvectors[:, best]
    topo = wsign * Ag[:, best]

    return {'ok': True, 'w': w, 'pow_ts': pow_ts, 'topo': topo,
            'r2': float(all_r2[best]), 'rank': int(best),
            'n_comp': len(use), 'ranks': [int(k) for k in use],
            'r2s': [float(all_r2[k]) for k in use],
            'all_r2': all_r2, 'best_possible_r2': (max(all_r2) if all_r2
                                                   else None)}


# =====================================================================
# SECTION 2: L1 GLM
# =====================================================================

def zscore_omitnan(x):
    """
    NaN-safe z-score with ddof=1 (sample standard deviation, N-1). A single finite value
    gives 0. The z-scores are the input to the L1 depth GLM.
    """
    m = np.isfinite(x)
    z = np.full_like(x, np.nan)
    if np.sum(m) >= 2:
        z[m] = (x[m] - np.mean(x[m])) / (np.std(x[m], ddof=1) + EPS)
    elif np.any(m):
        z[m] = 0
    return z


def fit_glm_wls(X, y, method='WLS'):
    """
    Participant-level GLM: WLS with Huber weights, or OLS. No cluster-robust variance.

    Returns:
      B     : beta coefficients [nPred] (for 1D y) or [nPred x nTime] (for 2D y)
      rv    : residual variance (scalar for 1D y)
      w_hub : Huber weights (ones for OLS)
    """
    n_pred = X.shape[1]
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_obs, n_time = y.shape

    # Regularized (X'X)^-1
    XtX = X.T @ X
    iXtX = np.linalg.solve(XtX + 1e-6 * np.trace(XtX) * np.eye(n_pred), np.eye(n_pred))
    df = n_obs - n_pred

    if method.upper() == 'WLS':
        # OLS initial
        B_ols = iXtX @ (X.T @ y)
        resid = y - X @ B_ols
        resid_norm = np.linalg.norm(resid, axis=1)
        k_hub = 1.345 * np.median(resid_norm)
        if k_hub < EPS:
            k_hub = 1.0
        w_hub = np.minimum(1.0, k_hub / np.maximum(resid_norm, EPS))
        W = np.diag(w_hub)
        XtWX = X.T @ W @ X
        iXtWX = np.linalg.solve(
            XtWX + 1e-6 * np.trace(XtWX) * np.eye(n_pred), np.eye(n_pred))
        B = iXtWX @ (X.T @ W @ y)
        Yhat = X @ B
        MSE = np.sum(((y - Yhat)**2) * w_hub[:, None], axis=0) / df
    else:
        # OLS
        B = iXtX @ (X.T @ y)
        Yhat = X @ B
        iXtWX = iXtX
        w_hub = np.ones(n_obs)
        MSE = np.sum((y - Yhat)**2, axis=0) / df

    # Residual variance for depth coefficient (column 1, 0-based)
    col_d = 1  # depth column (0-based: intercept=0, depth=1)
    if n_time == 1:
        rv = float(MSE[0] * iXtWX[col_d, col_d])
        B = B.ravel()
    else:
        rv = MSE * iXtWX[col_d, col_d]

    return B, rv, w_hub


# =====================================================================
# SECTION 3: L2 GROUP TESTS
# =====================================================================

def signflip_ttest(betas, weights, n_perm=1000):
    """
    Weighted sign-flip permutation test of the participant slopes. The SE uses Ws in the
    denominator, a non-finite t returns (NaN, NaN), and the p is add-one.

    Returns: (t_stat, p_value)
    """
    betas = betas.ravel()
    n = len(betas)
    w = weights.ravel() / np.mean(weights)
    Ws = np.sum(w)

    wm = np.sum(betas * w) / Ws
    res = betas - wm
    n_eff = Ws**2 / np.sum(w**2)
    SS = np.sum(res**2 * w)

    # SE denominator is Ws (not n_eff)
    t = wm / (np.sqrt(SS / (max(n_eff - 1, 1) * Ws)) + EPS)

    if not np.isfinite(t):
        return np.nan, np.nan

    # Permutation null
    rng = np.random.default_rng(42)
    tn = np.zeros(n_perm)
    for ip in range(n_perm):
        s = 2 * (rng.random(n) > 0.5).astype(float) - 1
        bp = betas * s
        wmp = np.sum(bp * w) / Ws
        rp = bp - wmp
        SSp = np.sum(rp**2 * w)
        tn[ip] = wmp / (np.sqrt(SSp / (max(n_eff - 1, 1) * Ws)) + EPS)

    # add-one permutation p (Phipson & Smyth 2010)
    p = (1 + np.sum(np.abs(tn) >= np.abs(t))) / (1 + n_perm)
    return t, p


def shrunk_ivw(svar, shrink_mode='median'):
    """
    Shrunk inverse-variance weights.
    w = 1/(v + c), c = median(v). Normalized to sum 1.

    Returns: (w, info_dict)
    """
    svar = svar.ravel()
    ok = np.isfinite(svar) & (svar > 0)

    if not np.any(ok):
        w = np.zeros_like(svar)
        return w, {'c': np.nan, 'nUsed': 0, 'eff_n': 0, 'wmax_ratio': np.nan}

    if isinstance(shrink_mode, (int, float)):
        c = shrink_mode
    elif shrink_mode == 'none':
        c = 0
    elif shrink_mode == 'median':
        c = np.median(svar[ok])
    else:
        raise ValueError(f"Unknown shrink_mode: {shrink_mode}")

    w = np.zeros_like(svar)
    w[ok] = 1.0 / (svar[ok] + c)
    w = w / np.sum(w)

    info = {
        'c': c,
        'nUsed': int(np.sum(ok)),
        'eff_n': 1.0 / np.sum(w**2),
        'wmax_ratio': np.max(w) / np.median(w[ok]),
    }
    return w, info


# =====================================================================
# DEPTH PROFILE PLOT
# =====================================================================

def plot_depth_profile(depth_levels, prof_mean, prof_sem, l2_table, cond_tag,
                        conf_key, output_path):
    """Depth profile bar plot with error bars."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    pairs = [('scalp', 'FMT_scalp', 'FMT scalp'),
             ('ged', 'FMT_GED', 'FMT GED')]

    for mi, (key, meas_name, title) in enumerate(pairs):
        ax = axes[mi]
        mn = prof_mean[key]
        se = prof_sem[key]
        ax.bar(depth_levels, mn, 0.6, color=[0.25, 0.5, 0.75])
        ax.errorbar(depth_levels, mn, se, fmt='k', linestyle='none',
                     linewidth=1.2, capsize=6)
        ax.axhline(0, color='k', linewidth=0.75)
        ax.set_xlabel('Depth level')
        ax.set_ylabel('Mean z-power (within-session)')
        ax.set_xlim(depth_levels[0] - 0.7, depth_levels[-1] + 0.7)
        ax.set_xticks(depth_levels)
        ax.grid(True, alpha=0.15)
        ax.set_title(title, fontsize=10)

    fig.suptitle(f'FMT depth profile - {cond_tag}, {conf_key}', fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f'  Saved: {output_path}')
