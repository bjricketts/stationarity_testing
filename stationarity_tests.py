"""
stationarity_tests.py
---------------------
Tests of second-order (spectral) stationarity for evenly sampled, Poisson
counting light curves.

H0 throughout: the *source* power spectrum (Poisson noise removed) does not
depend on time. All tests work on a time-frequency grid of K time blocks x J
frequency bins.

Poisson noise
=============
Counting noise is assumed Poisson (no dead time). In a segment with mean rate
r its periodogram level is

    N = 2 / r        (norm='frac', (rms/mean)^2 / Hz)
    N = 2 r          (norm='abs',  rms^2 / Hz)

N is computed per segment from that segment's own mean, so it follows any
change in the count rate. A block cell is C_kj = mean of n periodogram
ordinates with E[C_kj] = S_kj + N_k, and the source estimate is
S^_kj = C_kj - N_k. Subtracting a constant from a Gamma variable changes the
mean and variance of its logarithm, so the tests use

    Y_kj = ln(C_kj - N_k) - m(r_kj),   Var[Y_kj] = v(r_kj),   r_kj = N_k / S_j

where m and v are the exact moments of ln((C - N)/S) for a Gamma(n) cell,
tabulated numerically (for N = 0 they are psi(n) - ln n and psi'(n)), and
S_j is the time-averaged noise-subtracted power under H0. N_k differs
between blocks when the rate changes, so the moments are per cell and the
statistics are weighted. Y becomes skewed and the chi^2 approximation
degrades as the noise fraction grows, so the tests need a well-measured
source: only frequency bins where the Poisson level is at most
`max_noise_frac` of the total power, N / (S_j + N) <= max_noise_frac, and
where the predicted fractional error of a noise-subtracted cell,
(S_j + N) / (S_j sqrt(n)), is at most 0.5 enter the tests. With
noise='none' the tests reduce to the unsubtracted versions (N = 0, where
psi'(n) and the bias are exact).

Tests
=====
1. psr_test        Priestley & Subba Rao (1969)-style weighted analysis of
                   log block spectra: common time effect, time x frequency
                   interaction, their total, a smooth-trend statistic
                   (polynomials in time for each frequency bin), and the
                   largest single-bin trend. Permutation p-values shuffle
                   segments (and their noise levels) in time.
2. surrogate_test  Borgnat, Flandrin et al. (2010, IEEE TSP 58, 3459):
                   multitaper spectrogram (noise subtracted), variance over
                   time of a distance between local and mean spectra,
                   compared with phase-randomised surrogates.
3. bayes_test      Posterior probability of stationarity. H0: constant log
                   source spectrum. H1: a random subset of frequency bins has
                   log source power evolving smoothly in time. Gaussian
                   marginal likelihoods with the per-cell variances above.
4. sbb_bound       Sellke, Bayarri & Berger (2001) lower bound on P(H0 | p).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import stats
from scipy.signal.windows import dpss
from scipy.special import logsumexp

from stingray import Lightcurve
from stingray.fourier import poisson_level
from stingray.gti import bin_intervals_from_gtis
from stingray.powerspectrum import AveragedPowerspectrum


# =========================================================================
# Poisson noise level (stingray.fourier.poisson_level)
# =========================================================================

def noise_level(mean_counts_per_bin, dt, norm="frac", bkg_rate=0.0):
    """Poisson periodogram level of each segment, from its own mean rate.

    Wraps `stingray.fourier.poisson_level`: 2/r for norm='frac' and 2r for
    'abs', and with `bkg_rate` (a background rate included in the data) the
    level rises to 2 r / (r - bkg)^2, matching a fractional rms defined
    against the source rate.
    """
    rate = np.asarray(mean_counts_per_bin, dtype=float) / dt
    return np.array([poisson_level(norm=norm, meanrate=r, backrate=bkg_rate)
                     for r in np.atleast_1d(rate)])


# =========================================================================
# Time-frequency grid of block-averaged periodograms
# =========================================================================

@dataclass
class TFGrid:
    """Block-averaged periodograms with their Poisson levels.

    cells[k, j]   mean of `n` periodogram ordinates in block k, freq bin j
    noise[k]      mean Poisson level of the segments in block k
    seg_pgram     per-segment, per-frequency-bin averages (n_seg x J)
    seg_noise     per-segment Poisson level (n_seg,)
    seg_time      start time of each segment [s]
    S_hat[j]      time-averaged noise-subtracted power
    keep[j]       bins used by the tests (source well measured)
    noise_frac[j] time-averaged N / (S_hat + N)
    avg_ps        the stingray AveragedPowerspectrum the cells came from
    """
    freq: np.ndarray
    t_block: np.ndarray
    cells: np.ndarray
    noise: np.ndarray
    seg_pgram: np.ndarray
    seg_noise: np.ndarray
    seg_time: np.ndarray
    S_hat: np.ndarray
    keep: np.ndarray
    seg_per_block: int
    fbin: int
    seg_len: float
    norm: str
    noise_model: str
    max_noise_frac: float
    avg_ps: object = None          # the stingray AveragedPowerspectrum

    @property
    def n(self) -> int:
        """Independent exponential ordinates per cell."""
        return self.seg_per_block * self.fbin

    @property
    def source(self) -> np.ndarray:
        """Noise-subtracted cells (can be negative in noise-dominated bins)."""
        return self.cells - self.noise[:, None]

    @property
    def rel_err(self) -> np.ndarray:
        """Predicted fractional error of one noise-subtracted cell, per bin."""
        S = np.clip(self.S_hat, 1e-300, None)
        return (S + self.noise.mean()) / (S * np.sqrt(self.n))

    @property
    def noise_frac(self) -> np.ndarray:
        N = self.noise.mean()
        if N <= 0:
            return np.zeros_like(self.S_hat)
        return N / (np.clip(self.S_hat, 0, None) + N)


def as_lightcurve(counts=None, dt=None, lc=None, gti=None) -> Lightcurve:
    """Return a stingray Lightcurve, building one from counts if needed."""
    if lc is not None:
        return lc
    if counts is None or dt is None:
        raise ValueError("pass a Lightcurve, or counts and dt")
    counts = np.asarray(counts, dtype=float)
    t = (np.arange(counts.size) + 0.5) * dt
    if gti is None:
        gti = np.array([[0.0, counts.size * dt]])
    return Lightcurve(t, counts, dt=dt, gti=gti, skip_checks=True)


def segment_spectra(lc: Lightcurve, seg_len, norm="frac"):
    """Per-segment periodograms of a Lightcurve, respecting its GTIs.

    Uses `AveragedPowerspectrum(..., use_common_mean=False, save_all=True)`,
    so each segment is normalised by its own mean (which removes a linear
    rms-flux relation), and `stingray.gti.bin_intervals_from_gtis` for the
    segment boundaries, so no segment spans a gap or a GTI edge.

    Returns (freq, P, mu, seg_time, avg_ps): the frequency grid, the
    (n_seg x n_freq) periodograms, the mean counts per bin of each segment,
    each segment's start time, and the stingray object itself.
    """
    avg = AveragedPowerspectrum(lc, segment_size=seg_len, norm=norm,
                                use_common_mean=False, save_all=True,
                                silent=True)
    P = np.array(avg.cs_all, dtype=float)
    start, stop = bin_intervals_from_gtis(lc.gti, seg_len, lc.time, dt=lc.dt)
    counts = np.asarray(lc.counts, dtype=float)
    mu = np.array([counts[a:b].mean() for a, b in zip(start, stop)])
    seg_time = np.asarray(lc.time)[start] - lc.dt / 2
    if len(mu) != len(P):                     # stingray dropped a short segment
        k = min(len(mu), len(P))
        P, mu, seg_time = P[:k], mu[:k], seg_time[:k]
    return np.asarray(avg.freq, dtype=float), P, mu, seg_time, avg


def build_tf_grid(counts=None, dt=None, seg_len=8.0, n_blocks=16, fmin=0.25,
                  fmax=8.0, fbin=2, norm="frac", noise="poisson",
                  max_noise_frac=0.3, lc=None, gti=None, bkg_rate=0.0,
                  noise_scale=1.0, noise_level_value=None,
                  f_noise=None) -> TFGrid:
    """Periodograms -> K x J block cells, noise levels, bin selection.

    Input: a stingray `Lightcurve` (`lc`), or `counts` and `dt`, optionally
    with `gti`. Segmentation and the per-segment periodograms come from
    stingray (`segment_spectra`), so gaps and GTI edges are respected.

    noise:
      'poisson'   2/rate (frac) or 2*rate (abs), from each segment's own mean
      'none'      no subtraction
      'highfreq'  scale the Poisson shape to the mean power above `f_noise`
                  (default: the top third of the band up to the Nyquist
                  frequency); use when dead time changes the level
    noise_level_value  explicit level (scalar or one value per segment),
                  overriding the above; noise_scale multiplies whatever is used
    bkg_rate      background count rate included in the data [ct/s]; with
                  norm='frac' the power and the noise level are renormalised
                  to the source rate, so background dilution that changes with
                  the count rate does not look like non-stationarity
    max_noise_frac: keep bins where the Poisson level is at most this
    fraction of the total power (and the predicted fractional error of a
    noise-subtracted cell is <= 0.5). None keeps every bin with S_hat > 0.
    """
    lc = as_lightcurve(counts, dt, lc, gti)
    dt = lc.dt
    freq, P, mu, seg_time, avg_ps = segment_spectra(lc, seg_len, norm)
    rate = mu / dt

    if noise_level_value is not None:
        seg_noise = np.broadcast_to(np.asarray(noise_level_value, float),
                                    (P.shape[0],)).astype(float).copy()
    elif noise == "none":
        seg_noise = np.zeros_like(mu)
    elif noise in ("poisson", "highfreq"):
        seg_noise = noise_level(mu, dt, norm, bkg_rate)
    else:
        raise ValueError(noise)

    if bkg_rate and norm == "frac":
        src = rate - bkg_rate
        if np.any(src <= 0):
            raise ValueError("bkg_rate exceeds the count rate in some segments")
        P = P * ((rate / src) ** 2)[:, None]

    if noise == "highfreq" and noise_level_value is None:
        hi = f_noise if f_noise is not None else freq[-1] * 2 / 3
        band = freq >= hi
        if band.sum() < 5:
            raise ValueError("too few Fourier bins above f_noise")
        seg_noise = seg_noise * (P[:, band].mean() / seg_noise.mean())
    seg_noise = seg_noise * noise_scale

    sel = (freq >= fmin) & (freq <= fmax)
    freq, P = freq[sel], P[:, sel]
    J = freq.size // fbin
    if J < 1:
        raise ValueError("no frequency bin in [fmin, fmax] with this fbin")
    freq = freq[: J * fbin].reshape(J, fbin).mean(axis=1)
    P = P[:, : J * fbin].reshape(P.shape[0], J, fbin).mean(axis=2)
    spb = P.shape[0] // n_blocks
    if spb < 1:
        raise ValueError("fewer segments than blocks; shorten seg_len or "
                         "reduce n_blocks")
    P = P[: spb * n_blocks]
    seg_noise = seg_noise[: spb * n_blocks]
    seg_time = seg_time[: spb * n_blocks] - seg_time[0]
    cells = P.reshape(n_blocks, spb, J).mean(axis=1)
    nlev = seg_noise.reshape(n_blocks, spb).mean(axis=1)
    S_hat = (cells - nlev[:, None]).mean(axis=0)
    t_block = seg_time.reshape(n_blocks, spb).mean(axis=1)
    grid = TFGrid(freq, t_block, cells, nlev, P, seg_noise, seg_time, S_hat,
                  np.ones(J, bool), spb, fbin, seg_len, norm,
                  noise if noise_level_value is None else "explicit",
                  1.0 if max_noise_frac is None else max_noise_frac, avg_ps)
    keep = S_hat > 0
    if not np.all(seg_noise == 0) and max_noise_frac is not None:
        keep &= (grid.noise_frac <= max_noise_frac) & (grid.rel_err <= 0.5)
    grid.keep = keep
    if grid.keep.sum() < 1:
        raise ValueError("no frequency bin has enough source power above the "
                         "Poisson level; lengthen blocks, widen fbin or "
                         "raise max_noise_frac")
    return grid


def _block(seg_pgram, seg_noise, n_blocks):
    spb = seg_pgram.shape[0] // n_blocks
    c = seg_pgram[: spb * n_blocks].reshape(n_blocks, spb, -1).mean(axis=1)
    nl = seg_noise[: spb * n_blocks].reshape(n_blocks, spb).mean(axis=1)
    return c, nl


@lru_cache(maxsize=32)
def _log_moment_table(n, r_max=10.0, n_r=2001, n_q=4000, floor=0.01):
    """Mean and variance of ln(max((1 + r) X - r, floor)), X ~ Gamma(n)/n.

    This is ln((C - N) / S) for a cell C that averages n exponential
    ordinates with mean S + N, with r = N / S. Tabulated on r in
    [0, r_max]; midpoint quantile quadrature.
    """
    r = np.linspace(0.0, r_max, n_r)
    x = stats.gamma.ppf((np.arange(n_q) + 0.5) / n_q, n, scale=1.0 / n)
    L = np.log(np.maximum((1.0 + r[:, None]) * x[None, :] - r[:, None], floor))
    return r, L.mean(axis=1), L.var(axis=1)


def log_source(cells, noise, S_hat, n, floor=0.01):
    """Bias-corrected log source power and its variance, per cell.

    Y_kj = ln(max(C_kj - N_k, floor S_j)) - m(r_kj), Var = v(r_kj), where
    r_kj = N_k / S_j and m, v are the exact moments of that transform for a
    Gaussian-process periodogram (see `_log_moment_table`). For N = 0 these
    are psi(n) - ln n and psi'(n). Returns (Y, V, n_floored).
    """
    Sn = cells - noise[:, None]
    fl = floor * S_hat[None, :]
    n_floor = int(np.sum(Sn <= fl))
    Sn = np.maximum(Sn, fl)
    rt, mt, vt = _log_moment_table(int(n), floor=floor)
    r = noise[:, None] / S_hat[None, :]
    if np.any(r > rt[-1]):
        raise ValueError("Poisson level exceeds 10x the source power in a "
                         "selected bin; lower max_noise_frac")
    m = np.interp(r, rt, mt)
    V = np.interp(r, rt, vt)
    return np.log(Sn) - m, V, n_floor


# =========================================================================
# 1. Priestley-Subba Rao-style weighted log-spectral analysis
# =========================================================================

def time_basis(K, n_poly=3):
    """Orthonormal time contrasts, each orthogonal to the constant.

    n_poly=d: discrete Legendre-like polynomials of degree 1..d.
    n_poly=None (or >= K-1): full (K-1)-dim complement (any time dependence).
    """
    if n_poly is None or n_poly >= K - 1:
        A = np.column_stack([np.ones(K), np.eye(K)[:, : K - 1]])
    else:
        x = np.linspace(-1, 1, K)
        A = np.vander(x, n_poly + 1, increasing=True)
    Qm, _ = np.linalg.qr(A)
    return Qm[:, 1:]


def _wls_chi2(Y, W, X):
    """Weighted least-squares chi^2 of Y[:, j] on design X, for every j.

    Y, W: (K, J); X: (K, p). Returns chi2 (J,).
    """
    A = np.einsum("kp,kj,kq->jpq", X, W, X)
    b = np.einsum("kp,kj,kj->jp", X, W, Y)
    beta = np.linalg.solve(A, b[..., None])[..., 0]            # (J, p)
    r = Y - (X @ beta.T)
    return np.sum(W * r**2, axis=0)


def _psr_stats(Y, V, E):
    K, J = Y.shape
    W = 1.0 / V
    one = np.ones((K, 1))
    chi_const = _wls_chi2(Y, W, one)                             # (J,) ~ chi2_{K-1}
    chi_poly = _wls_chi2(Y, W, np.hstack([one, E]))
    S_trend_j = chi_const - chi_poly                             # ~ chi2_d
    S_total = chi_const.sum()                                    # ~ chi2_{J(K-1)}
    # common time effect: Y_kj = mu_j + g_k (g_0 = 0), weighted fit
    Xa = np.zeros((K * J, J + K - 1))
    Xa[np.arange(K * J), np.tile(np.arange(J), K)] = 1.0
    rows = np.repeat(np.arange(1, K), J)
    Xa[(np.arange(J, K * J)), J + rows - 1] = 1.0
    sw = np.sqrt(W.ravel())
    coef, *_ = np.linalg.lstsq(Xa * sw[:, None], Y.ravel() * sw, rcond=None)
    chi_add = np.sum(W.ravel() * (Y.ravel() - Xa @ coef) ** 2)
    S_T = S_total - chi_add                                      # ~ chi2_{K-1}
    S_IR = chi_add                                               # ~ chi2_{(K-1)(J-1)}
    mu = np.sum(W * Y, axis=0) / np.sum(W, axis=0)
    z = (Y - mu) * np.sqrt(W)
    return S_T, S_IR, S_trend_j, z


def psr_test(grid: TFGrid, n_poly=3, n_perm=2000, rng=None) -> dict:
    """PSR-style analysis on bias-corrected log source power (selected bins).

    Common time effect (K-1 dof) and time x frequency interaction
    ((K-1)(J-1) dof) spread their power over every cell. The trend statistic
    compares a constant with a degree-`n_poly` polynomial in time in each
    frequency bin (J*n_poly dof), and `trend_max` is the most significant
    single bin (calibrated by permutation only).
    """
    rng = np.random.default_rng(rng)
    keep = grid.keep
    K = grid.cells.shape[0]
    J = int(keep.sum())
    E = time_basis(K, n_poly)
    d = E.shape[1]
    S_hat = grid.S_hat[keep]
    Y, V, n_floor = log_source(grid.cells[:, keep], grid.noise, S_hat, grid.n)
    S_T, S_IR, S_tj, zk = _psr_stats(Y, V, E)
    S_tot = S_T + S_IR
    dof_T, dof_IR = K - 1, (K - 1) * (J - 1)
    S_trend = S_tj.sum()

    # full-size per-bin outputs (NaN for excluded bins)
    J_all = grid.freq.size
    S_trend_j = np.full(J_all, np.nan)
    S_trend_j[keep] = S_tj
    z = np.full((K, J_all), np.nan)
    z[:, keep] = zk
    out = dict(
        S_time=S_T, p_time=stats.chi2.sf(S_T, dof_T),
        S_interaction=S_IR, p_interaction=stats.chi2.sf(S_IR, dof_IR),
        S_total=S_tot, dof_total=dof_T + dof_IR,
        p_total=stats.chi2.sf(S_tot, dof_T + dof_IR),
        chi2_red=S_tot / (dof_T + dof_IR),
        S_trend=S_trend, dof_trend=J * d,
        p_trend=stats.chi2.sf(S_trend, J * d),
        S_trend_j=S_trend_j, p_trend_j=stats.chi2.sf(S_trend_j, d),
        trend_max=S_tj.max(), f_trend_max=grid.freq[keep][np.argmax(S_tj)],
        n_poly=d, z=z, n_bins_used=J, n_floored=n_floor,
    )
    if n_perm:
        seg = grid.seg_pgram[:, keep]
        null = np.empty((n_perm, 3))
        for i in range(n_perm):
            idx = rng.permutation(seg.shape[0])
            c, nl = _block(seg[idx], grid.seg_noise[idx], K)
            Yp, Vp, _ = log_source(c, nl, S_hat, grid.n)
            a, b, tj, _ = _psr_stats(Yp, Vp, E)
            null[i] = a + b, tj.sum(), tj.max()
        obs = np.array([S_tot, S_trend, S_tj.max()])
        pp = (1 + np.sum(null >= obs, axis=0)) / (n_perm + 1)
        out.update(p_total_perm=pp[0], p_trend_perm=pp[1],
                   p_trend_max_perm=pp[2], null_trend=null[:, 1],
                   null_total=null[:, 0])
    return out


# =========================================================================
# 2. Surrogate time-frequency test (Borgnat et al. 2010)
# =========================================================================

def multitaper_spectrogram(x, dt, win_len, n_tapers=5, hop=None):
    """Slepian multitaper spectrogram of mean-subtracted windows.

    Returns (t, f, S[t, f], mu[t]) with S in the same units as
    `segment_periodograms(norm='abs')` before division by dt^2, and mu the
    mean counts per bin of each window. f excludes DC.
    """
    nw = int(round(win_len / dt))
    hop = nw // 2 if hop is None else int(round(hop / dt))
    NW = (n_tapers + 1) / 2.0
    tapers = dpss(nw, NW, Kmax=n_tapers)            # unit-energy tapers
    starts = np.arange(0, x.size - nw + 1, hop)
    frames = np.lib.stride_tricks.sliding_window_view(x, nw)[starts]
    mu = frames.mean(axis=1)
    frames = frames - mu[:, None]
    F = np.fft.rfft(frames[:, None, :] * tapers[None], axis=2)
    # unit-energy tapers: sum w^2 = 1, so |F|^2 has white level mu (counts);
    # scale to match the rectangular-window periodogram 2 dt |F|^2 / nw
    S = 2.0 * dt * np.mean(np.abs(F) ** 2, axis=1)
    f = np.fft.rfftfreq(nw, dt)
    t = (starts + nw / 2) * dt
    return t, f[1:], S[:, 1:], mu


def _tf_distance_variance(S, alpha=0.3):
    """Var over time of alpha*KL + (1-alpha)*LSD between local and mean spectra."""
    G = S.mean(axis=0)
    p = S / S.sum(axis=1, keepdims=True)
    q = G / G.sum()
    kl = np.sum((p - q) * np.log(p / q), axis=1)          # symmetrised KL
    lsd = np.mean(np.abs(np.log(S / G)), axis=1)
    d = alpha * kl / kl.mean() + (1 - alpha) * lsd / lsd.mean()
    return np.var(d)


def phase_randomise(x, rng):
    """Surrogate with the same periodogram and random Fourier phases."""
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, X.size)
    ph[0] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=x.size) + x.mean()


def surrogate_test(counts, dt, win_len=8.0, n_tapers=5, n_surr=99,
                   freq_ranges=((0.25, 8.0),), norm="frac", noise="poisson",
                   rng=None) -> dict:
    """Borgnat et al. (2010) test with phase-randomised surrogates.

    Each window's spectrum is normalised (frac: by its mean squared) and its
    Poisson level (from the window's own mean) is subtracted; values are
    floored at 5 % of the time-averaged source power. Only frequencies inside
    `freq_ranges` are used; pass the selected TFGrid bins so both tests see
    the same well-measured frequencies.

    Phase randomisation spreads the Poisson noise evenly in time, so the
    surrogates keep a constant noise level; the subtraction uses each
    window's own mean for data and surrogates alike.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(counts, dtype=float)

    def stat(y):
        t, f, S, mu = multitaper_spectrogram(y, dt, win_len, n_tapers)
        sel = np.zeros(f.size, bool)
        for lo, hi in freq_ranges:
            sel |= (f >= lo) & (f <= hi)
        S = S[:, sel]
        if norm == "frac":
            S = S / mu[:, None] ** 2
        else:
            S = S / dt**2
        if noise == "poisson":
            S = S - noise_level(mu, dt, norm)[:, None]
            S = np.maximum(S, 0.05 * np.clip(S.mean(axis=0), 1e-300, None))
        return _tf_distance_variance(S)

    theta0 = stat(x)
    theta = np.array([stat(phase_randomise(x, rng)) for _ in range(n_surr)])
    a, loc, scale = stats.gamma.fit(theta, floc=0)
    return dict(
        theta=theta0, theta_surr=theta,
        INS=theta0 / np.median(theta),
        p_gamma=stats.gamma.sf(theta0, a, loc=loc, scale=scale),
        p_rank=(1 + np.sum(theta >= theta0)) / (n_surr + 1),
    )


def grid_freq_ranges(grid: TFGrid):
    """Contiguous frequency ranges covered by the selected grid bins."""
    df = grid.fbin / grid.seg_len
    lo = grid.freq - df / 2
    hi = grid.freq + df / 2
    ranges, cur = [], None
    for k, l, h in zip(grid.keep, lo, hi):
        if k:
            cur = [l, h] if cur is None else [cur[0], h]
        elif cur is not None:
            ranges.append(tuple(cur))
            cur = None
    if cur is not None:
        ranges.append(tuple(cur))
    return tuple(ranges)


# =========================================================================
# 3. Bayesian posterior probability of stationarity
# =========================================================================

def _gauss_marglik(Y, V, extra):
    """log N(Y | 1 mu, diag(V) + extra) with a flat prior on mu, per bin.

    Y, V: (K, J). extra: (..., K, K) broadcastable, added to the covariance.
    Constant terms shared by all models are dropped.
    """
    K, J = Y.shape
    C = extra + np.einsum("kj,kl->jkl", V, np.eye(K))           # (..., J, K, K)
    Ci = np.linalg.inv(C)
    _, logdet = np.linalg.slogdet(C)
    y = np.moveaxis(Y, 1, 0)                                     # (J, K)
    s1 = Ci.sum(axis=(-1, -2))                                   # 1^T C^-1 1
    Ciy = np.einsum("...jkl,jl->...jk", Ci, y)
    yCy = np.einsum("jk,...jk->...j", y, Ciy)
    oCy = Ciy.sum(axis=-1)
    quad = yCy - oCy**2 / s1
    return -0.5 * (logdet + np.log(s1) + quad)


def bayes_test(grid: TFGrid, n_poly=3, tau_range=(0.05, 2.0), n_tau=30,
               n_frac=30, prior_stationary=0.5) -> dict:
    """Posterior probability that the source power spectrum is constant.

    Data: bias-corrected log source power Y_kj with variance V_kj (see module
    docstring), in the selected bins.

    H0: Y_kj = mu_j + e_kj.
    H1: a random fraction pi of bins has Y_kj = mu_j + sum_m E_mk beta_jm
        + e_kj with beta_jm ~ N(0, tau^2); E are the time contrasts of
        `time_basis`. tau is in natural-log power (tau ~ 0.1 is ~10 %).
    mu_j has a flat prior in both models. Priors: tau log-uniform in
    `tau_range`, pi uniform on (0, 1].

    Per-bin outputs (NaN in unused bins):
      log10_BF_per_freq        inclusion Bayes factor of bin j, marginal over
                               tau, pi and the other bins' indicators
      log10_BF_alone_per_freq  B_j(tau) averaged over the tau prior, i.e. bin
                               j analysed on its own
      p_vary_per_freq          P(bin j varies | D, H1)
    """
    keep = grid.keep
    Y, V, _ = log_source(grid.cells[:, keep], grid.noise,
                         grid.S_hat[keep], grid.n)
    K, J = Y.shape
    E = time_basis(K, n_poly)
    EE = E @ E.T
    tau = np.geomspace(*tau_range, n_tau)
    l0 = _gauss_marglik(Y, V, np.zeros((1, 1, K, K)))[0]            # (J,)
    l1 = _gauss_marglik(Y, V, (tau**2)[:, None, None, None] * EE)   # (T, J)
    lnBj = l1 - l0[None, :]

    pis = (np.arange(n_frac) + 0.5) / n_frac
    lp = np.log(pis)[None, :, None] + lnBj[:, None, :]
    lq = np.log1p(-pis)[None, :, None] * np.ones_like(lp)
    per = np.logaddexp(lq, lp)                                      # (T, P, J)
    L = per.sum(axis=2)
    lnB10 = logsumexp(L) - np.log(L.size)
    prior_lo = np.log(prior_stationary) - np.log1p(-prior_stationary)
    post_lo = -lnB10 + prior_lo
    p0 = 1.0 / (1.0 + np.exp(-post_lo))

    lw = L - logsumexp(L)                         # log posterior of (tau, pi)
    w = np.exp(lw)
    # P(bin j varies | D, H1) and its complement, in log space
    ln_in = logsumexp(lw[:, :, None] + lp - per, axis=(0, 1))
    ln_out = logsumexp(lw[:, :, None] + lq - per, axis=(0, 1))
    p_var = np.full(grid.freq.size, np.nan)
    p_var[keep] = np.exp(ln_in)
    # Inclusion Bayes factor p(D | bin j varies) / p(D | bin j constant),
    # marginal over tau, pi and the other bins. With pi uniform the prior
    # inclusion odds are 1, so it equals the posterior inclusion odds.
    bf_incl = np.full(grid.freq.size, np.nan)
    bf_incl[keep] = (ln_in - ln_out) / np.log(10)
    # Standalone Bayes factor: bin j alone, B_j(tau) averaged over the tau
    # prior. Depends on tau_range through the Occam factor.
    bf_alone = np.full(grid.freq.size, np.nan)
    bf_alone[keep] = (logsumexp(lnBj, axis=0) - np.log(n_tau)) / np.log(10)
    imax = np.unravel_index(np.argmax(L), L.shape)
    return dict(
        log10_B01=-lnB10 / np.log(10),
        P_stationary=p0,
        tau_map=tau[imax[0]], pi_map=pis[imax[1]],
        tau_posterior=w.sum(axis=1), tau_grid=tau,
        log10_BF_per_freq=bf_incl,
        log10_BF_alone_per_freq=bf_alone,
        p_vary_per_freq=p_var,
        n_poly=E.shape[1],
    )


# =========================================================================
# 4. p-value calibration
# =========================================================================

def sbb_bound(p, prior_stationary=0.5):
    """Sellke-Bayarri-Berger (2001) lower bound on P(H0 | p).

    Bayes factor B01 >= -e p ln p for p < 1/e (no bound otherwise).
    """
    p = np.asarray(p, dtype=float)
    B = np.where(p < 1 / np.e, -np.e * p * np.log(np.clip(p, 1e-300, 1)), 1.0)
    odds = B * prior_stationary / (1 - prior_stationary)
    return odds / (1 + odds)


# =========================================================================
# Convenience wrapper
# =========================================================================

@dataclass
class StationarityReport:
    grid: TFGrid
    psr: dict
    surrogate: dict | None
    bayes: dict
    extra: dict = field(default_factory=dict)

    def p_values(self) -> dict:
        """Permutation p-values where available, else analytic."""
        p = self.psr
        out = {k: p[k] for k in ("p_total_perm", "p_trend_perm",
                                 "p_trend_max_perm") if k in p}
        if not out:
            out = dict(p_total=p["p_total"], p_trend=p["p_trend"])
        if self.surrogate is not None:
            out["p_surrogate"] = self.surrogate["p_gamma"]
        return out

    def summary(self) -> str:
        p, b, g = self.psr, self.bayes, self.grid
        perm = "p_total_perm" in p
        fk = g.freq[g.keep]
        lines = [
            f"grid: K={g.cells.shape[0]} blocks x {g.freq.size} freq bins "
            f"({g.freq[0]:.2f}-{g.freq[-1]:.2f} Hz), n={g.n} ordinates/cell, "
            f"segments {g.seg_len:g} s, norm={g.norm}",
            f"noise: {g.noise_model}, level {g.noise.min():.3g}-"
            f"{g.noise.max():.3g}; {p['n_bins_used']} bins used "
            f"({fk.min():.2f}-{fk.max():.2f} Hz, noise fraction <= "
            f"{g.max_noise_frac:g}); {p['n_floored']} cells floored",
            "PSR weighted log-analysis (chi2 p | permutation p):",
            f"  time effect         S={p['S_time']:8.1f}  p={p['p_time']:.3g}",
            f"  time x frequency    S={p['S_interaction']:8.1f}  "
            f"p={p['p_interaction']:.3g}",
            f"  total               S={p['S_total']:8.1f}  p={p['p_total']:.3g}"
            + (f" | {p['p_total_perm']:.3g}" if perm else ""),
            f"  trend (deg<={p['n_poly']})      S={p['S_trend']:8.1f}  "
            f"p={p['p_trend']:.3g}"
            + (f" | {p['p_trend_perm']:.3g}" if perm else ""),
            f"  max-bin trend       S={p['trend_max']:8.1f}  at "
            f"{p['f_trend_max']:.2f} Hz"
            + (f"  p_perm={p['p_trend_max_perm']:.3g}" if perm else ""),
        ]
        if self.surrogate is not None:
            s = self.surrogate
            lines += ["Surrogate TF test (Borgnat+10):",
                      f"  INS={s['INS']:.2f}  p_gamma={s['p_gamma']:.3g}  "
                      f"p_rank={s['p_rank']:.3g}"]
        pv = self.p_values()
        p_min = min(pv.values())
        lines += [
            "Bayesian (H0 constant spectrum vs H1 sparse smooth evolution):",
            f"  log10 B01={b['log10_B01']:.2f}  "
            f"P(stationary)={b['P_stationary']:.3g}  "
            f"(H1 MAP: tau={b['tau_map']:.2f}, frac bins={b['pi_map']:.2f})",
            f"SBB lower bound on P(stationary) from smallest p "
            f"({p_min:.3g}, uncorrected for {len(pv)} tests): "
            f"{float(sbb_bound(p_min)):.3g}",
        ]
        return "\n".join(lines)


def run_all(counts=None, dt=None, seg_len=8.0, n_blocks=16, fmin=0.25,
            fmax=8.0, fbin=2, norm="frac", noise="poisson",
            max_noise_frac=0.3, n_poly=3, n_perm=2000, n_surr=99,
            surrogate=True, prior_stationary=0.5, seed=None,
            lc=None, gti=None, bkg_rate=0.0, noise_scale=1.0,
            noise_level_value=None, f_noise=None) -> StationarityReport:
    """Build the grid and run every test.

    Takes a stingray `Lightcurve` (`lc`) or a `counts` array with `dt`. The
    surrogate test needs one uninterrupted stretch, so it is skipped when the
    light curve has more than one GTI.
    """
    rng = np.random.default_rng(seed)
    lc = as_lightcurve(counts, dt, lc, gti)
    grid = build_tf_grid(lc=lc, seg_len=seg_len, n_blocks=n_blocks, fmin=fmin,
                         fmax=fmax, fbin=fbin, norm=norm, noise=noise,
                         max_noise_frac=max_noise_frac, bkg_rate=bkg_rate,
                         noise_scale=noise_scale,
                         noise_level_value=noise_level_value, f_noise=f_noise)
    psr = psr_test(grid, n_poly=n_poly, n_perm=n_perm, rng=rng)
    sur = None
    if surrogate and len(lc.gti) == 1:
        sur = surrogate_test(np.asarray(lc.counts, float), lc.dt,
                             win_len=grid.seg_len, n_surr=n_surr,
                             freq_ranges=grid_freq_ranges(grid), norm=norm,
                             noise="none" if grid.noise_model == "none"
                             else "poisson", rng=rng)
    bay = bayes_test(grid, n_poly=n_poly, prior_stationary=prior_stationary)
    return StationarityReport(grid, psr, sur, bay)
