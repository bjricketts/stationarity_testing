"""
qpo_core.py
-----------
Signal-generation utilities reused from the rest of the QPO work, copied here so
that `stationarity_testing/` is self-contained (nothing imports from sibling
folders).

Provenance
==========
- lorentzian, dho_filter, ou_process, damped_oscillator_convolve,
  build_broadband (was `_build_broadband`), make_spectra:
      QPOs_all/QPO_sims/qpo_sims.py
- apply_filter (was `_apply`):
      QPOs_all/signal_interference/interference_sims.py

Changes relative to the originals
=================================
- ou_process: optional stationary initial state (default on); the original
  starts at the mean, so its variance ramps up over ~tau.
- damped_oscillator_convolve: identical update equations, but the per-step
  transition coefficients are precomputed with numpy (the original computed
  them inside the Python loop). The recursion itself is unchanged. If numba is
  installed the recursion is JIT-compiled.
- build_broadband: seeds numpy's global RNG around the stingray call, since
  stingray's Timmer-Koenig draw ignores `random_state` (so the original
  `_build_broadband(seed=...)` is not reproducible). The Simulator is run
  with poisson=False so the template carries no counting noise (the
  original used poisson=True at mean=100, which adds a white component).
  Also normalises the returned realisation to unit variance on
  request (`unit=True`) so callers can set the fractional rms directly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from stingray import Lightcurve, AveragedCrossspectrum, AveragedPowerspectrum
from stingray import simulator

try:  # optional speed-up for the time-varying oscillator recursion
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


# -------------------------------------------------------------------------
# From QPO_sims/qpo_sims.py
# -------------------------------------------------------------------------

def lorentzian(f, f0, q, norm):
    """Closed-form Lorentzian (no ndspec dependency)."""
    df = f0 / q
    return norm * (df / np.pi) / ((f - f0)**2 + df**2)


def dho_filter(x, dt, f0, zeta, gain=1.0):
    """Pass a real signal through a damped harmonic oscillator H(f).

    H(f) = w0^2 / (w0^2 - w^2 + 2j zeta w0 w),  w = 2*pi*f, w0 = 2*pi*f0.
    Q-factor is 1/(2*zeta).
    """
    N = len(x)
    freq = np.fft.rfftfreq(N, d=dt)
    w_ = 2 * np.pi * freq
    w0 = 2 * np.pi * f0
    H = w0**2 / (w0**2 - w_**2 + 2j * zeta * w0 * w_)
    X = np.fft.rfft(x - x.mean())
    return np.fft.irfft(gain * H * X, n=N)


def ou_process(N, dt, mean, tau, sigma, rng=None, stationary_start=True):
    """Discrete-time Ornstein-Uhlenbeck process.

    x[n+1] = mean + alpha*(x[n] - mean) + eta * xi[n],
    alpha = exp(-dt/tau), eta = sigma * sqrt(1 - alpha**2).
    Stationary distribution N(mean, sigma**2); zero-centred Lorentzian PSD
    with HWHM 1/(2*pi*tau).

    stationary_start=True draws the initial state from N(mean, sigma**2).
    The QPO_sims original starts at x = mean, so its variance ramps up over
    ~tau, which is itself a non-stationarity when tau is not << N*dt.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = np.exp(-dt / tau)
    eta = sigma * np.sqrt(1.0 - alpha**2)
    zi = None
    if stationary_start:
        zi = [alpha * sigma * rng.standard_normal()]
    xi = rng.standard_normal(N)
    if zi is None:
        y = lfilter([eta], [1.0, -alpha], xi)
    else:
        y, _ = lfilter([eta], [1.0, -alpha], xi, zi=zi)
    return mean + y


def _oscillator_coefficients(omega, gamma, dt):
    """Exact one-step transition matrix of x'' + 2 gamma x' + omega^2 x = 0."""
    wd_sq = omega * omega - gamma * gamma
    wd = np.where(wd_sq > 0, np.sqrt(np.abs(wd_sq)), 1e-12)
    e = np.exp(-gamma * dt)
    c = np.cos(wd * dt)
    s = np.sin(wd * dt)
    a11 = e * (c + (gamma / wd) * s)
    a12 = e * s / wd
    a21 = -e * (omega * omega / wd) * s
    a22 = e * (c - (gamma / wd) * s)
    return a11, a12, a21, a22


def _recursion_py(a11, a12, a21, a22, force, dt):
    N = force.size
    x = np.zeros(N)
    v = np.zeros(N)
    xp = vp = 0.0
    for n in range(1, N):
        xn = a11[n - 1] * xp + a12[n - 1] * vp
        vn = a21[n - 1] * xp + a22[n - 1] * vp + force[n - 1] * dt
        x[n] = xn
        v[n] = vn
        xp, vp = xn, vn
    return x


_recursion = njit(cache=True)(_recursion_py) if njit is not None else _recursion_py


def damped_oscillator_convolve(noise, omega, gamma, dt=1.0):
    """Drive a damped oscillator with time-varying omega and gamma.

    Solves x'' + 2 gamma(t) x' + omega(t)^2 x = noise(t) with a piecewise-exact
    state-space step. omega is angular frequency [rad/s], gamma the amplitude
    decay rate [1/s]; Q = omega / (2 gamma), PSD FWHM = gamma/pi [Hz].
    """
    noise = np.asarray(noise, dtype=float)
    N = noise.size
    omega = np.broadcast_to(np.asarray(omega, dtype=float), (N,))
    gamma = np.broadcast_to(np.asarray(gamma, dtype=float), (N,))
    a11, a12, a21, a22 = _oscillator_coefficients(omega, gamma, dt)
    return _recursion(a11, a12, a21, a22, noise, float(dt))


def build_broadband(dt, T_bins, rms=0.5, mean=100, seed=None, unit=False):
    """Two-Lorentzian broadband Timmer-Koenig realisation (stingray).

    Returns (broadband, t); `broadband` is mean-subtracted, and scaled to unit
    variance if `unit=True`.
    """
    kwargs = {} if seed is None else {"random_state": seed}
    sim = simulator.Simulator(N=int(T_bins), mean=mean, dt=dt, rms=rms,
                              poisson=False, **kwargs)
    w = np.fft.rfftfreq(sim.N, d=sim.dt)[1:]
    psd_model = lorentzian(w, 0.4, 0.3, 1.0) + lorentzian(w, 2.0, 0.5, 1.2)
    # stingray (2.x) draws the Timmer-Koenig Fourier amplitudes from the
    # global np.random state and ignores `random_state` there, so seed the
    # global state explicitly (and restore it afterwards).
    saved = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        lc_bb = sim.simulate(psd_model)
    finally:
        np.random.set_state(saved)
    bb = lc_bb.counts - lc_bb.counts.mean()
    if unit:
        bb = bb / bb.std()
    t = np.arange(lc_bb.n) * dt
    return bb, t


def make_spectra(lc1, lc2, segment_size):
    """Averaged fractional-rms power spectra and cross spectrum (stingray)."""
    ps1 = AveragedPowerspectrum.from_lightcurve(lc1, norm="frac",
                                                segment_size=segment_size)
    ps2 = AveragedPowerspectrum.from_lightcurve(lc2, norm="frac",
                                                segment_size=segment_size)
    cs = AveragedCrossspectrum.from_lightcurve(lc1, lc2, norm="frac",
                                               segment_size=segment_size)
    return ps1, ps2, cs


# -------------------------------------------------------------------------
# From signal_interference/interference_sims.py
# -------------------------------------------------------------------------

def apply_filter(x: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Convolve a real signal with a frequency-domain filter H (rfft grid)."""
    X = np.fft.rfft(x - x.mean())
    return np.fft.irfft(H * X, n=len(x))
