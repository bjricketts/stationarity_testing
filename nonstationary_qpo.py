#!/usr/bin/env python3
"""
nonstationary_qpo.py
--------------------
Simulate a QPO whose statistical properties change with time, then test the
light curve for (second-order) stationarity.

Model
=====
The QPO is a damped harmonic oscillator driven by white noise, integrated
with the time-varying state-space step from QPO_sims (`damped_oscillator_
convolve`):

    q'' + 2 gamma(t) q' + omega(t)^2 q = s(t) xi(t),
    omega = 2 pi f0(t),   gamma = omega / (2 Q(t)).

The driver is scaled by s(t) = 2 omega sqrt(gamma / dt), which keeps the
instantaneous variance of q at unity whatever f0(t) and Q(t) do, so a change
of centroid or coherence does not also change the QPO amplitude. The
amplitude is set separately by an envelope rms(t). A stationary broadband
component (the two-Lorentzian Timmer-Koenig realisation from QPO_sims) is
added, and the rate is Poisson sampled:

    rate(t) = mean(t) * (1 + bb_rms * b(t) + rms(t) * q(t))

Each parameter p in {f0, Q, rms, mean} evolves as p(t) = p0 * (1 + delta_p * g(t)),
with g(t) one of the time profiles below (|g| <= 1).

Models
======
  stationary   control: nothing varies                       (H0 true)
  fm_wander    control: f0 wanders as a stationary OU process
               (same idea as QPO_sims `fm`)                   (H0 true)
  lognormal    control: stationary, but rate = exp(...), giving a linear
               rms-flux relation (Uttley+05)                  (H0 true)
  low_rate     control: stationary at 500 ct/s, where the Poisson level
               dominates most bins                            (H0 true)
  rate_drift   control: the mean rate changes by +/-60 % while the
               fractional variability is fixed. The Poisson level 2/rate
               changes, the frac-normalised source spectrum does not
                                                              (H0 true for
                                                               the source)
  freq_drift   centroid frequency changes with time           (H0 false)
  rms_drift    QPO fractional rms changes with time           (H0 false)
  q_drift      QPO coherence Q changes with time              (H0 false)
  all          run every model above

Profiles (--profile): linear, step, sine, burst.

Examples
========
    python nonstationary_qpo.py freq_drift --outdir figures
    python nonstationary_qpo.py freq_drift --delta 0.05 --profile sine --period 256
    python nonstationary_qpo.py rms_drift --delta 0.3 --profile step
    python nonstationary_qpo.py stationary --no-surrogate
    python nonstationary_qpo.py all --outdir figures --seed 3
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, asdict

import numpy as np

warnings.filterwarnings("ignore", message=".*numba.*")
warnings.filterwarnings("ignore", message=".*SIMON says.*")

from stingray import Lightcurve  # noqa: E402

from qpo_core import (build_broadband, damped_oscillator_convolve,  # noqa: E402
                      ou_process)
from plotting import plot_report  # noqa: E402
from stationarity_tests import run_all  # noqa: E402


# =========================================================================
# Time profiles
# =========================================================================

def profile(t, kind, T, period=None, width=None):
    """Dimensionless evolution g(t) with |g| <= 1."""
    x = t / T
    if kind == "linear":
        return 2.0 * x - 1.0
    if kind == "step":
        return np.where(x < 0.5, -1.0, 1.0)
    if kind == "sine":
        P = T / 2 if period is None else period
        return np.sin(2 * np.pi * t / P)
    if kind == "burst":
        w = T / 10 if width is None else width
        return np.exp(-0.5 * ((t - T / 2) / w) ** 2)
    raise ValueError(kind)


# =========================================================================
# Simulation
# =========================================================================

@dataclass
class QPOSpec:
    dt: float = 1 / 512
    T: float = 1024.0
    mean: float = 1.0e4       # count rate [ct/s]
    f0: float = 2.0           # QPO centroid [Hz]
    Q: float = 8.0            # QPO quality factor
    rms: float = 0.10         # QPO fractional rms
    bb_rms: float = 0.20      # broadband fractional rms
    d_f0: float = 0.0         # fractional change amplitude of f0
    d_Q: float = 0.0          # ... of Q
    d_rms: float = 0.0        # ... of rms
    d_mean: float = 0.0       # ... of the mean count rate (fractional
                              #     variability, hence source power in
                              #     frac units, unchanged)
    profile: str = "linear"
    period: float | None = None
    width: float | None = None
    fm_sigma: float = 0.0     # stationary OU wander of f0 [Hz]
    fm_tau: float = 5.0       # OU correlation time [s]
    lognormal: bool = False
    poisson: bool = True


MODEL_DEFAULTS = {
    "stationary": {},
    "fm_wander":  dict(fm_sigma=0.15, fm_tau=5.0),
    "lognormal":  dict(lognormal=True, bb_rms=0.30),
    "low_rate":   dict(mean=500.0),
    "rate_drift": dict(d_mean=0.6),
    "freq_drift": dict(d_f0=0.10),
    "rms_drift":  dict(d_rms=0.50),
    "q_drift":    dict(d_Q=0.80),
}
NULL_MODELS = {"stationary", "fm_wander", "lognormal", "low_rate",
               "rate_drift"}


@dataclass
class SimOutput:
    t: np.ndarray
    counts: np.ndarray
    f0_t: np.ndarray
    Q_t: np.ndarray
    rms_t: np.ndarray
    mean_t: np.ndarray
    spec: QPOSpec

    def lightcurve(self) -> Lightcurve:
        return Lightcurve(self.t, self.counts, dt=self.spec.dt, skip_checks=True)


def simulate(spec: QPOSpec, seed=0) -> SimOutput:
    rng = np.random.default_rng(seed)
    dt = spec.dt
    N = int(round(spec.T / dt))
    t = np.arange(N) * dt
    g = profile(t, spec.profile, spec.T, spec.period, spec.width)

    f0_t = spec.f0 * (1.0 + spec.d_f0 * g)
    if spec.fm_sigma > 0:
        f0_t = f0_t + ou_process(N, dt, 0.0, spec.fm_tau, spec.fm_sigma, rng=rng)
    Q_t = spec.Q * (1.0 + spec.d_Q * g)
    rms_t = spec.rms * (1.0 + spec.d_rms * g)
    mean_t = spec.mean * (1.0 + spec.d_mean * g)
    if (np.any(f0_t <= 0) or np.any(Q_t <= 0.5) or np.any(rms_t < 0)
            or np.any(mean_t <= 0)):
        raise ValueError("parameter evolution leaves the physical range")

    omega = 2 * np.pi * f0_t
    gamma = omega / (2 * Q_t)
    drive = rng.standard_normal(N) * 2 * omega * np.sqrt(gamma / dt)
    # burn in so the oscillator starts in its stationary state
    # (20 decay times at the initial parameters)
    n_burn = int(min(N, 20 / gamma[0] / dt))
    full_drive = np.concatenate([rng.standard_normal(n_burn) * 2 * omega[0]
                                 * np.sqrt(gamma[0] / dt), drive])
    full_om = np.concatenate([np.full(n_burn, omega[0]), omega])
    full_ga = np.concatenate([np.full(n_burn, gamma[0]), gamma])
    q = damped_oscillator_convolve(full_drive, full_om, full_ga, dt)[n_burn:]

    b, _ = build_broadband(dt, N, seed=int(rng.integers(2**31)), unit=True)
    v = spec.bb_rms * b + rms_t * q
    if spec.lognormal:
        # exponentiated Gaussian: stationary, linear rms-flux relation.
        # Scale so the fractional rms is close to that of the linear model.
        s = np.sqrt(np.log1p(v.var()))
        y = s * v / v.std()
        rate = mean_t * np.exp(y - 0.5 * s**2)
    else:
        rate = mean_t * (1.0 + v)
    rate = np.clip(rate, 0.0, None)
    counts = rng.poisson(rate * dt) if spec.poisson else rate * dt
    return SimOutput(t, counts.astype(float), f0_t, Q_t, rms_t, mean_t, spec)


# =========================================================================
# Plotting (shared report figure in plotting.py)
# =========================================================================

def plot_result(sim: SimOutput, rep, savepath, title=""):
    """Report figure with the injected parameter evolution overlaid."""
    nb = int(round(1.0 / sim.spec.dt))
    t, spec = sim.t[::nb], sim.spec
    lines = [(t, sim.f0_t[::nb] / spec.f0, r"$f_0/f_{0,\rm ref}$", "C0"),
             (t, sim.Q_t[::nb] / spec.Q, r"$Q/Q_{\rm ref}$", "C2"),
             (t, sim.rms_t[::nb] / spec.rms, r"rms/rms$_{\rm ref}$", "C3"),
             (t, sim.mean_t[::nb] / spec.mean, r"rate/rate$_{\rm ref}$", "C1")]
    track = (t, sim.f0_t[::nb], sim.f0_t[::nb] / (2 * sim.Q_t[::nb]))
    plot_report(rep, savepath, sim.t, sim.counts, spec.dt, track=track,
                param_lines=lines, title=title)


# =========================================================================
# CLI
# =========================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=list(MODEL_DEFAULTS) + ["all"])
    p.add_argument("--outdir", default="figures")
    p.add_argument("--ext", default="png")
    p.add_argument("--seed", type=int, default=0)

    s = p.add_argument_group("simulation")
    s.add_argument("--T", type=float, help="duration [s] (1024)")
    s.add_argument("--dt", type=float, help="time bin [s] (1/512)")
    s.add_argument("--mean", type=float, help="count rate [ct/s] (1e4)")
    s.add_argument("--f0", type=float, help="QPO centroid [Hz] (2)")
    s.add_argument("--Q", type=float, help="QPO quality factor (8)")
    s.add_argument("--rms", type=float, help="QPO fractional rms (0.1)")
    s.add_argument("--bb-rms", type=float, help="broadband fractional rms (0.2)")
    s.add_argument("--delta", type=float,
                   help="fractional change of the model's drifting parameter")
    s.add_argument("--d-f0", type=float)
    s.add_argument("--d-Q", type=float)
    s.add_argument("--d-rms", type=float)
    s.add_argument("--d-mean", type=float,
                   help="fractional change of the mean count rate")
    s.add_argument("--profile", choices=["linear", "step", "sine", "burst"])
    s.add_argument("--period", type=float, help="sine period [s] (T/2)")
    s.add_argument("--width", type=float, help="burst Gaussian sigma [s] (T/10)")
    s.add_argument("--fm-sigma", type=float)
    s.add_argument("--fm-tau", type=float)
    s.add_argument("--no-poisson", action="store_true")

    t = p.add_argument_group("tests")
    t.add_argument("--seg", type=float, default=8.0, help="segment length [s]")
    t.add_argument("--blocks", type=int, default=16, help="number of time blocks")
    t.add_argument("--fmin", type=float, default=0.25)
    t.add_argument("--fmax", type=float, default=8.0)
    t.add_argument("--fbin", type=int, default=2,
                   help="Fourier bins averaged per frequency bin")
    t.add_argument("--norm", choices=["frac", "abs"], default="frac")
    t.add_argument("--noise", choices=["poisson", "none"], default="poisson",
                   help="Poisson-noise treatment (default: subtract 2/rate)")
    t.add_argument("--max-noise-frac", type=float, default=0.3,
                   help="use only bins where the Poisson level is at most "
                        "this fraction of the total power")
    t.add_argument("--n-poly", type=int, default=3,
                   help="polynomial degree of the time-evolution model; "
                        "0 = arbitrary time dependence")
    t.add_argument("--n-perm", type=int, default=2000)
    t.add_argument("--n-surr", type=int, default=99)
    t.add_argument("--no-surrogate", action="store_true")
    t.add_argument("--prior", type=float, default=0.5,
                   help="prior probability of stationarity")
    return p


_DRIFT_KEY = {"freq_drift": "d_f0", "rms_drift": "d_rms", "q_drift": "d_Q",
              "rate_drift": "d_mean"}


def spec_from_args(model, args) -> QPOSpec:
    kw = dict(MODEL_DEFAULTS[model])
    if args.delta is not None and model in _DRIFT_KEY:
        kw[_DRIFT_KEY[model]] = args.delta
    for name in ["T", "dt", "mean", "f0", "Q", "rms", "bb_rms", "d_f0", "d_Q",
                 "d_rms", "d_mean", "profile", "period", "width", "fm_sigma", "fm_tau"]:
        v = getattr(args, name)
        if v is not None:
            kw[name] = v
    kw["poisson"] = not args.no_poisson
    return QPOSpec(**kw)


def _jsonable(d):
    return {k: (float(v) if np.ndim(v) == 0 else None) for k, v in d.items()
            if np.ndim(v) == 0}


def run_one(model, args):
    spec = spec_from_args(model, args)
    print(f"== {model} ==")
    sim = simulate(spec, seed=args.seed)
    rep = run_all(sim.counts, spec.dt, seg_len=args.seg, n_blocks=args.blocks,
                  fmin=args.fmin, fmax=args.fmax, fbin=args.fbin,
                  norm=args.norm, noise=args.noise,
                  max_noise_frac=args.max_noise_frac, n_poly=args.n_poly or None,
                  n_perm=args.n_perm, n_surr=args.n_surr,
                  surrogate=not args.no_surrogate, prior_stationary=args.prior,
                  seed=args.seed + 1)
    print(rep.summary())
    truth = "stationary" if model in NULL_MODELS else "non-stationary"
    print(f"  (truth: {truth})")

    os.makedirs(args.outdir, exist_ok=True)
    tag = model if model in NULL_MODELS else f"{model}_{spec.profile}"
    plot_result(sim, rep, os.path.join(args.outdir, f"{tag}.{args.ext}"),
                title=f"{model} ({truth})")
    with open(os.path.join(args.outdir, f"{tag}.json"), "w") as fh:
        json.dump(dict(model=model, truth=truth, spec=asdict(spec),
                       psr=_jsonable(rep.psr),
                       surrogate=None if rep.surrogate is None
                       else _jsonable(rep.surrogate),
                       bayes=_jsonable(rep.bayes)), fh, indent=2)
    return rep


def main():
    args = build_parser().parse_args()
    models = list(MODEL_DEFAULTS) if args.model == "all" else [args.model]
    for m in models:
        run_one(m, args)


if __name__ == "__main__":
    main()
