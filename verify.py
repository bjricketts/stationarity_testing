#!/usr/bin/env python3
"""
verify.py
---------
Checks that the simulator does what it claims and that the stationarity tests
are calibrated. Run: `python verify.py [--nsim 40] [--workers 4]`
(exits non-zero if a check fails).

1. Simulator: the scaled time-varying oscillator has unit variance, and the
   imposed drift appears in the right place (QPO centroid moves by d_f0).
2. Calibration under H0 (stationary, lognormal rms-flux): the fraction of
   p < 0.05 is consistent with 5 %, and P(stationary) is rarely small.
   Includes a low count rate (500 ct/s) and a mean rate that changes by
   +/-60 % at fixed fractional variability, so the Poisson level changes
   while the source spectrum does not.
3. Without noise subtraction, that rate change is detected (a false
   positive), which is what the subtraction fixes.
4. Power under H1: strong drifts in f0 and rms are detected, also at
   1000 ct/s.
Monte Carlo checks the analytic (chi^2) and 200-draw permutation p-values
of the PSR statistics, and skips the surrogate test (slow).
"""
import argparse
import os
import sys

# one BLAS thread per worker (threaded BLAS + fork can deadlock)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
from multiprocessing import Pool

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

from qpo_core import damped_oscillator_convolve  # noqa: E402
from nonstationary_qpo import QPOSpec, simulate, MODEL_DEFAULTS  # noqa: E402
from stationarity_tests import build_tf_grid, psr_test, bayes_test  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAIL.append(name)


def one(job):
    model, kw, tkw, seed = job
    spec = QPOSpec(**{**MODEL_DEFAULTS[model], **kw})
    sim = simulate(spec, seed=seed)
    g = build_tf_grid(sim.counts, spec.dt, **tkw)
    r = psr_test(g, n_perm=200, rng=seed)
    b = bayes_test(g)
    return (r["p_total"], r["p_trend"], r["p_trend_max_perm"], b["P_stationary"],
            r["p_total_perm"], r["p_trend_perm"])


def binom_ok(k, n, p=0.05, alpha=0.01):
    """k false positives in n trials consistent with rate p (two-sided)."""
    return stats.binomtest(k, n, p).pvalue > alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    print("== 1. Simulator ==")
    rng = np.random.default_rng(0)
    dt, N = 1 / 512, 2**19
    w = 2 * np.pi * np.linspace(1.5, 3.0, N)          # chirping oscillator
    g = w / (2 * 8.0)
    x = damped_oscillator_convolve(rng.standard_normal(N) * 2 * w * np.sqrt(g / dt),
                                   w, g, dt)[4096:]
    halves = [x[: x.size // 2].var(), x[x.size // 2:].var()]
    check("unit variance with time-varying f0", all(abs(h - 1) < 0.1 for h in halves),
          f"var = {halves[0]:.3f}, {halves[1]:.3f}")

    spec = QPOSpec(d_f0=0.2, bb_rms=0.0, poisson=False)
    sim = simulate(spec, seed=1)
    grid = build_tf_grid(sim.counts, spec.dt, fbin=1, noise="none")
    peak = grid.freq[np.argmax(grid.cells, axis=1)]
    check("QPO centroid follows f0(t)",
          abs(peak[0] - 1.6) < 0.2 and abs(peak[-1] - 2.4) < 0.2,
          f"first block {peak[0]:.2f} Hz (expect ~1.61), "
          f"last {peak[-1]:.2f} Hz (expect ~2.39)")

    print(f"== 2-4. Monte Carlo ({a.nsim} sims per case) ==")
    # (label, model, sim kwargs, test kwargs, role)
    cases = [
        ("stationary", "stationary", {}, {}, "null"),
        ("lognormal", "lognormal", {}, {}, "null"),
        ("low_rate", "low_rate", {}, {}, "null"),
        ("rate_drift", "rate_drift", {"mean": 1000.0}, {}, "null"),
        ("rate_drift, no noise subtraction", "rate_drift", {"mean": 1000.0},
         {"noise": "none", "max_noise_frac": None}, "demo"),
        ("freq_drift", "freq_drift", {"d_f0": 0.10}, {}, "alt"),
        ("rms_drift", "rms_drift", {"d_rms": 0.6}, {}, "alt"),
        ("rms_drift @ 1000 ct/s", "rms_drift", {"d_rms": 0.6, "mean": 1000.0},
         {}, "alt"),
    ]
    jobs = [(m, kw, tkw, s) for _, m, kw, tkw, _ in cases for s in range(a.nsim)]
    with Pool(a.workers) as p:
        res = np.array(p.map(one, jobs)).reshape(len(cases), a.nsim, 6)

    for (m, _, _, _, role), r in zip(cases, res):
        cols = [0, 1, 4, 5, 2]
        names = ["total chi2", "trend chi2", "total perm", "trend perm",
                 "max-bin perm"]
        k = (r[:, cols] < 0.05).sum(axis=0)
        print(f"  {m}: p<0.05 counts (" + ", ".join(names) + ") = "
              f"{tuple(int(v) for v in k)}; median P(stat) = {np.median(r[:, 3]):.3g}")
        if role == "demo":
            check(f"{m}: changing Poisson level is (wrongly) detected, "
                  "showing why subtraction is needed",
                  np.mean(r[:, 0] < 0.05) > 0.5, f"{np.mean(r[:, 0] < 0.05):.2f}")
        elif role == "null":
            for name, kk in zip(names, k):
                check(f"{m}: {name} false-positive rate ~5 %", binom_ok(kk, a.nsim),
                      f"{kk}/{a.nsim}")
            check(f"{m}: P(stationary) < 0.05 in <= 5 % of sims",
                  np.mean(r[:, 3] < 0.05) <= 0.05 + 2 / a.nsim,
                  f"{np.mean(r[:, 3] < 0.05):.2f}")
        else:
            check(f"{m}: trend test detects in >= 90 %",
                  np.mean(r[:, 1] < 0.05) >= 0.9, f"{np.mean(r[:, 1] < 0.05):.2f}")
            check(f"{m}: median P(stationary) < 0.01",
                  np.median(r[:, 3]) < 0.01, f"{np.median(r[:, 3]):.2g}")

    print("\nALL CHECKS PASSED" if not FAIL else f"\nFAILED: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
