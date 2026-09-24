#!/usr/bin/env python3
"""
mc_fm_wander.py
---------------
Monte Carlo of the stationary frequency-wander model (`fm_wander`) as a
function of the wander correlation time tau.

f0(t) = f0 + OU(sigma, tau) is a stationary process, but when tau is not much
shorter than the analysis blocks the block spectra differ between blocks. This
script measures how often each test rejects stationarity as tau grows, which
is the false-positive rate while tau << block length and the rate of
"timescale-limited" detections beyond that.

For each tau it records, per simulation: analytic (chi^2) and permutation
p-values of the PSR total and trend statistics, the permutation p-value of
the max-bin statistic, P(stationary), and optionally the surrogate-test
p-value. It prints a table, writes a CSV of every simulation, and plots the
rejection rate against tau / block length.

Example
=======
    python mc_fm_wander.py --taus 1 5 20 64 256 --nsim 100 --workers 4
    python mc_fm_wander.py --taus 5 64 --nsim 50 --surrogate --n-surr 49
"""
import argparse
import csv
import os
import sys
import time
import warnings

# one BLAS thread per worker (threaded BLAS + fork can deadlock)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")

from nonstationary_qpo import QPOSpec, simulate  # noqa: E402
from stationarity_tests import (build_tf_grid, psr_test, bayes_test,  # noqa: E402
                                surrogate_test, grid_freq_ranges)

COLS = ["p_total_chi2", "p_trend_chi2", "p_total_perm", "p_trend_perm",
        "p_maxbin_perm", "P_stationary", "p_surrogate", "p_scan_perm"]
LABELS = ["total chi2", "trend chi2", "total perm", "trend perm",
          "max-bin perm", "P(stat)<0.05", "surrogate", "window scan perm"]


def one(job):
    tau, seed, a = job
    spec = QPOSpec(fm_sigma=a.sigma, fm_tau=tau, mean=a.mean, T=a.T)
    sim = simulate(spec, seed=seed)
    g = build_tf_grid(sim.counts, spec.dt, seg_len=a.seg, n_blocks=a.blocks,
                      fmin=a.fmin, fmax=a.fmax, fbin=a.fbin)
    r = psr_test(g, n_poly=a.n_poly, n_perm=a.n_perm, rng=seed)
    b = bayes_test(g, n_poly=a.n_poly)
    ps = np.nan
    if a.surrogate:
        ps = surrogate_test(sim.counts, spec.dt, win_len=a.seg,
                            n_surr=a.n_surr,
                            freq_ranges=grid_freq_ranges(g), rng=seed)["p_gamma"]
    return (tau, seed, r["p_total"], r["p_trend"], r["p_total_perm"],
            r["p_trend_perm"], r["p_trend_max_perm"], b["P_stationary"], ps,
            r["p_scan_perm"])


def wilson(k, n, z=1.96):
    """95 % Wilson interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[1, 5, 20, 64, 256], help="OU correlation times [s]")
    ap.add_argument("--nsim", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--sigma", type=float, default=0.15, help="OU std of f0 [Hz]")
    ap.add_argument("--mean", type=float, default=1.0e4, help="count rate")
    ap.add_argument("--T", type=float, default=1024.0, help="duration [s]")
    ap.add_argument("--seg", type=float, default=8.0)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--fmin", type=float, default=0.25)
    ap.add_argument("--fmax", type=float, default=8.0)
    ap.add_argument("--fbin", type=int, default=2)
    ap.add_argument("--n-poly", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--surrogate", action="store_true",
                    help="also run the surrogate test (slow)")
    ap.add_argument("--n-surr", type=int, default=49)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--outdir", default="mc_results")
    ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()

    block_len = a.T / a.blocks
    os.makedirs(a.outdir, exist_ok=True)
    jobs = [(tau, a.seed0 + s, a) for tau in a.taus for s in range(a.nsim)]
    print(f"{len(jobs)} simulations ({len(a.taus)} taus x {a.nsim}) on "
          f"{a.workers} workers; block length {block_len:g} s, "
          f"segment {a.seg:g} s")
    t0 = time.time()
    with Pool(a.workers) as p:
        rows = []
        for i, row in enumerate(p.imap_unordered(one, jobs, chunksize=1), 1):
            rows.append(row)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} done, {el:.0f} s elapsed, "
                      f"~{el / i * (len(jobs) - i):.0f} s left", flush=True)
    rows.sort()
    R = np.array(rows, dtype=float)

    csv_path = os.path.join(a.outdir, "fm_wander_tau_scan.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau", "seed"] + COLS)
        w.writerows(rows)

    # rejection rates
    use = [i for i, c in enumerate(COLS) if c != "p_surrogate" or a.surrogate]
    rates = np.full((len(a.taus), len(COLS)), np.nan)
    los, his = rates.copy(), rates.copy()
    head = f"{'tau [s]':>8} {'tau/block':>9} " + " ".join(
        f"{LABELS[i]:>16}" for i in use) + f" {'med P(stat)':>12}"
    print("\nFraction rejecting stationarity at "
          f"alpha = {a.alpha} (95 % interval in brackets)")
    print(head)
    for it, tau in enumerate(a.taus):
        sub = R[R[:, 0] == tau][:, 2:]
        n = len(sub)
        cells = []
        for i in use:
            k = int(np.sum(sub[:, i] < a.alpha))
            lo, hi = wilson(k, n)
            rates[it, i], los[it, i], his[it, i] = k / n, lo, hi
            cells.append(f"{k / n:5.2f}[{lo:.2f},{hi:.2f}]")
        print(f"{tau:8g} {tau / block_len:9.3g} " + " ".join(
            f"{c:>16}" for c in cells) + f" {np.median(sub[:, 5]):12.3g}")

    # uniformity of permutation p-values (KS), useful for the fast-wander end
    print("\nKS test of p-value uniformity (p_KS < 0.01 suggests miscalibration)")
    for it, tau in enumerate(a.taus):
        sub = R[R[:, 0] == tau][:, 2:]
        ks = {LABELS[i]: stats.kstest(sub[:, i], "uniform").pvalue
              for i in (0, 1, 2, 3)}
        print(f"  tau={tau:g}: " + ", ".join(f"{k} {v:.2g}" for k, v in ks.items()))

    # plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.array(a.taus) / block_len
    for i in use:
        ax.errorbar(x, rates[:, i],
                    yerr=[rates[:, i] - los[:, i], his[:, i] - rates[:, i]],
                    marker="o", ms=4, capsize=2, label=LABELS[i])
    ax.axhline(a.alpha, color="0.5", ls=":", lw=1)
    ax.axvline(1.0, color="0.5", ls="--", lw=1)
    ax.axvline(a.seg / block_len, color="0.7", ls="--", lw=1)
    ax.text(1.0, 1.02, "block", ha="center", fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.text(a.seg / block_len, 1.02, "segment", ha="center", fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\tau$ / block length")
    ax.set_ylabel(rf"fraction with $p<{a.alpha}$")
    ax.set_title(rf"fm_wander: $\sigma_f$={a.sigma} Hz, {a.nsim} sims per $\tau$")
    ax.legend(fontsize=8, loc="upper left")
    png = os.path.join(a.outdir, "fm_wander_tau_scan.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n-> {csv_path}\n-> {png}\n({time.time() - t0:.0f} s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
