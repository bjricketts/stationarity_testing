#!/usr/bin/env python3
"""
test_lightcurve.py
------------------
Test a real light curve for second-order stationarity.

Reads a FITS event list, a FITS light curve, a text/CSV file or a NumPy file
(see `lightcurve_io.py`), cuts it into gap-free segments inside the GTIs, runs
the tests of `stationarity_tests.py` and writes the same report figure the
simulator produces, plus a JSON summary.

Examples
========
    # NICER event list, 0.5-10 keV (PI in units of 10 eV), binned at 1/512 s
    python test_lightcurve.py ni1234_cl.evt --dt 0.0019531 --pi 50 1000 \\
        --seg 8 --blocks 16 --fmax 8

    # FITS light curve from xselect, and a background rate to undo dilution
    python test_lightcurve.py src.lc --bkg-rate 12.4

    # two-column text file of rate vs time, dead-time-affected noise level
    python test_lightcurve.py lc.txt --dt 0.008 --noise highfreq --f-noise 40

    # tell the tests what to resolve: 32 s segments, 8 blocks
    python test_lightcurve.py src.lc --seg 32 --blocks 8 --fbin 8

Notes
=====
- Segments never span a gap or a GTI boundary; leftover bins are dropped.
- Blocks are groups of consecutive segments, so with gaps they cover unequal
  spans of wall-clock time. The block times in the figure are the mean segment
  start time of each block.
- The surrogate test needs one uninterrupted stretch, so it is skipped when
  the light curve has gaps.
- Quote the permutation p-values. The chi^2 ones assume Gaussian statistics,
  which nonlinear variability (a wandering QPO, say) breaks.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*numba.*")

from lightcurve_io import (load_lightcurve, n_segments,  # noqa: E402
                           is_contiguous, describe)
from plotting import plot_report  # noqa: E402
from stationarity_tests import run_all  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="event list, light curve, text or NumPy file")

    r = p.add_argument_group("reading")
    r.add_argument("--dt", type=float,
                   help="time bin [s]; required for event lists")
    r.add_argument("--pi", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="PI/PHA filter for event lists")
    r.add_argument("--energy", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="ENERGY filter for event lists (needs an --rmf, "
                        "or an ENERGY column)")
    r.add_argument("--rmf", help="RMF used to convert PI to energy")
    r.add_argument("--tstart", type=float)
    r.add_argument("--tstop", type=float)
    r.add_argument("--columns", type=int, nargs=2, default=(0, 1),
                   metavar=("TIME", "Y"), help="text-file column indices")
    r.add_argument("--kind", choices=["auto", "rate", "counts"], default="auto",
                   help="what the text-file Y column holds")

    t = p.add_argument_group("tests")
    t.add_argument("--seg", type=float, default=8.0, help="segment length [s]")
    t.add_argument("--blocks", type=int, default=16, help="number of time blocks")
    t.add_argument("--fmin", type=float, default=0.25)
    t.add_argument("--fmax", type=float, default=8.0)
    t.add_argument("--fbin", type=int, default=2,
                   help="Fourier bins averaged per frequency bin")
    t.add_argument("--norm", choices=["frac", "abs"], default="frac")
    t.add_argument("--noise", choices=["poisson", "none", "highfreq"],
                   default="poisson")
    t.add_argument("--f-noise", type=float,
                   help="lower frequency for the 'highfreq' noise estimate")
    t.add_argument("--noise-level", type=float,
                   help="explicit noise level, in the chosen normalisation")
    t.add_argument("--noise-scale", type=float, default=1.0,
                   help="multiply the noise level (e.g. for dead time)")
    t.add_argument("--bkg-rate", type=float, default=0.0,
                   help="background rate included in the data [ct/s]")
    t.add_argument("--max-noise-frac", type=float, default=0.3)
    t.add_argument("--n-poly", type=int, default=3,
                   help="polynomial degree of the time-evolution model; "
                        "0 = arbitrary time dependence")
    t.add_argument("--n-perm", type=int, default=2000)
    t.add_argument("--n-surr", type=int, default=99)
    t.add_argument("--no-surrogate", action="store_true")
    t.add_argument("--prior", type=float, default=0.5,
                   help="prior probability of stationarity")
    t.add_argument("--seed", type=int, default=0)

    o = p.add_argument_group("output")
    o.add_argument("--outdir", default="results")
    o.add_argument("--label", help="output file stem (default: input name)")
    o.add_argument("--ext", default="png")
    o.add_argument("--lc-bin", type=float, default=1.0,
                   help="bin size for the light-curve panel [s]")
    return p


def main():
    args = build_parser().parse_args()
    trange = None
    if args.tstart is not None or args.tstop is not None:
        trange = (args.tstart if args.tstart is not None else -np.inf,
                  args.tstop if args.tstop is not None else np.inf)

    lc = load_lightcurve(args.path, dt=args.dt, pi_range=args.pi,
                         energy_range=args.energy, time_range=trange,
                         kind=args.kind, columns=tuple(args.columns),
                         rmf_file=args.rmf)
    print(f"== {os.path.basename(args.path)} ==")
    print("  " + describe(lc))

    n_seg = n_segments(lc, args.seg)
    contiguous = is_contiguous(lc)
    print(f"  {n_seg} gap-free segments of {args.seg:g} s "
          f"({'contiguous' if contiguous else 'with gaps'})")
    if n_seg < args.blocks:
        raise SystemExit(f"only {n_seg} segments for {args.blocks} "
                         "blocks; lower --blocks or shorten --seg")

    rep = run_all(lc=lc, seg_len=args.seg, n_blocks=args.blocks, fmin=args.fmin,
                  fmax=args.fmax, fbin=args.fbin, norm=args.norm,
                  noise=args.noise, max_noise_frac=args.max_noise_frac,
                  bkg_rate=args.bkg_rate, noise_scale=args.noise_scale,
                  noise_level_value=args.noise_level, f_noise=args.f_noise,
                  n_poly=args.n_poly or None, n_perm=args.n_perm,
                  n_surr=args.n_surr, surrogate=not args.no_surrogate,
                  prior_stationary=args.prior, seed=args.seed)
    print(rep.summary())
    if not contiguous and not args.no_surrogate:
        print("  (surrogate test skipped: the light curve has gaps)")

    os.makedirs(args.outdir, exist_ok=True)
    stem = args.label or os.path.basename(args.path).split(".")[0]
    t0 = lc.time[0] - lc.dt / 2
    plot_report(rep, os.path.join(args.outdir, f"{stem}.{args.ext}"),
                lc.time - lc.dt / 2 - t0, np.asarray(lc.counts, float), lc.dt,
                title=f"{stem}: {describe(lc)}", lc_bin=args.lc_bin)

    def scalars(d):
        return {k: float(v) for k, v in d.items() if np.ndim(v) == 0}

    out = dict(file=os.path.abspath(args.path), dt=lc.dt,
               mean_rate=float(np.mean(lc.counts) / lc.dt),
               n_segments=int(n_seg), tstart=float(lc.time[0] - lc.dt / 2),
               contiguous=bool(contiguous), settings=vars(args),
               grid=dict(n_blocks=int(rep.grid.cells.shape[0]),
                         n_freq=int(rep.grid.freq.size),
                         n_bins_used=int(rep.psr["n_bins_used"]),
                         n_per_cell=int(rep.grid.n),
                         noise_model=rep.grid.noise_model,
                         noise_level=float(rep.grid.noise.mean())),
               psr=scalars(rep.psr), bayes=scalars(rep.bayes),
               surrogate=None if rep.surrogate is None else scalars(rep.surrogate),
               meta={k: str(lc.__dict__.get(k)) for k in
                     ("mission", "instr", "mjdref") if lc.__dict__.get(k)})
    with open(os.path.join(args.outdir, f"{stem}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  -> {os.path.join(args.outdir, stem)}.json")


if __name__ == "__main__":
    main()
