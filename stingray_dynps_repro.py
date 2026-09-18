#!/usr/bin/env python3
"""
stingray_dynps_repro.py
-----------------------
Minimal reproduction for the `DynamicalPowerspectrum` normalisation issue
(stingray 2.3.2).

`AveragedPowerspectrum` has `use_common_mean`, which decides whether the
normalisation uses the mean count rate of the whole observation or of each
segment. `DynamicalPowerspectrum` builds its matrix from that class but has no
such argument, so every column is normalised by the common mean.

Test signal: a sinusoid at 2 Hz with a fixed fractional rms of 10 %, Poisson
sampled, whose mean count rate doubles halfway through the observation. The
fractional variability does not change, so with norm='frac' the columns of the
dynamical power spectrum should not change either.

Run: python stingray_dynps_repro.py
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from stingray import Lightcurve                                  # noqa: E402
from stingray.powerspectrum import (AveragedPowerspectrum,       # noqa: E402
                                    DynamicalPowerspectrum)
import stingray                                                  # noqa: E402


def make_lightcurve(seed=0, dt=1 / 64, T=512.0, f0=2.0, frac_rms=0.10,
                    rate_lo=200.0, rate_hi=400.0):
    """Constant fractional rms, mean count rate doubling at T/2."""
    rng = np.random.default_rng(seed)
    n = int(T / dt)
    t = (np.arange(n) + 0.5) * dt
    mean_rate = np.where(t < T / 2, rate_lo, rate_hi)
    rate = mean_rate * (1 + frac_rms * np.sqrt(2) * np.sin(2 * np.pi * f0 * t))
    counts = rng.poisson(rate * dt).astype(float)
    return Lightcurve(t, counts, dt=dt, skip_checks=True,
                      gti=np.array([[0.0, T]]))


def main():
    print(f"stingray {stingray.__version__}")
    lc = make_lightcurve()
    seg = 8.0

    # 1. the argument does not exist on the dynamical class
    try:
        DynamicalPowerspectrum(lc, segment_size=seg, norm="frac",
                               use_common_mean=False)
        print("1. DynamicalPowerspectrum accepted use_common_mean")
    except TypeError as exc:
        print(f"1. DynamicalPowerspectrum(use_common_mean=False) -> TypeError: {exc}")

    # 2. what the dynamical class gives, versus per-segment normalisation
    dyn = DynamicalPowerspectrum(lc, segment_size=seg, norm="frac")
    avg = AveragedPowerspectrum(lc, segment_size=seg, norm="frac",
                                use_common_mean=False, save_all=True,
                                silent=True)
    per_seg = np.array(avg.cs_all).T            # same shape as dyn.dyn_ps

    j = np.argmin(np.abs(dyn.freq - 2.0))       # the 2 Hz bin
    first = dyn.time < 256.0
    for name, mat in (("common mean (dynamical class)", dyn.dyn_ps),
                      ("per-segment mean (workaround)", per_seg)):
        lo = mat[j, first].mean()
        hi = mat[j, ~first].mean()
        print(f"2. {name:32s} power at 2 Hz: "
              f"first half {lo:.4f}, second half {hi:.4f}, ratio {hi / lo:.2f}")

    print("   rate ratio between halves: "
          f"{lc.counts[lc.time >= 256].mean() / lc.counts[lc.time < 256].mean():.2f}")
    print("   expected ratio for constant fractional rms: 1.00")

    # 3. the Poisson level, which follows each segment's own rate
    from stingray.fourier import poisson_level
    for r in (200.0, 400.0):
        print(f"3. poisson_level(frac, meanrate={r:.0f}) = "
              f"{poisson_level(norm='frac', meanrate=r):.5f}")
    print("   one level cannot describe both halves of a common-mean "
          "dynamical spectrum")


if __name__ == "__main__":
    main()
