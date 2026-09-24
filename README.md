# stationarity_testing

Tests whether the power spectrum of a light curve is constant in time (second-order stationarity), with a simulator of QPOs whose properties change in time for calibrating the tests. The folder has no dependencies on other code in `QPOs_all`.

## Quick start

```bash
pip install -r requirements.txt

# simulated light curves
python nonstationary_qpo.py freq_drift          # QPO centroid drifts by ±10 %
python nonstationary_qpo.py all                 # every model, figures in figures/

# a light curve from a file
python test_lightcurve.py src.lc
```

Each run prints a summary of the tests and writes a report figure and a JSON file of the statistics.

## Files

| File | Contents |
|------|----------|
| `stationarity_tests.py` | The tests. `run_all` runs all of them. |
| `lightcurve_io.py` | Reads files into a stingray `Lightcurve` and cuts it into segments that respect the GTIs. |
| `test_lightcurve.py` | Command-line interface for light curves from files. |
| `nonstationary_qpo.py` | Simulator and command-line interface for simulated light curves. |
| `qpo_core.py` | Signal generation: time-varying damped oscillator, Ornstein-Uhlenbeck process, broadband noise. |
| `plotting.py` | The report figure. |
| `verify.py` | False-positive and detection rates from simulations. |
| `mc_fm_wander.py` | Monte Carlo of the frequency-wander model against the wander timescale. |
| `stingray_dynps_repro.py` | Reproduction of the `DynamicalPowerspectrum` normalisation issue (see [stingray](#stingray)). |
| `method_explanations/` | Mathematical write-ups of the three tests (LaTeX and PDF). |

## Method

The light curve is cut into segments (default 8 s) that do not cross gaps. Each segment's periodogram is normalised by that segment's own mean rate, which removes a linear rms-flux relation. Segments are grouped into K time blocks (default 16) and Fourier bins into J frequency bins (default 2 Fourier bins each, 0.25 to 8 Hz). Each cell of the resulting K × J grid is the mean of n = 16 periodogram ordinates. The null hypothesis is that the source power in every frequency bin is the same in all blocks.

The tests:

1. **PSR log-ANOVA** (Priestley & Subba Rao 1969). The log of each cell's power has a known variance, so the question becomes an analysis of variance. Five statistics are reported: the total, the uniform time effect and time × frequency interaction that make it up, a smooth-trend statistic (`--n-poly` orthogonal polynomials in time per frequency bin), the largest single-bin trend, and a window scan. The window scan compares each run of consecutive blocks (up to `--scan-width`, default half the blocks) with the rest, in each frequency bin, and reports the largest contrast with its time and frequency; it detects short changes that a low-degree polynomial misses. Permutation p-values shuffle the segments in time. See `method_explanations/psr_method.pdf`.
2. **Surrogate test** (Borgnat et al. 2010). The statistic is the variance over time of the distance between each window's multitaper spectrum and the mean spectrum. It is compared with surrogates made by randomising the Fourier phases of the light curve. Reports an index of non-stationarity (INS) and a p-value. See `method_explanations/surrogate_method.pdf`.
3. **Bayesian test**. H0: each frequency bin has a constant log power. H1: a fraction π of bins has log power that follows a polynomial in time, with coefficients drawn from N(0, τ²). τ (log-uniform, 0.05 to 2) and π (uniform) are marginalised on a grid. Outputs are P(H0 | data) for a chosen prior (`--prior`, default 0.5) and a Bayes factor for each frequency bin (`log10_BF_per_freq`), which compares "this bin varies" with "this bin is constant". See `method_explanations/bayes_method.pdf`.
4. **SBB bound** (Sellke, Bayarri & Berger 2001). Converts the smallest reported p-value into a lower bound on P(H0 | p): P(H0 | p) ≥ 1/(1 + 1/(−e p ln p)). It does not correct for testing several statistics.

Quote the permutation p-values. The χ² p-values assume Gaussian statistics and are too small for nonlinear variability and at low count rates (see [Calibration](#calibration)).

### Poisson noise

The null hypothesis concerns the source spectrum, so the Poisson level is subtracted by default (`--noise poisson`; `--noise none` turns this off).

Each segment's level comes from `stingray.fourier.poisson_level` at that segment's mean rate r: 2/r in fractional normalisation, 2r in absolute normalisation, and 2r/(r − b)² with a background rate b (`--bkg-rate`). The level therefore follows changes in count rate. It is averaged over the segments of each block and subtracted from each cell.

Subtraction changes the mean and variance of the log power. The tests use Y = ln(C − N) − m(N/S), with variance v(N/S), where m and v are the exact moments of this transform for a Gamma-distributed cell, tabulated numerically. For N = 0 they are ψ(n) − ln n and ψ′(n). Because the moments differ between blocks when the rate changes, every statistic is a weighted fit.

Y becomes skewed as the noise fraction grows. Only frequency bins with N/(S + N) ≤ `--max-noise-frac` (default 0.3), and a predicted fractional error per cell of at most 0.5, enter the tests. At 500 ct/s with default settings this leaves 5 of 31 bins (0.3 to 2 Hz). Fewer blocks (`--blocks`) or wider bins (`--fbin`) admit more bins. Cells below 1 % of the mean source power after subtraction are floored, and the summary reports how many.

The permutation test moves each segment's noise level with its periodogram. The surrogate test subtracts each window's own Poisson level and uses the same frequency bins as the other tests. In the report figure the dynamic power spectrum, z map and block spectra are all noise subtracted, and unused bins are blank.

## Real data

```bash
# event list, PI 50-1000, binned at 1/512 s
python test_lightcurve.py ni1234_cl.evt --dt 0.0019531 --pi 50 1000 --seg 8 --blocks 16

# FITS light curve with a background rate
python test_lightcurve.py src.lc --bkg-rate 12.4

# text file (time, rate), noise level measured above 40 Hz
python test_lightcurve.py lc.txt --dt 0.008 --noise highfreq --f-noise 40
```

Output goes to `results/<stem>.png` and `results/<stem>.json`.

**Reading.** Event lists are read with `EventList.read` (fmt='hea'), filtered on PI (`--pi`) or energy (`--energy` with `--rmf`), and binned with `EventList.to_lc` at `--dt`. Binned FITS light curves are read with astropy, which handles RATE and COUNTS columns and drops bins with FRACEXP < 0.99. Text and CSV files (`--columns`, `--kind`) and `.npy`/`.npz` files are also supported.

**Gaps.** GTIs are taken from the file or inferred from gaps in the time axis. Segments are cut with `stingray.gti.bin_intervals_from_gtis` and never span a gap, since a gap distorts the periodogram. Bins left over at the end of each good interval are dropped. With gaps, blocks cover unequal spans of time, and the figure places each block at the mean start time of its segments. The surrogate test needs uninterrupted data and is skipped when there are gaps.

**Background and dead time.** `--bkg-rate` renormalises the fractional rms to the source rate, so that a background fraction varying with count rate is not detected as non-stationarity. The Poisson level assumes no dead time. If that does not hold, `--noise highfreq --f-noise F` scales the level to match the power above F Hz, and `--noise-level` or `--noise-scale` set it directly.

### From Python

`run_all` takes a stingray `Lightcurve` and returns a report object:

```python
import numpy as np
from stingray import Lightcurve
from stationarity_tests import run_all
from lightcurve_io import load_lightcurve

lc = Lightcurve(time, counts, dt=dt, gti=np.array([[0.0, 300.0], [400.0, 1024.0]]))
# or: lc = load_lightcurve("src.lc")       # FITS, text/CSV or NumPy

rep = run_all(lc=lc, seg_len=8.0, n_blocks=16, fmin=0.25, fmax=8.0,
              n_perm=2000, surrogate=False)

print(rep.summary())
rep.p_values()                     # permutation p-values of each statistic
rep.bayes["P_stationary"]          # posterior probability of stationarity
rep.bayes["log10_BF_per_freq"]     # per-bin Bayes factor (NaN in unused bins)
rep.grid.freq[rep.grid.keep]       # frequency bins used by the tests
rep.psr["z"]                       # K x J map of log source-power deviations
```

A counts array also works: `run_all(counts, dt, seg_len=8.0, ...)`. `lightcurve_io.run_lightcurve` is equivalent to `run_all(lc=...)`.

Other options: `fbin`, `norm`, `noise`, `max_noise_frac`, `bkg_rate`, `noise_scale`, `noise_level_value`, `f_noise`, `n_poly`, `n_surr`, `prior_stationary` and `seed`. `surrogate=True` adds the surrogate test, which takes most of the run time.

The report figure is made with `plotting.plot_report(rep, "out.png", time, counts, dt)`. The individual tests (`build_tf_grid`, `psr_test`, `bayes_test`, `surrogate_test`) can also be called directly.

## Simulation

The QPO is a damped harmonic oscillator driven by white noise, with coefficients that can change in time:

```
q'' + 2 gamma(t) q' + omega(t)^2 q = 2 omega sqrt(gamma/dt) xi(t),   omega = 2 pi f0(t),  gamma = omega/(2 Q(t))
rate(t) = mean(t) * (1 + bb_rms * b(t) + rms(t) * q(t))               Poisson sampled
```

The drive amplitude keeps Var[q] = 1 at all times, so changing `f0` or `Q` does not change the QPO amplitude, which is set by `rms(t)` alone. `b(t)` is a broadband realisation of two Lorentzians (Timmer & Koenig method). Each of `f0`, `Q`, `rms` and `mean` can evolve as `p0 (1 + delta g(t))`, where `g` is a `linear`, `step`, `sine` or `burst` profile.

| Model | Truth | Settings |
|-------|-------|----------|
| `stationary` | H0 | f0 = 2 Hz, Q = 8, QPO rms 10 %, broadband rms 20 %, 1e4 ct/s, 1024 s, dt = 1/512 s |
| `fm_wander` | H0 | f0 follows an OU process (σ = 0.15 Hz, τ = 5 s) |
| `lognormal` | H0 | exponentiated process with a linear rms-flux relation |
| `low_rate` | H0 | 500 ct/s; the Poisson level dominates most bins |
| `rate_drift` | H0 (source) | mean rate changes by ±60 % at fixed fractional variability; the Poisson level changes, the source spectrum does not |
| `freq_drift` | H1 | f0 changes by ±10 % |
| `rms_drift` | H1 | QPO rms changes by ±50 % |
| `q_drift` | H1 | Q changes by ±80 % |

Further examples:

```bash
python nonstationary_qpo.py freq_drift --delta 0.03 --profile step
python nonstationary_qpo.py rms_drift --profile sine --period 256 --n-poly 6
python nonstationary_qpo.py rate_drift --mean 1000 --noise none    # without noise subtraction
python nonstationary_qpo.py low_rate --max-noise-frac 0.5
python verify.py --nsim 40 --workers 4
```

Output goes to `figures/<model>[_<profile>].png` and a matching `.json`.

### Calibration

From `verify.py --nsim 100` with 200 permutations per simulation. Entries are the fraction of simulations with p < 0.05, given as χ² / permutation.

| Case | Total | Trend | Max-bin (perm.) | Median P(stationary) |
|------|-------|-------|-----------------|----------------------|
| stationary, 1e4 ct/s | 3 / 4 % | 4 / 4 % | 8 % | 0.71 |
| lognormal | 4 / 4 % | 7 / 5 % | 2 % | 0.73 |
| low_rate, 500 ct/s | 9 / 4 % | 6 / 4 % | 7 % | 0.63 |
| rate_drift, 1000 ct/s | 7 / 6 % | 8 / 6 % | 6 % | 0.67 |
| rate_drift, no noise subtraction | 100 / 100 % | 100 / 100 % | 99 % | 2e-22 |
| freq_drift ±10 % | 80 / 66 % | 100 / 99 % | 99 % | 2e-6 |
| rms_drift ±60 %, 1e4 ct/s | 100 / 97 % | 100 / 100 % | 100 % | 1e-18 |
| rms_drift ±60 %, 1000 ct/s | 100 / 100 % | 100 / 100 % | 100 % | 1e-16 |

Without noise subtraction a change in count rate alone is always detected, because the Poisson level 2/r changes with the rate. At 500 ct/s the χ² p-value of the total statistic rejects 9 % of stationary simulations, because the noise-subtracted log power is skewed; the permutation value stays at 4 %.

### Frequency wander and timescale

`fm_wander` is stationary, but its segment spectra differ from one another while the wander timescale τ is not short compared with a segment. `mc_fm_wander.py` measures the rejection rate of each statistic against τ and writes a CSV and a plot to `mc_results/`:

```bash
python mc_fm_wander.py --taus 1 5 20 64 256 --nsim 100 --workers 4
python mc_fm_wander.py --taus 1 5 20 64 256 --nsim 100 --workers 4 --surrogate   # slower
```

Rejection rates near 5 % for τ well below the segment length show that the tests are calibrated for this nonlinear, stationary process. Rates rise once τ approaches the block length (64 s by default), where the process is stationary only on timescales longer than the observation.

## Caveats

- **Timescale.** The segment length sets the timescale below which variability counts as stationary. The permutation test assumes segments are exchangeable, so a modulation with a correlation time comparable to a segment or longer is detected as non-stationarity. Choose `--seg` and `--blocks` for the timescale you want to test.
- **Red noise.** Power below 1/(segment length) correlates neighbouring segments, which affects both the χ² and permutation p-values. Check with `stationary` simulations using your own power spectrum.
- **Noise level.** The subtraction assumes pure Poisson noise. Dead time, pile-up and background subtraction change the level; measure it where the source is negligible and pass it with `--noise-level`, `--noise-scale` or `--noise highfreq`.
- **Low count rates.** Excluding noise-dominated bins costs sensitivity. The selection uses the time-averaged spectrum, so a feature that is strong for only part of the observation can be excluded.
- **Bayesian prior.** P(stationary) depends on the H1 prior. Alternatives with very small τ are almost identical to H0, so the lower limit of τ caps P(H0): stationary data give about 0.7. The model treats frequency bins as independent and approximates the log power as Gaussian, which is accurate for n ≳ 8 at low noise fractions.
- **Reproducibility.** stingray 2.x's `Simulator.simulate` ignores `random_state`, so `qpo_core.build_broadband` seeds the global NumPy state itself.
- **Parallel runs.** `verify.py` and `mc_fm_wander.py` use one BLAS thread per worker, since threaded BLAS in forked workers can deadlock.

## stingray

stingray provides the reading (`EventList.read`, `EventList.to_lc`, `Lightcurve`), GTI handling and segmentation (`bin_intervals_from_gtis`, `apply_gtis`, `truncate`), the per-segment periodograms (`AveragedPowerspectrum` with `use_common_mean=False, save_all=True`), the Poisson level (`fourier.poisson_level`), log-rebinning for the figure (`rebin_log`), and the broadband simulation (`stingray.simulator`). The multitaper spectrogram and the test statistics are implemented here.

`DynamicalPowerspectrum` would provide the time-frequency grid directly, but it normalises every segment by the mean rate of the whole observation, which reintroduces the rms-flux trend. It does not accept `use_common_mean`. `stingray_dynps_repro.py` demonstrates this. If the option is added, `segment_spectra` in `stationarity_tests.py` can use `DynamicalPowerspectrum`.

## References

- Borgnat, Flandrin, Honeine, Richard & Xiao 2010, IEEE TSP 58, 3459. Testing stationarity with surrogates: a time-frequency approach.
- Dwivedi & Subba Rao 2011, JTSA 32, 68; Jentsch & Subba Rao 2015, J. Econometrics 185, 124.
- Hübner et al. 2022, ApJS 259, 32. Pitfalls of periodograms: the non-stationarity bias in QPO analysis.
- Nason 2013, JRSS B 75, 879; von Sachs & Neumann 2000, JASA 95, 597. Wavelet-domain tests (R package `locits`).
- Paparoditis 2010, JASA 105, 839. Rolling local periodograms.
- Priestley & Subba Rao 1969, JRSS B 31, 140. Test for non-stationarity of time series.
- Rosen, Wood & Stoffer 2012, JASA 107, 1575. AdaptSPEC: Bayesian piecewise-stationary spectra.
- Sellke, Bayarri & Berger 2001, Am. Stat. 55, 62. Calibration of p-values.
- Vaughan, Edelson, Warwick & Uttley 2003, MNRAS 345, 1271. Tests for non-stationarity in X-ray light curves.
- van der Klis 1989, in Timing Neutron Stars (NATO ASI C262), 27. Poisson level and statistics of X-ray power spectra.
