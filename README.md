# stationarity_testing

Test whether a light curve's power spectrum is constant in time (second-order stationarity), and simulate QPOs whose properties change with time to calibrate those tests. The folder is self-contained.

## Files

| File | Contents |
|------|----------|
| `stationarity_tests.py` | The tests. Work on a counts array or on pre-cut segments. |
| `test_lightcurve.py` | Run the tests on a light curve from a file (CLI). |
| `lightcurve_io.py` | Reads files into a stingray `Lightcurve`, cuts it into GTI-respecting segments, and runs the tests on it. |
| `nonstationary_qpo.py` | Simulator and CLI. Runs the tests on each simulation. |
| `qpo_core.py` | Signal-generation utilities: `damped_oscillator_convolve`, `ou_process`, `build_broadband`, `dho_filter`, `lorentzian`, `make_spectra`, `apply_filter`. |
| `plotting.py` | The report figure, shared by the simulator and the data script. |
| `stingray_dynps_norm.md` | Suggested stingray change: per-segment normalisation in `DynamicalPowerspectrum`. |
| `verify.py` | Simulator checks, false-positive rates under H0, detection rates under H1. |
| `mc_fm_wander.py` | Monte Carlo of the frequency-wander model against the wander timescale tau. |

## Real data

`test_lightcurve.py` reads a light curve, cuts it into gap-free segments and runs the tests:

```bash
# event list, PI 50-1000, binned at 1/512 s
python test_lightcurve.py ni1234_cl.evt --dt 0.0019531 --pi 50 1000 --seg 8 --blocks 16

# FITS light curve, with a background rate that would otherwise dilute the rms
python test_lightcurve.py src.lc --bkg-rate 12.4

# text file (time, rate), noise level taken from high frequencies
python test_lightcurve.py lc.txt --dt 0.008 --noise highfreq --f-noise 40
```

It prints a summary and writes `results/<stem>.png` (the report figure) and `results/<stem>.json`.

Input is turned into a stingray `Lightcurve`. Event lists go through `EventList.read` (fmt='hea'), are filtered on PI (`--pi`) or energy (`--energy`, with `--rmf`), and are binned with `EventList.to_lc` at `--dt`. Binned FITS products are read with astropy, which keeps the RATE/COUNTS and FRACEXP handling explicit (bins with FRACEXP < 0.99 are dropped). Text/CSV (`--columns`, `--kind`) and `.npy`/`.npz` have small readers.

GTIs come from the file, or are inferred from gaps in the time axis, and bins outside them are dropped. Segmentation uses `stingray.gti.bin_intervals_from_gtis`, so segments never span a gap or a GTI boundary, because a segment containing a gap has a distorted periodogram; leftover bins at the end of each stretch are dropped. Blocks are groups of consecutive segments, so with gaps they cover unequal spans of clock time, and the figure uses each block's mean segment start time. The surrogate test needs one uninterrupted stretch and is skipped when the light curve has gaps.

`--bkg-rate` renormalises the fractional rms to the source rate, so background dilution that follows the count rate is not read as non-stationarity. The Poisson level assumes no dead time; `--noise highfreq` with `--f-noise` above the source band scales the level to the measured high-frequency power, and `--noise-level` or `--noise-scale` set it directly.

### Calling the tests from Python

`run_all` (or `run_lightcurve`, a thin alias in `lightcurve_io`) takes a stingray `Lightcurve` with its GTIs and returns a report object:

```python
import numpy as np
from stingray import Lightcurve
from stationarity_tests import run_all
from lightcurve_io import load_lightcurve

lc = Lightcurve(time, counts, dt=dt, gti=np.array([[0.0, 300.0], [400.0, 1024.0]]))
# or: lc = load_lightcurve("src.lc")       # FITS, text/CSV or NumPy -> Lightcurve

rep = run_all(lc=lc, seg_len=8.0, n_blocks=16, fmin=0.25, fmax=8.0,
              n_perm=2000, surrogate=False)

print(rep.summary())
rep.p_values()                    # permutation p-values of each statistic
rep.bayes["P_stationary"]         # posterior probability of stationarity
rep.grid.freq[rep.grid.keep]      # frequency bins that passed the noise cut
rep.psr["z"]                      # K x J map of log source-power deviations
```

Gaps are handled by the segmentation, so a light curve with several GTIs needs nothing extra; only the surrogate test is skipped, since it needs uninterrupted data.

A bare counts array works too, and is wrapped in a `Lightcurve` internally:

```python
rep = run_all(counts, dt, seg_len=8.0, n_blocks=16, n_perm=2000, surrogate=False)
```

Other options: `fbin`, `norm`, `noise`, `max_noise_frac`, `bkg_rate`, `noise_scale`, `noise_level_value`, `f_noise`, `n_poly`, `n_surr`, `prior_stationary` and `seed`. `surrogate=True` adds the surrogate test, which needs contiguous data and dominates the runtime.

The report figure is one more call: `plotting.plot_report(rep, "out.png", time, counts, dt)`. The individual tests (`build_tf_grid`, `psr_test`, `bayes_test`, `surrogate_test`) can be called directly too; `run_all` only chains them.

## Tests

Light curves are cut into segments (default 8 s) with `stingray.gti.bin_intervals_from_gtis`, and the per-segment periodograms come from `AveragedPowerspectrum(..., use_common_mean=False, save_all=True)`, so each segment is normalised by its own mean, which removes a linear rms-flux relation. Segments are grouped into K time blocks (default 16) and Fourier bins into J frequency bins (default 2 raw bins each, 0.25 to 8 Hz). Each cell is then an average of n = 16 periodogram ordinates.

1. PSR log-ANOVA (Priestley & Subba Rao 1969). Under H0 the log cell power has known variance, so the sums of squares are chi². The script reports the uniform-time and time × frequency effects, a trend statistic that projects each frequency bin's log power onto `--n-poly` orthogonal polynomials in time, and the largest single-bin trend (a scan statistic). Permutation p-values shuffle the segments in time and do not rely on the chi² approximation.
2. Surrogate test (Borgnat et al. 2010). Multitaper spectrogram, variance over time of a KL + log-spectral distance to the mean spectrum, compared with phase-randomised surrogates. Reports the index of non-stationarity (INS) and a gamma-fit p-value.
3. Bayesian P(stationary). H0: constant log source spectrum. H1: a random fraction pi of frequency bins has log power evolving as a polynomial in time, with coefficient prior N(0, tau²). The Bayes factor is analytic for each (tau, pi); tau (log-uniform, 0.05 to 2) and pi (uniform) are marginalised on a grid. The output is P(H0 | data) for the chosen prior (`--prior`, default 0.5), along with the posterior probability that each frequency bin varies.
4. SBB bound (Sellke, Bayarri & Berger 2001). For the smallest reported p-value, P(H0 | p) ≥ 1/(1 + 1/(−e p ln p)): the least favourable posterior probability over a broad class of alternatives. It does not correct for testing several statistics.

### Poisson noise

The null hypothesis concerns the source spectrum, so the counting noise is removed (`--noise poisson`, the default; `--noise none` switches it off).

Each segment's Poisson level comes from `stingray.fourier.poisson_level` with that segment's own mean rate r: N = 2/r in frac normalisation, 2r in abs normalisation, and 2r/(r − bkg)² when `--bkg-rate` is set. N therefore follows a changing count rate, and is tracked per segment and averaged per block. The source estimate for each cell is C − N.

Subtracting N changes the mean and variance of the log power, so the tests use Y = ln(C − N) − m(N/S) with variance v(N/S), where m and v are the exact moments of that transform for a Gamma(n) cell, tabulated numerically. For N = 0 they reduce to ψ(n) − ln n and ψ′(n). The moments differ between blocks when the rate changes, so every statistic is a weighted fit.

Y becomes skewed as the noise fraction grows, so only bins where N/(S + N) ≤ `--max-noise-frac` (default 0.3), and where a noise-subtracted cell has a predicted fractional error ≤ 0.5, enter the tests. At 500 ct/s with the default settings this leaves 5 of 31 bins (0.3 to 2 Hz); fewer blocks (`--blocks`) or wider bins (`--fbin`) bring more bins in. Cells that fall below 1 % of the mean source power after subtraction are floored, and the summary reports how many.

Permutation p-values shuffle segments together with their noise levels. The surrogate test subtracts each window's Poisson level and uses the same frequency bins. In the figures, the dynamic PSD, the z map and the block spectra are all noise subtracted, and unused bins are blank.

## Simulation

The QPO is a white-noise-driven damped oscillator integrated with a time-varying state-space step:

```
q'' + 2 gamma(t) q' + omega(t)^2 q = 2 omega sqrt(gamma/dt) xi(t),   omega = 2 pi f0(t),  gamma = omega/(2 Q(t))
rate(t) = mean(t) * (1 + bb_rms * b(t) + rms(t) * q(t))               Poisson sampled
```

The drive scaling keeps Var[q] = 1 at every instant, so drifts in `f0` or `Q` do not also change the QPO amplitude; amplitude changes come only from `rms(t)`. `b(t)` is a two-Lorentzian Timmer-Koenig broadband realisation. Each of `f0`, `Q`, `rms` and `mean` evolves as `p0 (1 + delta g(t))`, with `g` a `linear`, `step`, `sine` or `burst` profile.

| Model | Truth | Default |
|-------|-------|---------|
| `stationary` | H0 | f0 = 2 Hz, Q = 8, rms = 10 %, broadband 20 %, 1e4 ct/s, 1024 s at dt = 1/512 |
| `fm_wander` | H0 | f0 wanders as an OU process (sigma 0.15 Hz, tau 5 s) |
| `lognormal` | H0 | exponentiated process with a linear rms-flux relation |
| `low_rate` | H0 | 500 ct/s, so the Poisson level dominates most bins |
| `rate_drift` | H0 for the source | mean rate changes by ±60 % at fixed fractional variability, so the Poisson level 2/rate changes and the source spectrum does not |
| `freq_drift` | H1 | f0 changes by ±10 % |
| `rms_drift` | H1 | QPO rms changes by ±50 % |
| `q_drift` | H1 | Q changes by ±80 % |

```bash
pip install -r requirements.txt
python nonstationary_qpo.py all                                   # every model -> figures/
python nonstationary_qpo.py freq_drift --delta 0.03 --profile step
python nonstationary_qpo.py rms_drift --profile sine --period 256 --n-poly 6
python nonstationary_qpo.py rate_drift --mean 1000 --noise none    # effect of skipping subtraction
python nonstationary_qpo.py low_rate --max-noise-frac 0.5
python verify.py --nsim 40 --workers 4
```

Each run prints a summary and writes `figures/<model>[_<profile>].png` and a `.json` with the statistics.

### Calibration

From `verify.py --nsim 100`, with 200 permutations per simulation. Each entry is the fraction of simulations with p < 0.05 (chi² / permutation).

| Case | Total | Trend | Max-bin (perm) | Median P(stationary) |
|------|-------|-------|----------------|----------------------|
| stationary, 1e4 ct/s | 3 / 4 % | 4 / 4 % | 8 % | 0.71 |
| lognormal | 4 / 4 % | 7 / 5 % | 2 % | 0.73 |
| low_rate, 500 ct/s | 9 / 4 % | 6 / 4 % | 7 % | 0.63 |
| rate_drift, 1000 ct/s | 7 / 6 % | 8 / 6 % | 6 % | 0.67 |
| rate_drift, no noise subtraction | 100 / 100 % | 100 / 100 % | 99 % | 2e-22 |
| freq_drift ±10 % | 80 / 66 % | 100 / 99 % | 99 % | 2e-6 |
| rms_drift ±60 %, 1e4 ct/s | 100 / 97 % | 100 / 100 % | 100 % | 1e-18 |
| rms_drift ±60 %, 1000 ct/s | 100 / 100 % | 100 / 100 % | 100 % | 1e-16 |

Without noise subtraction a rate change alone is always detected, because the Poisson level 2/r moves with the rate. The analytic chi² for the total statistic runs high at low rates (9 %), since the noise-subtracted log power is skewed; the summary quotes the permutation p-values, and so should any reported result.

### Frequency wander and timescale

`fm_wander` is a stationary process, but the tests treat it as stationary only while tau is much shorter than the block length (64 s by default). `mc_fm_wander.py` measures the rejection rate of every statistic against tau / block length, and writes a CSV of all simulations and a plot to `mc_results/`:

```bash
python mc_fm_wander.py --taus 1 5 20 64 256 --nsim 100 --workers 4
python mc_fm_wander.py --taus 1 5 20 64 256 --nsim 100 --workers 4 --surrogate   # slower
```

Rates near 5 % for tau well below the segment length mean the tests are calibrated for this nonlinear but stationary process. Rising rates for tau of order a block length or longer are expected, because the process is then stationary only on timescales longer than the observation.

### What comes from stingray

Reading (`EventList.read`, `EventList.to_lc`, `Lightcurve`), GTI handling and segmentation (`bin_intervals_from_gtis`, `apply_gtis`, `truncate`), the per-segment periodograms (`AveragedPowerspectrum` with `use_common_mean=False, save_all=True`), the Poisson level (`stingray.fourier.poisson_level`, including its `backrate`), log-rebinning for the figure (`rebin_log`) and the simulator's broadband realisation (`stingray.simulator`).

What is not: the multitaper spectrogram of the surrogate test, and the statistics themselves. `DynamicalPowerspectrum` would cover the time-frequency grid, but it always normalises by the mean common to all segments, which reintroduces the rms-flux trend the tests must remove, and it exposes only one mean rate rather than one per segment. `stingray_dynps_norm.md` proposes adding `use_common_mean` (and a per-segment `meanrate`) to that class; with it, `segment_spectra` in `stationarity_tests.py` could call `DynamicalPowerspectrum` directly.

## Caveats

- Stationarity depends on the timescale. The segment length sets the timescale on which the tests treat variability as stationary. The permutation null assumes segments are exchangeable, so any modulation with a correlation time comparable to or longer than a segment counts as non-stationarity, and so do real differences between blocks. Choose `--seg` (and `--blocks`) for the timescale you want to test.
- Red noise below 1/segment correlates adjacent segments, which breaks the independence behind both the chi² and the permutation p-values. Check with `stationary` simulations that use your own PSD.
- The subtraction assumes pure Poisson noise. Dead time, pile-up or background subtraction change the noise level; measure it from frequencies where the source is negligible and pass it with `--noise-level`, `--noise-scale` or `--noise highfreq`.
- Excluding noise-dominated bins costs sensitivity at low count rates. The selection uses the time-averaged spectrum, so a feature that is strong only for part of the observation can fall below the cut.
- P(stationary) depends on the H1 prior. The smallest tau allowed caps how far P(H0) can rise, because H1 with a very small tau is almost indistinguishable from H0; stationary data here give P(H0) of about 0.7 to 0.8. The model treats frequency bins as independent and uses a Gaussian approximation for the log power, accurate for n of about 8 or more at low noise fractions.
- stingray 2.x's `Simulator.simulate` draws from the global `np.random` state and ignores `random_state`, so `qpo_core.build_broadband` seeds the global state itself to stay reproducible.
- `verify.py` and `mc_fm_wander.py` set one BLAS thread per worker, because threaded BLAS in forked workers can deadlock.

## References

- van der Klis 1989, in Timing Neutron Stars (NATO ASI C262), 27. Poisson level and statistics of X-ray power spectra.
- Priestley & Subba Rao 1969, JRSS B 31, 140. Test for non-stationarity of time series.
- Borgnat, Flandrin, Honeine, Richard & Xiao 2010, IEEE TSP 58, 3459. Testing stationarity with surrogates: a time-frequency approach.
- von Sachs & Neumann 2000, JASA 95, 597; Nason 2013, JRSS B 75, 879. Wavelet-domain tests (R package `locits`).
- Paparoditis 2010, JASA 105, 839 (rolling local periodograms); Dwivedi & Subba Rao 2011, JTSA 32, 68; Jentsch & Subba Rao 2015, J. Econometrics 185, 124.
- Rosen, Wood & Stoffer 2012, JASA 107, 1575. AdaptSPEC: Bayesian piecewise-stationary spectra.
- Vaughan, Edelson, Warwick & Uttley 2003, MNRAS 345, 1271. Tests for non-stationarity in X-ray light curves.
- Hübner et al. 2022, ApJS 259, 32. Pitfalls of periodograms: the non-stationarity bias in QPO analysis.
- Sellke, Bayarri & Berger 2001, Am. Stat. 55, 62. Calibration of p-values.
