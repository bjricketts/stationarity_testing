"""
lightcurve_io.py
----------------
Get a stingray `Lightcurve` from a file, cut it into GTI-respecting segments,
and run the stationarity tests on it.

Event lists are read with stingray (`EventList.read`, fmt='hea'), filtered on
PI or energy and binned with `EventList.to_lc`. Binned FITS products are read
with astropy, which keeps the RATE/COUNTS and FRACEXP handling explicit, and
text/CSV and NumPy inputs have small readers. Everything returns a
`Lightcurve`, so GTIs, truncation and splitting are stingray's.

Segments never span a gap or a GTI boundary: segmentation goes through
`stingray.gti.bin_intervals_from_gtis`, because a segment containing a gap has
a distorted periodogram.

Background and dead time are handled in `stationarity_tests.build_tf_grid`
(`bkg_rate`, `noise`, `noise_level`, `noise_scale`).
"""
from __future__ import annotations

import os

import numpy as np
from stingray import EventList, Lightcurve
from stingray.gti import bin_intervals_from_gtis, gti_border_bins  # noqa: F401

from stationarity_tests import run_all


# =========================================================================
# GTIs and segmentation
# =========================================================================

def gti_from_times(time, dt, tol=0.5):
    """GTIs implied by the gaps in an evenly binned time axis (bin starts).

    stingray's `create_gti_from_condition` needs a per-bin condition rather
    than the gap structure, so the intervals are built here.
    """
    time = np.asarray(time, dtype=float)
    brk = np.flatnonzero(np.diff(time) > dt * (1 + tol))
    starts = np.concatenate([[0], brk + 1])
    stops = np.concatenate([brk, [time.size - 1]])
    return np.column_stack([time[starts], time[stops] + dt])


def segment_bounds(lc: Lightcurve, seg_len):
    """Start/stop bin indices of each gap-free segment (stingray's GTI logic).

    Wraps `stingray.gti.bin_intervals_from_gtis`, so no segment spans a gap or
    a GTI boundary; leftover bins at the end of each GTI are dropped.
    """
    return bin_intervals_from_gtis(lc.gti, seg_len, lc.time, dt=lc.dt)


def segment_data(lc: Lightcurve, seg_len):
    """Gap-free segments of a Lightcurve: (counts array, start times)."""
    start, stop = segment_bounds(lc, seg_len)
    if not len(start):
        raise ValueError(f"no gap-free stretch of {seg_len:g} s in the data")
    counts = np.asarray(lc.counts, dtype=float)
    segs = np.array([counts[a:b] for a, b in zip(start, stop)])
    return segs, np.asarray(lc.time)[start] - lc.dt / 2


def n_segments(lc: Lightcurve, seg_len) -> int:
    """How many gap-free segments of `seg_len` the light curve holds."""
    return len(segment_bounds(lc, seg_len)[0])


def is_contiguous(lc: Lightcurve) -> bool:
    """True if the light curve has a single GTI."""
    return len(lc.gti) == 1


def describe(lc: Lightcurve) -> str:
    exp = float(np.sum(np.diff(lc.gti, axis=1))) if len(lc.gti) else lc.n * lc.dt
    span = lc.time[-1] - lc.time[0] + lc.dt
    txt = (f"{lc.n} bins of {lc.dt:g} s ({exp:.0f} s good time over "
           f"{span:.0f} s), mean rate {np.mean(lc.counts) / lc.dt:.1f} ct/s, "
           f"{len(lc.gti)} GTI(s)")
    inst = " ".join(str(lc.__dict__.get(k, "")) for k in ("mission", "instr")
                    if lc.__dict__.get(k))
    return txt + (f", {inst}" if inst.strip() else "")


# =========================================================================
# Readers
# =========================================================================

def _events_to_lc(ev: EventList, dt, pi_range=None, energy_range=None):
    mask = np.ones(ev.time.size, bool)
    if pi_range is not None:
        if getattr(ev, "pi", None) is None:
            raise ValueError("no PI column to filter on")
        mask &= (ev.pi >= pi_range[0]) & (ev.pi <= pi_range[1])
    if energy_range is not None:
        if getattr(ev, "energy", None) is None:
            raise ValueError("no ENERGY column to filter on (an RMF is needed "
                             "to convert PI to energy)")
        mask &= (ev.energy >= energy_range[0]) & (ev.energy <= energy_range[1])
    if not mask.all():
        ev = ev.apply_mask(mask)
    return ev.to_lc(dt)


def _read_fits_fallback(path, dt=None, pi_range=None, energy_range=None):
    """Read a FITS event list or light curve with astropy.

    Used when stingray's readers reject the file, which happens for products
    that are missing the OGIP timing keywords (TSTART, MJDREF, ...).
    """
    from astropy.io import fits

    def find(hdul, names, cols=()):
        for hdu in hdul:
            if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            have = [c.upper() for c in hdu.columns.names]
            if (any(n in (hdu.name or "").upper() for n in names)
                    and all(c in have for c in cols)):
                return hdu
        return None

    with fits.open(path, memmap=True) as hdul:
        gti_hdu = find(hdul, ("GTI",))
        gti = np.empty((0, 2))
        if gti_hdu is not None:
            gc = [c.upper() for c in gti_hdu.columns.names]
            a = "START" if "START" in gc else gc[0]
            b = "STOP" if "STOP" in gc else gc[1]
            gti = np.column_stack([np.asarray(gti_hdu.data[a], float),
                                   np.asarray(gti_hdu.data[b], float)])
        ev = find(hdul, ("EVENTS",), ("TIME",))
        if ev is not None:
            if dt is None:
                raise ValueError("pass dt to bin an event list")
            cols = {c.upper(): c for c in ev.columns.names}
            t = np.asarray(ev.data[cols["TIME"]], float)
            el = EventList(t, gti=gti if len(gti) else None)
            for attr, names in (("pi", ("PI", "PHA")), ("energy", ("ENERGY",))):
                col = next((cols[n] for n in names if n in cols), None)
                if col is not None:
                    setattr(el, attr, np.asarray(ev.data[col], float))
            if not len(gti):
                el.gti = np.array([[t.min(), t.max()]])
            return _events_to_lc(el, dt, pi_range, energy_range)

        hdu = find(hdul, ("RATE", "LIGHTCURVE", "LC"), ("TIME",))
        if hdu is None:
            raise ValueError(f"no EVENTS or RATE extension in {path}")
        cols = {c.upper(): c for c in hdu.columns.names}
        hdr = hdu.header
        t = np.asarray(hdu.data[cols["TIME"]], float)
        bin_dt = float(dt or hdr.get("TIMEDEL") or np.median(np.diff(t)))
        if "RATE" in cols:
            y = np.asarray(hdu.data[cols["RATE"]], float) * bin_dt
        elif "COUNTS" in cols:
            y = np.asarray(hdu.data[cols["COUNTS"]], float)
        else:
            raise ValueError("no RATE or COUNTS column")
        good = np.isfinite(y)
        if "FRACEXP" in cols:      # partial bins have the wrong noise level
            good &= np.asarray(hdu.data[cols["FRACEXP"]], float) > 0.99
        t, y = t[good], y[good]
        if not len(gti):
            gti = gti_from_times(t - bin_dt / 2, bin_dt)
    return Lightcurve(t, y, dt=bin_dt, gti=gti, skip_checks=True)


def read_text(path, dt=None, columns=(0, 1), kind="auto") -> Lightcurve:
    """Two- or three-column text/CSV light curve (time, rate|counts)."""
    delim = "," if path.lower().endswith(".csv") else None
    arr = np.atleast_2d(np.genfromtxt(path, delimiter=delim, comments="#"))
    if arr.shape[1] < 2:
        raise ValueError("need at least two columns (time, rate or counts)")
    t = arr[:, columns[0]].astype(float)
    y = arr[:, columns[1]].astype(float)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    bin_dt = float(dt or np.median(np.diff(t)))
    if kind == "auto":
        kind = "counts" if np.allclose(y, np.round(y)) and y.max() > 1 else "rate"
    counts = y if kind == "counts" else y * bin_dt
    return Lightcurve(t, counts, dt=bin_dt, skip_checks=True,
                      gti=gti_from_times(t, bin_dt))


def read_numpy(path, dt=None) -> Lightcurve:
    """.npy (counts array) or .npz with time/counts|rate/dt/gti keys."""
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.ndarray):
        if dt is None:
            raise ValueError("pass dt for a plain .npy counts array")
        c = obj.astype(float)
        t = (np.arange(c.size) + 0.5) * dt
        return Lightcurve(t, c, dt=dt, skip_checks=True,
                          gti=np.array([[0.0, c.size * dt]]))
    d = dict(obj)
    bin_dt = float(dt or d.get("dt", np.nan))
    t = d["time"].astype(float) if "time" in d else None
    if np.isnan(bin_dt):
        if t is None or t.size < 2:
            raise ValueError("cannot determine dt")
        bin_dt = float(np.median(np.diff(t)))
    if "counts" in d:
        c = d["counts"].astype(float)
    elif "rate" in d:
        c = d["rate"].astype(float) * bin_dt
    else:
        raise ValueError("npz needs a counts or rate array")
    if t is None:
        t = (np.arange(c.size) + 0.5) * bin_dt
    gti = d["gti"].astype(float) if "gti" in d else gti_from_times(t, bin_dt)
    return Lightcurve(t, c, dt=bin_dt, gti=gti, skip_checks=True)


def _has_events(path) -> bool:
    """True if the FITS file has an EVENTS extension with a TIME column."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul[1:]:
            cols = getattr(hdu, "columns", None)
            if cols is None:
                continue
            if ("EVENTS" in (hdu.name or "").upper()
                    and "TIME" in [c.upper() for c in cols.names]):
                return True
    return False


def load_lightcurve(path, dt=None, pi_range=None, energy_range=None,
                    time_range=None, kind="auto", columns=(0, 1),
                    rmf_file=None, verbose=True) -> Lightcurve:
    """Read a light curve from a file and return a stingray Lightcurve.

    FITS files are tried as an event list first (stingray `EventList.read`,
    binned at `dt`), then as a light curve (`Lightcurve.read`), then with an
    astropy fallback for files missing the OGIP keywords.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npy", ".npz"):
        lc = read_numpy(path, dt)
    elif ext in (".fits", ".fit", ".fts", ".evt", ".lc", ".gz", ".ftz"):
        lc = None
        if _has_events(path):
            if dt is None:
                raise ValueError("pass dt to bin an event list")
            try:
                ev = EventList.read(path, fmt="hea", rmf_file=rmf_file)
                lc = _events_to_lc(ev, dt, pi_range, energy_range)
            except Exception as exc:
                if verbose:
                    print(f"  (stingray's reader failed: {exc}; "
                          "falling back to astropy)")
        if lc is None:
            # binned products: read RATE/COUNTS and FRACEXP explicitly
            lc = _read_fits_fallback(path, dt, pi_range, energy_range)
        # drop bins outside the GTIs, so gaps are real gaps
        if lc.gti is not None and len(lc.gti) > 1:
            lc = lc.apply_gtis(inplace=False)
    else:
        lc = read_text(path, dt, columns, kind)

    if time_range is not None:
        lo = max(time_range[0], lc.time[0] - lc.dt / 2)
        hi = min(time_range[1], lc.time[-1] + lc.dt / 2)
        lc = lc.truncate(start=lo, stop=hi, method="time")
    if lc.gti is None or not len(lc.gti):
        lc.gti = gti_from_times(lc.time - lc.dt / 2, lc.dt)
    return lc


# =========================================================================
# Convenience: light curve in, report out
# =========================================================================

def run_lightcurve(lc: Lightcurve, seg_len=8.0, **kwargs):
    """Run the stationarity tests on a stingray Lightcurve.

    Passes the light curve, with its GTIs, to `stationarity_tests.run_all`,
    which does the segmentation through stingray and takes the rest of the
    options (n_blocks, fmin, fmax, fbin, norm, noise, bkg_rate, n_perm, ...).
    The surrogate test needs uninterrupted data, so it runs only for a light
    curve with a single GTI.
    """
    return run_all(lc=lc, seg_len=seg_len, **kwargs)
