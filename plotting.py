"""
plotting.py
-----------
Report figure for a stationarity test, used by both the simulator
(`nonstationary_qpo.py`) and the data script (`test_lightcurve.py`).

Panels: light curve; dynamic power spectrum (Poisson level removed, log
colour scale); the z map of log source-power deviations; the time-averaged
spectrum with the two most different blocks; and the per-bin trend p-value
and Bayes factor. Bins that the tests did not use are blank.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib import ticker



def plot_track(ax, track, color, outline):
    """Overlay (t, f0, hwhm): centroid solid, f0 +/- HWHM dashed."""
    if track is None:
        return
    t, f0, hw = track
    fx = [pe.Stroke(linewidth=2.2, foreground=outline, alpha=0.5), pe.Normal()]
    ax.plot(t, f0, color=color, lw=1.0, path_effects=fx)
    if hw is not None:
        for sgn in (-1, 1):
            ax.plot(t, f0 + sgn * hw, color=color, lw=0.8, ls="--",
                    path_effects=fx)

def most_different_blocks(z):
    """Indices (k, l), k < l, of the two blocks whose z vectors differ most.

    z is the K x J map rep.psr["z"] (NaN in unused bins). The distance is the
    mean squared difference over the bins finite in both blocks.
    """
    z = np.asarray(z, float)
    K = z.shape[0]
    if K < 2:
        return 0, 0
    diff = z[:, None, :] - z[None, :, :]                  # K x K x J
    with np.errstate(invalid="ignore"):
        D = np.nanmean(diff ** 2, axis=2)
    D = np.where(np.isfinite(D), D, -np.inf)
    np.fill_diagonal(D, -np.inf)
    k, l = np.unravel_index(np.argmax(D), D.shape)
    return (int(k), int(l)) if k < l else (int(l), int(k))


def _plain_log_axis(axis):
    """Log frequency axis labelled 0.2, 0.5, 1, 2, 5, ... in plain numbers.

    Majors at 1, 2 and 5 per decade, so a range under one decade still gets
    labels; minor ticks are unlabelled so they cannot overlap.
    """
    axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10)))
    axis.set_minor_formatter(ticker.NullFormatter())


def _lc_panel(ax, t, counts, dt, bin_s=1.0):
    """Count rate in `bin_s` bins, with gaps left blank."""
    t = np.asarray(t, float)
    nb = max(1, int(round(bin_s / dt)))
    m = counts.size // nb
    y = counts[: m * nb].reshape(m, nb).sum(axis=1) / (nb * dt)
    tb = t[: m * nb].reshape(m, nb)[:, 0]
    gap = np.diff(tb) > 1.5 * bin_s
    y = np.where(np.append(gap, False), np.nan, y)
    ax.plot(tb, y, lw=0.5, color="0.3")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Counts / s")
    return tb, y


def plot_report(rep, savepath, t, counts, dt, track=None,
                param_lines=(), title="", lc_bin=1.0):
    """Five-panel report figure.

    rep          StationarityReport
    t, counts    light curve (bin start times and counts per bin)
    track        (t, f0, hwhm) to overlay, e.g. injected QPO parameters
    param_lines  [(t, y, label, colour)] drawn on a twin axis of the light curve
    """
    g = rep.grid
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.4, 1.4], hspace=0.4,
                          wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    tb_lc, y_lc = _lc_panel(ax, t, counts, dt, lc_bin)
    
    # highlight the segments that make up the two most different blocks
    spb = g.seg_per_block
    kA, kB = most_different_blocks(rep.psr["z"])
    sel = ((kA, "C0", f"block {kA + 1}"), (kB, "C3", f"block {kB + 1}"))
    for k, col, lab in sel:
        m = np.zeros(tb_lc.size, bool)
        for s in g.seg_time[k * spb:(k + 1) * spb]:
            m |= (tb_lc >= s) & (tb_lc < s + g.seg_len)
        ax.plot(tb_lc, np.where(m, y_lc, np.nan), lw=0.5, color=col, label=lab)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(t[0], t[-1] + dt)
    if param_lines:
        ax2 = ax.twinx()
        for tt, yy, lab, col in param_lines:
            ax2.plot(tt, yy, color=col, lw=1.2, label=lab)
        ax2.set_ylabel("relative parameter")
        ax2.legend(loc="upper right", fontsize=8, ncol=len(param_lines))
    ax.set_title(title)

    # time-frequency axes: block centres and frequency bin edges
    tb = g.t_block
    edges_t = np.concatenate([[tb[0] - (tb[1] - tb[0]) / 2],
                              0.5 * (tb[1:] + tb[:-1]),
                              [tb[-1] + (tb[-1] - tb[-2]) / 2]])
    df = g.fbin / g.seg_len
    edges_f = np.append(g.freq - df / 2, g.freq[-1] + df / 2)

    ax = fig.add_subplot(gs[1, 0])
    src = np.where(g.keep[None, :], g.source, np.nan)
    fp = np.where(src > 0, src, np.nan) * g.freq[None, :]
    pc = ax.pcolormesh(edges_t, edges_f, fp.T, shading="flat", cmap="viridis",
                       norm=LogNorm(vmin=np.nanmin(fp), vmax=np.nanmax(fp)))
    plot_track(ax, track, "w", "k")
    for k, col, _ in sel:
            ax.axvline(tb[k], color=col, lw=1.2, ls="--",
                       path_effects=[pe.Stroke(linewidth=2.4, foreground="w",
                                               alpha=0.6), pe.Normal()])
    ax.set_yscale("log")
    _plain_log_axis(ax.yaxis)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(r"Dynamic PSD, $\nu (P_\nu - N)$")
    fig.colorbar(pc, ax=ax, pad=0.01)

    ax = fig.add_subplot(gs[1, 1])
    zlim = max(3.0, np.nanmax(np.abs(rep.psr["z"])))
    pc = ax.pcolormesh(edges_t, edges_f, rep.psr["z"].T, shading="flat",
                       cmap="PuOr", vmin=-zlim, vmax=zlim)
    plot_track(ax, track, "k", "w")
    for k, col, _ in sel:
            ax.axvline(tb[k], color=col, lw=1.2, ls="--")
    if "scan_blocks" in rep.psr:          # window picked by the window scan
        a, b = rep.psr["scan_blocks"]
        js = int(np.argmin(np.abs(g.freq - rep.psr["f_scan"])))
        ax.add_patch(Rectangle((edges_t[a], edges_f[js]),
                               edges_t[b] - edges_t[a],
                               edges_f[js + 1] - edges_f[js], fill=False,
                               ec="k", lw=1.5, zorder=5))
    ax.set_yscale("log")
    _plain_log_axis(ax.yaxis)
    ax.set_xlabel("Time (s)")
    ax.set_title(r"$z$: log source-power deviation (box: window scan)",
                 fontsize=10)
    fig.colorbar(pc, ax=ax, pad=0.01)

    # time-averaged spectrum over the full band (stingray, log-rebinned)
    ax = fig.add_subplot(gs[2, 0])
    N_all = g.seg_noise.mean()
    ps = g.avg_ps.rebin_log(f=0.03)
    fb = np.asarray(ps.freq, float)
    pb = np.asarray(ps.power, float).real
    ax.loglog(fb, fb * pb, color="0.6", lw=1, label="all data, raw")
    ax.loglog(fb, fb * (pb - N_all), color="k", lw=1,
              label="all data, noise subtracted")
    for k, col, lab in sel:
            ax.loglog(g.freq, g.freq * g.source[k], color=col, lw=1, alpha=0.8,
                      label=lab)
    if N_all > 0:
        ax.plot(fb, fb * N_all, color="0.4", ls=":", lw=1, label="noise level")
    lo = np.nanmin(np.where(g.keep, g.freq * g.S_hat, np.nan)) / 5
    ax.set_ylim(bottom=lo)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$\nu P_\nu$" + (r" (rms/mean)$^2$" if g.norm == "frac"
                                    else r" (rms$^2$)"))
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 1])
    ax.semilogx(g.freq, -np.log10(rep.psr["p_trend_j"]), color="k", lw=1,
                marker="o", ms=3, label=r"$-\log_{10} p$, trend per bin")
    ax.set_xlim(edges_f[0], edges_f[-1])
    ax.axhline(-np.log10(0.05), color="0.6", ls=":", lw=1)
    ax.set_ylim(0, max(1.6, 1.1 * np.nanmax(-np.log10(rep.psr["p_trend_j"]))))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$-\log_{10} p$")
    ax3 = ax.twinx()
    lbf = rep.bayes["log10_BF_per_freq"]
    ax3.semilogx(g.freq, lbf, color="C1", lw=1, marker="o", ms=3,
                 label=r"$\log_{10}$ BF$_j$ (bin varies : constant)")
    for y in (-1, 1):
        ax3.axhline(y, color="C1", ls=":", lw=0.8, alpha=0.6)
    ax3.axhline(0, color="C1", ls="-", lw=0.5, alpha=0.4)
    top = max(1.5, 1.1 * np.nanmax(np.abs(lbf)))
    ax3.set_ylim(-top, top)
    ax3.set_xlim(edges_f[0], edges_f[-1])
    _plain_log_axis(ax.xaxis)
    ax3.set_ylabel(r"$\log_{10}$ BF$_j$", color="C1")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("Per-bin trend p-value and Bayes factor "
                 "(unused bins omitted)")

    p, b = rep.psr, rep.bayes
    txt = [f"PSR total  p = {p.get('p_total_perm', p['p_total']):.2g}",
           f"PSR trend  p = {p.get('p_trend_perm', p['p_trend']):.2g}",
           f"max-bin    p = {p.get('p_trend_max_perm', np.nan):.2g}",
           f"window scan p = {p.get('p_scan_perm', np.nan):.2g}"]
    if rep.surrogate is not None:
        txt.append(f"Surrogate INS = {rep.surrogate['INS']:.2f}, "
                   f"p = {rep.surrogate['p_gamma']:.2g}")
    txt += [f"log10 B01 = {b['log10_B01']:.1f}",
            f"P(stationary) = {b['P_stationary']:.3g}"]
    fig.text(0.5, 0.045, "   |   ".join(txt[:4]) + "\n"
             + "   |   ".join(txt[4:]), ha="center", va="top",
             fontsize=9, family="monospace")

    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {savepath}")
