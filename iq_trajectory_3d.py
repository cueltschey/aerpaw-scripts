#!/usr/bin/env python3
"""
iq_trajectory_3d.py — 3D IQ constellation / trajectory plot

Plots I (real) vs Q (imaginary) vs time as a 3D scatter or line,
color-coded by time, amplitude, or phase.

Useful for:
  • Visualising 5G NR modulation order (QPSK, 16-QAM, 64-QAM …)
  • Detecting phase noise, frequency drift, IQ imbalance
  • Comparing clean vs distorted signal trajectories

Examples
--------
# Synthetic demo, colour by amplitude
python iq_trajectory_3d.py

# File input, scatter mode, colour by phase, viridis map
python iq_trajectory_3d.py capture.bin -r 30.72e6 \\
       --plot-type scatter --color-by phase --colormap viridis

# Line + scatter overlay, restrict time window
python iq_trajectory_3d.py capture.npy --plot-type both \\
       --t-start 0.002 --t-end 0.004 --stride 2 --colormap turbo

# Save to file without interactive window
python iq_trajectory_3d.py capture.bin -o traj.png --no-show
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ── Colormap catalogue ────────────────────────────────────────────────────────
COLORMAPS = [
    "plasma", "viridis", "inferno", "magma", "cividis",
    "turbo", "jet", "rainbow",
    "hot", "afmhot", "copper",
    "cool", "winter", "spring",
    "RdYlBu_r", "Spectral_r", "hsv",
    "gnuplot", "gnuplot2", "CMRmap",
]

STYLES = ["dark_background", "default", "seaborn-v0_8-dark", "ggplot", "bmh"]


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="3D IQ constellation / trajectory over time",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Omit INPUT to run with a synthetic 5G NR demo signal.",
    )

    # ── Input ──────────────────────────────────────────────────────────────
    p.add_argument(
        "input", nargs="?",
        help="Input file: raw float32 IQ binary, .npy complex array, or .csv (I,Q cols).",
    )
    p.add_argument(
        "--format", choices=["auto", "f32", "f64", "sc16", "npy", "csv"],
        default="auto",
        help="File format. f32=interleaved float32, sc16=interleaved int16.",
    )
    p.add_argument("--max-samples", "-n", type=int, default=None,
                   help="Cap on IQ samples to read from file.")

    # ── Signal parameters ──────────────────────────────────────────────────
    p.add_argument("--sample-rate", "-r", type=float, default=30.72e6,
                   help="Sample rate in Hz.")

    # ── Sample selection ───────────────────────────────────────────────────
    p.add_argument("--t-start", type=float, default=None,
                   help="Start time in seconds (clips from the beginning).")
    p.add_argument("--t-end", type=float, default=None,
                   help="End time in seconds (clips at end).")
    p.add_argument("--stride", type=int, default=1,
                   help="Plot every Nth sample (decimation for speed / clarity).")
    p.add_argument("--max-points", type=int, default=20_000,
                   help="Hard cap on displayed points (auto-strides if needed).")

    # ── Smoothing ──────────────────────────────────────────────────────────
    p.add_argument("--smooth", type=int, default=0,
                   help="Apply a running-average filter of this length before plotting (0=off).")

    # ── Plot type ──────────────────────────────────────────────────────────
    p.add_argument("--plot-type", choices=["scatter", "line", "both"], default="scatter",
                   help="Render as scatter points, a connected line, or both.")
    p.add_argument("--marker-size", type=float, default=4.0,
                   help="Scatter marker size in points².")
    p.add_argument("--line-width", type=float, default=0.8,
                   help="Line width for line / both modes.")

    # ── Color coding ───────────────────────────────────────────────────────
    p.add_argument("--color-by", choices=["time", "amplitude", "phase", "power_db"],
                   default="time",
                   help=(
                       "Variable used to drive the colormap: "
                       "time=sample index, amplitude=|IQ|, "
                       "phase=angle(IQ) in radians, power_db=10·log10(|IQ|²)."
                   ))
    p.add_argument("--colormap", "-c", default="plasma",
                   choices=COLORMAPS, metavar="CMAP",
                   help=f"Colormap. Choices: {', '.join(COLORMAPS)}.")
    p.add_argument("--vmin", type=float, default=None,
                   help="Color scale minimum (auto if omitted).")
    p.add_argument("--vmax", type=float, default=None,
                   help="Color scale maximum (auto if omitted).")

    # ── Reference rings ────────────────────────────────────────────────────
    p.add_argument("--no-rings", action="store_true",
                   help="Suppress reference unit-circle ring on the I/Q plane.")
    p.add_argument("--ring-radii", type=float, nargs="+", default=None,
                   help="Explicit ring radii to draw (overrides auto unit circle).")

    # ── Axes ───────────────────────────────────────────────────────────────
    p.add_argument("--time-unit", choices=["s", "ms", "us"], default="us",
                   help="Unit for the time (Z) axis.")
    p.add_argument("--iq-norm", action="store_true",
                   help="Normalise I and Q to unit peak before plotting.")

    # ── Appearance ─────────────────────────────────────────────────────────
    p.add_argument("--style", default="dark_background",
                   choices=STYLES, metavar="STYLE",
                   help=f"Matplotlib style. Choices: {', '.join(STYLES)}.")
    p.add_argument("--alpha", type=float, default=0.8,
                   help="Point / line opacity.")
    p.add_argument("--elev", type=float, default=25.0,
                   help="Elevation viewing angle in degrees.")
    p.add_argument("--azim", type=float, default=-50.0,
                   help="Azimuth viewing angle in degrees.")
    p.add_argument("--title", default=None,
                   help="Plot title (auto-generated if omitted).")
    p.add_argument("--figsize", type=float, nargs=2, default=[11, 9],
                   metavar=("W", "H"), help="Figure size in inches.")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    p.add_argument("--no-colorbar", action="store_true",
                   help="Suppress the colorbar.")

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument("--output", "-o", default=None,
                   help="Save figure to this path (png/pdf/svg).")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open an interactive window.")

    return p.parse_args()


# ── I/O ───────────────────────────────────────────────────────────────────────
def load_iq(path: str, fmt: str, max_samples) -> np.ndarray:
    if fmt == "auto":
        if path.endswith(".npy"):
            fmt = "npy"
        elif path.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = "f32"

    if fmt == "npy":
        data = np.load(path)
        samples = data if np.iscomplexobj(data) else data[0::2] + 1j * data[1::2]
    elif fmt == "csv":
        raw = np.loadtxt(path, delimiter=",")
        samples = (raw[:, 0] + 1j * raw[:, 1]) if raw.ndim == 2 else (raw[0::2] + 1j * raw[1::2])
    elif fmt == "sc16":
        raw = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
        samples = raw[0::2] + 1j * raw[1::2]
    elif fmt == "f64":
        raw = np.fromfile(path, dtype=np.float64)
        samples = raw[0::2] + 1j * raw[1::2]
    else:
        raw = np.fromfile(path, dtype=np.float32)
        samples = raw[0::2] + 1j * raw[1::2]

    if max_samples is not None:
        samples = samples[:max_samples]
    return samples.astype(np.complex64)


def make_synthetic_5g(sample_rate: float = 30.72e6, duration: float = 0.0002) -> np.ndarray:
    """
    Synthetic 5G NR-like OFDM IQ with:
      - 64-QAM-ish amplitude distribution
      - mild phase noise and IQ imbalance to make the trajectory interesting
    """
    rng = np.random.default_rng(1)
    n = int(sample_rate * duration)

    # QAM-64 alphabet (normalised)
    qam_levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float) / 7.0
    I = rng.choice(qam_levels, n)
    Q = rng.choice(qam_levels, n)
    signal = (I + 1j * Q).astype(np.complex64)

    # Add mild phase noise (Wiener process)
    phase_noise = np.cumsum(rng.normal(0, 0.008, n)).astype(np.float32)
    signal *= np.exp(1j * phase_noise)

    # Mild IQ imbalance
    signal = signal.real * 1.05 + 1j * signal.imag * 0.95

    # AWGN at ~28 dB SNR
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.04
    return signal + noise


# ── Colour array factory ──────────────────────────────────────────────────────
def build_color_values(samples: np.ndarray, method: str) -> np.ndarray:
    if method == "time":
        return np.arange(len(samples), dtype=float)
    elif method == "amplitude":
        return np.abs(samples)
    elif method == "phase":
        return np.angle(samples)          # −π … +π
    elif method == "power_db":
        return 10.0 * np.log10(np.abs(samples) ** 2 + 1e-30)
    raise ValueError(f"Unknown color-by: {method}")


# ── Thin wrapper around Line3DCollection for coloured 3-D line ────────────────
def colored_line3d(ax, x, y, z, c_vals, cmap, norm, lw, alpha):
    """Draw a 3-D line with per-segment colouring."""
    pts = np.stack([x, y, z], axis=1)          # (N, 3)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (N-1, 2, 3)
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=lw, alpha=alpha)
    lc.set_array(c_vals[:-1])
    ax.add_collection(lc)
    return lc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Load / generate ────────────────────────────────────────────────────
    if args.input:
        samples = load_iq(args.input, args.format, args.max_samples)
        src_label = args.input
    else:
        print("No input file — generating synthetic 5G NR 64-QAM IQ …", file=sys.stderr)
        samples = make_synthetic_5g(args.sample_rate)
        src_label = "synthetic 5G NR 64-QAM"

    total_n = len(samples)
    print(f"  {total_n:,} samples  |  {total_n / args.sample_rate * 1e6:.2f} µs", file=sys.stderr)

    # ── Time window clip ───────────────────────────────────────────────────
    i_start = int(args.t_start * args.sample_rate) if args.t_start is not None else 0
    i_end = int(args.t_end * args.sample_rate) if args.t_end is not None else total_n
    samples = samples[i_start:i_end]

    # ── Smoothing ──────────────────────────────────────────────────────────
    if args.smooth > 1:
        kernel = np.ones(args.smooth, dtype=np.float32) / args.smooth
        samples = (
            np.convolve(samples.real, kernel, mode="same") +
            1j * np.convolve(samples.imag, kernel, mode="same")
        ).astype(np.complex64)

    # ── Stride / cap ───────────────────────────────────────────────────────
    stride = args.stride
    if len(samples) // stride > args.max_points:
        stride = max(stride, len(samples) // args.max_points)
        print(f"  Auto-stride set to {stride} to keep ≤{args.max_points:,} points.", file=sys.stderr)
    samples = samples[::stride]

    # ── Normalise ──────────────────────────────────────────────────────────
    if args.iq_norm:
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak

    # ── Build axes ─────────────────────────────────────────────────────────
    time_scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[args.time_unit]
    t_abs = (i_start + np.arange(len(samples)) * stride) / args.sample_rate * time_scale
    I_vals = samples.real.astype(float)
    Q_vals = samples.imag.astype(float)

    # ── Colour values ──────────────────────────────────────────────────────
    c_vals = build_color_values(samples, args.color_by)
    vmin = args.vmin if args.vmin is not None else float(c_vals.min())
    vmax = args.vmax if args.vmax is not None else float(c_vals.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(args.colormap)

    print(
        f"  Plotting {len(samples):,} points  |  "
        f"color-by={args.color_by}  range=[{vmin:.3g}, {vmax:.3g}]",
        file=sys.stderr,
    )

    # ── Figure ─────────────────────────────────────────────────────────────
    plt.style.use(args.style)
    fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(c_vals)

    # ── Draw scatter ───────────────────────────────────────────────────────
    if args.plot_type in ("scatter", "both"):
        sc = ax.scatter(
            I_vals, Q_vals, t_abs,
            c=c_vals,
            cmap=cmap,
            norm=norm,
            s=args.marker_size,
            alpha=args.alpha,
            depthshade=True,
            linewidths=0,
        )

    # ── Draw coloured line ─────────────────────────────────────────────────
    if args.plot_type in ("line", "both"):
        colored_line3d(ax, I_vals, Q_vals, t_abs, c_vals, cmap, norm,
                       lw=args.line_width, alpha=args.alpha)
        # Manually set axis limits since Line3DCollection doesn't auto-expand
        margin = 0.05
        for setter, vals in [
            (ax.set_xlim, I_vals), (ax.set_ylim, Q_vals), (ax.set_zlim, t_abs)
        ]:
            lo, hi = vals.min(), vals.max()
            pad = (hi - lo) * margin or 0.1
            setter(lo - pad, hi + pad)

    # ── Reference rings ────────────────────────────────────────────────────
    if not args.no_rings:
        radii = args.ring_radii if args.ring_radii else [1.0]
        theta = np.linspace(0, 2 * np.pi, 256)
        z_ring = t_abs.min()
        ring_color = "white" if "dark" in args.style else "gray"
        for r in radii:
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    zs=z_ring, zdir="z",
                    color=ring_color, lw=0.6, alpha=0.4, linestyle="--")

    # ── Labels ─────────────────────────────────────────────────────────────
    ax.set_xlabel("I (real)", labelpad=10)
    ax.set_ylabel("Q (imag)", labelpad=10)
    ax.set_zlabel(f"Time ({args.time_unit})", labelpad=10)

    title = args.title or (
        f"5G IQ Trajectory  ·  color={args.color_by}  ·  {args.colormap}"
    )
    ax.set_title(title, pad=14)
    ax.view_init(elev=args.elev, azim=args.azim)

    if not args.no_colorbar:
        color_labels = {
            "time": f"Time ({args.time_unit})",
            "amplitude": "Amplitude |IQ|",
            "phase": "Phase (rad)",
            "power_db": "Power (dB)",
        }
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.45, aspect=14, pad=0.08)
        cbar.set_label(color_labels[args.color_by])

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved → {args.output}", file=sys.stderr)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
