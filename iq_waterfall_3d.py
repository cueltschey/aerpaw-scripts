#!/usr/bin/env python3
"""
iq_waterfall_3d.py — 3D waterfall spectrogram for IQ signal data

Reads raw IQ samples and renders a 3D height-map where:
  X = frequency bins (centered at --center-freq)
  Y = time (STFT frames)
  Z = power spectral density in dBFS

Supports raw float32/int16 binary, .npy complex arrays, and .csv.
If no input file is given, a synthetic 5G NR-like OFDM signal is generated.

Examples
--------
# Synthetic demo
python iq_waterfall_3d.py

# File input, custom colormap and view angle
python iq_waterfall_3d.py capture.bin --sample-rate 61.44e6 --fft-size 2048 \\
       --colormap turbo --elev 40 --azim -45 --output waterfall.png

# Tight power window, dark theme
python iq_waterfall_3d.py capture.npy -r 30.72e6 --vmin -80 --vmax -20 \\
       --colormap inferno --style dark_background
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import stft


# ── Colormap catalogue ────────────────────────────────────────────────────────
COLORMAPS = [
    "plasma", "viridis", "inferno", "magma", "cividis",
    "hot", "afmhot", "gist_heat",
    "turbo", "jet", "rainbow",
    "cool", "winter", "ocean",
    "RdYlBu_r", "Spectral_r", "coolwarm",
    "gnuplot", "gnuplot2", "CMRmap",
]

STYLES = ["dark_background", "default", "seaborn-v0_8-dark", "ggplot", "bmh"]


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="3D waterfall spectrogram for IQ signal data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Omit INPUT to run with a synthetic 5G NR demo signal.",
    )

    # ── Input ──────────────────────────────────────────────────────────────
    p.add_argument(
        "input", nargs="?",
        help="Input file: raw float32 IQ binary (.bin/.dat), .npy complex array, or .csv (I,Q cols).",
    )
    p.add_argument(
        "--format", choices=["auto", "f32", "f64", "sc16", "npy", "csv"],
        default="auto",
        help="File format. f32=interleaved float32, sc16=interleaved int16, auto=detect from extension.",
    )
    p.add_argument("--max-samples", "-n", type=int, default=None,
                   help="Cap on IQ samples to read from file.")

    # ── Signal parameters ──────────────────────────────────────────────────
    p.add_argument("--sample-rate", "-r", type=float, default=30.72e6,
                   help="Sample rate in Hz (30.72 MHz = standard 5G NR 100 MHz BW).")
    p.add_argument("--center-freq", "-f", type=float, default=3.5e9,
                   help="Center frequency in Hz (cosmetic, used for axis label).")

    # ── STFT parameters ────────────────────────────────────────────────────
    p.add_argument("--fft-size", type=int, default=1024,
                   help="FFT / STFT window length in samples.")
    p.add_argument("--overlap", type=float, default=0.75,
                   help="STFT window overlap fraction [0, 1).")
    p.add_argument("--window",
                   choices=["hann", "hamming", "blackman", "blackmanharris", "flattop", "boxcar"],
                   default="hann",
                   help="STFT window function.")
    p.add_argument("--max-frames", type=int, default=256,
                   help="Maximum time-frames on Y axis (decimated if the STFT yields more).")

    # ── Power scaling ──────────────────────────────────────────────────────
    p.add_argument("--vmin", type=float, default=None,
                   help="Minimum power (dBFS) for Z axis / color floor. Auto = 5th percentile.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Maximum power (dBFS) for Z axis / color ceiling. Auto = 99.5th percentile.")
    p.add_argument("--ref-power", type=float, default=1.0,
                   help="Reference power for dBFS = 10·log10(|X|²/ref).")

    # ── Appearance ─────────────────────────────────────────────────────────
    p.add_argument("--colormap", "-c", default="plasma",
                   choices=COLORMAPS, metavar="CMAP",
                   help=f"Colormap. Choices: {', '.join(COLORMAPS)}.")
    p.add_argument("--style", default="dark_background",
                   choices=STYLES, metavar="STYLE",
                   help=f"Matplotlib style. Choices: {', '.join(STYLES)}.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Surface opacity [0=transparent … 1=opaque].")
    p.add_argument("--elev", type=float, default=30.0,
                   help="Elevation viewing angle in degrees.")
    p.add_argument("--azim", type=float, default=-60.0,
                   help="Azimuth viewing angle in degrees.")
    p.add_argument("--title", default=None,
                   help="Plot title (auto-generated if omitted).")
    p.add_argument("--figsize", type=float, nargs=2, default=[13, 8],
                   metavar=("W", "H"), help="Figure size in inches.")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    p.add_argument("--no-colorbar", action="store_true",
                   help="Suppress the colorbar.")
    p.add_argument("--shade", action="store_true", default=False,
                   help="Enable matplotlib surface shading (can obscure colour detail).")

    # ── Axis units ─────────────────────────────────────────────────────────
    p.add_argument("--freq-unit", choices=["Hz", "kHz", "MHz", "GHz"], default="MHz",
                   help="Unit for the frequency axis.")
    p.add_argument("--time-unit", choices=["s", "ms", "us"], default="ms",
                   help="Unit for the time axis.")

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument("--output", "-o", default=None,
                   help="Save figure to this path (png/pdf/svg) instead of showing.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open an interactive window (useful with --output).")

    return p.parse_args()


# ── I/O ───────────────────────────────────────────────────────────────────────
def load_iq(path: str, fmt: str, max_samples) -> np.ndarray:
    """Return a complex64 1-D array of IQ samples."""
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

    else:  # f32
        raw = np.fromfile(path, dtype=np.float32)
        samples = raw[0::2] + 1j * raw[1::2]

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples.astype(np.complex64)


def make_synthetic_5g(sample_rate: float = 30.72e6, duration: float = 0.005) -> np.ndarray:
    """
    Synthetic 5G NR-like OFDM signal.

    Uses numerology μ=1 (30 kHz SCS), 132 active subcarriers, random QPSK
    symbols, plus additive white Gaussian noise at ~20 dB SNR.
    """
    rng = np.random.default_rng(0)
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate

    n_sc = 132                    # active subcarriers
    scs = 30e3                    # subcarrier spacing (Hz)
    freqs = (np.arange(n_sc) - n_sc // 2) * scs

    # Vary symbol power over OFDM slots to give the waterfall some texture
    n_slots = 14
    slot_len = n // n_slots
    signal = np.zeros(n, dtype=np.complex128)

    for slot in range(n_slots):
        sl = slice(slot * slot_len, (slot + 1) * slot_len)
        t_sl = t[sl]
        slot_pwr = rng.uniform(0.3, 1.0)
        for f in freqs:
            sym = rng.choice([-1, 1]) + 1j * rng.choice([-1, 1])
            signal[sl] += slot_pwr * sym * np.exp(2j * np.pi * f * t_sl)

    signal /= np.max(np.abs(signal))
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.056   # ~25 dB SNR
    return (signal + noise).astype(np.complex64)


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_surface_colored(ax, X, Y, Z, cmap_name, vmin, vmax, alpha, shade):
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    face_colors = cmap(norm(Z))
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=face_colors,
        shade=shade,
        alpha=alpha,
        linewidth=0,
        antialiased=True,
        rcount=Z.shape[1],
        ccount=Z.shape[0],
    )
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(Z)
    return surf, mappable


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Load / generate ────────────────────────────────────────────────────
    if args.input:
        samples = load_iq(args.input, args.format, args.max_samples)
        src_label = args.input
    else:
        print("No input file — generating synthetic 5G NR OFDM signal …", file=sys.stderr)
        samples = make_synthetic_5g(args.sample_rate)
        src_label = "synthetic 5G NR"

    print(
        f"  {len(samples):,} samples  |  "
        f"{len(samples) / args.sample_rate * 1e3:.3f} ms  |  "
        f"{args.sample_rate / 1e6:.3f} MS/s",
        file=sys.stderr,
    )

    # ── STFT ───────────────────────────────────────────────────────────────
    hop = max(1, int(args.fft_size * (1.0 - args.overlap)))
    freqs, times, Zxx = stft(
        samples,
        fs=args.sample_rate,
        window=args.window,
        nperseg=args.fft_size,
        noverlap=args.fft_size - hop,
        return_onesided=False,
    )
    # Centre-shift so DC is in the middle
    freqs = np.fft.fftshift(freqs)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # Power in dBFS  (n_freqs × n_frames)
    power_db = 10.0 * np.log10(np.abs(Zxx) ** 2 / args.ref_power + 1e-30)

    # Decimate frames to max_frames
    nf = power_db.shape[1]
    if nf > args.max_frames:
        idx = np.linspace(0, nf - 1, args.max_frames, dtype=int)
        power_db = power_db[:, idx]
        times = times[idx]

    # Clip to display range
    vmin = args.vmin if args.vmin is not None else float(np.percentile(power_db, 5))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(power_db, 99.5))
    power_db = np.clip(power_db, vmin, vmax)

    print(
        f"  Power range: {power_db.min():.1f} … {power_db.max():.1f} dBFS  "
        f"(display {vmin:.1f} … {vmax:.1f})",
        file=sys.stderr,
    )

    # ── Axis grids ─────────────────────────────────────────────────────────
    freq_scale = {"Hz": 1.0, "kHz": 1e-3, "MHz": 1e-6, "GHz": 1e-9}[args.freq_unit]
    time_scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[args.time_unit]

    F = freqs * freq_scale + args.center_freq * freq_scale   # (n_freqs,)
    T = times * time_scale                                   # (n_frames,)
    FF, TT = np.meshgrid(F, T, indexing="ij")               # both (n_freqs, n_frames)

    # ── Figure ─────────────────────────────────────────────────────────────
    plt.style.use(args.style)
    fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")

    surf, mappable = plot_surface_colored(
        ax, FF, TT, power_db,
        cmap_name=args.colormap,
        vmin=vmin, vmax=vmax,
        alpha=args.alpha,
        shade=args.shade,
    )

    ax.set_xlabel(f"Frequency ({args.freq_unit})", labelpad=12)
    ax.set_ylabel(f"Time ({args.time_unit})", labelpad=12)
    ax.set_zlabel("Power (dBFS)", labelpad=10)

    title = args.title or (
        f"5G IQ Waterfall  ·  {args.sample_rate / 1e6:.3g} MS/s  ·  "
        f"FFT {args.fft_size}  ·  {args.colormap}"
    )
    ax.set_title(title, pad=16)
    ax.view_init(elev=args.elev, azim=args.azim)

    if not args.no_colorbar:
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.45, aspect=14, pad=0.08)
        cbar.set_label("Power (dBFS)")

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved → {args.output}", file=sys.stderr)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
