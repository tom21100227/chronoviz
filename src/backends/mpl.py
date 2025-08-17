import subprocess, os
from pathlib import Path
from ..utils import _has_cmd, _compute_ylim, clamp_nan_2d
from ..writers import FFmpegWriter, FrameWriter, create_ffmpeg_writer
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # offscreen; keeps GUI out of the loop
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
from numba import njit
from typing import Tuple, List, Optional

USE_RGB24 = False

# global (before figure creation)
mpl.rcParams.update({
    "text.antialiased": True,        # cheaper text
    "mathtext.default": "regular",    # avoid mathtext layout
    "axes.unicode_minus": False,      # reduce glyph fallback
    "font.family": "DejaVu Sans",     # pin to a single known font
    "font.sans-serif": ["DejaVu Sans"],
    "path.simplify": True,
    "path.simplify_threshold": 1.0,
    "agg.path.chunksize": 10000,    # chunk large paths for less memory use
})

def _lightweight_axes(ax):
    # no grid, thin spines
    ax.grid(True, which="major", axis="both", alpha=0.15, linewidth=0.6)
    for s in ax.spines.values(): s.set_linewidth(1)

    # # few ticks, no minors
    # ax.xaxis.set_minor_locator(mticker.NullLocator())
    # ax.yaxis.set_minor_locator(mticker.NullLocator())
    # ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))
    # ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    fp = FontProperties(family="DejaVu Sans", size=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(fp)



@njit(cache=True)
def fill_window(sig, i, left, right, yout):
    L = left + right + 1
    for k in range(L): yout[k] = float('nan')
    N = sig.shape[0]
    s = i - left; e = i + right + 1
    if s < 0: s = 0
    if e > N: e = N
    dst = left - (i - s)
    for j in range(e - s):
        yout[dst + j] = sig[s + j]

def render_one_channel(
    signal: np.ndarray,
    out_path: str | Path,
    left: int,
    right: int,
    fps: float,
    size: tuple[int, int],
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    alpha: bool = False,
    writer: FrameWriter | None = None,
) -> Path:
    """
    Stream a sliding-window line plot to FFmpeg. Returns final output Path.

    Notes for speed:
      - Create once, update line data per frame (no re-plotting).
      - Autoscale is off; we only change xlim as we move the window.
      - Window is fixed length; edges are padded with NaN so draw cost stays flat.
      - If you make left/right very large, consider LOD min–max downsampling.
    """
    if not _has_cmd("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    sig = np.asarray(signal, dtype=np.float32).ravel()
    N = sig.size
    if N == 0:
        raise ValueError("signal is empty")
    
    if ylim is not None:
        # original_sig = sig.copy() # leave a copy here for if we decided to display true values later
        clamp_nan_2d(sig, ylim[0], ylim[1])

    # Figure/canvas setup to match exact pixel size
    W, H = int(size[0]), int(size[1])
    dpi = 100  # pick a simple DPI so figsize is exact pixels
    figsize = (W / dpi, H / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    _lightweight_axes(ax)
    
    # style: crisp & light for speed and readability
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # Fixed x-grid: [-left, ..., 0, ..., right]
    x = np.arange(-left, right + 1, dtype=np.float32)

    # Initial line (keep a handle; no markers/alpha—those are slow)
    (line,) = ax.plot(x, np.full_like(x, np.nan), lw=1.2, solid_joinstyle="bevel", solid_capstyle="butt", animated=True)

    # Axes limits: x is fixed; y either provided or computed globally
    ylo, yhi = _compute_ylim(sig, ylim)
    ax.set_xlim(-left, right)
    ax.set_ylim(ylo, yhi)

    # Optional vertical cursor at t=0 for reference (cheap extra artist)
    cursor = ax.axvline(0.0, lw=0.8, ls="--", color="0.4")

    if title:
        ax.set_title(title, fontsize=10)

    # Turn off autoscale to avoid recomputation on every set_data
    ax.set_autoscalex_on(False)
    ax.set_autoscaley_on(False)

    fig.tight_layout(pad=0)

    # First draw to allocate the renderer
    line.set_animated(True)
    fig.canvas.draw()

    background = fig.canvas.copy_from_bbox(ax.bbox)
    ywin = np.empty_like(x, dtype=np.float32)
    fill_window(sig, 0, left, right, ywin)  # JIT warmup

    # Create writer if not provided
    if writer is None:
        writer = create_ffmpeg_writer(Path(out_path), W, H, fps, alpha, use_rgb24=USE_RGB24)

    rgb = np.empty((H, W, 3), dtype=np.uint8) if USE_RGB24 else None

    try:
        # Render all frames: one frame per index to preserve original duration
        # (Your caller already set plot_fps = video_fps * ratio, so durations match.)
        for i in range(N):
            fill_window(sig, i, left, right, ywin)
            line.set_ydata(ywin)
            # If you want dynamic vertical scaling per-frame, uncomment:
            # ax.set_ylim(*_compute_ylim(y, ylim))

            fig.canvas.restore_region(background)
            ax.draw_artist(line)
            fig.canvas.blit(ax.bbox)           # cheap: only the axes region
            if USE_RGB24:
                argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(H, W, 4)
                rgb[...] = argb[:, :, 1:4]                   # ARGB → RGB (cheap memcpy in C)
                writer.write_frame(memoryview(rgb))
            else: 
                buf = memoryview(fig.canvas.buffer_rgba())   # no numpy, no .tobytes()
                writer.write_frame(buf)
    finally:
        plt.close(fig)
        final_path = writer.close()

    return final_path

def render_all_channels(
    signals: np.ndarray,
    out_path: str | Path,
    left: int,
    right: int,
    fps: float,
    size: tuple[int, int],
    ylim: tuple[float, float] | None = None,
    col_names: List[str] | None = None,
    alpha: bool = False,
    writer: FrameWriter | None = None,
) -> Path:
    """
    Combined-mode: all channels in one Axes (one line per channel), streamed to FFmpeg.
    """
    if not _has_cmd("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")

    sig = np.asarray(signals, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    N, C = sig.shape
    if N == 0:
        raise ValueError("signal is empty")

    if (col_names is None) or (len(col_names) != C):
        col_names = [f"ch{c}" for c in range(C)]

    if ylim is not None:
        # original_sig = sig.copy()
        clamp_nan_2d(sig, ylim[0], ylim[1])

    # Figure/canvas setup to match exact pixel size
    W, H = int(size[0]), int(size[1])
    dpi = 100
    figsize = (W / dpi, H / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    _lightweight_axes(ax)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # Shared x for all channels
    x = np.arange(-left, right + 1, dtype=np.float32)

    # Lines (animated for blitting)
    lines: list[plt.Line2D] = []
    for _ in range(C):
        (ln,) = ax.plot(
            x,
            np.full_like(x, np.nan),
            lw=1.2,
            antialiased=True,
            solid_joinstyle="bevel",
            solid_capstyle="butt",
            animated=True,
        )
        lines.append(ln)

    for ln in lines:
        ln.set_clip_on(False)
        ln.set_clip_path(None)
        ln.set_clip_box(None)
    # Limits (global y)
    ylo, yhi = _compute_ylim(sig, ylim)
    ax.set_xlim(-left, right)
    ax.set_ylim(ylo, yhi)
    ax.set_autoscalex_on(False)
    ax.set_autoscaley_on(False)

    # Cursor at t=0
    ax.axvline(0.0, lw=0.8, ls="--", color="0.4")

    # Optional legend (draw once; becomes part of the cached background)
    if col_names:
        leg = ax.legend(
            col_names,
            loc="upper right",
            frameon=True,
            framealpha=0.2,
            fontsize=8,
            ncol=1,
        )
        for legline in leg.get_lines():
            legline.set_linewidth(1.5)

    fig.tight_layout(pad=0)

    # First draw + cache background
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    # Per-channel window buffers (prealloc) and warm-up JIT
    L = left + right + 1
    ywin = np.empty((C, L), dtype=np.float32)
    for c in range(C):
        fill_window(sig[:, c], 0, left, right, ywin[c])

    # Create writer if not provided
    if writer is None:
        writer = create_ffmpeg_writer(Path(out_path), W, H, fps, alpha, use_rgb24=USE_RGB24)

    # Optional buffer for rgb24 path
    rgb = np.empty((H, W, 3), dtype=np.uint8) if USE_RGB24 else None

    try:
        for i in range(N):
            # Update all channel windows
            for c in range(C):
                fill_window(sig[:, c], i, left, right, ywin[c])
                lines[c].set_ydata(ywin[c])

            # Blit: restore cached bg, draw all lines, blit once
            fig.canvas.restore_region(background)
            for ln in lines:
                ax.draw_artist(ln)
            fig.canvas.blit(ax.bbox)

            # Write frame
            if USE_RGB24:
                # ARGB -> RGB (fast copy); then write 3B/px
                argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(H, W, 4)
                rgb[...] = argb[:, :, 1:4]
                writer.write_frame(memoryview(rgb))
            else:
                writer.write_frame(memoryview(fig.canvas.buffer_rgba()))
    finally:
        plt.close(fig)
        final_path = writer.close()

    return final_path
