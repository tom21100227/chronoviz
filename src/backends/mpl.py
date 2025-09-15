from pathlib import Path
from ..utils import _compute_ylim, clamp_nan_2d
from ..writers import FrameWriter, create_ffmpeg_writer
import numpy as np
import matplotlib as mpl

mpl.use("Agg")  # offscreen; keeps GUI out of the loop
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from numba import njit
from typing import Tuple, List

USE_RGB24 = False

# global (before figure creation)
mpl.rcParams.update(
    {
        "text.antialiased": True,  # cheaper text
        "mathtext.default": "regular",  # avoid mathtext layout
        "axes.unicode_minus": False,  # reduce glyph fallback
        "font.family": "DejaVu Sans",  # pin to a single known font
        "font.sans-serif": ["DejaVu Sans"],
        "path.simplify": True,
        "path.simplify_threshold": 1.0,
        # "agg.path.chunksize": 10000,  # chunk large paths for less memory use
    }
)


def _prepare_line_styles(
    line_kwargs: dict | List[dict] | None, num_channels: int
) -> List[dict]:
    """Prepare line kwargs for multiple channels, same as plt.plot() accepts."""
    default_kwargs = {
        "lw": 1.2,
        "antialiased": True,
        "solid_joinstyle": "bevel",
        "solid_capstyle": "butt",
        "animated": True,
    }

    if line_kwargs is None:
        # No custom styling - use defaults with automatic colors for multi-channel
        if num_channels == 1:
            return [default_kwargs.copy()]
        else:
            # Auto-assign colors from matplotlib's default color cycle
            styles = []
            colors = plt.get_cmap("tab10").colors
            for i in range(num_channels):
                kwargs = default_kwargs.copy()
                kwargs["color"] = colors[i % len(colors)]
                styles.append(kwargs)
            return styles

    if isinstance(line_kwargs, dict):
        # Single kwargs dict - apply to all channels
        if num_channels == 1:
            kwargs = default_kwargs.copy()
            kwargs.update(line_kwargs)
            return [kwargs]
        else:
            # Multi-channel: if no color specified, auto-assign colors
            styles = []
            colors = plt.get_cmap("tab10").colors
            for i in range(num_channels):
                kwargs = default_kwargs.copy()
                kwargs.update(line_kwargs)
                # Only auto-assign color if user didn't specify one
                if "color" not in line_kwargs and "c" not in line_kwargs:
                    kwargs["color"] = colors[i % len(colors)]
                styles.append(kwargs)
            return styles

    if isinstance(line_kwargs, list):
        # List of kwargs - pad with defaults if needed
        colors = plt.get_cmap("tab10").colors
        styles = []
        for i in range(num_channels):
            kwargs = default_kwargs.copy()
            if i < len(line_kwargs):
                kwargs.update(line_kwargs[i])
            # Only auto-assign color if user didn't specify one for this channel
            if i >= len(line_kwargs) or (
                "color" not in line_kwargs[i] and "c" not in line_kwargs[i]
            ):
                kwargs["color"] = colors[i % len(colors)]
            styles.append(kwargs)
        return styles

    return [default_kwargs.copy() for _ in range(num_channels)]


def _apply_axis_customizations(ax, xlabel=None, ylabel=None, axis_kwargs=None):
    """Apply user-specified axis customizations."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)

    if axis_kwargs:
        # Handle common axis customizations
        if "grid" in axis_kwargs:
            if axis_kwargs["grid"]:
                ax.grid(True, alpha=0.15, linewidth=0.6)
            else:
                ax.grid(False)

        if "spines" in axis_kwargs:
            spine_settings = axis_kwargs["spines"]
            for spine_name, visible in spine_settings.items():
                if spine_name in ax.spines:
                    ax.spines[spine_name].set_visible(visible)

        if "tick_params" in axis_kwargs:
            ax.tick_params(**axis_kwargs["tick_params"])

        if "facecolor" in axis_kwargs:
            ax.set_facecolor(axis_kwargs["facecolor"])


def _lightweight_axes(ax):
    # no grid, thin spines
    ax.grid(True, which="major", axis="both", alpha=0.15, linewidth=0.6)
    for s in ax.spines.values():
        s.set_linewidth(1)

    fp = FontProperties(family="DejaVu Sans", size=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(fp)


@njit(cache=True)
def fill_window(sig, i, left, right, yout):
    L = left + right + 1
    for k in range(L):
        yout[k] = float("nan")
    N = sig.shape[0]
    s = i - left
    e = i + right + 1
    if s < 0:
        s = 0
    if e > N:
        e = N
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
    line_kwargs: dict | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    axis_kwargs: dict | None = None,
) -> Path:
    """
    Stream a sliding-window line plot to FFmpeg. Returns final output Path.

    Parameters:
        line_kwargs: Keyword arguments passed to plt.plot() (e.g., linewidth=2.0, color='red', linestyle='--')
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        axis_kwargs: Additional axis formatting (e.g., {'grid': False, 'spines': {'top': False}})
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    N = sig.size
    if N == 0:
        raise ValueError("signal is empty")

    if ylim is not None:
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

    # Prepare line styling - merge user kwargs with defaults
    default_kwargs = {
        "lw": 1.2,
        "solid_joinstyle": "bevel",
        "solid_capstyle": "butt",
        "animated": True,
    }
    if line_kwargs:
        default_kwargs.update(line_kwargs)

    # Initial line (keep a handle; no markers/alpha—those are slow)
    (line,) = ax.plot(x, np.full_like(x, np.nan), **default_kwargs)

    # Axes limits: x is fixed; y either provided or computed globally
    ylo, yhi = _compute_ylim(sig, ylim)
    ax.set_xlim(-left, right)
    ax.set_ylim(ylo, yhi)

    # Optional vertical cursor at t=0 for reference (cheap extra artist)
    cursor = ax.axvline(0.0, lw=0.8, ls="--", color="0.4")

    if title:
        ax.set_title(title, fontsize=10)

    # Apply axis customizations
    _apply_axis_customizations(ax, xlabel, ylabel, axis_kwargs)

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
        writer = create_ffmpeg_writer(
            Path(out_path), W, H, fps, alpha, use_rgb24=USE_RGB24
        )

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
            fig.canvas.blit(ax.bbox)  # cheap: only the axes region
            if USE_RGB24:
                argb = np.frombuffer(
                    fig.canvas.tostring_argb(), dtype=np.uint8
                ).reshape(H, W, 4)
                rgb[...] = argb[:, :, 1:4]  # ARGB → RGB (cheap memcpy in C)
                writer.write_frame(memoryview(rgb))
            else:
                buf = memoryview(fig.canvas.buffer_rgba())  # no numpy, no .tobytes()
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
    line_kwargs: dict | List[dict] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    axis_kwargs: dict | None = None,
) -> Path:
    """
    Combined-mode: all channels in one Axes (one line per channel), streamed to FFmpeg.

    Parameters:
        line_kwargs: Either a single dict of plt.plot() kwargs (applied to all channels),
                    or a list of dicts (one per channel)
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        axis_kwargs: Additional axis formatting (e.g., {'grid': False, 'spines': {'top': False}})
    """
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

    # Prepare styles for each channel
    styles = _prepare_line_styles(line_kwargs, C)

    # Lines (animated for blitting)
    lines: list[plt.Line2D] = []
    for i in range(C):
        (ln,) = ax.plot(x, np.full_like(x, np.nan), **styles[i])
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

    # Apply axis customizations
    _apply_axis_customizations(ax, xlabel, ylabel, axis_kwargs)

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
        writer = create_ffmpeg_writer(
            Path(out_path), W, H, fps, alpha, use_rgb24=USE_RGB24
        )

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
                argb = np.frombuffer(
                    fig.canvas.tostring_argb(), dtype=np.uint8
                ).reshape(H, W, 4)
                rgb[...] = argb[:, :, 1:4]
                writer.write_frame(memoryview(rgb))
            else:
                writer.write_frame(memoryview(fig.canvas.buffer_rgba()))
    finally:
        plt.close(fig)
        final_path = writer.close()

    return final_path


def render_grid(
    signals: np.ndarray,
    out_path: str | Path,
    left: int,
    right: int,
    fps: float,
    grid: Tuple[int, int] | None,
    size: tuple[int, int],
    ylim: tuple[float, float] | None = None,
    col_names: List[str] | None = None,
    alpha: bool = False,
    writer: FrameWriter | None = None,
    line_kwargs: dict | List[dict] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    axis_kwargs: dict | None = None,
) -> Path:
    """
    Grid-mode: each channel in its own subplot, streamed to FFmpeg.

    Parameters:
        line_kwargs: Either a single dict of plt.plot() kwargs (applied to all channels),
                    or a list of dicts (one per channel)
        xlabel: Label for x-axis (applied to all subplots)
        ylabel: Label for y-axis (applied to all subplots)
        axis_kwargs: Additional axis formatting applied to all subplots
    """
    sig = np.asarray(signals, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    N, C = sig.shape
    if N == 0:
        raise ValueError("signal is empty")

    if (col_names is None) or (len(col_names) != C):
        col_names = [f"ch{c}" for c in range(C)]

    # Determine grid layout
    if grid is None:
        # Auto-determine grid: try to make it roughly square
        rows = int(np.ceil(np.sqrt(C)))
        cols = int(np.ceil(C / rows))
        grid = (rows, cols)
    else:
        rows, cols = grid
        if rows * cols < C:
            raise ValueError(f"Grid {grid} too small for {C} channels")

    if ylim is not None:
        clamp_nan_2d(sig, ylim[0], ylim[1])

    # Figure/canvas setup to match exact pixel size
    W, H = int(size[0]), int(size[1])
    dpi = 100
    figsize = (W / dpi, H / dpi)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Shared x for all channels
    x = np.arange(-left, right + 1, dtype=np.float32)

    # Prepare styles for each channel
    styles = _prepare_line_styles(line_kwargs, C)

    # Lines for each subplot (animated for blitting)
    lines: list[plt.Line2D] = []
    for c in range(C):
        ax = axes[c]
        _lightweight_axes(ax)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(1)

        (line,) = ax.plot(x, np.full_like(x, np.nan), **styles[c])
        lines.append(line)

        # Set limits for each subplot
        ylo, yhi = _compute_ylim(sig[:, c : c + 1], ylim)
        ax.set_xlim(-left, right)
        ax.set_ylim(ylo, yhi)
        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)

        # Cursor at t=0 for each subplot
        ax.axvline(0.0, lw=0.8, ls="--", color="0.4")

        # Title for each subplot
        ax.set_title(col_names[c], fontsize=8, pad=2)

        # Apply axis customizations to each subplot
        _apply_axis_customizations(ax, xlabel, ylabel, axis_kwargs)

    # Hide unused subplots
    for c in range(C, len(axes)):
        axes[c].set_visible(False)

    fig.tight_layout(pad=0.5)

    # First draw + cache background
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    # Per-channel window buffers (prealloc) and warm-up JIT
    L = left + right + 1
    ywin = np.empty((C, L), dtype=np.float32)
    for c in range(C):
        fill_window(sig[:, c], 0, left, right, ywin[c])

    # Create writer if not provided
    if writer is None:
        writer = create_ffmpeg_writer(
            Path(out_path), W, H, fps, alpha, use_rgb24=USE_RGB24
        )

    # Optional buffer for rgb24 path
    rgb = np.empty((H, W, 3), dtype=np.uint8) if USE_RGB24 else None

    try:
        for i in range(N):
            # Update all channel windows and lines
            for c in range(C):
                fill_window(sig[:, c], i, left, right, ywin[c])
                lines[c].set_ydata(ywin[c])

            # Blit: restore cached bg, draw all lines, blit once
            fig.canvas.restore_region(background)
            for c in range(C):
                axes[c].draw_artist(lines[c])
            fig.canvas.blit(fig.bbox)

            # Write frame
            if USE_RGB24:
                # ARGB -> RGB (fast copy); then write 3B/px
                argb = np.frombuffer(
                    fig.canvas.tostring_argb(), dtype=np.uint8
                ).reshape(H, W, 4)
                rgb[...] = argb[:, :, 1:4]
                writer.write_frame(memoryview(rgb))
            else:
                writer.write_frame(memoryview(fig.canvas.buffer_rgba()))
    finally:
        plt.close(fig)
        final_path = writer.close()

    return final_path


# === Bar plot renderers ===

def _bar_values_vec(sig: np.ndarray, i: int, agg: str, window: int) -> np.ndarray:
    N, C = sig.shape
    if agg == "instant" or window <= 1:
        idx = i if i < N else N - 1
        return sig[idx, :]
    # aggregate over [i-window+1, i]
    s = i - window + 1
    if s < 0:
        s = 0
    e = i + 1
    win = sig[s:e, :]
    if agg == "mean":
        return np.nanmean(win, axis=0)
    else:  # "max"
        return np.nanmax(win, axis=0)


def _pick_colors(n: int):
    colors = plt.get_cmap("tab10").colors
    return [colors[i % len(colors)] for i in range(n)]


def render_all_channels_bar(
    signals: np.ndarray,
    out_path: str | Path,
    fps: float,
    size: tuple[int, int],
    ylim: tuple[float, float] | None = None,
    col_names: List[str] | None = None,
    bar_mode: str = "grouped",
    bar_agg: str = "instant",
    bar_window: int = 1,
    alpha: bool = False,
    writer: FrameWriter | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    axis_kwargs: dict | None = None,
) -> Path:
    """Combined bar plot: all channels in one Axes, streamed to FFmpeg."""
    sig = np.asarray(signals, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    N, C = sig.shape
    if N == 0:
        raise ValueError("signal is empty")

    if (col_names is None) or (len(col_names) != C):
        col_names = [f"ch{c}" for c in range(C)]

    if ylim is not None:
        clamp_nan_2d(sig, ylim[0], ylim[1])

    W, H = int(size[0]), int(size[1])
    dpi = 100
    figsize = (W / dpi, H / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    _lightweight_axes(ax)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # y-limits
    ylo, yhi = _compute_ylim(sig, ylim)
    ax.set_ylim(ylo, yhi)
    ax.set_autoscaley_on(False)

    # x for grouped or stacked
    colors = _pick_colors(C)
    rects: List[plt.Rectangle] = []  # type: ignore[name-defined]
    if bar_mode == "stacked":
        x0 = 0.0
        width = 0.8
        bottom = 0.0
        for c in range(C):
            r = plt.Rectangle((x0 - width / 2, bottom), width, 0.0, color=colors[c], animated=True, label=col_names[c])
            ax.add_patch(r)
            rects.append(r)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xticks([])
    else:  # grouped
        xs = np.arange(C, dtype=float)
        width = 0.8
        for c in range(C):
            (r,) = ax.bar([xs[c]], [0.0], width=width, color=colors[c], label=col_names[c], animated=True)
            rects.append(r)
        ax.set_xlim(-0.5, C - 0.5)
        # lightweight tick labels
        ax.set_xticks(xs)
        if C <= 12:
            ax.set_xticklabels(col_names)
        else:
            ax.set_xticklabels([])

    # Labels and customizations
    _apply_axis_customizations(ax, xlabel, ylabel, axis_kwargs)

    # Draw and cache background
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    if writer is None:
        writer = create_ffmpeg_writer(Path(out_path), W, H, fps, alpha, use_rgb24=USE_RGB24)
    rgb = np.empty((H, W, 3), dtype=np.uint8) if USE_RGB24 else None

    try:
        for i in range(N):
            vals = _bar_values_vec(sig, i, bar_agg, int(max(1, bar_window)))
            # Update rectangles
            if bar_mode == "stacked":
                bottom = 0.0
                for c in range(C):
                    h = float(vals[c])
                    if h < 0:
                        h = 0.0  # keep simple: ignore negative in stacking
                    rects[c].set_y(bottom)
                    rects[c].set_height(h)
                    bottom += h
            else:
                for c in range(C):
                    rects[c].set_height(float(vals[c]))

            # Blit
            fig.canvas.restore_region(background)
            for r in rects:
                ax.draw_artist(r)
            fig.canvas.blit(ax.bbox)

            if USE_RGB24:
                argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(H, W, 4)
                rgb[...] = argb[:, :, 1:4]
                writer.write_frame(memoryview(rgb))
            else:
                writer.write_frame(memoryview(fig.canvas.buffer_rgba()))
    finally:
        plt.close(fig)
        final_path = writer.close()

    return final_path


def render_grid_bar(
    signals: np.ndarray,
    out_path: str | Path,
    fps: float,
    grid: Tuple[int, int] | None,
    size: tuple[int, int],
    ylim: tuple[float, float] | None = None,
    col_names: List[str] | None = None,
    bar_mode: str = "grouped",
    bar_agg: str = "instant",
    bar_window: int = 1,
    alpha: bool = False,
    writer: FrameWriter | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    axis_kwargs: dict | None = None,
) -> Path:
    """Grid of bars: each channel in its own subplot; one bar per axes."""
    sig = np.asarray(signals, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    N, C = sig.shape
    if N == 0:
        raise ValueError("signal is empty")

    if (col_names is None) or (len(col_names) != C):
        col_names = [f"ch{c}" for c in range(C)]

    if ylim is not None:
        clamp_nan_2d(sig, ylim[0], ylim[1])

    # Determine grid
    if grid is None:
        rows = int(np.ceil(np.sqrt(C)))
        cols = int(np.ceil(C / rows))
        grid = (rows, cols)
    rows, cols = grid
    if rows * cols < C:
        raise ValueError(f"Grid {grid} too small for {C} channels")

    W, H = int(size[0]), int(size[1])
    dpi = 100
    figsize = (W / dpi, H / dpi)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    colors = _pick_colors(C)
    rects: List[plt.Rectangle] = []  # type: ignore[name-defined]

    # Setup axes and one bar per channel
    for c in range(C):
        ax = axes[c]
        _lightweight_axes(ax)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(1)
        ylo, yhi = _compute_ylim(sig[:, c : c + 1], ylim)
        ax.set_ylim(ylo, yhi)
        ax.set_autoscaley_on(False)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xticks([])
        (r,) = ax.bar([0.0], [0.0], width=0.6, color=colors[c], animated=True)
        rects.append(r)
        ax.set_title(col_names[c], fontsize=8, pad=2)
        _apply_axis_customizations(ax, xlabel, ylabel, axis_kwargs)

    # Hide unused subplots
    for c in range(C, len(axes)):
        axes[c].set_visible(False)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    if writer is None:
        writer = create_ffmpeg_writer(Path(out_path), W, H, fps, alpha, use_rgb24=USE_RGB24)
    rgb = np.empty((H, W, 3), dtype=np.uint8) if USE_RGB24 else None

    try:
        for i in range(N):
            vals = _bar_values_vec(sig, i, bar_agg, int(max(1, bar_window)))
            for c in range(C):
                rects[c].set_height(float(vals[c]))

            fig.canvas.restore_region(background)
            for c in range(C):
                axes[c].draw_artist(rects[c])
            fig.canvas.blit(fig.bbox)

            if USE_RGB24:
                argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(H, W, 4)
                rgb[...] = argb[:, :, 1:4]
                writer.write_frame(memoryview(rgb))
            else:
                writer.write_frame(memoryview(fig.canvas.buffer_rgba()))
    finally:
        plt.close(fig)
        final_path = writer.close()

    return final_path
