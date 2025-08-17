import numpy as np
import pandas as pd
import pathlib, warnings, os, shutil, subprocess
from pathlib import Path
from typing import Optional
from typing import Tuple, Optional
from src.utils import _global_ylim
from src.backends.mpl import render_one_channel, render_all_channels

def generate_plot_videos(
    aligned_signal: np.ndarray,
    ratio: float,
    output_dir: str | Path,
    col_names: Optional[list[str]] = None,
    ylim: Optional[tuple[float, float]] = None,
    left: int = 250,
    right: int = 250,
    separate_videos: bool = False,
    combine_plots: bool = False,
    grid: Optional[tuple[int, int]] = (1, 1),
    video_fps: float = 30.0,
    plot_size: tuple[int, int] = (1280, 720),
    show_legend: bool = True,
    show_values: bool = False,
    backend: str = "matplotlib",
) -> bool:
    """
    Generate sliding-window plot video(s) from a pre-aligned signal.
    Output duration matches original video if plot_fps = video_fps * ratio.

    Parameters:
        - aligned_signal: 1D/2D array of aligned signal values
        - ratio: ratio of signal sampling rate to video frame rate
        - output_dir: directory to save the output video(s)
        - col_names: optional list of column names for the signal channels
        - ylim: optional tuple specifying y-axis limits for the plots. If None, auto-scale based on data.
        - left: number of signals to show before the current frame
        - right: number of signals to show after the current frame
        - separate_videos: if True, generate separate video for each channel
        - combine_plots: if True, combine all channels into a single plot, with different lines for legends
        - grid: tuple specifying the grid layout (rows, cols) for subplots when not combining plots, must be large enough to hold all channels
        - video_fps: frames per second for the output video(s)
        - plot_size: size of the output video frame (width, height)
        - show_legend: if True, show legend on the plots
        - show_values: if True, display current signal values on the plot
        - backend: plotting backend to use, currently only 'matplotlib' is supported,
          but I plan to support pygfx (or fastplotlib) in the future for better performance.

    Returns:
        - True if successful, False otherwise
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if separate_videos and combine_plots:
        raise ValueError("separate_videos and combine_plots cannot both be True.")

    sig = np.asarray(aligned_signal)
    if sig.ndim == 1:
        sig = sig[:, None]
    N, C = sig.shape

    if col_names is None or len(col_names) != C:
        col_names = [f"ch{c}" for c in range(C)]

    if left < 0 or right < 0:
        raise ValueError("left/right must be non-negative.")

    # fps to preserve original duration
    plot_fps = float(video_fps) * float(ratio)
    if plot_fps <= 0:
        raise ValueError("Computed plot_fps must be > 0 (check video_fps and ratio).")
    
    window_len = left + right + 1
    x = np.arange(-left, right + 1, dtype=float)
    
    # Global y-limits (stable visuals); renderers can still compute per-channel if you prefer
    global_ylim = _global_ylim(sig, ylim)

    # Heads-up for options we haven't implemented in renderers yet
    if show_legend and not combine_plots:
        warnings.warn("show_legend is only meaningful for 'combine_plots=True'. Ignoring.", RuntimeWarning)
    if show_values:
        warnings.warn("show_values not yet implemented in renderers. Ignoring.", RuntimeWarning)

    # Dispatch to the chosen renderer
    if separate_videos:
        # One file per channel
        for c in range(C):
            out = output_dir / f"{col_names[c]}_plot"
            # alpha=False by default; change to True if you want transparent .webm
            render_one_channel(
                signal=sig[:, c],
                out_path=out,
                left=left,
                right=right,
                fps=plot_fps,
                size=plot_size,
                ylim=global_ylim,     # or None if you want per-channel limits
                title=col_names[c],
                alpha=False,
            )
        return True

    if combine_plots:
        out = output_dir / "signals_plot_combined"
        render_all_channels(
            signals=sig,
            out_path=out,
            left=left,
            right=right,
            fps=plot_fps,
            size=plot_size,
            col_names=col_names if show_legend else None,
            ylim=global_ylim,
            alpha=False,             # set True to get transparent .webm
        )
        return True

    # Grid of subplots
    rows, cols = grid or (1, 1)
    if rows * cols < C:
        raise ValueError(f"grid {grid} too small for {C} channels.")

    out = output_dir / "signals_plot_grid"
    render_grid(
        signals=sig,
        out_path=out,
        left=left,
        right=right,
        fps=plot_fps,
        grid=(rows, cols),
        base_size=plot_size,
        col_names=col_names,
        ylim=global_ylim,
        alpha=False,                 # set True to get transparent .webm
    )
    return True
