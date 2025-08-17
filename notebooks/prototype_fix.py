from __future__ import annotations

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: chronoviz
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prototype
#
# The core function for this is literally a function create_visualization(path_to_vid, timeseries (dataframe, list, array), **kwargs for ffmpeg).
#
# It will take a video path, a timeseries (dataframe, list, array) and any kwargs for ffmpeg to create a new video with the timeseries overlaid on the video.

# %% [markdown]
# ## Alignment
# Given a video and a time series, we check if they're similar in length. If we can achieve frame-to-frame consistency, do that. If the time series is 0.5x, 1/3x, 1/4x, we should be able to handle that as well. If the time series is not on a fraction/integer scale, we should have options of stretch or leave the last few frames as blank.

# %%
import os, sys, shutil, pathlib, warnings, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import av

import fastplotlib as fpl
from rendercanvas.offscreen import OffscreenRenderCanvas

os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "1"

TEST_DATA_DIR = Path(pathlib.Path.home(), "PersonalProjects/chronoviz/test_data")

# %%
def get_video_timeline(video_path: str | Path):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    fps = float(video_stream.average_rate)  # assuming constant frame rate
    n_frames = video_stream.frames
    if n_frames is None:
        n_frames = sum(1 for _ in container.decode(video=0))
    timeline = np.arange(n_frames, dtype=float) / fps
    return fps, n_frames, timeline

# %%
def read_timeseries(path: str | Path, key: str = None) -> pd.DataFrame:
    ext = pathlib.Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".h5", ".hdf5"):
        if key is None:
            raise ValueError("Key must be provided for HDF5 files.")
        try:
            df = pd.read_hdf(path, key=key)
        except TypeError:
            import h5py
            with h5py.File(path, "r") as f:
                if key not in f:
                    raise KeyError(f"Key '{key}' not found. Available: {list(f.keys())}")
                data = f[key][:]
                df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df

# %%
def align_signal_cfr(video_times: np.ndarray, sig_values: np.ndarray, mode: str,
                     ratio: float = 1.0, padding_mode: str = "edge", **kwargs) -> np.ndarray:
    """
    Align signal values to target length int(len(video_times) * ratio).

    mode:
      - 'resample': interpolate signal to the target length
      - 'pad': pad/truncate to the target length
    """
    if sig_values.ndim == 2:
        # avoid in-place; align each channel then stack
        return np.stack(
            [align_signal_cfr(video_times, sig_values[:, c], mode, ratio, padding_mode, **kwargs)
             for c in range(sig_values.shape[1])],
            axis=1
        )

    target_len = int(len(video_times) * ratio)
    if len(sig_values) == target_len:
        return sig_values

    tol = 1e-3
    observed_ratio = (len(sig_values) / len(video_times)) if len(video_times) else np.nan
    if len(video_times) and abs(observed_ratio - ratio) > tol:
        print(f"Warning: observed ratio {observed_ratio:.6f} != expected {ratio:.6f}; results may be off.")

    match mode:
        case "resample":
            n = len(sig_values)
            xp = np.arange(n, dtype=float)
            xq = np.linspace(0.0, n - 1.0, target_len)
            return np.interp(xq, xp, sig_values)
        case "pad":
            if len(sig_values) >= target_len:
                return sig_values[:target_len]
            pad_width = target_len - len(sig_values)
            return np.pad(sig_values, (0, pad_width), mode=padding_mode, **kwargs)
        case _:
            raise ValueError(f"Unknown mode: {mode}")

# %% [markdown]
# ## Plotting (fastplotlib window-buffer)

# ----------
# helpers
# ----------
def _global_ylim(sig: np.ndarray, ylim: tuple[float, float] | None = None) -> tuple[float, float]:
    if ylim is not None:
        return float(ylim[0]), float(ylim[1])
    finite = np.isfinite(sig)
    if not finite.any():
        return (-1.0, 1.0)
    lo, hi = float(np.min(sig[finite])), float(np.max(sig[finite]))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    else:
        m = 0.05 * (hi - lo)
        lo, hi = lo - m, hi + m
    return (lo, hi)

def _compute_ylim(y: np.ndarray, ylim: tuple[float, float] | None = None) -> tuple[float, float]:
    if ylim is not None:
        return float(ylim[0]), float(ylim[1])
    finite = np.isfinite(y)
    if not finite.any():
        return (-1.0, 1.0)
    lo, hi = float(np.min(y[finite])), float(np.max(y[finite]))
    if lo == hi:
        lo -= 0.5; hi += 0.5
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)

def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None

def _ffmpeg_writer(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    alpha: bool,
    encoder: str = "auto",        # "auto", "h264_videotoolbox", "h264_nvenc", "libx264", etc.
    hevc: bool = False,
    qp: int | None = None,
    bitrate: str | None = None,
    extra_args: list[str] | None = None,
):
    """
    Returns (proc, out_path). Sets INPUT pix_fmt BEFORE -i - and matches what we write.
    """
    extra_args = extra_args or []
    ext = out_path.suffix.lower()

    # choose encoder
    sel = encoder
    if alpha:
        if sel == "auto":
            sel = "vp9"  # alpha-capable; browsers-friendly
        if sel == "vp9" and ext != ".webm":
            out_path = out_path.with_suffix(".webm")
        if sel == "prores" and ext != ".mov":
            out_path = out_path.with_suffix(".mov")
    else:
        if sel == "auto":
            if sys.platform == "darwin":
                sel = "hevc_videotoolbox" if hevc else "h264_videotoolbox"
            elif _has_cmd("nvidia-smi"):
                sel = "hevc_nvenc" if hevc else "h264_nvenc"
            else:
                sel = "libx264"
        if sel.startswith(("h264", "hevc", "libx264", "libx265")) and ext not in {".mp4", ".m4v", ".mov"}:
            out_path = out_path.with_suffix(".mp4")

    size = f"{width}x{height}"
    def base_in(in_pix: str) -> list[str]:
        return ["ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", in_pix,      # INPUT pix fmt BEFORE -i -
                "-s", size,
                "-r", f"{fps}",
                "-i", "-"]

    if alpha:
        # We will WRITE RGBA frames in the loop
        if sel == "vp9":
            cmd = base_in("rgba") + [
                "-an",
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",
                "-lossless", "1",
            ] + (extra_args or []) + [str(out_path)]
        elif sel == "prores":
            cmd = base_in("rgba") + [
                "-an",
                "-c:v", "prores_ks",
                "-profile:v", "4444",
                "-pix_fmt", "yuva444p10le",
                "-qscale:v", "8",
            ] + (extra_args or []) + [str(out_path)]
        else:
            raise ValueError("Alpha requires 'vp9' or 'prores'")
    else:
        # Opaque path: we WRITE RGB24 frames in the loop
        if sel in {"libx264", "libx265"}:
            codec = "libx265" if (sel == "libx265" or hevc) else "libx264"
            base_args = ["-an", "-c:v", codec, "-preset", "veryfast"]
            if bitrate:
                base_args += ["-b:v", bitrate]
            elif qp is not None:
                base_args += ["-crf", str(qp)]
            else:
                base_args += ["-crf", "18"]
            base_args += ["-pix_fmt", "yuv420p"]
            cmd = base_in("rgb24") + base_args + (extra_args or []) + [str(out_path)]
        else:
            base_args = ["-an", "-c:v", sel]
            if sel.endswith("_nvenc"):
                if qp is not None:
                    base_args += ["-rc", "constqp", "-qp", str(qp)]
                elif bitrate:
                    base_args += ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate]
                else:
                    base_args += ["-rc", "vbr", "-cq", "19"]
                base_args += ["-preset", "p4", "-tune", "ull", "-pix_fmt", "yuv420p", "-bf", "0", "-g", str(int(fps*2))]
            elif sel.endswith("_videotoolbox"):
                base_args += ["-b:v", bitrate or "6M", "-pix_fmt", "yuv420p"]
            elif sel.endswith("_qsv"):
                base_args += (["-b:v", bitrate] if bitrate else ["-global_quality", "23"])
                base_args += ["-pix_fmt", "yuv420p"]
            elif sel.endswith("_amf"):
                base_args += (["-b:v", bitrate] if bitrate else ["-quality", "balanced"])
                base_args += ["-pix_fmt", "yuv420p"]
            cmd = base_in("rgb24") + base_args + (extra_args or []) + [str(out_path)]

    print("Running ffmpeg command:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,  # avoid blocking on ffmpeg logs
        stderr=None,                # inherit so you SEE errors
        bufsize=0,
    )
    return proc, out_path

def _init_canvas(width: int, height: int, alpha: bool):
    canvas = OffscreenRenderCanvas(size=(width, height), pixel_ratio=1.0, format="rgba-u8")
    fig = fpl.Figure(size=(width, height), canvas=canvas)
    if alpha:
        try:
            fig.canvas.set_clear_color((0, 0, 0, 0))
        except Exception:
            pass
    # Create once; caller should call fig.show() once before loop
    return fig

def _grab_frame(fig, want_rgba: bool) -> memoryview:
    """Return a memoryview of HxWxC uint8 matching want_rgba (True=4ch, False=3ch)."""
    arr = fig.canvas.read_pixels()
    if arr is None:
        arr = np.asarray(fig.canvas.draw())
    if want_rgba:
        if arr.shape[-1] == 3:
            alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=-1)
        return memoryview(arr)
    else:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return memoryview(arr)

def _add_vertical_line(ax, x0: float, y0: float, y1: float, **kwargs):
    xs = np.array([x0, x0], dtype=np.float32)
    ys = np.array([y0, y1], dtype=np.float32)
    return ax.add_line(xs, ys, **kwargs)

# ----------
# Renderers (window-buffer)
# ----------
def render_one_channel(
    signal: np.ndarray,
    out_path: Path,
    left: int,
    right: int,
    fps: float,
    size: tuple[int, int] = (1280, 720),
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    alpha: bool = False,
) -> Path:
    sig = np.asarray(signal, dtype=np.float32)
    N = sig.shape[0]
    W = left + right + 1
    x = np.arange(-left, right + 1, dtype=np.float32)

    width, height = size
    fig = _init_canvas(width, height, alpha)
    ax = fig[0, 0]

    y0, y1 = _compute_ylim(sig, ylim)
    ax.camera.show_rect(-left, right, y0, y1)
    _add_vertical_line(ax, 0.0, y0, y1, thickness=1.25, opacity=0.7)
    if title:
        ax.add_text(title, offset=(10, 10))

    ywin = np.full(W, np.nan, dtype=np.float32)
    line = ax.add_line(x, ywin)

    fig.show()  # initialize once

    proc, out_path = _ffmpeg_writer(out_path, width, height, fps, alpha)

    want_rgba = bool(alpha)
    for t in range(N):
        s = max(0, t - left)
        e = min(N, t + right + 1)
        span = e - s
        if span > 0:
            ywin[:span] = sig[s:e]
        if span < W:
            ywin[span:] = np.nan
        line.set_data(y=ywin)
        frame = _grab_frame(fig, want_rgba=want_rgba)
        proc.stdin.write(frame)

    proc.stdin.close()
    proc.wait()
    return out_path

def render_all_channels(
    signals: np.ndarray,                # [T, C]
    out_path: Path,
    left: int,
    right: int,
    fps: float,
    size: tuple[int, int] = (1280, 720),
    col_names: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
    alpha: bool = False,
) -> Path:
    sig = np.asarray(signals, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    T, C = sig.shape
    names = col_names if (col_names and len(col_names) == C) else [f"ch{c}" for c in range(C)]

    W = left + right + 1
    x = np.arange(-left, right + 1, dtype=np.float32)

    width, height = size
    fig = _init_canvas(width, height, alpha)
    ax = fig[0, 0]

    y0, y1 = _compute_ylim(sig.reshape(-1), ylim)
    ax.camera.show_rect(-left, right, y0, y1)
    _add_vertical_line(ax, 0.0, y0, y1, thickness=1.25, opacity=0.7)

    lines = []
    for c in range(C):
        ywin = np.full(W, np.nan, dtype=np.float32)
        lines.append(ax.add_line(x, ywin))
        ax.add_text(names[c], offset=(10, 10 + 18 * (c + 1)))

    fig.show()

    proc, out_path = _ffmpeg_writer(out_path, width, height, fps, alpha)
    want_rgba = bool(alpha)

    for t in range(T):
        s = max(0, t - left)
        e = min(T, t + right + 1)
        span = e - s
        for c in range(C):
            ywin = lines[c].data[1]  # y reference (fastplotlib stores internally; safer to reassign)
            # safer: rebuild ywin each frame to avoid internal ref surprises
            ytmp = np.full(W, np.nan, dtype=np.float32)
            if span > 0:
                ytmp[:span] = sig[s:e, c]
            lines[c].set_data(x=x, y=ytmp)
        frame = _grab_frame(fig, want_rgba=want_rgba)
        proc.stdin.write(frame)

    proc.stdin.close()
    proc.wait()
    return out_path

def render_grid(
    signals: np.ndarray,                # [T, C]
    out_path: Path,
    left: int,
    right: int,
    fps: float,
    grid: tuple[int, int],              # (rows, cols)
    base_size: tuple[int, int] = (1280, 720),
    col_names: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
    alpha: bool = False,
) -> Path:
    sig = np.asarray(signals, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    T, C = sig.shape
    rows, cols = grid
    if rows * cols < C:
        raise ValueError(f"grid {grid} too small for {C} channels.")
    names = col_names if (col_names and len(col_names) == C) else [f"ch{c}" for c in range(C)]

    W = left + right + 1
    x = np.arange(-left, right + 1, dtype=np.float32)

    width, height = base_size
    per_row_h = max(240, height // max(1, rows))
    total_h = per_row_h * rows

    canvas = OffscreenRenderCanvas(size=(width, total_h), pixel_ratio=1.0, format="rgba-u8")
    fig = fpl.Figure(size=(width, total_h), shape=(rows, cols), canvas=canvas)

    lines = []
    idx = 0
    for r in range(rows):
        for cc in range(cols):
            if idx >= C:
                break
            ax = fig[r, cc]
            y0, y1 = _compute_ylim(sig[:, idx], ylim)
            ax.camera.show_rect(-left, right, y0, y1)
            _add_vertical_line(ax, 0.0, y0, y1, thickness=1.25, opacity=0.7)
            ax.add_text(names[idx], offset=(10, 10))
            ywin = np.full(W, np.nan, dtype=np.float32)
            line = ax.add_line(x, ywin)
            lines.append((idx, line))
            idx += 1

    fig.show()

    proc, out_path = _ffmpeg_writer(out_path, width, total_h, fps, alpha)
    want_rgba = bool(alpha)

    for t in range(T):
        s = max(0, t - left)
        e = min(T, t + right + 1)
        span = e - s
        for ch_idx, line in lines:
            ytmp = np.full(W, np.nan, dtype=np.float32)
            if span > 0:
                ytmp[:span] = sig[s:e, ch_idx]
            line.set_data(x=x, y=ytmp)
        frame = _grab_frame(fig, want_rgba=want_rgba)
        proc.stdin.write(frame)

    proc.stdin.close()
    proc.wait()
    return out_path

# ----------
# Orchestrator
# ----------
from typing import Optional

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
) -> bool:
    """
    Generate sliding-window plot video(s) from a pre-aligned signal.
    Output duration matches original video if plot_fps = video_fps * ratio.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if separate_videos and combine_plots:
        raise ValueError("separate_videos and combine_plots cannot both be True.")

    sig = np.asarray(aligned_signal)
    if sig.ndim == 1:
        sig = sig[:, None]
    N, C = sig.shape

    if not col_names or len(col_names) != C:
        col_names = [f"ch{c}" for c in range(C)]

    if left < 0 or right < 0:
        raise ValueError("left/right must be non-negative.")

    plot_fps = float(video_fps) * float(ratio)
    if plot_fps <= 0:
        raise ValueError("Computed plot_fps must be > 0 (check video_fps and ratio).")

    global_ylim = _global_ylim(sig, ylim)

    if show_legend and not combine_plots:
        warnings.warn("show_legend is only meaningful for 'combine_plots=True'. Ignoring.", RuntimeWarning)
    if show_values:
        warnings.warn("show_values not yet implemented in renderers. Ignoring.", RuntimeWarning)

    if separate_videos:
        for c in range(C):
            out = output_dir / f"{col_names[c]}_plot"
            render_one_channel(
                signal=sig[:, c],
                out_path=out,
                left=left,
                right=right,
                fps=plot_fps,
                size=plot_size,
                ylim=global_ylim,
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
            alpha=False,
        )
        return True

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
        alpha=False,
    )
    return True

# %%
# quick test (adjust paths for your env)
if __name__ == "__main__":
    test_vid_path = TEST_DATA_DIR / "slp/03.mp4"
    fps, num_frames, timeline = get_video_timeline(test_vid_path)
    df_h5 = read_timeseries(TEST_DATA_DIR / "slp/03.h5", key="data")
    print(f"Successfully read data with shape: {df_h5.shape}")
    df_csv = read_timeseries(TEST_DATA_DIR / "slp/03.csv")
    print(f"Successfully read data with shape: {df_csv.shape}")
    aligned = align_signal_cfr(timeline, df_h5.values, mode="resample", ratio=1.0, padding_mode="edge")

    generate_plot_videos(
        aligned_signal=aligned,
        ratio=1.0,
        output_dir=TEST_DATA_DIR / "output_plots",
        col_names=["track0", "track1"],
        grid=(1, 2),
    )
