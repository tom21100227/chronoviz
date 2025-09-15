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
# The core function for this is literallt a function create_visualization(path_to_vid, timeseries (dataframe, list, arrary), **kwargs for ffmpeg).
#
# It will take a video path, a timeseries (dataframe, list, array) and any kwargs for ffmpeg to create a new video with the timeseries overlaid on the video.

# %% [markdown]
# ## Alignment
# Given a video and a times series, we check if they're similar in length. If we can achieve frame to frame consistenty, do that, if the time series is 0.5x, 1/3x, 1/4x, we should be able to handle that as well. If the time series is not a on a fraction/integer scale, we should have options of stretch or leave the last few frames as blank.

# %%
import av
import numpy as np
import pandas as pd
import os
import sys
import shutil
import pathlib
from pathlib import Path
from rendercanvas.offscreen import OffscreenRenderCanvas
import warnings

os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "1"

TEST_DATA_DIR = Path(pathlib.Path.home(), "PersonalProjects/chronoviz/test_data")


# %%
def get_video_timeline(video_path: str | Path):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    fps = video_stream.average_rate  # assuming constant frame rate
    n_frames = video_stream.frames
    if n_frames is None:
        n_frames = sum(1 for _ in container.decode(video=0))
    timeline = np.arange(n_frames) / float(fps)
    return fps, n_frames, timeline


# %%
def read_timeseries(path: str | Path, key: str = None) -> pd.DataFrame:
    ext = pathlib.Path(path).suffix
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".h5", ".hdf5"]:
        if key is None:
            raise ValueError("Key must be provided for HDF5 files.")

        # Try pandas first, fall back to h5py for raw HDF5 files
        try:
            df = pd.read_hdf(path, key=key)
        except TypeError:
            import h5py

            # This is likely a raw HDF5 file, use h5py
            with h5py.File(path, "r") as f:
                if key not in f:
                    raise KeyError(
                        f"Key '{key}' not found in HDF5 file. Available keys: {list(f.keys())}"
                    )
                data = f[key][:]
                df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df


# %%
def align_signal_cfr(
    video_times: np.ndarray,
    sig_values: np.ndarray,
    mode: str,
    ratio: float = 1,
    padding_mode: str = "edge",
    **kwargs,
) -> np.ndarray:
    """
    Align signal values to video times.

    Parameters:
        - video_times: 1D array of video frame timestamps
        - sig_values: 1D/2D array of signal values, if 2D, the array should have shape (n_samples, n_channels)
        - mode: 'resample` for interpolating signal to video times,
                'pad' for padding/truncating signal to match video length
        - ratio: useful for when signal is at a different sampling rate than video,
                e.g. 0.5 for downsampling: signal is half the rate of video
                2.0 for upsampling: signal is twice the rate of video
        - padding_mode: if mode is 'pad', this specifies how to pad the signal,
                       e.g. 'edge' to pad with the last value, 'constant' to pad with zeros, etc.
        - **kwargs: additional keyword arguments for np.interp or np.pad

    Returns:
        - aligned signal values as 1D/2D array with length == int(len(video_times) * ratio)
    """
    if sig_values.ndim == 2:
        # avoid in-place modification; align each channel and stack
        return np.stack(
            [
                align_signal_cfr(
                    video_times, sig_values[:, c], mode, ratio, padding_mode, **kwargs
                )
                for c in range(sig_values.shape[1])
            ],
            axis=1,
        )

    target_signal_length = int(len(video_times) * ratio)
    if len(sig_values) == target_signal_length:
        # already the desired length for this ratio
        return sig_values

    ratio_mismatch_tolerance = 1e-3
    observed_ratio = len(sig_values) / len(video_times) if len(video_times) else np.nan
    if len(video_times) and abs(observed_ratio - ratio) > ratio_mismatch_tolerance:
        # if the observed ratio is significantly different from the expected ratio
        # this could indicate a mismatch in sampling rates or an error in the data
        print(
            f"Warning: Observed signal to frames ratio {observed_ratio} does not match expected ratio {ratio}. The alignment results may not be accurate."
        )

    match mode:
        case "resample":
            nums_signals = len(sig_values)
            xp = np.arange(nums_signals, dtype=float)
            xq = np.linspace(0.0, nums_signals - 1.0, target_signal_length)
            return np.interp(xq, xp, sig_values)
        case "pad":
            # truncate signal
            if len(sig_values) >= target_signal_length:
                return sig_values[:target_signal_length]

            pad_width = target_signal_length - len(sig_values)
            padded_values = np.pad(
                sig_values, (0, pad_width), mode=padding_mode, **kwargs
            )
            return padded_values
        case _:
            raise ValueError(f"Unknown mode: {mode}")


# %% [markdown]
# ### Now let's put it all together

# %%
test_vid_path = TEST_DATA_DIR / "slp/03.mp4"
fps, num_frames, timeline = get_video_timeline(test_vid_path)
# Test the updated function
df_h5 = read_timeseries(TEST_DATA_DIR / "slp/03.h5", key="data")
print(f"Successfully read data with shape: {df_h5.shape}")
df_csv = read_timeseries(TEST_DATA_DIR / "slp/03.csv")
print(f"Successfully read data with shape: {df_csv.shape}")

# Test the alignment function
aligned = align_signal_cfr(
    timeline, df_h5.values, mode="resample", ratio=1, padding_mode="edge"
)

# %% [markdown]
# ## Plotting
#
# Now that the alignment

# %%
from typing import Optional, Sequence, Tuple
from pathlib import Path
import subprocess
import fastplotlib as fpl

# ---------- helpers ----------


def _global_ylim(
    sig: np.ndarray, ylim: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """Compute global y-limits for a signal array."""
    if ylim is not None:
        return float(ylim[0]), float(ylim[1])
    finite = np.isfinite(sig)
    if not finite.any():
        return -1.0, 1.0
    lo, hi = float(np.min(sig[finite])), float(np.max(sig[finite]))
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    else:
        margin = (hi - lo) * 0.1
        lo, hi = lo - margin, hi + margin
    return lo, hi


def _compute_ylim(
    y: np.ndarray, ylim: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    if ylim is not None:
        return float(ylim[0]), float(ylim[1])
    finite = np.isfinite(y)
    if not finite.any():
        return -1.0, 1.0
    lo, hi = float(np.min(y[finite])), float(np.max(y[finite]))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _ffmpeg_writer(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    alpha: bool,
    encoder: str = "auto",  # e.g., "auto", "h264_nvenc", "h264_videotoolbox", "libx264", "hevc_nvenc", "vp9", "prores"
    hevc: bool = False,  # prefer HEVC (H.265) when using hardware encoders
    qp: (
        int | None
    ) = None,  # for NVENC const QP (e.g., 20). If None, use CRF/CBR defaults.
    bitrate: str | None = None,  # e.g., "6M"
    extra_args: list[str] | None = None,
):
    """
    Returns (Popen, out_path). Picks a hardware encoder when possible for opaque video.
    Alpha videos use CPU (VP9 or ProRes 4444).
    """
    extra_args = extra_args or []

    # --- Choose encoder & container ---
    sel_encoder = encoder
    ext = out_path.suffix.lower()

    if alpha:
        # Hardware encoders do not carry alpha; pick VP9 (webm) or ProRes 4444 (mov)
        if sel_encoder == "auto":
            # VP9 with alpha is broadly decodable in browsers; ProRes 4444 is faster encode but huge.
            sel_encoder = "vp9"  # or "prores" if you prefer pro workflows
        if sel_encoder == "vp9":
            if ext != ".webm":
                out_path = out_path.with_suffix(".webm")
        elif sel_encoder == "prores":
            if ext != ".mov":
                out_path = out_path.with_suffix(".mov")
        else:
            # force a valid alpha-capable choice
            sel_encoder = "vp9"
            if ext != ".webm":
                out_path = out_path.with_suffix(".webm")
    else:
        # Opaque path: try to auto-pick a HW encoder
        if sel_encoder == "auto":
            if sys.platform == "darwin":
                sel_encoder = "hevc_videotoolbox" if hevc else "h264_videotoolbox"
            else:
                # Try NVIDIA first
                if _has_cmd("nvidia-smi"):
                    sel_encoder = "hevc_nvenc" if hevc else "h264_nvenc"
                else:
                    # Try Intel QSV (Windows/Linux with supported iGPU/drivers)
                    # We can't reliably probe, so offer QSV next; otherwise AMF if on Windows
                    if sys.platform.startswith("win"):
                        sel_encoder = "h264_amf"  # AMD fallback on Windows
                    else:
                        sel_encoder = "libx264"  # portable CPU fallback

        # set default container
        if sel_encoder.startswith(
            ("h264", "hevc", "libx264", "libx265", "h264_", "hevc_")
        ):
            if ext not in {".mp4", ".m4v", ".mov"}:
                out_path = out_path.with_suffix(".mp4")
        elif sel_encoder in {"h264_amf", "h264_qsv"}:
            if ext not in {".mp4", ".m4v"}:
                out_path = out_path.with_suffix(".mp4")

    # --- Build ffmpeg command ---
    size = f"{width}x{height}"

    def _base_with_input_pixfmt(in_pix):
        return [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            in_pix,
            "-s",
            size,
            "-r",
            f"{fps}",
            "-i",
            "-",
        ]

    if alpha:
        # RGBA in, encode with alpha
        if sel_encoder == "vp9":
            cmd = (
                _base_with_input_pixfmt("rgba")
                + [
                    "-pix_fmt",
                    "rgba",
                    "-an",
                    "-c:v",
                    "libvpx-vp9",
                    "-pix_fmt",
                    "yuva420p",
                    "-lossless",
                    "1",  # fastest & big; or use -crf 28 for smaller
                ]
                + extra_args
                + [str(out_path)]
            )
        else:  # prores 4444
            cmd = (
                _base_with_input_pixfmt("rgba")
                + [
                    "-pix_fmt",
                    "rgba",
                    "-an",
                    "-c:v",
                    "prores_ks",
                    "-profile:v",
                    "4444",
                    "-pix_fmt",
                    "yuva444p10le",
                    "-qscale:v",
                    "8",  # quality; lower is higher quality (1–11)
                ]
                + extra_args
                + [str(out_path)]
            )
    else:
        # Opaque path: RGB in; ffmpeg will convert to YUV for the encoder
        in_pix = "rgb24"
        if sel_encoder in {"libx264", "libx265"}:
            # CPU encoders
            codec = "libx265" if (sel_encoder == "libx265" or hevc) else "libx264"
            base_args = [
                "-an",
                "-c:v",
                codec,
                "-preset",
                "veryfast",
            ]
            if bitrate:
                base_args += ["-b:v", bitrate]
            elif qp is not None:
                # libx264/libx265 use CRF, not QP
                crf = str(qp)  # reuse qp var as CRF for simplicity
                base_args += ["-crf", crf]
            else:
                base_args += ["-crf", "18"]
            # ensure browser-safe pix fmt
            base_args += ["-pix_fmt", "yuv420p"]
            cmd = base + pix_in + base_args + extra_args + [str(out_path)]
        else:
            # Hardware encoders
            base_args = ["-an", "-c:v", sel_encoder]
            # low-latency & sensible defaults
            if sel_encoder.endswith("_nvenc"):
                # NVENC rate control: const QP or CBR
                if qp is not None:
                    base_args += ["-rc", "constqp", "-qp", str(qp)]
                elif bitrate:
                    base_args += [
                        "-b:v",
                        bitrate,
                        "-maxrate",
                        bitrate,
                        "-bufsize",
                        bitrate,
                    ]
                else:
                    base_args += ["-rc", "vbr", "-cq", "19"]
                base_args += [
                    "-preset",
                    "p4",
                    "-tune",
                    "ull",
                    "-pix_fmt",
                    "yuv420p",
                    "-bf",
                    "0",
                    "-g",
                    str(int(fps * 2)),
                ]
            elif sel_encoder.endswith("_videotoolbox"):
                if bitrate:
                    base_args += ["-b:v", bitrate]
                else:
                    base_args += ["-b:v", "6M"]
                base_args += ["-pix_fmt", "yuv420p"]
            elif sel_encoder.endswith("_qsv"):
                if bitrate:
                    base_args += ["-b:v", bitrate]
                else:
                    base_args += ["-global_quality", "23"]
                base_args += ["-pix_fmt", "yuv420p"]
            elif sel_encoder.endswith("_amf"):
                if bitrate:
                    base_args += ["-b:v", bitrate]
                else:
                    base_args += ["-quality", "balanced"]
                base_args += ["-pix_fmt", "yuv420p"]

            cmd = base + pix_in + base_args + extra_args + [str(out_path)]

    # Spawn with unbuffered stdin; drop ffmpeg logs during tight loops

    print("Running ffmpeg command:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    return proc, out_path


def _init_canvas(width: int, height: int, alpha: bool):
    canvas = OffscreenRenderCanvas(
        size=(width, height), pixel_ratio=1.0, format="rgba-u8"
    )
    fig = fpl.Figure(size=(width, height), canvas=canvas)
    if alpha:
        # Try to make the background transparent. fastplotlib/pygfx usually respects RGBA clear.
        fig.canvas.set_clear_color((0, 0, 0, 0))
    return fig


def _read_frame(fig, alpha: bool) -> bytes:
    # fastplotlib returns HxWx[3 or 4] uint8
    fig.show()  # to trigger rendering
    arr = np.asarray(fig.canvas.draw())
    # Ensure channel count matches ffmpeg pix_fmt
    if alpha:
        if arr.shape[-1] == 3:
            # add opaque alpha if backend returned RGB
            arr = np.concatenate(
                [arr, np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)], axis=-1
            )
        return arr.tobytes()
    else:
        # drop alpha if present
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr.tobytes()


# ---------- 1) render one channel (single axis) ----------


def render_one_channel(
    signal: np.ndarray,
    out_path: Path,
    left: int,
    right: int,
    fps: float,
    size: Tuple[int, int] = (1280, 720),
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    alpha: bool = False,
) -> Path:
    signal = np.asarray(signal).astype(float)
    N = len(signal)
    W = left + right + 1
    x = np.arange(-left, right + 1, dtype=float)

    width, height = size
    fig = _init_canvas(width, height, alpha)
    ax = fig[0, 0]

    y0, y1 = _compute_ylim(signal, ylim)
    ax.camera.show_rect(left=-left, right=right, bottom=y0, top=y1)
    # Add vertical line at x=0
    vline_data = np.column_stack(
        [[0.0, 0.0], [y0, y1], [0.0, 0.0]]
    )  # Add z=0 for 3D coords
    ax.add_line(vline_data, thickness=1.5, alpha=0.7)
    if title:
        ax.add_text(title, offset=(10, 10, 0))

    ywin = np.full(W, np.nan, dtype=float)
    line_data = np.column_stack([x, ywin, np.zeros(W)])  # Add z=0 for 3D coords
    line = ax.add_line(line_data)

    proc, out_path = _ffmpeg_writer(out_path, width, height, fps, alpha)

    for t in range(N):
        s = max(0, t - left)
        e = min(N, t + right + 1)
        span = e - s
        ywin[:span] = signal[s:e]
        if span < W:
            ywin[span:] = np.nan
        # Update line data
        new_data = np.column_stack([x, ywin, np.zeros(W)])  # Add z=0 for 3D coords
        line.data = new_data
        proc.stdin.write(_read_frame(fig, alpha))

    proc.stdin.close()
    proc.wait()
    return out_path


# ---------- 2) render all channels into one plot (multiple lines on one axis) ----------


def render_all_channels(
    signals: np.ndarray,  # shape [T, C]
    out_path: Path,
    left: int,
    right: int,
    fps: float,
    size: Tuple[int, int] = (1280, 720),
    col_names: Optional[Sequence[str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    alpha: bool = False,
) -> Path:
    sig = np.asarray(signals).astype(float)
    if sig.ndim == 1:
        sig = sig[:, None]
    T, C = sig.shape
    names = (
        list(col_names)
        if (col_names and len(col_names) == C)
        else [f"ch{c}" for c in range(C)]
    )

    W = left + right + 1
    x = np.arange(-left, right + 1, dtype=float)

    width, height = size
    fig = _init_canvas(width, height, alpha)
    ax = fig[0, 0]

    y0, y1 = _compute_ylim(sig.reshape(-1), ylim)
    ax.camera.show_rect(left=-left, right=right, bottom=y0, top=y1)
    # Add vertical line at x=0
    vline_data = np.column_stack(
        [[0.0, 0.0], [y0, y1], [0.0, 0.0]]
    )  # Add z=0 for 3D coords
    ax.add_line(vline_data, thickness=1.5, alpha=0.7)

    lines = []
    for c in range(C):
        ywin = np.full(W, np.nan, dtype=float)
        line_data = np.column_stack([x, ywin, np.zeros(W)])  # Add z=0 for 3D coords
        lines.append(ax.add_line(line_data))
        ax.add_text(names[c], offset=(10, 10 + 18 * (c + 1), 0))

    proc, out_path = _ffmpeg_writer(out_path, width, height, fps, alpha)

    for t in range(T):
        s = max(0, t - left)
        e = min(T, t + right + 1)
        span = e - s
        for c in range(C):
            ywin = np.empty(W, dtype=float)
            ywin[:span] = sig[s:e, c]
            if span < W:
                ywin[span:] = np.nan
            # Update line data
            new_data = np.column_stack([x, ywin, np.zeros(W)])  # Add z=0 for 3D coords
            lines[c].data = new_data
        proc.stdin.write(_read_frame(fig, alpha))

    proc.stdin.close()
    proc.wait()
    return out_path


# ---------- 3) render grid of subplots (one channel per axis) ----------


def render_grid(
    signals: np.ndarray,  # shape [T, C]
    out_path: Path,
    left: int,
    right: int,
    fps: float,
    grid: Tuple[int, int],  # (rows, cols) must fit C
    base_size: Tuple[int, int] = (1280, 720),
    col_names: Optional[Sequence[str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    alpha: bool = False,
) -> Path:
    sig = np.asarray(signals).astype(float)
    if sig.ndim == 1:
        sig = sig[:, None]
    T, C = sig.shape
    rows, cols = grid
    if rows * cols < C:
        raise ValueError(f"grid {grid} too small for {C} channels.")
    names = (
        list(col_names)
        if (col_names and len(col_names) == C)
        else [f"ch{c}" for c in range(C)]
    )

    W = left + right + 1
    x = np.arange(-left, right + 1, dtype=float)

    # Scale total canvas height by rows to keep each subplot readable
    width, height = base_size
    per_row_h = max(240, height // max(1, rows))
    total_h = per_row_h * rows

    canvas = OffscreenRenderCanvas(
        size=(width, height), pixel_ratio=1.0, format="rgba-u8"
    )
    fig = fpl.Figure(size=(width, total_h), shape=(rows, cols), canvas=canvas)

    axes, lines = [], []
    idx = 0
    for r in range(rows):
        for cc in range(cols):
            if idx >= C:
                break
            ax = fig[r, cc]
            y0, y1 = _compute_ylim(sig[:, idx], ylim)
            ax.camera.show_rect(left=-left, right=right, bottom=y0, top=y1)
            # Add vertical line at x=0
            vline_data = np.column_stack(
                [[0.0, 0.0], [y0, y1], [0.0, 0.0]]
            )  # Add z=0 for 3D coords
            ax.add_line(vline_data, thickness=1.5, alpha=0.7)
            ax.add_text(names[idx], offset=(10, 10, 0))
            ywin = np.full(W, np.nan, dtype=float)
            line_data = np.column_stack([x, ywin, np.zeros(W)])  # Add z=0 for 3D coords
            line = ax.add_line(line_data)
            axes.append(ax)
            lines.append((idx, line))
            idx += 1

    proc, out_path = _ffmpeg_writer(out_path, width, total_h, fps, alpha)

    for t in range(T):
        s = max(0, t - left)
        e = min(T, t + right + 1)
        span = e - s
        for ch_idx, line in lines:
            ywin = np.empty(W, dtype=float)
            ywin[:span] = sig[s:e, ch_idx]
            if span < W:
                ywin[span:] = np.nan
            # Update line data
            new_data = np.column_stack([x, ywin, np.zeros(W)])  # Add z=0 for 3D coords
            line.data = new_data
        proc.stdin.write(_read_frame(fig, alpha))

    proc.stdin.close()
    proc.wait()
    return out_path


# %%
import os


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

    ylim = ylim or _compute_ylim(sig)

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

    # FPS to preserve original duration
    plot_fps = float(video_fps) * float(ratio)
    if plot_fps <= 0:
        raise ValueError("Computed plot_fps must be > 0 (check video_fps and ratio).")

    # Common window geometry
    window_len = (
        left + right + 1
    )  # (unused here directly, renderers compute their own buffers)

    # Global y-limits (stable visuals); renderers can still compute per-channel if you prefer
    global_ylim = _global_ylim(sig, ylim)

    # Heads-up for options we haven't implemented in renderers yet
    if show_legend and not combine_plots:
        warnings.warn(
            "show_legend is only meaningful for 'combine_plots=True'. Ignoring.",
            RuntimeWarning,
        )
    if show_values:
        warnings.warn(
            "show_values not yet implemented in renderers. Ignoring.", RuntimeWarning
        )

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
                ylim=global_ylim,  # or None if you want per-channel limits
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
            alpha=False,  # set True to get transparent .webm
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
        alpha=False,  # set True to get transparent .webm
    )
    return True


# %%
# test this out

generate_plot_videos(
    aligned_signal=aligned,
    ratio=1,
    output_dir=TEST_DATA_DIR / "output_plots",
    col_names=["track0", "track1"],
    grid=(1, 2),
)

# %% [markdown]
#
