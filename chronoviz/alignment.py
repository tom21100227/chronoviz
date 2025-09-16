"""
Module for aligning time series signals to video frame timelines.
"""

from __future__ import annotations

import av
import pathlib
import numpy as np
import pandas as pd
from pathlib import Path


def get_video_timeline(video_path: str | Path):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    fps = video_stream.average_rate  # assuming constant frame rate
    n_frames = video_stream.frames
    if n_frames is None:
        n_frames = sum(1 for _ in container.decode(video=0))
    timeline = np.arange(n_frames) / float(fps)
    return fps, n_frames, timeline


def read_timeseries(
    path: str | Path,
    key: str | None = None,
    *,
    time_col: str | None = None,
    time_key: str | None = None,
    time_units: str = "s",
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Read a time-series table from CSV/HDF5 and optionally extract a time vector.

    Parameters:
        - path: CSV or HDF5 file
        - key: dataset key for HDF5 when using pandas.read_hdf
        - time_col: optional column name in the table that contains time values
        - time_key: for raw HDF5 files (h5py), dataset key holding time values
        - time_units: units for the times (s, ms, us, ns)

    Returns:
        (df, times) where df is a pandas DataFrame of the signal table and
        times is a 1D ndarray in seconds or None if not found.
    """
    ext = pathlib.Path(path).suffix
    times: np.ndarray | None = None

    def _convert_units(arr: np.ndarray) -> np.ndarray:
        factor = {
            "s": 1.0,
            "ms": 1e-3,
            "us": 1e-6,
            "ns": 1e-9,
        }.get(time_units.lower(), None)
        if factor is None:
            raise ValueError(
                f"Unsupported time_units '{time_units}'. Use one of: s, ms, us, ns."
            )
        return arr.astype(float) * factor

    if ext == ".csv":
        df = pd.read_csv(path)
        # Auto-detect time col if not provided
        if time_col is None:
            for cand in ("time", "t"):
                if cand in df.columns:
                    time_col = cand
                    break
        if time_col is not None and time_col in df.columns:
            times = _convert_units(df[time_col].to_numpy())
    elif ext in [".h5", ".hdf5"]:
        # Try pandas first
        df = None
        if key is not None:
            try:
                df = pd.read_hdf(path, key=key)
            except TypeError:
                df = None
        if df is None:
            # Fall back to raw HDF5
            import h5py

            with h5py.File(path, "r") as f:
                if key is None:
                    raise ValueError(
                        "Key must be provided for HDF5 files when not using pandas-compatible stores."
                    )
                if key not in f:
                    raise KeyError(
                        f"Key '{key}' not found in HDF5 file. Available keys: {list(f.keys())}"
                    )
                data = f[key][:]
                df = pd.DataFrame(data)
                # time_key for raw datasets
                if time_key is not None and time_key in f:
                    times = _convert_units(np.asarray(f[time_key][:]))

        # If pandas path: try to pick time column
        if times is None and time_col is not None and time_col in df.columns:
            times = _convert_units(df[time_col].to_numpy())
        elif times is None:
            for cand in ("time", "t"):
                if cand in df.columns:
                    times = _convert_units(df[cand].to_numpy())
                    break
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df, times


def align_signal_cfr(
    video_times: np.ndarray,
    sig_values: np.ndarray,
    mode: str,
    ratio: float = 1,
    padding_mode: str = "edge",
    times: np.ndarray | None = None,
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
                    video_times,
                    sig_values[:, c],
                    mode,
                    ratio,
                    padding_mode,
                    times=times,
                    **kwargs,
                )
                for c in range(sig_values.shape[1])
            ],
            axis=1,
        )
    # Time-aware alignment when explicit sample times are provided
    if times is not None:
        times = np.asarray(times, dtype=float).ravel()
        y = np.asarray(sig_values, dtype=float).ravel()
        if times.shape[0] != y.shape[0]:
            raise ValueError(
                f"Length mismatch: times ({times.shape[0]}) vs samples ({y.shape[0]})."
            )
        # Ensure monotonic increasing times for interpolation
        if not np.all(np.diff(times) >= 0):
            order = np.argsort(times)
            times = times[order]
            y = y[order]
        # Use edge padding behavior via left/right for out-of-range
        return np.interp(video_times, times, y, left=y[0], right=y[-1])

    # Ratio-based alignment (legacy behavior)
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
