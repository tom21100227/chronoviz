import av, pathlib, numpy as np, pandas as pd
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
