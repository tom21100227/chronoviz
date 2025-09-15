import pandas as pd
from src.alignment import get_video_timeline, read_timeseries, align_signal_cfr
from src.plotting import generate_plot_videos
from src.combine import combine_videos
import pathlib
from pathlib import Path


def main():
    TEST_DATA_DIR = Path(
        pathlib.Path.home(), "PersonalProjects/chronoviz/test_data/single_mice"
    )
    video_path = TEST_DATA_DIR / "A01_20250820122555.predictions.slp.mp4"
    signal_path = TEST_DATA_DIR / "A01_20250820122555.csv"

    video_fps, n_frames, video_times = get_video_timeline(video_path)
    print(f"Video FPS: {video_fps}, n_frames: {n_frames}")
    df = read_timeseries(signal_path)

    # Pivot the dataframe to wide format
    df_pivot = df.pivot(index='frame', columns='roi_name', values='percentage_in_roi').reset_index()
    df_pivot = df_pivot.fillna(0)  # Fill NaNs for frames where a roi_name might be missing

    # Now, df_pivot.values will not include 'frame' if we select the right columns
    roi_columns = [col for col in df_pivot.columns if col not in ['frame', 'instance']]


    print(f"Signal columns: {roi_columns}")
    aligned_signal = align_signal_cfr(
        video_times=video_times,
        sig_values=df_pivot[roi_columns].values,
        ratio=1,
        mode="resample",
    )

    plot = generate_plot_videos(
        aligned_signal=aligned_signal,
        ratio=1.0,
        output_dir=TEST_DATA_DIR / "outputs",
        col_names=roi_columns,
        video_fps=video_fps,
        mode="grid",
        grid=(6, 1),
        ylim=(0, 100),
        plot_size=(320, 768),
        xlabel="Time (samples)",
        ylabel="Percentage in ROI",
    )

    combine_videos(
        video_path=video_path,
        plot_video_path=plot,
        output_path=TEST_DATA_DIR / "outputs" / "combined_output.mp4",
        ratio=1.0,
        position="right",
        overlay=False,
        cpu=True
    )


if __name__ == "__main__":
    main()
