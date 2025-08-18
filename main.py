from src.alignment import get_video_timeline, read_timeseries, align_signal_cfr
from src.plotting import generate_plot_videos
from src.combine import combine_videos
import pathlib
from pathlib import Path


def main():
    TEST_DATA_DIR = Path(
        pathlib.Path.home(), "PersonalProjects/chronoviz/test_data/slp"
    )
    video_path = TEST_DATA_DIR / "03.mp4"
    signal_path = TEST_DATA_DIR / "03.csv"

    video_fps, n_frames, video_times = get_video_timeline(video_path)
    print(f"Video FPS: {video_fps}, n_frames: {n_frames}")
    df = read_timeseries(signal_path)
    print(f"Signal columns: {df.columns.tolist()}")
    aligned_signal = align_signal_cfr(
        video_times=video_times,
        sig_values=df.values,
        ratio=1,
        mode="resample",
    )

    plot = generate_plot_videos(
        aligned_signal=aligned_signal,
        ratio=1.0,
        output_dir=TEST_DATA_DIR / "outputs",
        col_names=["track0", "track1"],
        video_fps=video_fps,
        mode="grid",
        grid=(2, 1),
        ylim=(0, 20),
        plot_size=(320, 512),
        xlabel="Time (samples)",
        ylabel="Speed (cm/s)",
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
