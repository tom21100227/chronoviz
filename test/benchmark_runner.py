"""
Simple benchmark runner for ChronoViz performance testing.

This provides a quick way to run performance benchmarks and collect
timing data for different aspects of the library.
"""

import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.plotting import generate_plot_videos
from src.alignment import align_signal_cfr


def generate_sine_waves(n_samples, n_channels=1, frequencies=None, noise_level=0.0):
    """Generate synthetic sine wave data for testing."""
    if frequencies is None:
        frequencies = [1.0 + i * 0.5 for i in range(n_channels)]

    t = np.linspace(0, n_samples / 1000.0, n_samples)
    signals = np.zeros((n_samples, n_channels))

    for i, freq in enumerate(frequencies[:n_channels]):
        signals[:, i] = np.sin(2 * np.pi * freq * t)
        if noise_level > 0:
            np.random.seed(42 + i)
            signals[:, i] += np.random.normal(0, noise_level, n_samples)

    return signals


@contextmanager
def temp_output_dir():
    """Context manager for temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_plotting_modes():
    """Benchmark different plotting modes with synthetic data."""
    print("Benchmarking Plotting Modes")
    print("=" * 40)

    # Test parameters
    n_samples = 10000
    n_channels = 4
    data = generate_sine_waves(n_samples, n_channels)

    modes = ["combine", "grid", "separate"]
    results = {}

    for mode in modes:
        print(f"Testing {mode} mode...")

        with temp_output_dir() as output_dir:
            start_time = time.perf_counter()

            result_path = generate_plot_videos(
                aligned_signal=data,
                ratio=1.0,
                output_dir=output_dir / "test_output",
                mode=mode,
                grid=(2, 2) if mode == "grid" else None,
                col_names=[f"Channel {i+1}" for i in range(n_channels)],
                video_fps=30.0,
                plot_size=(640, 480),
                left=50,
                right=50,
            )

            elapsed = time.perf_counter() - start_time
            results[mode] = elapsed

            print(f"  {mode}: {elapsed:.3f}s")

            # Check output exists
            if result_path.exists():
                if mode == "separate":
                    video_files = list(result_path.glob("*.mp4"))
                    print(f"  Generated {len(video_files)} separate videos")
                else:
                    print(f"  Generated video: {result_path}")

    print(f"\nFastest mode: {min(results.keys(), key=lambda k: results[k])}")
    return results


def benchmark_data_sizes():
    """Benchmark performance with different data sizes."""
    print("\nBenchmarking Data Sizes")
    print("=" * 40)

    sizes = [1000, 5000, 10000, 25000]
    n_channels = 2
    results = {}

    for n_samples in sizes:
        print(f"Testing {n_samples} samples...")

        data = generate_sine_waves(n_samples, n_channels)

        with temp_output_dir() as output_dir:
            start_time = time.perf_counter()

            generate_plot_videos(
                aligned_signal=data,
                ratio=1.0,
                output_dir=output_dir / "test_output",
                mode="combine",
                video_fps=30.0,
                plot_size=(640, 480),
                left=50,
                right=50,
            )

            elapsed = time.perf_counter() - start_time
            results[n_samples] = elapsed

            samples_per_sec = n_samples / elapsed
            print(
                f"  {n_samples} samples: {elapsed:.3f}s ({samples_per_sec:.0f} samples/sec)"
            )

    return results


def benchmark_channel_scaling():
    """Benchmark performance with different numbers of channels."""
    print("\nBenchmarking Channel Scaling")
    print("=" * 40)

    n_samples = 8000
    channel_counts = [1, 2, 4, 8, 16]
    results = {}

    for n_channels in channel_counts:
        print(f"Testing {n_channels} channels...")

        data = generate_sine_waves(n_samples, n_channels)

        with temp_output_dir() as output_dir:
            start_time = time.perf_counter()

            # Use grid mode for multi-channel
            if n_channels == 1:
                mode = "combine"
                grid = None
            else:
                mode = "grid"
                # Try to make a reasonable grid
                cols = min(4, n_channels)
                rows = (n_channels + cols - 1) // cols
                grid = (rows, cols)

            generate_plot_videos(
                aligned_signal=data,
                ratio=1.0,
                output_dir=output_dir / "test_output",
                mode=mode,
                grid=grid,
                video_fps=30.0,
                plot_size=(640, 480),
                left=50,
                right=50,
            )

            elapsed = time.perf_counter() - start_time
            results[n_channels] = elapsed

            channels_per_sec = n_channels / elapsed
            print(
                f"  {n_channels} channels: {elapsed:.3f}s ({channels_per_sec:.1f} channels/sec)"
            )

    return results


def benchmark_resolution_impact():
    """Benchmark performance with different video resolutions."""
    print("\nBenchmarking Resolution Impact")
    print("=" * 40)

    n_samples = 5000
    n_channels = 2
    data = generate_sine_waves(n_samples, n_channels)

    resolutions = [
        (320, 240),  # Low
        (640, 480),  # SD
        (1280, 720),  # HD
        (1920, 1080),  # Full HD
    ]

    results = {}

    for width, height in resolutions:
        print(f"Testing {width}x{height}...")

        with temp_output_dir() as output_dir:
            start_time = time.perf_counter()

            generate_plot_videos(
                aligned_signal=data,
                ratio=1.0,
                output_dir=output_dir / "test_output",
                mode="combine",
                video_fps=30.0,
                plot_size=(width, height),
                left=50,
                right=50,
            )

            elapsed = time.perf_counter() - start_time
            results[(width, height)] = elapsed

            pixels = width * height
            pixels_per_sec = pixels / elapsed
            print(
                f"  {width}x{height}: {elapsed:.3f}s ({pixels_per_sec/1000:.0f}k pixels/sec)"
            )

    return results


def benchmark_alignment():
    """Benchmark signal alignment performance."""
    print("\nBenchmarking Signal Alignment")
    print("=" * 40)

    sizes = [5000, 10000, 25000, 50000]
    n_channels = 4
    results = {}

    for n_samples in sizes:
        print(f"Testing alignment with {n_samples} samples...")

        data = generate_sine_waves(n_samples, n_channels)
        # Create video times (simulate different frame rate)
        video_times = np.linspace(0, n_samples / 1000.0, n_samples // 2)

        start_time = time.perf_counter()

        aligned = align_signal_cfr(
            video_times=video_times, sig_values=data, ratio=1.0, mode="resample"
        )

        elapsed = time.perf_counter() - start_time
        results[n_samples] = elapsed

        samples_per_sec = n_samples / elapsed
        print(
            f"  {n_samples} samples: {elapsed:.3f}s ({samples_per_sec:.0f} samples/sec)"
        )

    return results


def run_comprehensive_benchmark():
    """Run all benchmarks and summarize results."""
    print("ChronoViz Performance Benchmark Suite")
    print("Using Synthetic Sine Wave Data")
    print("=" * 50)

    try:
        # Run all benchmark categories
        plotting_results = benchmark_plotting_modes()
        size_results = benchmark_data_sizes()
        channel_results = benchmark_channel_scaling()
        resolution_results = benchmark_resolution_impact()
        alignment_results = benchmark_alignment()

        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)

        print(
            f"\nFastest plotting mode: {min(plotting_results.keys(), key=lambda k: plotting_results[k])}"
        )
        print(
            f"Peak throughput: {max(size_results.keys())/min(size_results.values()):.0f} samples/sec"
        )
        print(
            f"Best channel efficiency: {max(channel_results.keys())/min(channel_results.values()):.1f} channels/sec"
        )

        print(
            f"\nAlignment performance: {max(alignment_results.keys())/min(alignment_results.values()):.0f} samples/sec peak"
        )

        print("\nBenchmark completed successfully!")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_benchmark()
