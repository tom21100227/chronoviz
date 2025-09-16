"""
Generate sample videos using synthetic data for visual comparison.

This script creates several example videos using different synthetic
data types to help visualize the different test scenarios.
"""

import numpy as np
from pathlib import Path
import sys
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from chronoviz.plotting import generate_plot_videos
from test_data_generators import (
    generate_sine_waves,
    generate_complex_waveform,
    generate_step_function,
)


def create_sample_videos():
    """Create sample videos for different synthetic data types."""

    # Parameters
    n_samples = 3000  # Smaller for quick generation
    n_channels = 4
    output_base = Path(__file__).parent / "sample_outputs"

    # Clean and create output directory
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir()

    print("Generating sample videos with synthetic data...")
    print("=" * 50)

    # 1. Clean sine waves
    print("1. Generating clean sine waves...")
    sine_data = generate_sine_waves(n_samples, n_channels, frequencies=[1, 2, 3, 5])

    sine_path = generate_plot_videos(
        aligned_signal=sine_data,
        ratio=1.0,
        output_dir=output_base / "sine_waves",
        mode="grid",
        grid=(2, 2),
        col_names=["1 Hz", "2 Hz", "3 Hz", "5 Hz"],
        video_fps=30.0,
        plot_size=(800, 600),
        left=100,
        right=100,
        ylim=(-1.5, 1.5),
        xlabel="Sample",
        ylabel="Amplitude",
    )
    print(f"   Created: {sine_path}")

    # 2. Noisy sine waves
    print("2. Generating noisy sine waves...")
    noisy_data = generate_sine_waves(
        n_samples, n_channels, frequencies=[1, 2, 3, 5], noise_level=0.3
    )

    noisy_path = generate_plot_videos(
        aligned_signal=noisy_data,
        ratio=1.0,
        output_dir=output_base / "noisy_sine_waves",
        mode="grid",
        grid=(2, 2),
        col_names=["1 Hz + noise", "2 Hz + noise", "3 Hz + noise", "5 Hz + noise"],
        video_fps=30.0,
        plot_size=(800, 600),
        left=100,
        right=100,
        ylim=(-2.0, 2.0),
        xlabel="Sample",
        ylabel="Amplitude",
    )
    print(f"   Created: {noisy_path}")

    # 3. Complex waveforms
    print("3. Generating complex waveforms...")
    complex_data = generate_complex_waveform(n_samples, n_channels)

    complex_path = generate_plot_videos(
        aligned_signal=complex_data,
        ratio=1.0,
        output_dir=output_base / "complex_waveforms",
        mode="grid",
        grid=(2, 2),
        col_names=["Complex A", "Complex B", "Complex C", "Complex D"],
        video_fps=30.0,
        plot_size=(800, 600),
        left=100,
        right=100,
        xlabel="Sample",
        ylabel="Amplitude",
    )
    print(f"   Created: {complex_path}")

    # 4. Step functions
    print("4. Generating step functions...")
    step_data = generate_step_function(n_samples, n_channels, n_steps=8)

    step_path = generate_plot_videos(
        aligned_signal=step_data,
        ratio=1.0,
        output_dir=output_base / "step_functions",
        mode="grid",
        grid=(2, 2),
        col_names=["Steps A", "Steps B", "Steps C", "Steps D"],
        video_fps=30.0,
        plot_size=(800, 600),
        left=100,
        right=100,
        xlabel="Sample",
        ylabel="Value",
    )
    print(f"   Created: {step_path}")

    # 5. Combined view (all data types in one video)
    print("5. Generating combined comparison...")

    # Take one channel from each data type
    combined_data = np.column_stack(
        [sine_data[:, 0], noisy_data[:, 0], complex_data[:, 0], step_data[:, 0]]
    )

    combined_path = generate_plot_videos(
        aligned_signal=combined_data,
        ratio=1.0,
        output_dir=output_base / "comparison",
        mode="grid",
        grid=(2, 2),
        col_names=["Clean Sine", "Noisy Sine", "Complex", "Step Function"],
        video_fps=30.0,
        plot_size=(800, 600),
        left=100,
        right=100,
        xlabel="Sample",
        ylabel="Amplitude",
    )
    print(f"   Created: {combined_path}")

    # 6. Single combined plot version
    print("6. Generating overlaid comparison...")

    overlaid_path = generate_plot_videos(
        aligned_signal=combined_data,
        ratio=1.0,
        output_dir=output_base / "overlaid",
        mode="combine",
        col_names=["Clean Sine", "Noisy Sine", "Complex", "Step Function"],
        video_fps=30.0,
        plot_size=(800, 600),
        left=100,
        right=100,
        show_legend=True,
        xlabel="Sample",
        ylabel="Amplitude",
    )
    print(f"   Created: {overlaid_path}")

    print(f"\nAll sample videos created in: {output_base}")
    print("\nGenerated videos demonstrate:")
    print("- Clean synthetic sine waves")
    print("- Noisy signals")
    print("- Complex multi-harmonic waveforms")
    print("- Step functions with sharp transitions")
    print("- Side-by-side comparisons")
    print("- Overlaid multi-channel plots")


def create_performance_demo():
    """Create videos to demonstrate performance characteristics."""

    output_base = Path(__file__).parent / "performance_demo"
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir()

    print("\nGenerating performance demonstration videos...")
    print("=" * 50)

    # Different data sizes
    sizes = [1000, 5000, 10000]

    for size in sizes:
        print(f"Generating {size} sample video...")

        data = generate_sine_waves(size, 2, frequencies=[1, 3])

        import time

        start = time.perf_counter()

        path = generate_plot_videos(
            aligned_signal=data,
            ratio=1.0,
            output_dir=output_base / f"size_{size}",
            mode="combine",
            col_names=[f"1 Hz ({size} samples)", f"3 Hz ({size} samples)"],
            video_fps=30.0,
            plot_size=(640, 480),
            left=50,
            right=50,
            show_legend=True,
        )

        elapsed = time.perf_counter() - start
        throughput = size / elapsed

        print(f"   {size} samples: {elapsed:.2f}s ({throughput:.0f} samples/sec)")

    print(f"\nPerformance demo videos created in: {output_base}")


if __name__ == "__main__":
    print("ChronoViz Synthetic Data Sample Generator")
    print("=" * 50)

    try:
        create_sample_videos()
        create_performance_demo()
        print("\nSample generation completed successfully!")

    except Exception as e:
        print(f"Sample generation failed: {e}")
        import traceback

        traceback.print_exc()
