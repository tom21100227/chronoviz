"""
Performance profiler for ChronoViz to identify bottlenecks.

This module provides detailed timing information for different parts
of the rendering pipeline to help optimize performance.
"""

import time
import cProfile
import pstats
import io
from pathlib import Path
import sys
import tempfile
import shutil
from contextlib import contextmanager

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.plotting import generate_plot_videos
from src.alignment import align_signal_cfr

# Import data generators
from test_data_generators import generate_sine_waves, generate_complex_waveform


@contextmanager
def temp_output_dir():
    """Context manager for temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class PerformanceProfiler:
    """Profile ChronoViz performance and identify bottlenecks."""

    def __init__(self):
        self.results = {}

    def profile_plotting_pipeline(self, n_samples=5000, n_channels=4):
        """Profile the complete plotting pipeline."""
        print(
            f"Profiling plotting pipeline: {n_samples} samples, {n_channels} channels"
        )
        print("=" * 60)

        # Generate test data
        data = generate_sine_waves(n_samples, n_channels)

        with temp_output_dir() as output_dir:
            # Profile the complete pipeline
            profiler = cProfile.Profile()
            profiler.enable()

            start_time = time.perf_counter()

            result_path = generate_plot_videos(
                aligned_signal=data,
                ratio=1.0,
                output_dir=output_dir / "profile_output",
                mode="grid",
                grid=(2, 2),
                col_names=[f"Channel {i+1}" for i in range(n_channels)],
                video_fps=30.0,
                plot_size=(640, 480),
                left=50,
                right=50,
            )

            total_time = time.perf_counter() - start_time

            profiler.disable()

            # Analyze results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats("cumulative")
            ps.print_stats(20)  # Top 20 functions

            profile_output = s.getvalue()

            print(f"Total execution time: {total_time:.3f}s")
            print(f"Throughput: {n_samples/total_time:.0f} samples/sec")
            print("\nTop functions by cumulative time:")
            print("-" * 40)

            # Parse and display key statistics
            lines = profile_output.split("\n")
            for line in lines[5:25]:  # Skip header, show top 20
                if line.strip():
                    print(line)

            return {
                "total_time": total_time,
                "throughput": n_samples / total_time,
                "profile_data": profile_output,
            }

    def profile_alignment(self, n_samples=10000):
        """Profile signal alignment performance."""
        print(f"\nProfiling alignment: {n_samples} samples")
        print("=" * 40)

        # Generate test data
        data = generate_sine_waves(n_samples, 4)
        video_times = np.linspace(0, n_samples / 1000.0, n_samples // 2)

        # Profile alignment
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.perf_counter()

        aligned = align_signal_cfr(
            video_times=video_times, sig_values=data, ratio=1.0, mode="resample"
        )

        alignment_time = time.perf_counter() - start_time
        profiler.disable()

        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats("cumulative")
        ps.print_stats(10)  # Top 10 functions

        print(f"Alignment time: {alignment_time:.3f}s")
        print(f"Throughput: {n_samples/alignment_time:.0f} samples/sec")
        print("\nTop alignment functions:")
        print("-" * 30)

        lines = s.getvalue().split("\n")
        for line in lines[5:15]:  # Show top 10
            if line.strip():
                print(line)

        return {
            "alignment_time": alignment_time,
            "throughput": n_samples / alignment_time,
        }

    def memory_usage_analysis(self, n_samples=20000):
        """Analyze memory usage patterns."""
        print(f"\nMemory usage analysis: {n_samples} samples")
        print("=" * 40)

        try:
            import psutil

            process = psutil.Process()

            # Baseline memory
            baseline_mb = process.memory_info().rss / 1024 / 1024
            print(f"Baseline memory: {baseline_mb:.1f} MB")

            # Memory during data generation
            data = generate_complex_waveform(n_samples, 8)
            data_gen_mb = process.memory_info().rss / 1024 / 1024
            data_size_mb = data.nbytes / 1024 / 1024
            print(
                f"After data generation: {data_gen_mb:.1f} MB (+{data_gen_mb-baseline_mb:.1f} MB)"
            )
            print(f"Data size: {data_size_mb:.1f} MB")

            # Memory during plotting
            with temp_output_dir() as output_dir:
                start_plotting_mb = process.memory_info().rss / 1024 / 1024

                result_path = generate_plot_videos(
                    aligned_signal=data,
                    ratio=1.0,
                    output_dir=output_dir / "memory_test",
                    mode="grid",
                    grid=(4, 2),
                    video_fps=30.0,
                    plot_size=(640, 480),
                    left=100,
                    right=100,
                )

                peak_mb = process.memory_info().rss / 1024 / 1024
                print(
                    f"Peak during plotting: {peak_mb:.1f} MB (+{peak_mb-baseline_mb:.1f} MB)"
                )
                print(
                    f"Memory efficiency: {data_size_mb/(peak_mb-baseline_mb):.2f} (data_size/memory_used)"
                )

            # Memory after cleanup
            del data
            final_mb = process.memory_info().rss / 1024 / 1024
            print(f"After cleanup: {final_mb:.1f} MB")

        except ImportError:
            print("psutil not available for memory analysis")
            print("Install with: pip install psutil")

    def compare_data_types(self):
        """Compare performance across different synthetic data types."""
        print("\nComparing data type performance")
        print("=" * 40)

        n_samples, n_channels = 8000, 3

        data_types = [
            ("Sine waves", lambda: generate_sine_waves(n_samples, n_channels)),
            (
                "Complex waveform",
                lambda: generate_complex_waveform(n_samples, n_channels),
            ),
            (
                "Noisy sine",
                lambda: generate_sine_waves(n_samples, n_channels, noise_level=0.3),
            ),
        ]

        results = {}

        for name, generator in data_types:
            print(f"Testing {name}...")

            data = generator()

            with temp_output_dir() as output_dir:
                start_time = time.perf_counter()

                generate_plot_videos(
                    aligned_signal=data,
                    ratio=1.0,
                    output_dir=output_dir / "compare_test",
                    mode="combine",
                    video_fps=30.0,
                    plot_size=(640, 480),
                    left=50,
                    right=50,
                )

                elapsed = time.perf_counter() - start_time
                results[name] = elapsed

                print(f"  {name}: {elapsed:.3f}s ({n_samples/elapsed:.0f} samples/sec)")

        # Find fastest/slowest
        fastest = min(results.keys(), key=lambda k: results[k])
        slowest = max(results.keys(), key=lambda k: results[k])

        print(f"\nFastest: {fastest}")
        print(f"Slowest: {slowest}")
        print(f"Speed difference: {results[slowest]/results[fastest]:.1f}x")

        return results


def run_full_profile():
    """Run comprehensive performance profiling."""
    profiler = PerformanceProfiler()

    print("ChronoViz Performance Profiler")
    print("=" * 50)

    try:
        # Profile plotting pipeline
        plotting_results = profiler.profile_plotting_pipeline(5000, 4)

        # Profile alignment
        alignment_results = profiler.profile_alignment(10000)

        # Memory analysis
        profiler.memory_usage_analysis(15000)

        # Data type comparison
        comparison_results = profiler.compare_data_types()

        print("\n" + "=" * 50)
        print("PROFILING SUMMARY")
        print("=" * 50)
        print(f"Plotting throughput: {plotting_results['throughput']:.0f} samples/sec")
        print(
            f"Alignment throughput: {alignment_results['throughput']:.0f} samples/sec"
        )
        print("Profiling completed successfully!")

    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import numpy as np

    run_full_profile()
