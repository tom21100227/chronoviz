"""
Benchmark tests for ChronoViz using synthetic data.

This module provides comprehensive benchmarking tests for the chronoviz library
using synthetic data generators to test performance across various signal
types, data sizes, and rendering modes.
"""

import numpy as np
import pandas as pd
import tempfile
import time
import pytest
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional, Tuple, List
import shutil

# Import the modules we want to benchmark
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.plotting import generate_plot_videos
from src.alignment import align_signal_cfr
from src.backends.mpl import render_one_channel, render_all_channels, render_grid


# Synthetic data generators
def generate_sine_waves(
    n_samples: int,
    n_channels: int = 1,
    frequencies: Optional[List[float]] = None,
    sampling_rate: float = 1000.0,
    amplitude: float = 1.0,
    noise_level: float = 0.0,
    seed: int = 42
) -> np.ndarray:
    """Generate synthetic sine wave data."""
    np.random.seed(seed)
    
    if frequencies is None:
        frequencies = [1.0 + i * 0.5 for i in range(n_channels)]
    
    t = np.linspace(0, n_samples / sampling_rate, n_samples)
    signals = np.zeros((n_samples, n_channels))
    
    for i, freq in enumerate(frequencies[:n_channels]):
        signals[:, i] = amplitude * np.sin(2 * np.pi * freq * t)
        if noise_level > 0:
            signals[:, i] += np.random.normal(0, noise_level, n_samples)
    
    return signals


def generate_complex_waveform(
    n_samples: int,
    n_channels: int = 1,
    sampling_rate: float = 1000.0,
    seed: int = 42
) -> np.ndarray:
    """Generate complex synthetic waveforms with multiple frequency components."""
    np.random.seed(seed)
    
    t = np.linspace(0, n_samples / sampling_rate, n_samples)
    signals = np.zeros((n_samples, n_channels))
    
    for i in range(n_channels):
        # Base sine wave
        signals[:, i] = np.sin(2 * np.pi * (1 + i * 0.3) * t)
        # Add harmonics
        signals[:, i] += 0.5 * np.sin(2 * np.pi * (3 + i * 0.5) * t)
        signals[:, i] += 0.25 * np.sin(2 * np.pi * (5 + i * 0.7) * t)
        # Add some envelope modulation
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
        signals[:, i] *= envelope
        # Add noise
        signals[:, i] += np.random.normal(0, 0.1, n_samples)
        
        # Add some spikes/artifacts randomly
        spike_indices = np.random.choice(n_samples, size=n_samples//100, replace=False)
        signals[spike_indices, i] += np.random.normal(0, 2, len(spike_indices))
    
    return signals


def generate_step_function(
    n_samples: int,
    n_channels: int = 1,
    n_steps: int = 10,
    seed: int = 42
) -> np.ndarray:
    """Generate step function data (good for testing edge cases)."""
    np.random.seed(seed)
    
    signals = np.zeros((n_samples, n_channels))
    
    for i in range(n_channels):
        step_positions = np.sort(np.random.choice(n_samples, size=n_steps, replace=False))
        step_values = np.random.uniform(-5, 5, n_steps + 1)
        
        current_value = step_values[0]
        current_pos = 0
        
        for j, next_pos in enumerate(step_positions):
            signals[current_pos:next_pos, i] = current_value
            current_value = step_values[j + 1]
            current_pos = next_pos
        
        signals[current_pos:, i] = current_value
    
    return signals


# Benchmark fixtures and utilities
@contextmanager
def temp_output_dir():
    """Context manager for temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextmanager
def timer() -> Generator[dict, None, None]:
    """Context manager to time execution."""
    timing = {}
    start = time.perf_counter()
    yield timing
    timing['elapsed'] = time.perf_counter() - start


# Benchmark test classes
class TestSyntheticDataGeneration:
    """Test synthetic data generators for correctness and timing."""
    
    def test_sine_wave_generation(self):
        """Test sine wave generation."""
        with timer() as t:
            data = generate_sine_waves(10000, n_channels=4)
        
        assert data.shape == (10000, 4)
        assert not np.any(np.isnan(data))
        assert np.all(np.abs(data) <= 1.1)  # Within expected amplitude range
        print(f"Generated 10k samples x 4 channels sine waves in {t['elapsed']:.4f}s")
    
    def test_complex_waveform_generation(self):
        """Test complex waveform generation."""
        with timer() as t:
            data = generate_complex_waveform(50000, n_channels=8)
        
        assert data.shape == (50000, 8)
        assert not np.any(np.isnan(data))
        print(f"Generated 50k samples x 8 channels complex waveforms in {t['elapsed']:.4f}s")
    
    def test_step_function_generation(self):
        """Test step function generation."""
        with timer() as t:
            data = generate_step_function(20000, n_channels=2)
        
        assert data.shape == (20000, 2)
        assert not np.any(np.isnan(data))
        print(f"Generated 20k samples x 2 channels step functions in {t['elapsed']:.4f}s")


class TestPlottingPerformance:
    """Benchmark plotting performance with different data sizes and modes."""
    
    @pytest.mark.parametrize("n_samples", [1000, 5000, 10000, 25000])
    @pytest.mark.parametrize("n_channels", [1, 2, 4, 8])
    @pytest.mark.parametrize("mode", ["combine", "grid", "separate"])
    def test_plotting_performance_matrix(self, n_samples, n_channels, mode):
        """Comprehensive performance matrix test."""
        # Skip expensive combinations for CI
        if n_samples > 10000 and n_channels > 4:
            pytest.skip("Skipping expensive test combination")
        
        data = generate_sine_waves(n_samples, n_channels)
        
        with temp_output_dir() as output_dir:
            with timer() as t:
                result_path = generate_plot_videos(
                    aligned_signal=data,
                    ratio=1.0,
                    output_dir=output_dir / "test_output",
                    mode=mode,
                    grid=(2, max(1, n_channels // 2)) if mode == "grid" else None,
                    col_names=[f"ch{i}" for i in range(n_channels)],
                    video_fps=30.0,
                    plot_size=(640, 480),
                    left=50,
                    right=50
                )
            
            # Verify output was created
            assert result_path.exists()
            
            # Print performance metrics
            samples_per_sec = n_samples / t['elapsed']
            print(f"{mode} mode: {n_samples} samples x {n_channels} channels -> "
                  f"{t['elapsed']:.3f}s ({samples_per_sec:.0f} samples/sec)")
    
    def test_window_size_impact(self):
        """Test how window size (left/right) affects performance."""
        data = generate_sine_waves(10000, n_channels=2)
        
        window_sizes = [(25, 25), (50, 50), (100, 100), (250, 250)]
        
        for left, right in window_sizes:
            with temp_output_dir() as output_dir:
                with timer() as t:
                    generate_plot_videos(
                        aligned_signal=data,
                        ratio=1.0,
                        output_dir=output_dir / "test_output",
                        mode="combine",
                        video_fps=30.0,
                        plot_size=(640, 480),
                        left=left,
                        right=right
                    )
                
                print(f"Window size {left}+{right}: {t['elapsed']:.3f}s")
    
    def test_plot_size_impact(self):
        """Test how plot resolution affects performance."""
        data = generate_sine_waves(5000, n_channels=2)
        
        plot_sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
        
        for width, height in plot_sizes:
            with temp_output_dir() as output_dir:
                with timer() as t:
                    generate_plot_videos(
                        aligned_signal=data,
                        ratio=1.0,
                        output_dir=output_dir / "test_output",
                        mode="combine",
                        video_fps=30.0,
                        plot_size=(width, height),
                        left=50,
                        right=50
                    )
                
                pixels = width * height
                print(f"Resolution {width}x{height} ({pixels/1000:.0f}k pixels): {t['elapsed']:.3f}s")


class TestDataComplexityImpact:
    """Test how data complexity affects rendering performance."""
    
    def test_signal_complexity_comparison(self):
        """Compare performance across different signal types."""
        n_samples, n_channels = 10000, 3
        
        # Test different data types
        test_cases = [
            ("sine_waves", lambda: generate_sine_waves(n_samples, n_channels)),
            ("complex_waveform", lambda: generate_complex_waveform(n_samples, n_channels)),
            ("step_function", lambda: generate_step_function(n_samples, n_channels)),
            ("noisy_sine", lambda: generate_sine_waves(n_samples, n_channels, noise_level=0.5)),
        ]
        
        for name, generator in test_cases:
            data = generator()
            
            with temp_output_dir() as output_dir:
                with timer() as t:
                    generate_plot_videos(
                        aligned_signal=data,
                        ratio=1.0,
                        output_dir=output_dir / "test_output",
                        mode="grid",
                        grid=(3, 1),
                        video_fps=30.0,
                        plot_size=(640, 480),
                        left=100,
                        right=100
                    )
                
                print(f"{name}: {t['elapsed']:.3f}s")
    
    def test_high_frequency_content(self):
        """Test performance with high-frequency content."""
        n_samples = 10000
        frequencies = [1, 10, 50, 100]  # Different frequency content
        
        for freq in frequencies:
            data = generate_sine_waves(n_samples, n_channels=1, frequencies=[freq])
            
            with temp_output_dir() as output_dir:
                with timer() as t:
                    generate_plot_videos(
                        aligned_signal=data,
                        ratio=1.0,
                        output_dir=output_dir / "test_output",
                        mode="combine",
                        video_fps=30.0,
                        plot_size=(640, 480),
                        left=50,
                        right=50
                    )
                
                print(f"Frequency {freq} Hz: {t['elapsed']:.3f}s")


class TestAlignmentPerformance:
    """Benchmark signal alignment performance."""
    
    def test_alignment_modes(self):
        """Test different alignment modes."""
        n_samples = 20000
        data = generate_sine_waves(n_samples, n_channels=2)
        
        # Create fake video times
        video_times = np.linspace(0, n_samples / 1000.0, n_samples // 2)
        
        modes = ["resample", "interpolate"]
        
        for mode in modes:
            with timer() as t:
                aligned = align_signal_cfr(
                    video_times=video_times,
                    sig_values=data,
                    ratio=1.0,
                    mode=mode
                )
            
            print(f"Alignment mode '{mode}': {t['elapsed']:.3f}s")
            assert aligned.shape[0] == len(video_times)
    
    def test_alignment_scaling(self):
        """Test alignment performance with different data sizes."""
        base_samples = [5000, 10000, 20000, 50000]
        
        for n_samples in base_samples:
            data = generate_sine_waves(n_samples, n_channels=4)
            video_times = np.linspace(0, n_samples / 1000.0, n_samples // 2)
            
            with timer() as t:
                aligned = align_signal_cfr(
                    video_times=video_times,
                    sig_values=data,
                    ratio=1.0,
                    mode="resample"
                )
            
            samples_per_sec = n_samples / t['elapsed']
            print(f"Aligned {n_samples} samples in {t['elapsed']:.3f}s "
                  f"({samples_per_sec:.0f} samples/sec)")


class TestMemoryUsage:
    """Tests focused on memory usage patterns."""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # This should be one of the larger tests
        n_samples = 100000  # 100k samples
        n_channels = 8
        
        print(f"Testing large dataset: {n_samples} samples x {n_channels} channels")
        
        # Generate data in chunks to avoid memory issues during generation
        data = generate_sine_waves(n_samples, n_channels)
        data_size_mb = data.nbytes / (1024 * 1024)
        print(f"Data size: {data_size_mb:.1f} MB")
        
        with temp_output_dir() as output_dir:
            with timer() as t:
                result_path = generate_plot_videos(
                    aligned_signal=data,
                    ratio=1.0,
                    output_dir=output_dir / "test_output",
                    mode="grid",
                    grid=(4, 2),
                    video_fps=30.0,
                    plot_size=(1280, 720),
                    left=100,
                    right=100
                )
            
            throughput = data_size_mb / t['elapsed']
            print(f"Large dataset processing: {t['elapsed']:.2f}s ({throughput:.1f} MB/s)")


# Convenience function to run all benchmarks
def run_benchmarks():
    """Run all benchmark tests and collect results."""
    print("Running ChronoViz Synthetic Data Benchmarks")
    print("=" * 50)
    
    # You can run this manually or integrate with pytest
    test_classes = [
        TestSyntheticDataGeneration,
        TestPlottingPerformance,
        TestDataComplexityImpact,
        TestAlignmentPerformance,
        TestMemoryUsage,
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * len(test_class.__name__))
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_') and not method_name.endswith('_matrix'):
                print(f"  Running {method_name}...")
                try:
                    getattr(instance, method_name)()
                except Exception as e:
                    print(f"    ERROR: {e}")


if __name__ == "__main__":
    run_benchmarks()
