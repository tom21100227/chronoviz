"""
Unit tests for synthetic data generators and basic functionality.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def generate_sine_waves(
    n_samples: int,
    n_channels: int = 1,
    frequencies: Optional[List[float]] = None,
    sampling_rate: float = 1000.0,
    amplitude: float = 1.0,
    noise_level: float = 0.0,
    seed: int = 42,
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
    n_samples: int, n_channels: int = 1, sampling_rate: float = 1000.0, seed: int = 42
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
        spike_indices = np.random.choice(
            n_samples, size=n_samples // 100, replace=False
        )
        signals[spike_indices, i] += np.random.normal(0, 2, len(spike_indices))

    return signals


def generate_step_function(
    n_samples: int, n_channels: int = 1, n_steps: int = 10, seed: int = 42
) -> np.ndarray:
    """Generate step function data (good for testing edge cases)."""
    np.random.seed(seed)

    signals = np.zeros((n_samples, n_channels))

    for i in range(n_channels):
        step_positions = np.sort(
            np.random.choice(n_samples, size=n_steps, replace=False)
        )
        step_values = np.random.uniform(-5, 5, n_steps + 1)

        current_value = step_values[0]
        current_pos = 0

        for j, next_pos in enumerate(step_positions):
            signals[current_pos:next_pos, i] = current_value
            current_value = step_values[j + 1]
            current_pos = next_pos

        signals[current_pos:, i] = current_value

    return signals


class TestSyntheticDataGenerators:
    """Test synthetic data generators for correctness."""

    def test_sine_waves_basic(self):
        """Test basic sine wave generation."""
        n_samples, n_channels = 1000, 2
        data = generate_sine_waves(n_samples, n_channels)

        assert data.shape == (n_samples, n_channels)
        assert not np.any(np.isnan(data))
        assert np.all(np.abs(data) <= 1.1)  # Within expected range

        # Check that different channels have different frequencies
        assert not np.allclose(data[:, 0], data[:, 1])

    def test_sine_waves_with_noise(self):
        """Test sine wave generation with noise."""
        data = generate_sine_waves(1000, 1, noise_level=0.1)

        assert data.shape == (1000, 1)
        assert not np.any(np.isnan(data))
        # With noise, values should exceed pure sine range slightly
        assert np.any(np.abs(data) > 1.0)

    def test_complex_waveform(self):
        """Test complex waveform generation."""
        data = generate_complex_waveform(2000, 3)

        assert data.shape == (2000, 3)
        assert not np.any(np.isnan(data))

        # Should have more variation than simple sine waves
        for i in range(3):
            channel_data = data[:, i]
            std_dev = np.std(channel_data)
            assert std_dev > 0.5  # Should have reasonable variation

    def test_step_function(self):
        """Test step function generation."""
        data = generate_step_function(1000, 2, n_steps=5)

        assert data.shape == (1000, 2)
        assert not np.any(np.isnan(data))

        # Step functions should have relatively few unique values
        for i in range(2):
            unique_values = len(np.unique(data[:, i]))
            assert unique_values <= 10  # Should have distinct steps

    def test_reproducibility(self):
        """Test that generators produce reproducible results with same seed."""
        data1 = generate_sine_waves(500, 2, seed=42, noise_level=0.1)
        data2 = generate_sine_waves(500, 2, seed=42, noise_level=0.1)

        np.testing.assert_array_equal(data1, data2)

        # Different seeds should produce different results
        data3 = generate_sine_waves(500, 2, seed=123, noise_level=0.1)
        assert not np.allclose(data1, data3)

    def test_custom_frequencies(self):
        """Test sine wave generation with custom frequencies."""
        frequencies = [2.0, 5.0, 10.0]
        data = generate_sine_waves(
            1000, 3, frequencies=frequencies, sampling_rate=100.0
        )

        assert data.shape == (1000, 3)

        # Verify frequency content using NumPy FFT (basic check)
        for i, expected_freq in enumerate(frequencies):
            channel_fft = np.fft.fft(data[:, i])
            freqs = np.fft.fftfreq(1000, 1 / 100.0)

            # Find peak frequency
            peak_idx = np.argmax(np.abs(channel_fft[1 : len(freqs) // 2])) + 1
            peak_freq = abs(freqs[peak_idx])

            # Should be close to expected frequency
            assert abs(peak_freq - expected_freq) < 0.5


class TestDataProperties:
    """Test properties and edge cases of synthetic data."""

    def test_single_channel(self):
        """Test single channel data generation."""
        data = generate_sine_waves(100, 1)
        assert data.shape == (100, 1)

    def test_large_channel_count(self):
        """Test generation with many channels."""
        data = generate_sine_waves(100, 20)
        assert data.shape == (100, 20)

        # Each channel should be different
        for i in range(19):
            assert not np.allclose(data[:, i], data[:, i + 1])

    def test_small_sample_count(self):
        """Test with small sample counts."""
        data = generate_sine_waves(10, 2)
        assert data.shape == (10, 2)
        assert not np.any(np.isnan(data))

    def test_zero_noise(self):
        """Test explicitly with zero noise."""
        data = generate_sine_waves(1000, 1, noise_level=0.0)

        # Should be a clean sine wave
        t = np.linspace(0, 1, 1000)
        expected = np.sin(2 * np.pi * 1.0 * t)

        np.testing.assert_allclose(data[:, 0], expected, atol=1e-10)


if __name__ == "__main__":
    # Run tests manually
    test_gen = TestSyntheticDataGenerators()
    test_props = TestDataProperties()

    print("Running synthetic data generator tests...")

    # Run TestSyntheticDataGenerators tests
    for method_name in dir(test_gen):
        if method_name.startswith("test_"):
            print(f"  {method_name}...", end=" ")
            try:
                getattr(test_gen, method_name)()
                print("✓")
            except Exception as e:
                print(f"✗ {e}")

    # Run TestDataProperties tests
    for method_name in dir(test_props):
        if method_name.startswith("test_"):
            print(f"  {method_name}...", end=" ")
            try:
                getattr(test_props, method_name)()
                print("✓")
            except Exception as e:
                print(f"✗ {e}")

    print("Tests completed!")
