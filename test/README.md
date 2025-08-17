# ChronoViz Test Suite

This directory contains test files and benchmarks for the ChronoViz library, focusing on performance testing with synthetic data.

## Files

### `test_data_generators.py`
Unit tests for the synthetic data generators to ensure they produce valid test data:
- `generate_sine_waves()`: Creates clean sine waves with optional noise
- `generate_complex_waveform()`: Creates multi-harmonic signals with envelopes and artifacts  
- `generate_step_function()`: Creates step function signals for edge case testing

### `test_synthetic_benchmarks.py`
Comprehensive benchmark test suite using pytest-style tests:
- Tests plotting performance across different modes (combine, grid, separate)
- Tests scaling with data size, channel count, and resolution
- Tests data complexity impact on performance
- Tests alignment algorithm performance
- Tests memory usage with large datasets

### `benchmark_runner.py`
Simple standalone benchmark runner that provides quick performance metrics:
- Easy to run without pytest
- Focuses on core performance patterns
- Provides summary statistics and recommendations

### `performance_profiler.py`
Detailed performance profiler using Python's cProfile:
- Identifies specific function bottlenecks
- Provides memory usage analysis
- Compares performance across data types

### `generate_samples.py`
Creates sample videos using synthetic data:
- Demonstrates different data types visually
- Useful for validating synthetic data quality
- Creates comparison videos

### `benchmark_config.py`
Configuration file for customizing benchmark parameters:
- Adjustable test sizes and scenarios
- Hardware-specific settings
- Custom test configurations

## Running the Benchmarks

### Quick Benchmarks
For a quick performance overview, run the standalone benchmark runner:

```bash
cd test/
python benchmark_runner.py
```

This will test:
- Different plotting modes (combine, grid, separate)
- Data size scaling (1K to 25K samples)  
- Channel count scaling (1 to 16 channels)
- Resolution impact (320p to 1080p)
- Signal alignment performance

### Performance Profiling
For detailed bottleneck analysis:

```bash
python performance_profiler.py
```

This provides:
- Function-level timing breakdown
- Memory usage patterns
- Data type performance comparison

### Generate Sample Videos
To create example videos with synthetic data:

```bash
python generate_samples.py
```

This creates visual examples of:
- Different synthetic data types
- Performance scaling demonstrations

### Configuration
Customize benchmark parameters by editing `benchmark_config.py`:

```python
# Example: Run quick tests only
BENCHMARK_SIZES = [1000, 5000]  # Smaller test sizes
BENCHMARK_CHANNELS = [1, 2]      # Fewer channels
OUTPUT_SETTINGS["skip_slow_tests"] = True
```

### Comprehensive Benchmarks
For detailed testing using pytest:

```bash
# Run all benchmark tests
pytest test_synthetic_benchmarks.py -v

# Run specific test categories
pytest test_synthetic_benchmarks.py::TestPlottingPerformance -v
pytest test_synthetic_benchmarks.py::TestDataComplexityImpact -v

# Run performance matrix (warning: can be slow)
pytest test_synthetic_benchmarks.py::TestPlottingPerformance::test_plotting_performance_matrix -v
```

### Unit Tests
To verify the synthetic data generators work correctly:

```bash
python test_data_generators.py
# or
pytest test_data_generators.py -v
```

## Synthetic Data Types

The benchmarks use several types of synthetic data to test different performance scenarios:

### Sine Waves
- **Use case**: Baseline performance testing
- **Properties**: Smooth, predictable signals
- **Advantages**: Fast to generate, consistent results
- **Parameters**: Frequency, amplitude, noise level

### Complex Waveforms  
- **Use case**: Realistic signal complexity
- **Properties**: Multiple harmonics, envelope modulation, noise, artifacts
- **Advantages**: Tests rendering performance with detailed signals
- **Parameters**: Number of harmonics, modulation depth

### Step Functions
- **Use case**: Edge case testing
- **Properties**: Sharp transitions, constant segments
- **Advantages**: Tests rendering of discontinuities
- **Parameters**: Number of steps, step values

## Benchmark Categories

### Plotting Performance
Tests the core video generation pipeline:
- **Mode comparison**: combine vs grid vs separate rendering
- **Scaling**: How performance changes with data size/channels
- **Resolution impact**: Effect of output video resolution

### Data Complexity  
Tests how signal characteristics affect performance:
- **Signal type**: Sine vs complex vs step functions
- **Frequency content**: Low vs high frequency signals
- **Noise levels**: Clean vs noisy signals

### Alignment Performance
Tests the signal-to-video synchronization:
- **Algorithm modes**: Resample vs interpolate
- **Data sizes**: Scaling from 5K to 50K samples
- **Ratio effects**: Different video/signal rate ratios

### Memory Usage
Tests memory efficiency:
- **Large datasets**: 100K+ samples with multiple channels
- **Memory throughput**: MB/s processing rates
- **Memory patterns**: Peak usage during processing

## Performance Expectations

Based on typical hardware (modern laptop), expected performance ranges:

- **Plotting throughput**: 5,000-25,000 samples/sec
- **Channel scaling**: 2-10 channels/sec  
- **Alignment speed**: 10,000-100,000 samples/sec
- **Resolution impact**: 100k-2M pixels/sec

## Adding New Benchmarks

To add new benchmark tests:

1. **Synthetic data**: Add new generators to `test_data_generators.py`
2. **Performance tests**: Add test methods to `test_synthetic_benchmarks.py`
3. **Quick benchmarks**: Add functions to `benchmark_runner.py`

Follow the existing pattern:
- Use context managers for timing and cleanup
- Test multiple parameter combinations  
- Report throughput metrics (samples/sec, channels/sec, etc.)
- Clean up temporary files

## Dependencies

The benchmark tests require:
- `numpy` - For synthetic data generation
- `matplotlib` - For plotting (via ChronoViz)
- `pytest` - For running comprehensive test suite (optional)
- `pathlib`, `tempfile`, `shutil` - For file management

No additional dependencies beyond ChronoViz's requirements.
