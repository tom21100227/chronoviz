"""
Configuration file for ChronoViz benchmarks.

Modify these parameters to customize benchmark runs for your hardware
and testing needs.
"""

# Benchmark data sizes (number of samples)
BENCHMARK_SIZES = [1000, 5000, 10000, 25000]

# Benchmark channel counts
BENCHMARK_CHANNELS = [1, 2, 4, 8]

# Benchmark resolutions (width, height)
BENCHMARK_RESOLUTIONS = [
    (320, 240),   # Low resolution
    (640, 480),   # Standard definition  
    (1280, 720),  # HD
    (1920, 1080)  # Full HD
]

# Benchmark window sizes (left, right)
BENCHMARK_WINDOW_SIZES = [
    (25, 25),
    (50, 50), 
    (100, 100),
    (250, 250)
]

# Benchmark video frame rates
BENCHMARK_FRAME_RATES = [15, 24, 30, 60]

# Default synthetic data parameters
DEFAULT_SINE_FREQUENCIES = [1.0, 2.0, 3.0, 5.0, 8.0]
DEFAULT_SAMPLING_RATE = 1000.0
DEFAULT_NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5]

# Performance test thresholds (samples per second)
PERFORMANCE_THRESHOLDS = {
    "plotting_min": 1000,      # Minimum acceptable plotting speed
    "plotting_good": 5000,     # Good plotting performance
    "alignment_min": 10000,    # Minimum acceptable alignment speed  
    "alignment_good": 50000,   # Good alignment performance
}

# Memory usage limits (MB)
MEMORY_LIMITS = {
    "warning_mb": 1000,        # Warn if memory usage exceeds this
    "max_mb": 2000,           # Fail if memory usage exceeds this
}

# Test timeouts (seconds)
TEST_TIMEOUTS = {
    "single_test": 60,         # Timeout for individual tests
    "full_suite": 1800,       # Timeout for full benchmark suite (30 min)
}

# Output settings
OUTPUT_SETTINGS = {
    "temp_cleanup": True,      # Clean up temporary files after tests
    "save_videos": False,      # Save benchmark videos for inspection
    "verbose_output": True,    # Show detailed timing information
    "progress_bars": False,    # Show progress bars (requires tqdm)
}

# Hardware-specific settings
# Adjust these based on your system capabilities
HARDWARE_CONFIG = {
    "cpu_cores": None,         # None = auto-detect
    "memory_gb": None,         # None = auto-detect
    "skip_slow_tests": False,  # Skip tests that take > 30 seconds
    "parallel_tests": False,   # Run compatible tests in parallel
}

# Custom test configurations
# Add your own test scenarios here
CUSTOM_CONFIGS = {
    "quick_test": {
        "sizes": [1000, 5000],
        "channels": [1, 2],
        "resolutions": [(640, 480)],
        "modes": ["combine"]
    },
    
    "stress_test": {
        "sizes": [50000, 100000],
        "channels": [16, 32],
        "resolutions": [(1920, 1080)],
        "modes": ["grid", "separate"]
    },
    
    "memory_test": {
        "sizes": [100000],
        "channels": [8],
        "resolutions": [(1280, 720)],
        "modes": ["grid"]
    }
}
