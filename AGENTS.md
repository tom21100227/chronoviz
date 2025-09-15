# Project Overview

This project, `chronoviz`, is a Python tool for visualizing time series data synchronized with video. It is designed for high-performance offline rendering, using Matplotlib and FFmpeg to create synchronized visualizations of long recordings. The roadmap includes plans for a CLI, GPU acceleration, real-time streaming, and interactive playback.

The core pipeline consists of:
1.  **Alignment**: Reading time series data (from CSV or HDF5) and aligning it with the video's timeline. This is handled in `src/alignment.py`.
2.  **Plotting**: Generating plot videos from the aligned time series data using a sliding window approach. This is done in `src/plotting.py` with `matplotlib` as the current backend.
3.  **Combining**: Merging the original video with the generated plot videos using `ffmpeg`. This is managed by `src/combine.py`, which is GPU-aware for better performance.

## Building and Running

The project uses `uv` for dependency management.

**To install dependencies:**
```bash
uv pip install -r requirements.txt
```
or if you have `uv` installed:
```bash
uv sync
```

**To run the main application:**
```bash
python main.py
```
The main script in `main.py` processes a sample video and CSV file from the `test_data` directory. You can modify this file to process your own data.

**To run tests:**
```bash
pytest
```

## Development Conventions

- **Package Management**: The project uses `uv` and a `pyproject.toml` file to manage dependencies.
- **Code Style**: The project uses `black` for code formatting and `ruff` for linting.
- **Type Checking**: `mypy` is used for static type checking.
- **Testing**: `pytest` is used for testing.

You can run the linter and formatter with:
```bash
ruff check .
black .
```
