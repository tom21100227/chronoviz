# Project Overview

This project, `chronoviz`, is a Python tool for visualizing time series data synchronized with video. It is designed for high-performance offline rendering, using Matplotlib and FFmpeg to create synchronized visualizations of long recordings. The roadmap includes plans for a CLI, GPU acceleration, real-time streaming, and interactive playback.

The core pipeline consists of:
1.  **Alignment**: Reading time series data (from CSV or HDF5) and aligning it with the video's timeline. This is handled in `src/alignment.py`.
2.  **Plotting**: Generating plot videos from the aligned time series data using a sliding window approach. This is done in `src/plotting.py` with `matplotlib` as the current backend.
3.  **Combining**: Merging the original video with the generated plot videos using `ffmpeg`. This is managed by `src/combine.py`, which is GPU-aware for better performance.

## Building and Running

The project uses `uv` for dependency management.

Prerequisites
- Python 3.12+
- FFmpeg + FFprobe on PATH (`ffmpeg`, `ffprobe`)
- `uv` installed

Install environment
- `uv sync`
- Optional (to force install console scripts): `uv pip install -e .`

Run via CLI (preferred)
- Help: `chronoviz --help`
- One‑shot render (align → plots → combine):
  ```bash
  chronoviz render -v /path/to/video.mp4 -s /path/to/data.csv -o /tmp/final.mp4 -m grid --grid 6 1 --ylim 0 100 -c
  ```
- Plots only:
  ```bash
  chronoviz plots -s /path/to/data.csv -o /tmp/plots -m grid --grid 6 1
  ```
- Combine only:
  ```bash
  chronoviz combine -v /path/to/video.mp4 -P /tmp/plots/signals_plot_grid.mp4 -o /tmp/combined.mp4 -p right -c
  ```

Fallback (module or script)
- `python -m src.cli --help`
- `python main.py --help`

Notes
- ROI CSVs with columns `frame, roi_name, percentage_in_roi` are auto‑pivoted. For HDF5, pass `--signals-key`.
- Corner positions (`tr`, `tl`, `br`, `bl`) require `--overlay`. Use `-a/--alpha` to control transparency.
- Use `-c/--cpu` to force CPU paths; GPU is used opportunistically if detected.
- Auto legend: default on for `-m combine` with ≤10 channels; override with `-l/--legend` or `--no-legend`.
- Default output file (render): if `-o` omitted (and not `--plots-only`), writes `<video_stem>_with_plots.mp4` next to input.
- Verbosity: `-q/--quiet` to suppress non‑error prints; `--verbose` for extra diagnostics.

**To run tests:**
```bash
pytest -q
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

## Agent Hints
- Follow `STEPS.md` for milestone sequencing and human validation points.
- Keep changes focused; avoid refactors outside the current step.
- Prefer CPU paths for deterministic baseline performance; GPU is optional.
