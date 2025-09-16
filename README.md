# ChronoViz — Synchronized time series + video

ChronoViz is a high‑performance Python tool for rendering time series data synchronized with video. It focuses on fast offline rendering using Matplotlib + FFmpeg and can generate long, frame‑accurate visualizations.

Core pipeline
- Alignment: read time series (CSV/HDF5) and align to the video timeline
- Plotting: generate sliding‑window plot video(s)
- Combining: stitch original video and plot video(s) side‑by‑side or as an overlay

What’s included
- CLI with three subcommands: `plots`, `combine`, and `render` (one‑shot)
- GPU‑aware FFmpeg combining (CPU fallback for deterministic behavior)
- Support for ROI CSVs (`frame`, `roi_name`, `percentage_in_roi`) or wide numeric CSV/HDF5


## Quickstart

Prerequisites
- Python 3.12+
- FFmpeg and FFprobe on PATH (`ffmpeg`, `ffprobe`)
- uv (recommended) or pip

Create/refresh the environment
- uv: `uv sync`
- or editable install: `uv pip install -e .` (or `pip install -e .`)

CLI help
- `chronoviz --help`
- If the console script isn’t visible, use: `python -m chronoviz.cli --help` or `python main.py --help`

Examples
1) Generate plot video(s) from signals only
```
chronoviz plots -s /path/to/data.csv -o /tmp/plots -m grid --grid 6 1 --ylim 0 100 --plot-size 320 768 --fps 30 --left 250 --right 250
```

2) Combine original video with a plot video (stack right, CPU path)
```
chronoviz combine -v /path/to/video.mp4 -P /tmp/plots/signals_plot_grid.mp4 -o /tmp/combined.mp4 -p right -c
```

3) One‑shot render (align → plots → combine)
```
chronoviz render -v /path/to/video.mp4 -s /path/to/data.csv -o /tmp/final.mp4 -m grid --grid 6 1 --ylim 0 100 -c
```

Notes
- ROI CSVs with columns `frame, roi_name, percentage_in_roi` are auto‑pivoted to wide format. For HDF5, pass `--signals-key`.
- Corner positions (`tr`, `tl`, `br`, `bl`) require `--overlay`. Use `-a/--alpha` for transparency.
- Use `-c/--cpu` to force CPU paths (deterministic, broadly compatible). GPU paths are auto‑detected when available.
- Auto legend: enabled by default for `-m combine` when channel count ≤ 10. Use `-l/--legend` to force on or `--no-legend` to force off.
- Default output name (render): if `-o/--output` is omitted (and not `--plots-only`), output is `<video_stem>_with_plots.mp4` next to the input video.
- Verbosity: add `-q/--quiet` to suppress non‑error prints; `--verbose` for extra diagnostics.


## Development

Install deps
- `uv sync`

Run tests
- `pytest -q`

Lint/format
- `ruff check .`
- `black .`

Project layout
- `src/alignment.py` — read and align signals to video timeline
- `src/plotting.py` — generate plot video(s) using Matplotlib
- `src/combine.py` — stitch base video and plots via FFmpeg (GPU‑aware)
- `src/cli.py` — CLI entrypoint and subcommands
- `main.py` — thin wrapper delegating to `chronoviz.cli`

Roadmap
- [ ] Real‑time/interactive playback modes
- [ ] Faster GPU plotting backends (e.g., pygfx/fastplotlib)
- [ ] Additional data adapters and presets
