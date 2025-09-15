# ChronoViz Development Steps

This document tracks the current milestone plan. Each step lists changes, clear deliverables, example commands, and an agent handoff to let a human validate before proceeding.

Assumptions/prereqs:
- Python 3.12+, `uv` installed, and FFmpeg/FFprobe on PATH (`ffmpeg`, `ffprobe`).
- You provide your own video (`.mp4`) and signals CSV/HDF5 for testing.


## Summary of Completed Milestones
- CLI foundation: `plots`, `combine`, and one‑shot `render` commands.
- Packaged console entrypoint `chronoviz`.
- Docs updated (README/AGENTS) to reflect CLI usage and uv workflow.
- Polish: auto grid layout, auto legend for small `combine`, default labels, ffmpeg/ffprobe preflight, default render output name, quiet/verbose.


## Next Development Steps

### Step A — Bar Plot Support

Changes:
- Add bar plot rendering alongside line plots.
- New CLI flags:
  - `--style line|bar` (default: `line`)
  - `--bar-mode grouped|stacked` (default: `grouped`)
  - `--bar-agg instant|mean|max` (default: `instant`) with `--bar-window N` (frames) for smoothing when using `mean`/`max`.
- Supported in `grid` and `combine` modes.

Deliverables:
- `chronoviz plots ... --style bar` generates bar plot video(s) for grid/combine.
- `chronoviz render ... --style bar` produces final side‑by‑side/overlay output.

Example commands:
- `chronoviz plots -s data.csv -o /tmp/plots -m grid --style bar --bar-mode grouped`
- `chronoviz plots -s data.csv -o /tmp/plots -m combine --style bar --bar-mode stacked --bar-agg mean --bar-window 15`

Agent handoff:
- Stop after implementation. Human tests grouped/stacked and agg modes (instant/mean/max) for visual quality and performance.


### Step B — X‑Axis as Wall Time

Changes:
- X‑axis can reflect frames, seconds, or absolute wall time.
- New CLI: `--xaxis frames|seconds|absolute`.
  - `render` defaults to `seconds`; `plots` defaults to `frames` unless `--fps` provided.
- Absolute mode: add an overlay timestamp (HH:MM:SS.mmm) rather than dynamic tick relabeling.

Deliverables:
- `chronoviz render` shows seconds on x‑axis by default; `--xaxis absolute` adds a timestamp overlay when video timeline is available.
- `plots` supports `--xaxis seconds` when `--fps` is provided.

Example commands:
- `chronoviz render -v video.mp4 -s data.csv -o /tmp/out.mp4 --xaxis seconds`
- `chronoviz plots -s data.csv -o /tmp/plots --xaxis seconds --fps 30`
- `chronoviz render -v video.mp4 -s data.csv -o /tmp/out.mp4 --xaxis absolute`

Agent handoff:
- Stop after implementation. Human confirms x‑axis scaling and timestamp overlay are correct.


### Step C — Richer Data Adapters

Changes:
- Introduce `src/adapters` to handle multiple input shapes:
  - ROI CSV (`frame, roi_name, percentage_in_roi`)
  - Wide numeric CSV/HDF5, optional time column (`--time-col`),
  - HDF5 datasets with separate time/data (`--time-key`).
- Extend `read_timeseries` to return `(values, times|None, columns, meta)` and prefer times for alignment when available.
- Extend `align_signal_cfr` to accept time arrays for more precise resampling to video times.
- Optional `--units` to annotate axis labels.

Deliverables:
- CLI seamlessly loads different input formats; seconds/absolute x‑axis uses provided times when present.

Example commands:
- `chronoviz render -v video.mp4 -s data_with_time.csv --time-col timestamp -o /tmp/out.mp4 --xaxis seconds`
- `chronoviz render -v video.mp4 -s data.h5 --signals-key data --time-key time -o /tmp/out.mp4 --xaxis absolute`

Agent handoff:
- Stop after implementation. Human tests multiple file shapes and confirms alignment behavior.


### Step D — GPU Plotting Backend (Optional)

Changes:
- Add optional `pygfx`/`fastplotlib` backend for offscreen GPU plotting.
- New CLI: `--backend matplotlib|pygfx` (default: `matplotlib`).
- Feature parity goal: line + bar, grid/combine, decent color defaults.

Deliverables:
- `chronoviz plots/render --backend pygfx` runs and shows performance gains where supported.

Example commands:
- `chronoviz plots -s data.csv -o /tmp/plots -m grid --backend pygfx`
- `chronoviz render -v video.mp4 -s data.csv -o /tmp/out.mp4 -m combine --style bar --backend pygfx`

Agent handoff:
- Stop after implementation. Human benchmarks and decides whether to use GPU backend by default.


Notes
- Keep each step scoped and reversible; avoid refactors beyond scope.
- Favor minimal, testable increments; deliver and pause for human validation after each step.
