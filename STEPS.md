# ChronoViz Development Steps

This roadmap breaks work into small, testable milestones. Each step lists:
- What will change
- Clear deliverables (what you should be able to run/see)
- Agent handoff: stop here and let the human test before proceeding

Assumptions/prereqs:
- Python 3.12+, `uv` installed, and FFmpeg available on PATH (`ffmpeg`, `ffprobe`).
- You will provide your own video (`.mp4`) and signals CSV/HDF5 for testing.


## Step 0 — Environment Sanity Check

Changes:
- None (validation only).

Deliverables:
- `uv sync` completes.
- `ffmpeg -version` and `ffprobe -version` succeed.
- `pytest -q` runs (may skip heavy tests in CI or without media).

Commands:
- `uv sync`
- `ffmpeg -version`
- `ffprobe -version`
- `pytest -q`

Agent handoff:
- Stop after this step. Confirm tools are installed and tests run before continuing.


## Step 1 — Minimal CLI for Plot Videos (plots-only)

Changes:
- Add an argparse-based CLI that reads a signals file, aligns (if needed), and writes plot video(s) without combining with the base video yet.
- New command: `chronoviz plots` (or `python -m chronoviz` if entrypoint isn’t wired yet).
- Required args: `--signals`, `--output`, `--mode [grid|combine|separate]`. Optional: `--grid r c`, `--ylim lo hi`, `--plot-size w h`, `--fps`, `--left`, `--right`, `--ratio`.

Deliverables:
- Running the CLI generates plot video(s) in the specified output directory.
- Example (adjust paths):
  - `chronoviz plots --signals /path/to/data.csv --output /tmp/plots --mode grid --grid 6 1 --ylim 0 100 --plot-size 320 768 --fps 30 --left 250 --right 250`
  - Outputs: a `.mp4` (or `.webm` if alpha later) plot video in `/tmp/plots`.

Agent handoff:
- Stop here. Human validates plot video quality/performance and notes any UX tweaks before combining with a base video.


## Step 2 — CLI Entrypoint Packaging

Changes:
- Add `[project.scripts]` to `pyproject.toml` so `chronoviz` is a console command (e.g., `chronoviz = "main:cli_main"` or `chronoviz = "src.cli:main"`).
- Ensure module layout supports `python -m chronoviz` as well (optional but nice).

Deliverables:
- `chronoviz --help` works after `uv sync` or `pip install -e .`.
- Existing `plots` subcommand is discoverable under `chronoviz`.

Commands:
- `uv sync`
- `chronoviz --help`
- `chronoviz plots --help`

Agent handoff:
- Stop here. Human validates the packaging/entrypoint works in their environment.


## Step 3 — Add Video Combine Command (stack/overlay)

Changes:
- Add `chronoviz combine` command that merges a base video with a plot video using `src/combine.py`.
- Args: `--video`, `--plot-video`, `--output`, `--position [top|bottom|left|right|tr|tl|br|bl]`, `--overlay`, `--alpha`, `--ratio`, `--cpu`.
- Ensure GPU-aware path remains optional; provide `--cpu` for deterministic fallback.

Deliverables:
- Side-by-side or overlay output video created successfully.
- Example (stack right):
  - `chronoviz combine --video /path/to/source.mp4 --plot-video /tmp/plots/signals_plot_grid.mp4 --output /tmp/combined.mp4 --position right --cpu`
- Example (overlay top-right with 80% opacity):
  - `chronoviz combine --video /path/to/source.mp4 --plot-video /tmp/plots/signals_plot_grid.mp4 --output /tmp/overlay.mp4 --overlay --position tr --alpha 0.8 --cpu`

Agent handoff:
- Stop here. Human checks visual alignment, sizing, and encoder compatibility.


## Step 4 — One-Shot Pipeline Command (render)

Changes:
- Add `chronoviz render` that performs: read/align → plots → combine, in one command.
- Args unify from Steps 1 and 3: `--video`, `--signals`, plotting args, combine args, and `--plots-only` flag to skip combine.

Deliverables:
- Single command produces final combined video, or only plot videos when `--plots-only` is set.
- Example:
  - `chronoviz render --video /path/to/source.mp4 --signals /path/to/data.csv --mode grid --grid 6 1 --ylim 0 100 --plot-size 320 768 --position right --ratio 1.0 --cpu --output /tmp/final.mp4`

Agent handoff:
- Stop here. Human validates end-to-end flow and flags default choices to refine (e.g., defaults for grid, labels, legend).


## Step 5 — Documentation Cleanup

Changes:
- Update README.md and AGENTS.md:
  - Replace `requirements.txt` instructions with `uv sync` (project already uses `pyproject.toml`).
  - Add CLI usage examples from Steps 1–4.
  - Note that sample media is not bundled; user must supply `--video`/`--signals`.

Deliverables:
- Accurate setup/usage docs that match the implemented CLI.

Agent handoff:
- Stop here. Human reviews docs for clarity and completeness.


## Step 6 — Validation and Linting

Changes:
- Run and fix lint/format/type hints only where touched by CLI work.
- Add a couple of smoke tests for CLI argument parsing and plot generation using tiny synthetic data (no media files required for these tests).

Deliverables:
- `ruff check .` passes (or violations triaged).
- `black .` formats cleanly.
- `pytest -q` passes for new tests.

Commands:
- `ruff check .`
- `black .`
- `pytest -q`

Agent handoff:
- Stop here. Human validates CI hygiene and minimal test coverage.


## Step 7 — UX and Defaults Polish

Changes:
- Improve defaults: auto grid layout for N channels, reasonable `ylim` detection, human-friendly labels, optional legend in combine mode.
- Add `--xlabel`, `--ylabel`, and simple theming flags where low-effort.

Deliverables:
- Cleaner visuals with sensible defaults; fewer required flags for common cases.
- Example:
  - `chronoviz render --video v.mp4 --signals s.csv --mode grid --plots-only` should “just work” with decent layout.

Agent handoff:
- Stop here. Human confirms visual quality is acceptable for typical datasets.


## Step 8 — GPU Path Tuning (Optional)

Changes:
- Verify GPU backends selection heuristics on the target machine.
- Expose `--force-cpu-for-stack` toggle (already supported in code) and document when to use it.

Deliverables:
- Confirmed performance path (CPU-only vs GPU hybrid) with instructions for the user to switch.
- Example:
  - `chronoviz combine --video v.mp4 --plot-video p.mp4 --output out.mp4 --position right --cpu`
  - `chronoviz combine --video v.mp4 --plot-video p.mp4 --output out.mp4 --position right --alpha 1.0` (GPU where available)

Agent handoff:
- Stop here. Human decides whether to pursue deeper GPU integration or keep CPU defaults.


Notes
- Keep each step scoped and reversible; don’t refactor unrelated parts.
- Prefer adding CLI first, then tightening docs/tests, then polish.
- If the human reports blockers at any step, address them before proceeding.

