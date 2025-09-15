from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .alignment import read_timeseries, get_video_timeline, align_signal_cfr
from .plotting import generate_plot_videos
from .combine import combine_videos
from .utils import _has_cmd


def _infer_signal_and_columns(df: pd.DataFrame) -> tuple[np.ndarray, list[str], bool]:
    cols = set(df.columns.astype(str))

    # ROI pivot format
    if {"frame", "roi_name", "percentage_in_roi"}.issubset(cols):
        dfw = (
            df.pivot(index="frame", columns="roi_name", values="percentage_in_roi")
            .reset_index()
            .fillna(0)
        )
        channel_cols = [c for c in dfw.columns if c not in ("frame", "instance")]
        sig = dfw[channel_cols].to_numpy()
        return sig, [str(c) for c in channel_cols], True

    # Generic wide numeric
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if str(c) not in ("frame", "time", "t")]
    if not numeric_cols:
        raise ValueError(
            "No numeric signal columns found. Provide CSV/HDF5 with numeric columns, "
            "or ROI CSV with columns: frame, roi_name, percentage_in_roi."
        )
    sig = df[numeric_cols].to_numpy()
    return sig, [str(c) for c in numeric_cols], False


def _compute_legend(legend_flag: Optional[bool], mode: str, num_channels: int) -> bool:
    """Determine whether to show legend.

    - If legend_flag is True/False, honor it.
    - If None, enable for combine mode with small channel counts (<= 10).
    """
    if legend_flag is not None:
        return bool(legend_flag)
    return mode == "combine" and num_channels <= 10


def _ensure_tools(need_ffmpeg: bool = True, need_ffprobe: bool = False, quiet: bool = False) -> None:
    missing = []
    if need_ffmpeg and not _has_cmd("ffmpeg"):
        missing.append("ffmpeg")
    if need_ffprobe and not _has_cmd("ffprobe"):
        missing.append("ffprobe")
    if missing:
        msg = "Missing required tool(s): " + ", ".join(missing) + ". Install and retry."
        if quiet:
            raise SystemExit(2)
        raise SystemExit(msg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chronoviz", description="ChronoViz CLI")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error output")
    p.add_argument("--verbose", action="store_true", help="Extra diagnostics")
    sub = p.add_subparsers(dest="command", required=True)

    # plots
    plots = sub.add_parser("plots", help="Generate plot video(s) from a signals file (CSV/HDF5)")
    plots.add_argument("-s", "--signals", type=Path, required=True, help="Path to CSV/HDF5")
    plots.add_argument("--signals-key", type=str, default=None, help="Dataset key for HDF5")
    plots.add_argument("-o", "--output", type=Path, required=True, help="Directory to write plot video(s)")
    plots.add_argument("-m", "--mode", choices=("grid", "combine", "separate"), default="grid")
    plots.add_argument("--grid", nargs=2, metavar=("ROWS", "COLS"), type=int, default=None)
    plots.add_argument("--ylim", nargs=2, metavar=("LO", "HI"), type=float, default=None)
    plots.add_argument("--plot-size", nargs=2, metavar=("W", "H"), type=int, default=(640, 480))
    plots.add_argument("--fps", type=float, default=30.0)
    plots.add_argument("--left", type=int, default=250)
    plots.add_argument("--right", type=int, default=250)
    plots.add_argument("--ratio", type=float, default=1.0)
    plots.add_argument("--xlabel", type=str, default=None)
    plots.add_argument("--ylabel", type=str, default=None)
    legend_group = plots.add_mutually_exclusive_group()
    legend_group.add_argument("-l", "--legend", action="store_true", dest="legend", help="Force legend on (combine mode)")
    legend_group.add_argument("--no-legend", action="store_false", dest="legend", help="Force legend off")
    plots.set_defaults(legend=None)

    # combine
    combine = sub.add_parser("combine", help="Combine a base video with a plot video (stack or overlay)")
    combine.add_argument("-v", "--video", type=Path, required=True, help="Path to base video")
    combine.add_argument("-P", "--plot-video", dest="plot_video", type=Path, required=True, help="Path to plot video")
    combine.add_argument("-o", "--output", type=Path, required=True, help="Output video path")
    combine.add_argument("-p", "--position", choices=("top", "bottom", "left", "right", "tr", "tl", "br", "bl"), default="right")
    combine.add_argument("--overlay", action="store_true")
    combine.add_argument("-a", "--alpha", type=float, default=1.0)
    combine.add_argument("-r", "--ratio", type=float, default=1.0)
    combine.add_argument("-c", "--cpu", action="store_true")
    combine.add_argument("--force-cpu-for-stack", action="store_true")

    # render
    render = sub.add_parser("render", help="One-shot: read/align → plots → combine")
    render.add_argument("-v", "--video", type=Path, required=True, help="Path to base video")
    render.add_argument("-s", "--signals", type=Path, required=True, help="Path to CSV/HDF5")
    render.add_argument("-o", "--output", type=Path, required=False, help="Final combined video (.mp4). Required unless --plots-only")
    render.add_argument("-O", "--plots-output", type=Path, default=None, help="Directory for intermediate plot video(s)")
    render.add_argument("--signals-key", type=str, default=None, help="HDF5 dataset key")
    render.add_argument("--align-mode", choices=("resample", "pad"), default="resample")
    render.add_argument("--padding-mode", type=str, default="edge")
    render.add_argument("--ratio", type=float, default=1.0)
    render.add_argument("-m", "--mode", choices=("grid", "combine", "separate"), default="grid")
    render.add_argument("--grid", nargs=2, metavar=("ROWS", "COLS"), type=int, default=None)
    render.add_argument("--ylim", nargs=2, metavar=("LO", "HI"), type=float, default=None)
    render.add_argument("--plot-size", nargs=2, metavar=("W", "H"), type=int, default=(640, 480))
    render.add_argument("--left", type=int, default=250)
    render.add_argument("--right", type=int, default=250)
    render.add_argument("--xlabel", type=str, default=None)
    render.add_argument("--ylabel", type=str, default=None)
    legend_group_r = render.add_mutually_exclusive_group()
    legend_group_r.add_argument("-l", "--legend", action="store_true", dest="legend", help="Force legend on (combine mode)")
    legend_group_r.add_argument("--no-legend", action="store_false", dest="legend", help="Force legend off")
    render.set_defaults(legend=None)
    render.add_argument("-p", "--position", choices=("top", "bottom", "left", "right", "tr", "tl", "br", "bl"), default="right")
    render.add_argument("--overlay", action="store_true")
    render.add_argument("-a", "--alpha", type=float, default=1.0)
    render.add_argument("-c", "--cpu", action="store_true")
    render.add_argument("--force-cpu-for-stack", action="store_true")
    render.add_argument("--fps", type=float, default=None, help="Override detected base video FPS for plotting")
    render.add_argument("--plots-only", action="store_true", help="Generate plots only; skip combine")

    return p


def cmd_plots(args: argparse.Namespace) -> int:
    _ensure_tools(need_ffmpeg=True, need_ffprobe=False, quiet=args.quiet)
    df = read_timeseries(args.signals, key=args.signals_key)
    sig, col_names, is_roi = _infer_signal_and_columns(df)
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    C = sig.shape[1]

    # Defaults for labels
    xlabel = args.xlabel if args.xlabel is not None else "Time (frames)"
    default_ylabel = "Percentage in ROI" if is_roi else "Value"
    ylabel = args.ylabel if args.ylabel is not None else default_ylabel
    show_legend = _compute_legend(args.legend, args.mode, C)

    out = generate_plot_videos(
        aligned_signal=sig,
        ratio=float(args.ratio),
        output_dir=args.output,
        mode=args.mode,
        grid=tuple(args.grid) if args.grid is not None else None,
        col_names=col_names,
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        left=int(args.left),
        right=int(args.right),
        video_fps=float(args.fps),
        plot_size=tuple(args.plot_size),
        show_legend=show_legend,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    if not args.quiet:
        print(f"Wrote plots to: {out}")
    return 0


def _bool_flag(val: bool) -> bool:
    return bool(val)


def cmd_combine(args: argparse.Namespace) -> int:
    _ensure_tools(need_ffmpeg=True, need_ffprobe=True, quiet=args.quiet)
    if not args.overlay and args.position in ("tr", "tl", "br", "bl"):
        raise SystemExit("corner positions only valid with --overlay")
    if not args.overlay and args.alpha != 1.0:
        if not args.quiet:
            print("Warning: --alpha is only used with --overlay; ignoring.")

    ok = combine_videos(
        video_path=args.video,
        plot_video_path=args.plot_video,
        output_path=args.output,
        ratio=float(args.ratio),
        position=args.position,
        overlay=_bool_flag(args.overlay),
        alpha=float(args.alpha),
        cpu=_bool_flag(args.cpu),
        force_cpu_for_stack=_bool_flag(args.force_cpu_for_stack),
    )
    if not ok:
        return 1
    if not args.quiet:
        print(f"Wrote combined video to: {args.output}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    _ensure_tools(need_ffmpeg=True, need_ffprobe=True, quiet=args.quiet)
    # Default output if not provided and not plots-only
    if not args.plots_only and args.output is None:
        vid = Path(args.video)
        args.output = vid.with_name(f"{vid.stem}_with_plots.mp4")
    if args.mode == "separate" and not args.plots_only:
        raise SystemExit("combine step requires a single plot video; use mode grid or combine")
    if not args.overlay and args.position in ("tr", "tl", "br", "bl"):
        raise SystemExit("corner positions only valid with --overlay")
    if not args.overlay and args.alpha != 1.0:
        if not args.quiet:
            print("Warning: --alpha is only used with --overlay; ignoring.")

    video_fps, n_frames, video_times = get_video_timeline(args.video)
    base_fps = float(args.fps) if args.fps is not None else float(video_fps)

    df = read_timeseries(args.signals, key=args.signals_key)
    sig, col_names, is_roi = _infer_signal_and_columns(df)

    aligned = align_signal_cfr(
        video_times=video_times,
        sig_values=sig,
        mode=args.align_mode,
        ratio=float(args.ratio),
        padding_mode=args.padding_mode,
    )

    plots_dir = args.plots_output
    if plots_dir is None:
        if args.output is not None:
            plots_dir = args.output.parent / "plots"
        else:
            plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for labels & legend
    C = aligned.shape[1] if aligned.ndim == 2 else 1
    xlabel = args.xlabel if args.xlabel is not None else "Time (frames)"
    default_ylabel = "Percentage in ROI" if is_roi else "Value"
    ylabel = args.ylabel if args.ylabel is not None else default_ylabel
    show_legend = _compute_legend(args.legend, args.mode, C)

    plot_path = generate_plot_videos(
        aligned_signal=aligned,
        ratio=float(args.ratio),
        output_dir=plots_dir,
        mode=args.mode,
        grid=tuple(args.grid) if args.grid is not None else None,
        col_names=col_names,
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        left=int(args.left),
        right=int(args.right),
        video_fps=base_fps,
        plot_size=tuple(args.plot_size),
        show_legend=show_legend,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    if not args.quiet:
        print(f"Wrote plot video(s) to: {plot_path}")

    if args.plots_only:
        return 0

    ok = combine_videos(
        video_path=args.video,
        plot_video_path=plot_path,
        output_path=args.output,
        ratio=float(args.ratio),
        position=args.position,
        overlay=_bool_flag(args.overlay),
        alpha=float(args.alpha),
        cpu=_bool_flag(args.cpu),
        force_cpu_for_stack=_bool_flag(args.force_cpu_for_stack),
    )
    if not ok:
        return 1
    if not args.quiet:
        print(f"Wrote final combined video to: {args.output}")
    return 0


def cli_main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plots":
        return cmd_plots(args)
    if args.command == "combine":
        return cmd_combine(args)
    if args.command == "render":
        return cmd_render(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(cli_main())
