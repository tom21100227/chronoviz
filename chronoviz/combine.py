"""
Module for combining original videos with generated plot videos.
GPU-aware: uses CUDA/QSV/VAAPI/Video
"""

from __future__ import annotations

from pathlib import Path
import subprocess

from .utils import (
    get_video_height,
    pick_video_encoder,
    _ffmpeg_supports_filter,
    _ffmpeg_has_encoder,
    BACKENDS,
    GpuBackend,
)

COMBINE_FFMPEG_LOGLEVEL = "warning"


def _select_backend(require_alpha: bool) -> GpuBackend | None:
    """
    Choose the first backend whose encoder exists and that supports GPU scaling.
    We don't require GPU overlay here; we can still do GPU scale + CPU overlay.
    If alpha is required, we usually won't use the backend encoder (most hw encoders
    don't do yuva420p), but scaling can still help.
    """
    for be in BACKENDS:
        if not _ffmpeg_has_encoder(be.encoder):
            continue
        if not _ffmpeg_supports_filter(be.scale_filter):
            continue
        # OK to use this backend for scaling; overlay may still be CPU.

        print(f"Selected GPU backend: {be.name}")

        return be
    return None


def _overlay_coords(position: str) -> str:
    if position == "tr":
        return "main_w-overlay_w-10:10"
    if position == "tl":
        return "10:10"
    if position == "br":
        return "main_w-overlay_w-10:main_h-overlay_h-10"
    if position == "bl":
        return "10:main_h-overlay_h-10"
    raise ValueError(
        f"Position '{position}' not valid for overlay mode. Use 'tr', 'tl', 'br', or 'bl'."
    )


def combine_videos(
    video_path: str | Path,
    plot_video_path: str | Path,
    output_path: str | Path,
    ratio: float = 1.0,
    position: str = "bottom",  # "top", "bottom", "left", "right"
    overlay: bool = False,
    alpha: float = 1,
    cpu: bool = False,
    force_cpu_for_stack: bool = False,
) -> bool:
    """
    Combine original video with plot video. Prefers GPU paths when possible.

    Parameters:
        - video_path: Path to the original video file.
        - plot_video_path: Path to the generated plot video file.
        - output_path: Path to save the combined video.
        - ratio: fps ratio between plot video and original video. 2.0 means plot video is twice the fps of original video.
        - position: Position of the plot video relative to the original ("top", "bottom", "left", "right", "tr", "tl", "br", bl").
        - overlay: If True, overlay the plot on top of the original video. Note: position "tr", "tl", "br", "bl" only valid when overlay is True.
        - alpha: Transparency level for overlay (0.0 to 1.0).
        - cpu: Force CPU-only processing for all operations.
        - force_cpu_for_stack: Override default behavior for stack operations. If None, auto-decides based on platform (True for VideoToolbox/Apple Silicon).

    Returns:
        - True if successful, False otherwise.
    """
    video_path = Path(video_path)
    plot_video_path = Path(plot_video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_height = get_video_height(video_path)
    if video_height == 0:
        print("Could not determine video height, aborting.")
        return False

    # Try to use a GPU backend (for scale and possibly overlay)
    backend = _select_backend(require_alpha=(alpha < 1.0)) if not cpu else None
    if force_cpu_for_stack is None:
        force_cpu_for_stack = (
            not overlay  # Only affects stack operations
            and backend
            and backend.name == "videotoolbox"  # VideoToolbox has expensive transfers
        )
        if force_cpu_for_stack:
            print(
                "Auto-enabling CPU-only processing for stack operations on VideoToolbox to avoid GPU-CPU transfer overhead"
            )

    have_gpu_overlay = False
    if backend and backend.overlay_filter:
        have_gpu_overlay = _ffmpeg_supports_filter(backend.overlay_filter)

    # Pick an encoder:
    # - If alpha < 1 and we have a GPU backend, prefer its hw encoder.
    # - If alpha is requested or no backend, fall back to pick_video_encoder().
    if backend and alpha >= 1.0:
        encoder = backend.encoder
        encoder_args = list(backend.enc_args)
    else:
        # alpha=True typically implies VP9/AV1 software path
        encoder, encoder_args = pick_video_encoder(alpha=alpha < 1.0, cpu=True)

    # Build the setpts segment if fps ratio != 1
    if ratio != 1.0:
        setpts = f"[1:v]setpts={ratio}*PTS[plot_adj]"
        plot_lbl = "[plot_adj]"
    else:
        setpts = ""
        plot_lbl = "[1:v]"

    # Common ffmpeg preamble
    base = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        COMBINE_FFMPEG_LOGLEVEL,
        "-stats",
    ]

    # If we have a GPU backend and we are going to stay in hw frames for a while,
    # include its hw accel flags up front. (If we later use only CPU filters,
    # that’s still fine; hw flags just enable hw decode if possible.)
    if backend:
        base += backend.hw_flags

    # ===== OVERLAY MODE =====
    if overlay:
        # Position
        overlay_pos = _overlay_coords(position)

        # Case A: Full GPU path (no alpha + backend overlay filter exists)
        if backend and alpha >= 1.0 and have_gpu_overlay:
            pre = []
            if setpts:
                pre.append(setpts)

            # For VideoToolbox, upload both inputs to GPU
            if backend.needs_upload:
                # Upload main video to GPU
                pre.append("[0:v]hwupload=derive_device=videotoolbox[main_hw]")
                # Upload plot video to GPU (handling setpts label)
                if setpts:
                    pre.append(
                        f"{plot_lbl}hwupload=derive_device=videotoolbox[plot_hw]"
                    )
                else:
                    pre.append("[1:v]hwupload=derive_device=videotoolbox[plot_hw]")
                main_lbl = "[main_hw]"
                plot_lbl = "[plot_hw]"
            else:
                main_lbl = "[0:v]"
                # plot_lbl already set above

            # If you need to scale the overlay, prefer GPU scale to keep frames in hw.
            # Here we keep native size; add a scale if you want to enforce a specific overlay size.
            # Example: pre.append(f"{plot_lbl}{backend.scale_filter}=-2:1024[plot_hw]"); plot_lbl = "[plot_hw]"

            filter_complex = "; ".join(
                [*pre, f"{main_lbl}{plot_lbl}{backend.overlay_filter}={overlay_pos}"]
            )

            cmd = base + [
                "-i",
                str(video_path),
                "-i",
                str(plot_video_path),
                "-filter_complex",
                filter_complex,
                "-c:v",
                encoder,
                *encoder_args,
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

        # Case B: Alpha blending or no GPU overlay → GPU scale (if possible) then CPU overlay
        else:
            pre = []
            if setpts:
                pre.append(setpts)

            # If we have a backend, do GPU scale first to save CPU, then download once.
            if backend:
                # For VideoToolbox, upload to GPU first
                if backend.needs_upload:
                    pre.append(
                        f"{plot_lbl}hwupload=derive_device=videotoolbox[plot_hw]"
                    )
                    plot_lbl = "[plot_hw]"

                # If you want to resize the overlay prior to compositing, do it here:
                # example keeps original size; uncomment to enforce a height:
                # pre.append(f"{plot_lbl}{backend.scale_filter}=-2:1024[plot_hw]"); plot_lbl = "[plot_hw]"
                # Now download to system memory for CPU overlay / alpha
                pre.append(f"{plot_lbl}hwdownload,format={backend.sw_format}[plot_sw]")
                plot_lbl = "[plot_sw]"

            # If alpha < 1, do RGBA + colorchannelmixer on CPU
            if alpha < 1.0:
                pre.append(
                    f"{plot_lbl}format=rgba,colorchannelmixer=aa={alpha}[overlay_src]"
                )
                plot_lbl = "[overlay_src]"

            filter_complex = "; ".join([*pre, f"[0:v]{plot_lbl}overlay={overlay_pos}"])

            cmd = base + [
                "-i",
                str(video_path),
                "-i",
                str(plot_video_path),
                "-filter_complex",
                filter_complex,
                "-c:v",
                encoder,
                *encoder_args,
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

    # ===== STACK MODE (left/right/top/bottom) =====
    else:
        if position not in ("top", "bottom", "left", "right"):
            raise ValueError(
                f"Position '{position}' not valid for side-by-side mode. Use 'top', 'bottom', 'left', or 'right'."
            )

        pre = []
        if setpts:
            pre.append(setpts)

        # For stack operations, decide whether to use GPU or CPU processing
        # On VideoToolbox, CPU-only can be faster due to avoiding transfer overhead
        use_cpu_only = cpu or force_cpu_for_stack

        if use_cpu_only or not backend:
            # Pure CPU path for filtering - avoid GPU entirely to skip transfer overhead
            # But we can still use hardware encoder for final encoding step

            print("Using CPU-only path for stacking to avoid GPU-CPU transfer overhead")
            print(f"Final encoder: {encoder}")

            # CPU scaling
            if position in ("top", "bottom"):
                pre.append(
                    f"{plot_lbl}scale=iw*sar:ih,setsar=1,scale=-1:{video_height}[plot_sw]"
                )
            else:
                pre.append(
                    f"{plot_lbl}scale=iw*sar:ih,setsar=1,scale=-2:{video_height}[plot_sw]"
                )
            plot_lbl = "[plot_sw]"

            # Keep the original encoder selection (could be hardware encoder)
            # encoder and encoder_args were already set above
        else:
            # Original GPU->CPU hybrid path for other backends where transfer cost is lower
            print(f"Using GPU scaling + CPU stacking path with {backend.name}")

            # For VideoToolbox, upload to GPU first
            if backend.needs_upload:
                pre.append(
                    f"{plot_lbl}hwupload=derive_device=videotoolbox[plot_uploaded]"
                )
                plot_lbl = "[plot_uploaded]"

            # VideoToolbox scale_vt doesn't handle -2 (auto-calc) well, use software scaling
            if backend.needs_upload:
                # Download first, then use CPU scaling which handles -2 properly
                pre.append(
                    f"{plot_lbl}hwdownload,format={backend.sw_format}[plot_downloaded]"
                )
                if position in ("top", "bottom"):
                    # Match widths to a fixed target (avoid runtime dependent sizing)
                    pre.append(f"[plot_downloaded]scale=-1:{video_height}[plot_sw]")
                else:
                    # Match heights
                    pre.append(f"[plot_downloaded]scale=-2:{video_height}[plot_sw]")
                plot_lbl = "[plot_sw]"
            else:
                # CUDA/QSV/VAAPI can handle -2 in hardware scaling
                if position in ("top", "bottom"):
                    # Match widths to a fixed target (avoid runtime dependent sizing)
                    pre.append(
                        f"{plot_lbl}{backend.scale_filter}=-1:{video_height}[plot_hw]"
                    )
                else:
                    # Match heights
                    pre.append(
                        f"{plot_lbl}{backend.scale_filter}=-2:{video_height}[plot_hw]"
                    )
                plot_lbl = "[plot_hw]"
                pre.append(f"{plot_lbl}hwdownload,format={backend.sw_format}[plot_sw]")
                plot_lbl = "[plot_sw]"

        # Stack
        if position == "bottom":
            stack = f"[0:v]{plot_lbl}vstack=inputs=2"
        elif position == "top":
            stack = f"{plot_lbl}[0:v]vstack=inputs=2"
        elif position == "right":
            stack = f"[0:v]{plot_lbl}hstack=inputs=2"
        else:  # left
            stack = f"{plot_lbl}[0:v]hstack=inputs=2"

        filter_complex = "; ".join([*pre, stack])

        # Build ffmpeg command
        # For CPU-only filtering path, we still exclude GPU hw flags but can use HW encoder
        cmd_base = base
        if use_cpu_only and backend:
            # Remove GPU acceleration flags but keep device init for encoder
            cmd_base = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                COMBINE_FFMPEG_LOGLEVEL,
                "-stats",
            ]
            # Only add device init if needed for encoder (VideoToolbox needs it)
            if backend.name == "videotoolbox":
                cmd_base.extend(["-init_hw_device", "videotoolbox=vt"])

        cmd = cmd_base + [
            "-i",
            str(video_path),
            "-i",
            str(plot_video_path),
            "-filter_complex",
            filter_complex,
            "-c:v",
            encoder,
            *encoder_args,
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

    print(f"Running ffmpeg command:{" ".join(cmd)}")
    try:
        # Run ffmpeg (inherit stdio → progress bar; raise on error)
        subprocess.run(cmd, check=True)
        print(f"Successfully combined videos: {output_path}")
        return True

    except subprocess.CalledProcessError:
        print("ffmpeg failed.")
        return False
