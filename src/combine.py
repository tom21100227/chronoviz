"""
Module for combining original videos with generated plot videos.
"""

from pathlib import Path
import subprocess
from .utils import pick_video_encoder

COMBINE_FFMPEG_LOGLEVEL = "warning"


def combine_videos(
    video_path: str | Path,
    plot_video_path: str | Path,
    output_path: str | Path,
    ratio: float = 1.0,
    position: str = "bottom",  # "top", "bottom", "left", "right"
    overlay: bool = False,
    alpha: float = 0.5,
) -> bool:
    """
    Combine original video with plot video.

    Parameters:
        - video_path: Path to the original video file.
        - plot_video_path: Path to the generated plot video file.
        - output_path: Path to save the combined video.
        - ratio: fps ratio between plot video and original video. 2.0 means plot video is twice the fps of original video.
        - position: Position of the plot video relative to the original ("top", "bottom", "left", "right", "tr", "tl", "br", bl").
        - overlay: If True, overlay the plot on top of the original video. Note: position "tr", "tl", "br", "bl" only valid when overlay is True.
        - alpha: Transparency level for overlay (0.0 to 1.0).

    Returns:
        - True if successful, False otherwise.
    """
    video_path = Path(video_path)
    plot_video_path = Path(plot_video_path)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Determine encoder and settings (prefer hardware acceleration)
        encoder, encoder_args = pick_video_encoder(
            alpha=overlay and alpha < 1.0, cpu=True
        )

        # Handle fps ratio adjustment for plot video
        # If ratio > 1.0, plot video is faster than original, so we need to slow it down
        # If ratio < 1.0, plot video is slower than original, so we need to speed it up
        if ratio != 1.0:
            # setpts multiplier: larger values = slower playback, smaller = faster
            setpts_multiplier = ratio
            plot_filter = f"[1:v]setpts={setpts_multiplier}*PTS[plot_adjusted]"
            plot_input_label = "[plot_adjusted]"
        else:
            plot_filter = ""
            plot_input_label = "[1:v]"

        if overlay:
            # Overlay mode - plot video overlays on top of original
            if position == "tr":  # top-right
                overlay_pos = "main_w-overlay_w-10:10"
            elif position == "tl":  # top-left
                overlay_pos = "10:10"
            elif position == "br":  # bottom-right
                overlay_pos = "main_w-overlay_w-10:main_h-overlay_h-10"
            elif position == "bl":  # bottom-left
                overlay_pos = "10:main_h-overlay_h-10"
            else:
                raise ValueError(
                    f"Position '{position}' not valid for overlay mode. Use 'tr', 'tl', 'br', or 'bl'."
                )

            # Build filter complex for overlay with optional fps adjustment and alpha
            if plot_filter:
                if alpha < 1.0:
                    filter_complex = f"{plot_filter}; {plot_input_label}format=rgba,colorchannelmixer=aa={alpha}[overlay]; [0:v][overlay]overlay={overlay_pos}"
                else:
                    filter_complex = (
                        f"{plot_filter}; [0:v]{plot_input_label}overlay={overlay_pos}"
                    )
            else:
                if alpha < 1.0:
                    filter_complex = f"[1:v]format=rgba,colorchannelmixer=aa={alpha}[overlay]; [0:v][overlay]overlay={overlay_pos}"
                else:
                    filter_complex = f"[0:v][1:v]overlay={overlay_pos}"

            # Build ffmpeg command for overlay
            cmd = [
                "ffmpeg",
                "-y",  # -y to overwrite output file
                "-i",
                str(video_path),  # Main video input
                "-i",
                str(plot_video_path),  # Plot video input (overlay)
                "-filter_complex",
                filter_complex,
                "-c:v",
                encoder,  # Use selected encoder
                *encoder_args,  # Add encoder-specific arguments
                "-c:a",
                "copy",  # Copy audio from main video
                str(output_path),
            ]
        else:
            # Side-by-side/stacked mode - need to handle dimension compatibility
            if position in ["bottom", "top"]:
                # Vertical stacking: both videos need same width
                # Scale plot video to match original video's width
                if plot_filter:
                    # Chain with existing fps adjustment
                    scaling_filter = f"{plot_filter}; {plot_input_label}scale=iw*sar:ih,setsar=1,scale=1280:-2[plot_scaled]"
                    plot_input_label = "[plot_scaled]"
                else:
                    scaling_filter = (
                        "[1:v]scale=iw*sar:ih,setsar=1,scale=1280:-2[plot_scaled]"
                    )
                    plot_input_label = "[plot_scaled]"

                if position == "bottom":
                    # Stack vertically: original on top, plot on bottom
                    stack_filter = f"[0:v]{plot_input_label}vstack=inputs=2"
                else:  # position == "top"
                    # Stack vertically: plot on top, original on bottom
                    stack_filter = f"{plot_input_label}[0:v]vstack=inputs=2"

                filter_complex = f"{scaling_filter}; {stack_filter}"

            elif position in ["left", "right"]:
                # Horizontal stacking: both videos need same height
                # Scale plot video to match original video's height
                if plot_filter:
                    # Chain with existing fps adjustment
                    scaling_filter = f"{plot_filter}; {plot_input_label}scale=iw*sar:ih,setsar=1,scale=-2:1024[plot_scaled]"
                    plot_input_label = "[plot_scaled]"
                else:
                    scaling_filter = (
                        "[1:v]scale=iw*sar:ih,setsar=1,scale=-2:1024[plot_scaled]"
                    )
                    plot_input_label = "[plot_scaled]"

                if position == "right":
                    # Stack horizontally: original on left, plot on right
                    stack_filter = f"[0:v]{plot_input_label}hstack=inputs=2"
                else:  # position == "left"
                    # Stack horizontally: plot on left, original on right
                    stack_filter = f"{plot_input_label}[0:v]hstack=inputs=2"

                filter_complex = f"{scaling_filter}; {stack_filter}"
            else:
                raise ValueError(
                    f"Position '{position}' not valid for side-by-side mode. Use 'top', 'bottom', 'left', or 'right'."
                )

            # Build ffmpeg command for stacking
            cmd = [
                "ffmpeg",
                "-y",  # -y to overwrite output file
                "-loglevel",
                COMBINE_FFMPEG_LOGLEVEL,
                "-stats",
                "-i",
                str(video_path),  # Main video input
                "-i",
                str(plot_video_path),  # Plot video input
                "-filter_complex",
                filter_complex,
                "-c:v",
                encoder,  # Use selected encoder
                *encoder_args,  # Add encoder-specific arguments
                "-c:a",
                "copy",  # Copy audio from main video
                str(output_path),
            ]

        # Execute ffmpeg command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return False

        print(f"Successfully combined videos: {output_path}")
        return True

    except Exception as e:
        print(f"Error combining videos: {str(e)}")
        return False
