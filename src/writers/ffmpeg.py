"""
FFmpeg-based video writer implementation.
"""

import subprocess
import os
from pathlib import Path
from typing import Protocol, List
from ..utils import pick_video_encoder, _has_cmd

DEFAULT_FFMPEG_LOGLEVEL = "warning"


class FrameWriter(Protocol):
    """Protocol for writing video frames to different backends."""

    def write_frame(self, frame_data: memoryview) -> None:
        """Write a single frame of video data."""
        ...

    def close(self) -> Path:
        """Close the writer and return the output path."""
        ...


class FFmpegWriter:
    """Writer that pipes frames directly to FFmpeg via stdin."""

    def __init__(
        self,
        out_path: Path,
        w: int,
        h: int,
        fps: float,
        alpha: bool,
        input_pix: str,
        use_cpu: bool = True,
    ):
        if not _has_cmd("ffmpeg"):
            raise RuntimeError("ffmpeg not found on PATH.")
        self.out_path = self._ensure_correct_extension(out_path, alpha)
        cmd = self._build_ffmpeg_cmd(w, h, fps, alpha, input_pix, use_cpu)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.fd = self.proc.stdin.fileno()

    def _ensure_correct_extension(self, out_path: Path, alpha: bool) -> Path:
        """Ensure output path has correct extension for alpha/non-alpha."""
        out_path = Path(out_path)
        if alpha:
            if out_path.suffix.lower() != ".webm":
                out_path = out_path.with_suffix(".webm")
        else:
            if out_path.suffix.lower() != ".mp4":
                out_path = out_path.with_suffix(".mp4")
        return out_path

    def _build_ffmpeg_cmd(
        self, w: int, h: int, fps: float, alpha: bool, input_pix: str, use_cpu: bool
    ) -> List[str]:
        """Build the FFmpeg command arguments."""
        enc, enc_args = pick_video_encoder(alpha, cpu=use_cpu)
        return [
            "ffmpeg",
            "-loglevel",
            DEFAULT_FFMPEG_LOGLEVEL,
            "-stats",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            input_pix,
            "-s",
            f"{w}x{h}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            enc,
            *enc_args,
            str(self.out_path),
        ]

    def write_frame(self, frame_data: memoryview) -> None:
        """Write frame data directly to FFmpeg stdin (fast path)."""
        os.write(self.fd, frame_data)

    def close(self) -> Path:
        """Close FFmpeg process and return output path."""
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()
        return self.out_path


def create_ffmpeg_writer(
    out_path: Path,
    w: int,
    h: int,
    fps: float,
    alpha: bool,
    use_cpu: bool = True,
    use_rgb24: bool = False,
) -> FFmpegWriter:
    """Create an FFmpeg writer with the appropriate pixel format."""
    input_pix = "rgb24" if use_rgb24 else "rgba"
    return FFmpegWriter(out_path, w, h, fps, alpha, input_pix, use_cpu)
