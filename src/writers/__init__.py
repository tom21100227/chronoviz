"""
Video writers for different backends.
"""

from .ffmpeg import FFmpegWriter, FrameWriter, create_ffmpeg_writer

__all__ = ["FFmpegWriter", "FrameWriter", "create_ffmpeg_writer"]
