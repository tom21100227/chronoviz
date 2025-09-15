import platform, subprocess, shutil
import numpy as np
from numba import njit
from typing import Optional, Tuple, List
import warnings


def get_video_height(video_path: str) -> int:
    """Get the height of a video using ffprobe."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=height",
                "-of",
                "csv=s=x:p=0",
                str(video_path),
            ],
            text=True,
        )
        return int(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        warnings.warn(f"Could not get video height for {video_path}: {e}", RuntimeWarning)
        return 0


def _global_ylim(
    sig: np.ndarray, ylim: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """Compute global y-limits for a signal array."""
    if ylim is not None:
        return float(ylim[0]), float(ylim[1])
    finite = np.isfinite(sig)
    if not finite.any():
        return -1.0, 1.0
    lo, hi = float(np.min(sig[finite])), float(np.max(sig[finite]))
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    else:
        margin = (hi - lo) * 0.1
        lo, hi = lo - margin, hi + margin
    return lo, hi


def _compute_ylim(
    y: np.ndarray, ylim: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    if ylim is not None:
        pad = 0.05 * (ylim[0] - ylim[1])
        return float(ylim[0]) + pad, float(ylim[1]) - pad  # bleeding room
    finite = np.isfinite(y)
    if not finite.any():
        return -1.0, 1.0
    lo, hi = float(np.min(y[finite])), float(np.max(y[finite]))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _ffmpeg_has_encoder(name: str) -> bool:
    # Parse `ffmpeg -hide_banner -encoders` once; cache if you want
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True
        )
    except Exception:
        return False
    return any((" " + name) in line for line in out.splitlines())

def _ffmpeg_supports_filter(name: str) -> bool:
    """Return True if `ffmpeg -filters` lists the given filter name."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    try:
        out = subprocess.run(
            [ff, "-hide_banner", "-v", "0", "-filters"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return any(line.split()[1] == name for line in out.splitlines()[8:] if line.strip())
    except Exception:
        warnings(
            f"Could not determine if ffmpeg supports filter '{name}'; assuming not.",
            RuntimeWarning
        )
        return False

def pick_video_encoder(alpha: bool, cpu: bool = False) -> Tuple[str, List[str]]:
    """
    Returns (encoder_name, extra_args) for video.
    alpha=False -> yuv420p, alpha=True -> yuva420p capable only with VP9/AV1.
    """
    sys = platform.system().lower()

    if cpu:
        # Force software encoding
        if alpha and _ffmpeg_has_encoder("libvpx-vp9"):
            return "libvpx-vp9", [
                "-b:v",
                "0",
                "-crf",
                "28",
                "-row-mt",
                "1",
                "-pix_fmt",
                "yuva420p",
            ]
        return "libx264", ["-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p"]

    # macOS hardware (Metal/VideoToolbox)
    if sys == "darwin" and _ffmpeg_has_encoder("h264_videotoolbox") and not alpha:
        return "h264_videotoolbox", ["-b:v", "6M", "-pix_fmt", "yuv420p"]

    # NVIDIA (CUDA/NVENC)
    if _ffmpeg_has_encoder("h264_nvenc") and not alpha:
        # p7 fastest, p1 highest quality (ish). p6 is a nice speed/quality point.
        return "h264_nvenc", [
            "-preset",
            "p6",
            "-rc",
            "vbr",
            "-b:v",
            "6M",
            "-pix_fmt",
            "yuv420p",
        ]

    # Intel Quick Sync
    if _ffmpeg_has_encoder("h264_qsv") and not alpha:
        return "h264_qsv", ["-global_quality", "22", "-pix_fmt", "yuv420p"]

    # AMD AMF
    if _ffmpeg_has_encoder("h264_amf") and not alpha:
        return "h264_amf", ["-quality", "speed", "-pix_fmt", "yuv420p"]

    # If alpha requested, safest wide-availability is VP9
    if alpha and _ffmpeg_has_encoder("libvpx-vp9"):
        return "libvpx-vp9", [
            "-b:v",
            "0",
            "-crf",
            "28",
            "-row-mt",
            "1",
            "-pix_fmt",
            "yuva420p",
        ]

    # Fallback: software x264
    return "libx264", ["-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]


@njit(cache=True)
def clamp_nan_2d(sig, lo, hi):
    """In-place: set out-of-range to NaN. sig shape = (N,C) or (N,1)."""
    N = sig.shape[0]
    C = 1 if sig.ndim == 1 else sig.shape[1]
    if sig.ndim == 1:
        for i in range(N):
            v = sig[i]
            if v >= hi:
                sig[i] = hi - 0.01
            elif v <= lo:
                sig[i] = lo + 0.01
    else:
        for c in range(C):
            for i in range(N):
                v = sig[i, c]
                if v >= hi:
                    sig[i, c] = hi - 0.01
                elif v <= lo:
                    sig[i, c] = lo + 0.01

from dataclasses import dataclass

@dataclass
class GpuBackend:
    name: str
    hw_flags: list[str]
    scale_filter: str
    overlay_filter: str | None
    encoder: str
    enc_args: list[str]
    sw_format: str  # for hwdownload format before CPU filters
    needs_upload: bool = False  # VideoToolbox needs CPU->GPU upload

BACKENDS = [
    GpuBackend(
        name="cuda",
        hw_flags=["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
        scale_filter="scale_npp",
        overlay_filter="overlay_cuda",
        encoder="h264_nvenc",
        enc_args=["-preset", "p2", "-b:v", "6M", "-maxrate", "6M", "-bufsize", "12M", "-g", "240", "-bf", "2", "-pix_fmt", "yuv420p"],
        sw_format="nv12",
    ),
    GpuBackend(
        name="qsv",
        hw_flags=["-init_hw_device", "qsv=hw", "-filter_hw_device", "hw", "-hwaccel", "qsv"],
        scale_filter="scale_qsv",
        overlay_filter="overlay_qsv",
        encoder="h264_qsv",
        enc_args=["-global_quality", "23", "-g", "240", "-bf", "2", "-pix_fmt", "yuv420p"],
        sw_format="nv12",
    ),
    GpuBackend(
        name="vaapi",
        hw_flags=["-vaapi_device", "/dev/dri/renderD128", "-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"],
        scale_filter="scale_vaapi",
        overlay_filter="overlay_vaapi",
        encoder="h264_vaapi",
        enc_args=["-b:v", "6M", "-maxrate", "6M", "-bufsize", "12M", "-g", "240", "-bf", "2", "-pix_fmt", "nv12"],
        sw_format="nv12",
    ),
    GpuBackend(
        name="videotoolbox",
        hw_flags=["-init_hw_device", "videotoolbox=vt"],  # Just init the device
        scale_filter="scale_vt",
        overlay_filter="overlay_videotoolbox",
        encoder="h264_videotoolbox",
        enc_args=["-b:v", "6M", "-maxrate", "6M", "-bufsize", "12M", "-g", "240", "-bf", "0", "-profile:v", "high", "-level", "4.2", "-realtime", "true", "-allow_sw", "0", "-pix_fmt", "yuv420p"],
        sw_format="yuv420p",  # VideoToolbox uses yuv420p, not nv12
        needs_upload=True,  # VideoToolbox needs CPU->GPU upload
    ),
]
