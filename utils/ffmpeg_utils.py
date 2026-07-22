"""
MovieShort AI — FFmpeg utilities
"""
from pathlib import Path
from typing import Optional, Union
import re
import shutil
import subprocess

from config import VERTICAL_WIDTH, VERTICAL_HEIGHT, BANNER_TOP, BANNER_BOTTOM, ANTI_COPYRIGHT, AC_MIRROR, AC_CONTRAST, AC_BRIGHTNESS, AC_SATURATION, SUBTITLE_FONT, SUBTITLE_SIZE, SUBTITLE_COLOR, SUBTITLE_OUTLINE, SUBTITLE_BOLD, SUBTITLE_ITALIC, SUBTITLE_SHADOW


class FFmpegError(Exception):
    """Raised when an FFmpeg/FFprobe command fails."""


_GPU_ACCEL = None  # cached result

def _detect_gpu_accel():
    """Detect NVIDIA GPU availability for FFmpeg HW acceleration.
    
    Returns dict with hwaccel config or None if unavailable.
    Caches result in module global.
    """
    global _GPU_ACCEL
    if _GPU_ACCEL is not None:
        return _GPU_ACCEL
    
    try:
        # Check nvidia-smi
        subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5, check=True)
        # Check FFmpeg NVENC encoder
        encoders = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
        )
        has_nvenc = "nvenc" in encoders.stdout
        # Check FFmpeg NVDEC decoder
        decoders = subprocess.run(
            ["ffmpeg", "-decoders"], capture_output=True, text=True, timeout=10
        )
        has_nvdec = "nvdec" in decoders.stdout
        
        if has_nvenc and has_nvdec:
            _GPU_ACCEL = {"hwaccel": "cuda", "encoder": "h264_nvenc", "decoder": "h264_cuvid"}
            print("  GPU acceleration detected: NVENC+NVDEC available")
            return _GPU_ACCEL
        else:
            _GPU_ACCEL = None
            return None
    except (subprocess.SubprocessError, FileNotFoundError):
        _GPU_ACCEL = None
        return None


def _gpu_args(gpu_opts=None):
    """Build FFmpeg GPU acceleration arguments if available."""
    if gpu_opts is None:
        gpu_opts = _detect_gpu_accel()
    if gpu_opts:
        return ["-hwaccel", gpu_opts["hwaccel"], "-hwaccel_output_format", gpu_opts["hwaccel"]]
    return []


_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d+)?$")


def _validate_time(time_str: str, name: str) -> None:
    if not _TIME_RE.match(time_str):
        raise ValueError(
            f"{name} must be in HH:MM:SS format, got {time_str!r}"
        )
    h, m, s_part = int(time_str[:2]), int(time_str[3:5]), time_str[6:]
    # s_part could be "ss" or "ss.fff"
    s = int(float(s_part))
    if h > 23 or m > 59 or s > 59:
        raise ValueError(
            f"{name} has invalid time values "
            f"(got {h:02d}:{m:02d}:{s:02d})"
        )


def _validate_file(path: Union[str, Path], name: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{name} not found: {p}")
    return p


def _run(
    args: list,
    desc: str = "ffmpeg",
    cwd: Optional[Union[str, Path]] = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        errors="backslashreplace",
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode != 0:
        raise FFmpegError(
            f"{desc} failed (code {result.returncode}):\n{result.stderr.strip()}"
        )
    return result


def clip_video(
    input_path: Union[str, Path],
    start_time: str,
    end_time: str,
    output_path: Union[str, Path],
    gpu_opts=None,
) -> Path:
    _validate_file(input_path, "input_path")
    _validate_time(start_time, "start_time")
    _validate_time(end_time, "end_time")
    out = Path(output_path)
    gpu = _gpu_args(gpu_opts)
    cmd = ["ffmpeg", "-y"]
    if gpu:
        # GPU: -hwaccel before -i, input seeking (-ss before -i), nvenc encoder
        cmd.extend(gpu)
    cmd.extend(["-ss", start_time, "-to", end_time, "-i", str(input_path)])
    if gpu:
        cmd.extend(["-c:v", gpu_opts["encoder"], "-preset", "p1", "-cq", "23"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"])
    cmd.extend(["-c:a", "aac", str(out)])
    _run(cmd, desc="clip_video")
    return out


def embed_subtitles(
    video_path: Union[str, Path],
    subtitle_path: Union[str, Path],
    output_path: Union[str, Path],
    font_style: Optional[dict] = None,
    banner_top: int = BANNER_TOP,
    banner_bottom: int = BANNER_BOTTOM,
    gpu_opts=None,
) -> Path:
    _validate_file(video_path, "video_path")
    _validate_file(subtitle_path, "subtitle_path")
    out = Path(output_path)
    if font_style is None:
        font_style = {}
    fs = font_style
    # NOTE: MarginV intentionally omitted from force_style.
    # libass bug: MarginV >= 286 silently disables ALL subtitle rendering
    # (tested ffmpeg 8.1.2, libass 0.17.5). Without MarginV override,
    # subtitles default to the bottom of the content area — correct position
    # since blur_background overlays content just above the bottom banner.
    content_h = VERTICAL_HEIGHT - banner_top - banner_bottom
    style = (
        f"FontName={fs.get('font', SUBTITLE_FONT)},"
        f"FontSize={fs.get('size', SUBTITLE_SIZE)},"
        f"PrimaryColour={fs.get('color', SUBTITLE_COLOR)},"
        f"OutlineColour=&H00000000,"
        f"BackColour=&H00000000,"
        f"BorderStyle=1,"
        f"Outline={fs.get('outline', SUBTITLE_OUTLINE)},"
        f"Shadow={1 if fs.get('shadow', SUBTITLE_SHADOW) else 0},"
        f"Bold={1 if fs.get('bold', SUBTITLE_BOLD) else 0},"
        f"Italic={1 if fs.get('italic', SUBTITLE_ITALIC) else 0}"
    )
    # Copy SRT to output directory so we can reference it by filename
    # (avoids Windows drive-letter colons breaking ffmpeg's filter-parser).
    out_dir = out.parent
    local_srt = out_dir / f"{out.stem}.srt"
    shutil.copy2(subtitle_path, local_srt)
    filter_str = (
        "subtitles={}:force_style='{}':original_size=1080x{}"
    ).format(local_srt.name, style, content_h)
    gpu = _gpu_args(gpu_opts)
    cmd = ["ffmpeg", "-y"]
    if gpu:
        cmd.extend(gpu)
    cmd.extend([
        "-i", str(video_path),
        "-vf", filter_str,
        "-c:a", "copy",
        str(out),
    ])
    _run(cmd, desc="embed_subtitles", cwd=out_dir)
    # Clean up the temporary SRT copy
    local_srt.unlink(missing_ok=True)
    return out


def convert_to_vertical(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    anti_copyright: bool = True,
    banner_top: int = BANNER_TOP,
    banner_bottom: int = BANNER_BOTTOM,
    gpu_opts=None,
) -> Path:
    """Scale video to fill the content area, cropping overflow.

    Output is 1080 × content_h, zoomed so the content area is fully covered.
    Banner padding and subtitles are added in subsequent pipeline steps.
    """
    _validate_file(video_path, "video_path")
    out = Path(output_path)
    content_h = VERTICAL_HEIGHT - banner_top - banner_bottom
    ac_filters = []
    if anti_copyright:
        if AC_MIRROR:
            ac_filters.append("hflip")
        if AC_CONTRAST != 1.0 or AC_BRIGHTNESS != 0.0 or AC_SATURATION != 1.0:
            ac_filters.append(
                f"eq=contrast={AC_CONTRAST}:brightness={AC_BRIGHTNESS}:saturation={AC_SATURATION}"
            )
    ac_part = ",".join(ac_filters)
    ac_part = f",{ac_part}" if ac_part else ""
    filter_str = (
        "scale={}:{}:force_original_aspect_ratio=increase,"
        "crop={}:{}{}"
    ).format(VERTICAL_WIDTH, content_h,
             VERTICAL_WIDTH, content_h,
             ac_part)
    gpu = _gpu_args(gpu_opts)
    cmd = ["ffmpeg", "-y"]
    if gpu:
        cmd.extend(gpu)
    cmd.extend([
        "-i", str(video_path),
        "-vf", filter_str,
        "-c:a", "copy",
        str(out),
    ])
    _run(cmd, desc="convert_to_vertical")
    return out


def blur_background(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    enabled: bool = True,
    banner_top: int = BANNER_TOP,
    banner_bottom: int = BANNER_BOTTOM,
    gpu_opts=None,
) -> Path:
    """Blurred background effect: fills 1080×1920 with blurred video, clear fg centered.

    Takes a content-area video (1080 × content_h with subtitles already embedded)
    and produces a full 9:16 frame where:
    - Background: the same video scaled to fill 1080×1920 and heavily blurred
    - Foreground: the original clear video centered in the content area

    When enabled=False, just copies the input (no blur effect).
    """
    _validate_file(video_path, "video_path")
    out = Path(output_path)
    if not enabled:
        shutil.copy2(str(video_path), str(out))
        return out
    filter_complex = (
        "[0:v]scale={}:{}:force_original_aspect_ratio=increase,"
        "crop={}:{},boxblur=20:5[bg];"
        "[bg][0:v]overlay=0:{}"
    ).format(VERTICAL_WIDTH, VERTICAL_HEIGHT,
             VERTICAL_WIDTH, VERTICAL_HEIGHT,
             banner_top)
    gpu = _gpu_args(gpu_opts)
    cmd = ["ffmpeg", "-y"]
    if gpu:
        cmd.extend(gpu)
    cmd.extend([
        "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-c:a", "copy",
        str(out),
    ])
    _run(cmd, desc="blur_background")
    return out


def pad_with_banners(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    banner_top: int = BANNER_TOP,
    banner_bottom: int = BANNER_BOTTOM,
    gpu_opts=None,
) -> Path:
    """Pad a content-area video (1080 × content_h) to full 9:16 (1080 × 1920).

    Adds BANNER_TOP pixels of black at the top and BANNER_BOTTOM at the bottom,
    centering the input video within the content area.
    """
    _validate_file(video_path, "video_path")
    out = Path(output_path)
    pad_y = f"{banner_top}+(({VERTICAL_HEIGHT}-{banner_top}-{banner_bottom})-ih)/2"
    filter_str = "pad={}:{}:(ow-iw)/2:{}:black".format(
        VERTICAL_WIDTH, VERTICAL_HEIGHT, pad_y)
    gpu = _gpu_args(gpu_opts)
    cmd = ["ffmpeg", "-y"]
    if gpu:
        cmd.extend(gpu)
    cmd.extend([
        "-i", str(video_path),
        "-vf", filter_str,
        "-c:a", "copy",
        str(out),
    ])
    _run(cmd, desc="pad_with_banners")
    return out



