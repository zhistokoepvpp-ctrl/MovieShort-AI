"""
MovieShort AI — FFmpeg utilities
"""
from pathlib import Path
from typing import Optional, Union
import hashlib
import os
import re
import shutil
import subprocess
import time

from config import VERTICAL_WIDTH, VERTICAL_HEIGHT, BANNER_TOP, BANNER_BOTTOM, ANTI_COPYRIGHT, AC_MIRROR, AC_CONTRAST, AC_BRIGHTNESS, AC_SATURATION, SUBTITLE_FONT, SUBTITLE_SIZE, SUBTITLE_COLOR, SUBTITLE_OUTLINE, SUBTITLE_BOLD, SUBTITLE_ITALIC, SUBTITLE_SHADOW
from utils.font_manager import FONTS_DIR


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
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            errors="backslashreplace",
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"{desc} timed out after {timeout}s") from e
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
    """Cut a video clip from start_time to end_time.

    Uses input seeking (-ss before -i) with **software decode** for fast
    and frame-accurate seeking across all container/codec combinations.
    Duration is specified via -t (not -to) for deterministic behavior.
    GPU hardware decoding is NOT used — software decode ensures correct
    timestamp handling. GPU encoding (nvenc) IS used when available.
    """
    _validate_file(input_path, "input_path")
    _validate_time(start_time, "start_time")
    _validate_time(end_time, "end_time")
    out = Path(output_path)

    # Compute duration as -t (more deterministic than -to after -i)
    def _to_sec(t: str) -> float:
        parts = t.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    duration = _to_sec(end_time) - _to_sec(start_time)
    if duration <= 0:
        raise ValueError(
            f"end_time ({end_time}) must be after start_time ({start_time})"
        )

    # Input seeking (fast, keyframe-based) with software decode
    # NO -hwaccel cuda — software decode is reliable across all containers
    cmd = ["ffmpeg", "-y",
           "-ss", start_time, "-i", str(input_path),
           "-t", f"{duration:.3f}"]
    if gpu_opts:
        # GPU encoding only (nvenc) — decode is software
        cmd.extend(["-c:v", gpu_opts["encoder"], "-preset", "p1", "-cq", "23"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"])
    cmd.extend(["-c:a", "aac", str(out)])
    _run(cmd, desc="clip_video")
    return out


def _build_style_string(fs: dict) -> str:
    """Build the libass force_style string from a font_style dict.

    Shared by embed_subtitles (production) and render_full_preview
    so the two filter strings cannot drift. MarginV intentionally omitted —
    see the note in embed_subtitles.
    """
    return (
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


def embed_subtitles(
    video_path: Union[str, Path],
    subtitle_path: Union[str, Path],
    output_path: Union[str, Path],
    font_style: Optional[dict] = None,
    banner_top: int = BANNER_TOP,
    banner_bottom: int = BANNER_BOTTOM,
    gpu_opts=None,
    fontsdir: Optional[Union[str, Path]] = None,
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
    style = _build_style_string(fs)
    # Copy subtitle file to output directory under a FILTER-SAFE name.
    # out.stem (todo-1 clip names) contains ", " and "#" — an unquoted comma
    # splits ffmpeg's filtergraph ("No such filter: '...'"), so the copy gets
    # a content-hashed name instead of deriving from out.stem.
    # Suffix generalized: .srt and .ass — libass renders ASS natively, and
    # force_style overrides style-level params but inline {\fad}/{\move} tags survive.
    out_dir = out.parent
    safe_stem = hashlib.md5(out.stem.encode("utf-8")).hexdigest()[:12]
    local_srt = out_dir / f"{safe_stem}{Path(subtitle_path).suffix}"
    shutil.copy2(subtitle_path, local_srt)
    filter_str = (
        "subtitles={}:force_style='{}':original_size=1080x{}"
    ).format(local_srt.name, style, content_h)
    if fontsdir:
        # libass font lookup. An ABSOLUTE Windows path carries a drive colon,
        # which ffmpeg 8.1's filtergraph option splitter eats even when
        # single-quoted or backslash-escaped (empirically verified) → pass a
        # path RELATIVE to the ffmpeg cwd (= out_dir), forward slashes, quoted.
        try:
            fonts_arg = os.path.relpath(os.path.abspath(fontsdir), out_dir)
        except ValueError:  # fonts on a different drive than the output
            fonts_arg = os.path.abspath(fontsdir)
        filter_str += ":fontsdir='{}'".format(fonts_arg.replace("\\", "/"))
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
    # Clean up the temporary subtitle copy
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


def _probe_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise FFmpegError(
            f"ffprobe failed (code {result.returncode}):\n{result.stderr.strip()}"
        )
    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise FFmpegError(f"ffprobe returned no duration for {video_path}") from e


def _probe_height(video_path: Path) -> int:
    """Return video display height in pixels via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise FFmpegError(
            f"ffprobe failed (code {result.returncode}):\n{result.stderr.strip()}"
        )
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError) as e:
        raise FFmpegError(f"ffprobe returned no height for {video_path}") from e


def render_full_preview(
    video_path: Union[str, Path],
    srt_or_ass_path: Optional[Union[str, Path]],
    style_params: Optional[dict],
    options: dict,
    output_dir: Union[str, Path],
) -> str:
    """Render the FULL test video through the complete production chain
    (mirrors core/pipeline.py process_clip steps 3-5) and return the
    output .mp4 path.

    Chain:
      1. vertical: face_tracking → processor.apply_vertical_crop (lazy
         function-body import — core.processor imports ffmpeg_utils, a
         module-level import here would be circular); else
         convert_to_vertical(anti_copyright=...).
      2. subtitles: embed_subtitles with the editor style —
         skipped when srt_or_ass_path is None/empty.
      3. final frame: blur_background(enabled=blur) | pad_with_banners.

    NO -ss and NO -t anywhere: full source duration flows through, so the
    preview is exactly what production would produce for this settings set.

    options keys: banner_top:int, banner_bottom:int, blur:bool,
    anti_copyright:bool, face_tracking:bool.

    Timeout budget is max(180, duration*8)s per stage (computed once from
    _probe_duration). The chain helpers (convert_to_vertical /
    embed_subtitles / blur_background / pad_with_banners /
    apply_vertical_crop) do not expose a timeout parameter, so the budget
    cannot be threaded through them — they rely on their internal _run
    behavior; this function makes no direct ffmpeg subprocess calls itself.

    Intermediate files live in output_dir under unique timestamped names
    and are removed in a finally block.
    """
    # os.path.abspath, NOT Path.resolve() — Python 3.9/Windows resolve() of a
    # NONEXISTENT relative path returns it unchanged (still relative).
    video_path = os.path.abspath(video_path)
    _validate_file(video_path, "video_path")
    if srt_or_ass_path:
        srt_or_ass_path = os.path.abspath(srt_or_ass_path)
        _validate_file(srt_or_ass_path, "srt_or_ass_path")
    out_dir = Path(os.path.abspath(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    banner_top = int(options.get("banner_top", BANNER_TOP))
    banner_bottom = int(options.get("banner_bottom", BANNER_BOTTOM))
    blur = bool(options.get("blur", True))
    anti_copyright = bool(options.get("anti_copyright", ANTI_COPYRIGHT))
    face_tracking = bool(options.get("face_tracking", False))

    gpu_opts = _detect_gpu_accel()
    # Timeout budget per stage would be max(180, int(duration * 8))s, but the
    # chain helpers expose no timeout parameter (see docstring) — no direct
    # ffmpeg subprocess calls are made here, so nothing to thread it into.

    # Timestamped names: reruns never collide with a file Gradio is still serving.
    base = f"{Path(video_path).stem}_preview_{int(time.time() * 1000)}"
    tmp_vert = out_dir / f"{base}_vert.mp4"
    tmp_subs = out_dir / f"{base}_subs.mp4"
    out_mp4 = out_dir / f"{base}.mp4"
    try:
        # Step 1: vertical content area (1080 x content_h)
        if face_tracking:
            # Lazy import to avoid circular import (processor → ffmpeg_utils).
            from core.processor import apply_vertical_crop
            apply_vertical_crop(
                video_path, str(tmp_vert),
                anti_copyright=anti_copyright,
                banner_top=banner_top, banner_bottom=banner_bottom)
        else:
            convert_to_vertical(
                video_path, tmp_vert,
                anti_copyright=anti_copyright,
                banner_top=banner_top, banner_bottom=banner_bottom,
                gpu_opts=gpu_opts)

        # Step 2: subtitles on the content-area video (before banner padding)
        content_src = tmp_vert
        if srt_or_ass_path:
            embed_subtitles(
                tmp_vert, srt_or_ass_path, tmp_subs,
                font_style=style_params,
                banner_top=banner_top, banner_bottom=banner_bottom,
                gpu_opts=gpu_opts,
                fontsdir=os.path.abspath(FONTS_DIR))
            content_src = tmp_subs

        # Step 3: full 9:16 frame
        if blur:
            blur_background(content_src, out_mp4, enabled=True,
                            banner_top=banner_top, banner_bottom=banner_bottom,
                            gpu_opts=gpu_opts)
        else:
            pad_with_banners(content_src, out_mp4,
                             banner_top=banner_top, banner_bottom=banner_bottom,
                             gpu_opts=gpu_opts)
    finally:
        tmp_vert.unlink(missing_ok=True)
        tmp_subs.unlink(missing_ok=True)
    return str(out_mp4)



