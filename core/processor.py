"""Face tracking and vertical video cropping.

Uses OpenCV Haar cascade for face detection.
Cascade XML auto-downloads from GitHub on first use.
"""
from pathlib import Path
import time as time_module
import json
import os
import subprocess
from concurrent.futures import TimeoutError as _CFTimeoutError

import cv2
import numpy as np

import config

# ---------------------------------------------------------------------------
# Cascade cache
# ---------------------------------------------------------------------------

_MODEL_DIR = Path(__file__).parent.parent / "models"
_FACE_CASCADE = None  # type: cv2.CascadeClassifier | None


def _get_cascade() -> cv2.CascadeClassifier:
    """Return (and cache) the face cascade classifier."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = str(_MODEL_DIR / "haarcascade_frontalface_default.xml")
        if not Path(cascade_path).exists():
            import urllib.request
            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            url = (
                "https://raw.githubusercontent.com/opencv/opencv/master/"
                "data/haarcascades/haarcascade_frontalface_default.xml"
            )
            print("[face] Downloading Haar cascade...")
            urllib.request.urlretrieve(url, cascade_path)
            print("[face] Download complete.")
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
        if _FACE_CASCADE.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")
    return _FACE_CASCADE


# ---------------------------------------------------------------------------
# Single-frame helper  (used by the UI preview)
# ---------------------------------------------------------------------------


def analyze_faces_single_frame(image: np.ndarray) -> list[dict]:
    """Detect faces in a single cv2 image (BGR).

    Returns list of ``{x, y, w, h, confidence}`` dicts in **pixel**
    coordinates, or an empty list.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40),
    )
    return [
        {"x": x, "y": y, "w": w, "h": h, "confidence": 1.0}
        for (x, y, w, h) in rects
    ]


# ---------------------------------------------------------------------------
# Face cache helpers
# ---------------------------------------------------------------------------

import hashlib


def _get_face_cache_path(video_path: str) -> Path:
    """Return path to face cache JSON for the given video."""
    h = hashlib.md5(video_path.encode()).hexdigest()[:8]
    return Path(config.TEMP_DIR) / f"_face_cache_{h}.json"


def _load_face_cache(video_path: str):
    """Load cached face data if it exists and is newer than video."""
    cache_path = _get_face_cache_path(video_path)
    if not cache_path.exists():
        return None
    try:
        cache_mtime = cache_path.stat().st_mtime
        video_mtime = Path(video_path).stat().st_mtime
        if video_mtime > cache_mtime:
            print("  Face cache stale (video updated) — re-scanning")
            return None
        with open(cache_path, "r") as f:
            data = json.load(f)
        print(f"  Using cached face data ({len(data)} frames)")
        return data
    except Exception:
        return None


def _save_face_cache(video_path: str, face_data: list[dict]) -> None:
    """Save face data to cache JSON."""
    cache_path = _get_face_cache_path(video_path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(face_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Video-level face analysis
# ---------------------------------------------------------------------------


def analyze_faces(video_path: str, progress_callback=None) -> list[dict]:
    """Scan video for faces every ``FACE_TRACKING_INTERVAL`` frames.

    Downsamples frames before detection for 5-10x speedup.
    Prints frame-by-frame progress.

    Returns list of ``{frame_idx, x, y, w, h, num_faces}`` dicts.
    Entries with no faces contain ``None`` for coordinates.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    interval = config.FACE_TRACKING_INTERVAL

    if total <= 0:
        print(f"  ⚠ OpenCV не может декодировать видео (total_frames={total}) — "
              "пропускаем face tracking")
        cap.release()
        return []

    analyzed_count = total // interval

    cascade = _get_cascade()
    face_data: list[dict] = []
    last_print = 0
    _start = time_module.time()

    # Downscale target: max 360px on the longest side
    _MAX_DETECT_PX = 360

    # Jump directly to every Nth frame instead of reading all frames sequentially.
    # Sequential cap.read() decodes every single frame (~15ms each), which alone
    # takes 30s+ for a 90s clip.  Seeking by frame-index skips decode entirely.
    for frame_idx in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # Downscale for detection speed
        scale = min(_MAX_DETECT_PX / max(h, w), 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30),
        )

        if len(rects) > 0:
            # Scale bbox back to original coordinates
            best = max(rects, key=lambda r: r[2] * r[3])
            x, y, bw, bh = (int(v / scale) for v in best)
            face_data.append({
                "frame_idx": frame_idx,
                "x": float(x),
                "y": float(y),
                "w": float(bw),
                "h": float(bh),
                "num_faces": len(rects),
            })
        else:
            face_data.append({
                "frame_idx": frame_idx,
                "x": None, "y": None,
                "w": None, "h": None,
                "num_faces": 0,
            })

        # User-visible progress
        done = len(face_data)
        pct = done / max(analyzed_count, 1)
        elapsed = time_module.time() - _start
        if done % 10 == 0 or pct - last_print >= 0.05:
            eta = (elapsed / max(pct, 0.01) - elapsed) if pct > 0 else 0
            print(f"  Face scan: {pct:.0%}  ({done}/{analyzed_count} frames, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            last_print = pct

        if progress_callback:
            progress_callback(frame_idx / max(total, 1))

    cap.release()
    _elapsed = time_module.time() - _start
    print(f"  Face scan complete in {_elapsed:.0f}s, found {sum(1 for f in face_data if f['x'] is not None)}/{len(face_data)} frames with faces")
    _save_face_cache(video_path, face_data)
    return face_data


# ---------------------------------------------------------------------------
# Crop-path computation
# ---------------------------------------------------------------------------


def compute_crop_path(
    face_data: list[dict],
    video_width: int,
    video_height: int,
) -> list[dict]:
    """Compute smoothed 9:16 crop rectangles for each analyzed frame.

    Uses a moving-average window of 3 for smoothing.
    Returns list of ``{frame_idx, crop_x, crop_y, crop_w, crop_h}``.
    """
    target_ratio = 9 / 16
    target_w = video_width
    target_h = int(target_w / target_ratio)
    if target_h > video_height:
        target_h = video_height
        target_w = int(target_h * target_ratio)

    frames_with_faces = [
        (i, fd) for i, fd in enumerate(face_data) if fd["x"] is not None
    ]

    if not frames_with_faces:
        cx = video_width / 2
        cy = video_height / 2
        return [
            {
                "frame_idx": fd["frame_idx"],
                "crop_x": max(0, int(cx - target_w / 2)),
                "crop_y": max(0, int(cy - target_h / 2)),
                "crop_w": min(target_w, video_width),
                "crop_h": min(target_h, video_height),
            }
            for fd in face_data
        ]

    # Raw centers per frame index
    raw_centers: dict[int, tuple[float, float]] = {}
    for idx, fd in frames_with_faces:
        raw_centers[idx] = (fd["x"] + fd["w"] / 2, fd["y"] + fd["h"] / 2)

    all_indices = [fd["frame_idx"] for fd in face_data]
    smoothed: dict[int, tuple[float, float]] = {}
    window = 3

    for fi in all_indices:
        if fi in raw_centers:
            neighbors = [
                raw_centers[j] for j in raw_centers if abs(j - fi) <= window
            ]
            avg_x = float(np.mean([n[0] for n in neighbors]))
            avg_y = float(np.mean([n[1] for n in neighbors]))
        else:
            before = [(j, raw_centers[j]) for j in raw_centers if j < fi]
            after = [(j, raw_centers[j]) for j in raw_centers if j > fi]
            if before and after:
                b = max(before, key=lambda t: t[0])
                a = min(after, key=lambda t: t[0])
                t = (fi - b[0]) / max(a[0] - b[0], 1)
                avg_x = b[1][0] * (1 - t) + a[1][0] * t
                avg_y = b[1][1] * (1 - t) + a[1][1] * t
            elif before:
                avg_x, avg_y = max(before, key=lambda t: t[0])[1]
            else:
                avg_x, avg_y = min(after, key=lambda t: t[0])[1]
        smoothed[fi] = (avg_x, avg_y)

    result = []
    for fd in face_data:
        fi = fd["frame_idx"]
        cx, cy = smoothed.get(fi, (video_width / 2, video_height / 2))
        crop_x = int(cx - target_w / 2)
        crop_y = int(cy - target_h / 2)
        crop_x = max(0, min(crop_x, video_width - target_w))
        crop_y = max(0, min(crop_y, video_height - target_h))
        result.append({
            "frame_idx": fi,
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_w": target_w,
            "crop_h": target_h,
        })
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_video_fps(video_path: str) -> float:
    """Return video FPS using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    except Exception:
        return 0.0


def _get_video_width(video_path: str) -> int:
    """Return video width in pixels."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return w
    except Exception:
        return 0


def _get_video_height(video_path: str) -> int:
    """Return video height in pixels."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return h
    except Exception:
        return 0


def _center_crop_ffmpeg_fixed(video_path, output_path, target_w, content_h,
                               ac_part, progress_callback, clip_duration):
    """Run ffmpeg center crop with progress reporting."""
    import re as _re
    _time_re = _re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
    _proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf",
            f"scale={target_w}:{content_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{content_h}"
            f"{ac_part}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "copy",
            output_path,
        ],
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert _proc.stderr is not None
    for line in _proc.stderr:
        m = _time_re.search(line)
        if m and clip_duration > 0:
            h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            elapsed = h * 3600 + mnt * 60 + s
            pct = min(0.99, 0.6 + 0.39 * (elapsed / clip_duration))
            if progress_callback:
                progress_callback(pct)
    _proc.wait()
    if _proc.returncode != 0:
        raise subprocess.CalledProcessError(_proc.returncode, _proc.args)


def _face_tracking_crop(video_path, output_path, crop_path, fps,
                         anti_copyright=True, ac_part="", target_w=1080,
                         content_h=1320, progress_callback=None,
                         clip_duration=0):
    """Apply face-tracking crop using a single stable crop rectangle.

    Computes the median crop position (smoothed by compute_crop_path),
    scales coordinates to match the FFmpeg scale filter, and runs one
    FFmpeg command (no segment splitting — avoids ultra-short segment
    crashes and concat complexity).

    The filter chain:
      scale=W:H:force_original_aspect_ratio=increase  → fills content area
      crop=W:H:offset_x:0                             → shifts to follow face
      [anti-copyright filters]
    """
    if not crop_path:
        _center_crop_ffmpeg_fixed(video_path, output_path, target_w, content_h,
                                   ac_part, progress_callback, clip_duration)
        return

    # Get original video dimensions for coordinate scaling
    video_w = _get_video_width(video_path)
    video_h = _get_video_height(video_path)

    # Median crop position in original video coordinates
    crop_x_list = [entry["crop_x"] for entry in crop_path]
    crop_y_list = [entry["crop_y"] for entry in crop_path]
    crop_x_orig = int(np.median(crop_x_list))
    crop_y_orig = int(np.median(crop_y_list))

    # Scale factor for: scale=target_w:content_h:force_original_aspect_ratio=increase
    if video_w > 0 and video_h > 0:
        sf = max(target_w / video_w, content_h / video_h)
    else:
        sf = 1.0

    scaled_w = video_w * sf
    scaled_h = video_h * sf

    # Crop window start position in scaled coordinates
    offset_x = crop_x_orig * sf
    offset_y = crop_y_orig * sf

    # Clamp to valid range so the crop window stays within the scaled frame
    offset_x = max(0, min(int(offset_x), int(scaled_w - target_w)))
    offset_y = max(0, min(int(offset_y), int(scaled_h - content_h)))

    filter_str = (
        f"scale={target_w}:{content_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{content_h}:{offset_x}:{offset_y}"
        f"{ac_part}"
    )

    # Single ffmpeg pass with progress reporting
    import re as _re
    _time_re = _re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
    _proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", filter_str,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "copy",
            output_path,
        ],
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert _proc.stderr is not None
    for line in _proc.stderr:
        m = _time_re.search(line)
        if m and clip_duration > 0:
            h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            elapsed = h * 3600 + mnt * 60 + s
            pct = min(0.99, 0.6 + 0.39 * (elapsed / clip_duration))
            if progress_callback:
                progress_callback(pct)
    _proc.wait()
    if _proc.returncode != 0:
        raise subprocess.CalledProcessError(_proc.returncode, _proc.args)


def _get_clip_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe (cheap)."""
    import subprocess
    import json
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True, timeout=15,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Full vertical-crop pipeline
# ---------------------------------------------------------------------------


def _center_crop_ffmpeg(video_path, output_path, progress_callback=None,
                        anti_copyright=True, banner_top=None, banner_bottom=None):
    """Scale-to-fill content area (no banner padding, no face tracking)."""
    import subprocess
    bt = banner_top if banner_top is not None else config.BANNER_TOP
    bb = banner_bottom if banner_bottom is not None else config.BANNER_BOTTOM
    content_h = config.VERTICAL_HEIGHT - bt - bb
    ac_filters = []
    if anti_copyright:
        if config.AC_MIRROR:
            ac_filters.append("hflip")
        if config.AC_CONTRAST != 1.0 or config.AC_BRIGHTNESS != 0.0 or config.AC_SATURATION != 1.0:
            ac_filters.append(
                f"eq=contrast={config.AC_CONTRAST}:"
                f"brightness={config.AC_BRIGHTNESS}:"
                f"saturation={config.AC_SATURATION}"
            )
    ac_part = "," + ",".join(ac_filters) if ac_filters else ""
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf",
         f"scale={config.VERTICAL_WIDTH}:{content_h}:force_original_aspect_ratio=increase,"
         f"crop={config.VERTICAL_WIDTH}:{content_h}"
         f"{ac_part}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "copy", output_path],
        check=True, capture_output=True, timeout=120,
    )
    if progress_callback:
        progress_callback(1.0)
    return {"output_size": (config.VERTICAL_WIDTH, content_h),
            "faces_found": 0}


def apply_vertical_crop(
    video_path: str,
    output_path: str,
    progress_callback=None,
    anti_copyright: bool = True,
    banner_top: int = None,
    banner_bottom: int = None,
) -> dict:
    """Scale-to-fill the content area (no banner padding, center crop).

    Runs face analysis for stats/logging only — the source is scaled to fill
    the banner-padded content area (1080 × content_h), cropping overflow.
    Banner padding is added in a separate step after subtitle embedding.

    Returns metadata dict with ``output_size``, ``faces_found``.
    """
    import subprocess

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    cap.release()

    if progress_callback:
        progress_callback(0.05)

    # Face analysis with timeout (30s max) — try cache first
    face_data = _load_face_cache(video_path)
    if face_data is None:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(analyze_faces, video_path)
            try:
                face_data = fut.result(timeout=30)
            except _CFTimeoutError:
                print("  ⚠ Face scan timed out (30s) — using fallback")
                return _center_crop_ffmpeg(video_path, output_path, progress_callback,
                                           anti_copyright=anti_copyright,
                                           banner_top=banner_top, banner_bottom=banner_bottom)

    if progress_callback:
        progress_callback(0.5)

    bt = banner_top if banner_top is not None else config.BANNER_TOP
    bb = banner_bottom if banner_bottom is not None else config.BANNER_BOTTOM
    target_w = config.VERTICAL_WIDTH
    content_h = config.VERTICAL_HEIGHT - bt - bb

    ac_filters = []
    if anti_copyright:
        if config.AC_MIRROR:
            ac_filters.append("hflip")
        if config.AC_CONTRAST != 1.0 or config.AC_BRIGHTNESS != 0.0 or config.AC_SATURATION != 1.0:
            ac_filters.append(
                f"eq=contrast={config.AC_CONTRAST}:"
                f"brightness={config.AC_BRIGHTNESS}:"
                f"saturation={config.AC_SATURATION}"
            )
    ac_part = "," + ",".join(ac_filters) if ac_filters else ""

    # Log face analysis results
    valid = [fd for fd in face_data if fd["x"] is not None]
    faces_found = len(valid)
    total_frames = len(face_data)
    if faces_found > 0:
        avg_cx = float(np.mean([fd["x"] + fd["w"] / 2 for fd in valid]))
        avg_cy = float(np.mean([fd["y"] + fd["h"] / 2 for fd in valid]))
        print(f"  Faces found in {faces_found}/{total_frames} frames "
              f"(avg center: {avg_cx:.0f}, {avg_cy:.0f})")
    else:
        print(f"  No faces detected in {total_frames} frames")

    if progress_callback:
        progress_callback(0.6)

    clip_duration = _get_clip_duration(video_path)
    fps = _get_video_fps(video_path)

    # Decide: face-tracking crop or center crop
    if faces_found > 0 and fps > 0:
        print(f"  Applying face-tracking crop ({faces_found} face frames)...")
        video_width = _get_video_width(video_path)
        video_height = _get_video_height(video_path)
        if video_width > 0 and video_height > 0:
            crop_path = compute_crop_path(face_data, video_width, video_height)
            _face_tracking_crop(video_path, output_path, crop_path, fps,
                                anti_copyright=anti_copyright,
                                ac_part=ac_part, target_w=target_w,
                                content_h=content_h,
                                progress_callback=progress_callback,
                                clip_duration=clip_duration)
        else:
            _center_crop_ffmpeg_fixed(video_path, output_path, target_w, content_h,
                                      ac_part, progress_callback, clip_duration)
    else:
        print(f"  Using center crop (no face data)")
        _center_crop_ffmpeg_fixed(video_path, output_path, target_w, content_h,
                                  ac_part, progress_callback, clip_duration)

    if progress_callback:
        progress_callback(1.0)

    return {
        "output_size": (target_w, content_h),
        "faces_found": faces_found,
    }
