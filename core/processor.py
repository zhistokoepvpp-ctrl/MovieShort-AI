"""Face and person tracking with vertical video cropping.

Uses OpenCV Haar cascades (frontal + profile) for face detection,
and HOG person detector for body detection.
Cascade XMLs auto-download from GitHub on first use.
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
_PROFILE_CASCADE = None  # type: cv2.CascadeClassifier | None
_HOG = None  # type: cv2.HOGDescriptor | None


def _download_cascade(filename: str, url: str) -> Path:
    """Download a cascade XML if not present locally."""
    path = _MODEL_DIR / filename
    if not path.exists():
        import urllib.request
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[face] Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
        print(f"[face] Download complete: {filename}")
    return path


def _get_cascade() -> cv2.CascadeClassifier:
    """Return (and cache) the frontal face cascade classifier."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = _download_cascade(
            "haarcascade_frontalface_default.xml",
            "https://raw.githubusercontent.com/opencv/opencv/master/"
            "data/haarcascades/haarcascade_frontalface_default.xml",
        )
        _FACE_CASCADE = cv2.CascadeClassifier(str(path))
        if _FACE_CASCADE.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {path}")
    return _FACE_CASCADE


def _get_profile_cascade() -> cv2.CascadeClassifier:
    """Return (and cache) the profile face cascade classifier."""
    global _PROFILE_CASCADE
    if _PROFILE_CASCADE is None:
        path = _download_cascade(
            "haarcascade_profileface.xml",
            "https://raw.githubusercontent.com/opencv/opencv/master/"
            "data/haarcascades/haarcascade_profileface.xml",
        )
        _PROFILE_CASCADE = cv2.CascadeClassifier(str(path))
        if _PROFILE_CASCADE.empty():
            raise RuntimeError(f"Failed to load profile cascade: {path}")
    return _PROFILE_CASCADE


def _get_hog() -> cv2.HOGDescriptor:
    """Return (and cache) the HOG person detector."""
    global _HOG
    if _HOG is None:
        _HOG = cv2.HOGDescriptor()
        _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _HOG


# ---------------------------------------------------------------------------
# Single-frame helper  (used by the UI preview)
# ---------------------------------------------------------------------------


def analyze_persons_single_frame(image: np.ndarray) -> list[dict]:
    """Detect faces (frontal+profile) and persons in a single cv2 image (BGR).

    Returns list of ``{x, y, w, h, confidence, type}`` dicts in **pixel**
    coordinates.  ``type`` is "face" or "person".
    """
    detections = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Frontal face
    frontal_rects = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40),
    )
    for x, y, w, h in frontal_rects:
        detections.append({"x": x, "y": y, "w": w, "h": h,
                           "confidence": 1.0, "type": "face"})

    # Profile face (only if no frontal found — avoid double-counting)
    if not detections:
        # Left-facing profiles (direct)
        profile_rects = _get_profile_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40),
        )
        for x, y, w, h in profile_rects:
            detections.append({"x": x, "y": y, "w": w, "h": h,
                               "confidence": 1.0, "type": "face"})

        # Right-facing profiles (via horizontal flip)
        flipped = cv2.flip(gray, 1)
        profile_right = _get_profile_cascade().detectMultiScale(
            flipped, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40),
        )
        for x, y, w, h in profile_right:
            orig_x = gray.shape[1] - x - w
            detections.append({"x": orig_x, "y": y, "w": w, "h": h,
                               "confidence": 1.0, "type": "face"})

    # HOG person detector (only if no faces found — trigger on face-less runs)
    if not detections:
        hog = _get_hog()
        h, w = image.shape[:2]
        # HOG works best at 640px width
        scale = 640 / w if w > 640 else 1.0
        if scale < 1.0:
            small = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = image
        rects, weights = hog.detectMultiScale(small, winStride=(8, 8),
                                               padding=(4, 4), scale=1.05)
        for (x, y, w, h), weight in zip(rects, weights):
            # Scale back to original coordinates
            ox, oy, ow, oh = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
            detections.append({"x": ox, "y": oy, "w": ow, "h": oh,
                               "confidence": float(weight), "type": "person"})

    return detections


# Backward compat alias
analyze_faces_single_frame = analyze_persons_single_frame


# ---------------------------------------------------------------------------
# Face cache helpers
# ---------------------------------------------------------------------------

import hashlib


def _get_person_cache_path(video_path: str) -> Path:
    """Return path to person cache JSON for the given video."""
    h = hashlib.md5(video_path.encode()).hexdigest()[:8]
    return Path(config.TEMP_DIR) / f"_person_cache_{h}.json"


def _load_person_cache(video_path: str):
    """Load cached person data if it exists and is newer than video."""
    cache_path = _get_person_cache_path(video_path)
    if not cache_path.exists():
        return None
    try:
        cache_mtime = cache_path.stat().st_mtime
        video_mtime = Path(video_path).stat().st_mtime
        if video_mtime > cache_mtime:
            print("  Person cache stale (video updated) — re-scanning")
            return None
        with open(cache_path, "r") as f:
            data = json.load(f)
        print(f"  Using cached person data ({len(data)} frames)")
        return data
    except Exception:
        return None


def _save_person_cache(video_path: str, person_data: list[dict]) -> None:
    """Save person data to cache JSON."""
    cache_path = _get_person_cache_path(video_path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(person_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# Backward compat aliases
_get_face_cache_path = _get_person_cache_path
_load_face_cache = _load_person_cache
_save_face_cache = _save_person_cache


# ---------------------------------------------------------------------------
# Video-level face analysis
# ---------------------------------------------------------------------------


def analyze_persons(video_path: str, progress_callback=None) -> list[dict]:
    """Scan video for faces (frontal+profile) and persons every ~1 second.

    Reads frames SEQUENTIALLY (no frame-index seeking — unreliable on some
    codecs/containers).  Samples 1 frame per second for detection.

    Uses frontal face cascade, profile face cascade (if no frontal found),
    and HOG person detector (triggered after 5 consecutive face-less frames,
    then runs continuously until face reappears).

    Returns list of ``{frame_idx, x, y, w, h, num_faces, num_persons, type}`` dicts.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    # Sample 1 frame per second (sequential reading — no frame seeking)
    interval = max(1, int(fps))

    if total <= 0:
        print(f"  ⚠ OpenCV cannot decode video (total_frames={total}) — "
              "skipping person tracking")
        cap.release()
        return []

    analyzed_count = max(1, total // interval) if interval > 0 else 1
    cascade = _get_cascade()
    profile_cascade = _get_profile_cascade()
    hog = _get_hog()

    person_data: list[dict] = []
    last_print = 0
    _start = time_module.time()

    consecutive_no_face = 0
    HOG_TRIGGER_FRAMES = 0  # trigger HOG immediately on any face-less frame
    hog_active = False

    # Downscale target: max 360px on the longest side
    _MAX_DETECT_PX = 360

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every `interval`-th frame (= once per second)
        if frame_idx % interval != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        # Downscale for detection speed
        scale = min(_MAX_DETECT_PX / max(h, w), 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        detections = []
        num_faces = 0
        num_persons = 0

        # 1. Frontal face
        frontal_rects = cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30),
        )
        if len(frontal_rects) > 0:
            best = max(frontal_rects, key=lambda r: r[2] * r[3])
            x, y, bw, bh = (int(v / scale) for v in best)
            detections.append({"x": float(x), "y": float(y),
                               "w": float(bw), "h": float(bh),
                               "confidence": 1.0, "type": "face"})
            num_faces = len(frontal_rects)
            consecutive_no_face = 0
            hog_active = False
        else:
            # 2. Profile face (only if no frontal) — detect BOTH sides
            # Left-facing profiles (direct)
            profile_rects = profile_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30),
            )
            profile_results = list(profile_rects)

            # Right-facing profiles (via horizontal flip)
            flipped = cv2.flip(gray, 1)
            profile_right = profile_cascade.detectMultiScale(
                flipped, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30),
            )
            for x, y, w, h in profile_right:
                # Convert flipped coords back to original: x_orig = width - x - w
                orig_x = gray.shape[1] - x - w
                profile_results.append((orig_x, y, w, h))

            if len(profile_results) > 0:
                best = max(profile_results, key=lambda r: r[2] * r[3])
                x, y, bw, bh = (int(v / scale) for v in best)
                detections.append({"x": float(x), "y": float(y),
                                   "w": float(bw), "h": float(bh),
                                   "confidence": 1.0, "type": "face"})
                num_faces = len(profile_results)
                consecutive_no_face = 0
                hog_active = False
            else:
                consecutive_no_face += 1

            # 3. HOG person detector (trigger after run-length threshold,
            #    then stays active until face reappears)
            if consecutive_no_face >= HOG_TRIGGER_FRAMES or hog_active:
                hog_active = True
                rects, weights = hog.detectMultiScale(
                    small, winStride=(4, 4), padding=(4, 4), scale=1.03)
                if len(rects) > 0:
                    best_idx = np.argmax(weights)
                    x, y, bw, bh = rects[best_idx]
                    ox = int(x / scale)
                    oy = int(y / scale)
                    ow = int(bw / scale)
                    oh = int(bh / scale)
                    detections.append({"x": float(ox), "y": float(oy),
                                       "w": float(ow), "h": float(oh),
                                       "confidence": float(weights[best_idx]),
                                       "type": "person"})
                    num_persons = len(rects)

        if detections:
            best_det = max(detections, key=lambda d: d["w"] * d["h"])

            # Compute body-centered crop position
            if best_det["type"] == "face":
                # Face center horizontally, body center estimate below face
                cx = best_det["x"] + best_det["w"] / 2
                cy = best_det["y"] + best_det["h"] * 2  # heuristic: body below face
            else:
                # Person center (HOG returns full body)
                cx = best_det["x"] + best_det["w"] / 2
                cy = best_det["y"] + best_det["h"] / 2

            person_data.append({
                "frame_idx": frame_idx,
                "x": best_det["x"],
                "y": best_det["y"],
                "w": best_det["w"],
                "h": best_det["h"],
                "center_x": cx,
                "center_y": cy,
                "num_faces": num_faces,
                "num_persons": num_persons,
                "type": best_det["type"],
            })
        else:
            person_data.append({
                "frame_idx": frame_idx,
                "x": None, "y": None,
                "w": None, "h": None,
                "center_x": None, "center_y": None,
                "num_faces": 0,
                "num_persons": 0,
                "type": None,
            })

        # User-visible progress
        done = len(person_data)
        pct = done / max(analyzed_count, 1)
        elapsed = time_module.time() - _start
        if done % 10 == 0 or pct - last_print >= 0.05:
            eta = (elapsed / max(pct, 0.01) - elapsed) if pct > 0 else 0
            print(f"  Person scan: {pct:.0%}  ({done}/{analyzed_count} frames, "
                  f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            last_print = pct

        if progress_callback:
            progress_callback(frame_idx / max(total, 1))

        frame_idx += 1

    cap.release()
    _elapsed = time_module.time() - _start
    found = sum(1 for f in person_data if f["x"] is not None)
    print(f"  Person scan complete in {_elapsed:.0f}s, "
          f"found {found}/{len(person_data)} frames with faces/persons")
    _save_person_cache(video_path, person_data)
    return person_data


# Backward compat alias
analyze_faces = analyze_persons


# ---------------------------------------------------------------------------
# Crop-path computation
# ---------------------------------------------------------------------------


def compute_person_crop_path(
    person_data: list[dict],
    video_width: int,
    video_height: int,
) -> list[dict]:
    """Compute smoothed 9:16 crop rectangles for each analyzed frame.

    Uses person-centered variance metric (max(x_std, y_std)) for spread,
    moving-average window of 3 for smoothing.
    Returns list of ``{frame_idx, crop_x, crop_y, crop_w, crop_h}``.
    """
    target_ratio = 9 / 16
    target_w = video_width
    target_h = int(target_w / target_ratio)
    if target_h > video_height:
        target_h = video_height
        target_w = int(target_h * target_ratio)

    frames_with_persons = [
        (i, fd) for i, fd in enumerate(person_data) if fd["x"] is not None
    ]

    if not frames_with_persons:
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
            for fd in person_data
        ]

    # Raw centers per frame index — use stored body-centered coordinates
    raw_centers: dict[int, tuple[float, float]] = {}
    for idx, fd in frames_with_persons:
        raw_centers[idx] = (fd["center_x"], fd["center_y"])

    all_indices = [fd["frame_idx"] for fd in person_data]
    smoothed: dict[int, tuple[float, float]] = {}
    window = 3

    for fi in all_indices:
        if fi in raw_centers:
            # Smooth with neighbouring detections (window=3)
            neighbors = [
                raw_centers[j] for j in raw_centers if abs(j - fi) <= window
            ]
            avg_x = float(np.mean([n[0] for n in neighbors]))
            avg_y = float(np.mean([n[1] for n in neighbors]))
        else:
            before = [(j, raw_centers[j]) for j in raw_centers if j < fi]
            after = [(j, raw_centers[j]) for j in raw_centers if j > fi]
            if before:
                # Stay at last-known position — don't pull toward future
                avg_x, avg_y = max(before, key=lambda t: t[0])[1]
            elif after:
                # No past detection yet — use first future position
                avg_x, avg_y = min(after, key=lambda t: t[0])[1]
            else:
                # No detection at all — center frame (shouldn't happen)
                avg_x = video_width / 2
                avg_y = video_height / 2
        smoothed[fi] = (avg_x, avg_y)

    result = []
    for fd in person_data:
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


# Backward compat alias
compute_crop_path = compute_person_crop_path


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

    Computes the median crop position (smoothed by compute_person_crop_path),
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

    # Check variance — if high, use segment-based dynamic cropping
    crop_x_list = [entry["crop_x"] for entry in crop_path]
    crop_y_list = [entry["crop_y"] for entry in crop_path]
    x_std = float(np.std(crop_x_list)) if len(crop_x_list) > 1 else 0
    y_std = float(np.std(crop_y_list)) if len(crop_y_list) > 1 else 0
    max_std = max(x_std, y_std)

    # Dynamic crop threshold: 50px std deviation triggers segment-based approach
    DYNAMIC_CROP_THRESHOLD = 50.0

    if max_std > DYNAMIC_CROP_THRESHOLD:
        print(f"  High movement variance (std={max_std:.0f}px) — using segment-based dynamic crop")
        _segment_based_crop(video_path, output_path, crop_path, fps,
                           anti_copyright=anti_copyright, ac_part=ac_part,
                           target_w=target_w, content_h=content_h,
                           progress_callback=progress_callback,
                           clip_duration=clip_duration)
        return

    # Median crop position in original video coordinates
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


def _segment_based_crop(video_path, output_path, crop_path, fps,
                        anti_copyright=True, ac_part="", target_w=1080,
                        content_h=1320, progress_callback=None,
                        clip_duration=0):
    """Apply segment-based dynamic cropping for high-variance movement.

    Splits the clip into ~10s segments, each with its own crop position.
    Uses concat filter (not demuxer) to avoid AAC encoder delay issues.
    Input seeking (-ss before -i) for segment extraction.
    """
    import re as _re
    import tempfile

    video_w = _get_video_width(video_path)
    video_h = _get_video_height(video_path)

    # Segment duration in seconds
    SEGMENT_DURATION = 10.0
    total_frames = len(crop_path)
    frames_per_segment = int(SEGMENT_DURATION * fps)

    # Build segments
    segments = []
    seg_start = 0
    while seg_start < total_frames:
        seg_end = min(seg_start + frames_per_segment, total_frames)
        seg_frames = crop_path[seg_start:seg_end]

        # Median crop for this segment
        crop_x = int(np.median([e["crop_x"] for e in seg_frames]))
        crop_y = int(np.median([e["crop_y"] for e in seg_frames]))

        # Time bounds for this segment
        start_time_sec = seg_frames[0]["frame_idx"] / fps
        end_time_sec = seg_frames[-1]["frame_idx"] / fps + (1.0 / fps)

        segments.append({
            "start_sec": start_time_sec,
            "end_sec": end_time_sec,
            "crop_x": crop_x,
            "crop_y": crop_y,
        })
        seg_start = seg_end

    if not segments:
        _center_crop_ffmpeg_fixed(video_path, output_path, target_w, content_h,
                                   ac_part, progress_callback, clip_duration)
        return

    # Scale factor
    if video_w > 0 and video_h > 0:
        sf = max(target_w / video_w, content_h / video_h)
    else:
        sf = 1.0

    # Generate segment files
    tmp_dir = Path(tempfile.mkdtemp(prefix="ms_crop_"))
    segment_files = []
    concat_list_path = tmp_dir / "concat.txt"

    try:
        for i, seg in enumerate(segments):
            seg_file = tmp_dir / f"seg_{i:03d}.mp4"
            segment_files.append(seg_file)

            # Scale coordinates
            offset_x = max(0, min(int(seg["crop_x"] * sf), int(video_w * sf - target_w)))
            offset_y = max(0, min(int(seg["crop_y"] * sf), int(video_h * sf - content_h)))

            filter_str = (
                f"scale={target_w}:{content_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{content_h}:{offset_x}:{offset_y}"
                f"{ac_part}"
            )

            # Input seeking for each segment
            start_h = int(seg["start_sec"] // 3600)
            start_m = int((seg["start_sec"] % 3600) // 60)
            start_s = seg["start_sec"] % 60
            end_h = int(seg["end_sec"] // 3600)
            end_m = int((seg["end_sec"] % 3600) // 60)
            end_s = seg["end_sec"] % 60

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_h:02d}:{start_m:02d}:{start_s:06.3f}",
                "-to", f"{end_h:02d}:{end_m:02d}:{end_s:06.3f}",
                "-i", video_path,
                "-vf", filter_str,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac",
                str(seg_file),
            ]
            _run(cmd, desc=f"segment_{i}")

        # Write concat list
        with open(concat_list_path, "w") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")

        # Concat using filter (not demuxer — avoids AAC encoder delay)
        _time_re = _re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
        _proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list_path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac",
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

    finally:
        # Cleanup temp files
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


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

    Runs person analysis (faces + HOG) for stats/logging — the source is
    scaled to fill the banner-padded content area (1080 × content_h),
    cropping overflow. Banner padding is added in a separate step after
    subtitle embedding.

    Returns metadata dict with ``output_size``, ``faces_found``.
    """
    import subprocess

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    cap.release()

    if progress_callback:
        progress_callback(0.05)

    # Person analysis with timeout (30s max) — try cache first
    person_data = _load_person_cache(video_path)
    if person_data is None:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(analyze_persons, video_path)
            try:
                person_data = fut.result(timeout=30)
            except _CFTimeoutError:
                print("  ⚠ Person scan timed out (30s) — using fallback")
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

    # Log person analysis results
    valid = [fd for fd in person_data if fd["x"] is not None]
    faces_found = len(valid)
    total_frames = len(person_data)
    if faces_found > 0:
        avg_cx = float(np.mean([fd["x"] + fd["w"] / 2 for fd in valid]))
        avg_cy = float(np.mean([fd["y"] + fd["h"] / 2 for fd in valid]))
        types = [fd.get("type", "face") for fd in valid]
        face_count = types.count("face")
        person_count = types.count("person")
        print(f"  Detected {faces_found}/{total_frames} frames "
              f"({face_count} face, {person_count} person, avg center: {avg_cx:.0f}, {avg_cy:.0f})")
    else:
        print(f"  No faces/persons detected in {total_frames} frames")

    if progress_callback:
        progress_callback(0.6)

    clip_duration = _get_clip_duration(video_path)
    fps = _get_video_fps(video_path)

    # Decide: person-tracking crop or center crop
    if faces_found > 0 and fps > 0:
        print(f"  Applying person-tracking crop ({faces_found} frames)...")
        video_width = _get_video_width(video_path)
        video_height = _get_video_height(video_path)
        if video_width > 0 and video_height > 0:
            crop_path = compute_person_crop_path(person_data, video_width, video_height)
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
        print(f"  Using center crop (no person data)")
        _center_crop_ffmpeg_fixed(video_path, output_path, target_w, content_h,
                                  ac_part, progress_callback, clip_duration)

    if progress_callback:
        progress_callback(1.0)

    return {
        "output_size": (target_w, content_h),
        "faces_found": faces_found,
    }
