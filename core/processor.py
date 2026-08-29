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
    return Path(config.CACHE_DIR) / f"_person_cache_v2_{h}.json"


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


def _resolve_cut_flags(corr_values: list[float], cut_strong: float = 0.35,
                       cut_threshold: float = 0.5) -> list[bool]:
    """Resolve per-sample cut flags from sequential correlation values
    with hysteresis: strong cuts fire instantly; candidates need one-sample
    confirmation against the PRE-candidate frame. Unconfirmed candidates
    roll prev_hist back to the pre-candidate histogram."""
    flags = [False] * len(corr_values)
    i = 0
    while i < len(corr_values):
        c = corr_values[i]
        if c < cut_strong:
            flags[i] = True  # strong cut: confirmed instantly
            i += 1
        elif c < cut_threshold:
            # Candidate: confirmed only if the NEXT correlation (computed
            # against the same pre-candidate reference) also drops below the
            # threshold; the cut then lands on the candidate sample.
            if i + 1 < len(corr_values) and corr_values[i + 1] < cut_threshold:
                flags[i] = True
            i += 2  # confirmation sample is consumed either way
        else:
            i += 1
    return flags


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
    # Histogram correlation bands (v1-7 #3b hysteresis): below CUT_STRONG a
    # cut fires instantly; between CUT_STRONG and CUT_HIST_THRESHOLD it is
    # only a candidate needing one-sample confirmation.
    CUT_STRONG = 0.35
    CUT_HIST_THRESHOLD = 0.5

    # Diagnostics (task v1-7 #3a): MOVIESHORT_DEBUG_CUTS=1 logs per-sample
    # corr/cut/offset to .omo/evidence/cuts_debug.log — zero effect otherwise.
    _debug_cuts = os.environ.get("MOVIESHORT_DEBUG_CUTS") == "1"
    _debug_records: list[tuple] = []

    # Hold protocol (#3b): ref_hist is the last CONFIRMED frame's histogram.
    # While a candidate is unresolved the reference stays at the pre-candidate
    # histogram; final cut flags are resolved post-scan by _resolve_cut_flags.
    ref_hist = None
    corr_list: list[float] = []
    _pending_candidate = False

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

        # Shot-cut detection: grayscale histogram correlation of each sample
        # against the last CONFIRMED reference frame. Debounce (hysteresis)
        # is applied post-scan by _resolve_cut_flags; here we only stream
        # corr values and hold the reference while a candidate is unresolved.
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if ref_hist is None:
            _corr_s = "first"
            ref_hist = hist
        else:
            corr = float(cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL))
            corr_list.append(corr)
            _corr_s = f"{corr:.4f}"
            if _pending_candidate:
                # Candidate resolves NOW: this corr was computed against the
                # same pre-candidate reference. Below threshold → cut
                # confirmed (lands on the candidate sample) and the reference
                # commits to the confirming frame; otherwise unconfirmed →
                # reference stays at the pre-candidate histogram.
                if corr < CUT_HIST_THRESHOLD:
                    ref_hist = hist
                _pending_candidate = False
            elif corr < CUT_STRONG:
                ref_hist = hist  # strong cut: confirmed instantly
            elif corr >= CUT_HIST_THRESHOLD:
                ref_hist = hist  # clear match: advance reference
            else:
                _pending_candidate = True  # candidate band: hold reference

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
                "cut": False,  # placeholder — resolved post-scan (#3b)
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
                "cut": False,  # placeholder — resolved post-scan (#3b)
        })

        if _debug_cuts:
            fh, fw = frame.shape[:2]
            if detections:
                d = max(detections, key=lambda dd: dd["w"] * dd["h"])
                off_x = int(d["x"] + d["w"] / 2 - fw / 2)
                off_y = int(d["y"] + d["h"] / 2 - fh / 2)
            else:
                off_x = off_y = None
            _debug_records.append((_corr_s, off_x, off_y))

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

    # Resolve cut flags with hysteresis (v1-7 #3b anti-jitter debounce):
    # strong cuts instant, candidates need one-sample confirmation against
    # the pre-candidate frame. First sample of the scan is always a cut.
    flags = _resolve_cut_flags(corr_list, cut_threshold=CUT_HIST_THRESHOLD)
    cut_flags = [True] + flags
    for rec, cut in zip(person_data, cut_flags):
        rec["cut"] = cut

    if _debug_cuts and _debug_records:
        try:
            log_path = Path(".omo/evidence/cuts_debug.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"{i}\tcorr={c}\tcut={int(cut)}\toff=({ox},{oy})"
                     for i, ((c, ox, oy), cut)
                     in enumerate(zip(_debug_records, cut_flags))]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"=== {video_path} | {time_module.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"| samples={len(lines)} ===\n")
                f.write("\n".join(lines) + "\n")
            print(f"  [debug-cuts] wrote {len(lines)} samples -> {log_path}")
        except OSError as e:
            print(f"  [debug-cuts] write failed: {e}")
    _save_person_cache(video_path, person_data)
    return person_data


# Backward compat alias
analyze_faces = analyze_persons


# ---------------------------------------------------------------------------
# Crop-path computation
# ---------------------------------------------------------------------------

# Adjacent shots whose filled centers are closer than this fraction of the
# video width are merged into one shot (v1-7 #3b anti-jitter).
MERGE_DIST_PX_RATIO = 0.06


def compute_person_crop_path(
    person_data: list[dict],
    video_width: int,
    video_height: int,
) -> list[dict]:
    """Compute per-shot 9:16 crop rectangles for each analyzed frame.

    Records are grouped into shots at every ``cut=True`` flag; each shot's
    crop stays fixed on the median body-center of that shot's largest-area
    detections. Input without ``cut`` keys (legacy caches/fixtures) forms a
    single shot. Returns list of
    ``{frame_idx, shot_id, crop_x, crop_y, crop_w, crop_h}``.
    """
    target_ratio = 9 / 16
    target_w = video_width
    target_h = int(target_w / target_ratio)
    if target_h > video_height:
        target_h = video_height
        target_w = int(target_h * target_ratio)

    def _body_center(fd: dict) -> tuple[float, float]:
        """Body center of the largest detection (same heuristic as analyze_persons)."""
        if fd.get("center_x") is not None:
            return (fd["center_x"], fd["center_y"])
        det_type = fd.get("type") or (
            "face" if fd.get("num_faces", 0) > 0 else "person"
        )
        cy = fd["y"] + fd["h"] * 2 if det_type == "face" else fd["y"] + fd["h"] / 2
        return (fd["x"] + fd["w"] / 2, cy)

    # Split records into shots: a new group starts at every cut=True record.
    if any("cut" in fd for fd in person_data):
        shots: list[list[dict]] = []
        for fd in person_data:
            if fd.get("cut") or not shots:
                shots.append([fd])
            else:
                shots[-1].append(fd)
    else:
        shots = [list(person_data)]  # legacy input: whole clip is one shot

    # Per-shot center: median over that shot's detected frames.
    centers: list[tuple[float, float] | None] = []
    for shot in shots:
        pts = [_body_center(fd) for fd in shot if fd["x"] is not None]
        centers.append(
            (float(np.median([p[0] for p in pts])),
             float(np.median([p[1] for p in pts])))
            if pts else None
        )

    # Empty shots inherit previous shot's center, else next, else frame center.
    filled: list[tuple[float, float]] = []
    for i, c in enumerate(centers):
        if c is None:
            prev_c = next((centers[j] for j in range(i - 1, -1, -1)
                           if centers[j] is not None), None)
            next_c = next((centers[j] for j in range(i + 1, len(centers))
                           if centers[j] is not None), None)
            c = prev_c or next_c or (video_width / 2, video_height / 2)
        filled.append(c)

    # Merge adjacent shots whose filled centers are closer than MERGE_DIST_PX
    # (#3b): detection flicker splits one continuous shot into micro-shots,
    # each pulling the camera sideways. Merged shots recompute the median
    # over the union of their detected frames.
    merge_dist = int(MERGE_DIST_PX_RATIO * video_width)
    groups: list[list[dict]] = []
    group_centers: list[tuple[float, float]] = []
    for shot, c in zip(shots, filled):
        if group_centers:
            px, py = group_centers[-1]
            if ((px - c[0]) ** 2 + (py - c[1]) ** 2) ** 0.5 < merge_dist:
                groups[-1].extend(shot)
                pts = [_body_center(fd) for fd in groups[-1]
                       if fd["x"] is not None]
                if pts:
                    group_centers[-1] = (
                        float(np.median([p[0] for p in pts])),
                        float(np.median([p[1] for p in pts])))
                continue
        groups.append(list(shot))
        group_centers.append(c)

    result = []
    for shot_id, shot in enumerate(groups):
        cx, cy = group_centers[shot_id]
        crop_x = int(cx - target_w / 2)
        crop_y = int(cy - target_h / 2)
        crop_x = max(0, min(crop_x, video_width - target_w))
        crop_y = max(0, min(crop_y, video_height - target_h))
        for fd in shot:
            result.append({
                "frame_idx": fd["frame_idx"],
                "shot_id": shot_id,
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


# Max piecewise segments in one ffmpeg crop expression before forced merging
MAX_CROP_SEGMENTS = 60


def _crop_segments_from_path(crop_path, fps, video_size, target_size):
    """Per-shot crop offsets [(t_start_sec, offset_x, offset_y)].

    One segment per shot_id change in crop_path; offsets scaled/clamped to
    the FFmpeg scale filter; adjacent duplicates merged; count capped at
    MAX_CROP_SEGMENTS by collapsing shortest-duration adjacent pairs.
    """
    video_w, video_h = video_size
    target_w, content_h = target_size
    if video_w > 0 and video_h > 0:
        sf = max(target_w / video_w, content_h / video_h)
    else:
        sf = 1.0
    max_x = int(video_w * sf - target_w)
    max_y = int(video_h * sf - content_h)

    segments = []
    prev_shot = object()
    for entry in crop_path:
        shot = entry.get("shot_id")
        if shot == prev_shot:
            continue
        ox = max(0, min(int(entry["crop_x"] * sf), max_x))
        oy = max(0, min(int(entry["crop_y"] * sf), max_y))
        segments.append((entry["frame_idx"] / fps, ox, oy))
        prev_shot = shot

    # Merge adjacent segments with identical offsets
    merged = []
    for seg in segments:
        if merged and merged[-1][1] == seg[1] and merged[-1][2] == seg[2]:
            continue
        merged.append(seg)
    # ponytail: cap keeps the earlier pair member's offsets; fidelity is fine at <=60
    while len(merged) > MAX_CROP_SEGMENTS:
        shortest = min(range(len(merged) - 1),
                       key=lambda i: merged[i + 1][0] - merged[i][0])
        del merged[shortest + 1]
    return merged


def _build_crop_expression(segments):
    """Filter-ready piecewise-constant ffmpeg x(t)/y(t) expressions.

    x(t) = if(lt(t,t1),x0,if(lt(t,t2),x1,...,xn)); a single segment yields
    plain constants without lt(). Commas are backslash-escaped so the
    filtergraph parser does not split the expression into fake filters.
    """
    if len(segments) == 1:
        return str(segments[0][1]), str(segments[0][2])
    x_expr = str(segments[-1][1])
    y_expr = str(segments[-1][2])
    for t_start, x, y in reversed(segments[:-1]):
        x_expr = f"if(lt(t,{t_start}),{x},{x_expr})"
        y_expr = f"if(lt(t,{t_start}),{y},{y_expr})"
    # Escape commas for the ffmpeg filtergraph parser (", " splits filters)
    return x_expr.replace(",", "\\,"), y_expr.replace(",", "\\,")


def _face_tracking_crop(video_path, output_path, crop_path, fps,
                         anti_copyright=True, ac_part="", target_w=1080,
                         content_h=1320, progress_callback=None,
                         clip_duration=0):
    """Apply face-tracking crop as piecewise-constant per-shot offsets.

    Builds one segment per shot_id change in crop_path, scales each offset
    to match the FFmpeg scale filter, merges neighbors with identical
    offsets, and renders everything in ONE FFmpeg pass via time expressions
    x(t)/y(t) — the crop jumps instantly at shot cuts (no panning, no
    segment splitting, no concat).

    The filter chain:
      scale=W:H:force_original_aspect_ratio=increase  → fills content area
      crop=W:H:x(t):y(t)                              → follows active shot
      [anti-copyright filters]
    """
    if not crop_path:
        _center_crop_ffmpeg_fixed(video_path, output_path, target_w, content_h,
                                   ac_part, progress_callback, clip_duration)
        return

    # Get original video dimensions for coordinate scaling
    video_w = _get_video_width(video_path)
    video_h = _get_video_height(video_path)

    segments = _crop_segments_from_path(crop_path, fps,
                                        (video_w, video_h),
                                        (target_w, content_h))
    x_expr, y_expr = _build_crop_expression(segments)

    filter_str = (
        f"scale={target_w}:{content_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{content_h}:{x_expr}:{y_expr}"
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
