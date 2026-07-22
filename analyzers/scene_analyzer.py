"""
MovieShort AI — Scene detection and transcription analyzer.
"""
import hashlib
import json as _json
import re
import time as time_module
import subprocess
import os

from scenedetect import open_video, SceneManager, ContentDetector

from core.subtitle import transcribe, _transcribe_audio_file
import config


def _cleanup_temp_audio(path: str) -> None:
    """Safely remove temp audio file if it exists."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _compute_audio_rms(video_path: str, audio_path: str) -> list:
    """Compute RMS envelope (dBFS) from WAV audio, one value per 200ms window.

    Reads the 16-bit PCM WAV directly via Python (no ffmpeg astats dependency),
    properly parsing RIFF/WAV chunk structure to find the data chunk.
    Returns list of dBFS values or [] on error.
    """
    import struct
    import math
    window_s = 0.2   # 200ms per window

    try:
        with open(audio_path, "rb") as f:
            buf = f.read()
    except (IOError, OSError):
        return []

    if len(buf) < 44:
        return []

    # Parse RIFF/WAV chunk structure to locate the "data" chunk
    # WAV format: RIFF header (12 bytes) → chunks (type + size + data)
    # fmt chunk is usually first, but we skip through all chunks
    # to find the data chunk (can have LIST, fact, etc. in between)
    pos = 12  # skip RIFF header (4 RIFF + 4 size + 4 WAVE)
    sample_rate = 16000  # default fallback
    data_start = -1
    data_end = len(buf)

    while pos < min(len(buf), 256):  # scan first 256 bytes for headers
        if pos + 8 > len(buf):
            break
        chunk_id = buf[pos:pos + 4]
        if len(chunk_id) < 4:
            break
        chunk_size = struct.unpack_from("<I", buf, pos + 4)[0]
        chunk_data_start = pos + 8

        if chunk_id == b"fmt ":
            # fmt chunk: parse sample rate at offset 12 within chunk
            if chunk_data_start + 16 <= len(buf):
                sample_rate = struct.unpack_from("<I", buf, chunk_data_start + 4)[0]
                channels = struct.unpack_from("<H", buf, chunk_data_start + 2)[0]
        elif chunk_id == b"data":
            data_start = chunk_data_start
            data_end = chunk_data_start + chunk_size

        # Move to next chunk (chunks are padded to 2 bytes)
        pos = chunk_data_start + chunk_size
        if pos % 2:
            pos += 1

    if data_start < 0:
        return []  # no data chunk found

    # Read raw PCM data (16-bit signed little-endian)
    raw_pcm = buf[data_start:data_end]

    # 16-bit PCM: 2 bytes per sample
    bytes_per_sample = 2
    window_samples = int(window_s * sample_rate)
    if window_samples < 1:
        window_samples = 3200  # fallback for 16kHz

    window_bytes = window_samples * bytes_per_sample
    max_bytes = len(raw_pcm) - (len(raw_pcm) % window_bytes)

    rms_values = []
    for offset in range(0, max_bytes, window_bytes):
        chunk = raw_pcm[offset:offset + window_bytes]
        n_samples = len(chunk) // bytes_per_sample
        if n_samples < 1:
            continue
        samples = struct.unpack(f"<{n_samples}h", chunk)
        sq_sum = sum(s * s for s in samples)
        rms = math.sqrt(sq_sum / n_samples)
        dbfs = 20.0 * math.log10(rms / 32767.0) if rms > 0 else -100.0
        rms_values.append(dbfs)

    return rms_values


def _get_audio_rms(video_path, audio_path, language=None, model=None):
    """Get per-200ms RMS values, using cache when available."""
    if language is None:
        language = config.WHISPER_LANGUAGE
    if model is None:
        model = config.WHISPER_MODEL

    hash_input = f"{video_path}_{language}_{model}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    cache_file = os.path.join(str(config.TEMP_DIR), f"_audio_rms_{file_hash}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return _json.load(f)
        except Exception:
            pass

    rms = _compute_audio_rms(video_path, audio_path)

    try:
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        with open(cache_file, "w") as f:
            _json.dump(rms, f)
    except Exception:
        pass

    return rms
from utils import fmt_duration as _fmt_duration
from utils.ffmpeg_utils import _detect_gpu_accel
from utils import get_video_basename
# _fmt_duration is imported from utils — no local definition needed


def detect_scenes(video_path, threshold=None):
    """
    Detect scenes in a video using PySceneDetect ContentDetector.
    Uses frame_skip for speed and prints progress with ETA.

    Returns list of scenes: [{start, end, duration}].
    Each scene is a dict with float timestamps in seconds.
    """
    if threshold is None:
        threshold = config.SCENE_THRESHOLD

    frame_skip = getattr(config, 'SCENE_FRAME_SKIP', 2)

    print("=" * 50)
    print("SCENE DETECTION")
    print("=" * 50)
    print(f"Detector: ContentDetector (threshold={threshold})")
    print(f"Frame skip: {frame_skip} (processes every {frame_skip+1}th frame)")
    print()

    # Open video and get info
    video = open_video(str(video_path))
    duration_sec = video.duration.get_seconds()
    fps = video.frame_rate
    total_frames_est = int(duration_sec * fps)
    frames_to_process = total_frames_est // (frame_skip + 1)

    # Rough estimate: ~2000 frames/sec on CPU
    est_sec = max(1, frames_to_process / 2000)
    print(f"Video: {total_frames_est} frames @ {fps:.2f} fps")
    print(f"Duration: {duration_sec:.0f}s ({duration_sec/60:.1f}min)")
    print(f"Will process: ~{frames_to_process} frames (est. {est_sec:.0f}s / {est_sec/60:.1f}min)")
    print()

    # Use low-level SceneManager API
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    start_time = time_module.time()
    CHECK_INTERVAL = max(500, frames_to_process // 40)  # ~40 updates

    done_flag = False

    def progress_callback(frame: "np.ndarray", frame_number: int) -> None:
        nonlocal start_time, CHECK_INTERVAL, frames_to_process, done_flag
        if frame_number % CHECK_INTERVAL == 0 and frame_number > 0:
            elapsed = time_module.time() - start_time
            frames_done = min(frame_number, frames_to_process)
            pct = min(100.0, frames_done / frames_to_process * 100)
            # Skip if already at 100% (avoids repeated prints when estimate is exceeded)
            if pct >= 100.0 and done_flag:
                return
            rate = frames_done / max(elapsed, 0.1)
            remaining_frames = max(0, frames_to_process - frames_done)
            remaining = remaining_frames / max(rate, 0.1)
            print(f"  Scan: {pct:.0f}%  ({frames_done}/{frames_to_process})  "
                  f"elapsed: {_fmt_duration(elapsed)}  "
                  f"ETA: {_fmt_duration(remaining)}")
            if pct >= 100.0:
                done_flag = True

    print("Starting frame scan...")
    scene_manager.detect_scenes(
        video=video,
        frame_skip=frame_skip,
        callback=progress_callback,
    )

    elapsed = time_module.time() - start_time
    print(f"  Scan complete in {_fmt_duration(elapsed)}")

    scene_list = scene_manager.get_scene_list()
    scenes = []
    for start_time_ft, end_time_ft in scene_list:
        start_sec = start_time_ft.get_seconds()
        end_sec = end_time_ft.get_seconds()
        scenes.append({
            "start": start_sec,
            "end": end_sec,
            "duration": end_sec - start_sec,
        })

    print(f"  Found {len(scenes)} scenes")
    return scenes


def merge_short_scenes(scenes, min_duration=None, max_duration=None):
    """
    Merge short scenes with neighbours.

    Rules:
    1. Start a buffer with the first raw scene.
    2. If buffer < min_duration: merge next scene INTO buffer (build it up).
    3. If buffer >= min_duration:
       - If next scene is long (>= min_duration): buffer is final, start new buffer.
       - If next scene is short (< min_duration): DON'T extend old buffer.
         Start a NEW short buffer (will merge with subsequent short scenes).
       - If buffer >= max_duration: force-finalize, start new buffer.
    4. No snowball effect — short scenes never extend an already-adequate buffer.
    """
    if min_duration is None:
        min_duration = config.MIN_SCENE_DURATION
    if max_duration is None:
        max_duration = config.MAX_MERGE_DURATION

    if not scenes:
        return []

    merged = []
    buffer = scenes[0].copy()

    for i in range(1, len(scenes)):
        current = scenes[i].copy()

        # Force-finalize if buffer already at max
        if buffer["duration"] >= max_duration:
            merged.append(buffer)
            buffer = current
            continue

        # Buffer too short — merge current into it
        if buffer["duration"] < min_duration:
            buffer["end"] = current["end"]
            buffer["duration"] = buffer["end"] - buffer["start"]
        # Current is short — start a new buffer (don't extend old one)
        elif current["duration"] < min_duration:
            merged.append(buffer)
            buffer = current
        # Both long enough — start new buffer
        else:
            merged.append(buffer)
            buffer = current

    if buffer is not None:
        merged.append(buffer)

    print(f"  After merging short scenes: {len(merged)} scenes")
    return merged


def get_scene_transcripts(video_path, scenes, language=None):
    """
    Transcribe the FULL movie audio once, then map segments to scenes.
    Much faster than transcribing each scene separately.

    Args:
        video_path: path to video file
        scenes: list of {start, end, duration} dicts
        language: transcription language code (None = config default)

    Returns: [{start, end, duration, text}]
    """
    if not scenes:
        return []
    if language is None:
        language = config.WHISPER_LANGUAGE

    # Extract full audio once
    print("Extracting full audio for transcription...")
    print("  ffmpeg: extracting PCM audio (16kHz mono WAV)...")

    # Get video duration for progress tracking
    audio_duration = 0
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True, timeout=30
        )
        audio_duration = float(result.stdout.strip())
        est_extract = max(20, audio_duration / 250)
        print(f"  Audio duration: {audio_duration/60:.1f} min")
        gpu = _detect_gpu_accel()
        gpu_tag = "CUDA" if gpu else "CPU"
        print(f"  Est. extraction: ~{_fmt_duration(est_extract)} (ffmpeg, {gpu_tag})")
    except FileNotFoundError:
        print()
        print("  ⚠️  ffprobe/ffmpeg не найдены!")
        print("  Установи FFmpeg: https://ffmpeg.org/download.html")
        print("  Или через winget: winget install FFmpeg")
        print("  После установки перезапусти программу.")
        print()
        return [{"start": s["start"], "end": s["end"],
                 "duration": s["end"] - s["start"], "text": "",
                 "audio_peaks": {"peak_rms": 0.0, "loud_peak_count": 0,
                                 "silence_ratio": 1.0}}
                for s in scenes]
    except Exception:
        print("  Audio duration: unknown")
        print()

    temp_audio = os.path.join(str(config.TEMP_DIR), f"_full_audio_{os.getpid()}.wav")
    os.makedirs(config.TEMP_DIR, exist_ok=True)

    # Run ffmpeg with progress parsing (stderr has frame/time info)
    print("  Starting ffmpeg extraction...")
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-loglevel", "info",
             temp_audio],
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

        extract_start_time = time_module.time()
        last_report = 0.0
        time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d+)")
        for stderr_line in proc.stderr:
            m = time_pattern.search(stderr_line)
            if m:
                hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
                current_sec = hh * 3600 + mm * 60 + ss

                now = time_module.time()
                if now - last_report > 2.0:
                    last_report = now
                    wall_elapsed = now - extract_start_time
                    if audio_duration > 0:
                        pct = current_sec / audio_duration * 100
                        if current_sec > 0:
                            speed = current_sec / max(wall_elapsed, 0.1)
                            remaining_audio = audio_duration - current_sec
                            eta = remaining_audio / speed
                        else:
                            eta = 0
                        print(f"  Audio extraction: {pct:.0f}%  "
                              f"elapsed {_fmt_duration(wall_elapsed)}  "
                              f"ETA {_fmt_duration(eta)}")
                    else:
                        print(f"  Audio extraction: {_fmt_duration(current_sec)}...")

        proc.wait(timeout=900)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)

        print("  Audio extracted successfully.")
    except FileNotFoundError:
        print()
        print("  ⚠️  ffmpeg не найден!")
        print("  Установи FFmpeg: https://ffmpeg.org/download.html")
        print("  Или через winget (админ): winget install FFmpeg")
        print("  После установки перезапусти программу.")
        print()
        _cleanup_temp_audio(temp_audio)
        return [{"start": s["start"], "end": s["end"],
                 "duration": s["end"] - s["start"], "text": "",
                 "audio_peaks": {"peak_rms": 0.0, "loud_peak_count": 0,
                                 "silence_ratio": 1.0}}
                for s in scenes]
    except subprocess.TimeoutExpired:
        proc.kill()
        print("  Error: Audio extraction timed out (>15min)")
        _cleanup_temp_audio(temp_audio)
        return [{"start": s["start"], "end": s["end"],
                 "duration": s["end"] - s["start"], "text": "",
                 "audio_peaks": {"peak_rms": 0.0, "loud_peak_count": 0,
                                 "silence_ratio": 1.0}}
                for s in scenes]
    except subprocess.CalledProcessError:
        print("  Warning: Could not extract audio")
        _cleanup_temp_audio(temp_audio)
        return [{"start": s["start"], "end": s["end"],
                 "duration": s["end"] - s["start"], "text": "",
                 "audio_peaks": {"peak_rms": 0.0, "loud_peak_count": 0,
                                 "silence_ratio": 1.0}}
                for s in scenes]
    except BaseException:
        _cleanup_temp_audio(temp_audio)
        raise

    # Compute audio RMS envelope (per-200ms window) — cached per film
    print("Computing audio RMS envelope...")
    try:
        audio_rms = _get_audio_rms(video_path, temp_audio, language)
    except Exception:
        audio_rms = []
    if audio_rms:
        print(f"  Got {len(audio_rms)} RMS values ({len(audio_rms)/5:.1f}s of audio)")
    else:
        print("  RMS computation skipped or unavailable")

    # Transcribe full audio once (cached model)
    # (temp_audio is cleaned up inside _transcribe_audio_file)
    print()
    print("Starting Whisper transcription...")
    all_segments = _transcribe_audio_file(temp_audio, language)

    print(f"  Got {len(all_segments)} transcript segments total")

    # Save full transcript to JSON for reuse across clips
    from core.subtitle import save_segments_json
    video_basename = get_video_basename(video_path)
    hash_input = f"{video_path}_{language}_{config.WHISPER_MODEL}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    transcript_json = os.path.join(
        str(config.TEMP_DIR),
        f"full_transcript_{video_basename}_{file_hash}.json"
    )
    save_segments_json(all_segments, transcript_json)

    PAUSE_GAP = config.DIALOGUE_PAUSE_THRESHOLD

    # Map transcript segments to scenes by time overlap
    results = []
    for i, scene in enumerate(scenes):
        start_s = scene["start"]
        end_s = scene["end"]

        # Collect overlapping transcript segments (sorted by time)
        overlapping = []
        for seg in all_segments:
            if seg["start"] < end_s and seg["end"] > start_s:
                overlapping.append(seg)
        overlapping.sort(key=lambda s: s["start"])

        # Build ONE result per scene with advisory pause_points
        scene_text = " ".join(s["text"] for s in overlapping).strip()
        pause_points = []
        for j in range(len(overlapping) - 1):
            gap = overlapping[j + 1]["start"] - overlapping[j]["end"]
            if gap > PAUSE_GAP:
                pause_points.append({
                    "gap_start": overlapping[j]["end"],
                    "gap_end": overlapping[j + 1]["start"],
                })

        block = {
            "start": start_s,
            "end": end_s,
            "duration": end_s - start_s,
            "text": scene_text,
            "pause_points": pause_points,
        }

        # Compute audio_peaks from RMS envelope (5 values/sec = 200ms windows)
        if audio_rms:
            start_idx = int(start_s * 5)
            end_idx = int(end_s * 5)
            block_rms = audio_rms[max(0, start_idx):end_idx]
            if block_rms:
                peak_rms = max(block_rms)
                loud_peak_count = sum(1 for v in block_rms if v > -20)
                silence_ratio = sum(1 for v in block_rms if v < -50) / len(block_rms)
            else:
                peak_rms = 0.0
                loud_peak_count = 0
                silence_ratio = 1.0
        else:
            peak_rms = 0.0
            loud_peak_count = 0
            silence_ratio = 1.0
        block["audio_peaks"] = {
            "peak_rms": peak_rms,
            "loud_peak_count": loud_peak_count,
            "silence_ratio": silence_ratio,
        }
        if "cut_count" in scene:
            block["cut_count"] = scene["cut_count"]
        results.append(block)

        if (i + 1) % 20 == 0:
            print(f"  Mapped {i+1}/{len(scenes)} scenes to transcripts")

    # Cleanup temp audio (already handled by _transcribe_audio_file)
    return results


def detect_and_transcribe(video_path, language=None):
    """
    Convenience function: detect_scenes -> merge_short_scenes -> get_scene_transcripts.
    Returns final list.

    Args:
        video_path: path to video file
        language: transcription language code (None = config default)
    """
    scenes = detect_scenes(video_path)
    merged = merge_short_scenes(scenes)

    for m in merged:
        m["cut_count"] = sum(
            1 for r in scenes
            if r["start"] >= m["start"] and r["end"] <= m["end"]
        )

    if not merged:
        # Fallback: whole video as one scene
        print("  No scenes detected, treating whole video as one scene")
        merged = [{"start": 0, "end": 0, "duration": 0}]
        # Try to get video duration
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, check=True, timeout=30
            )
            duration = float(result.stdout.strip())
            merged = [{"start": 0, "end": duration, "duration": duration}]
        except Exception:
            pass

    return get_scene_transcripts(video_path, merged, language)
