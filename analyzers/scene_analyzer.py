"""
MovieShort AI — Scene detection and transcription analyzer.
"""
import hashlib
import re
import time as time_module
import subprocess
import os

from scenedetect import open_video, SceneManager, ContentDetector

from core.subtitle import transcribe, _transcribe_audio_file
import config
from utils import fmt_duration as _fmt_duration
from utils.ffmpeg_utils import _detect_gpu_accel
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

    def progress_callback(frame: "np.ndarray", frame_number: int) -> None:
        nonlocal start_time, CHECK_INTERVAL, frames_to_process
        if frame_number % CHECK_INTERVAL == 0 and frame_number > 0:
            elapsed = time_module.time() - start_time
            frames_done = min(frame_number, frames_to_process)
            pct = min(100.0, frames_done / frames_to_process * 100)
            # Skip if already at 100% (avoids repeated prints when estimate is exceeded)
            if pct >= 100.0 and hasattr(progress_callback, "_done"):
                return
            rate = frames_done / max(elapsed, 0.1)
            remaining_frames = max(0, frames_to_process - frames_done)
            remaining = remaining_frames / max(rate, 0.1)
            print(f"  Scan: {pct:.0f}%  ({frames_done}/{frames_to_process})  "
                  f"elapsed: {_fmt_duration(elapsed)}  "
                  f"ETA: {_fmt_duration(remaining)}")
            if pct >= 100.0:
                progress_callback._done = True

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
                 "duration": s["end"] - s["start"], "text": ""}
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
                    if audio_duration > 0:
                        pct = current_sec / audio_duration * 100
                        print(f"  Audio extraction: {pct:.0f}%  "
                              f"({_fmt_duration(current_sec)} / {_fmt_duration(audio_duration)})")
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
        return [{"start": s["start"], "end": s["end"],
                 "duration": s["end"] - s["start"], "text": ""}
                for s in scenes]
    except subprocess.TimeoutExpired:
        proc.kill()
        print("  Error: Audio extraction timed out (>15min)")
        return [{"start": s["start"], "end": s["end"],
                 "duration": s["end"] - s["start"], "text": ""}
                for s in scenes]
    except subprocess.CalledProcessError:
        print("  Warning: Could not extract audio")
        return [{"start": s["start"], "end": s["end"],
                 "duration": s["end"] - s["start"], "text": ""}
                for s in scenes]

    # Transcribe full audio once (cached model)
    print()
    print("Starting Whisper transcription...")
    all_segments = _transcribe_audio_file(temp_audio, language)

    print(f"  Got {len(all_segments)} transcript segments total")

    # Save full transcript to JSON for reuse across clips
    from core.subtitle import save_segments_json
    video_basename = os.path.basename(str(video_path)).split('.')[0]
    hash_input = f"{video_path}_{language}_{config.WHISPER_MODEL}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    transcript_json = os.path.join(
        str(config.TEMP_DIR),
        f"full_transcript_{video_basename}_{file_hash}.json"
    )
    save_segments_json(all_segments, transcript_json)

    MIN_SCENE = config.MIN_SCENE_DURATION
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

        # Split scene at dialogue pauses > threshold
        if len(overlapping) >= 2:
            sub_groups = []
            cur_group = [overlapping[0]]
            for j in range(1, len(overlapping)):
                gap = overlapping[j]["start"] - cur_group[-1]["end"]
                if gap > PAUSE_GAP:
                    sub_groups.append(cur_group)
                    cur_group = [overlapping[j]]
                else:
                    cur_group.append(overlapping[j])
            sub_groups.append(cur_group)

            # Build result from each subgroup that meets min duration
            for group in sub_groups:
                sub_text = " ".join(s["text"] for s in group).strip()
                if not sub_text:
                    continue
                sub_start = max(start_s, group[0]["start"])
                sub_end = min(end_s, group[-1]["end"])
                sub_dur = sub_end - sub_start
                # Allow scenes slightly shorter than min_duration if they have dialogue
                if sub_dur >= MIN_SCENE * 0.7:
                    results.append({
                        "start": sub_start,
                        "end": sub_end,
                        "duration": sub_dur,
                        "text": sub_text,
                    })
        else:
            # Single segment or none — use scene as-is
            scene_text = " ".join(s["text"] for s in overlapping).strip()
            results.append({
                "start": start_s,
                "end": end_s,
                "duration": end_s - start_s,
                "text": scene_text,
            })

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
