"""
MovieShort AI — Batch processor.
Processes a full movie: auto-detect best scenes → process each into a Short.
"""
import json
import os
import re
import shutil
from pathlib import Path

import config
from analyzers.detector import find_best_clips_standard
from analyzers.text_analyzer import ask_llm_context_mode, call_llm
from core.pipeline import process_clip
from core.subtitle import load_segments_json


def process_movie(video_path, settings=None):
    """
    Full auto pipeline: analyze movie → find best clips → process each → output.

    Args:
        video_path: path to movie file
        settings: dict with keys:
            - max_duration (int): max clip length in seconds (default 60)
            - min_duration (int): min clip length in seconds (default 15)
            - subtitles (bool): enable subtitles (default True)
            - face_tracking (bool): enable face tracking (default True)
            - api_key (str): API key
            - film_language (str): 'ru' or 'en'
            - anti_copyright (bool): enable anti-copyright measures
            - blur_background (bool): enable blurred background
            - banner_top (int): top banner padding
            - banner_bottom (int): bottom banner padding
            - num_clips (int): max number of clips to produce
            - score_threshold (float): minimum score for clip selection
            - auto_cleanup (bool): delete temp files after processing

    Returns:
        List of output file paths (None entries for failed clips).
    """
    if settings is None:
        settings = {}

    max_duration = settings.get("max_duration", config.DEFAULT_MAX_CLIP_DURATION)
    min_duration = settings.get("min_duration", config.DEFAULT_MIN_CLIP_DURATION)
    subtitles = settings.get("subtitles", True)
    face_tracking = settings.get("face_tracking", True)
    api_key = settings.get("api_key", config.GEMINI_API_KEY)
    film_language = settings.get("film_language", "ru")

    video_path = str(video_path)
    movie_title = settings.get("movie_title", "") or Path(video_path).stem
    # Sanitize for folder name (remove chars invalid on Windows)
    safe_name = re.sub(r'[\\/:*?"<>|]', '', movie_title).strip() or "untitled"

    # Create output subdirectory for this movie
    movie_output = config.OUTPUT_DIR / safe_name
    os.makedirs(movie_output, exist_ok=True)

    print(f"Processing movie: {os.path.basename(video_path)}")
    print(f"Options: subtitles={subtitles}, face_tracking={face_tracking}, language={film_language}")
    print()

    # Check settings
    analysis_mode = settings.get("analysis_mode", "context")
    llm_provider = settings.get("llm_provider", "gemini")

    # Step 1: Find best clips
    print("=" * 50)
    print("STEP 1: Finding best scenes...")
    print("=" * 50)

    best_scenes = None

    # Context mode: LLM sees real scene transcripts, picks by number
    if analysis_mode == "context" and movie_title and api_key:
        print(f"  Режим: контекстный ({llm_provider})")
        print(f"  → Context mode: LLM видит текст сцен, выбирает по номерам")
        best_scenes = find_best_clips_context(
            video_path, movie_title, api_key, llm_provider,
            max_duration, min_duration,
            num_clips=settings.get("num_clips", config.DEFAULT_NUM_CLIPS),
            score_threshold=settings.get("score_threshold", 7.0),
            language=film_language,
        )
        if best_scenes is None:
            print("  Context mode failed, falling back to standard mode...")

    # Standard mode: NO LLM, random selection
    if best_scenes is None:
        print(f"  Режим: стандартный")
        print(f"  → Standard mode: без LLM, названия не генерируются")
        best_scenes = find_best_clips_standard(
            video_path,
            max_duration=max_duration,
            min_duration=min_duration,
            num_clips=settings.get("num_clips", config.DEFAULT_NUM_CLIPS),
        )

    if movie_title:
        print(f"  Movie: {movie_title}")

    if not best_scenes:
        print("No suitable scenes found.")
        return []

    print()
    print("=" * 50)
    print(f"STEP 2: Processing {len(best_scenes)} clips...")
    print("=" * 50)

    # Find pre-transcribed transcript JSON — match by video+language+model hash
    import hashlib
    transcript_json = None
    video_basename = Path(video_path).stem.split('.')[0]
    hash_input = f"{video_path}_{film_language}_{config.WHISPER_MODEL}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    expected = str(config.TEMP_DIR / f"full_transcript_{video_basename}_{file_hash}.json")
    if os.path.exists(expected):
        transcript_json = expected
        print(f"Found pre-transcribed transcript: full_transcript_{video_basename}_{file_hash}.json")

    # Pre-load transcript segments for smart clip centering
    clip_segments = None
    if transcript_json and os.path.exists(transcript_json):
        try:
            clip_segments = load_segments_json(transcript_json)
        except Exception:
            pass

    # Step 2: Process each clip
    results = []
    base_options = {
        "subtitles": subtitles,
        "face_tracking": face_tracking,
        "max_duration": max_duration,
        "anti_copyright": settings.get("anti_copyright", config.DEFAULT_ANTI_COPYRIGHT),
        "blur_background": settings.get("blur_background", config.DEFAULT_BLUR_BACKGROUND),
        "banner_top": settings.get("banner_top", config.DEFAULT_BANNER_TOP),
        "banner_bottom": settings.get("banner_bottom", config.DEFAULT_BANNER_BOTTOM),
    }
    if transcript_json:
        base_options["transcript_path"] = transcript_json

    for i, scene in enumerate(best_scenes):
        scene_start = scene["start"]
        scene_end = scene["end"]
        scene_dur = scene_end - scene_start
        title = scene.get("title", "")

        # Smart centering: if scene is longer than max_duration, find best window
        if scene_dur > max_duration and clip_segments:
            new_start, new_end = _find_best_window(
                clip_segments, scene_start, scene_end, max_duration
            )
            if new_start is not None:
                orig_start_fmt = _format_time(scene_start)
                orig_end_fmt = _format_time(scene_end)
                scene_start, scene_end = new_start, new_end
                scene_dur = scene_end - scene_start
                print(f"  🎯 Сцена {orig_start_fmt}-{orig_end_fmt} ({scene_dur:.0f}s): "
                      f"центрирован на диалоге → {_format_time(scene_start)}-{_format_time(scene_end)}")

        print(f"\n--- Clip {i+1}/{len(best_scenes)}: "
              f"{_format_time(scene_start)} - {_format_time(scene_end)} ---")

        start = _format_time(scene_start)
        end = _format_time(scene_end)

        # Override output naming to put in movie subfolder
        result = process_clip(video_path, start, end, base_options, title=title)

        # Move to movie subfolder, preserving LLM title in filename
        if result and os.path.exists(result):
            src_name = Path(result).stem
            new_name = movie_output / f"{src_name}.mp4"
            shutil.move(result, new_name)
            results.append(str(new_name))
        else:
            results.append(None)

    # Summary
    done = sum(1 for r in results if r is not None)
    print(f"\n{'=' * 50}")
    print(f"Complete: {done}/{len(best_scenes)} clips ready")
    print(f"Output: {movie_output}")

    # Cost estimate (load from user config)
    try:
        from utils import user_config as _uc
        _cfg = _uc.load()
        cpm = _cfg.get("cost_per_minute", 0.0)
        if cpm > 0:
            import subprocess as _sp
            dur_str = _sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=30
            ).stdout.strip()
            dur_min = float(dur_str) / 60 if dur_str else 0
            est_cost = dur_min * cpm
            print(f"💰 Стоимость: ~{est_cost:.2f} руб ({cpm:.2f} руб/мин × {dur_min:.1f} мин)")
    except Exception:
        pass

    print(f"{'=' * 50}")

    # Print output list
    for i, r in enumerate(results):
        if r:
            fname = os.path.basename(r)
            print(f"  ✅ {fname}")
        else:
            print(f"  ❌ clip {i+1} — failed")

    # Auto-cleanup: delete temp files if enabled
    if settings.get("auto_cleanup", False) and os.path.exists(config.TEMP_DIR):
        for f in os.listdir(str(config.TEMP_DIR)):
            fp = os.path.join(str(config.TEMP_DIR), f)
            try:
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.unlink(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
            except Exception:
                pass
        print(f"  🧹 Временные файлы удалены")

    return results


def _format_time(seconds):
    """Format seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _snap_scene_boundary(clip_segments, scene_start, scene_end, max_dur):
    """Snap clip end to nearest sentence boundary within [max_dur-3, max_dur+5].

    Priority:
    1. Sentence end (. ! ?) in [max_dur-3, max_dur+5]
    2. Pause in dialogue >0.8s between whisper segments
    3. Word boundary (fallback)

    Returns (new_end, extended) where new_end <= scene_end.
    """
    if not clip_segments:
        return scene_start + max_dur, False

    # Filter segments in the [scene_start, scene_end] range
    overlapping = [
        s for s in clip_segments
        if s["start"] < scene_end and s["end"] > scene_start
    ]
    if not overlapping:
        return scene_start + max_dur, False

    # Search range: [max_dur - 3, max_dur + 5]
    search_start = scene_start + max(0, max_dur - 3)
    search_end = min(scene_end, scene_start + max_dur + 5)

    # Priority 1: Find sentence-ending punctuation in word timestamps
    best_end = None

    for seg in overlapping:
        seg_end = seg["end"]
        if search_start <= seg_end <= search_end:
            text = seg.get("text", "").strip()
            if text and text[-1] in ".!?…":
                if best_end is None or seg_end > best_end:
                    best_end = seg_end

    if best_end is not None:
        extended = best_end > scene_start + max_dur
        return best_end, extended

    # Priority 2: Find dialogue pause > 0.8s between consecutive segments
    sorted_segs = sorted(overlapping, key=lambda s: s["start"])
    for i in range(len(sorted_segs) - 1):
        gap = sorted_segs[i + 1]["start"] - sorted_segs[i]["end"]
        if gap > 0.8 and search_start <= sorted_segs[i]["end"] <= search_end:
            if best_end is None or sorted_segs[i]["end"] > best_end:
                best_end = sorted_segs[i]["end"]

    if best_end is not None:
        extended = best_end > scene_start + max_dur
        return best_end, extended

    # Priority 3: Word boundary — just use max_dur
    new_end = min(scene_end, scene_start + max_dur)
    return new_end, False


# ---------------------------------------------------------------------------
# Context Mode — LLM sees real scene transcripts, picks best scenes directly
# ---------------------------------------------------------------------------

def find_best_clips_context(video_path, movie_title, api_key, provider="gemini",
                            max_duration=60, min_duration=15,
                            num_clips=10, score_threshold=7.0, language="ru"):
    """Context mode: detect scenes, transcribe, send transcripts to LLM.

    LLM sees the real dialogue text of each scene and returns scene numbers
    directly — no matching, no stemming, no timing guesswork.

    Args:
        video_path: path to video file
        movie_title: movie name for LLM context
        api_key: API key
        provider: 'gemini' or 'yandex'
        max_duration: max clip length in seconds
        min_duration: min clip length in seconds
        num_clips: max number of clips to return (default 10)
        score_threshold: minimum score (default 7.0)
        language: 'ru' or 'en' for transcription and prompts

    Returns list of {start, end, duration, text, score, title} or None.
    """
    from analyzers.scene_analyzer import detect_and_transcribe

    print("[Context] Step 1: Detecting scenes and transcribing...")
    scenes = detect_and_transcribe(video_path, language=language)

    if not scenes:
        print("  No scenes detected, falling back to standard mode")
        return None

    total_duration = scenes[-1]["end"] if scenes else 0
    print(f"  Movie duration: {_format_time(total_duration)} ({total_duration:.0f}s)")
    print(f"  {len(scenes)} scenes with transcription")

    # Load clip_segments (whisper word-level timestamps) for smart snapping
    import hashlib
    clip_segments = None
    from core.subtitle import load_segments_json
    video_basename = os.path.basename(str(video_path)).split('.')[0]
    hash_input = f"{video_path}_{language}_{config.WHISPER_MODEL}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    transcript_path = str(config.TEMP_DIR / f"full_transcript_{video_basename}_{file_hash}.json")
    if os.path.exists(transcript_path):
        try:
            clip_segments = load_segments_json(transcript_path)
        except Exception:
            pass

    # Step 2: Send scene transcripts to LLM
    print("[Context] Step 2: Asking LLM to analyze scene transcripts...")
    rated = ask_llm_context_mode(
        scenes, movie_title, api_key, provider,
        language=language,
    )

    if not rated:
        print("  LLM returned no results, falling back to standard mode")
        return None

    print(f"  LLM rated {len(rated)} scenes")

    # Check for duplicate scene numbers
    seen_nums = set()
    duplicates = 0
    for item in rated:
        if item["scene_num"] in seen_nums:
            print(f"  ⚠️ Дубликат сцены {item['scene_num']} — пропускаю")
            duplicates += 1
        seen_nums.add(item["scene_num"])
    if duplicates:
        print(f"  {duplicates} duplicates ignored")

    # Step 3: Build clip list from LLM results
    print("[Context] Step 3: Building clip list...")
    found_clips = []

    for item in rated:
        scene_num = item.get("scene_num", 0)
        score = item.get("score", 5)
        title = item.get("title", "")

        # Convert 1-based to 0-based index
        idx = scene_num - 1
        if idx < 0 or idx >= len(scenes):
            print(f"  Scene {scene_num}: invalid number (have {len(scenes)} scenes)")
            continue

        scene = scenes[idx]
        start = scene["start"]
        end = scene["end"]
        duration = end - start

        print(f"  Scene {scene_num}: {_format_time(start)}-{_format_time(end)} "
              f"({duration:.0f}s, score={score}) {title}")

        # LLM Split-or-Keep: for long scenes with dialogue and API key
        from analyzers.text_analyzer import PROMPT_SCENE_SPLIT_OR_KEEP, _parse_split_or_keep

        if duration > 30 and clip_segments and api_key:
            # Build dialogue text for this scene
            overlapping = [
                s for s in clip_segments
                if s["start"] < end and s["end"] > start
            ]
            overlapping.sort(key=lambda s: s["start"])
            dialogue = " ".join(s.get("text", "") for s in overlapping).strip()

            if dialogue:
                try:
                    prompt = PROMPT_SCENE_SPLIT_OR_KEEP.format(
                        movie_title=movie_title,
                        dialogue=dialogue,
                        scene_duration=duration,
                    )
                    raw = call_llm(prompt, api_key, provider, max_tokens=512)
                    split_result = _parse_split_or_keep(raw, duration)

                    if split_result["decision"] == "keep":
                        # Entire scene as one clip — ignore max_duration
                        print(f"         LLM: ОДНА цельная сцена — {duration:.0f}s (max_duration ignored)")
                        found_clips.append({
                            "start": start,
                            "end": end,
                            "duration": duration,
                            "text": scene.get("text", ""),
                            "score": score,
                            "title": title[:40],
                        })
                        continue

                    elif split_result["decision"] == "split":
                        print(f"         LLM: НЕСКОЛЬКО частей — {len(split_result['parts'])} parts")
                        for rel_start, rel_end in split_result["parts"]:
                            abs_start = start + rel_start
                            abs_end = start + rel_end
                            snapped_end, _ = _snap_scene_boundary(
                                clip_segments, abs_start, abs_end, max_duration
                            )
                            part_dur = snapped_end - abs_start
                            found_clips.append({
                                "start": abs_start,
                                "end": snapped_end,
                                "duration": part_dur,
                                "text": scene.get("text", ""),
                                "score": score,
                                "title": title[:40],
                            })
                            print(f"           Часть: {_format_time(abs_start)}-{_format_time(snapped_end)} ({part_dur:.0f}s)")
                        continue
                except Exception:
                    print(f"         LLM split-or-keep failed, using smart snapping fallback")

        # Duration enforcement — expand short scenes to min_duration
        if duration < min_duration:
            # Priority: anchor expansion on sentence boundary (first whisper segment)
            if clip_segments:
                overlapping = sorted(
                    [s for s in clip_segments if s["start"] < end and s["end"] > start],
                    key=lambda s: s["start"],
                )
                if overlapping and overlapping[0]["start"] < start + min_duration:
                    sentence_start = overlapping[0]["start"]
                    new_start = max(start, sentence_start)
                    new_end = min(end, new_start + min_duration)
                    if new_end > end:
                        new_end = end
                        new_start = max(start, new_end - min_duration)
                    start, end = new_start, new_end
                else:
                    # Fallback: symmetric expansion
                    mid = (start + end) / 2
                    start = max(0, mid - min_duration / 2)
                    end = start + min_duration
            else:
                # No transcript available — simple center expansion
                mid = (start + end) / 2
                start = max(0, mid - min_duration / 2)
                end = start + min_duration
            duration = end - start
            print(f"         Увеличена до {duration:.0f}с (мин. длительность)")
        elif duration > max_duration:
            new_end, extended = _snap_scene_boundary(clip_segments, start, end, max_duration)
            if extended:
                print(f"         Укорочена до {new_end - start:.0f}с (снаппинг по границе предложения, +{new_end - (start + max_duration):.0f}с)")
            else:
                print(f"         Укорочена до {max_duration}с (макс. длительность)")
            end = new_end
            duration = end - start

        found_clips.append({
            "start": start,
            "end": end,
            "duration": duration,
            "text": scene.get("text", ""),
            "score": score,
            "title": title[:40],
        })

    if not found_clips:
        print("  No valid clips from LLM results")
        return None

    # Filter by score threshold
    filtered = [c for c in found_clips if c["score"] >= score_threshold]
    if not filtered:
        found_clips.sort(key=lambda x: x["score"], reverse=True)
        filtered = found_clips[:max(1, num_clips // 4)]
    else:
        filtered.sort(key=lambda x: x["score"], reverse=True)
        # Apply diversity filter to spread clips across timeline
        if total_duration > 0:
            filtered = _diversity_filter(filtered, num_clips, total_duration)
        else:
            filtered = filtered[:num_clips]

    filtered.sort(key=lambda x: x["start"])

    # Resolve overlaps between consecutive clips (from min-duration extension)
    for i in range(1, len(filtered)):
        prev = filtered[i-1]
        curr = filtered[i]
        if curr["start"] < prev["end"]:
            overlap_start = curr["start"]
            overlap_end = min(prev["end"], curr["end"])
            mid = (overlap_start + overlap_end) / 2
            if prev["end"] > mid:
                prev["end"] = mid
                prev["duration"] = prev["end"] - prev["start"]
            if curr["start"] < mid:
                curr["start"] = mid
                curr["duration"] = curr["end"] - curr["start"]
            print(f"         Перекрытие разрешено: разрез по {_format_time(mid)}")

    found_clips = filtered
    print(f"  Total: {len(found_clips)} clips ready (score ≥ {score_threshold})")
    return found_clips


# ---------------------------------------------------------------------------
# Diversity filter — spread selected clips across the movie timeline
# ---------------------------------------------------------------------------

def _diversity_filter(scenes, num_clips, total_duration):
    """Spread selected clips across the movie timeline.

    Divides the movie into num_clips segments and picks the best scene
    from each segment. If a segment has no qualifying scenes, fills
    from remaining (highest-score) scenes.

    Args:
        scenes: list of {start, end, score, ...} sorted by score desc
        num_clips: max number of clips to return
        total_duration: total movie duration in seconds

    Returns:
        list of scenes sorted by start time, max num_clips entries
    """
    if len(scenes) <= num_clips:
        return scenes

    segment_dur = total_duration / num_clips
    selected = []
    used_indices = set()

    for seg_idx in range(num_clips):
        seg_start = seg_idx * segment_dur
        seg_end = (seg_idx + 1) * segment_dur
        # Find best (highest score) scene in this segment
        candidates = [
            (i, s) for i, s in enumerate(scenes)
            if seg_start <= s["start"] < seg_end and i not in used_indices
        ]
        candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        if candidates:
            best_idx, best_scene = candidates[0]
            selected.append(best_scene)
            used_indices.add(best_idx)

    # If we have fewer than num_clips, fill from remaining top-scored
    if len(selected) < num_clips:
        remaining = [
            s for i, s in enumerate(scenes) if i not in used_indices
        ]
        remaining.sort(key=lambda x: x["score"], reverse=True)
        while len(selected) < num_clips and remaining:
            selected.append(remaining.pop(0))

    selected.sort(key=lambda x: x["start"])
    return selected


# ---------------------------------------------------------------------------
# Smart clip centering — find the best window by dialogue density
# ---------------------------------------------------------------------------

def _find_best_window(segments, scene_start, scene_end, window_dur):
    """Find the best `window_dur` window within [scene_start, scene_end].

    "Best" = the window with the highest word count (dialogue density).
    Falls back to the first window if no segments overlap the scene.

    Returns (new_start, new_end) or (None, None) if no adjustment needed.
    """
    # Filter segments overlapping this scene
    overlapping = [
        s for s in segments
        if s["start"] < scene_end and s["end"] > scene_start
    ]
    if not overlapping:
        return None, None  # no dialogue — keep original start

    scene_len = scene_end - scene_start
    window_dur = min(window_dur, scene_len)

    # Slide window across the scene, compute word count for each position
    best_start = scene_start
    best_words = -1

    # Step size = 1 second for precision
    step = 1.0
    max_start = scene_end - window_dur

    pos = scene_start
    while pos <= max_start:
        win_end = pos + window_dur
        # Count words in this window
        word_count = 0
        for s in overlapping:
            # Check if segment overlaps the window
            if s["start"] < win_end and s["end"] > pos:
                word_count += len(s["text"].split())
        if word_count > best_words:
            best_words = word_count
            best_start = pos
        pos += step

    new_start = best_start
    new_end = new_start + window_dur

    # Extend to natural boundaries:
    # - Start: nearest segment start (go back to where dialogue begins)
    for s in overlapping:
        if s["start"] < new_start and s["end"] > new_start:
            new_start = min(new_start, s["start"])
        elif s["start"] <= new_start < s["end"]:
            new_start = min(new_start, s["start"])

    # - End: nearest segment end (go forward to where dialogue ends)
    for s in overlapping:
        if s["start"] < new_end < s["end"]:
            new_end = max(new_end, s["end"])
        elif s["start"] >= new_end:
            # This segment starts after the window — might be part of the same scene
            # Don't extend past scene boundaries though
            pass

    # Clamp to scene boundaries
    new_start = max(scene_start, new_start)
    new_end = min(scene_end, new_end)

    # Ensure at least some minimum duration
    if new_end - new_start < 5:
        new_start = scene_start
        new_end = min(scene_end, scene_start + window_dur)

    return new_start, new_end
