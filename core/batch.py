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
from analyzers.text_analyzer import call_llm
from core.pipeline import process_clip, process_multiple
from core.subtitle import load_segments_json
from utils import get_video_basename


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
    video_basename = get_video_basename(video_path)
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

    # Step 2: Build timestamp list
    timestamps = []
    titles = []
    for scene in best_scenes:
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

        timestamps.append((_format_time(scene_start), _format_time(scene_end)))
        titles.append(title)

    # Step 3: Process clips in parallel
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

    results = process_multiple(video_path, timestamps, base_options, titles=titles, max_workers=2)

    # Move outputs to movie subfolder
    final_results = []
    for i, result in enumerate(results):
        if result and os.path.exists(result):
            src_name = Path(result).stem
            new_name = movie_output / f"{src_name}.mp4"
            shutil.move(result, new_name)
            final_results.append(str(new_name))
        else:
            final_results.append(None)

    results = final_results

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

def _validate_sub_clips(sub_clips, block_start, block_end, block_duration):
    """Validate sub-clips from LLM response.

    Applies per-sub-clip:
    1. Within block bounds
    2. Duration >= 20s (unless self-contained)
    3. Duration <= 75s
    4. Score 1-10
    5. No negative start or overflow

    Returns filtered list with logged drops.
    """
    valid = []
    for sc in sub_clips:
        sc_start = block_start + sc.get("start", 0)
        sc_end = block_start + sc.get("end", block_duration)
        sc_dur = sc_end - sc_start
        sc_score = sc.get("score", 5)
        sc_title = sc.get("title", "") or "untitled"
        sc_reason = sc.get("reason", "")

        if sc_start < 0 or sc_end > block_end:
            print(f"  ⛔ «{sc_title}» {sc_start:.0f}-{sc_end:.0f} — вне границ блока")
            continue
        if sc_dur < 20 and sc_reason not in ("самодостаточен", "self_contained"):
            print(f"  ⛔ «{sc_title}» {sc_start:.0f}-{sc_end:.0f} — слишком короткий ({sc_dur:.0f}s)")
            continue
        if sc_dur > 75:
            print(f"  ⛔ «{sc_title}» {sc_start:.0f}-{sc_end:.0f} — слишком длинный ({sc_dur:.0f}s)")
            continue
        if sc_score < 1 or sc_score > 10:
            print(f"  ⛔ «{sc_title}» — неверная оценка {sc_score}")
            continue

        valid.append({
            "start": sc_start,
            "end": sc_end,
            "duration": sc_dur,
            "text": sc.get("text", ""),
            "score": sc_score,
            "title": sc_title[:40],
        })
    return valid


def _expand_short_clips(clips, min_duration):
    """Expand clips shorter than min_duration symmetrically."""
    result = []
    for clip in clips:
        dur = clip["end"] - clip["start"]
        if dur < min_duration:
            mid = (clip["start"] + clip["end"]) / 2
            new_start = max(0, mid - min_duration / 2)
            new_end = new_start + min_duration
            clip["start"] = new_start
            clip["end"] = new_end
            clip["duration"] = new_end - new_start
        result.append(clip)
    return result


def _merge_blocks_for_llm(blocks, target_duration=120, max_duration=150):
    """Merge adjacent blocks into super-blocks of ~target_duration seconds.

    Preserves original scene boundaries within each super-block metadata.
    Combines text, pause_points, cut_count. Keeps audio_peaks from the
    longest constituent block.

    Args:
        blocks: list of block dicts from detect_and_transcribe()
        target_duration: target duration in seconds (default 120)
        max_duration: maximum duration before force-finalize (default 150)

    Returns:
        list of merged block dicts with same schema as input blocks
    """
    if not blocks:
        return []

    merged = []
    buffer = dict(blocks[0])  # shallow copy

    for b in blocks[1:]:
        b = dict(b)
        # Force-finalize if buffer already at max
        if buffer["duration"] >= max_duration:
            merged.append(buffer)
            buffer = b
            continue

        # If buffer is below target, merge this block in
        if buffer["duration"] < target_duration:
            buffer["end"] = b["end"]
            buffer["duration"] = buffer["end"] - buffer["start"]
            # Combine texts
            txt_a = buffer.get("text", "") or ""
            txt_b = b.get("text", "") or ""
            buffer["text"] = (txt_a + " " + txt_b).strip()
            # Merge pause_points
            buffer["pause_points"] = (
                buffer.get("pause_points", []) + b.get("pause_points", [])
            )
            # Sum cut_count
            buffer["cut_count"] = buffer.get("cut_count", 0) + b.get("cut_count", 0)
            # Keep audio_peaks from the longer sub-block
            if b.get("duration", 0) > buffer.get("_dominant_dur", 0):
                buffer["audio_peaks"] = b.get("audio_peaks", {})
                buffer["_dominant_dur"] = b.get("duration", 0)
        else:
            # Buffer is >= target — finalize, start new buffer
            merged.append(buffer)
            buffer = b

    if buffer:
        merged.append(buffer)

    # Strip internal helper fields
    for m in merged:
        m.pop("_dominant_dur", None)

    return merged


def find_best_clips_context(video_path, movie_title, api_key, provider="gemini",
                            max_duration=60, min_duration=15,
                            num_clips=10, score_threshold=7.0, language="ru"):
    """Context mode: detect blocks → LLM splits each block into sub-clips.

    Each block from detect_and_transcribe() carries metadata:
    pause_points, cut_count, audio_peaks. LLM decides boundaries and titles
    per block in one call. Falls back to smart centering when LLM returns empty.

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
    import json
    import time
    from pathlib import Path

    from analyzers.scene_analyzer import detect_and_transcribe
    from analyzers.text_analyzer import PROMPT_BATCH_TO_CLIPS, PROMPT_BATCH_TO_CLIPS_EN, _parse_batch_response

    video_basename = Path(video_path).stem
    total_start = time.time()
    print(f"\n🎬 {video_basename}: Context mode — block-based LLM pipeline")

    # Step 1: Detect scenes + transcribe (now returns blocks with metadata)
    print("[Context] Detecting scenes and transcribing...")
    blocks = detect_and_transcribe(video_path, language=language)

    if not blocks:
        print("  No blocks detected, falling back to standard mode")
        return None

    total_duration = blocks[-1]["end"] if blocks else 0
    print(f"  Movie duration: {_format_time(total_duration)} ({total_duration:.0f}s)")
    print(f"  {len(blocks)} blocks with transcription")

    # Step 1.5: Merge small blocks into super-blocks for LLM to work with
    before_merge = len(blocks)
    blocks = _merge_blocks_for_llm(blocks)
    print(f"  Merged {before_merge} → {len(blocks)} super-blocks "
          f"(target 120s, range 90-150s)")

    # Step 1.6: Filter out silent blocks (no dialogue) — user doesn't want clips from them
    before_filter = len(blocks)
    blocks = [b for b in blocks if b.get("text", "").strip()]
    filtered_silent = before_filter - len(blocks)
    if filtered_silent:
        print(f"  Filtered out {filtered_silent} silent block(s) (no dialogue)")

    if not blocks:
        print("  No blocks with dialogue — nothing to process")
        return None

    BATCH_SIZE = 2
    batch_template = PROMPT_BATCH_TO_CLIPS if language == "ru" else PROMPT_BATCH_TO_CLIPS_EN
    all_sub_clips = []
    total_batches = (len(blocks) + BATCH_SIZE - 1) // BATCH_SIZE

    # Step 2: Process blocks in batches through LLM
    print(f"[Context] Processing {len(blocks)} blocks in batches of {BATCH_SIZE} "
          f"(~{total_batches} LLM calls)...")
    for batch_idx in range(0, len(blocks), BATCH_SIZE):
        batch_blocks = blocks[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1

        # Build blocks_text for the combined prompt
        block_texts = []
        for i, block in enumerate(batch_blocks):
            global_idx = batch_idx + i
            block_start = block["start"]
            block_end = block["end"]
            block_dur = block_end - block_start
            dialogue = block.get("text", "").strip() or "(нет диалога)"
            cut_count = block.get("cut_count", 0)
            pause_points = block.get("pause_points", [])

            dialogue_preview = (dialogue[:80] + "...") if len(dialogue) > 80 else dialogue
            print(f"\n  Block {global_idx+1}/{len(blocks)}: {_format_time(block_start)}-{_format_time(block_end)} ({block_dur:.0f}s)")
            print(f"    Dialogue: {dialogue_preview}")
            print(f"    Cuts: {cut_count}, Pauses: {len(pause_points)}")

            if language == "ru":
                block_texts.append(
                    f"--- БЛОК {i} ({_format_time(block_start)}-{_format_time(block_end)}, {block_dur:.0f}s) ---\n"
                    f"Диалог: {dialogue}\n"
                    f"Смен кадра: {cut_count} (высокое = экшн)\n"
                    f"Паузы в диалоге: {pause_points}"
                )
            else:
                block_texts.append(
                    f"--- BLOCK {i} ({_format_time(block_start)}-{_format_time(block_end)}, {block_dur:.0f}s) ---\n"
                    f"Dialogue: {dialogue}\n"
                    f"Cut count: {cut_count} (high = action)\n"
                    f"Dialogue pauses: {pause_points}"
                )

        blocks_text = "\n\n".join(block_texts)
        prompt = batch_template.format(
            movie_name=movie_title,
            blocks_text=blocks_text,
        )

        print(f"\n  ── Batch {batch_num}/{total_batches} "
              f"(blocks {batch_idx+1}-{batch_idx+len(batch_blocks)}) ──")

        # Call LLM once for the entire batch
        raw_response = None
        try:
            raw_response = call_llm(prompt, api_key, provider, max_tokens=4096)
        except Exception as e:
            print(f"  ⚠️ Batch {batch_num} LLM failed: {e}")

        # Parse batch response into per-block clip lists
        if raw_response:
            block_start_times = [b["start"] for b in batch_blocks]
            batch_clips = _parse_batch_response(raw_response, block_start_times)
        else:
            batch_clips = {}

        # Log raw response if nothing was parsed
        if raw_response and not batch_clips:
            raw_short = raw_response.strip()
            if len(raw_short) > 500:
                raw_short = raw_short[:500] + "..."
            print(f"  ⚠️ Batch {batch_num}: LLM returned 0 parsed clips. Raw (truncated):")
            print(f"     {raw_short}")

        # Process each block in the batch
        for i, block in enumerate(batch_blocks):
            global_idx = batch_idx + i
            block_start = block["start"]
            block_end = block["end"]
            block_dur = block_end - block_start
            dialogue = block.get("text", "").strip()

            # Get clips for this block (absolute timestamps from batch parser)
            block_clips = batch_clips.get(i, [])

            # Validate — batch clips have absolute timestamps, so pass block_start=0
            valid_clips = _validate_sub_clips(block_clips, 0, block_end, block_dur) if block_clips else []
            for vc in valid_clips:
                vc["text"] = dialogue

            # Debug logging when LLM returns 0 valid clips for this block
            if len(valid_clips) == 0:
                if block_clips:
                    # Parser found clips but validation rejected all
                    for sc in block_clips:
                        sc_start = sc.get("start", 0)
                        sc_end = sc.get("end", block_end)
                        sc_dur = sc_end - sc_start
                        sc_score = sc.get("score", 5)
                        sc_title = sc.get("title", "") or "untitled"
                        sc_reason = sc.get("reason", "")
                        reasons = []
                        if sc_start < block_start or sc_end > block_end:
                            reasons.append("out_of_bounds")
                        if sc_dur < 20 and sc_reason not in ("самодостаточен", "self_contained"):
                            reasons.append(f"too_short({sc_dur:.0f}s)")
                        if sc_dur > 75:
                            reasons.append(f"too_long({sc_dur:.0f}s)")
                        if sc_score < 1 or sc_score > 10:
                            reasons.append(f"bad_score({sc_score})")
                        print(f"    ⛔ Rejected: «{sc_title}» {sc_start:.0f}-{sc_end:.0f}s "
                              f"dur={sc_dur:.0f}s score={sc_score} reason={'/'.join(reasons)}")
                else:
                    print(f"  ℹ️ Block {global_idx+1}: no clips from LLM")
            else:
                print(f"  ✅ Block {global_idx+1}: {len(valid_clips)} valid clip(s)")

            # Fallback: if LLM returned nothing but block has dialogue, use smart centering
            if not valid_clips and dialogue:
                segments = [{"start": block_start, "end": block_end, "text": dialogue}]
                fb_start, fb_end = _find_best_window(segments, block_start, block_end, max_duration)
                if fb_start is not None:
                    print(f"    → Fallback: smart centering ({fb_start:.0f}-{fb_end:.0f})")
                    valid_clips.append({
                        "start": fb_start,
                        "end": fb_end,
                        "duration": fb_end - fb_start,
                        "text": dialogue,
                        "score": 5.0,
                        "title": movie_title[:40],
                    })

            all_sub_clips.extend(valid_clips)

    if not all_sub_clips:
        print("  No valid clips from LLM results")
        return None

    # Step 3: Sort by start time
    all_sub_clips.sort(key=lambda x: x["start"])

    # Step 4: Score threshold filter
    before = len(all_sub_clips)
    filtered = [c for c in all_sub_clips if c["score"] >= score_threshold]
    if not filtered:
        # If nothing passes threshold, keep top N clips anyway
        all_sub_clips.sort(key=lambda x: x["score"], reverse=True)
        filtered = all_sub_clips[:max(1, num_clips // 4)]
        print(f"  Score threshold ({score_threshold}): no clips qualify, keeping top {len(filtered)}")
    else:
        print(f"  Score threshold ({score_threshold}): {before} → {len(filtered)} clip(s)")

    # Step 5: Diversity filter
    before = len(filtered)
    filtered.sort(key=lambda x: x["score"], reverse=True)
    if total_duration > 0:
        filtered = _diversity_filter(filtered, num_clips, total_duration)
    else:
        filtered = filtered[:num_clips]
    print(f"  Diversity filter: {before} → {len(filtered)} clip(s)")

    # Step 6: Deduplication
    before = len(filtered)
    filtered = _deduplicate_clips(filtered)
    print(f"  Dedup: {before} → {len(filtered)} clip(s)")

    # Step 7: Min duration expansion
    before = len(filtered)
    filtered = _expand_short_clips(filtered, min_duration)
    print(f"  Min duration expansion ({min_duration}s): {before} → {len(filtered)} clip(s)")

    # Step 8: Sort by start and resolve overlaps
    filtered.sort(key=lambda x: x["start"])
    for i in range(1, len(filtered)):
        prev = filtered[i-1]
        curr = filtered[i]
        if curr["start"] < prev["end"]:
            mid = (prev["end"] + curr["start"]) / 2
            if prev["end"] > mid:
                prev["end"] = mid
                prev["duration"] = prev["end"] - prev["start"]
            if curr["start"] < mid:
                curr["start"] = mid
                curr["duration"] = curr["end"] - curr["start"]
            print(f"  Перекрытие разрешено: разрез по {_format_time(mid)}")

    elapsed = time.time() - total_start
    print(f"\n✓ {len(filtered)} clip(s) selected in {elapsed:.0f}s")
    return filtered


# ---------------------------------------------------------------------------
# Diversity filter — spread selected clips across the movie timeline
# ---------------------------------------------------------------------------

def _diversity_filter(scenes, num_clips, total_duration, min_score=7.0):
    """Spread selected clips across the movie timeline.

    Divides the movie into num_clips segments and picks the best scene
    from each segment. If a segment has no qualifying scenes, fills
    from remaining (highest-score) scenes.

    Args:
        scenes: list of {start, end, score, ...} sorted by score desc
        num_clips: max number of clips to return
        total_duration: total movie duration in seconds
        min_score: minimum score for a scene to be a segment candidate (default 7.0)

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
            and s["score"] >= min_score
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


def _deduplicate_clips(clips, min_gap=120.0):
    sorted_clips = sorted(clips, key=lambda x: x["start"])
    kept = []
    for clip in sorted_clips:
        if not kept:
            kept.append(clip)
            continue
        gap = clip["start"] - kept[-1]["start"]
        if gap < min_gap:
            if clip["score"] > kept[-1]["score"]:
                removed = kept.pop()
                print(f"🗑️ Клип «{removed.get('title','')}» удалён (дубликат {clip.get('title','')}, дистанция {gap:.0f}s)")
                kept.append(clip)
            else:
                print(f"🗑️ Клип «{clip.get('title','')}» удалён (дубликат {kept[-1].get('title','')}, дистанция {gap:.0f}s)")
        else:
            kept.append(clip)
    return kept


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
