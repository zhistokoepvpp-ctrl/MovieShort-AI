"""
MovieShort AI — Pipeline orchestrator.
Connects FFmpeg clipping, subtitle generation, face tracking, and vertical crop.
"""
import os
import re
import subprocess
from pathlib import Path

import config
from utils.ffmpeg_utils import (
    clip_video, embed_subtitles,
    convert_to_vertical, FFmpegError,
    pad_with_banners, blur_background,
    _detect_gpu_accel,
)
from core.subtitle import (
    transcribe, generate_srt, generate_word_group_srt,
    load_segments_json, filter_segments_in_range
)
from core.processor import apply_vertical_crop


def _time_to_seconds(hh_mm_ss: str) -> float:
    """Convert 'HH:MM:SS' or 'MM:SS' to seconds."""
    parts = hh_mm_ss.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def process_clip(video_path, start_time, end_time, options=None, title=""):
    """
    Process a single clip: cut → subtitle → scale → embed subtitles → pad → export.

    Args:
        video_path: path to source video
        start_time: "HH:MM:SS"
        end_time: "HH:MM:SS"
        options: dict with keys:
            - subtitles (bool): generate and embed subtitles (default True)
            - face_tracking (bool): apply face-tracking vertical crop (default True)
            - max_duration (int): max clip length in sec (default 60)
            - anti_copyright (bool): enable anti-copyright measures (default True)
            - blur_background (bool): enable blurred background (default True)
            - banner_top (int): top banner padding (default 300)
            - banner_bottom (int): bottom banner padding (default 300)
            - font_style (dict): subtitle font settings
            - transcript_path (str): path to pre-transcribed JSON
        title: optional short title to include in output filename

    Returns:
        Path to the final output file, or None on failure.
    """
    if options is None:
        options = {}

    subtitles_enabled = options.get("subtitles", True)
    face_tracking_enabled = options.get("face_tracking", True)
    anti_copyright = options.get("anti_copyright", config.DEFAULT_ANTI_COPYRIGHT)
    blur_enabled = options.get("blur_background", config.DEFAULT_BLUR_BACKGROUND)
    banner_top = options.get("banner_top", config.DEFAULT_BANNER_TOP)
    banner_bottom = options.get("banner_bottom", config.DEFAULT_BANNER_BOTTOM)
    font_style = options.get("font_style")
    gpu_opts = _detect_gpu_accel()

    video_path = str(video_path)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if title:
        # Sanitize title for filename — preserve Cyrillic
        safe_title = re.sub(r'[^\w\s\-а-яА-ЯёЁ]', '', title).strip().replace(' ', '_')[:50]
        clip_name = safe_title
    else:
        stem = Path(video_path).stem
        safe_start = start_time.replace(":", "-")
        clip_name = f"{stem}_{safe_start}"

    raw_clip = str(config.TEMP_DIR / f"{clip_name}_raw.mp4")
    clip_with_audio = str(config.TEMP_DIR / f"{clip_name}_audio.mp4")
    srt_path = str(config.TEMP_DIR / f"{clip_name}.srt")
    vertical_clip = str(config.TEMP_DIR / f"{clip_name}_vert.mp4")
    final_output = str(config.OUTPUT_DIR / f"{clip_name}.mp4")

    # Initialize subtitled_clip before try so finally can reference it safely
    subtitled_clip = vertical_clip

    try:
        # Step 1: Cut the segment
        print(f"[1/5] Cutting {start_time} - {end_time}...")
        clip_video(video_path, start_time, end_time, raw_clip, gpu_opts=gpu_opts)

        # Step 2: Generate subtitles
        if subtitles_enabled:
            transcript_path = options.get("transcript_path")
            if transcript_path and os.path.exists(transcript_path):
                print("[2/5] Generating subtitles from pre-transcribed segments...")
                start_sec = _time_to_seconds(start_time)
                end_sec = _time_to_seconds(end_time)
                all_segments = load_segments_json(transcript_path)
                clip_segments = filter_segments_in_range(all_segments, start_sec, end_sec)
                if clip_segments:
                    generate_word_group_srt(clip_segments, srt_path)
                    print(f"      {len(clip_segments)} segments → word-group SRT")
                else:
                    print("      No speech in this clip — skipping subtitles")
                    subtitles_enabled = False
                # Use raw clip for face tracking (audio is already in raw_clip)
                clip_with_audio = raw_clip
            else:
                # Fallback: transcribe the clip
                print("[2/5] Transcribing clip audio...")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw_clip, "-c", "copy",
                     "-avoid_negative_ts", "make_zero", clip_with_audio],
                    capture_output=True, check=True, timeout=120
                )
                segments = transcribe(clip_with_audio)
                if segments:
                    generate_word_group_srt(segments, srt_path)
                    print(f"      {len(segments)} segments transcribed")
                else:
                    print("      No speech detected — skipping subtitles")
                    subtitles_enabled = False
        else:
            clip_with_audio = raw_clip

        # Step 3: Scale to fit content area (1080 × content_h, preserves aspect ratio)
        if face_tracking_enabled:
            print("[3/5] Scaling to Shorts format...")
            apply_vertical_crop(
                clip_with_audio if subtitles_enabled else raw_clip,
                vertical_clip,
                anti_copyright=anti_copyright,
                banner_top=banner_top,
                banner_bottom=banner_bottom,
            )
        else:
            print("[3/5] Scaling to Shorts format...")
            convert_to_vertical(
                clip_with_audio if subtitles_enabled else raw_clip,
                vertical_clip,
                anti_copyright=anti_copyright,
                banner_top=banner_top,
                banner_bottom=banner_bottom,
                gpu_opts=gpu_opts,
            )

        # Step 4: Embed subtitles (on content-area video, before banner padding)
        if subtitles_enabled and os.path.exists(srt_path):
            print("[4/5] Embedding subtitles...")
            subtitled_clip = str(config.TEMP_DIR / f"{clip_name}_subs.mp4")
            embed_subtitles(vertical_clip, srt_path, subtitled_clip,
                           font_style=font_style, banner_top=banner_top,
                           banner_bottom=banner_bottom, gpu_opts=gpu_opts)
        else:
            print("[4/5] Skipping subtitle embed...")

        # Step 5: Blurred background → full 9:16 output
        if blur_enabled:
            print("[5/5] Adding blurred background...")
            blur_background(subtitled_clip, final_output,
                          enabled=True, banner_top=banner_top, banner_bottom=banner_bottom,
                          gpu_opts=gpu_opts)
        else:
            print("[5/5] Padding to full frame (no blur)...")
            # If blur disabled, pad the content-area video to full 9:16
            pad_with_banners(subtitled_clip, final_output,
                           banner_top=banner_top, banner_bottom=banner_bottom,
                           gpu_opts=gpu_opts)

        print(f"Done! → {final_output}")
        return final_output

    except FFmpegError as e:
        print(f"FFmpeg error: {e}")
        return None
    except subprocess.TimeoutExpired:
        print("Processing timed out")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        # Cleanup temp files
        cleanup_files = [raw_clip, clip_with_audio, srt_path, vertical_clip]
        if subtitled_clip != vertical_clip:
            cleanup_files.append(subtitled_clip)
        for f in cleanup_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


def process_multiple(video_path, timestamps_list, options=None, titles=None, max_workers=2):
    """
    Process multiple clips from one video using parallel workers.

    Args:
        video_path: path to source video
        timestamps_list: list of (start_time, end_time) tuples
        options: dict of options (passed to process_clip)
        titles: optional list of scene titles (same length as timestamps_list)
        max_workers: max parallel workers (default 2, 1 = sequential)

    Returns:
        List of paths to output files.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(timestamps_list)
    if total == 0:
        return []

    if titles is None:
        titles = [""] * total

    if max_workers <= 1:
        # Sequential mode
        results = []
        for i, (start, end) in enumerate(timestamps_list):
            print(f"\n--- Clip {i+1}/{total}: {start} - {end} ---")
            result = process_clip(video_path, start, end, options, title=titles[i])
            results.append(result)
        done = sum(1 for r in results if r is not None)
        print(f"\nDone: {done}/{total} clips processed")
        return results

    # Parallel mode
    print(f"\nProcessing {total} clips with max_workers={max_workers}...")
    results = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, (start, end) in enumerate(timestamps_list):
            future = executor.submit(process_clip, video_path, start, end, options, title=titles[i])
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                status = "✅" if result else "❌"
                print(f"  Clip {idx+1}/{total} {status}")
            except Exception as e:
                print(f"  Clip {idx+1}/{total} ❌ failed: {e}")
                results[idx] = None

    done = sum(1 for r in results if r is not None)
    print(f"\nDone: {done}/{total} clips processed (parallel)")
    return results
