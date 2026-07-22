"""
MovieShort AI — Standard detection (no LLM, random selection).
"""
import config
from analyzers.scene_analyzer import detect_and_transcribe


def find_best_clips_standard(video_path, max_duration=60, min_duration=15, num_clips=10):
    """
    Standard mode: NO LLM. PySceneDetect → filter by duration → random selection.

    Args:
        video_path: path to video file
        max_duration: max clip duration in seconds
        min_duration: min clip duration in seconds
        num_clips: number of clips to select (default 10)
    """
    scenes = detect_and_transcribe(video_path)
    if not scenes:
        return []
    scenes = [s for s in scenes if s["duration"] >= min_duration]
    if not scenes:
        return []
    import random
    scenes_copy = scenes.copy()
    random.shuffle(scenes_copy)
    selected = scenes_copy[:num_clips]
    for s in selected:
        s["score"] = random.randint(5, 8)
        s["title"] = ""
    return selected
