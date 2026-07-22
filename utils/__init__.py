"""
MovieShort AI — Shared utilities.
"""
from pathlib import Path


def get_video_basename(video_path: str) -> str:
    """Extract the base filename without extension using Path.stem.
    
    For 'movie.2024.mp4' returns 'movie.2024', for 'movie.mp4' returns 'movie'.
    Consistent across all modules — fixes hash mismatch for multi-dot filenames.
    """
    return Path(video_path).stem


def fmt_duration(seconds: float) -> str:
    """Format seconds to human-readable string (e.g. '2m 34s', '1h 02m')."""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"
