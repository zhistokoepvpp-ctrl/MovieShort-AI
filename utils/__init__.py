"""
MovieShort AI — Shared utilities.
"""


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
