"""
Small helpers shared by split.py (Step 4) and snap.py (Step 5).

Kept in one place so both steps use exactly the same length limits and the
same idea of "where is someone speaking".
"""

import csv
import json
import re
from pathlib import Path


# ----------------------------------------------------------------------------
# Chunk length limits per duration tier 
# ----------------------------------------------------------------------------

# tier -> (min seconds, max seconds)
TIER_LIMITS = {
    "short": (5.0, 30.0),
    "medium": (8.0, 90.0),
    "long": (10.0, 120.0),
}
# A video with no tier label (a real customer corpus) gets the medium limits.
FALLBACK_TIER = "medium"


def limits_for(meta: dict, min_override=None, max_override=None) -> tuple:
    """Return (min_len, max_len) for this video. Command-line values win."""
    min_len, max_len = TIER_LIMITS.get(meta.get("tier"), TIER_LIMITS[FALLBACK_TIER])
    if min_override is not None:
        min_len = min_override
    if max_override is not None:
        max_len = max_override
    return min_len, max_len


# ----------------------------------------------------------------------------
# Speech from subtitles
# ----------------------------------------------------------------------------

def is_rolling(subtitles: list) -> bool:
    """
    True for YouTube-style "rolling" auto-captions, where each line stays on
    screen until the line after next appears, so almost every line overlaps
    the next one. Checked on the whole file: more than half the lines overlap.
    """
    lines = sorted((float(s["start"]), float(s["end"])) for s in subtitles if s["end"] > s["start"])
    if len(lines) < 2:
        return False
    overlaps = sum(lines[i + 1][0] < lines[i][1] for i in range(len(lines) - 1))
    return overlaps > (len(lines) - 1) / 2


def speech_intervals(subtitles: list) -> list:
    """
    Turn subtitle lines into a sorted list of (start, end) "someone is talking"
    intervals that never overlap.

    Rolling auto-captions: a line's words are spoken between its own start and
    the start of the next line; after that it is only still on screen. So each
    line is cut back to end where the next one starts. The lines then touch,
    and every line break becomes a place where a boundary may go.

    Other subtitles: lines that overlap are joined into one interval, because
    the moment between two overlapping lines is NOT a pause. Lines that only
    touch (one ends exactly when the next starts) are kept apart: the moment
    between them is a line break, which is a fair place to cut.
    """
    lines = sorted((float(s["start"]), float(s["end"])) for s in subtitles if s["end"] > s["start"])
    if is_rolling(subtitles):
        lines = [(start, min(end, lines[i + 1][0]) if i + 1 < len(lines) else end)
                 for i, (start, end) in enumerate(lines)]
        lines = [(start, end) for start, end in lines if end > start]
    joined = []
    for start, end in lines:
        if joined and start < joined[-1][1]:            # overlaps the previous one
            joined[-1] = (joined[-1][0], max(joined[-1][1], end))
        else:
            joined.append((start, end))
    return joined


def speech_gaps(intervals: list) -> list:
    """The pauses between speech intervals, as (gap_start, gap_end). Can be zero length."""
    return [(intervals[i][1], intervals[i + 1][0]) for i in range(len(intervals) - 1)]


def line_at(subtitles: list, t: float):
    """The text of the subtitle line being spoken at time t, or None."""
    for line in subtitles:
        if line["start"] < t < line["end"]:
            return re.sub(r"<[^>]+>", "", line["text"]).strip()
    return None


# ----------------------------------------------------------------------------
# Small things
# ----------------------------------------------------------------------------

def clock(seconds: float) -> str:
    """
    Seconds as a video-player time, so you can jump to it by hand:
    75.4 -> "1:15.4",   3725.0 -> "1:02:05.0"
    """
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes)}:{secs:04.1f}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list, columns: list):
    # utf-8-sig so Excel shows non-English subtitle text correctly.
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})


def length_report(chunks: list, min_len: float, max_len: float) -> dict:
    """Counts used by both logs to show the limits really hold."""
    lengths = [c["end"] - c["start"] for c in chunks]
    return {
        "chunks": len(chunks),
        "shortest_s": round(min(lengths), 1) if lengths else None,
        "longest_s": round(max(lengths), 1) if lengths else None,
        "below_min": sum(1 for x in lengths if x < min_len - 1e-6),
        "above_max": sum(1 for x in lengths if x > max_len + 1e-6),
    }
