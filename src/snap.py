"""
Step 5 of the video chunking pipeline (see final-approach.md): snap boundaries to sentences.

Input, per video:
    <work>/<video_id>/split.json    the chunks after Step 4 (split.py)
    <work>/<video_id>/meta.json     subtitles, tier (decode.py)

What it does, per video:
  Looks at every boundary between two chunks. If it falls while someone is
  speaking (inside a subtitle line), it moves it to the nearest pause between
  lines, but only if that pause is within 1.5 s. Otherwise the boundary stays
  where the picture changed: a clear visual change wins over a sentence.

  A move is also refused if it would break the length limits, i.e. make
  either neighbouring chunk shorter than the minimum or longer than the
  maximum (see "Why snapping never breaks the limits" below). The next-nearest
  pause is tried instead, and if none works the boundary stays.

  Boundaries are never removed or added here: the number of chunks in and out
  is always the same.

Output, per video:
    <work>/<video_id>/snapped.json  the chunks after this step, plus this video's log
And for the whole run (open them in Excel):
    <work>/snap_log.csv             one row per video: boundaries checked / moved / kept
    <work>/snap_moves.csv           one row per boundary that fell inside speech: where it
                                    was, where it went (video-player timestamps), the words
                                    it was cutting through, and why it moved or stayed

Install (pip only):
    pip install numpy

Usage:
    python snap.py --work decoded
    python snap.py --work decoded --videos YtAL8y2lACs
    python snap.py --work decoded --redo
"""

import argparse
import time
from pathlib import Path

from chunk_common import (clock, length_report, limits_for, line_at, read_json, speech_gaps,
                          speech_intervals, write_csv, write_json)


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

MAX_SHIFT = 1.5      # never move a boundary further than this (seconds)
INSET = 0.2          # land this far inside a pause, not on the last syllable (if the pause is long enough)
EDGE_TOLERANCE = 0.05  # a boundary this close to the start/end of a line already counts as "between lines"


# ----------------------------------------------------------------------------
# Why snapping never breaks the limits
# ----------------------------------------------------------------------------
#
# Moving a boundary by up to 1.5 s makes one neighbour longer and the other
# shorter. Without a check, that could:
#   - push a 29.5 s chunk to 31 s (over the short-tier max of 30 s), or
#   - shrink a 6 s chunk to 4.5 s (under the short-tier min of 5 s), or
#   - make two boundaries meet at the same pause, leaving a 0 s chunk.
# So a move is accepted only if, for BOTH neighbours, the new length is
#   >= the minimum (or no shorter than it already was, if it was already too short), and
#   <= the maximum (or no longer than it already was, if it was already too long).
# "Never make it worse" also covers the 0 s case, so boundaries can never
# cross or merge.


def allowed(new_length: float, old_length: float, min_len: float, max_len: float) -> bool:
    lowest = min(min_len, old_length)
    highest = max(max_len, old_length)
    return lowest - 1e-6 <= new_length <= highest + 1e-6


def inside_speech(t: float, intervals: list) -> bool:
    """True if t falls clearly inside a stretch of speech."""
    return any(start + EDGE_TOLERANCE < t < end - EDGE_TOLERANCE for start, end in intervals)


def snap_targets(t: float, gaps: list) -> list:
    """
    All places within MAX_SHIFT of t where a boundary could go, nearest first.
    For each pause, the target is the point in the pause nearest to t, moved
    INSET seconds into the pause when the pause is long enough for that.
    """
    targets = []
    for start, end in gaps:
        if end < t - MAX_SHIFT or start > t + MAX_SHIFT:
            continue
        inset = min(INSET, (end - start) / 2)
        target = min(max(t, start + inset), end - inset)
        if abs(target - t) <= MAX_SHIFT:
            targets.append(target)
    return sorted(set(targets), key=lambda x: abs(x - t))


def snap_boundaries(chunks: list, subtitles: list, min_len: float, max_len: float) -> tuple:
    """
    Walk the boundaries left to right and snap the ones inside speech.
    Returns (new_chunks, rows) where rows describes each boundary that was inside speech.
    """
    intervals = speech_intervals(subtitles)
    gaps = speech_gaps(intervals)
    chunks = [dict(c) for c in chunks]
    rows = []

    for i in range(1, len(chunks)):
        left, right = chunks[i - 1], chunks[i]
        t = right["start"]                       # the boundary between left and right
        if not inside_speech(t, intervals):
            continue

        row = {"boundary_reason": right.get("start_reason", ""), "was_s": t, "was_at": clock(t),
               "cutting_through": line_at(subtitles, t) or ""}
        old_left, old_right = t - left["start"], right["end"] - t
        targets = snap_targets(t, gaps)

        moved_to = None
        for target in targets:
            new_left, new_right = target - left["start"], right["end"] - target
            if (allowed(new_left, old_left, min_len, max_len)
                    and allowed(new_right, old_right, min_len, max_len)):
                moved_to = target
                break

        if moved_to is not None:
            left["end"] = right["start"] = moved_to
            right["snapped_from"] = t
            row.update(outcome="snapped", now_s=round(moved_to, 3), now_at=clock(moved_to),
                       shift_s=round(moved_to - t, 2))
        elif targets:
            row.update(outcome="kept: move would break min/max", now_s=t, now_at=clock(t), shift_s=0.0)
        else:
            row.update(outcome=f"kept: no pause within {MAX_SHIFT} s", now_s=t, now_at=clock(t), shift_s=0.0)
        rows.append(row)

    for i, c in enumerate(chunks):
        c["index"] = i
        c["duration"] = round(c["end"] - c["start"], 3)
    return chunks, rows


# ----------------------------------------------------------------------------
# One video, start to finish
# ----------------------------------------------------------------------------

def process_one_video(video_dir: Path, min_override, max_override, redo: bool) -> tuple:
    """Returns (log_row, move_rows) for the two CSV files."""
    video_id = video_dir.name
    out_file = video_dir / "snapped.json"

    if out_file.exists() and not redo:
        saved = read_json(out_file)
        return dict(saved["log"], status="already done"), saved["moves"]

    started = time.time()
    try:
        meta = read_json(video_dir / "meta.json")
        chunks = read_json(video_dir / "split.json")["chunks"]
        subtitles = meta.get("subtitles", [])
        min_len, max_len = limits_for(meta, min_override, max_override)

        new_chunks, moves = snap_boundaries(chunks, subtitles, min_len, max_len)
        for m in moves:
            m["video_id"] = video_id

        before = length_report(chunks, min_len, max_len)
        after = length_report(new_chunks, min_len, max_len)
        shifts = [abs(m["shift_s"]) for m in moves if m["outcome"] == "snapped"]
        outcomes = [m["outcome"] for m in moves]
        log = {
            "video_id": video_id,
            "status": "ok",
            "domain": meta.get("domain"),
            "tier": meta.get("tier"),
            "subtitle_lines": len(subtitles),
            "min_s": min_len,
            "max_s": max_len,
            "boundaries": max(0, len(chunks) - 1),
            "inside_speech": len(moves),
            "snapped": len(shifts),
            "kept_no_pause_near": sum(o.startswith("kept: no pause") for o in outcomes),
            "kept_would_break_limits": sum(o.startswith("kept: move would") for o in outcomes),
            "average_shift_s": round(sum(shifts) / len(shifts), 2) if shifts else None,
            "largest_shift_s": round(max(shifts), 2) if shifts else None,
            "chunks_in": before["chunks"],
            "chunks_out": after["chunks"],          # always equal to chunks_in
            "below_min_in": before["below_min"],
            "below_min_out": after["below_min"],    # never more than below_min_in
            "above_max_in": before["above_max"],
            "above_max_out": after["above_max"],    # never more than above_max_in
            "seconds_taken": round(time.time() - started, 2),
        }
        write_json(out_file, {"video_id": video_id, "domain": meta.get("domain"), "tier": meta.get("tier"),
                              "min_s": min_len, "max_s": max_len, "log": log, "moves": moves,
                              "chunks": new_chunks})
        return log, moves

    except Exception as error:   # one bad video must not stop the others
        return {"video_id": video_id, "status": f"error: {error}"}, []


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

LOG_COLUMNS = ["video_id", "status", "domain", "tier", "subtitle_lines", "min_s", "max_s",
               "boundaries", "inside_speech", "snapped", "kept_no_pause_near", "kept_would_break_limits",
               "average_shift_s", "largest_shift_s", "chunks_in", "chunks_out",
               "below_min_in", "below_min_out", "above_max_in", "above_max_out", "seconds_taken"]
MOVE_COLUMNS = ["video_id", "was_at", "now_at", "shift_s", "outcome", "boundary_reason",
                "cutting_through", "was_s", "now_s"]


def main():
    parser = argparse.ArgumentParser(description="Step 5: snap chunk boundaries to pauses in speech.")
    parser.add_argument("--work", type=Path, default=Path("work"), help="the folder the earlier steps wrote to")
    parser.add_argument("--videos", nargs="*", help="only these video ids (default: all)")
    parser.add_argument("--min", type=float, default=None, help="minimum chunk seconds (default: by tier; use the same value as split.py)")
    parser.add_argument("--max", type=float, default=None, help="maximum chunk seconds (default: by tier; use the same value as split.py)")
    parser.add_argument("--redo", action="store_true", help="recompute videos already done")
    args = parser.parse_args()

    video_dirs = sorted(d for d in args.work.iterdir() if (d / "split.json").exists())
    if args.videos:
        video_dirs = [d for d in video_dirs if d.name in set(args.videos)]
    if not video_dirs:
        raise SystemExit(f"No videos with split.json found in {args.work}. Run split.py first.")

    logs, all_moves = [], []
    run_started = time.time()
    for n, video_dir in enumerate(video_dirs, 1):
        log, moves = process_one_video(video_dir, args.min, args.max, args.redo)
        logs.append(log)
        all_moves.extend(moves)
        if log["status"] in ("ok", "already done"):
            if not log["subtitle_lines"]:
                details = "no subtitles, nothing to snap"
            else:
                details = (f"{log['inside_speech']} of {log['boundaries']} boundaries inside speech"
                           f"  |  {log['snapped']} snapped, {log['kept_no_pause_near']} no pause near,"
                           f" {log['kept_would_break_limits']} kept for limits")
        else:
            details = ""
        print(f"  [done {n}/{len(video_dirs)}] {log['video_id']}: {log['status']}  {details}"
              f"  |  total elapsed {time.time() - run_started:.0f} s", flush=True)

    write_csv(args.work / "snap_log.csv", logs, LOG_COLUMNS)
    write_csv(args.work / "snap_moves.csv", all_moves, MOVE_COLUMNS)
    failed = [r for r in logs if r["status"].startswith("error")]
    print(f"Done. {len(logs) - len(failed)} ok, {len(failed)} failed. "
          f"{sum(r.get('snapped', 0) for r in logs)} boundaries snapped in total.")
    print(f"Logs: {args.work / 'snap_log.csv'} (per video) and {args.work / 'snap_moves.csv'} (every boundary inside speech)")


if __name__ == "__main__":
    main()
