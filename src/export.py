"""
Step 8 of the video chunking pipeline (see final-approach.md): write the final output.

Input, per video:
    <work>/<video_id>/snapped.json  the final chunks (snap.py)
    <work>/<video_id>/shots.json    the raw shots (shots.py), to count shots per chunk
    <work>/<video_id>/meta.json     duration, subtitles, domain, tier (decode.py)

What it does:
  Keeps every chunk boundary exactly as it is. For each chunk it adds:
    - a padded window, 1 s wider on each side (never before 0 or past the end
      of the video), for the indexer to use as context
    - the number of camera shots inside it
    - the subtitle lines spoken inside it
    - why the chunk starts where it does: "video start", "shot" (a camera cut
      from TransNet V2) or "split" (Step 4 cut a chunk that was too long), and
      whether Step 5 then moved that boundary to a pause in speech ("snapped")

Output (for the whole corpus):
    <work>/chunks.jsonl    one JSON record per chunk: the hand-off to the indexer
    <work>/chunks.csv      the same records, to open in Excel

Install (pip only):
    nothing beyond what snap.py needs

Usage:
    python export.py --work decoded
    python export.py --work decoded --padding 2
"""

import argparse
import json
import re
from pathlib import Path

from chunk_common import clock, read_json, write_csv


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

PADDING = 1.0     # seconds of context added on each side (final-approach.md, Step 8)


# ----------------------------------------------------------------------------
# One chunk
# ----------------------------------------------------------------------------

def clean(text: str) -> str:
    """Subtitle text without <font ...> tags and extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def boundary_reason(chunk: dict) -> str:
    """Which rule made the cut at the start of this chunk: video start / shot / split."""
    reason = chunk.get("start_reason") or "shot"
    if reason == "video start":
        return "video start"
    if reason.startswith("split"):
        return "split"
    return "shot"


def count_shots(start: float, end: float, shots: list) -> int:
    """How many raw shots are inside [start, end). At least 1.

    A shot counts if at least 0.5 s of it (or half of it, if it is shorter than
    1 s) is inside, so a boundary snapped a little way into the next shot does
    not add that shot.
    """
    count = 0
    for s in shots:
        overlap = min(end, s["end"]) - max(start, s["start"])
        if overlap >= min(0.5, (s["end"] - s["start"]) / 2) and overlap > 0:
            count += 1
    return max(1, count)


def subtitle_text(start: float, end: float, subtitles: list) -> str:
    """The subtitle lines that start inside [start, end), joined.

    Using the start time means each line goes to exactly one chunk, even for
    rolling auto-captions where every line overlaps the next one.
    """
    return " ".join(clean(s["text"]) for s in subtitles if start <= s["start"] < end and s.get("text"))


# ----------------------------------------------------------------------------
# One video
# ----------------------------------------------------------------------------

def export_one_video(video_dir: Path, padding: float) -> list:
    video_id = video_dir.name
    meta = read_json(video_dir / "meta.json")
    chunks = sorted(read_json(video_dir / "snapped.json")["chunks"], key=lambda c: c["start"])
    shots = read_json(video_dir / "shots.json")["shots"]
    subtitles = meta.get("subtitles", [])
    duration = (meta.get("probe") or {}).get("duration_s") or (chunks[-1]["end"] if chunks else 0.0)

    records = []
    for i, c in enumerate(chunks):
        start, end = float(c["start"]), float(c["end"])
        records.append({
            "video_id": video_id,
            "chunk_id": f"{video_id}_{i:04d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "padded_start": round(max(0.0, start - padding), 3),
            "padded_end": round(min(duration, end + padding), 3),
            "start_at": clock(start),
            "end_at": clock(end),
            "n_shots": count_shots(start, end, shots),
            "subtitle_text": subtitle_text(start, end, subtitles),
            "boundary_reason": boundary_reason(c),
            "boundary_detail": c.get("start_reason") or "shot",
            "snapped": "snapped_from" in c,
            "snapped_from": round(c["snapped_from"], 3) if "snapped_from" in c else None,
            "domain": meta.get("domain"),
            "tier": meta.get("tier"),
        })
    return records


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

COLUMNS = ["video_id", "chunk_id", "start", "end", "duration", "padded_start", "padded_end",
           "start_at", "end_at", "n_shots", "boundary_reason", "boundary_detail", "snapped", "snapped_from",
           "domain", "tier", "subtitle_text"]


def main():
    parser = argparse.ArgumentParser(description="Step 8: write the final chunks for the indexer.")
    parser.add_argument("--work", type=Path, default=Path("work"), help="the folder the earlier steps wrote to")
    parser.add_argument("--videos", nargs="*", help="only these video ids (default: all)")
    parser.add_argument("--padding", type=float, default=PADDING, help=f"context seconds per side (default {PADDING:g})")
    args = parser.parse_args()

    video_dirs = sorted(d for d in args.work.iterdir() if (d / "snapped.json").exists())
    if args.videos:
        video_dirs = [d for d in video_dirs if d.name in set(args.videos)]
    if not video_dirs:
        raise SystemExit(f"No videos with snapped.json found in {args.work}. Run snap.py first.")

    all_records, failed = [], []
    for video_dir in video_dirs:
        try:
            all_records.extend(export_one_video(video_dir, args.padding))
        except Exception as error:   # one bad video must not stop the others
            failed.append(video_dir.name)
            print(f"  [error] {video_dir.name}: {error}", flush=True)

    with (args.work / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_csv(args.work / "chunks.csv", all_records, COLUMNS)

    reasons = {}
    for r in all_records:
        reasons[r["boundary_reason"]] = reasons.get(r["boundary_reason"], 0) + 1
    snapped = sum(r["snapped"] for r in all_records)
    print(f"Done. {len(all_records)} chunks from {len(video_dirs) - len(failed)} video(s), {len(failed)} failed.")
    print("  Chunk starts by reason: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
          + f"  |  {snapped} boundaries snapped to a pause")
    print(f"Output: {args.work / 'chunks.jsonl'} (for the indexer) and {args.work / 'chunks.csv'} (for Excel)")


if __name__ == "__main__":
    main()
