"""
Layer 1 of the test plan (see final-approach.md, section 5): sanity checks on the final chunks.

Input, per video:
    <work>/<video_id>/snapped.json  the final chunks (snap.py)
    <work>/<video_id>/meta.json     duration, subtitles, domain, tier (decode.py)

What it checks, per video:
  1. Coverage. The chunks start at 0, end at the end of the video, and follow
     each other with no gap and no overlap. No chunk has zero or negative length.
  2. Limits. No chunk is shorter than the tier's minimum or longer than its
     maximum. The one allowed exception: a video shorter than the minimum is a
     single chunk.
  3. Mid-sentence boundaries. For videos with subtitles, the share of
     boundaries that still fall inside a line. For comparison, the same share
     for plain fixed 30 s windows on the same video.
  4. Outliers. Chunk counts that look wrong for the video: a long video with
     fewer than 5 chunks, a short one with more than 15, or a chunk rate far
     from the other videos of the same domain and tier.

Each video gets a verdict:
    PASS   nothing to look at
    WARN   worth a look by eye (outliers, many mid-sentence boundaries), not a bug
    FAIL   a real problem: coverage broken or a limit broken. Fix before testing further.

Output (open them in Excel):
    <work>/check_report.csv     one row per video: numbers, verdict and the reasons
    <work>/check_summary.csv    one row per domain and tier: chunk counts and lengths
And a short summary printed at the end.

Install (pip only):
    nothing beyond what snap.py needs

Usage:
    python check.py --work decoded
    python check.py --work decoded --videos qd2ivr-5oEM S8w44Tl5atk
"""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from chunk_common import clock, limits_for, read_json, speech_intervals, write_csv


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

EDGE_TOLERANCE = 0.5     # the first chunk may start up to 0.5 s late, the last end up to 0.5 s early
JOIN_TOLERANCE = 0.01    # neighbouring chunks must meet within 0.01 s (no gap, no overlap)
LIMIT_TOLERANCE = 0.01   # rounding slack when comparing a length to min/max
LINE_TOLERANCE = 0.05    # a boundary this close to the start/end of a line counts as "between lines" (same as snap.py)
BASELINE_WINDOW = 30.0   # fixed-window baseline for the mid-sentence comparison

# Outlier rules (final-approach.md, section 5, Layer 1)
LONG_MIN_CHUNKS = 5            # a long video (30-60 min) with fewer chunks than this is suspicious
SHORT_MAX_CHUNKS = 15          # a short video (< 2 min) with more chunks than this is suspicious
RATE_FACTOR = 2.5              # chunks per minute this many times above/below the median of the same domain and tier
RATE_MIN_MINUTES = 1.0         # videos shorter than this are too short for the rate comparison
MID_SENTENCE_WARN = 0.30       # warn if more than 30% of boundaries still cut through a line


# ----------------------------------------------------------------------------
# The checks
# ----------------------------------------------------------------------------

def check_coverage(chunks: list, duration: float) -> list:
    """Problems with how the chunks cover the video. Empty list = fine."""
    problems = []
    if not chunks:
        return ["no chunks at all"]

    if chunks[0]["start"] > EDGE_TOLERANCE:
        problems.append(f"first chunk starts at {clock(chunks[0]['start'])}, not at 0")
    if duration and chunks[-1]["end"] < duration - EDGE_TOLERANCE:
        problems.append(f"last chunk ends at {clock(chunks[-1]['end'])}, video ends at {clock(duration)}")
    if duration and chunks[-1]["end"] > duration + EDGE_TOLERANCE:
        problems.append(f"last chunk ends at {clock(chunks[-1]['end'])}, after the video ends ({clock(duration)})")

    for left, right in zip(chunks, chunks[1:]):
        gap = right["start"] - left["end"]
        if gap > JOIN_TOLERANCE:
            problems.append(f"gap of {gap:.2f} s at {clock(left['end'])}")
        elif gap < -JOIN_TOLERANCE:
            problems.append(f"overlap of {-gap:.2f} s at {clock(right['start'])}")

    for c in chunks:
        if c["end"] - c["start"] <= 0:
            problems.append(f"zero or negative length chunk at {clock(c['start'])}")
    return problems


def check_limits(chunks: list, duration: float, min_len: float, max_len: float) -> list:
    """Chunks shorter than min or longer than max. Empty list = fine."""
    # A video shorter than the minimum can only be one short chunk; that's fine.
    if len(chunks) == 1 and duration and duration < min_len:
        return []
    problems = []
    for c in chunks:
        length = c["end"] - c["start"]
        if length < min_len - LIMIT_TOLERANCE:
            problems.append(f"{length:.1f} s chunk at {clock(c['start'])} is under the {min_len:g} s minimum")
        elif length > max_len + LIMIT_TOLERANCE:
            problems.append(f"{length:.1f} s chunk at {clock(c['start'])} is over the {max_len:g} s maximum")
    return problems


def share_inside_speech(boundaries: list, intervals: list):
    """Share of boundaries (0-1) that fall inside a line of speech. None if there are no boundaries."""
    if not boundaries:
        return None
    inside = sum(any(s + LINE_TOLERANCE < t < e - LINE_TOLERANCE for s, e in intervals) for t in boundaries)
    return inside / len(boundaries)


def fixed_window_boundaries(duration: float) -> list:
    """The boundaries plain 30 s windows would give: 30, 60, 90, ... up to the end."""
    boundaries, t = [], BASELINE_WINDOW
    while duration and t < duration - 1e-6:
        boundaries.append(t)
        t += BASELINE_WINDOW
    return boundaries


# ----------------------------------------------------------------------------
# One video
# ----------------------------------------------------------------------------

def check_one_video(video_dir: Path) -> dict:
    video_id = video_dir.name
    try:
        meta = read_json(video_dir / "meta.json")
        chunks = sorted(read_json(video_dir / "snapped.json")["chunks"], key=lambda c: c["start"])
    except Exception as error:   # a missing or broken file is reported, not fatal
        return {"video_id": video_id, "verdict": "FAIL", "reasons": f"could not read files: {error}"}

    duration = (meta.get("probe") or {}).get("duration_s") or (chunks[-1]["end"] if chunks else 0)
    min_len, max_len = limits_for(meta)
    lengths = [c["end"] - c["start"] for c in chunks]
    boundaries = [c["start"] for c in chunks[1:]]

    subtitles = meta.get("subtitles", [])
    intervals = speech_intervals(subtitles) if subtitles else []
    mid = share_inside_speech(boundaries, intervals) if subtitles else None
    mid_fixed = share_inside_speech(fixed_window_boundaries(duration), intervals) if subtitles else None

    coverage = check_coverage(chunks, duration)
    limits = check_limits(chunks, duration, min_len, max_len)

    return {
        "video_id": video_id,
        "domain": meta.get("domain"),
        "tier": meta.get("tier"),
        "duration_min": round(duration / 60, 2) if duration else None,
        "min_s": min_len,
        "max_s": max_len,
        "chunks": len(chunks),
        "chunks_per_min": round(len(chunks) / (duration / 60), 2) if duration else None,
        "median_chunk_s": round(statistics.median(lengths), 1) if lengths else None,
        "shortest_chunk_s": round(min(lengths), 1) if lengths else None,
        "longest_chunk_s": round(max(lengths), 1) if lengths else None,
        "subtitle_lines": len(subtitles),
        "mid_sentence_share": round(mid, 3) if mid is not None else None,
        "mid_sentence_share_fixed_30s": round(mid_fixed, 3) if mid_fixed is not None else None,
        "coverage_problems": len(coverage),
        "limit_problems": len(limits),
        # filled in later, once all videos are known
        "verdict": None,
        "reasons": None,
        "_fails": coverage + limits,
    }


# ----------------------------------------------------------------------------
# Verdicts (need all videos, for the per-domain comparison)
# ----------------------------------------------------------------------------

def add_verdicts(rows: list):
    rates = defaultdict(list)
    for r in rows:
        if r.get("chunks_per_min"):
            rates[(r["domain"], r["tier"])].append(r["chunks_per_min"])
    group_median = {g: statistics.median(v) for g, v in rates.items()}

    for r in rows:
        if r["verdict"] == "FAIL":          # already failed on reading the files
            continue
        fails = r.pop("_fails")
        warns = []

        if r["tier"] == "long" and r["chunks"] < LONG_MIN_CHUNKS:
            warns.append(f"long video with only {r['chunks']} chunks")
        if r["tier"] == "short" and r["chunks"] > SHORT_MAX_CHUNKS:
            warns.append(f"short video with {r['chunks']} chunks")
        median = group_median.get((r["domain"], r["tier"]))
        rate = r["chunks_per_min"]
        if median and rate and (r["duration_min"] or 0) >= RATE_MIN_MINUTES:
            group = f"{r['domain']} {r['tier']}"
            if rate > median * RATE_FACTOR:
                warns.append(f"{rate} chunks/min, far above the {group} median of {median:.2f}")
            elif rate < median / RATE_FACTOR:
                warns.append(f"{rate} chunks/min, far below the {group} median of {median:.2f}")
        if r["mid_sentence_share"] is not None and r["mid_sentence_share"] > MID_SENTENCE_WARN:
            warns.append(f"{r['mid_sentence_share']:.0%} of boundaries cut through a line")

        r["verdict"] = "FAIL" if fails else ("WARN" if warns else "PASS")
        r["reasons"] = "; ".join(fails[:5] + (["..."] if len(fails) > 5 else []) + warns)


def summary_rows(rows: list) -> list:
    """One row per (domain, tier), plus a total row."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("chunks") is not None:
            groups[(r["domain"], r["tier"])].append(r)
            groups[("ALL", "ALL")].append(r)

    out = []
    for (domain, tier), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] == "ALL", str(kv[0]))):
        minutes = sum(r["duration_min"] or 0 for r in rs)
        chunks = sum(r["chunks"] for r in rs)
        mids = [r["mid_sentence_share"] for r in rs if r["mid_sentence_share"] is not None]
        fixed = [r["mid_sentence_share_fixed_30s"] for r in rs if r["mid_sentence_share_fixed_30s"] is not None]
        medians = [r["median_chunk_s"] for r in rs if r["median_chunk_s"] is not None]
        out.append({
            "domain": domain,
            "tier": tier,
            "videos": len(rs),
            "minutes": round(minutes, 1),
            "chunks": chunks,
            "chunks_per_hour": round(chunks / minutes * 60, 1) if minutes else None,
            "median_chunk_s": round(statistics.median(medians), 1) if medians else None,
            "mid_sentence_share": round(statistics.mean(mids), 3) if mids else None,
            "mid_sentence_share_fixed_30s": round(statistics.mean(fixed), 3) if fixed else None,
            "pass": sum(r["verdict"] == "PASS" for r in rs),
            "warn": sum(r["verdict"] == "WARN" for r in rs),
            "fail": sum(r["verdict"] == "FAIL" for r in rs),
        })
    return out


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

REPORT_COLUMNS = ["video_id", "verdict", "reasons", "domain", "tier", "duration_min", "min_s", "max_s",
                  "chunks", "chunks_per_min", "median_chunk_s", "shortest_chunk_s", "longest_chunk_s",
                  "subtitle_lines", "mid_sentence_share", "mid_sentence_share_fixed_30s",
                  "coverage_problems", "limit_problems"]
SUMMARY_COLUMNS = ["domain", "tier", "videos", "minutes", "chunks", "chunks_per_hour", "median_chunk_s",
                   "mid_sentence_share", "mid_sentence_share_fixed_30s", "pass", "warn", "fail"]


def main():
    parser = argparse.ArgumentParser(description="Layer 1 sanity checks on the final chunks (snapped.json).")
    parser.add_argument("--work", type=Path, default=Path("work"), help="the folder the earlier steps wrote to")
    parser.add_argument("--videos", nargs="*", help="only these video ids (default: all)")
    args = parser.parse_args()

    video_dirs = sorted(d for d in args.work.iterdir() if (d / "snapped.json").exists())
    if args.videos:
        video_dirs = [d for d in video_dirs if d.name in set(args.videos)]
    if not video_dirs:
        raise SystemExit(f"No videos with snapped.json found in {args.work}. Run snap.py first.")

    print(f"Checking {len(video_dirs)} video(s)...", flush=True)
    rows = [check_one_video(d) for d in video_dirs]
    add_verdicts(rows)
    summary = summary_rows(rows)

    write_csv(args.work / "check_report.csv", rows, REPORT_COLUMNS)
    write_csv(args.work / "check_summary.csv", summary, SUMMARY_COLUMNS)

    # Short printed summary: totals first, then every video that needs a look.
    total = summary[-1] if summary else {}
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("PASS", "WARN", "FAIL")}
    print()
    print(f"  {counts['PASS']} PASS  |  {counts['WARN']} WARN  |  {counts['FAIL']} FAIL")
    if total:
        print(f"  {total['chunks']} chunks over {total['minutes']} min of video "
              f"= {total['chunks_per_hour']} chunks per hour, median chunk {total['median_chunk_s']} s")
        if total["mid_sentence_share"] is not None:
            print(f"  Boundaries cutting through a subtitle line: {total['mid_sentence_share']:.0%} "
                  f"(fixed 30 s windows would give {total['mid_sentence_share_fixed_30s']:.0%})")
    for verdict in ("FAIL", "WARN"):
        for r in rows:
            if r["verdict"] == verdict:
                print(f"  [{verdict}] {r['video_id']}  ({r.get('domain')}, {r.get('tier')}): {r['reasons']}")
    print()
    print(f"Reports: {args.work / 'check_report.csv'} (per video) and "
          f"{args.work / 'check_summary.csv'} (per domain and tier)")
    if counts["FAIL"]:
        print("Fix the FAIL videos before going on to Layer 2 and 3.")


if __name__ == "__main__":
    main()
