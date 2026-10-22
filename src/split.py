"""
Step 4 of the video chunking pipeline: split chunks that are too long.

Input, per video (from earlier steps):
    <work>/<video_id>/merged.json   Step 3 output, if it exists (its "chunks" list)
    <work>/<video_id>/shots.json    otherwise the raw shots from Step 2
    <work>/<video_id>/meta.json     tier, domain, subtitles, audio facts (decode.py)

What it does, per video:
  Every chunk longer than the maximum for the video's tier (short 30 s,
  medium 90 s, long 120 s) is cut into equal-ish pieces of about 80% of the
  maximum: a 300 s chunk with a 120 s maximum becomes 4 pieces of about 75 s.
  Each cut is placed near its ideal spot (+-25% of a piece), at the best
  point there, in order of preference:
    1. the longest pause between subtitle lines (at least 0.7 s);
    2. the longest silence in the audio;
    3. a plain break between two subtitle lines (a pause shorter than 0.7 s);
    4. exactly the ideal spot.
  Every cut also leaves enough room for the remaining pieces, so a split can
  never make a piece shorter than the minimum or longer than the maximum (see
  the note "Why splitting never breaks the limits" below).

Output, per video:
    <work>/<video_id>/split.json    the chunks after this step, plus this video's log
And for the whole run (open them in Excel):
    <work>/split_log.csv            one row per video: how many chunks were split, how
                                    many cuts were added and how, longest chunk before/after
    <work>/split_cuts.csv           one row per NEW cut, with its time as a video-player
                                    timestamp (e.g. 12:03.4), so you can check it by hand

Install (pip only):
    pip install av numpy

Usage:
    python split.py --work decoded --data data/vmme
    python split.py --work decoded --data data/vmme --videos S8w44Tl5atk 7fnVcXfPe5c
    python split.py --work decoded --data data/vmme --redo
--data is the dataset folder (with videos/ in it). It is only needed to read
the audio, for videos that have no usable subtitle pause.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np

from chunk_common import (clock, length_report, limits_for, read_json, speech_gaps,
                          speech_intervals, write_csv, write_json)


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

# A pause between subtitle lines must be at least this long to count as a
# "real" pause (choice 1). Shorter breaks are only used as choice 3.
MIN_SUBTITLE_PAUSE = 0.7

# A too-long chunk is cut into pieces of about this share of the maximum
# (0.8 x 120 s = 96 s for long videos). Aiming a bit below the maximum leaves
# each cut room to move to a nearby pause.
TARGET_SHARE = 0.8

# How far a cut may move from its ideal spot to reach a pause, as a share of
# one piece's length. 0.25 = pieces end up between 75% and 125% of equal.
SEARCH_SHARE = 0.25

# Audio silence (choice 2). The audio is measured in 50 ms steps.
AUDIO_STEP = 0.05            # seconds per loudness value
AUDIO_RATE = 16000           # resample audio to this many samples per second
SILENCE_MAX_DB = -40.0       # never call anything louder than this "silence"
SILENCE_ABOVE_QUIETEST = 10  # ...and it must be within 10 dB of the video's quietest 10%
MIN_SILENCE = 0.3            # a silence must last at least this long


# ----------------------------------------------------------------------------
# Why splitting never breaks the limits
# ----------------------------------------------------------------------------
#
# A chunk of length L > max is cut into n pieces, with n = ceil(L / (0.8 * max)),
# or n = ceil(L / max) if the first choice would make pieces shorter than min.
# Either way:
#   - L / n <= max, so n pieces are enough;
#   - L / n >= min, so n pieces are not too many (with ceil(L / max) pieces,
#     L / n > max / 2 >= min, as long as max >= 2 * min).
#
# The cuts are placed left to right. Say m pieces are still to be made from
# the R seconds after the last cut. The next cut must leave:
#   - this piece between min and max:            between last + min and last + max
#   - room for the other m-1 pieces to fit too:  between end - (m-1)*max and end - (m-1)*min
# These four limits always overlap (because m*min <= R <= m*max), and the
# ideal spot (last + R/m) is always inside them. So whatever pause we pick
# inside that window, every piece ends up between min and max.
#
# The only way this can fail is if max < 2 * min: then a chunk just over max
# cannot be cut into two pieces that are both >= min. The tier limits
# (5/30, 8/90, 10/120) are far from that, and main() refuses --min/--max
# values that break it.


# ----------------------------------------------------------------------------
# Reading the input chunks
# ----------------------------------------------------------------------------

def load_input_chunks(video_dir: Path) -> tuple:
    """
    Return (chunks, source). Each chunk is a dict with start, end (seconds),
    n_shots, and start_reason (why a chunk starts there).
    Uses Step 3's output when it exists, else the raw shots from Step 2.
    """
    merged_file = video_dir / "merged.json"
    if merged_file.exists():
        chunks = read_json(merged_file)["chunks"]
        return [{"start": c["start"], "end": c["end"], "n_shots": c.get("n_shots", 1),
                 "start_reason": c.get("start_reason", "shot")} for c in chunks], "step3 merged.json"

    shots = read_json(video_dir / "shots.json")["shots"]
    chunks = [{"start": s["start"], "end": s["end"], "n_shots": 1,
               "start_reason": "video start" if i == 0 else "shot"} for i, s in enumerate(shots)]
    return chunks, "step2 shots.json"


# ----------------------------------------------------------------------------
# Audio loudness (only used when subtitles don't help)
# ----------------------------------------------------------------------------

def load_loudness(video_path: Path, cache_file: Path) -> np.ndarray:
    """
    Loudness of the audio in decibels, one value per 50 ms.
    0 dB is the loudest possible sound, -60 dB is very quiet.
    Saved to audio_db.npy the first time, so the audio is decoded only once.
    """
    if cache_file.exists():
        return np.load(cache_file)

    import av    # imported here: most videos never need the audio

    samples = []
    with av.open(str(video_path)) as container:
        stream = container.streams.audio[0]
        # Turn any audio (stereo, 44.1 kHz, ...) into one 16 kHz channel of 16-bit numbers.
        resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_RATE)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                samples.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):           # flush what's left inside the resampler
            samples.append(out.to_ndarray().reshape(-1))

    audio = np.concatenate(samples).astype(np.float32) / 32768.0 if samples else np.zeros(0, np.float32)
    step = int(AUDIO_RATE * AUDIO_STEP)
    n = len(audio) // step
    blocks = audio[:n * step].reshape(n, step)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    loudness = 20 * np.log10(np.maximum(rms, 1e-6))   # -120 dB for digital silence
    np.save(cache_file, loudness.astype(np.float32))
    return loudness


def quiet_runs(loudness: np.ndarray) -> list:
    """
    Stretches of silence as (start_s, end_s). "Silence" adapts to the video
    (within 10 dB of its quietest moments) but is never louder than -40 dB,
    so music or crowd noise is not mistaken for a pause.
    """
    if len(loudness) == 0:
        return []
    limit = min(SILENCE_MAX_DB, float(np.percentile(loudness, 10)) + SILENCE_ABOVE_QUIETEST)
    quiet = loudness < limit
    runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j + 1 < len(quiet) and quiet[j + 1]:
                j += 1
            runs.append((i * AUDIO_STEP, (j + 1) * AUDIO_STEP))
            i = j + 1
        else:
            i += 1
    return runs


# ----------------------------------------------------------------------------
# Choosing where to split one chunk
# ----------------------------------------------------------------------------

def best_pause(pauses: list, lo: float, hi: float, ideal: float, min_length: float):
    """
    From a list of pauses (start, end), pick the longest one that reaches into
    the allowed window [lo, hi] and is at least min_length long. Ties go to the
    pause nearest the ideal spot. Returns (cut_time, pause_length) or None.
    The cut goes in the middle of the pause, moved inside the window if needed.
    """
    best = None
    for start, end in pauses:
        length = end - start
        if length < min_length or end < lo or start > hi:
            continue
        cut = min(max((start + end) / 2, lo, start), hi, end)   # middle of the pause, kept in the window
        key = (length, -abs(cut - ideal))
        if best is None or key > best[0]:
            best = (key, cut, length)
    return None if best is None else (best[1], best[2])


def choose_cut(lo: float, hi: float, ideal: float, gaps: list, get_quiet_runs) -> tuple:
    """
    Pick one cut inside the window [lo, hi], preferring pauses.
    Returns (cut_time, method, detail).
    """
    # 1. A real pause in the speech.
    found = best_pause(gaps, lo, hi, ideal, MIN_SUBTITLE_PAUSE)
    if found:
        return found[0], "subtitle pause", f"pause {found[1]:.1f} s"

    # 2. Silence in the audio.
    found = best_pause(get_quiet_runs(), lo, hi, ideal, MIN_SILENCE)
    if found:
        return found[0], "silence", f"silence {found[1]:.1f} s"

    # 3. Any break between two subtitle lines, even a very short one.
    found = best_pause(gaps, lo, hi, ideal, 0.0)
    if found:
        return found[0], "line break", f"pause {found[1]:.2f} s"

    # 4. Nothing to go on: cut at the ideal spot.
    return ideal, "even split", ""


def split_one_chunk(start: float, end: float, min_len: float, max_len: float, gaps: list, get_quiet_runs) -> list:
    """
    Cut [start, end] into the fewest pieces that are all <= max_len.
    Returns the list of cuts as (cut_time, method, detail), left to right.
    """
    length = end - start
    # Aim for pieces of about 80% of the maximum, so each cut has room to move
    # to a pause. Use fewer pieces only if that would make them too short.
    n_pieces = math.ceil(length / (TARGET_SHARE * max_len))
    if length / n_pieces < min_len:
        n_pieces = math.ceil(length / max_len - 1e-9)
    cuts, last = [], start
    for pieces_left in range(n_pieces, 1, -1):
        remaining = end - last
        ideal = last + remaining / pieces_left
        reach = SEARCH_SHARE * remaining / pieces_left
        # The window: near the ideal spot, AND leaving every piece within the limits.
        lo = max(ideal - reach, last + min_len, end - (pieces_left - 1) * max_len)
        hi = min(ideal + reach, last + max_len, end - (pieces_left - 1) * min_len)
        cut, method, detail = choose_cut(lo, hi, ideal, gaps, get_quiet_runs)
        cuts.append((cut, method, detail))
        last = cut
    return cuts


def split_chunks(chunks: list, min_len: float, max_len: float, gaps: list, get_quiet_runs) -> tuple:
    """
    Split every chunk longer than max_len.
    Returns (new_chunks, cuts), where cuts describes each new cut.
    """
    result, cut_rows = [], []
    for chunk in chunks:
        if chunk["end"] - chunk["start"] <= max_len:
            result.append(dict(chunk))
            continue

        cuts = split_one_chunk(chunk["start"], chunk["end"], min_len, max_len, gaps, get_quiet_runs)
        edges = [chunk["start"]] + [c[0] for c in cuts] + [chunk["end"]]
        for k in range(len(edges) - 1):
            piece = dict(chunk, start=edges[k], end=edges[k + 1])
            if k > 0:
                piece["start_reason"] = f"split: {cuts[k - 1][1]}"
            result.append(piece)
        for k, (cut, method, detail) in enumerate(cuts):
            cut_rows.append({
                "chunk_start": chunk["start"], "chunk_end": chunk["end"],
                "cut_s": round(cut, 3), "method": method, "detail": detail,
                "piece_before_s": round(edges[k + 1] - edges[k], 1),
                "piece_after_s": round(edges[k + 2] - edges[k + 1], 1),
            })
    return result, cut_rows


# ----------------------------------------------------------------------------
# One video, start to finish
# ----------------------------------------------------------------------------

def process_one_video(video_dir: Path, data_dir, min_override, max_override, redo: bool) -> tuple:
    """Returns (log_row, cut_rows) for the two CSV files."""
    video_id = video_dir.name
    out_file = video_dir / "split.json"

    if out_file.exists() and not redo:
        saved = read_json(out_file)
        return dict(saved["log"], status="already done"), saved["cuts"]

    started = time.time()
    try:
        meta = read_json(video_dir / "meta.json")
        chunks, source = load_input_chunks(video_dir)
        min_len, max_len = limits_for(meta, min_override, max_override)
        gaps = speech_gaps(speech_intervals(meta.get("subtitles", [])))

        # Audio is read only the first time a chunk needs it, then remembered.
        audio_cache = {}

        def get_quiet_runs():
            if "runs" not in audio_cache:
                audio_cache["runs"] = []
                probe = meta.get("probe", {})
                if probe.get("has_audio") and data_dir is not None:
                    video_path = Path(data_dir) / "videos" / probe["file"]
                    if video_path.exists():
                        print(f"  [audio] {video_id}: reading the audio to find silences...", flush=True)
                        audio_cache["runs"] = quiet_runs(load_loudness(video_path, video_dir / "audio_db.npy"))
            return audio_cache["runs"]

        new_chunks, cuts = split_chunks(chunks, min_len, max_len, gaps, get_quiet_runs)
        for i, c in enumerate(new_chunks):
            c["index"] = i
            c["duration"] = round(c["end"] - c["start"], 3)

        # Add readable timestamps so the CSV can be checked against a video player.
        for c in cuts:
            c["video_id"] = video_id
            c["chunk"] = f"{clock(c['chunk_start'])} - {clock(c['chunk_end'])}"
            c["cut_at"] = clock(c["cut_s"])

        before = length_report(chunks, min_len, max_len)
        after = length_report(new_chunks, min_len, max_len)
        methods = [c["method"] for c in cuts]
        log = {
            "video_id": video_id,
            "status": "ok",
            "domain": meta.get("domain"),
            "tier": meta.get("tier"),
            "input": source,
            "min_s": min_len,
            "max_s": max_len,
            "chunks_in": before["chunks"],
            "too_long_in": before["above_max"],
            "chunks_split": len({(c["chunk_start"], c["chunk_end"]) for c in cuts}),
            "cuts_added": len(cuts),
            "by_subtitle_pause": methods.count("subtitle pause"),
            "by_silence": methods.count("silence"),
            "by_line_break": methods.count("line break"),
            "by_even_split": methods.count("even split"),
            "chunks_out": after["chunks"],
            "longest_in_s": before["longest_s"],
            "longest_out_s": after["longest_s"],
            "shortest_new_piece_s": min((min(c["piece_before_s"], c["piece_after_s"]) for c in cuts), default=None),
            "below_min_in": before["below_min"],     # too-short chunks that came in (Step 3's job)
            "below_min_out": after["below_min"],     # should equal below_min_in: splitting adds none
            "above_max_out": after["above_max"],     # should be 0
            "seconds_taken": round(time.time() - started, 2),
        }
        write_json(out_file, {"video_id": video_id, "domain": meta.get("domain"), "tier": meta.get("tier"),
                              "min_s": min_len, "max_s": max_len, "log": log, "cuts": cuts,
                              "chunks": new_chunks})
        return log, cuts

    except Exception as error:   # one bad video must not stop the others
        return {"video_id": video_id, "status": f"error: {error}"}, []


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

LOG_COLUMNS = ["video_id", "status", "domain", "tier", "input", "min_s", "max_s",
               "chunks_in", "too_long_in", "chunks_split", "cuts_added",
               "by_subtitle_pause", "by_silence", "by_line_break", "by_even_split",
               "chunks_out", "longest_in_s", "longest_out_s", "shortest_new_piece_s",
               "below_min_in", "below_min_out", "above_max_out", "seconds_taken"]
CUT_COLUMNS = ["video_id", "chunk", "cut_at", "method", "detail", "piece_before_s", "piece_after_s",
               "chunk_start", "chunk_end", "cut_s"]


def main():
    parser = argparse.ArgumentParser(description="Step 4: split chunks that are too long.")
    parser.add_argument("--work", type=Path, default=Path("work"), help="the folder decode.py and shots.py wrote to")
    parser.add_argument("--data", type=Path, default=None,
                        help="dataset folder with videos/ (needed only to read audio for silences)")
    parser.add_argument("--videos", nargs="*", help="only these video ids (default: all)")
    parser.add_argument("--min", type=float, default=None, help="minimum chunk seconds for every video (default: by tier)")
    parser.add_argument("--max", type=float, default=None, help="maximum chunk seconds for every video (default: by tier)")
    parser.add_argument("--redo", action="store_true", help="recompute videos already done")
    args = parser.parse_args()
    if args.min is not None and args.max is not None and args.max < 2 * args.min:
        raise SystemExit("--max must be at least twice --min, or a long chunk can't be split safely")

    video_dirs = sorted(d for d in args.work.iterdir() if (d / "shots.json").exists())
    if args.videos:
        video_dirs = [d for d in video_dirs if d.name in set(args.videos)]
    if not video_dirs:
        raise SystemExit(f"No videos with shots.json found in {args.work}. Run shots.py first.")
    if args.data is None:
        print("Note: no --data given, so silences can't be used; videos without subtitle "
              "pauses will be split at line breaks or evenly.")

    logs, all_cuts = [], []
    run_started = time.time()
    for n, video_dir in enumerate(video_dirs, 1):
        log, cuts = process_one_video(video_dir, args.data, args.min, args.max, args.redo)
        logs.append(log)
        all_cuts.extend(cuts)
        if log["status"] in ("ok", "already done"):
            details = (f"{log['chunks_split']} chunks split, {log['cuts_added']} cuts added"
                       f"  |  {log['chunks_in']} -> {log['chunks_out']} chunks"
                       f"  |  longest {log['longest_in_s']} -> {log['longest_out_s']} s")
        else:
            details = ""
        print(f"  [done {n}/{len(video_dirs)}] {log['video_id']}: {log['status']}  {details}"
              f"  |  total elapsed {time.time() - run_started:.0f} s", flush=True)

    write_csv(args.work / "split_log.csv", logs, LOG_COLUMNS)
    all_cuts.sort(key=lambda c: (c["video_id"], c["cut_s"]))
    write_csv(args.work / "split_cuts.csv", all_cuts, CUT_COLUMNS)
    failed = [r for r in logs if r["status"].startswith("error")]
    print(f"Done. {len(logs) - len(failed)} ok, {len(failed)} failed. "
          f"{sum(r.get('cuts_added', 0) for r in logs)} cuts added in total.")
    print(f"Logs: {args.work / 'split_log.csv'} (per video) and {args.work / 'split_cuts.csv'} (every new cut)")


if __name__ == "__main__":
    main()
