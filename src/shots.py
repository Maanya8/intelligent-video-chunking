"""
Step 2 of the video chunking pipeline (see final-approach.md): find the camera cuts.

Input:  the output of decode.py (Steps 0-1), one folder per video:
            <work>/<video_id>/frames_48x27.npy   every frame, 48x27 RGB
            <work>/<video_id>/timestamps.npy     time in seconds of each frame
            <work>/<video_id>/meta.json          domain, tier, probe facts

What it does, per video:
  1. Feed the tiny frames to TransNet V2, 100 frames at a time.
  2. Get a score from 0 to 1 for every frame: "is there a cut here?"
  3. Frames above a threshold are "transition frames". Neighbouring transition
     frames are grouped into one transition (a fade can span many frames).
  4. The shots are the stretches between transitions, in seconds.

Output, per video:
            <work>/<video_id>/shots.json         the shots and transitions found
            <work>/<video_id>/cut_scores.npy     the raw score of every frame (for tuning)
And for the whole run:
            <work>/shots_summary.csv             one row per video

Install (pip only):
    pip install torch transnetv2-pytorch numpy
    (for an NVIDIA GPU, install torch from the command at pytorch.org first)

Usage:
    python shots.py --work work
    python shots.py --work work --videos KlbDo9o5SC8 026dzf-vc5g
    python shots.py --work work --threshold 0.55      # same threshold for every video
    python shots.py --work work --redo                # recompute videos already done

Faster on a CPU:
    python shots.py --work work --workers 2           # 2 videos at once, CPU threads split between them
    python shots.py --work work --frame-skip 2        # score every 2nd frame: about 2x faster
    python shots.py --work work --batch 32            # more windows per model call
These can be combined. Results from --frame-skip are slightly less exact: hard
cuts are still found, but very short fades can be missed and cut times are
accurate to 2 frames (~0.07 s at 30 fps) instead of 1.
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

# How TransNet V2 reads the video. These match the original paper's code, so
# the scores are the same as the reference implementation.
WINDOW = 100          # frames the model sees at once
STEP = 50             # how far the window moves each time
KEEP_FROM = 25        # of each 100-frame window, keep the scores of frames 25..74,
KEEP_TO = 75          # where the model can see 25 frames of context on both sides

# How many 100-frame windows to send to the model in one go (--batch). Bigger
# is faster on a GPU; on a CPU it helps a little. Lower it if memory runs out.
DEFAULT_WINDOWS_PER_BATCH = 8

# Cut threshold per domain (see final-approach.md, Step 2).
DEFAULT_THRESHOLD = 0.5
DOMAIN_THRESHOLDS = {
    "Life Record": 0.6,            # handheld, shaky footage looks like cuts
    "Artistic Performance": 0.6,   # stage lighting flashes look like cuts
}

# Print a progress line every time this much more of a video is scored.
PROGRESS_STEP = 0.10


# ----------------------------------------------------------------------------
# Loading the model
# ----------------------------------------------------------------------------

def load_model(device: str, cpu_threads: int = None):
    """
    Load TransNet V2 with its trained weights (they ship inside the pip package).
    `device` is "auto", "cpu" or "cuda". "auto" picks the GPU when there is one.
    `cpu_threads` limits how many CPU cores this process uses (None = all).
    Imported here so the rest of the file can be read and tested without torch.
    """
    import torch
    from transnetv2_pytorch import TransNetV2

    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransNetV2(device=device)
    model.eval()
    return model, device


def make_scorer(model, device: str):
    """
    Wrap the model in a plain function: windows in (numpy), scores out (numpy).

    Input:  uint8 array of shape (number_of_windows, 100, 27, 48, 3)
    Output: float array of shape (number_of_windows, 100), one score per frame.

    TransNet V2 has two output heads. We use the first ("single frame"), which
    is trained to fire on the frame where a cut happens. This is the head the
    original authors use to find shots.
    """
    import torch

    def score_windows(windows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            batch = torch.from_numpy(windows).to(device)
            single_frame, _all_frames = model.predict_raw(batch)   # both already 0..1
            return single_frame[:, :, 0].float().cpu().numpy()

    return score_windows


# ----------------------------------------------------------------------------
# Scoring every frame of one video
# ----------------------------------------------------------------------------

def pad_frames(frames: np.ndarray) -> np.ndarray:
    """
    Add copies of the first and last frame so that:
      - the very first real frame also gets 25 frames of "context" before it;
      - the total length fits a whole number of 50-frame steps.
    Example: 130 frames -> 25 copies of frame 0, the 130 frames, then 45 copies
    of the last frame = 200 frames = 3 windows (0-99, 50-149, 100-199).
    """
    n = len(frames)
    pad_start = KEEP_FROM
    leftover = n % STEP
    pad_end = KEEP_FROM + (STEP - leftover if leftover else 0)
    return np.concatenate([
        np.repeat(frames[:1], pad_start, axis=0),
        frames,
        np.repeat(frames[-1:], pad_end, axis=0),
    ])


def score_all_frames(frames: np.ndarray, score_windows, video_id: str,
                     windows_per_batch: int = DEFAULT_WINDOWS_PER_BATCH) -> np.ndarray:
    """
    Slide a 100-frame window over the video in steps of 50 and keep the middle
    50 scores of each window. Together those middles cover every frame once.
    Prints progress every 10%.
    """
    n = len(frames)
    padded = pad_frames(frames)
    window_starts = list(range(0, len(padded) - WINDOW + 1, STEP))

    scores = []
    started = time.time()
    next_report = PROGRESS_STEP

    for i in range(0, len(window_starts), windows_per_batch):
        starts = window_starts[i:i + windows_per_batch]
        windows = np.stack([padded[s:s + WINDOW] for s in starts])
        window_scores = score_windows(windows)                 # (batch, 100)
        scores.extend(window_scores[:, KEEP_FROM:KEEP_TO].reshape(-1))

        done = min(len(scores), n) / n
        if done >= next_report and done < 1.0:
            elapsed = time.time() - started
            speed = min(len(scores), n) / max(elapsed, 1e-6)
            left = (n - min(len(scores), n)) / max(speed, 1e-6)
            print(f"  [{done:4.0%}] {video_id}  {min(len(scores), n)}/{n} frames"
                  f"  |  {speed:,.0f} frames/s  |  ~{left:.0f} s left", flush=True)
            while next_report <= done:
                next_report += PROGRESS_STEP

    return np.asarray(scores[:n], dtype=np.float32)     # drop the padded tail


# ----------------------------------------------------------------------------
# From scores to transitions and shots
# ----------------------------------------------------------------------------

def find_transitions(scores: np.ndarray, threshold: float) -> list:
    """
    Group neighbouring above-threshold frames into transitions.
    Returns a list of (first_frame, last_frame, peak_score).

    Example with threshold 0.5:
        scores: 0.01 0.02 0.97 0.03 ... 0.6 0.8 0.7 0.1
        -> one hard cut at the 0.97 frame, one 3-frame fade at 0.6 0.8 0.7
    """
    above = scores > threshold
    transitions = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            transitions.append((i, j, float(scores[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    return transitions


def transitions_to_shots(transitions: list, timestamps: np.ndarray, video_end: float) -> list:
    """
    Turn transitions into shots measured in seconds.

    Each transition becomes ONE cut time:
      - a hard cut (1 frame):        the time of that frame;
      - a fade or dissolve (frames a..b): halfway between frame a and frame b.
    Shots are then simply the spans between consecutive cut times, from the
    start of the video to its end, so the shots cover the whole video with no
    gaps and no overlaps.
    """
    n = len(timestamps)
    cut_points = []          # (cut_time, cut_frame, peak_score)
    for first, last, peak in transitions:
        # A "transition" touching the very first frame is just the video starting.
        if first == 0:
            continue
        cut_time = (timestamps[first] + timestamps[last]) / 2
        cut_frame = (first + last) // 2
        cut_points.append((float(cut_time), int(cut_frame), peak))

    shots = []
    start_time, start_frame, start_score = float(timestamps[0]), 0, None
    for cut_time, cut_frame, peak in cut_points + [(video_end, n, None)]:
        if cut_time <= start_time:           # skip a cut that lands on the shot start
            continue
        shots.append({
            "index": len(shots),
            "start": round(start_time, 3),
            "end": round(cut_time, 3),
            "duration": round(cut_time - start_time, 3),
            "start_frame": start_frame,
            "end_frame": cut_frame - 1,       # last frame that belongs to this shot
            "cut_score_at_start": None if start_score is None else round(start_score, 3),
        })
        start_time, start_frame, start_score = cut_time, cut_frame, peak
    return shots


# ----------------------------------------------------------------------------
# One video, start to finish
# ----------------------------------------------------------------------------

def threshold_for(meta: dict, override) -> float:
    """--threshold on the command line wins; otherwise pick by domain."""
    if override is not None:
        return override
    return DOMAIN_THRESHOLDS.get(meta.get("domain"), DEFAULT_THRESHOLD)


def process_one_video(video_dir: Path, score_windows, device: str, threshold_override, redo: bool,
                      frame_skip: int = 1, windows_per_batch: int = DEFAULT_WINDOWS_PER_BATCH) -> dict:
    video_id = video_dir.name
    out_file = video_dir / "shots.json"

    # Resume support: skip videos finished in an earlier run.
    if out_file.exists() and not redo:
        result = json.loads(out_file.read_text(encoding="utf-8"))
        return summary_row(result, status="already done")

    try:
        meta = json.loads((video_dir / "meta.json").read_text(encoding="utf-8"))
        all_frames = np.load(video_dir / "frames_48x27.npy")
        all_timestamps = np.load(video_dir / "timestamps.npy")
        if len(all_frames) != len(all_timestamps):
            raise ValueError("frames and timestamps have different lengths; rerun decode.py")

        threshold = threshold_for(meta, threshold_override)
        # The video ends one frame after the last frame's timestamp.
        frame_gap = float(np.median(np.diff(all_timestamps))) if len(all_timestamps) > 1 else 1 / 30
        video_end = float(all_timestamps[-1]) + frame_gap

        # Frame skip: keep every Nth frame (and its timestamp). The model then
        # sees a lower frame rate, e.g. 30 fps -> 15 fps with --frame-skip 2.
        # Everything below works on these kept frames; frame numbers are mapped
        # back to the original video at the end.
        frames = all_frames[::frame_skip]
        timestamps = all_timestamps[::frame_skip]

        skip_note = f"  |  every {frame_skip} frames" if frame_skip > 1 else ""
        print(f"  [start] {video_id}  {len(frames):,} frames{skip_note}  |  domain: {meta.get('domain')}"
              f"  |  threshold {threshold}", flush=True)
        started = time.time()

        scores = score_all_frames(frames, score_windows, video_id, windows_per_batch)
        transitions = find_transitions(scores, threshold)
        shots = transitions_to_shots(transitions, timestamps, video_end)

        # Map frame numbers from "kept frames" back to "original frames".
        if frame_skip > 1:
            last_original = len(all_frames) - 1
            for shot in shots:
                shot["start_frame"] = shot["start_frame"] * frame_skip
                shot["end_frame"] = min(last_original, (shot["end_frame"] + 1) * frame_skip - 1)
            transitions = [(a * frame_skip, min(last_original, b * frame_skip), p) for a, b, p in transitions]

        result = {
            "video_id": video_id,
            "domain": meta.get("domain"),
            "tier": meta.get("tier"),
            "threshold": threshold,
            "device": device,
            "frame_skip": frame_skip,            # cut_scores.npy has one score per KEPT frame
            "frames": int(len(all_frames)),
            "video_seconds": round(video_end - float(timestamps[0]), 3),
            "seconds_taken": round(time.time() - started, 1),
            "transitions": [
                {"first_frame": a, "last_frame": b, "frames": b - a + 1,
                 "start": round(float(all_timestamps[a]), 3), "end": round(float(all_timestamps[b]), 3),
                 "peak_score": round(p, 3)}
                for a, b, p in transitions
            ],
            "shots": shots,
        }
        np.save(video_dir / "cut_scores.npy", scores)
        # Written last, so a crash halfway never leaves a "finished" marker.
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return summary_row(result, status="ok")

    except Exception as error:   # one bad video must not stop the others
        return {"video_id": video_id, "status": f"error: {error}"}


def summary_row(result: dict, status: str) -> dict:
    """The key numbers for one video, for the CSV and the progress line."""
    shots = result.get("shots", [])
    lengths = [s["duration"] for s in shots]
    minutes = result.get("video_seconds", 0) / 60
    return {
        "video_id": result.get("video_id"),
        "status": status,
        "domain": result.get("domain"),
        "tier": result.get("tier"),
        "threshold": result.get("threshold"),
        "video_minutes": round(minutes, 2),
        "shots": len(shots),
        "shots_per_minute": round(len(shots) / minutes, 1) if minutes else None,
        "median_shot_s": round(float(np.median(lengths)), 1) if lengths else None,
        "longest_shot_s": round(max(lengths), 1) if lengths else None,
        "soft_transitions": sum(1 for t in result.get("transitions", []) if t["frames"] > result.get("frame_skip", 1)),
        "frame_skip": result.get("frame_skip"),
        "seconds_taken": result.get("seconds_taken"),
    }


# ----------------------------------------------------------------------------
# Running several videos at once (--workers)
# ----------------------------------------------------------------------------

# Each worker process loads its own copy of the model once, when it starts,
# and keeps it here. (Processes do not share memory, so it cannot be passed in.)
_worker_scorer = None
_worker_device = None


def _start_worker(device: str, cpu_threads: int):
    """Runs once inside each worker process: load the model with its share of CPU threads."""
    global _worker_scorer, _worker_device
    model, _worker_device = load_model(device, cpu_threads)
    _worker_scorer = make_scorer(model, _worker_device)


def _worker_job(video_dir, threshold_override, redo, frame_skip, windows_per_batch):
    """Runs inside a worker process for one video, using that worker's model."""
    return process_one_video(video_dir, _worker_scorer, _worker_device, threshold_override, redo,
                             frame_skip, windows_per_batch)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2: find shots with TransNet V2.")
    parser.add_argument("--work", type=Path, default=Path("work"),
                        help="the --out folder of decode.py (default: work)")
    parser.add_argument("--videos", nargs="*", help="only these video ids (default: all)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="use this cut threshold for every video (default: per domain)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="where to run the model (default: GPU if available)")
    parser.add_argument("--redo", action="store_true", help="recompute videos already done")
    parser.add_argument("--workers", type=int, default=1,
                        help="videos processed at the same time (default 1). On a CPU, try 2; "
                             "the CPU cores are split evenly between workers")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="score every Nth frame (default 1 = every frame). 2 is about 2x faster; "
                             "keep it so the kept frame rate stays at 12 fps or more")
    parser.add_argument("--batch", type=int, default=DEFAULT_WINDOWS_PER_BATCH,
                        help=f"100-frame windows per model call (default {DEFAULT_WINDOWS_PER_BATCH}); try 32 on a CPU")
    args = parser.parse_args()
    if args.workers < 1 or args.frame_skip < 1 or args.batch < 1:
        raise SystemExit("--workers, --frame-skip and --batch must be 1 or more")

    # Only folders where decode.py finished (it writes meta.json last).
    video_dirs = sorted(d for d in args.work.iterdir() if (d / "meta.json").exists())
    if args.videos:
        wanted = set(args.videos)
        video_dirs = [d for d in video_dirs if d.name in wanted]
    if not video_dirs:
        raise SystemExit(f"No decoded videos found in {args.work}. Run decode.py first.")

    rows = []
    run_started = time.time()

    def report_finished(row):
        """One line per finished video, with an overall counter."""
        rows.append(row)
        details = ""
        if row["status"] in ("ok", "already done"):
            details = (f"  |  {row['shots']} shots, median {row['median_shot_s']} s"
                       f"  |  {row['seconds_taken']} s")
        print(f"  [done {len(rows)}/{len(video_dirs)}] {row['video_id']}: {row['status']}{details}"
              f"  |  total elapsed {time.time() - run_started:.0f} s", flush=True)

    if args.workers == 1:
        # One process: load the model once and reuse it for every video.
        print("Loading TransNet V2...", flush=True)
        model, device = load_model(args.device)
        score_windows = make_scorer(model, device)
        print(f"Model ready on {device}. Processing {len(video_dirs)} video(s)...", flush=True)
        for video_dir in video_dirs:
            report_finished(process_one_video(video_dir, score_windows, device, args.threshold,
                                              args.redo, args.frame_skip, args.batch))
    else:
        # Several processes, each with its own model and an equal share of the
        # CPU cores. On a CPU this usually beats one process using every core,
        # because a single model call can't keep all cores busy all the time.
        # (On a GPU, keep --workers 1: the workers would share one card.)
        cores = os.cpu_count() or 2
        threads_each = max(1, cores // args.workers)
        print(f"Starting {args.workers} workers, {threads_each} CPU threads each "
              f"({cores} available). Each loads its own copy of TransNet V2...", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_start_worker,
                                 initargs=(args.device, threads_each)) as pool:
            jobs = [pool.submit(_worker_job, d, args.threshold, args.redo, args.frame_skip, args.batch)
                    for d in video_dirs]
            # Report each video the moment it finishes, not in list order.
            for job in as_completed(jobs):
                report_finished(job.result())

    columns = list(summary_row({}, "").keys())
    with (args.work / "shots_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})

    failed = [r for r in rows if r["status"].startswith("error")]
    print(f"Done. {len(rows) - len(failed)} ok, {len(failed)} failed."
          f" Summary: {args.work / 'shots_summary.csv'}")


if __name__ == "__main__":
    main()
