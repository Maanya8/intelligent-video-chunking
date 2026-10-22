"""
Steps 0 and 1 of the video chunking pipeline .

  Step 0 - Probe:  read basic facts about each video (length, fps, size, audio),
                   load its subtitles, and look up its domain and duration tier.
  Step 1 - Decode: turn the video into pictures ONCE and keep two small copies:
                   A) every frame shrunk to 48x27   -> for TransNet V2 (Step 2)
                   B) one frame per second, 224 px  -> for image embeddings (Step 3)

Everything here installs with pip:
    pip install av numpy pillow

--workers is how many videos the script processes at the same time.

--workers 1 (the default): one video at a time. Video 1 finishes, then video 2 starts, and so on.
--workers 3: three separate Python processes run side by side, each on its own video. When one finishes, it picks up the next video in the list.

Usage (from the folder that contains the downloaded dataset):
    python decode.py --data data/vmme --out work
    python decode.py --data data/vmme --out work --videos KlbDo9o5SC8 026dzf-vc5g
    python decode.py --data data/vmme --out work --workers 4

Progress: each video prints a line when it starts, every 10% while decoding
(with speed and time left), and when it finishes, e.g.
    [start] KlbDo9o5SC8  (9.5 min of video)
    [ 40%] KlbDo9o5SC8  228/570 s of video  |  12.3x real time  |  ~28 s left
    [done 3/90] KlbDo9o5SC8: ok  (47 s)

What it writes, per video, in  <out>/<video_id>/ :
    meta.json            Step 0 facts + subtitle lines + what Step 1 produced
    frames_48x27.npy     copy A: array of shape (N, 27, 48, 3), uint8 RGB
    timestamps.npy       copy A: time in seconds of each of those N frames
    frames_1fps/*.jpg    copy B: one picture per second of video
    frames_1fps.json     copy B: the time in seconds of each picture
And for the whole run:
    <out>/probe.csv      one row per video, handy to open in Excel
"""

import argparse
import csv
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

# Copy A: the size TransNet V2 was trained on (width x height).
TRANSNET_WIDTH = 48
TRANSNET_HEIGHT = 27

# Copy B: one picture every this many seconds, scaled so the SHORT side is this
# many pixels. Embedding models (CLIP / SigLIP) expect about 224 px.
SAMPLE_EVERY_SECONDS = 1.0
EMBED_SHORT_SIDE = 224
JPEG_QUALITY = 90

# Copy A is collected in batches of this many frames, so a 60-minute video
# does not need one giant Python list of 100,000 small arrays.
BATCH_SIZE = 2000

# Print a progress line every time this much more of the video is decoded.
PROGRESS_STEP = 0.10      # 10%
# For videos whose length is unknown, print every this many seconds instead.
PROGRESS_EVERY_SECONDS = 15


# ----------------------------------------------------------------------------
# Step 0a: look up each video's labels in the question file
# ----------------------------------------------------------------------------

def load_video_labels(data_dir: Path) -> dict:
    """
    Read qa/test.jsonl and return {video_id: {"tier": ..., "domain": ..., "sub_category": ...}}.

    The file has one question per line and 3 questions per video, so we keep
    the first line we see for each video. A real customer corpus will not have
    these labels; then this function simply returns an empty dict.
    """
    labels = {}
    qa_file = data_dir / "qa" / "test.jsonl"
    if not qa_file.exists():
        return labels

    with qa_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = row["video_id"]
            if video_id not in labels:
                labels[video_id] = {
                    "tier": row.get("duration"),          # "short" / "medium" / "long"
                    "domain": row.get("domain"),
                    "sub_category": row.get("sub_category"),
                }
    return labels


# ----------------------------------------------------------------------------
# Step 0b: read the subtitle file (.srt)
# ----------------------------------------------------------------------------

# An .srt timestamp looks like 00:01:02,500  (hours:minutes:seconds,milliseconds)
SRT_TIME = r"(\d+):(\d+):(\d+)[,.](\d+)"
SRT_TIMING_LINE = re.compile(SRT_TIME + r"\s*-->\s*" + SRT_TIME)


def srt_time_to_seconds(hours, minutes, seconds, millis) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def load_subtitles(srt_path: Path) -> list:
    """
    Parse an .srt file into a list of {"start": s, "end": s, "text": "..."}.

    An .srt file is made of blocks separated by blank lines:
        12
        00:01:02,500 --> 00:01:05,000
        The text, possibly
        over two lines
    We only need the timing line and the text lines. Returns [] if the video
    has no subtitles (12 of the 90 videos in this dataset don't).
    """
    if not srt_path.exists():
        return []

    # utf-8-sig also strips the invisible "byte order mark" some .srt files start with.
    content = srt_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = []
    for block in re.split(r"\n\s*\n", content.strip()):
        block_lines = block.strip().splitlines()
        for i, line in enumerate(block_lines):
            match = SRT_TIMING_LINE.search(line)
            if match:
                start = srt_time_to_seconds(*match.groups()[:4])
                end = srt_time_to_seconds(*match.groups()[4:])
                text = " ".join(l.strip() for l in block_lines[i + 1:]).strip()
                if text:
                    lines.append({"start": round(start, 3), "end": round(end, 3), "text": text})
                break
    lines.sort(key=lambda s: s["start"])
    return lines


# ----------------------------------------------------------------------------
# Step 0c: read the facts stored in the video file's header
# ----------------------------------------------------------------------------

def probe_video(video_path: Path) -> dict:
    """
    Open the file and read its header only (no decoding yet): codec, size,
    frame rate, duration, bitrate, and whether there is an audio track.
    """
    with av.open(str(video_path)) as container:
        video = container.streams.video[0]

        # Duration: prefer the video stream's own value; fall back to the
        # container's. PyAV gives the container duration in microseconds.
        if video.duration is not None and video.time_base is not None:
            duration = float(video.duration * video.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        else:
            duration = None

        fps = float(video.average_rate) if video.average_rate else None

        return {
            "file": video_path.name,
            "size_bytes": video_path.stat().st_size,
            "codec": video.codec_context.name,
            "width": video.codec_context.width,
            "height": video.codec_context.height,
            "fps": round(fps, 3) if fps else None,
            "duration_s": round(duration, 3) if duration else None,
            "frame_count_in_header": video.frames or None,   # 0 means "unknown"
            "bitrate_bps": container.bit_rate or None,
            "has_audio": len(container.streams.audio) > 0,
        }


# ----------------------------------------------------------------------------
# Step 1: decode once, keep two small copies
# ----------------------------------------------------------------------------

def size_for_short_side(width: int, height: int, short_side: int) -> tuple:
    """Scale (width, height) so the shorter side becomes `short_side`, keeping the shape.
    Sizes are rounded to even numbers, which video scalers prefer."""
    scale = short_side / min(width, height)
    new_w = max(2, round(width * scale / 2) * 2)
    new_h = max(2, round(height * scale / 2) * 2)
    return new_w, new_h


def print_progress(video_id: str, video_seconds: float, duration: float, started: float):
    """One progress line: how far into the video we are, how fast, and time left."""
    elapsed = max(time.time() - started, 1e-6)
    speed = video_seconds / elapsed                     # seconds of video per second of work
    if duration:
        percent = min(video_seconds / duration, 1.0)
        left = (duration - video_seconds) / speed if speed > 0 else 0
        print(f"  [{percent:4.0%}] {video_id}  {video_seconds:.0f}/{duration:.0f} s of video"
              f"  |  {speed:.1f}x real time  |  ~{left:.0f} s left", flush=True)
    else:
        print(f"  [ ...] {video_id}  {video_seconds:.0f} s of video decoded"
              f"  |  {speed:.1f}x real time", flush=True)


def decode_video(video_path: Path, out_dir: Path, duration: float = None) -> dict:
    """
    Walk through every frame of the video exactly once. For each frame:
      - always keep a 48x27 RGB copy (copy A) and its timestamp;
      - if a full second has passed since the last saved picture, also save
        a 224 px JPEG (copy B).
    The full-size frame is never stored, so memory stays small.
    `duration` (from Step 0) is only used to show progress as a percentage.
    """
    keyframe_dir = out_dir / "frames_1fps"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    small_batches = []      # copy A, gathered in batches of BATCH_SIZE frames
    current_batch = []
    timestamps = []         # copy A: time of every frame, in seconds
    keyframe_times = []     # copy B: time of every saved picture

    next_sample_time = 0.0  # when the next 1-per-second picture is due
    last_time = -1.0

    # Progress bookkeeping.
    video_id = video_path.stem
    started = time.time()
    next_progress_fraction = PROGRESS_STEP
    next_progress_wallclock = started + PROGRESS_EVERY_SECONDS

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # Let the decoder use several CPU threads. This is the single biggest
        # speed-up for H.264 and does not change the output.
        stream.thread_type = "AUTO"
        embed_w, embed_h = size_for_short_side(
            stream.codec_context.width, stream.codec_context.height, EMBED_SHORT_SIDE
        )

        for frame in container.decode(stream):
            # The frame's own timestamp. Safer than "frame number / fps"
            # because some videos have a slightly uneven frame rate.
            t = frame.time if frame.time is not None else last_time + 1 / 30
            # Rarely a broken file repeats or reorders a timestamp; keep time moving forward.
            if t <= last_time:
                t = last_time + 1e-3
            last_time = t

            # Copy A: let the decoder's own scaler shrink the frame. This is
            # much faster than converting to a full-size picture first.
            tiny = frame.reformat(width=TRANSNET_WIDTH, height=TRANSNET_HEIGHT, format="rgb24")
            current_batch.append(tiny.to_ndarray())
            timestamps.append(t)
            if len(current_batch) == BATCH_SIZE:
                small_batches.append(np.stack(current_batch))
                current_batch = []

            # Copy B: the first frame at or after each whole second.
            if t >= next_sample_time:
                picture = frame.reformat(width=embed_w, height=embed_h, format="rgb24").to_image()
                index = len(keyframe_times)
                picture.save(keyframe_dir / f"{index:05d}.jpg", quality=JPEG_QUALITY)
                keyframe_times.append(round(t, 3))
                # Next sample one second after the slot we just filled. If the
                # video skipped ahead, jump to the next slot after `t`.
                while next_sample_time <= t:
                    next_sample_time += SAMPLE_EVERY_SECONDS

            # Progress: every 10% of the video, or every 15 s if length is unknown.
            if duration:
                if t / duration >= next_progress_fraction and next_progress_fraction < 1.0:
                    print_progress(video_id, t, duration, started)
                    while next_progress_fraction <= t / duration:
                        next_progress_fraction += PROGRESS_STEP
            elif time.time() >= next_progress_wallclock:
                print_progress(video_id, t, None, started)
                next_progress_wallclock = time.time() + PROGRESS_EVERY_SECONDS

    if current_batch:
        small_batches.append(np.stack(current_batch))
    if not small_batches:
        raise RuntimeError("no frames could be decoded")

    frames = np.concatenate(small_batches)                 # shape (N, 27, 48, 3)
    np.save(out_dir / "frames_48x27.npy", frames)
    np.save(out_dir / "timestamps.npy", np.asarray(timestamps, dtype=np.float64))
    (out_dir / "frames_1fps.json").write_text(json.dumps(keyframe_times))

    return {
        "decoded_frames": int(frames.shape[0]),
        "first_frame_s": round(timestamps[0], 3),
        "last_frame_s": round(timestamps[-1], 3),
        "pictures_1fps": len(keyframe_times),
        "embed_picture_size": [embed_w, embed_h],
    }


# ----------------------------------------------------------------------------
# One video, start to finish
# ----------------------------------------------------------------------------

def process_one_video(video_path: Path, data_dir: Path, out_root: Path, labels: dict) -> dict:
    """Run Step 0 and Step 1 for one video and write meta.json. Returns a summary row."""
    video_id = video_path.stem
    out_dir = out_root / video_id
    meta_file = out_dir / "meta.json"

    # Resume support: skip videos finished in an earlier run.
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return summary_row(video_id, meta, status="already done")

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        # ---- Step 0 ----
        meta = {"video_id": video_id}
        meta.update(labels.get(video_id, {"tier": None, "domain": None, "sub_category": None}))
        meta["probe"] = probe_video(video_path)
        meta["subtitles"] = load_subtitles(data_dir / "subtitles" / f"{video_id}.srt")

        duration = meta["probe"]["duration_s"]
        length = f"{duration / 60:.1f} min of video" if duration else "length unknown"
        print(f"  [start] {video_id}  ({length})", flush=True)

        # ---- Step 1 ----
        meta["decode"] = decode_video(video_path, out_dir, duration)
        meta["decode"]["seconds_taken"] = round(time.time() - started, 1)

        # Written last, so a crash halfway never leaves a "finished" marker.
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary_row(video_id, meta, status="ok")

    except Exception as error:  # one bad file must not stop the other 89
        return {"video_id": video_id, "status": f"error: {error}"}


def summary_row(video_id: str, meta: dict, status: str) -> dict:
    """Flatten the interesting parts of meta.json into one CSV row."""
    probe, decode = meta.get("probe", {}), meta.get("decode", {})
    return {
        "video_id": video_id,
        "status": status,
        "tier": meta.get("tier"),
        "domain": meta.get("domain"),
        "codec": probe.get("codec"),
        "width": probe.get("width"),
        "height": probe.get("height"),
        "fps": probe.get("fps"),
        "duration_s": probe.get("duration_s"),
        "bitrate_bps": probe.get("bitrate_bps"),
        "size_mb": round(probe.get("size_bytes", 0) / 1e6, 1),
        "has_audio": probe.get("has_audio"),
        "subtitle_lines": len(meta.get("subtitles", [])),
        "decoded_frames": decode.get("decoded_frames"),
        "pictures_1fps": decode.get("pictures_1fps"),
        "decode_seconds": decode.get("seconds_taken"),
    }


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Steps 0-1: probe and decode videos for chunking.")
    parser.add_argument("--data", type=Path, required=True,
                        help="dataset folder that contains videos/, subtitles/ and qa/")
    parser.add_argument("--out", type=Path, default=Path("work"),
                        help="where to write the per-video results (default: work)")
    parser.add_argument("--videos", nargs="*",
                        help="only these video ids (default: every .mp4 in videos/)")
    parser.add_argument("--workers", type=int, default=1,
                        help="videos processed at the same time (default 1; try 2-4)")
    args = parser.parse_args()

    video_paths = sorted((args.data / "videos").glob("*.mp4"))
    if args.videos:
        wanted = set(args.videos)
        video_paths = [p for p in video_paths if p.stem in wanted]
    if not video_paths:
        raise SystemExit(f"No .mp4 files found in {args.data / 'videos'}")

    labels = load_video_labels(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(video_paths)} video(s) with {args.workers} worker(s)...")

    rows = []
    run_started = time.time()

    def report_finished(row):
        """Print one line per finished video, with an overall counter."""
        rows.append(row)
        took = row.get("decode_seconds")
        took_text = f"  ({took:.0f} s)" if isinstance(took, (int, float)) and row["status"] == "ok" else ""
        print(f"  [done {len(rows)}/{len(video_paths)}] {row['video_id']}: {row['status']}{took_text}"
              f"  |  total elapsed {time.time() - run_started:.0f} s", flush=True)

    if args.workers == 1:
        for path in video_paths:
            report_finished(process_one_video(path, args.data, args.out, labels))
    else:
        # Separate processes, one video each. Each decoder already uses threads,
        # so a few workers is enough; too many just fight over the CPU.
        # as_completed() reports each video the moment it finishes, instead of
        # waiting for them in list order (which made the run look stuck).
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(process_one_video, p, args.data, args.out, labels) for p in video_paths]
            for job in as_completed(jobs):
                report_finished(job.result())

    # One summary table for the whole run.
    columns = list(summary_row("", {}, "").keys())
    with (args.out / "probe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})

    failed = [r for r in rows if r["status"].startswith("error")]
    print(f"Done. {len(rows) - len(failed)} ok, {len(failed)} failed. Summary: {args.out / 'probe.csv'}")


if __name__ == "__main__":
    # Needed on Windows so worker processes start cleanly.
    main()
