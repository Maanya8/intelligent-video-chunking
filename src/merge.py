"""
Step 3 of the video chunking pipeline (see final-approach.md): merge similar neighbouring shots.

Input, per video (from earlier steps):
    <work>/<video_id>/shots.json         the shots from Step 2 (shots.py)
    <work>/<video_id>/frames_1fps/*.jpg  one picture per second (decode.py)
    <work>/<video_id>/frames_1fps.json   the time of each of those pictures
    <work>/<video_id>/meta.json          tier and domain

What it does, per video:
  1. Turn every 1-per-second picture into an "embedding" with CLIP: a list of
     512 numbers that describes what the picture shows. Two pictures of the
     same scene get similar lists, even from a different camera angle.
     (Saved to frame_embeddings.npy, so this slow part runs only once.)
  2. Describe each shot by the average embedding of its pictures.
  3. Walk through the shots in order and join a shot onto the chunk before it when:
       a) the chunk before is still shorter than the minimum length, or
       b) the shot itself is shorter than the minimum length (unless it looks
          more like the NEXT shot: then it starts a new chunk, and rule a)
          joins the next shot onto it), or
       c) the shot looks like the chunk (similarity >= 0.85; 0.80 for sports).
     b) and c) only join if the result stays within the maximum length; if it
     wouldn't, the cut is kept, so a long scene is broken at a real camera cut
     instead of being split at an arbitrary point later.
  4. If the very last chunk is still too short, join it onto the one before.

  a) has no maximum check, because a too-short chunk must go somewhere. That is
  the only way a merge can make a chunk longer than the maximum, and Step 4
  (split.py) runs next and splits any such chunk.

Output, per video:
    <work>/<video_id>/merged.json     the chunks after this step, plus this video's log
And for the whole run (open them in Excel):
    <work>/merge_log.csv              one row per video: shots in, chunks out, how many
                                      cuts were removed and why
    <work>/merge_events.csv           one row per removed cut (and per cut kept only
                                      because of the maximum), with video-player
                                      timestamps and the similarity, to check by hand

Install (pip only):
    pip install torch open_clip_torch pillow numpy
The CLIP weights (about 600 MB) download automatically the first time.

Usage:
    python merge.py --work decoded
    python merge.py --work decoded --videos 1NYQf_OXDqI RwdvZLz-aSk
    python merge.py --work decoded --similarity 0.9    # same threshold for every video
    python merge.py --work decoded --redo              # redo the merging (embeddings are reused)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chunk_common import clock, length_report, limits_for, read_json, write_csv, write_json


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

# The image model. ViT-B-32 is the smallest common CLIP model: good enough to
# tell "same scene" from "different scene", and the fastest on a CPU.
# (A bigger model or SigLIP would be a little more accurate and several
# times slower; worth trying only if the merges look wrong.)
CLIP_MODEL = "ViT-B-32"
CLIP_WEIGHTS = "laion2b_s34b_b79k"
IMAGES_PER_BATCH = 64

# Similarity needed to join a shot onto the chunk before it (1.0 = identical).
DEFAULT_SIMILARITY = 0.85
DOMAIN_SIMILARITY = {
    "Sports Competition": 0.80,   # replays and wide/close shots of the same play should join
}

PROGRESS_STEP = 0.10


# ----------------------------------------------------------------------------
# Embeddings (the slow part, cached per video)
# ----------------------------------------------------------------------------

def load_clip(device: str):
    """Load CLIP and return a function: list of PIL images -> (n, 512) unit-length array."""
    import torch
    import open_clip

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
    model = model.to(device).eval()

    def embed(images) -> np.ndarray:
        with torch.no_grad():
            batch = torch.stack([preprocess(img) for img in images]).to(device)
            features = model.encode_image(batch).float()
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()

    return embed, device


def frame_embeddings(video_dir: Path, frame_times: list, embed) -> np.ndarray:
    """
    One embedding per 1-per-second picture, shape (n_pictures, 512).
    Read from frame_embeddings.npy if an earlier run made it.
    """
    cache = video_dir / "frame_embeddings.npy"
    if cache.exists():
        saved = np.load(cache)
        if len(saved) == len(frame_times):
            return saved.astype(np.float32)

    from PIL import Image

    video_id = video_dir.name
    n = len(frame_times)
    print(f"  [embed] {video_id}: {n} pictures", flush=True)
    started, next_report = time.time(), PROGRESS_STEP
    parts = []
    for i in range(0, n, IMAGES_PER_BATCH):
        images = [Image.open(video_dir / "frames_1fps" / f"{k:05d}.jpg").convert("RGB")
                  for k in range(i, min(i + IMAGES_PER_BATCH, n))]
        parts.append(embed(images))
        done = min(i + IMAGES_PER_BATCH, n) / n
        if done >= next_report and done < 1.0:
            speed = (i + len(images)) / max(time.time() - started, 1e-6)
            print(f"  [{done:4.0%}] {video_id}  {i + len(images)}/{n} pictures  |  {speed:.0f} pictures/s"
                  f"  |  ~{(n - i - len(images)) / max(speed, 1e-6):.0f} s left", flush=True)
            while next_report <= done:
                next_report += PROGRESS_STEP

    embeddings = np.concatenate(parts) if parts else np.zeros((0, 512), np.float32)
    np.save(cache, embeddings.astype(np.float16))     # half precision: half the disk, same results
    return embeddings.astype(np.float32)


def shot_embedding(start: float, end: float, frame_times: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Average embedding of the pictures inside [start, end). A shot shorter than
    a second may contain no picture; then use the picture nearest its middle.
    Returned as a SUM plus a count, so chunks can be averaged cheaply later.
    """
    inside = np.where((frame_times >= start) & (frame_times < end))[0]
    if len(inside) == 0:
        inside = [int(np.argmin(np.abs(frame_times - (start + end) / 2)))]
    return embeddings[inside].sum(axis=0), len(inside)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ----------------------------------------------------------------------------
# The merging itself
# ----------------------------------------------------------------------------

def merge_shots(shots: list, frame_times: np.ndarray, embeddings: np.ndarray,
                min_len: float, max_len: float, similarity: float) -> tuple:
    """
    Walk through the shots and join them into chunks (rules a-c and the last-chunk
    rule in the notes at the top). Returns (chunks, events), where events lists
    every removed cut and every similar cut that was kept because of max_len.
    """
    chunks, events = [], []

    def new_chunk(shot, emb_sum, count, reason, sim):
        chunks.append({"start": shot["start"], "end": shot["end"], "n_shots": 1,
                       "start_reason": reason, "start_similarity": sim,
                       "_sum": emb_sum.copy(), "_count": count})

    shot_embs = [shot_embedding(s["start"], s["end"], frame_times, embeddings) for s in shots]

    for i, shot in enumerate(shots):
        emb_sum, count = shot_embs[i]
        if not chunks:
            new_chunk(shot, emb_sum, count, "video start", None)
            continue

        chunk = chunks[-1]
        sim = round(cosine(chunk["_sum"], emb_sum), 3)
        chunk_len = chunk["end"] - chunk["start"]
        shot_len = shot["end"] - shot["start"]
        fits = shot["end"] - chunk["start"] <= max_len

        reason = None
        if chunk_len < min_len:
            reason = "chunk before too short"
        elif shot_len < min_len and fits:
            # A short shot belongs with whichever side it looks more like. If that is
            # the NEXT shot, start a new chunk here; rule a) then pulls the next shot in.
            looks_like_next = i + 1 < len(shots) and cosine(shot_embs[i + 1][0], emb_sum) > sim
            if not looks_like_next:
                reason = "shot too short"
        elif sim >= similarity and fits:
            reason = "similar"

        if reason is None:
            if (sim >= similarity or shot_len < min_len) and not fits:
                events.append(event_row(chunk, shot, sim, "kept: join would pass max", None))
            new_chunk(shot, emb_sum, count, "shot", sim)
            continue

        events.append(event_row(chunk, shot, sim, f"merged: {reason}", shot["end"] - chunk["start"]))
        chunk["end"] = shot["end"]
        chunk["n_shots"] += 1
        chunk["_sum"] += emb_sum
        chunk["_count"] += count

    # The last chunk has nothing after it to absorb it, so join it backwards.
    if len(chunks) > 1 and chunks[-1]["end"] - chunks[-1]["start"] < min_len:
        last, before = chunks.pop(), chunks[-1]
        sim = round(cosine(before["_sum"], last["_sum"]), 3)
        events.append(event_row(before, last, sim, "merged: last chunk too short", last["end"] - before["start"]))
        before["end"] = last["end"]
        before["n_shots"] += last["n_shots"]

    for k, c in enumerate(chunks):
        del c["_sum"], c["_count"]
        c["index"] = k
        c["duration"] = round(c["end"] - c["start"], 3)
    return chunks, events


def event_row(chunk: dict, shot: dict, sim, action: str, new_length) -> dict:
    """One line of merge_events.csv: the cut between `chunk` and `shot`."""
    return {
        "cut_at": clock(shot["start"]),
        "action": action,
        "similarity": sim,
        "chunk_before": f"{clock(chunk['start'])} - {clock(chunk['end'])}",
        "shot_after": f"{clock(shot['start'])} - {clock(shot['end'])}",
        "chunk_before_s": round(chunk["end"] - chunk["start"], 1),
        "shot_after_s": round(shot["end"] - shot["start"], 1),
        "joined_s": None if new_length is None else round(new_length, 1),
        "cut_s": shot["start"],
    }


# ----------------------------------------------------------------------------
# One video, start to finish
# ----------------------------------------------------------------------------

def similarity_for(meta: dict, override) -> float:
    """--similarity on the command line wins; otherwise pick by domain."""
    if override is not None:
        return override
    return DOMAIN_SIMILARITY.get(meta.get("domain"), DEFAULT_SIMILARITY)


def process_one_video(video_dir: Path, get_embedder, similarity_override, min_override, max_override,
                      redo: bool) -> tuple:
    """Returns (log_row, event_rows) for the two CSV files."""
    video_id = video_dir.name
    out_file = video_dir / "merged.json"

    if out_file.exists() and not redo:
        saved = read_json(out_file)
        return dict(saved["log"], status="already done"), saved["events"]

    started = time.time()
    try:
        meta = read_json(video_dir / "meta.json")
        shots = read_json(video_dir / "shots.json")["shots"]
        frame_times = np.asarray(json.loads((video_dir / "frames_1fps.json").read_text()), dtype=np.float64)
        min_len, max_len = limits_for(meta, min_override, max_override)
        similarity = similarity_for(meta, similarity_override)

        embeddings = frame_embeddings(video_dir, list(frame_times), get_embedder)
        embed_seconds = round(time.time() - started, 1)

        chunks, events = merge_shots(shots, frame_times, embeddings, min_len, max_len, similarity)
        for e in events:
            e["video_id"] = video_id

        actions = [e["action"] for e in events]
        after = length_report(chunks, min_len, max_len)
        lengths = sorted(c["duration"] for c in chunks)
        log = {
            "video_id": video_id,
            "status": "ok",
            "domain": meta.get("domain"),
            "tier": meta.get("tier"),
            "min_s": min_len,
            "max_s": max_len,
            "similarity": similarity,
            "shots_in": len(shots),
            "chunks_out": after["chunks"],
            "cuts_removed": len(shots) - after["chunks"],
            "by_similar": actions.count("merged: similar"),
            "by_short_shot": actions.count("merged: shot too short"),
            "by_short_chunk": actions.count("merged: chunk before too short"),
            "by_short_last_chunk": actions.count("merged: last chunk too short"),
            "kept_for_max": actions.count("kept: join would pass max"),
            "median_chunk_s": lengths[len(lengths) // 2] if lengths else None,
            "longest_chunk_s": after["longest_s"],
            "below_min_out": after["below_min"],     # only if the whole video is shorter than min
            "above_max_out": after["above_max"],     # long single shots; split.py fixes these
            "embed_seconds": embed_seconds,
            "seconds_taken": round(time.time() - started, 1),
        }
        write_json(out_file, {"video_id": video_id, "domain": meta.get("domain"), "tier": meta.get("tier"),
                              "min_s": min_len, "max_s": max_len, "similarity": similarity,
                              "log": log, "events": events, "chunks": chunks})
        return log, events

    except Exception as error:   # one bad video must not stop the others
        return {"video_id": video_id, "status": f"error: {error}"}, []


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

LOG_COLUMNS = ["video_id", "status", "domain", "tier", "min_s", "max_s", "similarity",
               "shots_in", "chunks_out", "cuts_removed", "by_similar", "by_short_shot", "by_short_chunk",
               "by_short_last_chunk", "kept_for_max", "median_chunk_s", "longest_chunk_s",
               "below_min_out", "above_max_out", "embed_seconds", "seconds_taken"]
EVENT_COLUMNS = ["video_id", "cut_at", "action", "similarity", "chunk_before", "shot_after",
                 "chunk_before_s", "shot_after_s", "joined_s", "cut_s"]


class LazyEmbedder:
    """Loads CLIP only when the first video actually needs new embeddings."""
    def __init__(self, device):
        self.device, self.embed = device, None

    def __call__(self, images):
        if self.embed is None:
            print("Loading CLIP (downloads about 600 MB the first time)...", flush=True)
            self.embed, self.device = load_clip(self.device)
            print(f"CLIP ready on {self.device}.", flush=True)
        return self.embed(images)


def main():
    parser = argparse.ArgumentParser(description="Step 3: merge similar neighbouring shots.")
    parser.add_argument("--work", type=Path, default=Path("work"), help="the folder decode.py and shots.py wrote to")
    parser.add_argument("--videos", nargs="*", help="only these video ids (default: all)")
    parser.add_argument("--similarity", type=float, default=None,
                        help="similarity needed to merge, for every video (default: 0.85, sports 0.80)")
    parser.add_argument("--min", type=float, default=None, help="minimum chunk seconds (default: by tier)")
    parser.add_argument("--max", type=float, default=None, help="maximum chunk seconds (default: by tier)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--redo", action="store_true",
                        help="redo the merging for videos already done (embeddings are still reused)")
    args = parser.parse_args()

    video_dirs = sorted(d for d in args.work.iterdir() if (d / "shots.json").exists())
    if args.videos:
        video_dirs = [d for d in video_dirs if d.name in set(args.videos)]
    if not video_dirs:
        raise SystemExit(f"No videos with shots.json found in {args.work}. Run shots.py first.")

    embedder = LazyEmbedder(args.device)
    logs, all_events = [], []
    run_started = time.time()
    for n, video_dir in enumerate(video_dirs, 1):
        log, events = process_one_video(video_dir, embedder, args.similarity, args.min, args.max, args.redo)
        logs.append(log)
        all_events.extend(events)
        details = ""
        if log["status"] in ("ok", "already done"):
            details = (f"{log['shots_in']} shots -> {log['chunks_out']} chunks"
                       f"  |  {log['cuts_removed']} cuts removed ({log['by_similar']} similar,"
                       f" {log['by_short_shot'] + log['by_short_chunk'] + log['by_short_last_chunk']} too short)")
        print(f"  [done {n}/{len(video_dirs)}] {log['video_id']}: {log['status']}  {details}"
              f"  |  total elapsed {time.time() - run_started:.0f} s", flush=True)

    all_events.sort(key=lambda e: (e["video_id"], e["cut_s"]))
    write_csv(args.work / "merge_log.csv", logs, LOG_COLUMNS)
    write_csv(args.work / "merge_events.csv", all_events, EVENT_COLUMNS)
    failed = [r for r in logs if r["status"].startswith("error")]
    print(f"Done. {len(logs) - len(failed)} ok, {len(failed)} failed. "
          f"{sum(r.get('cuts_removed', 0) for r in logs)} cuts removed in total.")
    print(f"Logs: {args.work / 'merge_log.csv'} (per video) and {args.work / 'merge_events.csv'} (every merge)")
    print("Next: python split.py --redo ... and python snap.py --redo ..., so they use these chunks.")


if __name__ == "__main__":
    main()
