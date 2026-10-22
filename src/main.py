"""
Run the whole video chunking pipeline with one command.

Steps, in order (each one is its own script in this folder):
    decode   Steps 0-1  probe each video and decode it once      (decode.py)
    shots    Step 2     find camera cuts with TransNet V2         (shots.py)
    merge    Step 3     merge similar neighbouring shots (CLIP)   (merge.py)
    split    Step 4     split chunks that are too long            (split.py)
    snap     Step 5     move boundaries to pauses in speech       (snap.py)
    check    Layer 1    sanity checks on the final chunks         (check.py)
    export   Step 8     write chunks.jsonl / chunks.csv           (export.py)

Running this again is safe. The slow steps (decode, shots, merge) skip videos
they have already finished. The fast steps (split, snap) always recompute, so
they never keep results built from an older merge. After each step, main.py
checks that every video got that step's output file, and stops with a clear
message if one is missing (for example when the CLIP weights could not be
downloaded), instead of carrying on with incomplete results.

Output:
    <work>/chunks.jsonl        the final chunks, one JSON record per chunk
    <work>/chunks.csv          the same, for Excel
    <work>/check_report.csv    sanity-check verdict per video
    plus each step's own logs (merge_log.csv, split_log.csv, snap_log.csv, ...)

Usage:
    python src/main.py --data data/vmme --work work
    python src/main.py --data data/vmme --work work --videos 026dzf-vc5g KlbDo9o5SC8
    python src/main.py --data data/vmme --work work --steps check export
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["decode", "shots", "merge", "split", "snap", "check", "export"]


def step_command(step: str, args) -> list:
    """The command line for one step."""
    work, data = str(args.work), str(args.data)
    videos = ["--videos", *args.videos] if args.videos else []
    commands = {
        "decode": ["decode.py", "--data", data, "--out", work, "--workers", str(args.workers)],
        "shots": ["shots.py", "--work", work, "--workers", str(args.workers),
                  "--frame-skip", str(args.frame_skip), "--device", args.device],
        "merge": ["merge.py", "--work", work, "--device", args.device]
                 + (["--similarity", str(args.similarity)] if args.similarity else []),
        "split": ["split.py", "--work", work, "--data", data, "--redo"],
        "snap": ["snap.py", "--work", work, "--redo"],
        "check": ["check.py", "--work", work],
        "export": ["export.py", "--work", work],
    }
    script, *rest = commands[step]
    return [sys.executable, str(HERE / script), *rest, *videos]


# The file each step writes per video, used to check that no video was left out.
STEP_OUTPUT = {"decode": "meta.json", "shots": "shots.json", "merge": "merged.json",
               "split": "split.json", "snap": "snapped.json"}


def missing_outputs(step: str, args) -> list:
    """Video ids that should have this step's output file but don't."""
    if step not in STEP_OUTPUT:
        return []
    if args.videos:
        expected = list(args.videos)
    else:
        expected = sorted(p.stem for p in (args.data / "videos").glob("*.mp4"))
    return [v for v in expected if not (args.work / v / STEP_OUTPUT[step]).exists()]


def main():
    parser = argparse.ArgumentParser(description="Run the video chunking pipeline end to end.")
    parser.add_argument("--data", type=Path, required=True,
                        help="dataset folder with videos/, subtitles/ and qa/ (see download_data.py)")
    parser.add_argument("--work", type=Path, default=Path("work"), help="where all results go (default: work)")
    parser.add_argument("--videos", nargs="*", help="only these video ids (default: every video in <data>/videos)")
    parser.add_argument("--steps", nargs="*", default=STEPS, choices=STEPS, help="run only these steps")
    parser.add_argument("--workers", type=int, default=2, help="videos processed at the same time (default 2)")
    parser.add_argument("--frame-skip", type=int, default=2,
                        help="TransNet V2 scores every Nth frame (default 2: about 2x faster on CPU)")
    parser.add_argument("--similarity", type=float, default=0.75, help="merge similarity threshold (default 0.75)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if "decode" in args.steps and not (args.data / "videos").is_dir():
        raise SystemExit(f"No videos/ folder in {args.data}. Run src/download_data.py first.")

    started = time.time()
    for step in STEPS:
        if step not in args.steps:
            continue
        command = step_command(step, args)
        print(f"\n=== {step} ===\n$ {' '.join(command)}", flush=True)
        step_started = time.time()
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(f"Step '{step}' failed (exit code {result.returncode}). "
                             f"Fix it and rerun; finished work is kept.")
        missing = missing_outputs(step, args)
        if missing:
            raise SystemExit(f"Step '{step}' did not finish for {len(missing)} video(s): {', '.join(missing)}. "
                             f"See the [error] lines above. Fix it and rerun; finished work is kept.")
        print(f"=== {step} done in {time.time() - step_started:.0f} s ===", flush=True)

    print(f"\nPipeline finished in {(time.time() - started) / 60:.1f} min. "
          f"Final chunks: {args.work / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
