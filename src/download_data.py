"""
Download the Video-MME sample dataset from Hugging Face (non-interactive, safe to rerun).

    https://huggingface.co/datasets/aintropy-ai/video-mme-sample10

By default it downloads the questions, subtitles and a small sample of 3 videos
(one short, one medium, one long: about 15-30 min of pipeline time on a CPU).
Use --all for all 90 videos (about 11 GB, and many hours of CPU time).
Files that are already there are not downloaded again.

Usage:
    python src/download_data.py --out data/vmme
    python src/download_data.py --out data/vmme --all
    python src/download_data.py --out data/vmme --videos 026dzf-vc5g KlbDo9o5SC8
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "aintropy-ai/video-mme-sample10"
# One short (~1 min), one medium (~13 min) and one long (~33 min) video.
SAMPLE_VIDEOS = ["026dzf-vc5g", "KlbDo9o5SC8", "LCfBYE97rFk"]
SMALL_FILES = ["qa/*", "subtitles/*", "manifest.json", "sample_ids.txt"]


def main():
    parser = argparse.ArgumentParser(description="Download the Video-MME sample dataset.")
    parser.add_argument("--out", type=Path, default=Path("data/vmme"), help="where to put it (default: data/vmme)")
    parser.add_argument("--all", action="store_true", help="download all 90 videos (about 11 GB)")
    parser.add_argument("--videos", nargs="*", help="download these video ids (default: a 3-video sample)")
    args = parser.parse_args()

    if args.all:
        patterns = None                     # everything
    else:
        videos = args.videos or SAMPLE_VIDEOS
        patterns = SMALL_FILES + [f"videos/{v}.mp4" for v in videos]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} ({'all files' if args.all else ', '.join(patterns)}) to {args.out} ...", flush=True)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(args.out), allow_patterns=patterns)
    count = len(list((args.out / "videos").glob("*.mp4")))
    print(f"Done. {count} video(s) in {args.out / 'videos'}.")


if __name__ == "__main__":
    main()
