# Intelligent video chunking

Cuts every video in a corpus into chunks that each hold one searchable thing (a scene, a play, a stretch of a talk). An indexer downstream can then caption and search them. The pipeline runs per video and only compares neighbouring shots, so its cost grows linearly with the size of the corpus.

Developed and evaluated on the [Video-MME sample](https://huggingface.co/datasets/aintropy-ai/video-mme-sample10): 90 YouTube videos, 25.9 hours, 6 domains, 30 short / 30 medium / 30 long.

## How it works

1. **Decode once** (PyAV): keep tiny 48×27 frames for shot detection and one 224 px frame per second for embeddings.
2. **Find camera cuts** with [TransNet V2](https://arxiv.org/abs/2008.04838). The threshold is 0.5, or 0.6 for handheld vlogs and stage performances.
3. **Merge neighbouring shots** that look alike (CLIP ViT-B-32, cosine ≥ 0.75), or that are shorter than the minimum chunk length.
4. **Split chunks that are too long** at a pause between subtitle lines, then at an audio silence, and as a last resort at an even point.
5. **Snap boundaries to speech pauses** within ±1.5 s, so a chunk doesn't start mid-sentence. YouTube "rolling" auto-captions are detected and handled.
6. **Write the output**: one record per chunk, with a ±1 s padded window for context.

Chunk length limits by duration tier: short 5–30 s, medium 8–90 s, long 10–120 s.

## Repository structure

```
run.yaml                 setup / run / test commands for the evaluator
requirements.txt         pip dependencies (CPU is enough; GPU is used if present)
src/
  main.py                runs the whole pipeline (decode → shots → merge → split → snap → check → export)
  download_data.py       downloads the dataset from Hugging Face (3-video sample by default, --all for 90)
  decode.py              Steps 0–1: probe each video, decode once, save tiny frames + 1 fps pictures
  shots.py               Step 2: TransNet V2 shot detection, per-domain threshold, optional frame skip
  merge.py               Step 3: merge similar or too-short neighbouring shots using CLIP embeddings
  split.py               Step 4: split chunks over the max length at subtitle pauses / silences
  snap.py                Step 5: move boundaries to pauses in speech without breaking length limits
  chunk_common.py        shared helpers: tier limits, subtitle speech intervals (incl. rolling captions), CSV/JSON I/O
  export.py              Step 8: write work/chunks.jsonl and work/chunks.csv (the hand-off to the indexer)
  check.py               sanity checks: coverage, length limits, mid-sentence cuts vs 30 s windows, outliers
  qa_test.py             question test: Qwen3-VL answers the dataset's questions from the top 3 chunks
```

## Quick start

The evaluator runs these automatically from `run.yaml`. To run them by hand, use the same commands in the same order:

```bash
pip install -r requirements.txt
python src/download_data.py --out data/vmme          # add --all for all 90 videos (~11 GB)
python src/main.py --data data/vmme --work work      # final output: work/chunks.jsonl
python src/check.py --work work
```

Every step skips videos it has already finished, so reruns are safe. Optional question test (slow on CPU):
`python src/qa_test.py --work work --data data/vmme --per-video 1`

## Output format

`work/chunks.jsonl` has one JSON record per chunk:
`video_id`, `chunk_id`, `start`, `end`, `padded_start`, `padded_end`, `n_shots`, `subtitle_text`, `boundary_reason` (`video start` / `shot` / `split`), `snapped`, `domain`, `tier`.

## Results (all 90 videos)

### Chunks

| | Count | Per hour of video |
|---|---|---|
| TransNet V2 shots (raw) | 13,952 | 539 |
| After merging (Step 3) | 2,786 | 108 |
| **Final chunks** | **2,903** | **112** |
| Fixed 30 s windows (baseline) | ~3,150 | 122 |

Median final chunk length is 18 s. We produce 4.8× fewer chunks than raw shots, which cuts the indexer's per-chunk bill by about 80%. Against fixed windows, the chunk count is about the same, but the boundaries fall at real scene changes and sentence breaks (see below).

### Sanity checks (`check.py`)

| Check | Result |
|---|---|
| Coverage (no gaps, no overlaps, start to end) | 90 / 90 videos pass |
| Length limits (min/max per tier) | 0 violations |
| Verdicts | 71 PASS, 19 WARN, **0 FAIL** |
| Boundaries that cut through a subtitle line | **19%** (fixed 30 s windows: **88%**) |

The WARNs are not bugs. They are short videos with very few boundaries (one bad cut out of 2 reads as 50%), videos with sparse subtitle lines, and single-camera videos split only by the length cap.

Rolling-caption handling made a big difference. Before it, only 23% of mid-sentence boundaries could be snapped to a pause; after it, 78% could.

### Question answering (`qa_test.py`)

Qwen3-VL-2B-Instruct (CPU) answers each multiple-choice question using only the top 3 chunks, ranked by the same CLIP + BM25 retriever for every method. Each chunk contributes up to 8 frames plus its subtitles. One question per video (90 questions).

| Method | Accuracy |
|---|---|
| **Ours** (final chunks) | _TBD_ |
| Fixed 30 s windows | _TBD_ |
| Raw TransNet V2 shots | _TBD_ |

A small model scores lower than a large one in absolute terms. Every method uses the same model, so the comparison between methods is fair. Raw shots get fewer frames per chunk because their chunks are only a few seconds long, which is part of what this test measures.

### Boundary F1

Not measured: it needs hand-labelled boundaries. For reference, TransNet V2 reports shot-detection F1 of 96.2 on BBC Planet Earth, 93.9 on RAI and 77.9 on ClipShots.

## Cost and scaling (measured on all 90 videos)

Measured on an ordinary desktop CPU (no GPU), with 2 videos processed in parallel for decode and shot detection, and TransNet V2 scoring every 2nd frame.

| Step | Machine-hours | Seconds per minute of video | Share |
|---|---|---|---|
| Decode | 0.85 | 2.0 | 13% |
| Shot detection (TransNet V2) | 4.25 | 9.9 | 64% |
| Merge (CLIP embeddings) | 1.52 | 3.5 | 23% |
| Split + snap | ~0 | ~0 | <1% |
| **Total** | **6.62** | **15.4** | |

- **Cost:** about **$2.25 for the whole 25.9-hour corpus**, or **~$0.09 per hour of video**. This assumes a comparable cloud machine at $0.34/hour. A GPU would make shot detection and CLIP, 87% of the time, much faster.
- **Scaling is linear.** Time per minute of video is the same for medium and long videos (15.0 vs 15.2 s/min). Short videos are a bit higher (21.4 s/min) because each video has a fixed start-up cost. No step looks across videos.
- **Projection:** about 2,560 machine-hours (~$870) for 10,000 hours of video. Videos are independent, so this splits across machines; 100 machines would finish in about a day.

## Known limits

- **Fixed cameras** (surveillance, meetings without slides): TransNet V2 finds almost no cuts, so chunks come only from the length cap. These need an activity- or speech-based splitter, which would require a hybrid approach involving a router which was deliberately not chosen.
- **Sparse subtitles:** when the nearest line break is more than 1.5 s away, the camera cut wins and the boundary stays mid-sentence.
- **Domain labels** come from the dataset. A real customer corpus would need a cheap property probe (cut rate, speech share, camera motion) to choose the thresholds.
- **Thresholds** (0.5 / 0.6 for cuts, 0.75 for merging, tier lengths) are starting values, not tuned on held-out data.
