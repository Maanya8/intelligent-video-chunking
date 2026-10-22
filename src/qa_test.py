"""
Layer 3 of the test plan : can a model answer
the dataset's questions from only the top 3 chunks?

This is the "real" test. It compares chunking methods by how useful their
chunks are, not by how they look:
    ours      the final chunks (snapped.json, after Steps 2-5)
    fixed30   plain 30 s windows, the simplest baseline
    shots     raw TransNet V2 shots (shots.json), no merging/splitting/snapping

For every question and every method:
  1. Rank that video's chunks with a simple retriever that is the SAME for
     every method, so only the chunking changes:
       - picture match: CLIP text embedding of the question vs the average CLIP
         embedding of the chunk's 1-per-second pictures (already saved by merge.py)
       - word match: BM25 between the question and the chunk's subtitle text
     Both scores are put on the same scale (z-scores within the video) and added.
  2. Take the top 3 chunks, put them back in time order, and give a small
     vision-language model up to 8 pictures per chunk plus the chunk's subtitles.
  3. Ask the multiple-choice question and read back one letter (A-D).

Accuracy per method is the result. A small model scores lower than a big one,
but every method uses the same model, so the comparison is fair.

Input:
    <data>/qa/test.jsonl                     the questions (dataset)
    <work>/<video_id>/meta.json              duration, subtitles (decode.py)
    <work>/<video_id>/frames_1fps/*.jpg      pictures (decode.py)
    <work>/<video_id>/frames_1fps.json       picture times (decode.py)
    <work>/<video_id>/frame_embeddings.npy   CLIP picture embeddings (merge.py)
    <work>/<video_id>/shots.json             raw shots (shots.py)
    <work>/<video_id>/snapped.json           final chunks (snap.py)

Output (resumable: answers already in qa_results.jsonl are skipped on a rerun):
    <work>/qa_results.jsonl   one line per (question, method): chunks used, model reply, right or wrong
    <work>/qa_summary.csv     accuracy per method: overall, and by domain, tier and task type

Install (pip only):
    pip install -U transformers accelerate torch open_clip_torch pillow numpy
    (Qwen3-VL needs a recent transformers, 4.57 or newer.)

Usage:
    # 1. Try 3 questions first, to see the speed on your machine:
    python qa_test.py --work decoded --data data\\vmme --questions 3
    # 2. The trimmed run: 1 question per video (90 questions x 3 methods):
    python qa_test.py --work decoded --data data\\vmme --per-video 1
    # 3. Everything (270 questions x 3 methods). Already-answered ones are skipped:
    python qa_test.py --work decoded --data data\\vmme
    # Just rebuild qa_summary.csv from what is already answered:
    python qa_test.py --work decoded --data data\\vmme --summary-only
"""

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from chunk_common import clock, read_json, write_csv


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
METHODS = ["ours", "fixed30", "shots"]
FIXED_WINDOW = 30.0          # seconds, for the fixed30 baseline
TOP_K = 3                    # chunks given to the model per question
FRAMES_PER_CHUNK = 8         # at most this many pictures per chunk (fewer if the chunk is short)
SUBTITLE_CHARS = 600         # at most this much subtitle text per chunk
MAX_NEW_TOKENS = 8           # the answer is one letter; a few extra tokens in case it says "Answer: B"

CLIP_MODEL = "ViT-B-32"              # must match merge.py, so text and picture embeddings agree
CLIP_WEIGHTS = "laion2b_s34b_b79k"

LETTERS = "ABCD"


# ----------------------------------------------------------------------------
# Questions
# ----------------------------------------------------------------------------

def load_questions(data_dir: Path) -> list:
    """All questions from qa/test.jsonl, with options normalised to ["A. ...", "B. ...", ...]."""
    questions = []
    with (data_dir / "qa" / "test.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            options = row.get("options") or []
            if isinstance(options, dict):
                options = [f"{k}. {v}" for k, v in sorted(options.items())]
            options = [o if re.match(r"^[A-D][.)]\s", o) else f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
            row["options"] = options
            row["answer"] = str(row.get("answer", "")).strip()[:1].upper()
            questions.append(row)
    return questions


def pick_questions(questions: list, per_video, limit) -> list:
    """The first `per_video` questions of each video, then the first `limit` of those."""
    if per_video:
        seen = Counter()
        kept = []
        for q in questions:
            if seen[q["video_id"]] < per_video:
                kept.append(q)
                seen[q["video_id"]] += 1
        questions = kept
    if limit:
        questions = questions[:limit]
    return questions


# ----------------------------------------------------------------------------
# One video: its chunks for each method, and what's in them
# ----------------------------------------------------------------------------

def clean(text: str) -> str:
    """Subtitle text without <font ...> tags and extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def fixed_windows(duration: float) -> list:
    chunks, t = [], 0.0
    while t < duration - 1e-6:
        chunks.append({"start": t, "end": min(t + FIXED_WINDOW, duration)})
        t += FIXED_WINDOW
    return chunks


class Video:
    """Everything about one video that the retriever and the model need."""

    def __init__(self, video_dir: Path):
        self.dir = video_dir
        self.meta = read_json(video_dir / "meta.json")
        self.duration = (self.meta.get("probe") or {}).get("duration_s") or 0.0
        self.subtitles = [dict(s, text=clean(s.get("text", ""))) for s in self.meta.get("subtitles", [])]
        self.frame_times = np.asarray(json.loads((video_dir / "frames_1fps.json").read_text()), dtype=np.float64)
        self.frame_embeddings = np.load(video_dir / "frame_embeddings.npy").astype(np.float32)
        if not self.duration and len(self.frame_times):
            self.duration = float(self.frame_times[-1]) + 1.0

        self.chunks = {
            "ours": read_json(video_dir / "snapped.json")["chunks"],
            "fixed30": fixed_windows(self.duration),
            "shots": read_json(video_dir / "shots.json")["shots"],
        }

    def frame_indices(self, start: float, end: float) -> np.ndarray:
        """Indices of the 1-per-second pictures inside [start, end). The nearest one if none."""
        inside = np.where((self.frame_times >= start) & (self.frame_times < end))[0]
        if len(inside) == 0 and len(self.frame_times):
            inside = np.array([int(np.argmin(np.abs(self.frame_times - (start + end) / 2)))])
        return inside

    def subtitle_text(self, start: float, end: float) -> str:
        """The subtitle lines that start inside [start, end), joined."""
        return " ".join(s["text"] for s in self.subtitles if start <= s["start"] < end)


# ----------------------------------------------------------------------------
# The retriever (identical for every method)
# ----------------------------------------------------------------------------

def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def bm25_scores(query: str, documents: list, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Classic BM25 word-match score of `query` against each document."""
    docs = [tokenize(d) for d in documents]
    n = len(docs)
    avg_len = (sum(len(d) for d in docs) / n) if n else 0
    doc_freq = Counter(word for d in docs for word in set(d))
    scores = np.zeros(n, dtype=np.float64)
    for word in set(tokenize(query)):
        if word not in doc_freq:
            continue
        idf = math.log(1 + (n - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(word)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / max(avg_len, 1e-6)))
    return scores


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else np.zeros_like(x)


def load_clip_text():
    """Return a function: text -> unit-length (512,) CLIP embedding."""
    import torch
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
    model = model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    def embed_text(text: str) -> np.ndarray:
        with torch.no_grad():
            features = model.encode_text(tokenizer([text])).float()[0]
            return (features / features.norm()).numpy()

    return embed_text


def rank_chunks(video: Video, chunks: list, query: str, query_embedding: np.ndarray) -> list:
    """Indices of `chunks`, best match first."""
    picture = np.zeros(len(chunks))
    for i, c in enumerate(chunks):
        mean = video.frame_embeddings[video.frame_indices(c["start"], c["end"])].mean(axis=0)
        picture[i] = float(mean @ query_embedding) / max(float(np.linalg.norm(mean)), 1e-9)
    score = zscore(picture)
    if video.subtitles:
        score = score + zscore(bm25_scores(query, [video.subtitle_text(c["start"], c["end"]) for c in chunks]))
    return list(np.argsort(-score, kind="stable"))


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------

def load_vlm(model_name: str, threads):
    """Load the vision-language model. Returns a function: messages -> reply text."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if threads:
        torch.set_num_threads(threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    def ask(messages: list) -> str:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

    return ask, device


def build_messages(video: Video, chunks: list, question: dict) -> list:
    """One user turn: for each chunk its time, its pictures and its subtitles; then the question."""
    from PIL import Image

    content = [{"type": "text", "text": f"Below are {len(chunks)} clips from one video, in time order."}]
    for n, c in enumerate(chunks, 1):
        indices = video.frame_indices(c["start"], c["end"])
        if len(indices) > FRAMES_PER_CHUNK:
            indices = indices[np.linspace(0, len(indices) - 1, FRAMES_PER_CHUNK).round().astype(int)]
        content.append({"type": "text", "text": f"\nClip {n} ({clock(c['start'])} to {clock(c['end'])}):"})
        for k in indices:
            image = Image.open(video.dir / "frames_1fps" / f"{int(k):05d}.jpg").convert("RGB")
            content.append({"type": "image", "image": image})
        subtitles = video.subtitle_text(c["start"], c["end"])
        if subtitles:
            content.append({"type": "text", "text": f"Speech in clip {n}: {subtitles[:SUBTITLE_CHARS]}"})

    options = "\n".join(question["options"])
    content.append({"type": "text", "text": (
        f"\nQuestion: {question['question']}\n{options}\n"
        "Answer with only the letter (A, B, C or D) of the correct option.")})
    return [{"role": "user", "content": content}]


def read_letter(reply: str):
    """The first standalone A-D in the reply, or None."""
    match = re.search(r"\b([A-D])\b", reply.upper())
    return match.group(1) if match else None


# ----------------------------------------------------------------------------
# Results and summary
# ----------------------------------------------------------------------------

def load_done(results_file: Path) -> dict:
    done = {}
    if results_file.exists():
        for line in results_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(str(row["question_id"]), row["method"])] = row
    return done


def summary(rows: list) -> list:
    """Accuracy per method, overall and per domain / tier / task type."""
    out = []
    for group_by in ("all", "domain", "tier", "task_type"):
        groups = defaultdict(lambda: defaultdict(list))
        for r in rows:
            key = "all" if group_by == "all" else (r.get(group_by) or "unknown")
            groups[key][r["method"]].append(r["correct"])
        for key in sorted(groups):
            row = {"group_by": group_by, "group": key}
            for method in METHODS:
                answers = groups[key].get(method, [])
                row[f"{method}_n"] = len(answers)
                row[f"{method}_accuracy"] = round(sum(answers) / len(answers), 3) if answers else None
            out.append(row)
    return out


SUMMARY_COLUMNS = ["group_by", "group"] + [f"{m}_{x}" for m in METHODS for x in ("accuracy", "n")]


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer 3: answer the dataset's questions from the top 3 chunks.")
    parser.add_argument("--work", type=Path, default=Path("work"), help="the folder the earlier steps wrote to")
    parser.add_argument("--data", type=Path, required=True, help="the dataset folder (with qa/test.jsonl)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default {DEFAULT_MODEL})")
    parser.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    parser.add_argument("--per-video", type=int, default=None, help="only the first N questions of each video")
    parser.add_argument("--questions", type=int, default=None, help="stop after this many questions (for a trial)")
    parser.add_argument("--videos", nargs="*", help="only these video ids (default: all)")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for the model (default: all)")
    parser.add_argument("--summary-only", action="store_true", help="only rebuild qa_summary.csv")
    args = parser.parse_args()

    results_file = args.work / "qa_results.jsonl"
    done = load_done(results_file)

    if not args.summary_only:
        questions = load_questions(args.data)
        if args.videos:
            questions = [q for q in questions if q["video_id"] in set(args.videos)]
        questions = [q for q in questions if (args.work / q["video_id"] / "snapped.json").exists()]
        questions = pick_questions(questions, args.per_video, args.questions)
        todo = [(q, m) for q in questions for m in args.methods if (str(q["question_id"]), m) not in done]
        print(f"{len(questions)} question(s) x {len(args.methods)} method(s): "
              f"{len(todo)} to answer, {len(questions) * len(args.methods) - len(todo)} already done.", flush=True)

        if todo:
            print("Loading CLIP (for ranking) and the model "
                  f"{args.model} (first time: a few GB download)...", flush=True)
            embed_text = load_clip_text()
            ask, device = load_vlm(args.model, args.threads)
            print(f"Model ready on {device}.", flush=True)

            videos = {}
            running = defaultdict(list)
            started = time.time()
            with results_file.open("a", encoding="utf-8") as out:
                for n, (q, method) in enumerate(todo, 1):
                    video_id = q["video_id"]
                    if video_id not in videos:
                        videos.clear()           # keep only one video in memory
                        videos[video_id] = Video(args.work / video_id)
                    video = videos[video_id]

                    query = q["question"] + " " + " ".join(q["options"])
                    chunks = video.chunks[method]
                    top = sorted(rank_chunks(video, chunks, query, embed_text(query))[:TOP_K],
                                 key=lambda i: chunks[i]["start"])
                    chosen = [chunks[i] for i in top]

                    call_started = time.time()
                    try:
                        reply = ask(build_messages(video, chosen, q))
                    except Exception as error:   # one failed call must not stop the run
                        reply = f"error: {error}"
                    letter = read_letter(reply) if not reply.startswith("error:") else None

                    row = {
                        "question_id": q["question_id"], "video_id": video_id, "method": method,
                        "domain": q.get("domain"), "tier": q.get("duration"), "task_type": q.get("task_type"),
                        "chunks_used": [[round(c["start"], 2), round(c["end"], 2)] for c in chosen],
                        "chunks_in_video": len(chunks),
                        "reply": reply, "predicted": letter, "answer": q["answer"],
                        "correct": letter == q["answer"],
                        "seconds": round(time.time() - call_started, 1),
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    done[(str(q["question_id"]), method)] = row
                    running[method].append(row["correct"])

                    elapsed = time.time() - started
                    left = elapsed / n * (len(todo) - n)
                    so_far = "  ".join(f"{m} {sum(v)}/{len(v)}" for m, v in running.items())
                    mark = "right" if row["correct"] else ("ERROR" if letter is None else "wrong")
                    print(f"  [{n}/{len(todo)}] {video_id} q{q['question_id']} {method:7s}: "
                          f"{letter or '-'} vs {q['answer']} ({mark}, {row['seconds']} s)  |  "
                          f"so far: {so_far}  |  ~{left / 60:.0f} min left", flush=True)

    rows = list(done.values())
    table = summary(rows)
    write_csv(args.work / "qa_summary.csv", table, SUMMARY_COLUMNS)

    print()
    for r in table:
        if r["group_by"] == "all":
            for method in METHODS:
                if r[f"{method}_n"]:
                    print(f"  {method:8s} accuracy {r[f'{method}_accuracy']:.1%}  ({r[f'{method}_n']} questions)")
    errors = sum(r["predicted"] is None for r in rows)
    if errors:
        print(f"  {errors} answer(s) had no readable letter or failed; they count as wrong. See 'reply' in qa_results.jsonl.")
    print(f"Results: {results_file} (every answer) and {args.work / 'qa_summary.csv'} (accuracy by group)")


if __name__ == "__main__":
    main()
