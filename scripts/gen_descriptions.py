"""Frames -> Qwen2.5-VL -> purely observational descriptions (facial actions / posture), cached to JSONL.

The prompt forbids emotion/sentiment words so the description stays a "visual evidence" channel.
Resumable: clips already in the output JSONL are skipped.

Usage:
  python gen_descriptions.py --splits ~/work/msa-pilot/outputs/splits/emomm_chsims_base.csv \
      --video-root ~/work/msa-pilot/data/ch-sims-v2s/Raw --out ~/work/msa-pilot/outputs/desc_qwen7b.jsonl
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL_DIR = os.path.expanduser("~/work/msa-pilot/models/Qwen2.5-VL-7B-Instruct")

PROMPT = (
    "You are given {n} frames (F1..F{n}) sampled in order from a short video clip of one speaker. "
    "List the visible facial actions and head/gaze movements, ONE action per bullet. "
    "Each bullet MUST start with the specific frame index or indices where that action is visible, "
    "e.g. '[F3] lip corners pulled up' or '[F2,F5] eyes closed'. Use a full range like [F1-F{n}] only if the "
    "action is clearly present in every frame; prefer describing changes between frames. "
    "Use only physical descriptions: brows raised/lowered/drawn together, eyes closed/widened/narrowed, "
    "cheeks raised, nose wrinkled, lip corners pulled up/down, lips pressed/parted, jaw dropped, "
    "head tilted/turned/nodding/still, gaze down/up/sideways/forward. "
    "FORBIDDEN: any emotion, mood, attitude or intention word (e.g. happy, sad, angry, surprised, relaxed, "
    "engaged, interested, friendly, confident, serious, attentive) and any inferential phrase "
    "(suggesting, indicating, showing that, expressing, appears to, seems, as if, possibly, likely). "
    "Cover the brows, eyes, mouth/lips and head; subtle actions count. "
    "If a face is not visible in a frame, say '[Fk] face not visible'. "
    "Write 4 to 8 bullets, English only."
)


# v3 vocabulary: fine-grained, purely physical head actions matching head_actions.py classes.
PROMPT_V3 = PROMPT.replace(
    "head tilted/turned/nodding/still, gaze down/up/sideways/forward. ",
    "head actions ONLY from this list: nodding (head moves down-up repeatedly), shaking (head moves left-right "
    "repeatedly), tilted to one side, turned away / face to the side, head down, head up, leaning forward, "
    "leaning back, small head movement while talking, head still; gaze down/up/sideways/forward. "
    "A nod is a physical movement only: never describe it as agreement or approval. "
    "Distinguish a brief blink from eyes kept closed. ")
assert PROMPT_V3 != PROMPT, "PROMPT_V3 replacement anchor not found"


def sample_frames(path, num_frames, max_side=448):
    """Uniformly sample frames with torchcodec (needs LD_LIBRARY_PATH from ~/.venv/bin/activate)."""
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(path)
    n = dec.metadata.num_frames
    idx = np.linspace(0, n - 1, num_frames).round().astype(int).tolist()
    batch = dec.get_frames_at(indices=idx)  # uint8 [N, C, H, W]
    fps = dec.metadata.average_fps or 25.0
    frames = []
    for t in batch.data:
        im = Image.fromarray(t.permute(1, 2, 0).numpy())
        s = max_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)))
        frames.append(im)
    return frames, [round(i / fps, 2) for i in idx]


def load_model(model_dir=MODEL_DIR):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.eval()
    proc = AutoProcessor.from_pretrained(model_dir)
    return model, proc


@torch.inference_mode()
def describe(model, proc, frames, times, max_new_tokens=200, prompt=None):
    content = []
    for i, (im, t) in enumerate(zip(frames, times), 1):
        content.append({"type": "text", "text": f"F{i} (t={t}s):"})
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": (prompt or PROMPT).format(n=len(frames))})
    msgs = [{"role": "user", "content": content}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip(), inputs["input_ids"].shape[1]


def video_path(root, key, exts=(".mp4", ".avi", ".mkv")):
    # CH-SIMS layout is assumed to be Raw/<video_id>/<clip_id>.<ext>; adjust once the data lands.
    vid, clip = key.split("/")
    for e in exts:
        p = os.path.join(root, vid, f"{clip}{e}")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="csv with a 'key' column (video_id/clip_id)")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    keys = pd.read_csv(a.splits)["key"].drop_duplicates().tolist()
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [k for k in keys if k not in done][: a.limit or None]
    print(f"{len(keys)} clips, {len(done)} cached, {len(todo)} to do")
    if not todo:
        return

    model, proc = load_model()
    with open(a.out, "a") as f:
        for k in todo:
            p = video_path(a.video_root, k)
            if p is None:
                print("[skip] no video for", k)
                continue
            t0 = time.time()
            frames, times = sample_frames(p, a.num_frames)
            text, n_tok = describe(model, proc, frames, times)
            rec = {"key": k, "model": os.path.basename(MODEL_DIR), "num_frames": len(frames),
                   "times": times, "prompt_tokens": n_tok, "desc": text, "sec": round(time.time() - t0, 2)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
