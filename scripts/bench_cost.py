"""Computational cost for the KBS paper (2026-11-11): Py-Feat extraction time per clip (8 frames, Detectorv2 on GPU, as in emo_frames.py)
and BERT forward latency for the transcript alone vs. transcript + 8 evidence tokens. Writes outputs/full/cost_bench.json."""
import json, os, time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.expanduser("~/work/msa-pilot"); F = os.path.join(ROOT, "outputs/full")
clips = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str)
test = clips[clips.split == "test"].key.tolist()
rng = np.random.default_rng(0); sample = list(rng.choice(test, 60, replace=False))
out = {"gpu": torch.cuda.get_device_name(0)}

# ---- Py-Feat: per-clip extraction of 8 frames (face detection + AU + emotion in one pass)
from feat import Detectorv2
det = Detectorv2(device="cuda", identity_model=None)
times = []
for n, k in enumerate(sample):
    d = os.path.join(F, "frames", k.replace("/", "__"))
    paths = [p for p in (os.path.join(d, f"F{i}.jpg") for i in range(1, 9)) if os.path.exists(p)]
    torch.cuda.synchronize(); t = time.perf_counter()
    det.detect(paths, data_type="image", batch_size=8)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    if n >= 10:  # first 10 clips are warm-up
        times.append(dt)
out["pyfeat_sec_per_clip"] = {"mean": float(np.mean(times)), "median": float(np.median(times)), "n": len(times)}
del det; torch.cuda.empty_cache()

# ---- BERT forward latency: text vs text + 8 evidence positions
from transformers import AutoTokenizer, BertModel
name = os.path.join(ROOT, "models", "bert-base-chinese")
tok = AutoTokenizer.from_pretrained(name); bert = BertModel.from_pretrained(name).cuda().eval()
lens = [len(tok(t)["input_ids"]) for t in clips[clips.split == "test"].text.fillna("")] if "text" in clips else [40]
L = int(np.median(lens)); out["median_text_tokens"] = L
for bs in (1, 32):
    for extra in (0, 8):
        x = torch.randint(1000, 5000, (bs, L + extra), device="cuda")
        tt = torch.zeros_like(x); tt[:, L:] = 1
        with torch.no_grad():
            for _ in range(10):
                bert(input_ids=x, token_type_ids=tt)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(50):
                bert(input_ids=x, token_type_ids=tt)
            torch.cuda.synchronize()
        out[f"bert_ms_bs{bs}_plus{extra}"] = (time.perf_counter() - t) / 50 * 1000
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(F, "cost_bench.json"), "w"), indent=1)
