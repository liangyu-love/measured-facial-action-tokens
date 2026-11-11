"""v2-prompt descriptions for all 4403 clips with any Qwen VL model (+ optional LoRA adapter merged), bf16.
Test split first, so test-set evaluation can start early. Resumable.
Usage: python gen_full.py --model Qwen3-VL-8B-Instruct --out outputs/full_q3vl8b [--adapter models/sft_q3vl4b_lora]
"""
import argparse
import json
import os
import time

import pandas as pd
import torch

from gen_descriptions import describe
from scale_desc import ROOT, frames448, load

F = os.path.join(ROOT, "outputs/full")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    model, proc = load(a.model, adapter=os.path.join(ROOT, a.adapter) if a.adapter else None)
    od = os.path.join(out, "desc.jsonl")
    done = {json.loads(l)["key"] for l in open(od)} if os.path.exists(od) else set()
    clips = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str)
    keys = clips.sort_values("split", key=lambda s: s.map({"test": 0, "valid": 1, "train": 2})).key.tolist()
    with open(od, "a") as f:
        for n, k in enumerate(keys, 1):
            if k in done:
                continue
            d = os.path.join(F, "frames", k.replace("/", "__"))
            info = json.load(open(os.path.join(d, "frames.json")))
            t = time.time()
            text, _ = describe(model, proc, frames448(d), info["times"])
            f.write(json.dumps({"key": k, "desc": text, "sec": round(time.time() - t, 2),
                                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 500 == 0:
                print(f"{a.model} {n}/{len(keys)}", flush=True)


if __name__ == "__main__":
    main()
