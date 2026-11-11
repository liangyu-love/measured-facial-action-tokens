"""Model-scale check on the 200 pilot Conflict clips: same v2 prompt, same 8 frames (448 px max side) as the
Qwen2.5-VL-7B run. Any Qwen VL checkpoint under models/ (loaded via AutoModelForImageTextToText); --nf4 loads 4-bit.
Usage: python scale_desc.py --model Qwen3-VL-8B-Instruct --tag q3vl8b [--nf4]
Output: outputs/pilot_scale/<tag>/desc.jsonl (resumable), meta.json
"""
import argparse
import json
import os
import time

import pandas as pd
import torch
from PIL import Image

from gen_descriptions import describe

ROOT = os.path.expanduser("~/work/msa-pilot")
P = os.path.join(ROOT, "outputs/pilot")


def load(model_name, nf4=False, adapter=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    mdir = os.path.join(ROOT, "models", model_name)
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    if nf4:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(mdir, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.eval(), AutoProcessor.from_pretrained(mdir)


def frames448(d):
    ims = []
    for i in range(1, 9):
        im = Image.open(os.path.join(d, f"F{i}.jpg")).convert("RGB")
        s = 448 / max(im.size)
        ims.append(im.resize((round(im.width * s), round(im.height * s))) if s < 1 else im)
    return ims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nf4", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, "outputs/pilot_scale", a.tag); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    model, proc = load(a.model, a.nf4)
    meta = {"model": a.model, "quant": "nf4" if a.nf4 else "bf16", "load_sec": round(time.time() - t0, 1),
            "vram_after_load_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}
    print(meta, flush=True)
    od = os.path.join(out, "desc.jsonl")
    done = {json.loads(l)["key"] for l in open(od)} if os.path.exists(od) else set()
    keys = pd.read_csv(os.path.join(P, "pilot_clips.csv"), dtype=str).key.tolist()
    peak = 0.0
    with open(od, "a") as f:
        for k in keys:
            if k in done:
                continue
            d = os.path.join(P, "frames", k.replace("/", "__"))
            info = json.load(open(os.path.join(d, "frames.json")))
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
            text, n_tok = describe(model, proc, frames448(d), info["times"])
            torch.cuda.synchronize()
            pk = torch.cuda.max_memory_allocated() / 2**30; peak = max(peak, pk)
            f.write(json.dumps({"key": k, "desc": text, "prompt_tokens": n_tok, "sec": round(time.time() - t, 2),
                                "peak_gib": round(pk, 2)}, ensure_ascii=False) + "\n"); f.flush()
    meta["peak_gib"] = round(peak, 2)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print("done", meta)


if __name__ == "__main__":
    main()
