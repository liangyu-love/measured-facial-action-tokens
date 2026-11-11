"""Generate v2-prompt descriptions for all 4403 clips with Qwen2.5-VL-3B: --variant zs (zero-shot) or sft (LoRA merged).
Both in bf16 for comparability. Output: outputs/sft3b/<variant>/desc.jsonl (resumable)."""
import argparse
import json
import os
import time

import pandas as pd
import torch

from gen_descriptions import describe
from sft_qwen3b import ADIR, F, MDIR, O, frames_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["zs", "sft"])
    a = ap.parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MDIR, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                               device_map="cuda")
    if a.variant == "sft":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADIR).merge_and_unload()
    model.eval()
    proc = AutoProcessor.from_pretrained(MDIR)
    out = os.path.join(O, a.variant); os.makedirs(out, exist_ok=True)
    od = os.path.join(out, "desc.jsonl")
    done = {json.loads(l)["key"] for l in open(od)} if os.path.exists(od) else set()
    clips = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str)
    # test split first so the verifier evaluation can start early
    keys = clips.sort_values("split", key=lambda s: s.map({"test": 0, "valid": 1, "train": 2})).key.tolist()
    with open(od, "a") as f:
        for n, k in enumerate(keys, 1):
            if k in done:
                continue
            info = json.load(open(os.path.join(F, "frames", k.replace("/", "__"), "frames.json")))
            t = time.time()
            text, n_tok = describe(model, proc, frames_of(k), info["times"])
            f.write(json.dumps({"key": k, "desc": text, "sec": round(time.time() - t, 2),
                                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 500 == 0:
                print(f"{a.variant} {n}/{len(keys)}", flush=True)


if __name__ == "__main__":
    main()
