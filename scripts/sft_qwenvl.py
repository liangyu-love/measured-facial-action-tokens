"""Small SFT ablation: QLoRA on a Qwen VL model (default Qwen3-VL-4B-Instruct), train split only, targets from sft_data.py.
Input format identical to zero-shot inference (gen_descriptions.describe): interleaved "Fi (t=..s):" + image, then
the v2 PROMPT; loss only on the assistant answer. LoRA on language-model projections only (vision tower frozen).
Output: models/sft3b_lora/ (adapter), outputs/sft3b/train_log.jsonl
"""
import json
import math
import os
import random
import re
import time

import torch
from PIL import Image

from gen_descriptions import PROMPT

ROOT = os.path.expanduser("~/work/msa-pilot")
F = os.path.join(ROOT, "outputs/full")
O = os.path.join(ROOT, os.environ.get("SFT_DIR", "outputs/sft_q3vl4b"))
MDIR = os.path.join(ROOT, "models", os.environ.get("SFT_MODEL", "Qwen3-VL-4B-Instruct"))
ADIR = os.path.join(ROOT, os.environ.get("SFT_ADAPTER", "models/sft_q3vl4b_lora"))
EPOCHS, ACCUM, LR, SEED = 1, 8, 1e-4, 42


def frames_of(key):
    d = os.path.join(F, "frames", key.replace("/", "__"))
    ims = []
    for i in range(1, 9):
        im = Image.open(os.path.join(d, f"F{i}.jpg")).convert("RGB")
        s = 448 / max(im.size)
        ims.append(im.resize((round(im.width * s), round(im.height * s))) if s < 1 else im)
    return ims


def messages(ims, times, answer=None):
    content = []
    for i, (im, t) in enumerate(zip(ims, times), 1):
        content += [{"type": "text", "text": f"F{i} (t={t}s):"}, {"type": "image", "image": im}]
    content.append({"type": "text", "text": PROMPT.format(n=len(ims))})
    m = [{"role": "user", "content": content}]
    if answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return m


def encode(proc, ex):
    ims = frames_of(ex["key"])
    full = proc.apply_chat_template(messages(ims, ex["times"], ex["target"]), tokenize=True, return_dict=True,
                                    return_tensors="pt")
    prompt = proc.apply_chat_template(messages(ims, ex["times"]), tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt")
    labels = full["input_ids"].clone()
    labels[:, : prompt["input_ids"].shape[1]] = -100
    full["labels"] = labels
    return full


def main():
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
    random.seed(SEED); torch.manual_seed(SEED)
    proc = AutoProcessor.from_pretrained(MDIR)
    model = AutoModelForImageTextToText.from_pretrained(
        MDIR, device_map="cuda", attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    names = [n for n, _ in model.named_modules()]
    lm = [n for n in names if re.search(r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$", n)
          and "visual" not in n]
    assert lm, "no language-model projection modules found"
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=lm, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train = [json.loads(l) for l in open(os.path.join(O, "train.jsonl"))]
    valid = [json.loads(l) for l in open(os.path.join(O, "valid.jsonl"))][:100]
    lim = int(os.environ.get("SFT_LIMIT", 0))  # smoke test: tiny run, adapter not saved
    if lim:
        train, valid = train[:lim], valid[:4]
    steps = math.ceil(len(train) * EPOCHS / ACCUM)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    log = open(os.path.join(O, "train_log.jsonl"), "w")

    @torch.no_grad()
    def vloss():
        model.eval(); tot = 0.0
        for ex in valid:
            b = {k: v.cuda() for k, v in encode(proc, ex).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot += model(**b).loss.item()
        model.train()
        return tot / len(valid)

    log.write(json.dumps({"step": 0, "valid_loss": vloss()}) + "\n"); log.flush()
    model.train(); t0 = time.time(); run, n = 0.0, 0
    for ep in range(EPOCHS):
        random.shuffle(train)
        for i, ex in enumerate(train, 1):
            b = {k: v.cuda() for k, v in encode(proc, ex).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss / ACCUM
            loss.backward(); run += loss.item() * ACCUM; n += 1
            if i % ACCUM == 0 or i == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            if i % 80 == 0:
                rec = {"ep": ep, "i": i, "train_loss": round(run / n, 4), "sec": round(time.time() - t0),
                       "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True); run, n = 0.0, 0
    log.write(json.dumps({"step": "end", "valid_loss": vloss(), "sec": round(time.time() - t0)}) + "\n"); log.close()
    if lim:
        print("smoke ok"); return
    model.save_pretrained(ADIR)
    print("saved", ADIR)


if __name__ == "__main__":
    main()
