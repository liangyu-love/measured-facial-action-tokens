"""Paper-2 pilot C, part 2: same LLM backbone, but it SEES the 8 video frames (Qwen3-VL-4B with its vision tower, frozen; LoRA on the LLM).
  --variant vid      8 frames (each resized to 336x192 -> 77 visual tokens, 616 in total) + transcript
  --variant vid_au   same + the 8 measured face tokens (written into 8 reserved slot tokens via a hook on the embedding layer)
  --variant vid_exp  8 frames + transcript + ONE expert-opinion token: the [DIFF] input of llm_diff.py au_tri_trust (frozen cross-fitted
                     text / visual / speech expert readings + whom-to-trust head, aux weight 0.1), no measured face tokens.
                     Test conditions: full | shufD (expert readings permuted across test clips)
Same prompt, readout (last position -> linear head), loss and schedule as llm_pilot.py; batch 4 x 4 accumulation = effective 16.
Stage `prep` caches the resized frames once: outputs/full/frames336.npy (N, 8, 192, 336, 3) uint8.
Output: outputs/full/llm_pilot/<variant>_seed<s>.json + _test_pred.csv
Usage: python llm_vid.py --stage prep ;  python llm_vid.py --variant vid --seed 1
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from train_bert import FULL, metrics

MDIR = os.path.expanduser(os.environ.get("LLM_DIR", "~/work/msa-pilot/models/Qwen3-VL-4B-Instruct"))
OUT = os.path.join(FULL, os.environ.get("LLM_OUT", "llm_pilot"))
CACHE = os.path.join(FULL, "frames336.npy")
DEV = os.environ.get("LLM_DEV", "cuda")
PROMPT = "判断说话人的情感倾向，从负面到正面。\n话语："
IMG = "<|vision_start|><|image_pad|><|vision_end|>"
SLOT = "<|object_ref_start|>"           # reserved token; its embedding is replaced by a measured face token
NF, W, H = 8, 336, 192


def load_clip(key):
    d = os.path.join(FULL, "frames", key.replace("/", "__"))
    out = np.zeros((NF, H, W, 3), np.uint8)
    for i in range(NF):
        p = os.path.join(d, f"F{i + 1}.jpg")
        if os.path.exists(p):
            out[i] = np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BICUBIC))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="train", choices=["prep", "train"])
    ap.add_argument("--variant", choices=["vid", "vid_au", "vid_exp"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    if a.stage == "prep":
        if os.path.exists(CACHE):
            print("exists", CACHE); return
        from multiprocessing import Pool
        with Pool(12) as pool:
            arr = np.stack(pool.map(load_clip, list(df.key), chunksize=32))
        np.save(CACHE, arr); print("saved", arr.shape, "empty frames:", int((arr.reshape(len(df), NF, -1).max(-1) == 0).sum())); return

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, get_linear_schedule_with_warmup
    from au_soft2 import extra_inputs, flags
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    out_json = os.path.join(OUT, tag + ".json")
    smoke = int(os.environ.get("LLM_SMOKE", 0))
    if os.path.exists(out_json) and not smoke:
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    use_face, use_exp = a.variant == "vid_au", a.variant == "vid_exp"
    if use_exp:
        from llm_diff import diff_feats
        YT, YV = torch.tensor(df.label_T.values, dtype=torch.float32), torch.tensor(df.label_V.values, dtype=torch.float32)
        rt = pd.read_csv(os.path.join(FULL, "expert_oof", f"seed{a.seed}.csv")).set_index("key").reindex(df.key)
        ra = pd.read_csv(os.path.join(FULL, "expert_oof", f"audio_hubert_seed{a.seed}.csv")).set_index("key").reindex(df.key)
        R = torch.tensor(np.concatenate([rt[["yv_oof", "yt_oof", "sv_oof", "st_oof"]].values, ra[["ya", "sa"]].values], 1), dtype=torch.float32)
        assert not torch.isnan(R).any(), "missing expert readings"
    fr, _ = extra_inputs(df, flags("soft_emo"))
    FR = torch.tensor(np.ascontiguousarray(fr)); Y = torch.tensor(df.label.values, dtype=torch.float32)
    TRUST = (Y - YT).abs() - (Y - YV).abs() if use_exp else None
    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    if smoke:
        sp = {k: v[:smoke] for k, v in sp.items()}
    FRAMES = np.load(CACHE, mmap_mode="r") if os.path.exists(CACHE) else None
    proc = AutoProcessor.from_pretrained(MDIR); tok = proc.tokenizer; tok.padding_side = "left"
    slot_id = tok.convert_tokens_to_ids(SLOT)
    texts = [IMG * NF + PROMPT + t + ("\n表情：" + SLOT * NF if use_face else "") + ("\n参考：" + SLOT if use_exp else "") + "\n情感：" for t in df.text]
    dt = torch.bfloat16 if DEV == "cuda" else torch.float32
    full = AutoModelForImageTextToText.from_pretrained(MDIR, dtype=dt, device_map=DEV)
    base = full.model; del full.lm_head
    for p in base.parameters():
        p.requires_grad_(False)
    emb = base.language_model.embed_tokens; d = emb.embedding_dim
    with torch.no_grad():
        emb_norm = emb.weight[:20000].float().norm(dim=-1).mean().item()
    base = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"))

    class Extra(nn.Module):
        def __init__(self):
            super().__init__()
            if use_face:
                self.proj = nn.Sequential(nn.Linear(fr.shape[-1], d), nn.LayerNorm(d))
                self.fpos = nn.Parameter(torch.zeros(NF, d)); nn.init.normal_(self.fpos, std=0.02)
                self.scale = nn.Parameter(torch.tensor(emb_norm / d ** 0.5))
            if use_exp:
                self.htr = nn.Sequential(nn.Linear(15, 64), nn.GELU(), nn.Linear(64, 1))
                self.mproj = nn.Sequential(nn.Linear(16, d), nn.LayerNorm(d))
                self.scale = nn.Parameter(torch.tensor(emb_norm / d ** 0.5))
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1), nn.Linear(d, 1))

    ex = Extra().to(DEV); cur = {}

    def hook(_m, inp, out):                                  # write the measured face tokens into the reserved slots
        if "soft" in cur:
            m = inp[0] == slot_id
            if m.any():
                out = out.clone(); out[m] = cur["soft"].reshape(-1, d).to(out.dtype)
        return out
    emb.register_forward_hook(hook)

    def forward(idx, zero_face=False, r=None):
        imgs = [[Image.fromarray(np.asarray(FRAMES[i, f])) if FRAMES is not None else Image.fromarray(load_clip(df.key[i])[f]) for f in range(NF)] for i in idx]
        x = proc(text=[texts[i] for i in idx], images=imgs, return_tensors="pt", padding=True).to(DEV)
        x["pixel_values"] = x["pixel_values"].to(dt)
        if use_face:
            f = (torch.zeros(len(idx), *FR.shape[1:]) if zero_face else FR[idx]).to(DEV)
            cur["soft"] = (ex.proj(f) + ex.fpos) * ex.scale
        trp = None
        if use_exp:
            m = diff_feats((R[idx] if r is None else r).to(DEV), True)
            trp = ex.htr(m).squeeze(-1); cur["soft"] = ex.mproj(torch.cat([m, trp.unsqueeze(-1)], -1)) * ex.scale
        h = base(**x).last_hidden_state[:, -1]
        cur.pop("soft", None)
        return ex.head(h.float()).squeeze(-1), trp

    AC = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if DEV == "cuda" else (lambda: torch.autocast("cpu", enabled=False))
    lora = [p for n, p in base.named_parameters() if p.requires_grad]
    new = list(ex.parameters())
    print(f"[{tag}] trainable: lora {sum(p.numel() for p in lora) / 1e6:.1f}M, new {sum(p.numel() for p in new) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": new, "lr": 1e-3}], weight_decay=0.01)
    bs = int(os.environ.get("LLM_BS", 2)); acc = 16 // bs      # bs 4 ran out of memory on the 24GB card; effective batch stays 16
    if smoke:
        bs, acc = 2, 1
    steps = (len(sp["train"]) + bs * acc - 1) // (bs * acc) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def predict(idx, cond="full"):
        base.eval(); ex.eval(); o = []
        perm = idx[np.random.RandomState(0).permutation(len(idx))]
        for i in range(0, len(idx), 8):
            with AC():
                o.append(forward(idx[i:i + 8], cond == "noV", R[perm[i:i + 8]] if cond == "shufD" else None)[0].float().cpu())
        base.train(); ex.train(); return torch.cat(o).numpy()

    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for n, i in enumerate(range(0, len(perm), bs)):
            j = perm[i:i + bs]
            with AC():
                y, tr = forward(j)
            loss = nn.functional.l1_loss(y.float(), Y[j].to(DEV))
            if use_exp:
                loss = loss + 0.1 * nn.functional.mse_loss(tr.float(), TRUST[j].to(DEV))
            (loss / acc).backward()
            if (n + 1) % acc == 0 or i + bs >= len(perm):
                nn.utils.clip_grad_norm_(lora + new, 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            if n == 20:
                print(f"[{tag}] 20 steps in {time.time() - t0:.0f}s, max mem {torch.cuda.max_memory_allocated() / 2 ** 30 if DEV == 'cuda' else 0:.1f}G", flush=True)
        vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])); hist.append(vm)
        print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = ({n: p.detach().cpu().clone() for n, p in base.named_parameters() if p.requires_grad}, {k: v.detach().cpu().clone() for k, v in ex.state_dict().items()})
        if ep - best_ep >= 2:
            break
    with torch.no_grad():
        for n, p in base.named_parameters():
            if p.requires_grad:
                p.copy_(state[0][n].to(p.device, p.dtype))
    ex.load_state_dict(state[1])
    te = df.iloc[sp["test"]].reset_index(drop=True)
    res = {"variant": a.variant, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0), "valid": hist[best_ep], "test": {}}
    preds = te[["key", "label", "label_T", "label_V", "category", "conflict_mods"]].copy()
    conf = (te.category == "conflict").values
    for c in (["full", "noV"] if use_face else ["full", "shufD"] if use_exp else ["full"]):
        p = predict(sp["test"], c); preds["pred_" + c] = p
        res["test"][c] = {"all": metrics(te.label, p), "conflict": metrics(te.label[conf], p[conf]), "align": metrics(te.label[~conf], p[~conf])}
        print(f"[{tag}] {c:5s} test {res['test'][c]['all']} | conflict acc2={res['test'][c]['conflict']['acc2']:.4f}", flush=True)
    if not smoke:
        json.dump(res, open(out_json, "w"), indent=1)
        preds.to_csv(os.path.join(OUT, tag + "_test_pred.csv"), index=False)


if __name__ == "__main__":
    main()
