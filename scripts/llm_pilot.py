"""Paper-2 pilot C: measured facial action tokens on an LLM backbone (CH-SIMS v2).

Backbone: the language model of Qwen3-VL-4B-Instruct (vision tower dropped, no video/image input), LoRA r=16 on attention projections.
Input:  "<prompt>话语：{text}\n" [+ "表情：" + 8 measured face tokens] + "\n情感："  -> hidden state of the last position -> linear head.
The 8 face tokens are the same 33-d per-frame measurements as au_soft+emo (20 AU + face flag + 7 emotion probs + 5 EMFACS products),
projected by Linear + LayerNorm and rescaled to the LLM's token-embedding norm.
  --variant text   transcript only
  --variant au     transcript + measured face tokens   (also scored with the face tokens zeroed: cond noV)
Recipe: 5 epochs, bs 16, L1, AdamW (LoRA 1e-4, new modules 1e-3), 10% warmup, best epoch on valid MAE (patience 2).
Output: outputs/full/llm_pilot/<variant>_seed<s>.json + _test_pred.csv.   Usage: python llm_pilot.py --variant au --seed 1
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer, get_linear_schedule_with_warmup

from au_soft import NF
from au_soft2 import extra_inputs, flags
from train_bert import FULL, metrics

MDIR = os.path.expanduser(os.environ.get("LLM_DIR", "~/work/msa-pilot/models/Qwen3-VL-4B-Instruct"))
OUT = os.path.join(FULL, os.environ.get("LLM_OUT", "llm_pilot"))
AC = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
PROMPT = "判断说话人的情感倾向，从负面到正面。\n话语："


class Net(nn.Module):
    def __init__(self, lm, emb, d_frame, use_face, emb_norm):
        super().__init__()
        self.lm, self.emb, self.use_face = lm, emb, use_face
        d = emb.embedding_dim
        if use_face:
            self.proj = nn.Sequential(nn.Linear(d_frame, d), nn.LayerNorm(d))
            self.fpos = nn.Parameter(torch.zeros(NF, d)); nn.init.normal_(self.fpos, std=0.02)
            self.scale = nn.Parameter(torch.tensor(emb_norm / d ** 0.5))      # LayerNorm output has norm ~ sqrt(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, pre_ids, pre_att, suf_ids, frames):
        B = pre_ids.size(0)
        parts, masks = [self.emb(pre_ids)], [pre_att]
        if self.use_face:
            soft = ((self.proj(frames) + self.fpos) * self.scale).to(parts[0].dtype)
            parts.append(soft); masks.append(torch.ones(B, NF, dtype=pre_att.dtype, device=pre_att.device))
        parts.append(self.emb(suf_ids).unsqueeze(0).expand(B, -1, -1))
        masks.append(torch.ones(B, suf_ids.size(0), dtype=pre_att.dtype, device=pre_att.device))
        h = self.lm(inputs_embeds=torch.cat(parts, 1), attention_mask=torch.cat(masks, 1)).last_hidden_state[:, -1]
        return self.head(h.float()).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["text", "au"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    out_json = os.path.join(OUT, tag + ".json")
    smoke = int(os.environ.get("LLM_SMOKE", 0))
    if os.path.exists(out_json) and not smoke:
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    fr, _ = extra_inputs(df, flags("soft_emo"))
    FR = torch.tensor(np.ascontiguousarray(fr)); Y = torch.tensor(df.label.values, dtype=torch.float32)
    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    if smoke:
        sp = {k: v[:smoke] for k, v in sp.items()}
    tok = AutoTokenizer.from_pretrained(MDIR); tok.padding_side = "left"
    use_face = a.variant == "au"
    pre = [PROMPT + t + ("\n表情：" if use_face else "") for t in df.text]
    suf_ids = torch.tensor(tok("\n情感：", add_special_tokens=False).input_ids).cuda()

    full = AutoModelForImageTextToText.from_pretrained(MDIR, dtype=torch.bfloat16, device_map="cuda")
    lm = full.model.language_model; emb = lm.embed_tokens
    del full.model.visual, full.lm_head; torch.cuda.empty_cache()
    for p in lm.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        emb_norm = emb.weight[:20000].float().norm(dim=-1).mean().item()
    lm = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model = Net(lm, emb, fr.shape[-1], use_face, emb_norm).cuda()
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    new = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    print(f"[{tag}] trainable: lora {sum(p.numel() for p in lora) / 1e6:.1f}M, new {sum(p.numel() for p in new) / 1e6:.2f}M; emb norm {emb_norm:.3f}", flush=True)
    opt = torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": new, "lr": 1e-3}], weight_decay=0.01)
    bs = 16; steps = (len(sp["train"]) + bs - 1) // bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); gen = torch.Generator().manual_seed(a.seed)

    def batch(idx, zero_face=False):
        x = tok([pre[i] for i in idx], return_tensors="pt", padding=True)
        f = torch.zeros(len(idx), *FR.shape[1:]) if zero_face else FR[idx]
        return x.input_ids.cuda(), x.attention_mask.cuda(), suf_ids, f.cuda()

    @torch.no_grad()
    def predict(idx, zero_face=False):
        model.eval(); o = []
        for i in range(0, len(idx), 64):
            with AC():
                o.append(model(*batch(idx[i:i + 64], zero_face)).float().cpu())
        model.train(); return torch.cat(o).numpy()

    train_keys = [n for n, p in model.named_parameters() if p.requires_grad]
    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            with AC():
                y = model(*batch(j))
            loss = nn.functional.l1_loss(y.float(), Y[j].cuda())
            loss.backward(); nn.utils.clip_grad_norm_(lora + new, 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])); hist.append(vm)
        print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            sd = model.state_dict(); state = {k: sd[k].detach().cpu().clone() for k in train_keys}
        if ep - best_ep >= 2:
            break
    model.load_state_dict(state, strict=False)
    te = df.iloc[sp["test"]].reset_index(drop=True)
    res = {"variant": a.variant, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0), "valid": hist[best_ep], "test": {}}
    preds = te[["key", "label", "label_T", "label_V", "category", "conflict_mods"]].copy()
    conf = (te.category == "conflict").values
    for c in (["full", "noV"] if use_face else ["full"]):
        p = predict(sp["test"], zero_face=(c == "noV")); preds["pred_" + c] = p
        res["test"][c] = {"all": metrics(te.label, p), "conflict": metrics(te.label[conf], p[conf]), "align": metrics(te.label[~conf], p[~conf])}
        print(f"[{tag}] {c:5s} test {res['test'][c]['all']} | conflict acc2={res['test'][c]['conflict']['acc2']:.4f}", flush=True)
    if not smoke:
        json.dump(res, open(out_json, "w"), indent=1)
        preds.to_csv(os.path.join(OUT, tag + "_test_pred.csv"), index=False)
        va = df.iloc[sp["valid"]]                               # valid predictions: late-average weights are chosen on valid only
        pd.DataFrame({"key": va.key.values, "label": va.label.values, "pred_full": predict(sp["valid"])}).to_csv(os.path.join(OUT, tag + "_valid_pred.csv"), index=False)


if __name__ == "__main__":
    main()
