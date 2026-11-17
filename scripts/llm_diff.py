"""Paper-2 pilot C, part 3: the paper-1 analysis modules (expert-anchored discrepancy token, whom-to-trust head) on the LLM backbone.

Base = llm_pilot.py `au` (Qwen3-VL-4B language model, LoRA, transcript + 8 measured face tokens). One extra [DIFF] token is appended
after the face tokens. Its input is the same scalar vector as diff3.py --precomp oof, from the frozen, cross-fitted unimodal experts
(outputs/full/expert_oof/seed<s>.csv, audio_hubert_seed<s>.csv; train clips use out-of-fold readings):
  au_diff        [yv-yt, |yv-yt|, yv, yt, sign agree, sv, st]
  au_tri         + speech expert: [ya, sa, ya-yt, |ya-yt|, sign agree, yv-ya, |yv-ya|, sign agree]
  au_tri_trust   + whom-to-trust head on that vector (target |y-y_T| - |y-y_V| from the true unimodal labels, train only, weight 0.1),
                 its prediction is fed back as one more [DIFF] feature
  au_spk         control without experts: the 8 face tokens + ONE speech token (frozen HuBERT clip feature, learned layer mix), no [DIFF]
--exp m            experts trained on the OVERALL label only (expert_oof_m/, oof_experts_m.py); the whom-to-trust target is then
                   |y - yt_reading| - |y - yv_reading| (train, out-of-fold), so no unimodal label is used anywhere. Tag suffix _m.
--vctl zero|shuf   capacity control of paper 1: visual expert reading (and its std) zeroed / shuffled across clips
Test conditions: full | noV (face tokens zeroed, [DIFF] intact) | shufD ([DIFF] input permuted across test clips)
Output: outputs/full/llm_pilot/<variant>[_v0|_vs]_seed<s>.json + _test_pred.csv.  Usage: python llm_diff.py --variant au_tri_trust --seed 1
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
DEV = os.environ.get("LLM_DEV", "cuda")
PROMPT = "判断说话人的情感倾向，从负面到正面。\n话语："


def diff_feats(R, audio):
    """R: (B, 6) = [yv, yt, sv, st, ya, sa] -> scalar DIFF input (same order as diff3.py precomp)."""
    yv, yt, sv, st, ya, sa = R.unbind(-1)
    f = [yv - yt, (yv - yt).abs(), yv, yt, ((yv * yt) > 0).float(), sv, st]
    if audio:
        f += [ya, sa, ya - yt, (ya - yt).abs(), ((ya * yt) > 0).float(), yv - ya, (yv - ya).abs(), ((yv * ya) > 0).float()]
    return torch.stack(f, -1)


class Net(nn.Module):
    def __init__(self, lm, emb, d_frame, emb_norm, audio, trust, spk=False, res=False, tri=False):
        super().__init__()
        self.lm, self.emb, self.audio, self.trust, self.spk, self.res, self.tri = lm, emb, audio, trust, spk, res, tri
        d = emb.embedding_dim; k = 15 if audio else 7
        self.proj = nn.Sequential(nn.Linear(d_frame, d), nn.LayerNorm(d))
        self.fpos = nn.Parameter(torch.zeros(NF, d)); nn.init.normal_(self.fpos, std=0.02)
        self.scale = nn.Parameter(torch.tensor(emb_norm / d ** 0.5))
        nt = 3 if tri else (1 if trust else 0)                  # tri: which of text / face / speech expert is closest to the truth
        if nt:
            self.htr = nn.Sequential(nn.Linear(k, 64), nn.GELU(), nn.Linear(64, nt))
        if spk:
            self.lw = nn.Parameter(torch.zeros(13)); self.aproj = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d))
        else:
            self.mproj = nn.Sequential(nn.Linear(k + nt, d), nn.LayerNorm(d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1), nn.Linear(d, 1))
        if res:                                                 # residual on the expert mean: the model starts exactly at the plain expert average
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, pre_ids, pre_att, suf_ids, frames, R, aud=None):
        B = pre_ids.size(0); trp = None
        if self.spk:
            extra = self.aproj((torch.softmax(self.lw, 0)[None, :, None] * aud).sum(1))
        else:
            m = diff_feats(R, self.audio)
            if self.tri:
                trp = self.htr(m); m = torch.cat([m, torch.softmax(trp, -1)], -1)
            elif self.trust:
                trp = self.htr(m).squeeze(-1); m = torch.cat([m, trp.unsqueeze(-1)], -1)
            extra = self.mproj(m)
        e0 = self.emb(pre_ids)
        soft = (torch.cat([self.proj(frames) + self.fpos, extra.unsqueeze(1)], 1) * self.scale).to(e0.dtype)
        ones = lambda n: torch.ones(B, n, dtype=pre_att.dtype, device=pre_att.device)
        x = torch.cat([e0, soft, self.emb(suf_ids).unsqueeze(0).expand(B, -1, -1)], 1)
        att = torch.cat([pre_att, ones(NF + 1), ones(suf_ids.size(0))], 1)
        h = self.lm(inputs_embeds=x, attention_mask=att).last_hidden_state[:, -1]
        y = self.head(h.float()).squeeze(-1)
        if self.res:
            y = y + R[:, [1, 0, 4]].float().mean(1)
        return y, trp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["au_diff", "au_tri", "au_tri_trust", "au_spk", "au_res", "au_adj"])
    ap.add_argument("--exp", default="", choices=["", "m"])
    ap.add_argument("--vctl", default="", choices=["", "zero", "shuf"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.variant}{ {'': '', 'zero': '_v0', 'shuf': '_vs'}[a.vctl] }{'_m' if a.exp else ''}_seed{a.seed}"
    out_json = os.path.join(OUT, tag + ".json")
    smoke = int(os.environ.get("LLM_SMOKE", 0))
    if os.path.exists(out_json) and not smoke:
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    audio, trust, spk = a.variant != "au_diff", a.variant == "au_tri_trust", a.variant == "au_spk"
    res_, tri = a.variant in ("au_res", "au_adj"), a.variant == "au_adj"
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    fr, _ = extra_inputs(df, flags("soft_emo"))
    FR = torch.tensor(np.ascontiguousarray(fr)); Y = torch.tensor(df.label.values, dtype=torch.float32)
    YT, YV = torch.tensor(df.label_T.values, dtype=torch.float32), torch.tensor(df.label_V.values, dtype=torch.float32)
    TRUST = (Y - YT).abs() - (Y - YV).abs()                     # > 0: overall label closer to the face label than to the text label
    rt = pd.read_csv(os.path.join(FULL, "expert_oof" + ("_m" if a.exp else ""), f"seed{a.seed}.csv")).set_index("key").reindex(df.key)
    ra = rt if a.exp else pd.read_csv(os.path.join(FULL, "expert_oof", f"audio_hubert_seed{a.seed}.csv")).set_index("key").reindex(df.key)
    R = torch.tensor(np.concatenate([rt[["yv_oof", "yt_oof", "sv_oof", "st_oof"]].values, ra[["ya", "sa"]].values], 1), dtype=torch.float32)
    assert not torch.isnan(R).any(), "missing expert readings"
    if a.exp:                                                   # no unimodal labels: trust target from the expert readings themselves
        TRUST = (Y - R[:, 1]).abs() - (Y - R[:, 0]).abs()
    HUB = None
    if spk:
        HUB = torch.tensor(np.load(os.path.join(FULL, "hubert_layers.npy")).astype(np.float32)); tr_ = df.index[df.split == "train"].values
        HUB = ((HUB - HUB[tr_].mean(0)) / (HUB[tr_].std(0) + 1e-5)).clamp(-6, 6)
    if a.vctl == "zero":
        R[:, [0, 2]] = 0.0
    elif a.vctl == "shuf":
        R[:, [0, 2]] = R[torch.randperm(len(R), generator=torch.Generator().manual_seed(0))][:, [0, 2]]
    TRI = (Y[:, None] - R[:, [1, 0, 4]]).abs().argmin(1)        # 0 text / 1 face / 2 speech expert closest to the overall label (no unimodal label)
    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    if smoke:
        sp = {k: v[:smoke] for k, v in sp.items()}
    tok = AutoTokenizer.from_pretrained(MDIR); tok.padding_side = "left"
    pre = [PROMPT + t + "\n表情：" for t in df.text]
    suf_ids = torch.tensor(tok("\n情感：", add_special_tokens=False).input_ids).to(DEV)
    dt = torch.bfloat16 if DEV == "cuda" else torch.float32
    full = AutoModelForImageTextToText.from_pretrained(MDIR, dtype=dt, device_map=DEV)
    lm = full.model.language_model; emb = lm.embed_tokens
    del full.model.visual, full.lm_head
    for p in lm.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        emb_norm = emb.weight[:20000].float().norm(dim=-1).mean().item()
    lm = get_peft_model(lm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model = Net(lm, emb, fr.shape[-1], emb_norm, audio, trust, spk, res_, tri).to(DEV)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    new = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    opt = torch.optim.AdamW([{"params": lora, "lr": 1e-4}, {"params": new, "lr": 1e-3}], weight_decay=0.01)
    bs = 16; steps = (len(sp["train"]) + bs - 1) // bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); gen = torch.Generator().manual_seed(a.seed)
    AC = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if DEV == "cuda" else (lambda: torch.autocast("cpu", enabled=False))

    def batch(idx, zero_face=False, r=None):
        x = tok([pre[i] for i in idx], return_tensors="pt", padding=True)
        f = torch.zeros(len(idx), *FR.shape[1:]) if zero_face else FR[idx]
        return x.input_ids.to(DEV), x.attention_mask.to(DEV), suf_ids, f.to(DEV), (R[idx] if r is None else r).to(DEV), (HUB[idx].to(DEV) if spk else None)

    @torch.no_grad()
    def predict(idx, cond="full"):
        model.eval(); o, t = [], []
        perm = idx[np.random.RandomState(0).permutation(len(idx))]
        for i in range(0, len(idx), 64):
            with AC():
                y, tr = model(*batch(idx[i:i + 64], zero_face=(cond == "noV"), r=R[perm[i:i + 64]] if cond == "shufD" else None))
            if tri:
                tr = tr.argmax(-1)
            o.append(y.float().cpu()); t.append(tr.float().cpu() if tr is not None else torch.zeros(len(y)))
        model.train(); return torch.cat(o).numpy(), torch.cat(t).numpy()

    train_keys = [n for n, p in model.named_parameters() if p.requires_grad]
    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            with AC():
                y, tr = model(*batch(j))
            loss = nn.functional.l1_loss(y.float(), Y[j].to(DEV))
            if trust:
                loss = loss + 0.1 * nn.functional.mse_loss(tr.float(), TRUST[j].to(DEV))
            if tri:
                loss = loss + 0.1 * nn.functional.cross_entropy(tr.float(), TRI[j].to(DEV))
            loss.backward(); nn.utils.clip_grad_norm_(lora + new, 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])[0]); hist.append(vm)
        print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            sd = model.state_dict(); state = {k: sd[k].detach().cpu().clone() for k in train_keys}
        if ep - best_ep >= 2:
            break
    model.load_state_dict(state, strict=False)
    te = df.iloc[sp["test"]].reset_index(drop=True)
    res = {"variant": a.variant, "vctl": a.vctl, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0), "valid": hist[best_ep], "test": {}}
    preds = te[["key", "label", "label_T", "label_V", "category", "conflict_mods"]].copy()
    conf = (te.category == "conflict").values
    for c in ("full", "noV", "shufD"):
        p, tr = predict(sp["test"], c); preds["pred_" + c] = p
        res["test"][c] = {"all": metrics(te.label, p), "conflict": metrics(te.label[conf], p[conf]), "align": metrics(te.label[~conf], p[~conf])}
        print(f"[{tag}] {c:5s} test {res['test'][c]['all']} | conflict acc2={res['test'][c]['conflict']['acc2']:.4f}", flush=True)
        if c == "full" and trust and len(te) > 2:
            res["trust_corr_test"] = float(np.corrcoef(tr, TRUST[sp["test"]].numpy())[0, 1])
        if c == "full" and tri:
            res["trust_acc_test"] = float((tr == TRI[sp["test"]].numpy()).mean()); preds["trust_pred"] = tr
            print(f"[{tag}] closest-expert head acc {res['trust_acc_test']:.3f} (class share {np.bincount(TRI[sp['test']].numpy(), minlength=3) / len(te)})", flush=True)
    for nm, col in (("rt", 1), ("rv", 0), ("ra", 4)):           # expert readings, for the late-average baseline
        preds[nm] = R[sp["test"], col].numpy()
    va = df.iloc[sp["valid"]]
    vpred = pd.DataFrame({"key": va.key.values, "label": va.label.values, "pred_full": predict(sp["valid"])[0],
                          "rt": R[sp["valid"], 1].numpy(), "rv": R[sp["valid"], 0].numpy(), "ra": R[sp["valid"], 4].numpy()})
    if not smoke:
        json.dump(res, open(out_json, "w"), indent=1)
        preds.to_csv(os.path.join(OUT, tag + "_test_pred.csv"), index=False)
        vpred.to_csv(os.path.join(OUT, tag + "_valid_pred.csv"), index=False)


if __name__ == "__main__":
    main()
