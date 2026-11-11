"""Conflict gate on AU soft tokens (bert-base-chinese).
Gate input z = [mean of transcript input embeddings (layer-0), 8-frame mean of the raw AU vector, reliability feats
(face rate, mean FaceScore, AU strength, #salient)].
  soft_gate      scalar g = sigmoid(MLP(z)) multiplies the 8 AU soft-token input embeddings
  soft_gate_aux  + BCE(g, conflict) with conflict = sign(label_T) != sign(label_V) (train split), weight 0.1
  soft_tokgate   per-frame g_f = sigmoid(MLP([z, AU vector of frame f])) + the same aux loss (on every g_f)
Gate bias initialised to +2 (g ~ 0.88). Same protocol as au_soft (6 ep, lr 2e-5, bs 16, best epoch by valid MAE).
Usage: python gate_soft.py --variant soft_gate_aux --seed 1
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from au_soft import NF, build_inputs
from gate_bert import reliability_feats
from train_bert import BERT, FULL, metrics

AUX_W = 0.1


class GateSoft(nn.Module):
    def __init__(self, variant, d_frame, n_rel=4):
        super().__init__()
        self.variant = variant
        self.enc = AutoModel.from_pretrained(BERT)
        d = self.enc.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(d_frame, d), nn.LayerNorm(d))
        self.fpos = nn.Parameter(torch.randn(NF, d) * 0.02)
        zin = d + d_frame + n_rel + (d_frame if variant == "soft_tokgate" else 0)
        self.gate = nn.Sequential(nn.Linear(zin, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.constant_(self.gate[-1].bias, 2.0)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, ids, att, tt, frames, rel):
        emb = self.enc.embeddings.word_embeddings(ids)
        m = att.unsqueeze(-1).to(emb.dtype)
        tmean = (emb * m).sum(1) / m.sum(1).clamp(min=1)            # layer-0 transcript summary
        z = torch.cat([tmean.float(), frames.mean(1), rel], -1)
        if self.variant == "soft_tokgate":
            zf = torch.cat([z.unsqueeze(1).expand(-1, NF, -1), frames], -1)
            glog = self.gate(zf).squeeze(-1)                         # (B, 8)
            g = torch.sigmoid(glog).unsqueeze(-1)
        else:
            glog = self.gate(z).squeeze(-1)                          # (B,)
            g = torch.sigmoid(glog)[:, None, None]
        soft = ((self.proj(frames) + self.fpos) * g).to(emb.dtype)
        emb = torch.cat([emb, soft], 1)
        att = torch.cat([att, torch.ones(att.size(0), NF, dtype=att.dtype, device=att.device)], 1)
        tt = torch.cat([tt, torch.ones(tt.size(0), NF, dtype=tt.dtype, device=tt.device)], 1)
        h = self.enc(inputs_embeds=emb, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0]
        return self.head(h).squeeze(-1), glog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["soft_gate", "soft_gate_aux", "soft_tokgate"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    out_dir = os.path.join(FULL, "results_softgate"); os.makedirs(out_dir, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    if os.path.exists(os.path.join(out_dir, tag + ".json")):
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    frames, _, _ = build_inputs(df, "soft")
    FR = torch.tensor(frames)
    rel = reliability_feats(df.key.tolist())
    tr = (df.split == "train").values
    rel = (rel - rel[tr].mean(0)) / (rel[tr].std(0) + 1e-6)
    R = torch.tensor(rel.astype(np.float32))
    CT = torch.tensor((np.sign(df.label_T) != np.sign(df.label_V)).astype(np.float32).values)
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(list(df.text), truncation=True, max_length=128)
    Y = torch.tensor(df.label.values, dtype=torch.float32)

    def batch(idx):
        L = max(len(enc["input_ids"][i]) for i in idx)
        p = lambda k, v: torch.tensor([enc[k][i] + [v] * (L - len(enc[k][i])) for i in idx])
        return p("input_ids", tok.pad_token_id).cuda(), p("attention_mask", 0).cuda(), p("token_type_ids", 0).cuda(), \
            FR[idx].cuda(), R[idx].cuda()

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    model = GateSoft(a.variant, frames.shape[-1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def predict(idx):
        model.eval(); Ys, Gs = [], []
        for i in range(0, len(idx), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, gl = model(*batch(idx[i:i + 64]))
            Ys.append(y.float().cpu()); Gs.append(torch.sigmoid(gl.float()).reshape(len(y), -1).mean(1).cpu())
        model.train()
        return torch.cat(Ys).numpy(), torch.cat(Gs).numpy()

    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, gl = model(*batch(j))
            loss = nn.functional.mse_loss(y.float(), Y[j].cuda())
            if a.variant != "soft_gate":
                tgt = CT[j].cuda()
                tgt = tgt.unsqueeze(-1).expand_as(gl) if gl.dim() == 2 else tgt
                loss = loss + AUX_W * nn.functional.binary_cross_entropy_with_logits(gl.float(), tgt)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])[0]); hist.append(vm)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f}", flush=True)
    model.load_state_dict(state)
    pt, gt = predict(sp["test"]); te = df.iloc[sp["test"]].reset_index(drop=True)
    res = {"variant": a.variant, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0, 1),
           "valid": hist[best_ep], "test": metrics(te.label, pt)}
    for name, m in (("conflict", te.category == "conflict"), ("align", te.category.isin(["equal", "spne"]))):
        res[f"test_{name}"] = metrics(te.label[m], pt[m.values])
    pd.DataFrame({"key": te.key, "label": te.label, "label_T": te.label_T, "label_V": te.label_V,
                  "category": te.category, "pred": pt, "gate": gt}).to_csv(os.path.join(out_dir, tag + "_test_pred.csv"), index=False)
    json.dump(res, open(os.path.join(out_dir, tag + ".json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "test_align")}))


if __name__ == "__main__":
    main()
