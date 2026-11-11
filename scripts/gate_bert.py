"""Conflict-aware reliability gate over text + AU evidence (CH-SIMS v2, official split).

Shared bert-base-chinese encoder, two passes:
  h_t = CLS(transcript)            -> y_t  (aux target: label_T)
  h_e = CLS(AU evidence text)      -> y_e  (aux target: label_V)
Gate input z = [h_t, h_e, |y_t - y_e|, reliability feats (face rate, mean FaceScore, AU signal strength, #salient AUs)]
  conflict head  c = sigmoid(MLP_c(z))           (aux target: sign(label_V) != sign(label_T))
  gate           g = sigmoid(MLP_g([z, c]))      weight on the evidence branch
  prediction     y = head([h_t, g * h_e, g * (y_e - y_t)])
Variants (--variant): gate (no aux), gate_aux (unimodal + conflict aux losses), nogate (g fixed to 1, late fusion).
Best epoch by valid MAE; metrics on full test + Conflict/Align; per-sample gate values saved for analysis.
Usage: python gate_bert.py --variant gate_aux --seed 1
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_bert import BERT, FULL, jl, metrics

AUX_W_UNI, AUX_W_CONF = 0.3, 0.2


def reliability_feats(keys):
    au = pd.read_csv(os.path.join(FULL, "au_frames.csv"))
    aucols = [c for c in au.columns if re.fullmatch(r"AU\d+", c)]
    au["au_max"] = au[aucols].max(axis=1)
    au["n_sal"] = (au[aucols] >= 0.5).sum(axis=1)
    g = au.groupby("key")
    f = pd.DataFrame({"face_rate": g.face.mean(),
                      "face_score": g.face_score.mean() if "face_score" in au else 0.0,
                      "au_strength": g.au_max.mean(), "n_salient": g.n_sal.mean() / 5.0}).fillna(0.0)
    return f.reindex(keys).fillna(0.0).values.astype(np.float32)


class GateModel(nn.Module):
    def __init__(self, variant, n_rel=4):
        super().__init__()
        self.variant = variant
        self.enc = AutoModel.from_pretrained(BERT)
        d = self.enc.config.hidden_size
        self.head_t, self.head_e = nn.Linear(d, 1), nn.Linear(d, 1)
        zin = 2 * d + 1 + n_rel
        self.conf = nn.Sequential(nn.Linear(zin, 128), nn.GELU(), nn.Linear(128, 1))
        self.gate = nn.Sequential(nn.Linear(zin + 1, 128), nn.GELU(), nn.Linear(128, 1))
        self.out = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * d + 1, 1))

    def cls(self, b):
        return self.enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        token_type_ids=b["token_type_ids"]).last_hidden_state[:, 0]

    def forward(self, bt, be, rel):
        ht, he = self.cls(bt), self.cls(be)
        yt, ye = self.head_t(ht).squeeze(-1), self.head_e(he).squeeze(-1)
        z = torch.cat([ht, he, (yt - ye).abs().unsqueeze(-1).detach(), rel], -1)
        c = self.conf(z).squeeze(-1)
        if self.variant == "nogate":
            g = torch.ones_like(yt)
        else:
            g = torch.sigmoid(self.gate(torch.cat([z, torch.sigmoid(c).unsqueeze(-1)], -1))).squeeze(-1)
        y = self.out(torch.cat([ht, g.unsqueeze(-1) * he, (g * (ye - yt)).unsqueeze(-1)], -1)).squeeze(-1)
        return y, yt, ye, c, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["gate", "gate_aux", "nogate"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--evidence", default="au", help="au = Face sentence of measure_text")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    out_dir = os.path.join(FULL, "results_gate"); os.makedirs(out_dir, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    if os.path.exists(os.path.join(out_dir, tag + ".json")):
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    vis = jl(os.path.join(FULL, "measure_text.jsonl"), "visual_text")
    face = lambda t: (re.search(r"Face: .*?(?= (Head|Gaze): |$)", t) or [""])[0] if t else ""
    df["evid"] = [face(vis.get(k, "")) or "Face: not detected." for k in df.key]
    rel = reliability_feats(df.key.tolist())
    df["conf_t"] = (np.sign(df.label_V) != np.sign(df.label_T)).astype(np.float32)
    tok = AutoTokenizer.from_pretrained(BERT)

    def enc(texts, n):
        e = tok(list(texts), truncation=True, max_length=n)
        return [{k: e[k][i] for k in ("input_ids", "attention_mask", "token_type_ids")} for i in range(len(texts))]

    def pad(items):
        L = max(len(x["input_ids"]) for x in items)
        return {k: torch.tensor([x[k] + [tok.pad_token_id if k == "input_ids" else 0] * (L - len(x[k])) for x in items])
                for k in ("input_ids", "attention_mask", "token_type_ids")}

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    ET, EE = enc(df.text, 128), enc(df.evid, 192)
    Y = torch.tensor(df.label.values, dtype=torch.float32)
    YT, YV = torch.tensor(df.label_T.values, dtype=torch.float32), torch.tensor(df.label_V.values, dtype=torch.float32)
    CT, R = torch.tensor(df.conf_t.values), torch.tensor(rel)

    def batches(idx, bs, shuffle, gen=None):
        idx = idx[torch.randperm(len(idx), generator=gen).numpy()] if shuffle else idx
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            yield j, pad([ET[x] for x in j]), pad([EE[x] for x in j])

    model = GateModel(a.variant).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)
    cu = lambda b: {k: v.cuda() for k, v in b.items()}

    @torch.no_grad()
    def predict(idx):
        model.eval(); outs = []
        for j, bt, be in batches(idx, 64, False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(cu(bt), cu(be), R[j].cuda())
            outs.append(torch.stack([x.float() for x in o], -1).cpu())
        model.train()
        return torch.cat(outs).numpy()  # columns: y, yt, ye, c(logit), g

    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        for j, bt, be in batches(sp["train"], a.bs, True, gen):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, yt, ye, c, g = model(cu(bt), cu(be), R[j].cuda())
            y, yt, ye, c = y.float(), yt.float(), ye.float(), c.float()
            loss = nn.functional.mse_loss(y, Y[j].cuda())
            if a.variant == "gate_aux":
                loss = loss + AUX_W_UNI * (nn.functional.mse_loss(yt, YT[j].cuda()) + nn.functional.mse_loss(ye, YV[j].cuda())) \
                    + AUX_W_CONF * nn.functional.binary_cross_entropy_with_logits(c, CT[j].cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        pv = predict(sp["valid"])
        vm = metrics(df.label.values[sp["valid"]], pv[:, 0]); hist.append(vm)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f}", flush=True)
    model.load_state_dict(state)
    pt = predict(sp["test"]); te = df.iloc[sp["test"]].reset_index(drop=True)
    res = {"variant": a.variant, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0, 1),
           "valid": hist[best_ep], "test": metrics(te.label, pt[:, 0])}
    for name, m in (("conflict", te.category == "conflict"), ("align", te.category.isin(["equal", "spne"]))):
        res[f"test_{name}"] = metrics(te.label[m], pt[m.values, 0])
    pd.DataFrame({"key": te.key, "label": te.label, "label_T": te.label_T, "label_V": te.label_V, "category": te.category,
                  "pred": pt[:, 0], "y_t": pt[:, 1], "y_e": pt[:, 2], "conflict_logit": pt[:, 3], "gate": pt[:, 4]}
                 ).to_csv(os.path.join(out_dir, tag + "_test_pred.csv"), index=False)
    json.dump(res, open(os.path.join(out_dir, tag + ".json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "test_align")}))


if __name__ == "__main__":
    main()
