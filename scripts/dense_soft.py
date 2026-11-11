"""Frame densification on CH-SIMS v2 (face-crop py-feat AUs + emotion probs along 10 fps tracks).
Per-frame vector = AUs + face flag + 7 emotion probs + 5 EMFACS products (same as au_soft+emo).
Variants (--variant):
  crop8    the original 8 sampled frames, face-cropped (control: better AU estimation only)
  seg8     10 fps frames split into 8 equal segments; per-segment mean ++ max over face frames (+ face fraction) -> 8 tokens
  dense16  0.25 s bins (mean); > 16 bins pooled into 16 groups, < 16 padded + masked -> up to 16 tokens
Baseline: 8 full-frame tokens au_soft+emo (results_acc/soft_emo). Same protocol (6 ep, lr 2e-5, bs 16, valid MAE).
Usage: python dense_soft.py --variant seg8 --seed 1
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

from train_bert import BERT, FULL, add_tune_args, loss_fn, make_opt, metrics, save_valid_only, stop_early

OUT = os.path.join(FULL, os.environ.get("DENSE_RESULTS_DIR", "results_dense"))
EMO = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
EMFACS = [("AU06", "AU12"), ("AU01", "AU04", "AU15"), ("AU04", "AU07"), ("AU09", "AU10"), ("AU01", "AU02", "AU05")]


def frame_vectors(d, aucols):
    a = d[aucols].fillna(0).values
    idx = {c: i for i, c in enumerate(aucols)}
    em = np.stack([np.prod([a[:, idx[c]] for c in combo], 0) for combo in EMFACS], -1)
    v = np.concatenate([a, d[["face"]].values, d[EMO].fillna(0).values, em], -1).astype(np.float32)
    v[d.face.values == 0] = 0.0  # no face -> zero vector (face flag 0)
    return v


def build(df, variant):
    pos = {k: i for i, k in enumerate(df.key)}
    if variant == "crop8":
        d = pd.read_csv(os.path.join(FULL, "crop8_au.csv"))
        aucols = sorted(c for c in d.columns if re.fullmatch(r"AU\d+", c))
        d = d[d.key.isin(pos)]
        X = np.zeros((len(df), 8, len(aucols) + 1 + len(EMO) + len(EMFACS)), np.float32)
        X[d.key.map(pos).values, d.frame.values - 1] = frame_vectors(d, aucols)
        return X, np.ones((len(df), 8), np.int64)
    d = pd.read_csv(os.path.join(FULL, "dense_au.csv"))
    aucols = sorted(c for c in d.columns if re.fullmatch(r"AU\d+", c))
    D = len(aucols) + 1 + len(EMO) + len(EMFACS)
    T = 8 if variant == "seg8" else 16
    X = np.zeros((len(df), T, 2 * D if variant == "seg8" else D), np.float32)
    Mk = np.zeros((len(df), T), np.int64)
    for k, g in d[d.key.isin(pos)].groupby("key"):
        g = g.sort_values("t"); v = frame_vectors(g, aucols); f = g.face.values.astype(bool); i = pos[k]
        if variant == "seg8":
            for s, ix in enumerate(np.array_split(np.arange(len(g)), 8)):
                ix = ix[f[ix]] if f[ix].any() else ix
                if len(ix):
                    X[i, s] = np.concatenate([v[ix].mean(0), v[ix].max(0)])
                Mk[i, s] = 1
        else:
            bins = np.floor(g.t.values / 0.25).astype(int)
            bvecs = [v[bins == b][f[bins == b]].mean(0) if f[bins == b].any() else np.zeros(D, np.float32) for b in np.unique(bins)]
            bvecs = np.stack(bvecs)
            if len(bvecs) > T:
                bvecs = np.stack([bvecs[ix].mean(0) for ix in np.array_split(np.arange(len(bvecs)), T)])
            X[i, :len(bvecs)] = bvecs; Mk[i, :len(bvecs)] = 1
    Mk[Mk.sum(1) == 0, 0] = 1  # never fully masked
    return X, Mk


class Model(nn.Module):
    def __init__(self, d_in, T):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BERT)
        d = self.enc.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
        self.fpos = nn.Parameter(torch.zeros(T, d)); nn.init.normal_(self.fpos, std=0.02)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, ids, att, tt, X, Mk):
        emb = self.enc.embeddings.word_embeddings(ids)
        soft = (self.proj(X) + self.fpos).to(emb.dtype)
        emb = torch.cat([emb, soft], 1)
        att = torch.cat([att, Mk.to(att.dtype)], 1)
        tt = torch.cat([tt, torch.ones_like(Mk, dtype=tt.dtype)], 1)
        return self.head(self.enc(inputs_embeds=emb, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0]).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["crop8", "seg8", "dense16"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    add_tune_args(ap)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    if os.path.exists(os.path.join(OUT, tag + ".json")):
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    X, Mk = build(df, a.variant)
    XT, MT = torch.tensor(X), torch.tensor(Mk)
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(list(df.text), truncation=True, max_length=128)
    Y = torch.tensor(df.label.values, dtype=torch.float32)

    def batch(idx):
        L = max(len(enc["input_ids"][i]) for i in idx)
        p = lambda k, v: torch.tensor([enc[k][i] + [v] * (L - len(enc[k][i])) for i in idx])
        return p("input_ids", tok.pad_token_id).cuda(), p("attention_mask", 0).cuda(), p("token_type_ids", 0).cuda(), \
            XT[idx].cuda(), MT[idx].cuda()

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    model = Model(X.shape[-1], X.shape[1]).cuda()
    opt = make_opt(model, a)
    steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def predict(idx):
        model.eval(); out = []
        for i in range(0, len(idx), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(*batch(idx[i:i + 64])).float().cpu())
        model.train()
        return torch.cat(out).numpy()

    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(*batch(j))
            loss = loss_fn(a)(y.float(), Y[j].cuda())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])); hist.append(vm)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f}", flush=True)
        if stop_early(a, ep, best_ep):
            break
    if a.valid_only:
        save_valid_only(a, os.path.join(OUT, tag + ".json"), hist, best_ep); print("valid", hist[best_ep]); return
    model.load_state_dict(state)
    pt = predict(sp["test"]); te = df.iloc[sp["test"]].reset_index(drop=True)
    res = {"variant": a.variant, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0, 1),
           "n_tokens_mean": float(Mk.sum(1).mean()), "valid": hist[best_ep], "test": metrics(te.label, pt)}
    for name, m in (("conflict", te.category == "conflict"), ("align", te.category.isin(["equal", "spne"]))):
        res[f"test_{name}"] = metrics(te.label[m], pt[m.values])
    pd.DataFrame({"key": te.key, "label": te.label, "category": te.category, "pred": pt}).to_csv(
        os.path.join(OUT, tag + "_test_pred.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "test_align")}))


if __name__ == "__main__":
    main()
