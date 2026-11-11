"""Cross-fitted (out-of-fold) unimodal expert readings for CH-SIMS v2 (fixes stacking leakage).
For seed s: 5-fold GroupKFold over the TRAIN split (groups = source video). For each fold, a visual expert (evq.py recipe:
raw8emo, 2 layers, AdamW 1e-3, 40 epochs, best on valid MAE) and a text expert (texp.py recipe: BERT, AdamW 2e-5, 6 epochs,
warmup 10%, best on valid MAE) are trained on the other 4 folds and predict the held-out fold.
Valid/test readings come from the existing full-train experts (results_evq / results_texp, same seed).
Also writes in-sample readings (full experts on every clip) for the control. Each reading = mean and MC-dropout std (8 passes).
Output: outputs/full/expert_oof/seed<s>.csv  (key, yv_oof, yt_oof, sv_oof, st_oof, yv_in, yt_in, sv_in, st_in)
Usage: python oof_experts.py --seed 1"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from transformers import get_linear_schedule_with_warmup

from au_soft2 import extra_inputs
from evq import VisOnly
from texp import TextExpert, tokenize
from train_bert import FULL

OUT = os.path.join(FULL, "expert_oof")


def train_visual(X, Y, tr, va, seed):
    torch.manual_seed(seed); m = VisOnly(X.shape[-1], X.shape[1]).cuda(); M = torch.ones(X.shape[:2], dtype=torch.long)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4); g = torch.Generator().manual_seed(seed); best, state = 1e9, None
    for ep in range(40):
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            loss = nn.functional.mse_loss(m(X[j].cuda(), M[j].cuda()), Y[j].cuda()); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        m.eval()
        with torch.no_grad():
            v = (m(X[va].cuda(), M[va].cuda()).cpu() - Y[va]).abs().mean().item()
        m.train()
        if v < best:
            best, state = v, {k: x.detach().clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state); return m.eval()


def train_text(batch, Y, tr, va, seed, epochs=6, drop=0.1):
    torch.manual_seed(seed); m = TextExpert(drop).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
    steps = (len(tr) + 15) // 16 * epochs; sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    g = torch.Generator().manual_seed(seed); best, state = 1e9, None
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), 16):
            j = perm[i:i + 16]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(*batch(j))
            loss = nn.functional.mse_loss(y.float(), Y[j].cuda()); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        p = predict_text(m, batch, va, mc=False)[0]
        v = np.abs(p - Y[va].numpy()).mean()
        if v < best:
            best, state = v, {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state); return m.eval()


@torch.no_grad()
def predict_visual(m, X, idx):
    M = torch.ones(len(idx), X.shape[1], dtype=torch.long).cuda(); x = X[idx].cuda()
    m.eval(); y = m(x, M).cpu().numpy()
    m.train(); s = torch.stack([m(x, M) for _ in range(8)]).std(0).cpu().numpy(); m.eval()
    return y, s


@torch.no_grad()
def predict_text(m, batch, idx, mc=True):
    ys, ss = [], []
    for i in range(0, len(idx), 64):
        b = batch(idx[i:i + 64])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.eval(); ys.append(m(*b).float().cpu())
            if mc:
                m.train(); ss.append(torch.stack([m(*b).float() for _ in range(8)]).std(0).cpu()); m.eval()
    return torch.cat(ys).numpy(), (torch.cat(ss).numpy() if mc else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); out = os.path.join(OUT, f"seed{a.seed}.csv")
    if os.path.exists(out):
        print("exists"); return
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    fr, _ = extra_inputs(df, {"stat": False, "emo": True, "diff": False, "vaux": 0.0, "taux": 0.0, "pre": None})
    X = torch.tensor(fr); YV = torch.tensor(df.label_V.values, dtype=torch.float32); YT = torch.tensor(df.label_T.values, dtype=torch.float32)
    batch = tokenize(df)
    tr = df.index[df.split == "train"].values; va = df.index[df.split == "valid"].values; allix = df.index.values
    res = pd.DataFrame({"key": df.key})
    for c in ("yv_oof", "yt_oof", "sv_oof", "st_oof"):
        res[c] = np.nan
    for k, (a_, b_) in enumerate(GroupKFold(5).split(tr, groups=df.video_id.values[tr])):
        fit, hold = tr[a_], tr[b_]
        vm = train_visual(X, YV, fit, va, a.seed * 100 + k); y, s = predict_visual(vm, X, hold)
        res.loc[hold, "yv_oof"], res.loc[hold, "sv_oof"] = y, s
        tm = train_text(batch, YT, fit, va, a.seed * 100 + k); y, s = predict_text(tm, batch, hold)
        res.loc[hold, "yt_oof"], res.loc[hold, "st_oof"] = y, s
        del tm; torch.cuda.empty_cache(); print(f"seed {a.seed} fold {k} done", flush=True)
    vck = torch.load(os.path.join(FULL, "results_evq", f"raw8emo_seed{a.seed}.pt"), map_location="cpu", weights_only=False)
    vfull = VisOnly(vck["d_in"], vck["T"], vck.get("layers", 2)); vfull.load_state_dict(vck["state"], strict=False); vfull.cuda().eval()
    tfull = TextExpert(); tfull.load_state_dict(torch.load(os.path.join(FULL, "results_texp", f"seed{a.seed}.pt"), map_location="cpu"), strict=False); tfull.cuda().eval()
    yv, sv = predict_visual(vfull, X, allix); yt, st = predict_text(tfull, batch, allix)
    res["yv_in"], res["sv_in"], res["yt_in"], res["st_in"] = yv, sv, yt, st
    nt = df.split != "train"  # valid/test: full-train experts in both versions
    for c in ("yv", "yt", "sv", "st"):
        res.loc[nt, f"{c}_oof"] = res.loc[nt, f"{c}_in"]
    res.to_csv(out, index=False)
    from scipy.stats import pearsonr
    trm = (df.split == "train").values
    print(f"seed {a.seed}: TRAIN Corr OOF visual {pearsonr(res.yv_oof[trm], df.label_V[trm])[0]:.3f} text {pearsonr(res.yt_oof[trm], df.label_T[trm])[0]:.3f} | "
          f"in-sample visual {pearsonr(res.yv_in[trm], df.label_V[trm])[0]:.3f} text {pearsonr(res.yt_in[trm], df.label_T[trm])[0]:.3f}")


if __name__ == "__main__":
    main()
