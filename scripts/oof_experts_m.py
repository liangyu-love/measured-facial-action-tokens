"""Unimodal experts WITHOUT unimodal labels (CH-SIMS v2): same three experts and recipes as oof_experts.py / audio_expert_hubert.py
(visual: VisOnly on the 8 measured frames; text: BERT; speech: frozen HuBERT + MLP), but every expert is trained on the OVERALL label.
Train clips get 5-fold out-of-fold readings (GroupKFold by video); valid/test readings come from an expert trained on the whole train split.
This is the version that transfers to datasets without unimodal annotation (MOSI / MOSEI).
Output: outputs/full/expert_oof_m/seed<s>.csv  (key, yv_oof, yt_oof, sv_oof, st_oof, ya, sa).   Usage: python oof_experts_m.py --seed 1"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold

from au_soft2 import extra_inputs
from oof_experts import predict_text, predict_visual, train_text, train_visual
from texp import tokenize
from train_bert import FULL

OUT = os.path.join(FULL, "expert_oof_m")


class AExp(nn.Module):
    def __init__(self):
        super().__init__(); self.w = nn.Parameter(torch.zeros(13))
        self.mlp = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.mlp((torch.softmax(self.w, 0)[None, :, None] * x).sum(1)).squeeze(-1)


def train_audio(X, Y, idx, va, seed):
    torch.manual_seed(seed); m = AExp().cuda(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    g = torch.Generator().manual_seed(seed); best, state = 1e9, None
    for ep in range(60):
        perm = idx[torch.randperm(len(idx), generator=g).numpy()]
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]; loss = nn.functional.mse_loss(m(X[j].cuda()), Y[j].cuda()); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        m.eval()
        with torch.no_grad():
            v = (m(X[va].cuda()).cpu() - Y[va]).abs().mean().item()
        m.train()
        if v < best:
            best, state = v, {k: x.detach().clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state); return m.eval()


@torch.no_grad()
def predict_audio(m, X, idx):
    x = X[idx].cuda(); m.eval(); y = m(x).cpu().numpy(); m.train(); s = torch.stack([m(x) for _ in range(8)]).std(0).cpu().numpy(); m.eval()
    return y, s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); out = os.path.join(OUT, f"seed{a.seed}.csv")
    if os.path.exists(out):
        print("exists"); return
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    fr, _ = extra_inputs(df, {"stat": False, "emo": True, "diff": False, "vaux": 0.0, "taux": 0.0, "pre": None})
    XV = torch.tensor(fr); Y = torch.tensor(df.label.values, dtype=torch.float32)
    batch = tokenize(df)
    tr = df.index[df.split == "train"].values; va = df.index[df.split == "valid"].values; nt = df.index[df.split != "train"].values
    H = torch.tensor(np.load(os.path.join(FULL, "hubert_layers.npy")).astype(np.float32))
    XA = ((H - H[tr].mean(0)) / (H[tr].std(0) + 1e-5)).clamp(-6, 6)
    res = pd.DataFrame({"key": df.key})
    cols = ("yv_oof", "sv_oof", "yt_oof", "st_oof", "ya", "sa")
    for c in cols:
        res[c] = np.nan
    folds = [(tr[a_], tr[b_]) for a_, b_ in GroupKFold(5).split(tr, groups=df.video_id.values[tr])] + [(tr, nt)]   # last: full-train experts
    for k, (fit, hold) in enumerate(folds):
        sd = a.seed * 100 + k
        m = train_visual(XV, Y, fit, va, sd); res.loc[hold, "yv_oof"], res.loc[hold, "sv_oof"] = predict_visual(m, XV, hold)
        m = train_audio(XA, Y, fit, va, sd); res.loc[hold, "ya"], res.loc[hold, "sa"] = predict_audio(m, XA, hold)
        m = train_text(batch, Y, fit, va, sd); res.loc[hold, "yt_oof"], res.loc[hold, "st_oof"] = predict_text(m, batch, hold)
        del m; torch.cuda.empty_cache(); print(f"seed {a.seed} fold {k} done", flush=True)
    assert not res[list(cols)].isna().any().any()
    res.to_csv(out, index=False)
    for name, ix in (("train_oof", tr), ("test", df.index[df.split == "test"].values)):
        print(f"seed {a.seed} {name} Corr with overall label: " + " ".join(f"{c} {pearsonr(res[c][ix], df.label[ix])[0]:.3f}" for c in ("yv_oof", "yt_oof", "ya")), flush=True)


if __name__ == "__main__":
    main()
