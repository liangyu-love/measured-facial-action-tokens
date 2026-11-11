"""Audio expert on frozen Chinese-HuBERT features (CH-SIMS v2): learned softmax weights over the 13 layers -> MLP (768 -> 256 -> 1),
predicting label_A. AdamW 1e-3, 60 epochs, best epoch on valid MAE. Cross-fitted: 5-fold GroupKFold over TRAIN (groups = video) gives
out-of-fold readings for train clips; a full-train expert gives valid/test readings. MC-dropout std as uncertainty. Seeds 1-5.
Outputs: outputs/full/expert_oof/audio_hubert_seed<s>.csv (key, ya, sa) and audio_hubert_report.json (valid/test Corr, MAE vs label_A;
eGeMAPS expert for comparison). Usage: python audio_expert_hubert.py"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold

from train_bert import FULL

H = torch.tensor(np.load(os.path.join(FULL, "hubert_layers.npy")).astype(np.float32))
c = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
Y = torch.tensor(c.label_A.values, dtype=torch.float32)
tr = c.index[c.split == "train"].values; va = c.index[c.split == "valid"].values; te = c.index[c.split == "test"].values
mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-5
X = ((H - mu) / sd).clamp(-6, 6)


class AExp(nn.Module):
    def __init__(self):
        super().__init__(); self.w = nn.Parameter(torch.zeros(13))
        self.mlp = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.mlp((torch.softmax(self.w, 0)[None, :, None] * x).sum(1)).squeeze(-1)


def fit(idx, seed):
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
def read(m, idx):
    x = X[idx].cuda(); m.eval(); y = m(x).cpu().numpy(); m.train(); s = torch.stack([m(x) for _ in range(8)]).std(0).cpu().numpy(); m.eval()
    return y, s


SEEDS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 6))
os.makedirs(os.path.join(FULL, "expert_oof"), exist_ok=True); rep = {"seeds": {}}
for seed in SEEDS:
    if os.path.exists(os.path.join(FULL, "expert_oof", f"audio_hubert_seed{seed}.csv")) and len(sys.argv) > 1:
        print("exists", seed); continue
    ya, sa = np.full(len(c), np.nan), np.full(len(c), np.nan)
    for k, (a_, b_) in enumerate(GroupKFold(5).split(tr, groups=c.video_id.values[tr])):
        mk = fit(tr[a_], seed * 100 + k); y, s = read(mk, tr[b_]); ya[tr[b_]], sa[tr[b_]] = y, s
    mf = fit(tr, seed); rest = np.concatenate([va, te]); y, s = read(mf, rest); ya[rest], sa[rest] = y, s
    yin, _ = read(mf, tr)
    pd.DataFrame({"key": c.key, "ya": ya, "sa": sa}).to_csv(os.path.join(FULL, "expert_oof", f"audio_hubert_seed{seed}.csv"), index=False)
    r = {s_: {"corr": float(pearsonr(ya[ix], c.label_A.values[ix])[0]), "mae": float(np.abs(ya[ix] - c.label_A.values[ix]).mean())}
         for s_, ix in (("train_oof", tr), ("valid", va), ("test", te))}
    r["train_insample_corr"] = float(pearsonr(yin, c.label_A.values[tr])[0])
    rep["seeds"][seed] = r; print(seed, json.dumps(r), flush=True)
rep["mean"] = {} if not rep["seeds"] else {s_: {k: float(np.mean([rep["seeds"][s][s_][k] for s in rep["seeds"]])) for k in ("corr", "mae")} for s_ in ("train_oof", "valid", "test")}
eg = [json.load(open(p)) for p in glob.glob(os.path.join(FULL, "results_aexp", "seed*.json"))]
rep["egemaps_expert_test"] = {"corr": float(np.mean([r["test_labelA"]["corr"] for r in eg])), "mae": float(np.mean([r["test_labelA"]["mae"] for r in eg])), "n_seeds": len(eg)} if eg else None
rep["pass_threshold_corr_0.5_on_valid"] = bool(rep["mean"]) and rep["mean"]["valid"]["corr"] >= 0.5
json.dump(rep, open(os.path.join(FULL, "expert_oof", "audio_hubert_report.json" if len(sys.argv) == 1 else f"audio_hubert_report_{sys.argv[1].replace(',', '_')}.json"), "w"), indent=2)
print("MEAN", json.dumps(rep["mean"]), "| eGeMAPS", json.dumps(rep["egemaps_expert_test"]), "| PASS", rep["pass_threshold_corr_0.5_on_valid"])
