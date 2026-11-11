"""Audio-only unimodal expert (CH-SIMS v2): clip-level eGeMAPS-88 (train-standardised, clipped to +-6) -> MLP
(88 -> 128, LN, GELU, dropout 0.1 -> 128 = z_a) -> Linear(128, 1) predicting label_A. AdamW 1e-3, 60 epochs, best by valid MAE.
Output: outputs/full/results_aexp/seed<s>.pt ({state, mu, sd}) + .json.  Usage: python aexp.py --seed 1"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_bert import FULL, metrics

OUT = os.path.join(os.path.expanduser("~/work/msa-pilot/outputs/full"), os.environ.get("EXPERT_DIR_A", "results_aexp"))


def audio_feats(df, mu=None, sd=None):
    a = pd.read_csv(os.path.join(os.path.expanduser("~/work/msa-pilot/outputs/full"), "audio_egemaps88.csv")).set_index("key").reindex(df.key)
    av = a.values.astype(np.float32)
    if mu is None:
        tr = (df.split == "train").values
        mu, sd = np.nanmean(av[tr], 0), np.nanstd(av[tr], 0) + 1e-6
    return np.nan_to_num((av - mu) / sd).clip(-6, 6).astype(np.float32), mu, sd


class AudioExpert(nn.Module):
    def __init__(self, d_in=88):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 128), nn.LayerNorm(128))
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(128, 1)

    def encode(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.drop(self.encode(x))).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"seed{a.seed}.pt")
    if os.path.exists(out):
        print("exists"); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    X, mu, sd = audio_feats(df)
    X = torch.tensor(X); Y = torch.tensor(df.label_A.values, dtype=torch.float32)
    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    model = AudioExpert(X.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def pred(idx):
        model.eval(); p = model(X[idx].cuda()).cpu().numpy(); model.train(); return p

    best = 1e9
    for ep in range(60):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            loss = nn.functional.mse_loss(model(X[j].cuda()), Y[j].cuda())
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(Y[sp["valid"]].numpy(), pred(sp["valid"]))
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    te = df.iloc[sp["test"]]; pt = pred(sp["test"]); cm = (te.category == "conflict").values
    res = {"seed": a.seed, "best_epoch": best_ep, "valid_mae": best, "test_labelA": metrics(te.label_A.values, pt),
           "test_labelA_conflict": metrics(te.label_A.values[cm], pt[cm])}
    torch.save({"state": state, "mu": mu, "sd": sd, "d_in": X.shape[1]}, out)
    json.dump(res, open(out.replace(".pt", ".json"), "w"), indent=2)
    print(json.dumps(res["test_labelA"]))


if __name__ == "__main__":
    main()
