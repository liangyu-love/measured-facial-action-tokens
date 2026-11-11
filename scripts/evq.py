"""Evidence-quality yardstick (CH-SIMS v2): a VISUAL-ONLY model (no text) predicting label_V from frame vectors.
Small transformer (2 layers, d=128, 4 heads) + masked mean pooling; Adam 1e-3, 40 epochs, best epoch by valid MAE.
Evidence variants (--ev):
  raw8      8 full-frame tokens: AUs + face flag
  raw8emo   + 7 emotion probs + EMFACS products (au_soft+emo input)
  dense     best densified variant (--dense_variant, default from valid MAE of results_dense)
  rel / z   raw8emo with per-source-video normalisation (median-subtracted / z-scored; unsupervised)
Output: outputs/full/results_evq/<ev>_seed<s>.json (test Corr / Acc-2 / MAE on label_V)
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_bert import FULL, metrics

OUT = os.path.join(FULL, os.environ.get("EXPERT_DIR_V", "results_evq"))


def evidence(df, ev, dense_variant):
    if ev == "dense":
        from dense_soft import build
        return build(df, dense_variant)
    from au_soft2 import extra_inputs
    fl = {"stat": False, "emo": ev != "raw8", "diff": False, "vaux": 0.0, "taux": 0.0, "pre": None}
    if ev in ("rel", "z"):
        os.environ["MSA_NORM"] = ev
    fr, _ = extra_inputs(df, fl)
    os.environ.pop("MSA_NORM", None)
    return fr, np.ones(fr.shape[:2], np.int64)


class VisOnly(nn.Module):
    def __init__(self, d_in, T, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, 128), nn.LayerNorm(128))
        self.pos = nn.Parameter(torch.randn(T, 128) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True), layers)
        self.head = nn.Linear(128, 1)
        self.pol = nn.Linear(128, 1)  # optional polarity head (sign BCE), training aid only

    def encode(self, x, m):
        """-> frame-level hidden states (B, T, 128), pooled clip vector (B, 128)"""
        h = self.tf(self.inp(x) + self.pos, src_key_padding_mask=(m == 0))
        w = m.unsqueeze(-1).to(h.dtype)
        return h, (h * w).sum(1) / w.sum(1).clamp(min=1)

    def forward(self, x, m):
        return self.head(self.encode(x, m)[1]).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev", required=True, choices=["raw8", "raw8emo", "dense", "rel", "z"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dense_variant", default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tag", default="", help="suffix for the output files (expert tuning grid)")
    ap.add_argument("--pol", type=float, default=0.0, help="weight of the sign-BCE polarity head (|label_V| > 0.1)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{a.ev}_seed{a.seed}{a.tag}.json")
    if os.path.exists(out):
        print("exists"); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    dv = a.dense_variant
    if a.ev == "dense" and not dv:
        dv = min(("crop8", "seg8", "dense16"), key=lambda v: np.mean(
            [json.load(open(p))["valid"]["mae"] for p in glob.glob(os.path.join(FULL, "results_dense", f"{v}_seed*.json"))] or [9]))
    X, M = evidence(df, a.ev, dv)
    X, M = torch.tensor(X, dtype=torch.float32), torch.tensor(M)
    Y = torch.tensor(df.label_V.values, dtype=torch.float32)
    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    model = VisOnly(X.shape[-1], X.shape[1], a.layers).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def pred(idx):
        model.eval(); p = model(X[idx].cuda(), M[idx].cuda()).cpu().numpy(); model.train(); return p

    best, state = 1e9, None
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            zz = model.encode(X[j].cuda(), M[j].cuda())[1]; yy = Y[j].cuda()
            loss = nn.functional.mse_loss(model.head(zz).squeeze(-1), yy)
            if a.pol and (yy.abs() > 0.1).any():
                mk = yy.abs() > 0.1
                loss = loss + a.pol * nn.functional.binary_cross_entropy_with_logits(model.pol(zz).squeeze(-1)[mk], (yy[mk] > 0).float())
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(Y[sp["valid"]].numpy(), pred(sp["valid"]))
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    te = df.iloc[sp["test"]]
    pt = pred(sp["test"])
    from texp import pol_acc
    va = df.iloc[sp["valid"]]; pv = pred(sp["valid"]); vcm = (va.category == "conflict").values; tcm = (te.category == "conflict").values
    res = {"ev": a.ev, "dense_variant": dv, "seed": a.seed, "best_epoch": best_ep, "valid_mae": best, "pol": a.pol,
           "valid_conflict_polacc": pol_acc(va.label_V.values[vcm], pv[vcm]), "test_conflict_polacc": pol_acc(te.label_V.values[tcm], pt[tcm]),
           "test_polacc": pol_acc(te.label_V.values, pt),
           "test_labelV": metrics(te.label_V.values, pt),
           "test_labelV_conflict": metrics(te.label_V.values[(te.category == "conflict").values], pt[(te.category == "conflict").values])}
    torch.save({"state": state, "d_in": X.shape[-1], "T": X.shape[1], "layers": a.layers, "lr": a.lr, "epochs": a.epochs, "ev": a.ev, "dense_variant": dv},
               out.replace(".json", ".pt"))  # frozen evidence encoder for pre_soft.py
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res["test_labelV"]))


if __name__ == "__main__":
    main()
