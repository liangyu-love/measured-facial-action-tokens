"""Stacking-leakage check (CPU): expert readings on TRAIN vs VALID vs TEST for the experts used by diff3_unc
(results_evq raw8emo + results_texp, seed given). Corr with the unimodal label, polarity accuracy excluding neutral (|y|>0.1),
AUROC of |yv - yt| for strict T-V opposite clips. Usage: CUDA_VISIBLE_DEVICES= python expert_leak.py --seed 1"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
from evq import VisOnly  # noqa: E402
from texp import TextExpert, pol_acc, tokenize  # noqa: E402
from train_bert import FULL  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--vdir", default="results_evq"); ap.add_argument("--tdir", default="results_texp"); a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
from au_soft2 import extra_inputs  # noqa: E402
fr, _ = extra_inputs(df, {"stat": False, "emo": True, "diff": False, "vaux": 0.0, "taux": 0.0, "pre": None})
vck = torch.load(os.path.join(FULL, a.vdir, f"raw8emo_seed{a.seed}.pt"), map_location="cpu", weights_only=False)
vis = VisOnly(vck["d_in"], vck["T"], vck.get("layers", 2)); vis.load_state_dict(vck["state"], strict=False); vis.eval()
tex = TextExpert(); tex.load_state_dict(torch.load(os.path.join(FULL, a.tdir, f"seed{a.seed}.pt"), map_location="cpu"), strict=False); tex.eval()
tb = tokenize(df)
with torch.no_grad():
    yv = vis(torch.tensor(fr), torch.ones(fr.shape[:2], dtype=torch.long)).numpy()
    yt = np.concatenate([tex(*tb(list(range(i, min(i + 64, len(df)))))).numpy() for i in range(0, len(df), 64)])
df["yv"], df["yt"] = yv, yt
df.to_csv(os.path.join(FULL, f"expert_readings_seed{a.seed}.csv"), columns=["key", "split", "category", "label", "label_T", "label_V", "yv", "yt"], index=False)
sg = np.sign
print(f"seed {a.seed}: experts {a.vdir} / {a.tdir}")
print("| split | n | visual Corr(yv,label_V) | visual polarity acc | text Corr(yt,label_T) | text polarity acc | AUROC |yv-yt| strict T-V (n pos) |")
print("|---|---|---|---|---|---|---|")
for s in ("train", "valid", "test"):
    d = df[df.split == s]; strict = ((sg(d.label_T) * sg(d.label_V)) < 0).astype(int)
    print(f"| {s} | {len(d)} | {pearsonr(d.yv, d.label_V)[0]:.3f} | {pol_acc(d.label_V, d.yv):.3f} | {pearsonr(d.yt, d.label_T)[0]:.3f} | {pol_acc(d.label_T, d.yt):.3f} | "
          f"{roc_auc_score(strict, (d.yv - d.yt).abs()):.3f} ({int(strict.sum())}) |")
