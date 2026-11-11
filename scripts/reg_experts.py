"""Less-overfit experts (diff3_reg): text expert 2 epochs + dropout 0.3 (best epoch on valid); visual expert = the existing
valid-early-stopped expert (results_evq). In-sample readings (mean + MC-dropout std) for every clip.
Output: outputs/full/expert_oof/seed<s>_reg.csv (key, yv_reg, yt_reg, sv_reg, st_reg) and train/valid/test Corr per expert."""
import argparse, os
import numpy as np, pandas as pd, torch
from scipy.stats import pearsonr
from au_soft2 import extra_inputs
from evq import VisOnly
from oof_experts import OUT, predict_text, predict_visual, train_text
from texp import tokenize
from train_bert import FULL
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args()
os.makedirs(OUT, exist_ok=True); out = os.path.join(OUT, f"seed{a.seed}_reg.csv")
df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
fr, _ = extra_inputs(df, {"stat": False, "emo": True, "diff": False, "vaux": 0.0, "taux": 0.0, "pre": None})
X = torch.tensor(fr); YT = torch.tensor(df.label_T.values, dtype=torch.float32); batch = tokenize(df)
tr = df.index[df.split == "train"].values; va = df.index[df.split == "valid"].values; allix = df.index.values
tm = train_text(batch, YT, tr, va, a.seed, epochs=2, drop=0.3); yt, st = predict_text(tm, batch, allix)
vck = torch.load(os.path.join(FULL, "results_evq", f"raw8emo_seed{a.seed}.pt"), map_location="cpu", weights_only=False)
vm = VisOnly(vck["d_in"], vck["T"], vck.get("layers", 2)); vm.load_state_dict(vck["state"], strict=False); vm.cuda().eval()
yv, sv = predict_visual(vm, X, allix)
pd.DataFrame({"key": df.key, "yv_reg": yv, "yt_reg": yt, "sv_reg": sv, "st_reg": st}).to_csv(out, index=False)
for s in ("train", "valid", "test"):
    m = (df.split == s).values
    print(f"seed {a.seed} {s}: text(reg) Corr {pearsonr(yt[m], df.label_T[m])[0]:.3f} | visual Corr {pearsonr(yv[m], df.label_V[m])[0]:.3f}", flush=True)
