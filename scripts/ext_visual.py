"""External visual expert (no training, no target-dataset labels): py-feat emotion probabilities (trained on external face data)
-> per-frame valence = P(happy) - (P(sad) + P(anger) + P(disgust) + P(fear)); clip reading = mean over face frames, uncertainty =
std over face frames (0 / 0 when no face). Writes <MSA_DATA>/expert_ext_visual.csv (key, yv_ext, sv_ext) and, where unimodal
labels exist (CH-SIMS v2), reports Corr with label_V, polarity accuracy (|label_V| > 0.1) and strict T-V conflict AUROC later.
Usage: MSA_DATA=outputs/full python ext_visual.py"""
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from train_bert import FULL

c = pd.read_csv(os.path.join(FULL, "clips.csv"))
e = pd.read_csv(os.path.join(FULL, "emo_frames.csv"))
e = e[(e.face == 1) & e.frame.between(1, 8)].copy()
e["val"] = e.Happy.fillna(0) - (e.Sad.fillna(0) + e.Anger.fillna(0) + e.Disgust.fillna(0) + e.Fear.fillna(0))
g = e.groupby("key").val
out = pd.DataFrame({"key": c.key}).set_index("key")
out["yv_ext"] = g.mean().reindex(out.index).fillna(0.0)
out["sv_ext"] = g.std(ddof=0).reindex(out.index).fillna(0.0)
out.reset_index().to_csv(os.path.join(FULL, "expert_ext_visual.csv"), index=False)
print("clips", len(out), "with face", int(g.size().reindex(out.index).notna().sum()))
if c.label_V.notna().all():
    c = c.set_index("key"); o = out.loc[c.index]
    for s in ("train", "valid", "test"):
        m = (c.split == s).values; y = c.label_V.values[m]; p = o.yv_ext.values[m]; nz = np.abs(y) > 0.1
        print(f"{s}: Corr(yv_ext, label_V) {pearsonr(p, y)[0]:.3f} | polarity acc {(np.sign(p[nz]) == np.sign(y[nz])).mean():.3f} (n={int(nz.sum())})")
