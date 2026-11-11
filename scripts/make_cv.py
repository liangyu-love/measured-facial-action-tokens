"""SIMS v2 5-fold cross-validation splits (supplementary analysis): GroupKFold over ALL 4403 clips by source video; in each fold the
held-out videos are test, 15% of the remaining videos (GroupShuffleSplit, seeded) are valid, the rest train. Each fold dir gets its own
clips.csv and symlinks to the shared feature tables. -> outputs/full_cv/fold<k>/"""
import os

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

F = os.path.expanduser("~/work/msa-pilot/outputs/full"); CV = os.path.expanduser("~/work/msa-pilot/outputs/full_cv")
c = pd.read_csv(os.path.join(F, "clips.csv"), dtype={"video_id": str, "clip_id": str})
for k, (rest, test) in enumerate(GroupKFold(5).split(c, groups=c.video_id)):
    d = os.path.join(CV, f"fold{k}"); os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    cc = c.copy(); cc["split"] = "train"; cc.loc[cc.index[test], "split"] = "test"
    r = cc.index[rest]
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=k).split(r, groups=cc.video_id.values[rest]))
    cc.loc[r[va], "split"] = "valid"
    cc.to_csv(os.path.join(d, "clips.csv"), index=False)
    for f in ("au_frames.csv", "emo_frames.csv"):
        dst = os.path.join(d, f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(F, f), dst)
    print(k, cc.split.value_counts().to_dict())
