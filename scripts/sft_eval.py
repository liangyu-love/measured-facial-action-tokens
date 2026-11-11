"""Slice the SFT ablation by py-feat failure cases on the TEST split.
Slices: face_missing (face absent in >=1 of 8 frames), low_conf (mean FaceScore in bottom quartile of test),
profile (median |yaw| > 30 deg, dense pose), multi_face (>=2 faces in >50% of dense frames), any_failure, clean.
Reports (a) v3b verifier hallucination / unverifiable per slice for zs vs sft descriptions,
        (b) downstream BERT metrics per slice (mean±std over seeds) for text / tmpl_au / vlm (7B) / vlm3b_zs / vlm3b_sft.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from train_bert import metrics

ROOT = os.path.expanduser("~/work/msa-pilot")
F = os.path.join(ROOT, "outputs/full")
O = os.path.join(ROOT, os.environ.get("SFT_DIR", "outputs/sft_q3vl4b"))
clips = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str)
test = clips[clips.split == "test"].key
au = pd.read_csv(os.path.join(F, "au_frames.csv"))
au = au[au.key.isin(test)]
g = au.groupby("key")
fs = g.face_score.mean() if "face_score" in au else g.face.mean() * 0
dense = pd.read_csv(os.path.join(F, "dense/dense_pose.csv"))
dense = dense[dense.key.isin(test)]
yaw = dense[dense.face == 1].groupby("key").Yaw.apply(lambda s: float(np.median(np.abs(np.degrees(s)))))
multi = dense.groupby("key").n_faces.apply(lambda s: float((s.fillna(0) >= 2).mean()))
S = pd.DataFrame(index=test)
S["face_missing"] = g.face.mean().reindex(test).fillna(0) < 1
S["low_conf"] = fs.reindex(test) <= fs.quantile(0.25)
S["profile"] = yaw.reindex(test).fillna(90) > 30
S["multi_face"] = multi.reindex(test).fillna(0) > 0.5
S["any_failure"] = S[["face_missing", "low_conf", "profile", "multi_face"]].any(axis=1)
S["clean"] = ~S.any_failure
S["all"] = True
slices = ["all", "clean", "any_failure", "face_missing", "low_conf", "profile", "multi_face"]
lines = ["## slice sizes (test)", ", ".join(f"{s}={int(S[s].sum())}" for s in slices)]

# (a) verifier
lines += ["\n## v3b verifier on test descriptions: hallucination (unsupported/verifiable) | unverifiable | n_claims"]
for v in ("zs", "sft"):
    p = os.path.join(O, v, "score_test", "claims_v3b.csv")
    if not os.path.exists(p):
        continue
    c = pd.read_csv(p)
    row = []
    for s in slices:
        cs = c[c.key.isin(S.index[S[s]])]
        ver = cs[cs.verdict_v3 != "unverifiable"]
        row.append(f"{s}: {(ver.verdict_v3 == 'unsupported').mean():.3f} | {1 - len(ver) / max(1, len(cs)):.3f} | {len(cs)}")
    lines.append(f"- {v}: " + " ; ".join(row))
    sm = os.path.join(O, v, "score_test", "summary_v3b.json")
    if os.path.exists(sm):
        j = json.load(open(sm))
        lines.append(f"  {v} overall: halluc {j['hallucination_rate']} vs shuffled {j['shuffled_control_rate']}; "
                     f"by_type {j['by_type']}; specific-frame {j.get('specific_frame_claim_rate')}; style {j.get('style')}")

# (b) downstream BERT
lines += ["\n## downstream BERT per slice: Acc-2 / MAE / Corr (mean±std over seeds)"]
for cond in ("text", "tmpl_au", "vlm", "vlm8b", "vlm4b_zs", "vlm4b_sft"):
    files = sorted(glob.glob(os.path.join(F, "results", f"{cond}_seed*_test_pred.csv")))
    if not files:
        continue
    per = {s: [] for s in slices}
    for fp in files:
        d = pd.read_csv(fp).set_index("key")
        for s in slices:
            k = S.index[S[s]]
            dd = d.loc[d.index.intersection(k)]
            per[s].append(metrics(dd.label.values, dd.pred.values))
    fmt = lambda xs, m: f"{np.mean([x[m] for x in xs]):.3f}±{np.std([x[m] for x in xs], ddof=1) if len(xs) > 1 else 0:.3f}"
    lines.append(f"- {cond} ({len(files)} seeds): " + " ; ".join(
        f"{s}: {fmt(per[s], 'acc2')} / {fmt(per[s], 'mae')} / {fmt(per[s], 'corr')}" for s in slices))
txt = "\n".join(lines)
open(os.path.join(O, "sft_results.md"), "w").write(txt)
S.to_csv(os.path.join(O, "test_slices.csv"))
print(txt)
