"""Quantitative interpretability on ALL Conflict test clips (CH-SIMS v2) for a trained diff3_unc model.
Per-frame visual-expert reading: the frozen visual expert pools frame states by a (masked) mean and applies a linear head,
so yv = mean_f head(h_f) exactly; yv_f = head(h_f) is frame f's reading. A face frame "contradicts the text" if
sign(yv_f) != sign(yt) (yt = frozen text expert reading).
Two frame-importance scores per clip:
  sim:  post-hoc layer-8 cosine between [CLS] and each of the 8 evidence soft tokens (READ-style map)
  diff: |delta prediction| when frame f is removed from the VISUAL EXPERT only (masked out of the pooling), i.e. the
        effect that flows through the [DIFF] token / scalars, raw evidence tokens untouched
Metrics: hit@1 (top frame is a contradicting frame) vs the random-frame baseline |C|/|face frames| (same clips);
mean score on contradicting vs non-contradicting frames; a permutation test (shuffle frame scores within clip, 5k).
Usage: python interp3.py --ckpt <diff3 save> --seed 1  -> outputs/full/interp3_results.md
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from diff3 import Diff3
from evq import VisOnly
from texp import TextExpert, tokenize
from train_bert import FULL

SIMS = os.path.expanduser("~/work/msa-pilot/outputs/full")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    vck = torch.load(os.path.join(SIMS, "results_evq", f"raw8emo_seed{a.seed}.pt"), map_location="cpu")
    vis = VisOnly(vck["d_in"], vck["T"]); texp = TextExpert() if ck["flags"]["texp"] else None
    model = Diff3(vis, texp, ck["d_frame"], ck["flags"]); model.load_state_dict(ck["state"]); model.cuda().eval()
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    from au_soft2 import extra_inputs
    fr, _ = extra_inputs(df, {"stat": False, "emo": True, "diff": False, "vaux": 0.0, "taux": 0.0, "pre": None})
    import re
    nau = len([c for c in pd.read_csv(os.path.join(FULL, "au_frames.csv"), nrows=1).columns if re.fullmatch(r"AU\d+", c)])
    tb = tokenize(df)
    cap = {}
    model.enc.encoder.layer[7].register_forward_hook(lambda m, i, o: cap.__setitem__("h8", o[0] if isinstance(o, tuple) else o))
    idx = df.index[(df.split == "test") & (df.category == "conflict")].values
    rows = []
    torch.manual_seed(0)
    with torch.no_grad():
        for i in idx:
            ids, att, tt = tb([i]); x = torch.tensor(fr[[i]]).cuda(); msk = torch.ones(1, 8, dtype=torch.long).cuda()
            face = fr[i, :, nau] > 0
            if face.sum() < 2:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y0, aux = model(ids, att, tt, x, msk)
                h8 = cap["h8"].float()[0]; L = ids.shape[1]
                cls, ev = h8[0], h8[L:L + 8]
                sim = torch.nn.functional.cosine_similarity(ev, cls[None], dim=-1).cpu().numpy()
                h, _ = model.vis.encode(x, msk); yvf = model.vis.head(h).float()[0, :, 0].cpu().numpy()
                eff = np.zeros(8)
                for f in range(8):
                    m2 = msk.clone(); m2[0, f] = 0
                    yf, _ = model(ids, att, tt, x, m2)
                    eff[f] = abs(float(yf) - float(y0))
            yt = float(aux["yt"])
            contra = face & (np.sign(yvf) != np.sign(yt))
            rows.append({"key": df.key[i], "yt": yt, "yv": float(aux["yv"]), "label_T": df.label_T[i], "label_V": df.label_V[i],
                         "face": face, "contra": contra, "sim": sim, "eff": eff})
    R = []
    rng = np.random.default_rng(0)
    for name in ("sim", "eff"):
        hits, base, cs, ns, perm_hits = [], [], [], [], []
        for r in rows:
            if not r["contra"].any() or r["contra"].sum() == r["face"].sum():
                continue  # need both kinds of face frames for a contrast
            s = np.where(r["face"], r[name], -np.inf)
            hits.append(r["contra"][int(np.argmax(s))]); base.append(r["contra"].sum() / r["face"].sum())
            cs.append(r[name][r["contra"]].mean()); ns.append(r[name][r["face"] & ~r["contra"]].mean())
        hits, base = np.array(hits, float), np.array(base)
        # permutation: shuffle scores among the clip's face frames
        null = []
        for _ in range(5000):
            hh = []
            for r in rows:
                if not r["contra"].any() or r["contra"].sum() == r["face"].sum():
                    continue
                fi = np.where(r["face"])[0]; hh.append(r["contra"][rng.choice(fi)])
            null.append(np.mean(hh))
        p = (np.sum(np.array(null) >= hits.mean()) + 1) / 5001
        R.append({"score": name, "n_clips": len(hits), "hit@1": hits.mean(), "random_baseline": base.mean(), "lift": hits.mean() / base.mean(),
                  "perm_p": p, "mean_contra": np.mean(cs), "mean_noncontra": np.mean(ns)})
    n_any = sum(r["contra"].any() for r in rows)
    agree = np.mean([np.sign(r["yv"]) == np.sign(r["label_V"]) for r in rows])
    L = [f"## Interpretability on CH-SIMS v2 Conflict test clips (diff3_unc, seed {a.seed})",
         f"- clips analysed (>=2 face frames): {len(rows)}; with >=1 text-contradicting face frame: {n_any}; "
         f"visual expert polarity agrees with label_V on {agree:.3f} of them",
         "| frame score | clips (mixed frames) | hit@1 (top frame contradicts text) | random-frame baseline | lift | permutation p | mean score: contradicting / other frames |",
         "|---|---|---|---|---|---|---|"]
    for r in R:
        L.append(f"| {'post-hoc layer-8 similarity' if r['score'] == 'sim' else 'DIFF-path occlusion |Δŷ|'} | {r['n_clips']} | {r['hit@1']:.3f} | "
                 f"{r['random_baseline']:.3f} | {r['lift']:.2f}× | {r['perm_p']:.4f} | {r['mean_contra']:.4f} / {r['mean_noncontra']:.4f} |")
    txt = "\n".join(L)
    open(os.path.join(FULL, "interp3_results.md"), "w").write(txt)
    pd.DataFrame([{**{k: v for k, v in r.items() if k not in ("face", "contra", "sim", "eff")},
                   "face": "".join(map(str, r["face"].astype(int))), "contra": "".join(map(str, r["contra"].astype(int))),
                   "sim": ",".join(f"{v:.4f}" for v in r["sim"]), "eff": ",".join(f"{v:.5f}" for v in r["eff"])} for r in rows]).to_csv(
        os.path.join(FULL, "interp3_frames.csv"), index=False)
    print(txt)


if __name__ == "__main__":
    main()
