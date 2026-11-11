"""SFT targets for the Qwen2.5-VL-3B ablation: measured, thresholded py-feat actions with frame indices.
Physical actions only, phrased in the v2-prompt vocabulary (so the v3b verifier can map every phrase).
Train split = SFT data; valid split = loss monitoring. Test split is never used for training.
Output: outputs/sft3b/{train,valid}.jsonl  (key, times, target)
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/work/msa-pilot")
F = os.path.join(ROOT, "outputs/full")
O = os.path.join(ROOT, "outputs/sft3b"); os.makedirs(O, exist_ok=True)
NF = 8
AU_PH = {"AU01": "inner brows raised", "AU02": "outer brows raised", "AU04": "brows lowered and drawn together",
         "AU05": "eyes widened", "AU06": "cheeks raised", "AU07": "eyelids tightened", "AU09": "nose wrinkled",
         "AU10": "upper lip raised", "AU12": "lip corners pulled up", "AU15": "lip corners pulled down",
         "AU20": "lips stretched", "AU23": "lips tightened", "AU24": "lips pressed together", "AU25": "lips parted",
         "AU26": "jaw dropped", "AU43": "eyes closed"}
HEAD_PH = {"nod": "head nodding", "shake": "head shaking", "tilt": "head tilted to one side",
           "turn_away": "head turned to the side", "head_down": "head down", "head_up": "head up",
           "lean_fwd": "leaning forward", "lean_back": "leaning back", "speech_sway": "slight head movement while talking",
           "still": "head still"}
MAX_AU = 4


def span(frames):
    out, s = [], None
    for f in sorted(frames) + [None]:
        if s is None:
            s = p = f
        elif f is not None and f == p + 1:
            p = f
        else:
            out.append(f"F{s}" if s == p else f"F{s}-F{p}"); s = p = f
    return ",".join(out)


clips = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str)
au = pd.read_csv(os.path.join(F, "au_frames.csv")).merge(clips[["key", "split"]], on="key")
q90 = {c: float(au[(au.split == "train") & (au.face == 1)][c].quantile(0.90)) for c in AU_PH if c in au}
heads = {json.loads(l)["key"]: json.loads(l)["spans"] for l in open(os.path.join(F, "dense/head_actions.jsonl"))}
times = {}
for k in clips.key:
    p = os.path.join(F, "frames", k.replace("/", "__"), "frames.json")
    if os.path.exists(p):
        times[k] = json.load(open(p))["times"]

stats = {"train": 0, "valid": 0, "no_face": 0}
fo = {s: open(os.path.join(O, f"{s}.jsonl"), "w") for s in ("train", "valid")}
for k, g in au.groupby("key"):
    sp = g.split.iloc[0]
    if sp not in fo or k not in times:
        continue
    gf = g[g.face == 1].sort_values("frame")
    lines = []
    miss = sorted(set(range(1, NF + 1)) - set(gf.frame))
    if gf.empty:
        lines.append(f"[F1-F{NF}] face not visible"); stats["no_face"] += 1
    else:
        items = []
        for c, ph in AU_PH.items():
            if c not in gf:
                continue
            fr = gf.loc[(gf[c] >= 0.5) & (gf[c] >= q90[c]), "frame"].astype(int).tolist()
            if fr:
                items.append((-gf[c].max(), f"[{span(fr)}] {ph}"))
        lines += [t for _, t in sorted(items)[:MAX_AU]]
        # head: v3b spans mapped onto the sampled frames
        t = times[k]
        for s in heads.get(k, []):
            fr = [i + 1 for i, tt in enumerate(t) if s["t0"] - 1e-6 <= tt <= s["t1"] + 1e-6]
            if fr:
                lines.append(f"[{span(fr)}] {HEAD_PH[s['label']]}")
        # gaze (iris visible only)
        gz = []
        vis = gf[gf.eyeBlink < 0.4] if "eyeBlink" in gf else gf
        for cols, ph in ((["eyeLookDown"], "gaze down"), (["eyeLookIn", "eyeLookOut"], "gaze to the side")):
            fr = vis.loc[(vis[cols].max(axis=1) >= 0.5), "frame"].astype(int).tolist()
            if fr:
                gz.append(f"[{span(fr)}] {ph}")
        lines += gz or ([f"[{span(vis.frame.astype(int).tolist())}] gaze forward"] if len(vis) else [])
        if miss:
            lines.append(f"[{span(miss)}] face not visible")
    lines = lines[:8]
    fo[sp].write(json.dumps({"key": k, "times": times[k], "target": "\n".join("- " + l for l in lines)}) + "\n")
    stats[sp] += 1
for f in fo.values():
    f.close()
print(stats)
