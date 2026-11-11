"""py-feat measurements (AU / head pose / gaze blendshapes on 8 frames) -> grounded English template.

Intensity bins use quantiles fit on TRAIN-split frames only (no test leakage):
  level(AU) = slight (>q75) / moderate (>q90) / strong (>q97) relative to that AU's train distribution.
Example: "Face: [F3-F5] strong lip corner puller (AU12); [F1-F2] moderate brow lowerer (AU04).
          Head: yaw +18 deg; pitch range 12 deg (nodding). Gaze: down in F2-F4."

Usage: python measure_text.py --clips outputs/full/clips.csv --au outputs/full/au_frames.csv --out outputs/full/measure_text.jsonl
"""
import argparse
import json

import numpy as np
import pandas as pd

NF = 8
AU_NAMES = {
    "AU01": "inner brow raiser", "AU02": "outer brow raiser", "AU04": "brow lowerer", "AU05": "upper lid raiser",
    "AU06": "cheek raiser", "AU07": "lid tightener", "AU09": "nose wrinkler", "AU10": "upper lip raiser",
    "AU12": "lip corner puller", "AU14": "dimpler", "AU15": "lip corner depressor", "AU17": "chin raiser",
    "AU20": "lip stretcher", "AU23": "lip tightener", "AU24": "lip pressor", "AU25": "lips part",
    "AU26": "jaw drop", "AU28": "lip suck", "AU43": "eyes closed",
}
GAZE = {"eyeLookDown": "down", "eyeLookUp": "up", "eyeLookIn": "sideways", "eyeLookOut": "sideways"}
LV = [(0.97, "strong"), (0.90, "moderate"), (0.75, "slight")]
MAX_AU = 5  # keep the template short: the most salient AUs only


def spans(frames):
    """[1,2,3,5] -> 'F1-F3,F5'"""
    out, s = [], None
    for f in sorted(frames) + [None]:
        if s is None:
            s = p = f
        elif f is not None and f == p + 1:
            p = f
        else:
            out.append(f"F{s}" if s == p else f"F{s}-F{p}")
            s = p = f
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--au", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-actions", default=None,
                    help="head_actions.jsonl (v3 taxonomy spans); replaces the range-based Head sentence")
    a = ap.parse_args()
    HEAD_TXT = {"nod": "nodding", "shake": "shaking head", "lean_fwd": "leaning forward", "lean_back": "leaning back",
                "turn_away": "face turned to the side", "head_down": "head down", "head_up": "head up",
                "tilt": "head tilted", "speech_sway": "small head movement while talking", "still": "head still"}
    head_spans = {}
    if a.head_actions:
        for l in open(a.head_actions):
            r = json.loads(l)
            head_spans[r["key"]] = "; ".join(f"{s['t0']:.1f}-{s['t1']:.1f}s {HEAD_TXT[s['label']]}" for s in r["spans"])

    cl = pd.read_csv(a.clips, dtype=str)[["key", "split"]]
    au = pd.read_csv(a.au).merge(cl, on="key")
    pc = ["Pitch", "Roll", "Yaw"]
    if np.nanpercentile(au[pc].abs().values, 99) < 3.2:
        au[pc] = np.degrees(au[pc])
    face = au[au.face == 1]
    tr = face[face.split == "train"]
    aus = [c for c in AU_NAMES if c in au.columns]
    q = {c: {p: float(tr[c].quantile(p)) for p, _ in LV} for c in aus}
    gq = {c: float(tr[c].quantile(0.90)) for c in GAZE if c in au.columns}
    # clip-level pose stats thresholds from train clips
    g_tr = tr.groupby("key")
    rng_tr = (g_tr[pc].max() - g_tr[pc].min())
    mean_tr = g_tr[pc].mean().abs()
    q_rng = {c: float(rng_tr[c].quantile(0.75)) for c in pc}
    q_mean = {c: float(mean_tr[c].quantile(0.75)) for c in pc}

    n_out = 0
    with open(a.out, "w") as f:
        for k, g in au.groupby("key"):
            gf = g[g.face == 1].sort_values("frame")
            parts = []
            missing = sorted(set(range(1, NF + 1)) - set(gf.frame))
            if gf.empty:
                f.write(json.dumps({"key": k, "visual_text": "Face: not detected in any frame."}) + "\n"); n_out += 1
                continue
            # --- AUs: per AU, frames at each level; rank by peak percentile
            items = []
            for c in aus:
                best_lv, fr = None, []
                for i, (p, name) in enumerate(LV):
                    hit = gf.loc[gf[c] > q[c][p], "frame"].tolist()
                    if hit:
                        best_lv, fr, rank = name, hit, i
                        break
                if best_lv:
                    items.append((rank, -gf[c].max(), f"[{spans(fr)}] {best_lv} {AU_NAMES[c]} ({c})"))
            items.sort()
            parts.append("Face: " + ("; ".join(t for *_, t in items[:MAX_AU]) if items else "no salient action units")
                         + (f"; face not visible in {spans(missing)}" if missing else "") + ".")
            # --- head pose
            head = []
            m = gf[pc].mean(); r = gf[pc].max() - gf[pc].min()
            for c, word in (("Yaw", "turned"), ("Pitch", "pitched"), ("Roll", "tilted")):
                if abs(m[c]) > q_mean[c]:
                    head.append(f"{word} ({c.lower()} {m[c]:+.0f} deg)")
            for c, word in (("Pitch", "nodding"), ("Yaw", "turning"), ("Roll", "tilting")):
                if r[c] > q_rng[c]:
                    head.append(f"{word} ({c.lower()} range {r[c]:.0f} deg)")
            if k in head_spans:  # v3: taxonomy spans (physical actions only)
                parts.append("Head: " + head_spans[k] + ".")
            else:
                parts.append("Head: " + (", ".join(head) if head else "mostly still and frontal") + ".")
            # --- gaze
            gz = {}
            for c, word in GAZE.items():
                if c in gq:
                    fr = gf.loc[gf[c] > gq[c], "frame"].tolist()
                    if fr:
                        gz.setdefault(word, set()).update(fr)
            parts.append("Gaze: " + ("; ".join(f"{w} in {spans(fr)}" for w, fr in gz.items()) if gz else "forward") + ".")
            f.write(json.dumps({"key": k, "visual_text": " ".join(parts)}, ensure_ascii=False) + "\n")
            n_out += 1
    print("wrote", n_out, "->", a.out)


if __name__ == "__main__":
    main()
