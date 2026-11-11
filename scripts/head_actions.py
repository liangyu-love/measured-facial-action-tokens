"""Fine-grained head-action taxonomy from dense (10 fps) py-feat pose + face bbox.

Classes (physical actions only; a nod is pitch oscillation, never "agreement"):
  nod          pitch oscillation  (high-passed pitch, vertical translation regressed out, >=2 velocity reversals)
  shake        yaw oscillation    (same on yaw, horizontal translation regressed out)
  lean_fwd/back face bbox scale change (monotonic log-scale change)
  turn_away    sustained yaw offset (|median yaw|)
  head_down/up sustained pitch offset (py-feat sign: +pitch = head up, checked visually)
  tilt         sustained roll offset
  speech_sway  irregular multi-axis motion with no oscillation/offset pattern (not a gesture)
  still        all angle ranges small
Each 1 s segment gets ONE label = first class in PRIORITY with score >= 0.5 (fallback sway/still); equal
neighbours are merged into spans. Every class score is kept so claims can be verified per class.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/work/msa-pilot")
O = os.path.join(ROOT, "outputs/pilot_v3_scoring")
PRIORITY = ["nod", "shake", "lean_fwd", "lean_back", "turn_away", "head_down", "head_up", "tilt", "speech_sway", "still"]
TAU = {"nod": 3.0, "shake": 4.0, "lean": 0.06, "turn_away": 20.0, "pitch_off": 10.0, "tilt": 10.0, "move": 4.0}
DOMINANCE = None  # v3b: main-axis ptp must be >= DOMINANCE x the other axes' ptp in the same window
O_OUT = None      # v3b: output dir override


def configure(version):
    """v3 = original; v3b = 5 deg amplitude + single-axis dominance."""
    global DOMINANCE, O_OUT
    if version == "v3b":
        TAU.update(nod=5.0, shake=5.0)
        DOMINANCE = 1.5
        O_OUT = os.path.join(ROOT, "outputs/pilot_v3b_scoring")
SEG, CTX = 1.0, 0.3
sig = lambda e, tau: float(1 / (1 + np.exp(-(e - tau) / (tau / 4))))


def hp(x, n=11):
    return x - pd.Series(x).rolling(n, center=True, min_periods=1).mean().values


def osc(ang, trans, t, lo, hi, others=()):
    """peak-to-peak of the de-trended, translation-regressed angle in a <=1.2 s window with >=2 reversals.
    v3b: the window only counts if this axis dominates the other axes (DOMINANCE x their high-passed ptp)."""
    p, tr = hp(ang), hp(trans)
    if np.std(tr) > 1e-6:
        p = p - (np.dot(p, tr) / np.dot(tr, tr)) * tr
    p = pd.Series(p).rolling(3, center=True, min_periods=1).mean().values
    oth = [pd.Series(hp(o)).rolling(3, center=True, min_periods=1).mean().values for o in others]
    best = 0.0
    for i in np.where((t >= lo) & (t <= hi))[0]:
        w = (t >= t[i]) & (t <= min(t[i] + 1.2, hi))
        if w.sum() < 4:
            continue
        v = np.diff(p[w]); v = v[np.abs(v) > 0.3]
        if len(v) > 1 and np.sum(np.sign(v[1:]) != np.sign(v[:-1])) >= 2:
            amp = float(p[w].max() - p[w].min())
            if DOMINANCE and oth and amp < DOMINANCE * max(float(o[w].max() - o[w].min()) for o in oth):
                continue
            best = max(best, amp)
    return best


def class_scores(g, lo, hi):
    """scores for every class on [lo, hi] (g = whole clip dense series, for context)."""
    t = g.t.values
    seg = g[(g.t >= lo) & (g.t <= hi)]
    if len(seg) < 3:
        return None
    fw = g.fw.values.clip(1e-3)
    nod = osc(g.Pitch.values, g.cy.values / fw, t, lo, hi, others=(g.Yaw.values, g.Roll.values))
    shake = osc(g.Yaw.values, g.cx.values / fw, t, lo, hi, others=(g.Pitch.values, g.Roll.values))
    ls = np.log(seg.fw.values.clip(1e-3)); k = max(1, len(ls) // 3)
    net = float(np.median(ls[-k:]) - np.median(ls[:k])); tv = float(np.abs(np.diff(ls)).sum()) or 1e-6
    mono = abs(net) / tv
    lean = sig(abs(net), TAU["lean"]) if mono >= 0.6 else 0.0
    mp, my, mr = (float(np.median(seg[c])) for c in ("Pitch", "Yaw", "Roll"))
    rng = float(max(seg[c].max() - seg[c].min() for c in ("Pitch", "Yaw", "Roll")))
    s = {"nod": sig(nod, TAU["nod"]) if nod else 0.0,
         "shake": sig(shake, TAU["shake"]) if shake else 0.0,
         "lean_fwd": lean if net > 0 else 0.0, "lean_back": lean if net < 0 else 0.0,
         "turn_away": sig(abs(my), TAU["turn_away"]),
         "head_down": sig(-mp, TAU["pitch_off"]) if mp < 0 else 0.0,
         "head_up": sig(mp, TAU["pitch_off"]) if mp > 0 else 0.0,
         "tilt": sig(abs(mr), TAU["tilt"]),
         "still": 1 - sig(rng, TAU["move"])}
    s["speech_sway"] = sig(rng, TAU["move"]) * (1 - max(s["nod"], s["shake"], lean))
    ev = {"nod_ptp": round(nod, 1), "shake_ptp": round(shake, 1), "dlogscale": round(net, 3),
          "pitch": round(mp, 1), "yaw": round(my, 1), "roll": round(mr, 1), "range": round(rng, 1)}
    return s, ev


def label(s):
    for c in PRIORITY[:-2]:
        if s[c] >= 0.5:
            return c, s[c]
    return ("still", s["still"]) if s["still"] >= 0.5 else ("speech_sway", s["speech_sway"])


def segment_clip(g):
    g = g.sort_values("t").reset_index(drop=True)
    if len(g) < 3:
        return []
    T = g.t.max()
    edges = list(np.arange(0, T, SEG))
    if len(edges) > 1 and T - edges[-1] < 0.5:
        edges.pop()
    segs = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else T
        r = class_scores(g, lo, hi)
        if r is None:
            continue
        s, ev = r
        lab, sc = label(s)
        segs.append({"t0": round(lo, 2), "t1": round(hi, 2), "label": lab, "score": round(sc, 3),
                     "scores": {k: round(v, 3) for k, v in s.items()}, "ev": ev})
    spans = []
    for sgm in segs:  # merge equal neighbours
        if spans and spans[-1]["label"] == sgm["label"] and abs(spans[-1]["t1"] - sgm["t0"]) < 1e-6:
            spans[-1]["t1"] = sgm["t1"]; spans[-1]["score"] = max(spans[-1]["score"], sgm["score"])
        else:
            spans.append({k: sgm[k] for k in ("t0", "t1", "label", "score")})
    return segs, spans


def load_dense(path=None):
    d = pd.read_csv(path or os.path.join(O, "dense_pose.csv"))
    d = d[d.face == 1].copy()
    if np.nanpercentile(d[["Pitch", "Yaw", "Roll"]].abs().values, 99) < 3.2:
        d[["Pitch", "Yaw", "Roll"]] = np.degrees(d[["Pitch", "Yaw", "Roll"]])
    return d


def main():
    import sys
    # usage: head_actions.py [v3|v3b] [dense_pose.csv] [out_dir]
    configure(sys.argv[1] if len(sys.argv) > 1 else "v3")
    out_dir = sys.argv[3] if len(sys.argv) > 3 else (O_OUT or O)
    os.makedirs(out_dir, exist_ok=True)
    d = load_dense(sys.argv[2] if len(sys.argv) > 2 else None)
    out, stats = [], []
    for k, g in d.groupby("key"):
        r = segment_clip(g)
        if not r:
            continue
        segs, spans = r
        out.append({"key": k, "spans": spans, "segments": segs})
        for s in segs:
            stats.append({"key": k, "label": s["label"], "dur": s["t1"] - s["t0"]})
    with open(os.path.join(out_dir, "head_actions.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    st = pd.DataFrame(stats)
    dist = {"clips": int(st.key.nunique()),
            "time_share": (st.groupby("label").dur.sum() / st.dur.sum()).round(3).reindex(PRIORITY).fillna(0).to_dict(),
            "clips_with_label": st.groupby("label").key.nunique().reindex(PRIORITY).fillna(0).astype(int).to_dict()}
    json.dump(dist, open(os.path.join(out_dir, "head_action_distribution.json"), "w"), indent=2)
    print(json.dumps(dist, indent=2))


if __name__ == "__main__":
    main()
