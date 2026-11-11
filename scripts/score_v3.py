"""v3 verifier for pilot v2 claims (keeps v2 outputs untouched). CPU only.

Changes vs v2 (motivated by manual check v2):
  * every claim gets a continuous score in [0,1] = sigmoid((evidence - tau) / (tau/4)); verdict = score >= 0.5
  * nod   : dense 10 fps pitch, high-passed (minus 1 s moving average) and with vertical face translation
            regressed out; evidence = peak-to-peak in a 1.2 s window that has >= 2 velocity sign reversals
  * turn  : sustained yaw offset (|median yaw|) OR monotonic yaw change (net/total-variation >= 0.6), not range
  * still / tilt : dense pose over the cited span
  * lips part (AU25/26): tau calibrated on manual items (leave-one-out reported)
  * gaze down/up: only frames with eyeBlink < 0.5 count (squint guard)
Outputs -> outputs/pilot_v3_scoring/
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import pilot  # noqa: E402  (claim extraction identical to v2)

ROOT = os.path.expanduser("~/work/msa-pilot")
FULL_MODE = os.environ.get("SCORE_BASE") == "full"  # score on the full 4403-clip set (8-frame AUs + 10 fps pose)
P = os.path.join(ROOT, "outputs/full" if FULL_MODE else "outputs/pilot")
O = os.path.join(ROOT, "outputs/full/dense" if FULL_MODE else "outputs/pilot_v3_scoring")
NF = 8
TAU = {"au": 0.5, "lips": None, "gaze": 0.5, "blink_guard": 0.5,
       "nod": 3.0, "turn_off": 15.0, "turn_net": 10.0, "mono": 0.6, "still": 6.0, "tilt": 10.0}

sig = lambda e, tau: float(1 / (1 + np.exp(-(e - tau) / (tau / 4))))

au = pd.read_csv(os.path.join(P, "au_frames.csv"))
au_by = {k: g.set_index("frame") for k, g in au[au.face == 1].groupby("key")}
dense = pd.read_csv(os.path.join(O, "dense_pose.csv"))
dense = dense[dense.face == 1].copy()
if np.nanpercentile(dense[["Pitch", "Yaw", "Roll"]].abs().values, 99) < 3.2:
    dense[["Pitch", "Yaw", "Roll"]] = np.degrees(dense[["Pitch", "Yaw", "Roll"]])
VERSION = os.environ.get("SCORE_VERSION", "v3")
OUT = O
if VERSION == "v3b":
    OUT = os.path.join(ROOT, "outputs/pilot_v3b_scoring")
    import head_actions as _HA
    _HA.configure("v3b")
    if not FULL_MODE:  # full-res eye crops exist only for the 200 pilot clips; full mode uses the 640 px pass
        eyes = pd.read_csv(os.path.join(OUT, "dense_eyes.csv"))
        ec = ["eyeBlink", "eyeLookDown", "eyeLookUp", "eyeLookIn", "eyeLookOut"]
        eyes.loc[eyes.eye_face != 1, ec] = np.nan  # no face in the full-res crop -> unknown
        dense = dense.drop(columns=ec).merge(eyes[["key", "frame_idx"] + ec], on=["key", "frame_idx"], how="left")
dense_by = {k: g.sort_values("t").reset_index(drop=True) for k, g in dense.groupby("key")}
OUT = os.environ.get("SCORE_OUT", OUT)
os.makedirs(OUT, exist_ok=True)
finfo = {}


def times(key):
    if key not in finfo:
        finfo[key] = json.load(open(os.path.join(P, "frames", key.replace("/", "__"), "frames.json")))
    return finfo[key]["times"]


def span(c, key):
    fr = c["frames"] or list(range(1, NF + 1))
    t = times(key)
    lo, hi = t[min(fr) - 1], t[max(fr) - 1]
    if hi - lo < 1.0:  # single / close frames: centre a ~1 s window
        m = (lo + hi) / 2; lo, hi = m - 0.6, m + 0.6
    return lo - 0.2, hi + 0.2


def hp(x, n=11):
    return x - pd.Series(x).rolling(n, center=True, min_periods=1).mean().values


def nod_evidence(g):
    p, cy = hp(g.Pitch.values), hp(g.cy.values / g.fw.values.clip(1e-3))
    if np.std(cy) > 1e-6:  # regress out vertical translation (body sway / leaning)
        a = np.dot(p, cy) / np.dot(cy, cy); p = p - a * cy
    p = pd.Series(p).rolling(3, center=True, min_periods=1).mean().values
    t = g.t.values; v = np.diff(p); best = 0.0
    for i in range(len(t)):
        w = (t >= t[i]) & (t <= t[i] + 1.2)
        if w.sum() < 4:
            continue
        vw = v[w[:-1] & w[1:]] if len(v) else v
        vw = vw[np.abs(vw) > 0.3]
        rev = int(np.sum(np.sign(vw[1:]) != np.sign(vw[:-1]))) if len(vw) > 1 else 0
        if rev >= 2:
            best = max(best, float(p[w].max() - p[w].min()))
    return best


def turn_score(g):
    y = pd.Series(g.Yaw.values).rolling(3, center=True, min_periods=1).median().values
    off = float(abs(np.median(y)))
    k = max(1, len(y) // 4)
    net = float(abs(np.median(y[-k:]) - np.median(y[:k])))
    tv = float(np.abs(np.diff(y)).sum()) or 1e-6
    mono = net / tv
    s_net = sig(net, TAU["turn_net"]) if mono >= TAU["mono"] else 0.0
    return max(sig(off, TAU["turn_off"]), s_net), {"yaw_off": off, "yaw_net": net, "mono": round(mono, 2)}


import head_actions as HA  # noqa: E402

HEAD_MAP = [(re.compile(p, re.I), c) for p, c in [
    (r"\bnod", "nod"),
    (r"shak\w* (his |her |the )?head|head\s+(is\s+)?shak", "shake"),
    (r"lean\w*\s+(\w+\s+)?(forward|in\b|closer|toward)", "lean_fwd"),
    (r"lean\w*\s+(\w+\s+)?(back|away)", "lean_back"),
    (r"(tilt\w*|lower\w*|bow\w*|dip\w*)\s+(\w+\s+){0,2}(forward|down)|head\s+(is\s+)?(\w+\s+)?(down|lowered|bowed)|(lower\w*|bow\w*) (his |her |the )?head",
     "head_down"),
    (r"(tilt\w*|rais\w*|lift\w*)\s+(\w+\s+){0,2}(back|up|upward)|chin\s+(is\s+)?(up|raised|lifted)|head\s+(is\s+)?(\w+\s+)?(up|raised)|(rais|lift)\w* (his |her |the )?head",
     "head_up"),
    (r"tilt", "tilt"),
    (r"turn\w*|rotat\w*|look\w* (away|aside)|fac\w* (away|to the side)", "turn_away"),
    (r"head\s+(remains?|stays?|is|kept|held)\s+(\w+\s+)?(still|steady|stable|stationary|unchanged)|no (\w+ )?head (movement|motion)|head still",
     "still"),
    (r"(head|body)\s+(\w+\s+)?(mov\w*|sway\w*|bob\w*|shift\w*|bounc\w*)|slight (head )?movement", "speech_sway"),
]]
HEADISH = re.compile(r"head|nod|tilt|lean|chin|turn", re.I)
HEAD_SEGS = {}


def head_class(c):
    if c["type"] not in ("pose", "unmapped") or not HEADISH.search(c["clause"]):
        return None
    for rx, cls in HEAD_MAP:
        if rx.search(c["clause"]):
            return cls
    return None


def head_score(c, key):
    if key not in HEAD_SEGS:
        g = dense_by.get(key)
        r = HA.segment_clip(g) if g is not None else None
        HEAD_SEGS[key] = r[0] if r else []
    segs = HEAD_SEGS[key]
    lo, hi = span(c, key)
    ov = [s for s in segs if s["t1"] > lo and s["t0"] < hi]
    if not ov:
        return np.nan, {}
    cls = c["head_class"]
    pool = ["speech_sway", "nod", "shake"] if cls == "speech_sway" else [cls]  # "head moving" = any oscillatory motion
    s = max(sg["scores"][p] for sg in ov for p in pool)
    ev = {"class": cls, "labels": [sg["label"] for sg in ov]}
    return (1 - s if c["negated"] and cls != "still" else s), ev


BLINK = re.compile(r"blink", re.I)
AGREE = re.compile(r"\b(agree\w*|approv\w*|acknowledg\w*|affirm\w*|consent\w*|yes)\b", re.I)


def closed_runs(g, thr=0.5):
    """durations (s) of consecutive dense frames with eyeBlink >= thr, plus their max eyeBlink"""
    runs, cur = [], []
    for b in g.eyeBlink.values:
        if b >= thr:
            cur.append(b)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return [(len(r) / 10.0, max(r)) for r in runs]


def eye_score(c, key):
    """eyes closed vs blink by duration; gaze only when the iris is visible (eyeBlink low)."""
    g = dense_by.get(key)
    if g is None:
        return np.nan, {}
    lo, hi = span(c, key)
    g = g[(g.t >= lo) & (g.t <= hi)]
    if len(g) < 3 or g.eyeBlink.isna().all():
        return np.nan, {}
    a = c["action"]
    if a in ("eye closure", "blink"):
        runs = closed_runs(g)
        longest = max((d for d, _ in runs), default=0.0)
        if a == "eye closure":  # sustained closure only (>= ~0.4 s)
            s = sig(longest, 0.4); ev = {"closed_s": longest}
        else:  # blink = short closure (< 0.4 s)
            short = [m for d, m in runs if d < 0.4]
            s = sig(max(short), 0.5) if short else 0.0; ev = {"n_blinks": len(short), "closed_s": longest}
        return (1 - s if c["negated"] else s), ev
    cols = [x for x in c["aus"].split("|") if x in g.columns]
    vis = g[g.eyeBlink < 0.4]  # iris visible
    if c["kind"] == "absent":  # gaze forward
        if vis.empty:
            return np.nan, {"reason": "eyes not visible"}
        e = float(vis[cols].max(axis=1).max()); s = 1 - sig(e, TAU["gaze"])
        return (1 - s if c["negated"] else s), {"e": round(e, 3)}
    if vis.empty:  # eyelids low throughout: squint / eyes closed, not gaze
        return (1.0 if c["negated"] else 0.0), {"reason": "eyelids low (squint/closed)",
                                                 "blink_min": round(float(g.eyeBlink.min()), 2)}
    e = float(vis[cols].max(axis=1).max()); s = sig(e, TAU["gaze"])
    return (1 - s if c["negated"] else s), {"e": round(e, 3), "visible_frames": int(len(vis))}


def score(c, src=None):
    """-> (score in [0,1] or nan, evidence dict). nan = unverifiable."""
    key = src or c["key"]
    t, a = c["type"], c["action"]
    if c.get("head_class"):
        return head_score(c, key)
    if t in ("unmapped", "pose") or not c["aus"]:  # head phrases outside the taxonomy -> unverifiable
        return np.nan, {}
    if a in ("eye closure", "blink") or t == "gaze":
        return eye_score(c, key)
    if t == "AU":
        g = au_by.get(key)
        if g is None:
            return np.nan, {}
        g = g[g.index.isin(c["frames"] or range(1, NF + 1))]
        if t == "gaze" and c["kind"] == "present" and "eyeBlink" in g:
            g = g[g.eyeBlink < TAU["blink_guard"]]  # squint guard
        cols = [x for x in c["aus"].split("|") if x in g.columns]
        if g.empty or not cols:
            return np.nan, {}
        e = float(g[cols].max(axis=1).max())
        tau = TAU["lips"] if a == "lips part / jaw drop" and TAU["lips"] else (TAU["gaze"] if t == "gaze" else TAU["au"])
        s = sig(e, tau)
        if c["kind"] == "absent":
            s = 1 - s
        return (1 - s if c["negated"] else s), {"e": round(e, 3), "tau": tau}
    # pose: dense series over the cited span
    g = dense_by.get(key)
    if g is None:
        return np.nan, {}
    lo, hi = span(c, key)
    g = g[(g.t >= lo) & (g.t <= hi)]
    if len(g) < 4:
        return np.nan, {}
    if c["kind"] == "move":  # nod
        e = nod_evidence(g); s = sig(e, TAU["nod"]) if e > 0 else 0.0; ev = {"nod_ptp": round(e, 2)}
    elif c["kind"] == "move_or_static":  # turn
        s, ev = turn_score(g)
    elif c["kind"] == "still":
        e = float(max(g[x].max() - g[x].min() for x in ("Pitch", "Yaw", "Roll")))
        s = 1 - sig(e, TAU["still"]); ev = {"ptp": round(e, 2)}
    else:  # tilt
        col = "Pitch" if re.search(r"forward|back|down|up", c["clause"], re.I) else "Roll"
        e = float(abs(np.median(g[col]))); s = sig(e, TAU["tilt"]); ev = {col: round(e, 2)}
    return (1 - s if c["negated"] else s), ev


def kappa(h, m):
    h, m = np.asarray(h), np.asarray(m)
    po = (h == m).mean(); ph, pm = (h == "是").mean(), (m == "是").mean()
    pe = ph * pm + (1 - ph) * (1 - pm)
    return float(po), float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    os.makedirs(OUT, exist_ok=True)
    # generalised: SCORE_DESC = descriptions to score, SCORE_KEYS = csv with 'key' column restricting the clips
    desc = [json.loads(l) for l in open(os.environ.get("SCORE_DESC", os.path.join(P, "desc.jsonl"))) if l.strip()]
    if os.environ.get("SCORE_KEYS"):
        ks = set(pd.read_csv(os.environ["SCORE_KEYS"], dtype=str).key)
        desc = [d for d in desc if d["key"] in ks]
    lines_all = [l for d in desc for l in d["desc"].splitlines() if l.strip()]
    style = {"n_desc": len(desc),
             "emotion_leak_desc_rate": round(float(np.mean([bool(pilot.EMO.search(d["desc"])) for d in desc])), 3),
             "inference_desc_rate": round(float(np.mean([bool(pilot.INFER.search(d["desc"])) for d in desc])), 3),
             "frame_ref_line_rate": round(float(np.mean([bool(pilot.FREF.search(l)) for l in lines_all])), 3),
             "mean_sec": round(float(np.mean([d.get("sec", np.nan) for d in desc])), 2) if desc and "sec" in desc[0] else None,
             "max_peak_gib": max((d.get("peak_gib", 0) for d in desc), default=None)}
    claims = [c for d in desc for c in pilot.extract_claims(d["key"], d["desc"])]
    for c in claims:  # separate claim type for blinks (v2 lumped them into eye closure)
        c["orig_action"] = c["action"]
        if c["action"] == "eye closure" and BLINK.search(c["clause"]):
            c["action"] = "blink"
        c["head_class"] = head_class(c)
        if c["head_class"]:
            c["type"], c["action"] = "head", "head:" + c["head_class"]
    agree_lines = sum(bool(AGREE.search(l)) for d in desc for l in d["desc"].splitlines())
    man =pd.read_csv(os.path.join(ROOT, "outputs/manual_check_v2_merged.csv"), encoding="utf-8-sig", dtype=str).fillna("")
    man = man[man.answer.isin(["是", "否"])]

    external = bool(os.environ.get("SCORE_DESC"))  # scoring another model's descriptions
    if external:  # calibration/manual items belong to the 7B claims: reuse the 7B-calibrated tau
        TAU["lips"] = 0.09
        LIPS_CAL = {"tau": 0.09, "note": "fixed from Qwen-7B v3b calibration"}
        man = man.iloc[0:0]

    # ---- calibrate lips-part tau on manual items (report LOO)
    def lips_e(r):
        c = next(c for c in claims if c["key"] == r.key and c["orig_action"] == r.action and c["clause"] == r.clause)
        return score(c)[1].get("e", np.nan), c
    if not external:
        lp = man[man.action == "lips part / jaw drop"]
        E = np.array([lips_e(r)[0] for r in lp.itertuples()]); H = (lp.answer == "是").values
        grid = np.round(np.arange(0.05, 0.61, 0.01), 2)
        acc = lambda idx, tau: np.mean((E[idx] >= tau) == H[idx])
        allidx = np.arange(len(E))
        best = grid[np.argmax([acc(allidx, t) for t in grid])]
        loo = []
        for i in allidx:
            rest = allidx[allidx != i]
            tb = grid[np.argmax([acc(rest, t) for t in grid])]
            loo.append((E[i] >= tb) == H[i])
        TAU["lips"] = float(best)
        print(f"lips tau={best} in-sample acc={acc(allidx, best):.2f} LOO acc={np.mean(loo):.2f} (n={len(E)})")
        LIPS_CAL = {"tau": float(best), "in_sample_acc": round(float(acc(allidx, best)), 3),
                    "loo_acc": round(float(np.mean(loo)), 3), "n": int(len(E))}

    # ---- score all claims + shuffled control
    rng = np.random.default_rng(42)
    keys = sorted(au_by)
    rows = []
    for c in claims:
        s, ev = score(c)
        rows.append({**{k: c[k] for k in ("key", "type", "action", "kind", "negated", "clause", "line")},
                     "frames": ",".join(f"F{i}" for i in c["frames"]), "score_v3": s,
                     "verdict_v3": "unverifiable" if np.isnan(s) else ("supported" if s >= 0.5 else "unsupported"),
                     "evidence": json.dumps(ev)})
    df = pd.DataFrame(rows)
    ctrl = []
    for _ in range(20):
        for c, v in zip(claims, df.verdict_v3):
            if v == "unverifiable":
                continue
            s, _ = score(c, src=rng.choice([k for k in keys if k != c["key"]]))
            if not np.isnan(s):
                ctrl.append((c["type"], s >= 0.5))
    ctrl = pd.DataFrame(ctrl, columns=["type", "sup"])
    ver = df[df.verdict_v3 != "unverifiable"]
    summ = {
        "n_claims": len(df), "verifiable": len(ver), "unverifiable_rate": round(1 - len(ver) / len(df), 3),
        "hallucination_rate": round(float((ver.verdict_v3 == "unsupported").mean()), 3),
        "shuffled_control_rate": round(float((~ctrl.sup).mean()), 3), "control_n": len(ctrl),
        "by_type": {t: f"{(g.verdict_v3 == 'unsupported').mean():.3f} (n={len(g)}) vs ctrl "
                       f"{(~ctrl[ctrl.type == t].sup).mean():.3f}" for t, g in ver.groupby("type")},
        "by_action": {a: f"{(g.verdict_v3 == 'unsupported').mean():.2f} (n={len(g)})" for a, g in ver.groupby("action")},
        "mean_score_by_type": ver.groupby("type").score_v3.mean().round(3).to_dict(),
        "tau": TAU, "lips_calibration": LIPS_CAL, "version": VERSION,
        "nod_as_agreement_lines": int(agree_lines),  # interpretive reading of nods ("in agreement") -> leakage
        "style": style,
        "specific_frame_claim_rate": round(float(np.mean([0 < len(c["frames"]) < NF for c in claims])), 3) if claims else None,
        "claims_per_desc": round(len(claims) / max(1, len(desc)), 2),
    }

    # ---- manual agreement, v2 vs v3 on the same items
    mrows = []
    for r in man.itertuples():
        c = next((c for c in claims if c["key"] == r.key and c["orig_action"] == r.action and c["clause"] == r.clause), None)
        s, ev = score(c) if c else (np.nan, {})
        mrows.append({"id": r.id, "action": r.action, "human": r.answer,
                      "v2": "是" if r.pf == "supported" else "否",
                      "v3": "看不清" if np.isnan(s) else ("是" if s >= 0.5 else "否"),
                      "score_v3": None if np.isnan(s) else round(s, 3), "evidence": json.dumps(ev), "note": r.note})
    me = pd.DataFrame(mrows, columns=["id", "action", "human", "v2", "v3", "score_v3", "evidence", "note"])
    motion = me.action.isin(["head nod / pitch change", "head turn / yaw"])
    res = {}
    for name, m in ((() if external else (("all", me.index == me.index), ("static", ~motion), ("motion", motion)))):
        g = me[m]
        g3 = g[g.v3 != "看不清"]
        res[name] = {"n": int(len(g)), "v2_agree_kappa": [round(x, 3) for x in kappa(g.human, g.v2)],
                     "v3_n_decided": int(len(g3)), "v3_agree_kappa": [round(x, 3) for x in kappa(g3.human, g3.v3)]}
    summ["manual_v2_vs_v3"] = res if not external else "n/a (external descriptions)"
    df.to_csv(os.path.join(OUT, f"claims_{VERSION}.csv"), index=False)
    me.to_csv(os.path.join(OUT, f"manual_eval_{VERSION}.csv"), index=False, encoding="utf-8-sig")
    json.dump(summ, open(os.path.join(OUT, f"summary_{VERSION}.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    print(me[["id", "action", "human", "v2", "v3", "score_v3", "evidence", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
