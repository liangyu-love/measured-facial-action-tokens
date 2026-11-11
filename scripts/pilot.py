"""Go/no-go pilot: are Qwen2.5-VL visual descriptions grounded in measurable facial actions?

Stages (each caches to ~/work/msa-pilot/outputs/pilot/, re-runnable):
  sample  : 200 Conflict clips (seed 42), stratified by which modalities conflict  -> pilot_clips.csv
  frames  : 8 uniformly sampled frames per clip, saved as JPEG (shared by py-feat and Qwen) -> frames/
  au      : py-feat Detectorv2 AUs per frame (largest face)                          -> au_frames.csv
  desc    : Qwen2.5-VL-7B descriptions from the same 8 frames                         -> desc.jsonl
  score   : lexicon phrase->AU claims, verdicts vs AU intensities on cited frames,
            hallucination rate, emotion-word leakage, shuffled-clip control, 50-claim manual sheet
(prosody: run prosody_text.py extract/describe with --splits pilot_clips.csv)
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/work/msa-pilot")
OUT = os.path.join(ROOT, "outputs/pilot")
RAW = os.path.join(ROOT, os.environ.get("MSA_RAW", "data/chsims2/Raw"))
NF = 8

# ---------------------------------------------------------------- lexicon: phrase regex -> AUs
LEX = [
    (r"smil|grin|(mouth|lip)[- ]?corners?\s+(are\s+|is\s+)?(pulled|turned|curved|drawn|raised|lifted)\s*(up|upward)?|corners of (the )?(mouth|lips) (are )?(pulled |turned |curved |raised |lifted )?(up|upward)",
     ["AU12"], "lip corner puller"),
    (r"(mouth|lip)[- ]?corners?\s+(are\s+)?(pulled|turned|drawn)\s+down|corners of (the )?(mouth|lips) (are )?(pulled |turned |drawn )?down|downturned",
     ["AU15"], "lip corner depressor"),
    (r"frown|furrow|brows?\s+(are\s+)?(lowered|knit|drawn together|pulled together|drawn down)|eyebrows?\s+(are\s+)?(lowered|knit|drawn together|pulled together)",
     ["AU04"], "brow lowerer"),
    (r"(brows?|eyebrows?)\s+(are\s+|is\s+)?(slightly\s+)?(raised|lifted|arched|go up)|rais(ed|es|ing) (the |his |her )?(eye)?brows?",
     ["AU01", "AU02"], "brow raiser"),
    (r"(lips?\s+(are\s+)?(pressed|tightened|compressed|pursed))|press(ed|es|ing) (the |his |her )?lips|tight(ened)? lips",
     ["AU24", "AU23"], "lip press/tightener"),
    (r"jaw\s+(drops?|dropped|lowered|open)|mouth\s+(is\s+|slightly\s+|wide\s+)*(open|opened|opens)|open(s|ed)? (his |her |the )?mouth|lips?\s+(are\s+)?(slightly\s+)?(parted|apart|separated)",
     ["AU25", "AU26"], "lips part / jaw drop"),
    (r"eyes?\s+(are\s+)?(closed|shut|closing)|blink|clos(es|ed|ing) (his |her |the )?eyes",
     ["AU43"], "eye closure"),
    (r"eyes?\s+(are\s+|open\s+)?(wide|widened|widening)|wide[- ]open eyes|eyes open wide",
     ["AU05"], "upper lid raiser"),
    (r"squint|narrow(ed|ing)? (his |her |the )?eyes|eyes?\s+(are\s+)?narrow|cheeks?\s+(are\s+)?(raised|lifted|puffed)",
     ["AU06", "AU07"], "cheek raiser / lid tightener"),
    (r"(wrinkl|scrunch)\w* (the |his |her )?nose|nose\s+(is\s+)?(wrinkled|scrunched)",
     ["AU09"], "nose wrinkler"),
    (r"upper lip\s+(is\s+)?(raised|curled|lifted)|sneer",
     ["AU10"], "upper lip raiser"),
    (r"chin\s+(is\s+)?(raised|pushed up|wrinkled)",
     ["AU17"], "chin raiser"),
    (r"lips?\s+(are\s+)?(stretched|pulled back|drawn back)",
     ["AU20"], "lip stretcher"),
]
LEX = [(re.compile(p, re.I), aus, name) for p, aus, name in LEX]

# head pose claims -> py-feat Pitch/Roll/Yaw (degrees). kind: "static" = |angle| on cited frames,
# "move" = range of angle over cited frames (+-1 neighbour), "still" = max range of all angles stays small.
POSE = [
    (r"\bnod\w*|head\s+(moves?|moving|bobs?)\s+(up and down|down and up)|(lower|rais|lift)(s|ed|ing)? (his |her |the )?head",
     ["Pitch"], "move", "head nod / pitch change"),
    (r"(turn|rotat)(s|ed|ing)? (his |her |the )?head|head\s+(is\s+|slightly\s+)*(turned|turns|turning|rotated)|shak(es|ing) (his |her |the )?head",
     ["Yaw"], "move_or_static", "head turn / yaw"),
    (r"tilt\w*|head\s+(is\s+)?(cocked|leaning|leans|inclined)",
     ["Roll", "Pitch"], "static", "head tilt"),
    (r"head\s+(remains?|stays?|is|kept|held)\s+(mostly\s+|relatively\s+|fairly\s+)?(still|steady|stable|stationary|unchanged)|no (significant |noticeable |visible )?head (movement|motion)",
     ["Pitch", "Roll", "Yaw"], "still", "head still"),
]
POSE = [(re.compile(p, re.I), cols, kind, name) for p, cols, kind, name in POSE]

# gaze claims -> ARKit-style blendshapes from py-feat (0..1, max of left/right eye)
GAZE = [
    (r"(look\w*|gaz\w*|eyes?)\s+(\w+\s+){0,2}(down|downward|lowered)|(look\w*|gaz\w*)\s+at (the )?(floor|ground|table)|downcast",
     ["eyeLookDown"], "present", "gaze down"),
    (r"(look\w*|gaz\w*|eyes?)\s+(\w+\s+){0,2}(up|upward)\b", ["eyeLookUp"], "present", "gaze up"),
    (r"(look\w*|gaz\w*|eyes?)\s+(\w+\s+){0,3}(left|right|sideways|side|away|aside)",
     ["eyeLookIn", "eyeLookOut"], "present", "gaze sideways"),
    (r"(look\w*|gaz\w*|eyes?)\s+(\w+\s+){0,3}(straight|forward|ahead|directly|at the camera|into the camera|at the viewer)|direct gaze",
     ["eyeLookDown", "eyeLookUp", "eyeLookIn", "eyeLookOut"], "absent", "gaze forward"),
]
GAZE = [(re.compile(p, re.I), cols, kind, name) for p, cols, kind, name in GAZE]
BLEND = ["eyeLookDown", "eyeLookUp", "eyeLookIn", "eyeLookOut", "eyeBlink", "jawOpen", "mouthSmile"]

NEG = re.compile(r"\b(no|not|without|never|neither|nor|absence of|lack of|n't)\b", re.I)
UNMAPPED = re.compile(r"head|gaze|look(s|ing)?|eyes? (are )?(focused|fixed)|posture|shoulder|hand|gesture|body|lean", re.I)

# Emotion/attitude words. Muscle-state words (relaxed, calm, tense, neutral) are NOT counted: in this setting
# they mostly describe muscles ("mouth corners relaxed"), so they were false positives in v0.
EMO = re.compile(
    r"\b(happ\w*|joy\w*|sad\w*|sorrow\w*|ang(ry|er)\w*|annoy\w*|irritat\w*|furious|rage|fear\w*|afraid|scared|"
    r"anxi\w*|nervous\w*|worr\w*|surpris\w*|shock\w*|astonish\w*|disgust\w*|contempt\w*|excit\w*|amus\w*|"
    r"cheer\w*|delight\w*|pleas(ed|ure|ant)|content(ed|ment)|frustrat\w*|upset|depress\w*|"
    r"emotion\w*|mood|sentiment|positive|negative|friendly|hostile|confident|confus\w*|bored|embarrass\w*|"
    r"stress\w*|serious|enthusias\w*|grief|lonely|proud|shy|interest(ed)?|engag(ed|ement)|attentive\w*|"
    r"thoughtful|contemplat\w*|determin\w*|skeptic\w*|doubt\w*|concern\w*)\b", re.I)
INFER = re.compile(r"\b(indicat\w*|suggest\w*|express\w*|convey\w*|show(s|ing)? (signs of|that)|appear(s)? to|"
                   r"seem(s)?|as if|possibly|likely|perhaps|probably)\b", re.I)
FREF = re.compile(r"F(\d+)(?:\s*[-–~to]+\s*F?(\d+))?")


# ---------------------------------------------------------------- stages
def st_sample(a):
    base = pd.read_csv(os.path.join(ROOT, "outputs/splits/emomm_chsims_base.csv"), dtype=str)
    c = base[base.subset == "Conflict"].copy()
    n = a.n
    alloc = (c.conflict_mods.value_counts(normalize=True) * n).round().astype(int)
    alloc.iloc[0] += n - alloc.sum()  # fix rounding so total == n
    parts = [c[c.conflict_mods == m].sample(n=min(k, (c.conflict_mods == m).sum()), random_state=a.seed)
             for m, k in alloc.items()]
    s = pd.concat(parts).sort_values("key").reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    s.to_csv(os.path.join(OUT, "pilot_clips.csv"), index=False)
    print("pilot clips", len(s), s.conflict_mods.value_counts().to_dict())


CLIPS = os.path.join(OUT, "pilot_clips.csv")


def clips():
    return pd.read_csv(CLIPS, dtype=str)


def frame_dir(key):
    return os.path.join(OUT, "frames", key.replace("/", "__"))


def frame_one(k):
    from torchcodec.decoders import VideoDecoder
    from PIL import Image
    d = frame_dir(k)
    p = os.path.join(RAW, f"{k}.mp4")
    if os.path.exists(os.path.join(d, "frames.json")):
        return 0
    if not os.path.exists(p):
        print("[missing video]", k, flush=True); return 0
    try:
        dec = VideoDecoder(p)
        n = dec.metadata.num_frames
        fps = dec.metadata.average_fps or 25.0
        idx = np.linspace(0, n - 1, NF).round().astype(int).tolist()
        os.makedirs(d, exist_ok=True)
        for i, t in enumerate(dec.get_frames_at(indices=idx).data, 1):
            Image.fromarray(t.permute(1, 2, 0).numpy()).save(os.path.join(d, f"F{i}.jpg"), quality=92)
        info = {"key": k, "num_frames": n, "fps": fps, "idx": idx, "times": [round(i / fps, 2) for i in idx],
                "duration": round(n / fps, 2), "size": list(t.shape[1:])}
        json.dump(info, open(os.path.join(d, "frames.json"), "w"))  # written last = completion marker
        return 1
    except Exception as e:
        print("[frame error]", k, repr(e)[:200], flush=True); return 0


def st_frames(a):
    from multiprocessing import get_context
    keys = clips().key.tolist()
    with get_context("fork").Pool(a.workers) as pool:  # fork: children inherit OUT/CLIPS overrides
        n = 0
        for i, r in enumerate(pool.imap_unordered(frame_one, keys, chunksize=8), 1):
            n += r
            if i % 500 == 0:
                print(f"frames {i}/{len(keys)}", flush=True)
    print("framed", n, "new clips")


def st_au(a):
    import torch
    from feat import Detectorv2
    out = os.path.join(OUT, "au_frames.csv")
    parts = os.path.join(OUT, "au_parts")
    os.makedirs(parts, exist_ok=True)
    done = set()
    for p in sorted(os.listdir(parts)):
        done |= set(pd.read_csv(os.path.join(parts, p), usecols=["key"], dtype=str).key)
    if os.path.exists(out) and not os.listdir(parts):  # legacy single-file cache
        done |= set(pd.read_csv(out, usecols=["key"], dtype=str).key)
    if os.environ.get("AU_MEM_FRAC"):  # cap per-worker VRAM so a huge batch raises OOM (-> per-frame fallback) instead of starving siblings
        torch.cuda.set_per_process_memory_fraction(float(os.environ["AU_MEM_FRAC"]))
    det = Detectorv2(device="cuda", identity_model=None)
    rows = []
    allk = list(clips().key)
    todo = [k for k in allk if k not in done]
    for ex in filter(None, os.environ.get("AU_EXCLUDE_SHARDS", "").split(",")):  # skip slices still owned by running workers
        ei, en = (int(x) for x in ex.split("/"))
        own = set(allk[ei::en]); todo = [k for k in todo if k not in own]
    si, sn = (int(x) for x in a.shard.split("/"))
    todo = todo[si::sn]
    print(len(todo), "clips for AU", flush=True)
    nflush = [0]

    def flush():
        if rows:
            nflush[0] += 1
            pd.DataFrame(rows).to_csv(os.path.join(parts, f"part_s{si}of{sn}_{nflush[0]:05d}_{os.getpid()}.csv"), index=False)
            rows.clear()

    for n_done, k in enumerate(todo, 1):
        d = frame_dir(k)
        if not os.path.exists(os.path.join(d, "frames.json")):
            continue
        paths = [os.path.join(d, f"F{i}.jpg") for i in range(1, NF + 1)]
        kw = {"progress_bar": False} if "progress_bar" in det.detect.__code__.co_varnames else {}
        try:
            fex = pd.DataFrame(det.detect(paths, data_type="image", batch_size=NF, **kw))
        except torch.OutOfMemoryError:  # rare high-res / many-face clips: same detector, one frame at a time
            torch.cuda.empty_cache()
            fex = pd.concat([pd.DataFrame(det.detect([q], data_type="image", batch_size=1, **kw)) for q in paths], ignore_index=True)
            torch.cuda.empty_cache()
            print("oom-fallback", k, flush=True)
        au_cols = [c for c in fex.columns if re.fullmatch(r"AU\d+", str(c))]
        fex["area"] = fex["FaceRectWidth"] * fex["FaceRectHeight"]
        for i, pth in enumerate(paths, 1):
            g = fex[fex["input"].astype(str) == pth].dropna(subset=au_cols)
            r = {"key": k, "frame": i, "face": int(len(g) > 0)}
            if len(g):
                best = g.sort_values("area").iloc[-1]  # largest face = speaker
                r.update({c: float(best[c]) for c in au_cols})
                r["face_score"] = float(best.get("FaceScore", np.nan))
                for c in ("Pitch", "Roll", "Yaw", "gaze_pitch", "gaze_yaw"):
                    r[c] = float(best[c]) if c in best else np.nan
                for b in BLEND:  # max over left/right eye (or the single column)
                    cs = [c for c in (f"{b}Left", f"{b}Right", b) if c in best.index]
                    r[b] = float(max(best[c] for c in cs)) if cs else np.nan
                for e in ("Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"):  # same pass: emotions
                    if e in best.index:
                        r[e] = float(best[e])
            rows.append(r)
        if n_done % 200 == 0:
            flush(); print(f"au {n_done}/{len(todo)}", flush=True)
    flush()
    dfs = [pd.read_csv(os.path.join(parts, p)) for p in sorted(os.listdir(parts))]
    if os.path.exists(out) and not dfs:
        return
    df = pd.concat(dfs).drop_duplicates(["key", "frame"]) if dfs else pd.DataFrame()
    df.to_csv(out, index=False)
    print("au rows", len(df), "faces found", int(df.face.sum()) if len(df) else 0)


def st_desc(a):
    import time
    import torch
    from PIL import Image
    from gen_descriptions import describe, load_model
    out = os.path.join(OUT, "desc.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["key"] for l in open(out) if l.strip()}
    todo = [k for k in clips().key if k not in done and os.path.exists(os.path.join(frame_dir(k), "frames.json"))]
    print(len(todo), "clips to describe")
    if not todo:
        return
    model, proc = load_model()
    with open(out, "a") as f:
        for k in todo:
            d = frame_dir(k)
            info = json.load(open(os.path.join(d, "frames.json")))
            frames = []
            for i in range(1, NF + 1):
                im = Image.open(os.path.join(d, f"F{i}.jpg")).convert("RGB")
                s = 448 / max(im.size)
                frames.append(im.resize((round(im.width * s), round(im.height * s))) if s < 1 else im)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            text, n_tok = describe(model, proc, frames, info["times"])
            f.write(json.dumps({"key": k, "desc": text, "prompt_tokens": n_tok, "sec": round(time.time() - t0, 2),
                                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
                               ensure_ascii=False) + "\n")
            f.flush()


# ---------------------------------------------------------------- scoring
def parse_frames(line):
    fr = set()
    for m in FREF.finditer(line):
        lo = int(m.group(1)); hi = int(m.group(2) or lo)
        fr.update(range(min(lo, hi), max(lo, hi) + 1))
    return sorted(f for f in fr if 1 <= f <= NF)


def extract_claims(key, desc):
    claims = []
    for line in [l.strip(" -*•\t") for l in desc.splitlines() if l.strip()]:
        frames = parse_frames(line)
        # split a bullet into clauses so "no smile, brows raised" is handled per clause
        for clause in re.split(r"[;,.]| and | while | but ", line):
            if not clause.strip() or re.fullmatch(r"\s*\[?F[\d,\-\s F]*\]?\s*", clause):
                continue
            neg = bool(NEG.search(clause))
            base = {"key": key, "line": line, "clause": clause.strip(), "frames": frames, "negated": neg,
                    "specific_frames": 0 < len(frames) < NF}
            hit = False
            for rx, aus, name in LEX:
                if rx.search(clause):
                    hit = True
                    claims.append({**base, "type": "AU", "action": name, "aus": "|".join(aus), "kind": "present"})
            for rx, cols, kind, name in POSE:
                if rx.search(clause):
                    hit = True
                    claims.append({**base, "type": "pose", "action": name, "aus": "|".join(cols), "kind": kind,
                                   "negated": neg and kind != "still"})  # "no head movement" is itself a still claim
            for rx, cols, kind, name in GAZE:
                if rx.search(clause):
                    hit = True
                    claims.append({**base, "type": "gaze", "action": name, "aus": "|".join(cols), "kind": kind})
            if not hit and UNMAPPED.search(clause):
                claims.append({**base, "type": "unmapped", "action": "other head/gaze/posture/hands",
                               "aus": "", "kind": ""})
    return claims


# strictness levels: AU / blendshape intensity threshold, and matching angle thresholds in degrees
LEVELS = {0.3: {"deg": 5, "move": 4, "still": 10},
          0.5: {"deg": 10, "move": 8, "still": 6},
          0.7: {"deg": 15, "move": 12, "still": 3}}


def verdict(c, au_by_key, thr, src_key=None):
    """supported / unsupported / unverifiable for one claim against py-feat outputs on its cited frames."""
    if not c["aus"]:
        return "unverifiable", np.nan
    g = au_by_key.get(src_key or c["key"])
    if g is None:
        return "unverifiable", np.nan
    fr = c["frames"] or list(range(1, NF + 1))
    if c["type"] == "pose" and c["kind"] != "static":
        # movement needs >= 2 frames: widen a single cited frame to its neighbours
        fr = sorted({f + d for f in fr for d in (-1, 0, 1) if 1 <= f + d <= NF}) if len(fr) < 2 else fr
    g = g[g.frame.isin(fr) & (g.face == 1)]
    cols = [x for x in c["aus"].split("|") if x in g.columns]
    if g.empty or not cols or g[cols].isna().all().all():
        return "unverifiable", np.nan
    L = LEVELS[thr]
    if c["type"] == "pose":
        ang = g[cols].abs()
        rng = (g[cols].max() - g[cols].min())
        if c["kind"] == "still":
            v = float(rng.max()); ok = v <= L["still"]
        elif c["kind"] == "move":
            v = float(rng.max()); ok = v >= L["move"]
        elif c["kind"] == "move_or_static":  # "head turned" may be a held pose or a movement
            v = float(max(rng.max(), ang.max().max())); ok = rng.max() >= L["move"] or ang.max().max() >= L["deg"]
        else:
            v = float(ang.max().max()); ok = v >= L["deg"]
    else:  # AU or gaze blendshape intensity
        v = float(g[cols].max(axis=1).max())  # strongest cited frame, strongest unit of the set
        ok = v < thr if c["kind"] == "absent" else v >= thr
    if c["negated"]:  # "no smile" is supported when the unit stays low on the cited frames
        ok = not ok
    return ("supported" if ok else "unsupported"), v


def st_score(a):
    desc = [json.loads(l) for l in open(os.path.join(OUT, "desc.jsonl")) if l.strip()]
    au = pd.read_csv(os.path.join(OUT, "au_frames.csv"))
    # py-feat pose units: convert to degrees if the values look like radians
    pose_cols = [c for c in ("Pitch", "Roll", "Yaw") if c in au.columns]
    pose_unit = "deg"
    if pose_cols and np.nanpercentile(au[pose_cols].abs().values, 99) < 3.2:
        au[pose_cols] = np.degrees(au[pose_cols]); pose_unit = "rad->deg"
    au_by_key = {k: g for k, g in au.groupby("key")}
    claims = [c for d in desc for c in extract_claims(d["key"], d["desc"])]
    cdf = pd.DataFrame(claims)
    rng = np.random.default_rng(a.seed)
    keys = sorted(au_by_key)

    res = {"n_clips_described": len(desc), "n_claims": len(cdf),
           "claims_per_desc": round(len(cdf) / max(1, len(desc)), 2),
           "claims_by_type": cdf["type"].value_counts().to_dict(),
           "specific_frame_claim_rate": round(float(cdf["specific_frames"].mean()), 3),
           "pose_unit": pose_unit,
           "pose_abs_deg_p50_p90": {c: [round(float(np.nanpercentile(au[c].abs(), q)), 1) for q in (50, 90)]
                                    for c in pose_cols}}
    # emotion-word leakage (per description and per bullet line)
    lines = [(d["key"], l) for d in desc for l in d["desc"].splitlines() if l.strip()]
    res["emotion_leak_desc_rate"] = float(np.mean([bool(EMO.search(d["desc"])) for d in desc]))
    res["emotion_leak_line_rate"] = float(np.mean([bool(EMO.search(l)) for _, l in lines]))
    res["inference_phrase_desc_rate"] = float(np.mean([bool(INFER.search(d["desc"])) for d in desc]))
    res["frame_ref_line_rate"] = float(np.mean([bool(FREF.search(l)) for _, l in lines]))
    words = pd.Series([m.group(0).lower() for d in desc for m in EMO.finditer(d["desc"])])
    res["top_emotion_words"] = words.value_counts().head(10).to_dict()

    for thr in (0.3, 0.5, 0.7):
        v = [verdict(c, au_by_key, thr) for c in claims]
        cdf[f"verdict@{thr}"] = [x[0] for x in v]
        cdf["au_value"] = [x[1] for x in v]
        ver = cdf[cdf[f"verdict@{thr}"] != "unverifiable"]
        # control: same claims scored against a random OTHER clip's AUs (chance level of being "supported")
        # 20 independent shuffles, pooled, so the control is not a single noisy draw
        mask = (cdf[f"verdict@{thr}"] != "unverifiable").values
        ctrl_all, type_all = [], []
        for _ in range(20):
            cv = [verdict(c, au_by_key, thr, src_key=rng.choice([k for k in keys if k != c["key"]]))[0]
                  for c in claims]
            ctrl_all += list(np.array(cv)[mask]); type_all += list(cdf["type"].values[mask])
        ctrl = pd.Series(ctrl_all)
        cver = ctrl[ctrl != "unverifiable"]
        res[f"thr{thr}"] = {
            "verifiable": int(len(ver)),
            "unverifiable": int((cdf[f"verdict@{thr}"] == "unverifiable").sum()),
            "unverifiable_rate": round(float((cdf[f"verdict@{thr}"] == "unverifiable").mean()), 3),
            "hallucination_by_type": ver.groupby("type")[f"verdict@{thr}"].apply(
                lambda s: f"{(s == 'unsupported').mean():.3f} (n={len(s)})").to_dict(),
            "control_by_type": pd.DataFrame({"t": type_all, "v": ctrl_all}).query("v != 'unverifiable'").groupby("t")["v"].apply(
                lambda s: f"{(s == 'unsupported').mean():.3f} (n={len(s)})").to_dict(),
            "hallucination_rate": round(float((ver[f"verdict@{thr}"] == "unsupported").mean()), 3) if len(ver) else None,
            "shuffled_control_hallucination_rate": round(float((cver == "unsupported").mean()), 3) if len(cver) else None,
            "by_action": ver.groupby("action")[f"verdict@{thr}"].apply(lambda s: f"{(s == 'unsupported').mean():.2f} (n={len(s)})").to_dict(),
        }
    cdf["frames"] = cdf["frames"].apply(lambda x: ",".join(f"F{i}" for i in x))
    cdf.to_csv(os.path.join(OUT, "claims_scored.csv"), index=False)

    # 50 random claims for manual checking, with frame image paths
    pick = cdf[cdf.aus != ""].sample(n=min(50, (cdf.aus != "").sum()), random_state=a.seed).copy()
    pick["frame_images"] = [";".join(os.path.join(frame_dir(k), f"{f}.jpg") for f in (fr.split(",") if fr else [f"F{i}" for i in range(1, NF + 1)]))
                            for k, fr in zip(pick.key, pick.frames)]
    pick["manual_verdict"] = ""
    pick["manual_note"] = ""
    pick[["key", "frames", "clause", "line", "type", "action", "aus", "negated", "au_value", "verdict@0.5",
          "frame_images", "manual_verdict", "manual_note"]].to_csv(os.path.join(OUT, "manual_check_50.csv"), index=False)

    json.dump(res, open(os.path.join(OUT, "pilot_summary.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def st_verify(a):
    """Keep only description lines whose verifiable claims are all supported at thr=0.5.
    Lines with unsupported claims are dropped; lines with no verifiable claim are kept, marked '(unverified)'."""
    thr = 0.5
    desc = [json.loads(l) for l in open(os.path.join(OUT, "desc.jsonl")) if l.strip()]
    au = pd.read_csv(os.path.join(OUT, "au_frames.csv"))
    pc = [c for c in ("Pitch", "Roll", "Yaw") if c in au.columns]
    if pc and np.nanpercentile(au[pc].abs().values, 99) < 3.2:
        au[pc] = np.degrees(au[pc])
    au_by_key = {k: g for k, g in au.groupby("key")}
    out_rows, claim_rows, stats = [], [], {"kept": 0, "dropped": 0, "unverified": 0}
    for d in desc:
        kept = []
        for raw in [l for l in d["desc"].splitlines() if l.strip()]:
            line = raw.strip(" -*•\t")
            cl = extract_claims(d["key"], line)
            vs = [verdict(c, au_by_key, thr)[0] for c in cl]
            for c, v in zip(cl, vs):
                claim_rows.append({"key": d["key"], "line": line, "action": c["action"], "type": c["type"], "verdict": v})
            if "unsupported" in vs:
                stats["dropped"] += 1
            elif "supported" in vs:
                kept.append(line); stats["kept"] += 1
            else:
                kept.append(f"{line} (unverified)"); stats["unverified"] += 1
        out_rows.append({"key": d["key"], "desc": "\n".join(kept)})
    with open(os.path.join(OUT, "desc_verified.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(claim_rows).to_csv(os.path.join(OUT, "claims_verdict_05.csv"), index=False)
    print("verify lines:", stats, "empty descs after filter:", sum(not r["desc"] for r in out_rows))


def main():
    global OUT, CLIPS
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "frames", "au", "desc", "score", "verify"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT, help="output dir (default: outputs/pilot)")
    ap.add_argument("--clips", default=None, help="clip csv with a 'key' column (default: <out>/pilot_clips.csv)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", default="0/1", help="au stage: i/n slice of the remaining clips")
    a = ap.parse_args()
    OUT = os.path.expanduser(a.out)
    CLIPS = a.clips or os.path.join(OUT, "pilot_clips.csv")
    globals()[f"st_{a.stage}"](a)


if __name__ == "__main__":
    main()
