"""Faithful DEVA-style visual descriptions on the 10 fps py-feat AU tracks (dense_au.csv), following DEVA's EDG rules:
an AU is 'present' in a frame if its intensity >= 0.5 (py-feat has no binary AU_c); a candidate if present in >= 3 CONSECUTIVE 10 fps samples (~0.3 s; DEVA: >= 3 video frames ~0.1 s, so ours is stricter);
candidates ranked by duration (longest consecutive run, in frames) and the top-k kept; fixed template. Audio text is reused from
deva_text.jsonl (4 prosodic values, tertiles, fixed template). Writes deva_text_dense_k{3,5,8}.jsonl (k chosen later on validation).
Usage: python deva_dense.py"""
import json
import os

import numpy as np
import pandas as pd

# same Chinese AU phrases / templates as the original DEVA-style baseline (deva_text.py; not imported because it runs on import)
AU_TXT = {"AU01": "抬起内眉", "AU02": "抬起外眉", "AU04": "低眉", "AU05": "抬起上眼睑", "AU06": "抬起脸颊",
          "AU07": "紧闭眼皮", "AU09": "皱鼻头", "AU10": "扬起上唇", "AU12": "拉唇角", "AU15": "下压唇角",
          "AU20": "拉伸嘴唇", "AU23": "抿紧嘴唇", "AU25": "分开嘴唇", "AU26": "掉下巴", "AU28": "嘬嘴", "AU43": "眨眼"}
T = {"v_pre": "说话人做出了这样的表情：", "v_none": "没有明显的表情动作", "v_noface": "说话人做出了这样的表情：未检测到人脸。", "sep": "，", "end": "。"}

F = os.path.expanduser("~/work/msa-pilot/outputs/full")
clips = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str)
au = pd.read_csv(os.path.join(F, "dense_au.csv"))
audio = {json.loads(l)["key"]: json.loads(l)["audio_text"] for l in open(os.path.join(F, "deva_text.jsonl"))}
dur = {}  # key -> list of (duration, AU)
for k, g in au[au.face == 1].groupby("key"):
    g = g.sort_values("t"); tt = g.t.values; cand = []
    for c in AU_TXT:
        if c not in g:
            continue
        on = g[c].values >= 0.5; best = run = 0; prev = None
        for t_i, o in zip(tt, on):  # consecutive = adjacent 10 fps samples (time gap <= 0.15 s)
            run = (run + 1 if (o and prev is not None and t_i - prev <= 0.15 and run > 0) else (1 if o else 0)); prev = t_i
            best = max(best, run)
        if best >= 3:
            cand.append((best, c))
    dur[k] = sorted(cand, key=lambda x: -x[0])
stats = []
for kk in (3, 5, 8):
    with open(os.path.join(F, f"deva_text_dense_k{kk}.jsonl"), "w") as f:
        for key in clips.key:
            if key not in dur:
                v = T["v_noface"]
            else:
                sel = [c for _, c in dur[key][:kk]]
                v = T["v_pre"] + (T["sep"].join(AU_TXT[c] for c in sel) if sel else T["v_none"]) + T["end"]
            f.write(json.dumps({"key": key, "vision_text": v, "audio_text": audio.get(key, "")}, ensure_ascii=False) + "\n")
n = np.array([len(v) for v in dur.values()])
print("clips with face", len(dur), "| candidates per clip: mean %.2f median %d | zero-candidate clips %d" % (n.mean(), np.median(n), int((n == 0).sum())))
print("example:", next(iter(open(os.path.join(F, "deva_text_dense_k5.jsonl")))).strip()[:200])
