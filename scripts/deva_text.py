"""DEVA-style (arXiv 2412.10460) Chinese emotional descriptions, re-implemented from the paper / public repo description.
Deviations (documented): py-feat AUs on 8 sampled frames instead of OpenFace AU*_c at full frame rate
(presence = intensity >= 0.5, "sustained" = present in >= 2 consecutive sampled frames instead of >= 3 video frames);
eGeMAPS functionals instead of IS10_paraling; tertile bins fit on TRAIN only (repo bins use the whole dataset).
Output: outputs/full/deva_text.jsonl with fields vision_text, audio_text.
"""
import json
import os

import numpy as np
import pandas as pd

F = os.path.join(os.path.expanduser("~/work/msa-pilot"), os.environ.get("MSA_DATA", "outputs/full"))
LANG = os.environ.get("DEVA_LANG", "zh")  # "en" for CMU-MOSI (DEVA used English templates there)
AU_ZH = {"AU01": "抬起内眉", "AU02": "抬起外眉", "AU04": "低眉", "AU05": "抬起上眼睑", "AU06": "抬起脸颊",
         "AU07": "紧闭眼皮", "AU09": "皱鼻头", "AU10": "扬起上唇", "AU12": "拉唇角", "AU15": "下压唇角",
         "AU20": "拉伸嘴唇", "AU23": "抿紧嘴唇", "AU25": "分开嘴唇", "AU26": "掉下巴", "AU28": "嘬嘴",
         "AU43": "眨眼"}  # DEVA uses AU45 (blink); py-feat provides AU43 (eyes closed)
MAX_AU = 8
AU_EN = {"AU01": "raise inner brow", "AU02": "raise outer brow", "AU04": "lower brow", "AU05": "raise upper lid",
         "AU06": "raise cheek", "AU07": "tighten lid", "AU09": "wrinkle nose", "AU10": "raise upper lip",
         "AU12": "pull lip corner", "AU15": "depress lip corner", "AU20": "stretch lip", "AU23": "tighten lip",
         "AU25": "part lip", "AU26": "drop jaw", "AU28": "suck lip", "AU43": "blink"}  # DEVA's English AU phrases
EN = LANG == "en"
AU_TXT = AU_EN if EN else AU_ZH
T = {"v_pre": ("The speaker made such an expression: " if EN else "说话人做出了这样的表情："),
     "v_none": ("no obvious facial action" if EN else "没有明显的表情动作"),
     "v_noface": ("The speaker made such an expression: face not detected." if EN else "说话人做出了这样的表情：未检测到人脸。"),
     "a_pre": ("The speaker's voice: " if EN else "说话人的声音特点："), "sep": (", " if EN else "，"), "end": ("." if EN else "。")}

clips = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str)[["key", "split"]]
au = pd.read_csv(os.path.join(F, "au_frames.csv"))
pros = pd.read_csv(os.path.join(F, "prosody_feats.csv")).merge(clips, on="key")

# ---- audio: tertiles on train only
A = ({"loud": ("low volume", "medium volume", "high volume"), "jitter": ("low jitter", "medium jitter", "high jitter"),
      "shimmer": ("low shimmer", "medium shimmer", "high shimmer"), "pitch": ("low pitch", "medium pitch", "high pitch")} if EN else
     {"loud": ("音量较小", "音量中等", "音量较大"), "jitter": ("频率变化较小", "频率变化中等", "频率变化较大"),
      "shimmer": ("振幅变化较小", "振幅变化中等", "振幅变化较大"), "pitch": ("音调较低", "音调中等", "音调较高")})
tr = pros[pros.split == "train"]
q = {c: tr[c].quantile([1 / 3, 2 / 3]).values for c in A}
audio = {}
for r in pros.itertuples(index=False):
    parts = [A[c][int(getattr(r, c) >= q[c][0]) + int(getattr(r, c) >= q[c][1])] for c in A]
    audio[r.key] = T["a_pre"] + T["sep"].join(parts) + T["end"]

# ---- vision: AUs present in >= 2 consecutive sampled frames, ranked by count, max 8
vision = {}
for k, g in au[au.face == 1].groupby("key"):
    g = g.sort_values("frame")
    sel = []
    for c in AU_TXT:
        if c not in g:
            continue
        on = (g[c].values >= 0.5).astype(int)
        fr = g.frame.values
        run, best, prev = 0, 0, None
        for f_i, o in zip(fr, on):
            if not o:
                run = 0
            elif run > 0 and prev is not None and f_i == prev + 1:
                run += 1
            else:
                run = 1
            prev = f_i
            best = max(best, run)
        if best >= 2:
            sel.append((int(on.sum()), c))
    sel = [c for _, c in sorted(sel, key=lambda x: -x[0])][:MAX_AU]
    vision[k] = T["v_pre"] + (T["sep"].join(AU_TXT[c] for c in sel) if sel else T["v_none"]) + T["end"]

with open(os.path.join(F, "deva_text.jsonl"), "w") as f:
    for k in clips.key:
        f.write(json.dumps({"key": k, "vision_text": vision.get(k, T["v_noface"]),
                            "audio_text": audio.get(k, "")}, ensure_ascii=False) + "\n")
print("wrote", len(clips), "| example:", vision.get(clips.key[0]), audio.get(clips.key[0]))
