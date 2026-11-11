"""Densified AU evidence for CH-SIMS v2: face crops (square, 2.2x face box, from the full-res frame, padded to 448)
along the existing 10 fps face tracks (outputs/full/dense/dense_pose.csv) + crops at the original 8 sampled frames
(nearest track box). py-feat (largest face in the crop): AUs + 7-class emotion probabilities + face flag.
Outputs (after `merge`): outputs/full/dense_au.csv (key, frame_idx, t, ...), outputs/full/crop8_au.csv (key, frame 1..8, ...)
Usage: python dense_au.py extract --shard i/n  |  python dense_au.py merge
"""
import json
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/work/msa-pilot")
F = os.path.join(ROOT, "outputs/full")
RAW = os.path.join(ROOT, "data/chsims2/Raw")
EMO = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
SIDE = 448


def extract(shard):
    import torch
    from PIL import Image, ImageOps
    from torchcodec.decoders import VideoDecoder
    from feat import Detectorv2
    torch.set_num_threads(4)
    si, sn = (int(x) for x in shard.split("/"))
    parts = os.path.join(F, "denseau_parts"); os.makedirs(parts, exist_ok=True)
    tracks = pd.read_csv(os.path.join(F, "dense/dense_pose.csv"))
    tracks = {k: g.sort_values("frame_idx") for k, g in tracks[tracks.face == 1].groupby("key")}
    keys = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str).key.tolist()[si::sn]
    det = Detectorv2(device="cuda", identity_model=None)
    n_done = 0
    for k in keys:
        fp = os.path.join(parts, k.replace("/", "__") + ".csv")
        if os.path.exists(fp) or k not in tracks:
            continue
        tr = tracks[k]
        info = json.load(open(os.path.join(F, "frames", k.replace("/", "__"), "frames.json")))
        dense_idx = tr.frame_idx.astype(int).tolist()
        orig_idx = [int(x) for x in info["idx"]]
        want = sorted(set(dense_idx) | set(orig_idx))
        dec = VideoDecoder(os.path.join(RAW, f"{k}.mp4"))
        fps = dec.metadata.average_fps or 25.0
        frames = dec.get_frames_at(indices=want).data
        ti = tr.frame_idx.values
        rows = []
        with tempfile.TemporaryDirectory() as td:
            paths, meta = [], []
            for fi, t in zip(want, frames):
                r = tr.iloc[int(np.abs(ti - fi).argmin())]  # nearest track box
                im = Image.fromarray(t.permute(1, 2, 0).numpy()); W, H = im.size
                cx, cy, s = r.cx * W, r.cy * H, 1.1 * r.fw * W
                box = (int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s)))
                p = os.path.join(td, f"{fi:06d}.jpg")
                ImageOps.pad(im.crop(box), (SIDE, SIDE), color=(0, 0, 0)).save(p, quality=95)
                paths.append(p); meta.append(fi)
            fex = pd.DataFrame(det.detect(paths, data_type="image", batch_size=16))
            aucols = [c for c in fex.columns if re.fullmatch(r"AU\d+", str(c))]
            fex["area"] = fex.FaceRectWidth * fex.FaceRectHeight
            for p, fi in zip(paths, meta):
                g = fex[(fex["input"].astype(str) == p) & fex.FaceRectWidth.notna()]
                o = {"key": k, "frame_idx": fi, "t": round(fi / fps, 3), "is_dense": int(fi in set(dense_idx)),
                     "orig_pos": (orig_idx.index(fi) + 1) if fi in orig_idx else 0, "face": int(len(g) > 0)}
                if len(g):
                    b = g.sort_values("area").iloc[-1]
                    o.update({c: float(b[c]) for c in aucols}); o.update({e: float(b[e]) for e in EMO})
                    o["face_score"] = float(b.FaceScore)
                rows.append(o)
        pd.DataFrame(rows).to_csv(fp, index=False)
        n_done += 1
        if n_done % 100 == 0:
            print(f"shard {shard}: {n_done}/{len(keys)}", flush=True)
    print("shard done", shard, n_done)


def merge():
    parts = os.path.join(F, "denseau_parts")
    d = pd.concat([pd.read_csv(os.path.join(parts, p)) for p in sorted(os.listdir(parts))], ignore_index=True)
    dense = d[d.is_dense == 1].drop(columns=["is_dense", "orig_pos"])
    crop8 = d[d.orig_pos > 0].copy()
    crop8["frame"] = crop8.orig_pos
    crop8 = crop8.drop(columns=["is_dense", "orig_pos"]).drop_duplicates(["key", "frame"])
    dense.to_csv(os.path.join(F, "dense_au.csv"), index=False)
    crop8.to_csv(os.path.join(F, "crop8_au.csv"), index=False)
    print("dense rows", len(dense), "clips", dense.key.nunique(), "frames/clip median", int(dense.groupby("key").size().median()),
          "face", round(dense.face.mean(), 3), "| crop8 rows", len(crop8), "face", round(crop8.face.mean(), 3))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[sys.argv.index("--shard") + 1] if "--shard" in sys.argv else "0/1")
    else:
        merge()
