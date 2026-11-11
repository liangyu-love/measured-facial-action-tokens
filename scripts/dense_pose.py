"""Dense (10 fps) head pose + blink/gaze blendshapes for pilot clips, CPU only.
Face continuity: start from the largest face, then follow the face nearest the previous one.
Output: outputs/pilot_v3_scoring/dense_pose.csv  (resumable via per-clip parts)
Usage: CUDA_VISIBLE_DEVICES= nice -n 19 python dense_pose.py [--limit N]
"""
import argparse
import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
ROOT = os.path.expanduser("~/work/msa-pilot")
OUT = os.path.join(ROOT, "outputs/pilot_v3_scoring")
RAW = os.path.join(ROOT, "data/chsims2/Raw")
FPS = 10
W = 640


def pick(g, prev):
    g = g.assign(area=g.FaceRectWidth * g.FaceRectHeight,
                 cx=g.FaceRectX + g.FaceRectWidth / 2, cy=g.FaceRectY + g.FaceRectHeight / 2)
    if prev is None or len(g) == 1:
        return g.sort_values("area").iloc[-1]
    d = np.hypot(g.cx - prev[0], g.cy - prev[1]) / max(prev[2], 1)
    big = g.area.max()
    g = g[g.area >= 0.35 * big]  # ignore tiny background faces
    return g.loc[d[g.index].idxmin()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clips", default=os.path.join(ROOT, "outputs/pilot/pilot_clips.csv"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th clip starting at i")
    a = ap.parse_args()
    out_dir = a.out
    si, sn = (int(x) for x in a.shard.split("/"))
    from PIL import Image
    from torchcodec.decoders import VideoDecoder
    from feat import Detectorv2
    parts = os.path.join(out_dir, "dense_parts")
    os.makedirs(parts, exist_ok=True)
    keys = pd.read_csv(a.clips, dtype=str).key.tolist()
    keys = keys[: a.limit or None][si::sn]
    det = Detectorv2(device=a.device, identity_model=None)
    t_all = time.time()
    for n, k in enumerate(keys, 1):
        fp = os.path.join(parts, k.replace("/", "__") + ".csv")
        if os.path.exists(fp):
            continue
        dec = VideoDecoder(os.path.join(RAW, f"{k}.mp4"))
        fps = dec.metadata.average_fps or 25.0
        nfr = dec.metadata.num_frames
        idx = np.unique(np.round(np.arange(0, nfr, fps / FPS)).astype(int).clip(0, nfr - 1)).tolist()
        batch = dec.get_frames_at(indices=idx).data
        rows = []
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, t in zip(idx, batch):
                im = Image.fromarray(t.permute(1, 2, 0).numpy())
                s = W / im.width
                im = im.resize((W, round(im.height * s)))
                p = os.path.join(td, f"{i:06d}.jpg"); im.save(p, quality=90); paths.append(p)
            H = im.height
            fex = pd.DataFrame(det.detect(paths, data_type="image", batch_size=16))
            prev = None
            for i, p in zip(idx, paths):
                g = fex[(fex["input"].astype(str) == p) & fex.FaceRectWidth.notna()]
                r = {"key": k, "frame_idx": i, "t": round(i / fps, 3), "face": int(len(g) > 0)}
                if len(g):
                    b = pick(g, prev)
                    prev = (b.FaceRectX + b.FaceRectWidth / 2, b.FaceRectY + b.FaceRectHeight / 2, b.FaceRectWidth)
                    r.update({"Pitch": b.Pitch, "Roll": b.Roll, "Yaw": b.Yaw,
                              "cx": prev[0] / W, "cy": prev[1] / H, "fw": b.FaceRectWidth / W,
                              "n_faces": len(g)})
                    for bs in ("eyeBlink", "eyeLookDown", "eyeLookUp", "eyeLookIn", "eyeLookOut"):
                        cs = [c for c in (f"{bs}Left", f"{bs}Right") if c in b.index]
                        r[bs] = float(max(b[c] for c in cs)) if cs else np.nan
                rows.append(r)
        pd.DataFrame(rows).to_csv(fp, index=False)
        if n % 10 == 0 or n == len(keys):
            print(f"[{time.strftime('%T')}] {n}/{len(keys)} clips, {time.time() - t_all:.0f}s", flush=True)
    df = pd.concat([pd.read_csv(os.path.join(parts, p)) for p in sorted(os.listdir(parts))])
    df.to_csv(os.path.join(out_dir, "dense_pose.csv"), index=False)
    print("dense rows", len(df), "clips", df.key.nunique())


if __name__ == "__main__":
    main()
