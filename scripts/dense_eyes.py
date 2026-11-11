"""Eye/gaze blendshapes at ORIGINAL resolution for the dense (10 fps) frames of pilot clips. CPU only.
Uses the face box found by dense_pose.py (640 px pass), crops a square 2.2x box from the full-res frame
(so face pixels are native resolution) and re-runs py-feat on the crop.
Output: outputs/pilot_v3b_scoring/dense_eyes.csv (resumable per clip)
"""
import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
ROOT = os.path.expanduser("~/work/msa-pilot")
V3 = os.path.join(ROOT, "outputs/pilot_v3_scoring")
O = os.path.join(ROOT, "outputs/pilot_v3b_scoring")
RAW = os.path.join(ROOT, "data/chsims2/Raw")
COLS = ("eyeBlink", "eyeLookDown", "eyeLookUp", "eyeLookIn", "eyeLookOut")


def main():
    from PIL import Image, ImageOps
    from torchcodec.decoders import VideoDecoder
    from feat import Detectorv2
    parts = os.path.join(O, "eyes_parts"); os.makedirs(parts, exist_ok=True)
    dense = pd.read_csv(os.path.join(V3, "dense_pose.csv"))
    dense = dense[dense.face == 1]
    det = Detectorv2(device="cpu", identity_model=None)
    t0 = time.time()
    keys = sorted(dense.key.unique())
    for n, (k) in enumerate(keys, 1):
        fp = os.path.join(parts, k.replace("/", "__") + ".csv")
        if os.path.exists(fp):
            continue
        g = dense[dense.key == k].sort_values("frame_idx")
        dec = VideoDecoder(os.path.join(RAW, f"{k}.mp4"))
        frames = dec.get_frames_at(indices=g.frame_idx.astype(int).tolist()).data
        rows = []
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for (_, r), t in zip(g.iterrows(), frames):
                im = Image.fromarray(t.permute(1, 2, 0).numpy()); W, H = im.size
                cx, cy, fw = r.cx * W, r.cy * H, r.fw * W
                s = 1.1 * fw  # half side of a 2.2x square
                box = (int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s)))
                # pad to square + fixed size so the batch detector gets equal-size inputs (face ~250 px)
                crop = ImageOps.pad(im.crop(box), (560, 560), color=(0, 0, 0))
                p = os.path.join(td, f"{int(r.frame_idx):06d}.jpg"); crop.save(p, quality=95); paths.append(p)
            fex = pd.DataFrame(det.detect(paths, data_type="image", batch_size=16))
            for (_, r), p in zip(g.iterrows(), paths):
                h = fex[(fex["input"].astype(str) == p) & fex.FaceRectWidth.notna()]
                o = {"key": k, "frame_idx": int(r.frame_idx), "t": r.t, "eye_face": int(len(h) > 0)}
                if len(h):
                    b = h.assign(a=h.FaceRectWidth * h.FaceRectHeight).sort_values("a").iloc[-1]
                    for c in COLS:
                        cs = [x for x in (f"{c}Left", f"{c}Right") if x in b.index]
                        o[c] = float(max(b[x] for x in cs)) if cs else np.nan
                    o["crop_face_px"] = float(b.FaceRectWidth)
                rows.append(o)
        pd.DataFrame(rows).to_csv(fp, index=False)
        if n % 10 == 0:
            print(f"[{time.strftime('%T')}] {n}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    df = pd.concat([pd.read_csv(os.path.join(parts, p)) for p in sorted(os.listdir(parts))])
    df.to_csv(os.path.join(O, "dense_eyes.csv"), index=False)
    print("eye rows", len(df), "face rate", round(df.eye_face.mean(), 3))


if __name__ == "__main__":
    main()
