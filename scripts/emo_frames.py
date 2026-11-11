"""py-feat 7-class emotion probabilities for the same 8 frames per clip (largest face), resumable in parts.
Output: outputs/full/emo_frames.csv (key, frame, face, Neutral..Anger)"""
import os

import pandas as pd
import torch

ROOT = os.path.expanduser("~/work/msa-pilot")
F = os.path.join(ROOT, os.environ.get("MSA_DATA", "outputs/full"))
EMO = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]


def main():
    from feat import Detectorv2
    parts = os.path.join(F, "emo_parts"); os.makedirs(parts, exist_ok=True)
    done = set()
    for p in os.listdir(parts):
        done |= set(pd.read_csv(os.path.join(parts, p), usecols=["key"], dtype=str).key)
    keys = [k for k in pd.read_csv(os.path.join(F, "clips.csv"), dtype=str).key if k not in done]
    det = Detectorv2(device="cuda", identity_model=None)
    rows = []
    for n, k in enumerate(keys, 1):
        d = os.path.join(F, "frames", k.replace("/", "__"))
        paths = [os.path.join(d, f"F{i}.jpg") for i in range(1, 9)]
        paths = [p for p in paths if os.path.exists(p)]
        fex = pd.DataFrame(det.detect(paths, data_type="image", batch_size=8))
        fex["area"] = fex.FaceRectWidth * fex.FaceRectHeight
        for i, p in enumerate(paths, 1):
            g = fex[(fex["input"].astype(str) == p) & fex.FaceRectWidth.notna()]
            r = {"key": k, "frame": i, "face": int(len(g) > 0)}
            if len(g):
                b = g.sort_values("area").iloc[-1]
                r.update({e: float(b[e]) for e in EMO})
            rows.append(r)
        if n % 200 == 0 or n == len(keys):
            pd.DataFrame(rows).to_csv(os.path.join(parts, f"part_{len(os.listdir(parts)):05d}.csv"), index=False)
            rows = []
            print(f"{n}/{len(keys)}", flush=True)
    df = pd.concat([pd.read_csv(os.path.join(parts, p)) for p in sorted(os.listdir(parts))]).drop_duplicates(["key", "frame"])
    df.to_csv(os.path.join(F, "emo_frames.csv"), index=False)
    print("emo rows", len(df))


if __name__ == "__main__":
    main()
