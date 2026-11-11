"""CLIP ViT-B/16 image embeddings of the same 8 sampled frames per clip (generic-visual-feature baseline).
Output: outputs/full/clip_b16_frames.npy (N, 8, 512) in clips.csv order (zeros if a frame is missing)."""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = os.path.expanduser("~/work/msa-pilot")
F = os.path.join(ROOT, "outputs/full")
M = os.path.join(ROOT, "models/clip-vit-base-patch16")
keys = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str).key.tolist()
model = CLIPModel.from_pretrained(M, dtype=torch.float16).cuda().eval()
proc = CLIPProcessor.from_pretrained(M)
out = np.zeros((len(keys), 8, model.config.projection_dim), np.float16)
paths = [(i, f, os.path.join(F, "frames", k.replace("/", "__"), f"F{f + 1}.jpg")) for i, k in enumerate(keys) for f in range(8)]
paths = [p for p in paths if os.path.exists(p[2])]
with torch.no_grad():
    for b in range(0, len(paths), 128):
        chunk = paths[b:b + 128]
        ims = [Image.open(p).convert("RGB") for _, _, p in chunk]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        emb = model.get_image_features(pixel_values=px)
        emb = getattr(emb, "pooler_output", emb)
        e = emb.float().cpu().numpy()
        for (i, f, _), v in zip(chunk, e):
            out[i, f] = v
        if b % 6400 == 0:
            print(b, "/", len(paths), flush=True)
np.save(os.path.join(F, "clip_b16_frames.npy"), out)
print("saved", out.shape)
