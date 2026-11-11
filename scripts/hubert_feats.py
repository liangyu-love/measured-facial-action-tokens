"""Utterance-level speech features from Chinese HuBERT (TencentGameMate/chinese-hubert-base, frozen): for each clip wav (16 kHz mono),
mean-pool every layer's hidden states over time -> (13 layers, 768). Saved as float16 npy in clips.csv order + a key list.
Usage: MSA_DATA=outputs/full python hubert_feats.py"""
import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from train_bert import FULL

M = os.path.expanduser("~/work/msa-pilot/models/chinese-hubert-base")
c = pd.read_csv(os.path.join(FULL, "clips.csv"))
fe = Wav2Vec2FeatureExtractor.from_pretrained(M); m = HubertModel.from_pretrained(M).cuda().eval()
out = np.zeros((len(c), 13, 768), np.float16); miss = 0
with torch.no_grad():
    for i, k in enumerate(c.key):
        p = os.path.join(FULL, "wav", k.replace("/", "__") + ".wav")
        if not os.path.exists(p):
            miss += 1; continue
        w, sr = sf.read(p, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        w = w[: 16000 * 20] if len(w) > 16000 * 20 else w  # cap 20 s
        if len(w) < 400:
            miss += 1; continue
        x = fe(w, sampling_rate=16000, return_tensors="pt").input_values.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = m(x, output_hidden_states=True).hidden_states
        out[i] = torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy().astype(np.float16)
        if i % 500 == 0:
            print(i, flush=True)
np.save(os.path.join(FULL, "hubert_layers.npy"), out)
print("done", out.shape, "missing", miss)
