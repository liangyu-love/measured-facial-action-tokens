"""External text expert (no target-dataset labels): BERT of the target language fine-tuned ONLY on a public text sentiment corpus
(English SST-2 -> MOSI / MOSEI; Chinese ChnSentiCorp -> CH-SIMS v2), binary cross-entropy, 2 epochs, lr 2e-5, bs 32,
best epoch on the corpus' own validation split. Reading on every clip transcript: y_t = 2*P(pos) - 1 in [-1, 1],
uncertainty = MC-dropout std (8 passes). Merges with expert_ext_visual.csv -> <MSA_DATA>/expert_ext.csv (key, yv_ext, yt_ext, sv_ext, st_ext).
Corpora are read from ~/work/msa-pilot/data/public/{sst2,chnsenticorp}/*.parquet (downloaded separately, research use only).
Usage: MSA_DATA=outputs/full ENCODER_DIR=.../bert-base-chinese python ext_text.py --corpus chnsenticorp"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_bert import BERT, FULL

PUB = os.path.expanduser("~/work/msa-pilot/data/public")


def read_corpus(name):
    fs = {s: glob.glob(os.path.join(PUB, name, "**", f"{s}-*.parquet"), recursive=True) for s in ("train", "validation")}
    tr, va = pd.read_parquet(fs["train"][0]), pd.read_parquet(fs["validation"][0])
    tcol = "sentence" if "sentence" in tr.columns else "text"
    return list(tr[tcol]), tr["label"].values.astype(float), list(va[tcol]), va["label"].values.astype(float)


class Clf(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = AutoModel.from_pretrained(BERT); self.drop = nn.Dropout(0.1); self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, **b):
        return self.head(self.drop(self.enc(**b).last_hidden_state[:, 0])).squeeze(-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", required=True, choices=["sst2", "chnsenticorp"]); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = lambda xs: {k: v.cuda() for k, v in tok(xs, truncation=True, max_length=128, padding=True, return_tensors="pt").items()}
    Xtr, ytr, Xva, yva = read_corpus(a.corpus)
    m = Clf().cuda(); opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
    steps = (len(Xtr) + 31) // 32 * 2; sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    g = torch.Generator().manual_seed(a.seed); best, state = -1, None

    @torch.no_grad()
    def prob(xs, mc=False):
        out, sd = [], []
        for i in range(0, len(xs), 64):
            b = enc(xs[i:i + 64])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m.eval(); out.append(torch.sigmoid(m(**b).float()).cpu())
                if mc:
                    m.train(); sd.append(torch.stack([2 * torch.sigmoid(m(**b).float()) - 1 for _ in range(8)]).std(0).cpu()); m.eval()
        return torch.cat(out).numpy(), (torch.cat(sd).numpy() if mc else None)

    for ep in range(2):
        perm = torch.randperm(len(Xtr), generator=g).numpy()
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(**enc([Xtr[k] for k in j]))
            loss = nn.functional.binary_cross_entropy_with_logits(lg.float(), torch.tensor(ytr[j], dtype=torch.float32).cuda())
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        acc = ((prob(Xva)[0] > 0.5) == (yva > 0.5)).mean(); print(f"[{a.corpus}] ep{ep} corpus-valid acc {acc:.4f}", flush=True)
        if acc > best:
            best, state = acc, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    c = pd.read_csv(os.path.join(FULL, "clips.csv"))
    p, sd = prob(list(c.text.astype(str)), mc=True)
    vis = pd.read_csv(os.path.join(FULL, "expert_ext_visual.csv")).set_index("key").reindex(c.key)
    out = pd.DataFrame({"key": c.key, "yv_ext": vis.yv_ext.values, "yt_ext": 2 * p - 1, "sv_ext": vis.sv_ext.values, "st_ext": sd})
    out.to_csv(os.path.join(FULL, "expert_ext.csv"), index=False)
    print(f"corpus valid acc {best:.4f}; wrote {len(out)} readings")
    if c.label_T.notna().all():
        sg = np.sign
        for s in ("train", "valid", "test"):
            mm = (c.split == s).values; y = c.label_T.values[mm]; q = out.yt_ext.values[mm]; nz = np.abs(y) > 0.1
            print(f"{s}: Corr(yt_ext, label_T) {pearsonr(q, y)[0]:.3f} | polarity acc {(sg(q[nz]) == sg(y[nz])).mean():.3f}")
        from sklearn.metrics import roc_auc_score
        t = c.split == "test"; strict = ((sg(c.label_T[t]) * sg(c.label_V[t])) < 0).astype(int)
        print(f"test AUROC |yv_ext - yt_ext| for strict T-V conflict: {roc_auc_score(strict, (out.yv_ext[t] - out.yt_ext[t]).abs()):.3f} (n={int(strict.sum())})")


if __name__ == "__main__":
    main()
