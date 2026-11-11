"""Text-only unimodal expert (CH-SIMS v2): BERT [CLS] -> Linear(768,128)+LN (z_t) -> Linear(128,1) predicting label_T.
Train split only, 6 epochs, lr 2e-5, bs 16, warmup 10%, best epoch by valid MAE on label_T. Mirrors evq.py's visual expert.
Output: outputs/full/results_texp/seed<s>.pt (state_dict) + seed<s>.json (valid/test label_T metrics).
Usage: python texp.py --seed 1
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_bert import BERT, FULL, metrics

OUT = os.path.join(os.path.expanduser("~/work/msa-pilot/outputs/full"), os.environ.get("EXPERT_DIR_T", "results_texp"))


class TextExpert(nn.Module):
    def __init__(self, drop=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BERT)
        self.proj = nn.Sequential(nn.Linear(self.enc.config.hidden_size, 128), nn.LayerNorm(128))
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(128, 1)
        self.pol = nn.Linear(128, 1)  # optional polarity head (sign BCE), training aid only

    def encode(self, ids, att, tt):
        return self.proj(self.drop(self.enc(input_ids=ids, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0]))

    def forward(self, ids, att, tt):
        return self.head(self.encode(ids, att, tt)).squeeze(-1)


def pol_acc(y, p, thr=0.1):
    """polarity accuracy of a regression reading: sign(pred) == sign(label) on clips with |label| > thr"""
    y, p = np.asarray(y, float), np.asarray(p, float); m = np.abs(y) > thr
    return float((np.sign(p[m]) == np.sign(y[m])).mean()) if m.any() else float("nan")


def tokenize(df):
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(list(df.text), truncation=True, max_length=128)
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = [[0] * len(x) for x in enc["input_ids"]]

    def batch(idx):
        L = max(len(enc["input_ids"][i]) for i in idx)
        p = lambda k, v: torch.tensor([enc[k][i] + [v] * (L - len(enc[k][i])) for i in idx])
        return p("input_ids", tok.pad_token_id).cuda(), p("attention_mask", 0).cuda(), p("token_type_ids", 0).cuda()
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pol", type=float, default=0.0, help="weight of the sign-BCE polarity head (|label_T| > 0.1)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"seed{a.seed}{a.tag}.pt")
    if os.path.exists(out):
        print("exists"); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    batch = tokenize(df)
    Y = torch.tensor(df.label_T.values, dtype=torch.float32)
    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    if os.environ.get("PRE_SMOKE"):
        sp = {k: v[:int(os.environ["PRE_SMOKE"])] for k, v in sp.items()}
    model = TextExpert().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(sp["train"]) + 15) // 16 * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def pred(idx):
        model.eval(); out_ = []
        for i in range(0, len(idx), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_.append(model(*batch(idx[i:i + 64])).float().cpu())
        model.train(); return torch.cat(out_).numpy()

    best, hist = 1e9, []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), 16):
            j = perm[i:i + 16]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.encode(*batch(j)); y = model.head(z).squeeze(-1); lg = model.pol(z).squeeze(-1)
            loss = nn.functional.mse_loss(y.float(), Y[j].cuda())
            if a.pol:
                yy = Y[j].cuda(); mk = yy.abs() > 0.1
                if mk.any():
                    loss = loss + a.pol * nn.functional.binary_cross_entropy_with_logits(lg.float()[mk], (yy[mk] > 0).float())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(Y[sp["valid"]].numpy(), pred(sp["valid"])); hist.append(vm)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[texp seed{a.seed}] ep{ep} valid label_T mae={vm['mae']:.4f} corr={vm['corr']:.4f}", flush=True)
    model.load_state_dict(state)
    te = df.iloc[sp["test"]]; pt = pred(sp["test"]); cm = (te.category == "conflict").values
    va = df.iloc[sp["valid"]]; pv = pred(sp["valid"]); vcm = (va.category == "conflict").values
    res = {"seed": a.seed, "lr": a.lr, "epochs": a.epochs, "pol": a.pol, "best_epoch": best_ep, "valid_labelT": hist[best_ep], "test_labelT": metrics(te.label_T.values, pt),
           "test_labelT_conflict": metrics(te.label_T.values[cm], pt[cm]),
           "valid_conflict_polacc": pol_acc(va.label_T.values[vcm], pv[vcm]), "test_conflict_polacc": pol_acc(te.label_T.values[cm], pt[cm]),
           "test_polacc": pol_acc(te.label_T.values, pt)}
    torch.save(state, out)
    json.dump(res, open(out.replace(".pt", ".json"), "w"), indent=2)
    print(json.dumps(res["test_labelT"]))


if __name__ == "__main__":
    main()
