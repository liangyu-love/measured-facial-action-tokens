"""DIFF v2: refined text-face discrepancy token on top of au_soft (bert-base-chinese, 6 ep, lr 2e-5, bs 16, valid MAE).
  t = [CLS] of a text-only forward pass of the same BERT (stop-grad);  a = mean of the AU soft-token projections
  z_t = P_t(t), z_v = P_v(a)  (d=128);  aux heads: z_t -> label_T, z_v -> label_V  (MSE, lambda1 = 0.2, train only)
  m = [z_v - z_t, |z_v - z_t|, z_v * z_t] -> linear -> 768 -> one [DIFF] soft token appended after the AU tokens
  optional conflict head on m -> |label_T - label_V|  (MSE, lambda2 = 0.1)
Variants: diff2_nosup (no aux), diff2_uni (lambda1), diff2_full (lambda1 + lambda2). --stat adds the A clip-level token.
Diagnostics (test): shuffle [DIFF] across samples -> delta metrics; Spearman(||z_v - z_t||, |label_T - label_V|);
AUROC of ||z_v - z_t|| for sign conflict (sign(label_T) != sign(label_V)).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from au_soft import NF
from au_soft2 import extra_inputs
from train_bert import BERT, FULL, add_tune_args, loss_fn, make_opt, metrics, save_valid_only, stop_early

OUT = os.path.join(FULL, os.environ.get("DIFF2_RESULTS_DIR", "results_diff2"))


class Diff2(nn.Module):
    def __init__(self, d_frame, d_stat, use_stat):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BERT)
        d = self.enc.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(d_frame, d), nn.LayerNorm(d))
        self.fpos = nn.Parameter(torch.zeros(NF, d)); nn.init.normal_(self.fpos, std=0.02)
        self.use_stat = use_stat
        if use_stat:
            self.sproj = nn.Sequential(nn.Linear(d_stat, d), nn.LayerNorm(d))
        self.Pt, self.Pv = nn.Linear(d, 128), nn.Linear(d, 128)
        self.ht, self.hv = nn.Linear(128, 1), nn.Linear(128, 1)
        self.mproj = nn.Sequential(nn.Linear(3 * 128, d), nn.LayerNorm(d))
        self.hc = nn.Linear(3 * 128, 1)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, ids, att, tt, frames, stat, m_override=None):
        with torch.no_grad():  # text-only pass of the same encoder, stop-grad
            t = self.enc(input_ids=ids, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0].float()
        emb = self.enc.embeddings.word_embeddings(ids)
        au = self.proj(frames) + self.fpos
        zt, zv = self.Pt(t), self.Pv(au.float().mean(1))
        m = torch.cat([zv - zt, (zv - zt).abs(), zv * zt], -1)
        m_in = m if m_override is None else m_override
        toks = [au]
        if self.use_stat:
            toks.append(self.sproj(stat).unsqueeze(1))
        toks.append(self.mproj(m_in).unsqueeze(1))
        soft = torch.cat(toks, 1).to(emb.dtype); n = soft.size(1)
        emb = torch.cat([emb, soft], 1)
        att2 = torch.cat([att, torch.ones(att.size(0), n, dtype=att.dtype, device=att.device)], 1)
        tt2 = torch.cat([tt, torch.ones(tt.size(0), n, dtype=tt.dtype, device=tt.device)], 1)
        h = self.enc(inputs_embeds=emb, attention_mask=att2, token_type_ids=tt2).last_hidden_state
        return self.head(h[:, 0]).squeeze(-1), {"yt": self.ht(zt).squeeze(-1), "yv": self.hv(zv).squeeze(-1),
                                                  "c": self.hc(m).squeeze(-1), "m": m, "gap": (zv - zt).norm(dim=-1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["diff2_nosup", "diff2_uni", "diff2_full"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stat", action="store_true")
    ap.add_argument("--emo", action="store_true", help="add emotion probs + EMFACS to the frame vectors")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    add_tune_args(ap)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    if os.path.exists(os.path.join(OUT, tag + ".json")):
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    lam1 = 0.2 if a.variant in ("diff2_uni", "diff2_full") else 0.0
    lam2 = 0.1 if a.variant == "diff2_full" else 0.0
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    fl = {"stat": a.stat, "emo": a.emo, "diff": False, "vaux": 0.0, "taux": 0.0, "pre": None}
    fr, stat = extra_inputs(df, fl)
    FR = torch.tensor(fr); ST = torch.tensor(stat) if stat is not None else torch.zeros(len(df), 1)
    Y = torch.tensor(df.label.values, dtype=torch.float32)
    YT, YV = torch.tensor(df.label_T.values, dtype=torch.float32), torch.tensor(df.label_V.values, dtype=torch.float32)
    GAP = (YT - YV).abs()
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(list(df.text), truncation=True, max_length=128)

    def batch(idx):
        L = max(len(enc["input_ids"][i]) for i in idx)
        p = lambda k, v: torch.tensor([enc[k][i] + [v] * (L - len(enc[k][i])) for i in idx])
        return p("input_ids", tok.pad_token_id).cuda(), p("attention_mask", 0).cuda(), p("token_type_ids", 0).cuda(), \
            FR[idx].cuda(), ST[idx].cuda()

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    if os.environ.get("PRE_SMOKE"):  # tiny CPU smoke test only
        sp = {k: v[:int(os.environ["PRE_SMOKE"])] for k, v in sp.items()}
    model = Diff2(fr.shape[-1], ST.shape[1], a.stat).cuda()
    opt = make_opt(model, a)
    steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def predict(idx, m_all=None):
        model.eval(); Ys, Ms, Gs = [], [], []
        for i in range(0, len(idx), 64):
            mo = None if m_all is None else m_all[i:i + 64].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux = model(*batch(idx[i:i + 64]), m_override=mo)
            Ys.append(y.float().cpu()); Ms.append(aux["m"].float().cpu()); Gs.append(aux["gap"].float().cpu())
        model.train()
        return torch.cat(Ys).numpy(), torch.cat(Ms), torch.cat(Gs).numpy()

    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux = model(*batch(j))
            loss = loss_fn(a)(y.float(), Y[j].cuda())
            if lam1:
                loss = loss + lam1 * (nn.functional.mse_loss(aux["yt"].float(), YT[j].cuda())
                                      + nn.functional.mse_loss(aux["yv"].float(), YV[j].cuda()))
            if lam2:
                loss = loss + lam2 * nn.functional.mse_loss(aux["c"].float(), GAP[j].cuda())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])[0]); hist.append(vm)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f}", flush=True)
        if stop_early(a, ep, best_ep):
            break
    if a.valid_only:
        save_valid_only(a, os.path.join(OUT, tag + ".json"), hist, best_ep); print("valid", hist[best_ep]); return
    model.load_state_dict(state)
    te_idx = sp["test"]; te = df.iloc[te_idx].reset_index(drop=True)
    pt, M, gap = predict(te_idx)
    res = {"variant": a.variant, "stat": a.stat, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0, 1),
           "valid": hist[best_ep], "test": metrics(te.label, pt)}
    for name, m in (("conflict", te.category == "conflict"), ("align", te.category.isin(["equal", "spne"]))):
        res[f"test_{name}"] = metrics(te.label[m], pt[m.values])
    # diagnostics
    perm = torch.randperm(len(te_idx), generator=torch.Generator().manual_seed(a.seed))
    ps, _, _ = predict(te_idx, m_all=M[perm])
    full0, full1 = metrics(te.label, pt), metrics(te.label, ps)
    cm = (te.category == "conflict").values
    conf0, conf1 = metrics(te.label[cm], pt[cm]), metrics(te.label[cm], ps[cm])
    tgap = (te.label_T - te.label_V).abs().values
    sign_conf = (np.sign(te.label_T) != np.sign(te.label_V)).astype(int).values
    res["diag"] = {"shuffle_delta_full": {k: full1[k] - full0[k] for k in ("acc2", "mae", "corr")},
                   "shuffle_delta_conflict": {k: conf1[k] - conf0[k] for k in ("acc2", "mae", "corr")},
                   "mean_abs_dpred_shuffle": float(np.abs(ps - pt).mean()),
                   "spearman_gap_vs_trueTVgap": float(spearmanr(gap, tgap)[0]),
                   "auroc_gap_sign_conflict": float(roc_auc_score(sign_conf, gap))}
    pd.DataFrame({"key": te.key, "label": te.label, "category": te.category, "pred": pt, "gap": gap}).to_csv(
        os.path.join(OUT, tag + "_test_pred.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "diag")}))


if __name__ == "__main__":
    main()
