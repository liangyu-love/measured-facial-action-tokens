"""Pretrained visual evidence encoder plugged into BERT.
The encoder is evq.py's visual-only model trained on CH-SIMS v2 TRAIN label_V (same seed as this run), then FROZEN.
Evidence variant = best non-dense evq variant by mean valid MAE (raw8 / raw8emo / rel / z; dense is not available off-SIMS).
  au_soft_pre: the encoder's frame-level hidden states (8 x 128) -> linear -> 768 soft tokens (replace raw AU tokens).
  diff2_pre:   raw evidence soft tokens (as au_soft+emo) + one [DIFF] token from m = [zv-zt, |zv-zt|, zv*zt, yv-yt],
               zv = frozen pooled encoder state, yv = frozen label_V head(zv); zt = P_t(text-only CLS, stop-grad),
               yt = the same FROZEN head(zt), trained to read label_T (lambda 0.2) so zt lives in the visual-sentiment space.
               Datasets without unimodal labels (MOSI / MOSEI): lambda = 0 (no proxy supervision).
Same training recipe as diff2 / au_soft (6 ep, lr 2e-5, bs 16, best epoch by valid MAE). Works for MSA_DATASET=sims2|mosi|mosei.
Diagnostics (test): shuffle evidence across samples (both); shuffle [DIFF] (diff2_pre); with unimodal labels also
Spearman(|yv-yt|, |label_T-label_V|), AUROC for sign conflict, encoder yv Corr with label_V; always yv Corr with label.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from au_soft import NF
from evq import VisOnly
from train_bert import BERT, FULL, add_tune_args, loss_fn, make_opt, metrics, save_valid_only, stop_early

SIMS = os.path.expanduser("~/work/msa-pilot/outputs/full")
OUT = os.path.join(FULL, os.environ.get("PRE_RESULTS_DIR", "results_pre"))


def pick_ev():
    vm = {ev: np.mean([json.load(open(p))["valid_mae"] for p in glob.glob(os.path.join(SIMS, os.environ.get("EXPERT_DIR_V", "results_evq"), f"{ev}_seed*.json"))] or [9])
          for ev in ("raw8", "raw8emo", "rel", "z")}
    return min(vm, key=vm.get)


class Pre(nn.Module):
    def __init__(self, vis, d_frame, variant):
        super().__init__()
        self.variant = variant
        self.enc = AutoModel.from_pretrained(BERT)
        d = self.enc.config.hidden_size
        self.vis = vis
        for p in self.vis.parameters():
            p.requires_grad_(False)
        self.proj = nn.Sequential(nn.Linear(128 if variant == "au_soft_pre" else d_frame, d), nn.LayerNorm(d))
        self.fpos = nn.Parameter(torch.zeros(NF, d)); nn.init.normal_(self.fpos, std=0.02)
        if variant == "diff2_pre":
            self.Pt = nn.Linear(d, 128)
            self.mproj = nn.Sequential(nn.Linear(3 * 128 + 1, d), nn.LayerNorm(d))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, ids, att, tt, x, msk, m_override=None):
        self.vis.eval()
        with torch.no_grad():
            h, zv = self.vis.encode(x, msk)
            yv = self.vis.head(zv).squeeze(-1)
        emb = self.enc.embeddings.word_embeddings(ids)
        aux = {"yv": yv}
        if self.variant == "au_soft_pre":
            toks = [self.proj(h) + self.fpos]
        else:
            with torch.no_grad():
                t = self.enc(input_ids=ids, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0]
            zt = self.Pt(t)
            yt = self.vis.head(zt).squeeze(-1)  # frozen head, gradient flows into Pt
            m = torch.cat([zv - zt, (zv - zt).abs(), zv * zt, (yv - yt).unsqueeze(-1)], -1)
            toks = [self.proj(x) + self.fpos, self.mproj(m if m_override is None else m_override).unsqueeze(1)]
            aux.update(yt=yt, m=m, gap=(yv - yt).abs())
        soft = torch.cat(toks, 1).to(emb.dtype); n = soft.size(1)
        emb = torch.cat([emb, soft], 1)
        att2 = torch.cat([att, torch.ones(att.size(0), n, dtype=att.dtype, device=att.device)], 1)
        tt2 = torch.cat([tt, torch.ones(tt.size(0), n, dtype=tt.dtype, device=tt.device)], 1)
        hb = self.enc(inputs_embeds=emb, attention_mask=att2, token_type_ids=tt2).last_hidden_state
        return self.head(hb[:, 0]).squeeze(-1), aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["au_soft_pre", "diff2_pre"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ev", default=None, help="evidence variant of the SIMS encoder (default: best by valid MAE)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    add_tune_args(ap)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    if os.path.exists(os.path.join(OUT, tag + ".json")):
        print("exists", tag); return
    ev = a.ev or pick_ev()
    ck = torch.load(os.path.join(SIMS, os.environ.get("EXPERT_DIR_V", "results_evq"), f"{ev}_seed{a.seed}.pt"), map_location="cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    from au_soft2 import extra_inputs
    if ev in ("rel", "z"):
        os.environ["MSA_NORM"] = ev
    fr, _ = extra_inputs(df, {"stat": False, "emo": ev != "raw8", "diff": False, "vaux": 0.0, "taux": 0.0, "pre": None})
    os.environ.pop("MSA_NORM", None)
    assert fr.shape[1:] == (ck["T"], ck["d_in"]), (fr.shape, ck["T"], ck["d_in"])
    vis = VisOnly(ck["d_in"], ck["T"], ck.get("layers", 2)); vis.load_state_dict(ck["state"], strict=False)
    FR = torch.tensor(fr); MK = torch.ones(fr.shape[:2], dtype=torch.long)
    Y = torch.tensor(df.label.values, dtype=torch.float32)
    has_uni = df.label_T.notna().all() and df.label_V.notna().all()
    YT = torch.tensor(df.label_T.values if has_uni else df.label.values, dtype=torch.float32)
    lam = 0.2 if (a.variant == "diff2_pre" and has_uni) else 0.0
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(list(df.text), truncation=True, max_length=128)
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = [[0] * len(x) for x in enc["input_ids"]]

    def batch(idx):
        L = max(len(enc["input_ids"][i]) for i in idx)
        p = lambda k, v: torch.tensor([enc[k][i] + [v] * (L - len(enc[k][i])) for i in idx])
        return p("input_ids", tok.pad_token_id).cuda(), p("attention_mask", 0).cuda(), p("token_type_ids", 0).cuda(), \
            FR[idx].cuda(), MK[idx].cuda()

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    if os.environ.get("PRE_SMOKE"):  # tiny CPU smoke test only
        sp = {k: v[:int(os.environ["PRE_SMOKE"])] for k, v in sp.items()}
    model = Pre(vis, fr.shape[-1], a.variant).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_opt(model, a)
    steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def predict(idx, x_perm=None, m_all=None):
        model.eval(); out = {"y": [], "yv": [], "m": [], "gap": []}
        for i in range(0, len(idx), 64):
            b = list(batch(idx[i:i + 64]))
            if x_perm is not None:
                b[3] = FR[x_perm[i:i + 64]].cuda()
            mo = None if m_all is None else m_all[i:i + 64].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux = model(*b, m_override=mo)
            out["y"].append(y.float().cpu()); out["yv"].append(aux["yv"].float().cpu())
            if "m" in aux:
                out["m"].append(aux["m"].float().cpu()); out["gap"].append(aux["gap"].float().cpu())
        model.train()
        return {k: torch.cat(v) for k, v in out.items() if v}

    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux = model(*batch(j))
            loss = loss_fn(a)(y.float(), Y[j].cuda())
            if lam:
                loss = loss + lam * nn.functional.mse_loss(aux["yt"].float(), YT[j].cuda())
            loss.backward(); nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])["y"].numpy()); hist.append(vm)
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
    P = predict(te_idx); pt = P["y"].numpy()
    res = {"variant": a.variant, "ev": ev, "seed": a.seed, "lambda_T": lam, "best_epoch": best_ep,
           "sec": round(time.time() - t0, 1), "valid": hist[best_ep], "test": metrics(te.label, pt)}
    cm = (te.category == "conflict").values
    for name, m in (("conflict", cm), ("align", te.category.isin(["equal", "spne"]).values)):
        res[f"test_{name}"] = metrics(te.label[m], pt[m])
    # diagnostics
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(te_idx), generator=g).numpy()
    d = {}
    ps = predict(te_idx, x_perm=te_idx[perm])["y"].numpy()
    f0, f1 = metrics(te.label, pt), metrics(te.label, ps)
    d["shuffle_evidence_delta_full"] = {k: f1[k] - f0[k] for k in ("acc2", "mae", "corr")}
    if cm.any():
        c0, c1 = metrics(te.label[cm], pt[cm]), metrics(te.label[cm], ps[cm])
        d["shuffle_evidence_delta_conflict"] = {k: c1[k] - c0[k] for k in ("acc2", "mae", "corr")}
    yv = P["yv"].numpy()
    d["encoder_yv_corr_label"] = float(pearsonr(yv, te.label)[0])
    if has_uni:
        d["encoder_yv_corr_labelV"] = float(pearsonr(yv, te.label_V)[0])
    if a.variant == "diff2_pre":
        pm = predict(te_idx, m_all=P["m"][perm])["y"].numpy()
        f1 = metrics(te.label, pm)
        d["shuffle_diff_delta_full"] = {k: f1[k] - f0[k] for k in ("acc2", "mae", "corr")}
        if cm.any():
            c0, c1 = metrics(te.label[cm], pt[cm]), metrics(te.label[cm], pm[cm])
            d["shuffle_diff_delta_conflict"] = {k: c1[k] - c0[k] for k in ("acc2", "mae", "corr")}
        d["mean_abs_dpred_shuffle_diff"] = float(np.abs(pm - pt).mean())
        if has_uni:
            gap = P["gap"].numpy()
            d["spearman_gap_vs_trueTVgap"] = float(spearmanr(gap, (te.label_T - te.label_V).abs())[0])
            d["auroc_gap_sign_conflict"] = float(roc_auc_score((np.sign(te.label_T) != np.sign(te.label_V)).astype(int), gap))
    res["diag"] = d
    pd.DataFrame({"key": te.key, "label": te.label, "category": te.category, "pred": pt, "yv": yv}).to_csv(
        os.path.join(OUT, tag + "_test_pred.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "diag")}))


if __name__ == "__main__":
    main()
