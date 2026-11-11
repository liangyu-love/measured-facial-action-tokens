"""Accuracy round on top of au_soft (bert-base-chinese, same protocol: 6 ep, lr 2e-5 for everything, bs 16, valid MAE).
One change per variant (--variant):
  soft_stat     + 1 clip-level token: per-AU max / std / fraction of frames active (> train q75), face rate
  soft_vaux01   + aux head on mean(AU soft-token outputs) -> label_V, MSE weight 0.1 (train split)
  soft_vaux03   same, weight 0.3
  soft_vtaux03  weight 0.3 label_V head + weight 0.3 label_T head on [CLS]
  soft_emo      frame vector += 7 py-feat emotion probs + EMFACS products (AU6*AU12, AU1*AU4*AU15, AU4*AU7, AU9*AU10, AU1*AU2*AU5)
  soft_diff     + 1 discrepancy token: W1 * mean(AU token inputs) - W2 * mean(transcript layer-0 embeddings)
  xattn_pre     + 4 tokens from a 2-layer pre-fusion: (1) text word queries cross-attend to the 8 AU tokens,
                (2) 4 learned latent queries pool layer-1 outputs (Perceiver style). Raw 8 AU tokens are KEPT.
  selfattn_pre  control: identical module/params, but layer 1 = self-attention over AU tokens (no text query)
  soft_combo    combination given by env AUSOFT2_COMBO, e.g. "stat,vaux03,emo"
Usage: python au_soft2.py --variant soft_stat --seed 1 [--maps]
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from au_soft import NF, build_inputs
from train_bert import BERT, FULL, add_tune_args, loss_fn, make_opt, metrics, pol_acc2, save_valid_only, stop_early

OUT = os.path.join(FULL, os.environ.get("AUSOFT2_RESULTS_DIR", "results_acc"))
EMO = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
EMFACS = [("AU06", "AU12"), ("AU01", "AU04", "AU15"), ("AU04", "AU07"), ("AU09", "AU10"), ("AU01", "AU02", "AU05")]


def flags(variant):
    if variant == "soft_combo":
        f = set(os.environ.get("AUSOFT2_COMBO", "").split(","))
    else:
        f = {variant.replace("soft_", "")}
    return {"stat": "stat" in f, "emo": "emo" in f, "diff": "diff" in f,
            "vaux": 0.1 if "vaux01" in f else (0.3 if ("vaux03" in f or "vtaux03" in f) else 0.0),
            "taux": 0.3 if "vtaux03" in f else 0.0,
            "pre": "x" if variant == "xattn_pre" else ("self" if variant == "selfattn_pre" else None)}


def video_normalize(fr, vids, face, nau, mode, n_emo=0):
    """Per source video, over all its clips' face frames (all splits, labels never used):
    rel: x - median(x);  z: (x - mean(x)) / std(x). Applied to AU (and emotion) dims; face flag / EMFACS untouched."""
    out = fr.copy()
    cols = list(range(nau)) + (list(range(nau + 1, nau + 1 + n_emo)) if n_emo else [])
    for v in np.unique(vids):
        m = vids == v
        f = face[m] > 0                                   # (n_clips, 8)
        X = fr[m][..., cols]                              # (n_clips, 8, C)
        if not f.any():
            continue
        flat = X[f]
        if mode == "rel":
            X = X - np.median(flat, 0)
        else:
            X = (X - flat.mean(0)) / (flat.std(0) + 1e-3)
        X[~f] = 0.0
        sub = out[m]; sub[..., cols] = X; out[m] = sub
    return out.astype(np.float32)


def extra_inputs(df, fl):
    frames, _, _ = build_inputs(df, "soft")  # (N, 8, nAU + 1)
    aucols = sorted(c for c in pd.read_csv(os.path.join(FULL, "au_frames.csv"), nrows=1).columns if re.fullmatch(r"AU\d+", c))
    nau = len(aucols)
    A, face = frames[..., :nau], frames[..., nau]
    tr = (df.split == "train").values
    parts = [frames]
    if fl["emo"]:
        e = pd.read_csv(os.path.join(FULL, "emo_frames.csv"))
        pos = {k: i for i, k in enumerate(df.key)}
        E = np.zeros((len(df), NF, len(EMO)), np.float32)
        e = e[e.key.isin(pos) & (e.face == 1) & e.frame.between(1, NF)]
        E[e.key.map(pos).values, e.frame.values - 1] = e[EMO].fillna(0).values
        idx = {c: i for i, c in enumerate(aucols)}
        P = np.stack([np.prod([A[..., idx[c]] for c in combo], 0) for combo in EMFACS], -1).astype(np.float32)
        parts += [E, P]
    fr = np.concatenate(parts, -1).astype(np.float32)
    norm = os.environ.get("MSA_NORM", "")  # unsupervised per-source-video normalisation (no labels): rel | z
    if norm:
        fr = video_normalize(fr, df.video_id.values, face, nau, norm, n_emo=len(EMO) if fl["emo"] else 0)
    stat = None
    if fl["stat"]:
        trA = A[tr][face[tr] > 0]
        q75 = np.quantile(trA, 0.75, axis=0)
        cnt = np.maximum(face.sum(1, keepdims=True), 1)
        active = ((A > q75) & (face[..., None] > 0)).sum(1) / cnt
        s = np.concatenate([A.max(1), A.std(1), active, face.mean(1, keepdims=True)], 1).astype(np.float32)
        stat = (s - s[tr].mean(0)) / (s[tr].std(0) + 1e-6)
    return fr, stat


class PreFusion(nn.Module):
    """layer 1: cross-attn (text queries -> AU tokens) or self-attn (AU tokens); layer 2: 4 latent queries pool layer-1 output."""
    def __init__(self, d, mode, n_lat=4, heads=8):
        super().__init__()
        self.mode = mode
        self.att1 = nn.MultiheadAttention(d, heads, batch_first=True); self.ln1 = nn.LayerNorm(d)
        self.ff1 = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)); self.ln1b = nn.LayerNorm(d)
        self.lat = nn.Parameter(torch.randn(n_lat, d) * 0.02)
        self.att2 = nn.MultiheadAttention(d, heads, batch_first=True); self.ln2 = nn.LayerNorm(d)
        self.ff2 = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)); self.ln2b = nn.LayerNorm(d)

    def forward(self, text, text_pad, au):
        if self.mode == "x":
            h, w = self.att1(text, au, au, need_weights=True, average_attn_weights=True)  # (B, Lt, 8)
            h = self.ln1(text + h); pad = text_pad
        else:
            h, w = self.att1(au, au, au, need_weights=True, average_attn_weights=True)
            h = self.ln1(au + h); pad = None
        h = self.ln1b(h + self.ff1(h))
        q = self.lat.unsqueeze(0).expand(h.size(0), -1, -1)
        z, _ = self.att2(q, h, h, key_padding_mask=pad)
        z = self.ln2(q + z)
        return self.ln2b(z + self.ff2(z)), w


class Model(nn.Module):
    def __init__(self, fl, d_frame, d_stat):
        super().__init__()
        self.fl = fl
        self.enc = AutoModel.from_pretrained(BERT)
        d = self.enc.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(d_frame, d), nn.LayerNorm(d))
        self.fpos = nn.Parameter(torch.zeros(NF, d)); nn.init.normal_(self.fpos, std=0.02)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))
        if fl["stat"]:
            self.sproj = nn.Sequential(nn.Linear(d_stat, d), nn.LayerNorm(d))
        if fl["diff"]:
            self.w1, self.w2 = nn.Linear(d, d), nn.Linear(d, d); self.dln = nn.LayerNorm(d)
        if fl["vaux"]:
            self.vhead = nn.Linear(d, 1)
        if fl["taux"]:
            self.thead = nn.Linear(d, 1)
        if fl["pre"]:
            self.pre = PreFusion(d, fl["pre"])

    def forward(self, ids, att, tt, frames, stat):
        emb = self.enc.embeddings.word_embeddings(ids)
        au = self.proj(frames) + self.fpos                                   # (B, 8, d)
        toks, wmap = [au], None
        if self.fl["stat"]:
            toks.append(self.sproj(stat).unsqueeze(1))
        if self.fl["diff"]:
            m = att.unsqueeze(-1).to(emb.dtype)
            tmean = (emb * m).sum(1) / m.sum(1).clamp(min=1)
            toks.append(self.dln(self.w1(au.mean(1)) - self.w2(tmean.float())).unsqueeze(1))
        if self.fl["pre"]:
            z, wmap = self.pre(emb.float(), att == 0, au)
            toks.append(z)
        soft = torch.cat(toks, 1).to(emb.dtype)
        n = soft.size(1); L = emb.size(1)
        emb = torch.cat([emb, soft], 1)
        att2 = torch.cat([att, torch.ones(att.size(0), n, dtype=att.dtype, device=att.device)], 1)
        tt2 = torch.cat([tt, torch.ones(tt.size(0), n, dtype=tt.dtype, device=tt.device)], 1)
        h = self.enc(inputs_embeds=emb, attention_mask=att2, token_type_ids=tt2).last_hidden_state
        y = self.head(h[:, 0]).squeeze(-1)
        aux = {}
        if hasattr(self, "phead"):  # polarity head (--polw)
            aux["pol"] = self.phead(h[:, 0]).squeeze(-1)
        if self.fl["vaux"]:
            aux["v"] = self.vhead(h[:, L:L + NF].mean(1)).squeeze(-1)
        if self.fl["taux"]:
            aux["t"] = self.thead(h[:, 0]).squeeze(-1)
        return y, aux, wmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["soft_stat", "soft_vaux01", "soft_vaux03", "soft_vtaux03", "soft_emo",
                                                         "soft_diff", "xattn_pre", "selfattn_pre", "soft_combo"])
    ap.add_argument("--seed", type=int, required=True)
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
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    fl = flags(a.variant)
    fr, stat = extra_inputs(df, fl)
    FR = torch.tensor(fr); ST = torch.tensor(stat) if stat is not None else torch.zeros(len(df), 1)
    YV, YT = torch.tensor(df.label_V.values, dtype=torch.float32), torch.tensor(df.label_T.values, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(list(df.text), truncation=True, max_length=128)
    Y = torch.tensor(df.label.values, dtype=torch.float32)

    def batch(idx):
        L = max(len(enc["input_ids"][i]) for i in idx)
        p = lambda k, v: torch.tensor([enc[k][i] + [v] * (L - len(enc[k][i])) for i in idx])
        return p("input_ids", tok.pad_token_id).cuda(), p("attention_mask", 0).cuda(), p("token_type_ids", 0).cuda(), \
            FR[idx].cuda(), ST[idx].cuda()

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    if os.environ.get("PRE_SMOKE"):  # tiny CPU smoke test only
        sp = {k: v[:int(os.environ["PRE_SMOKE"])] for k, v in sp.items()}
    model = Model(fl, fr.shape[-1], ST.shape[1]).cuda()
    if a.polw:
        model.phead = nn.Linear(model.enc.config.hidden_size, 1).cuda()
    opt = make_opt(model, a)
    steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)

    PL = []  # polarity logits of the last predict() call

    @torch.no_grad()
    def predict(idx, maps=False):
        model.eval(); Ys, W = [], []; PL.clear()
        for i in range(0, len(idx), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, ax, w = model(*batch(idx[i:i + 64]))
            Ys.append(y.float().cpu())
            if "pol" in ax:
                PL.append(ax["pol"].float().cpu())
            if maps and w is not None:
                W += [x.float().cpu() for x in w]
        model.train()
        return torch.cat(Ys).numpy(), W

    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux, _ = model(*batch(j))
            loss = loss_fn(a)(y.float(), Y[j].cuda())
            if a.polw:
                loss = loss + a.polw * nn.functional.binary_cross_entropy_with_logits(aux["pol"].float(), (Y[j] > 0).float().cuda())
            if "v" in aux:
                loss = loss + fl["vaux"] * nn.functional.mse_loss(aux["v"].float(), YV[j].cuda())
            if "t" in aux:
                loss = loss + fl["taux"] * nn.functional.mse_loss(aux["t"].float(), YT[j].cuda())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])[0])
        if a.polw:
            vm["acc2_pol"] = pol_acc2(df.label.values[sp["valid"]], torch.cat(PL).numpy())
        hist.append(vm)
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
    pt, W = predict(te_idx, maps=(a.variant == "xattn_pre" and a.seed == 1))
    lt = torch.cat(PL).numpy() if PL else None
    res = {"variant": a.variant, "flags": fl, "combo": os.environ.get("AUSOFT2_COMBO") if a.variant == "soft_combo" else None,
           "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0, 1), "valid": hist[best_ep], "test": metrics(te.label, pt)}
    for name, m in (("conflict", te.category == "conflict"), ("align", te.category.isin(["equal", "spne"]))):
        res[f"test_{name}"] = metrics(te.label[m], pt[m.values])
    if a.polw:
        res["test_pol_acc2"] = pol_acc2(te.label, lt); res["polw"] = a.polw
    pd.DataFrame({"key": te.key, "label": te.label, "category": te.category, "pred": pt, **({"plogit": lt} if a.polw else {})}).to_csv(
        os.path.join(OUT, tag + "_test_pred.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=2)
    if W:  # word x frame cross-attention heatmaps for the 10 most text/face-discrepant Conflict clips
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        md = os.path.join(FULL, "xattn_maps"); os.makedirs(md, exist_ok=True)
        conf = np.where(te.category.values == "conflict")[0]
        pick = conf[np.argsort(-np.abs(te.label_V.values[conf] - te.label_T.values[conf]))][:10]
        lines = ["| map | key | tokens (row order) | label | label_T | label_V | pred |", "|---|---|---|---|---|---|---|"]
        for n, i in enumerate(pick, 1):
            ids = enc["input_ids"][te_idx[i]]
            w = W[i][: len(ids)].numpy()
            fig, ax = plt.subplots(figsize=(4.5, max(3, 0.18 * len(ids))))
            im = ax.imshow(w, aspect="auto", cmap="magma")
            ax.set_xticks(range(NF)); ax.set_xticklabels([f"F{f + 1}" for f in range(NF)])
            ax.set_yticks(range(len(ids))); ax.set_yticklabels([str(t) for t in range(len(ids))], fontsize=6)
            r = te.iloc[i]
            ax.set_title(f"{r.key}\nT {r.label_T:+.1f} V {r.label_V:+.1f} y {r.label:+.1f} pred {pt[i]:+.2f}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.05); fig.tight_layout()
            fn = f"xattn{n:02d}_{r.key.replace('/', '__')}.png"; fig.savefig(os.path.join(md, fn), dpi=130); plt.close(fig)
            toks = " ".join(f"{t}:{s}" for t, s in enumerate(tok.convert_ids_to_tokens(ids)))
            lines.append(f"| {fn} | {r.key} | {toks} | {r.label:+.1f} | {r.label_T:+.1f} | {r.label_V:+.1f} | {pt[i]:+.2f} |")
        open(os.path.join(md, "index.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "test_align")}))


if __name__ == "__main__":
    main()
