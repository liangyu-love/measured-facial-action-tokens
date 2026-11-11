"""READ/<SEG>-style evidence attending on top of AU soft tokens (bert-base-chinese, eager attention for rollout).

Variants (--variant):
  fa      one soft token per (frame, AU): intensity * AU-identity embedding + frame-position embedding (8 x nAU tokens)
  evi     au_soft frame tokens (8) + [EVI] token after [CLS]; S = softmax(cos(h_EVI, h_ev)/tau) over last-layer
          evidence tokens; head on [CLS] ++ [EVI] ++ sum(S * h_ev)
  fa_evi  fa tokens + [EVI] head (S is a frame x AU map)
After training: test predictions + (evi variants) deletion faithfulness test (top-k / random-k / bottom-k by S and
by attention rollout, evidence tokens masked out of attention) and, with --maps, frame x AU heatmaps for Conflict clips.
Usage: python evi_bert.py --variant fa_evi --seed 1 [--maps]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from au_soft import NF, build_inputs
from train_bert import BERT, FULL, metrics

TAU = 0.1


class EviModel(nn.Module):
    def __init__(self, variant, d_frame, n_au):
        super().__init__()
        self.variant, self.n_au = variant, n_au
        self.enc = AutoModel.from_pretrained(BERT, attn_implementation="eager")
        d = self.enc.config.hidden_size
        self.fpos = nn.Parameter(torch.randn(NF, d) * 0.02)
        # v2 fix (scale-matched init + LayerNorm + frame shuffling + S position-mean) also for the round-3 variants
        self.v2 = "evi2" in variant or variant in ("fa2", "fa_evi_2stage", "fa_evi_vsup")
        if variant.startswith("fa"):
            sc = float(self.enc.embeddings.word_embeddings.weight.std()) if self.v2 else 0.02
            self.au_id = nn.Parameter(torch.randn(n_au, d) * sc)     # AU-identity embedding
            self.face = nn.Parameter(torch.randn(d) * sc)            # added to frames with a detected face
            if self.v2:
                with torch.no_grad():
                    self.fpos.normal_(0, sc)
                self.tok_ln = nn.LayerNorm(d)
        else:
            self.proj = nn.Sequential(nn.Linear(d_frame, d), nn.LayerNorm(d))
        self.use_evi = "evi" in variant
        self.n_reg = 4 if variant.endswith("_reg") else 0  # register tokens: excluded from S and pooling
        if self.n_reg:
            self.reg = nn.Parameter(torch.randn(self.n_reg, d) * 0.02)
        if self.use_evi:
            self.evi = nn.Parameter(torch.randn(d) * 0.02)
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(3 * d, 1))
        else:
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))
        if variant == "fa_evi_vsup":  # S-pooled evidence must predict the visual unimodal label (non-circular signal)
            self.vhead = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))

    def evidence_tokens(self, frames):
        if self.variant.startswith("fa"):
            inten = frames[..., : self.n_au]                       # (B, 8, nAU)
            facef = frames[..., self.n_au: self.n_au + 1]          # (B, 8, 1)
            tok = inten.unsqueeze(-1) * self.au_id + (self.fpos + 0)[None, :, None, :] \
                + facef.unsqueeze(-1) * self.face
            self._perm = None
            if self.v2:
                tok = self.tok_ln(tok)
                if self.training:  # random frame order per sample: frame identity only via fpos
                    perm = torch.argsort(torch.rand(tok.size(0), NF, device=tok.device), 1)
                    tok = torch.gather(tok, 1, perm[:, :, None, None].expand_as(tok))
                    self._perm = perm
            return tok.flatten(1, 2)                               # (B, 8*nAU, d), frame-major
        return self.proj(frames) + self.fpos

    def forward(self, ids, att, tt, frames, ev_drop=None, attn=False, post_layers=None):
        B = ids.size(0)
        emb = self.enc.embeddings.word_embeddings(ids)
        ev = self.evidence_tokens(frames).to(emb.dtype)
        n_ev = ev.size(1)
        if self.use_evi:  # [CLS] [EVI] (+ registers) text... [SEP]
            extra = self.evi.to(emb.dtype).expand(B, 1, -1)
            if self.n_reg:
                extra = torch.cat([extra, self.reg.to(emb.dtype).expand(B, -1, -1)], 1)
            n_x = extra.size(1)
            emb = torch.cat([emb[:, :1], extra, emb[:, 1:]], 1)
            att = torch.cat([att[:, :1], torch.ones(B, n_x, dtype=att.dtype, device=att.device), att[:, 1:]], 1)
            tt = torch.cat([tt[:, :1], tt[:, :1].expand(B, n_x), tt[:, 1:]], 1)
        ev_start = emb.size(1)
        ev_att = torch.ones(B, n_ev, dtype=att.dtype, device=att.device)
        if ev_drop is not None:  # deletion test: masked evidence tokens are removed from attention
            ev_att = ev_att * (~ev_drop).to(att.dtype)
        emb = torch.cat([emb, ev], 1)
        att = torch.cat([att, ev_att], 1)
        tt = torch.cat([tt, torch.ones(B, n_ev, dtype=tt.dtype, device=tt.device)], 1)
        out = self.enc(inputs_embeds=emb, attention_mask=att, token_type_ids=tt, output_attentions=attn,
                       output_hidden_states=bool(post_layers))
        h = out.last_hidden_state
        S, extras = None, {}
        if post_layers:  # READ-style post-hoc map: cos(h_[CLS], h_evidence) at chosen layers, no training
            for L in post_layers:
                hs = out.hidden_states[L]
                sim = torch.cosine_similarity(hs[:, 0:1].float(), hs[:, ev_start:].float(), dim=-1) / TAU
                sim = (sim - sim.mean(0, keepdim=True)).masked_fill(ev_att == 0, -1e4)
                extras[f"post_L{L}"] = torch.softmax(sim, -1)
        if self.use_evi:
            hev = h[:, ev_start:]
            sim = torch.cosine_similarity(h[:, 1:2].float(), hev.float(), dim=-1) / TAU
            if self.v2:  # remove per-position bias (batch mean of the similarity at each evidence position)
                sim = sim - sim.mean(0, keepdim=True)
            sim = sim.masked_fill(ev_att == 0, -1e4)
            S = torch.softmax(sim, -1)
            if self.v2 and getattr(self, "_perm", None) is not None:  # map back to canonical frame order
                Sv = S.view(B, NF, -1)
                S = torch.zeros_like(Sv).scatter_(1, self._perm[:, :, None].expand_as(Sv), Sv).view(B, -1)
            pooled = (S.unsqueeze(-1) * hev.float()).sum(1)
            y = self.head(torch.cat([h[:, 0].float(), h[:, 1].float(), pooled], -1)).squeeze(-1)
            if hasattr(self, "vhead"):
                extras["vpred"] = self.vhead(pooled).squeeze(-1)
        else:
            y = self.head(h[:, 0].float()).squeeze(-1)
        rollout = None
        if attn:  # attention rollout (Abnar & Zuidema 2020) from [CLS] to evidence tokens
            R = None
            for A in out.attentions:
                a = A.float().mean(1)
                a = a + torch.eye(a.size(-1), device=a.device)
                a = a / a.sum(-1, keepdim=True)
                R = a if R is None else a @ R
            rollout = R[:, 0, ev_start:]
        return y, S, rollout, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fa", "evi", "fa_evi", "fa_evi_reg", "fa_evi2", "fa_evi2_sup"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--maps", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(FULL, "results_evi"); os.makedirs(out_dir, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    if os.path.exists(os.path.join(out_dir, tag + ".json")) and not a.maps:
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    frames, _, _ = build_inputs(df, "soft")                       # (N, 8, nAU + 1): AU intensities + face flag
    import re
    aucols = sorted(c for c in pd.read_csv(os.path.join(FULL, "au_frames.csv"), nrows=1).columns if re.fullmatch(r"AU\d+", c))
    n_au = len(aucols)
    FR = torch.tensor(frames)
    trm = (df.split == "train").values
    a_int, face_on = frames[..., :n_au], frames[..., n_au] > 0
    q90 = np.array([np.quantile(a_int[trm][..., c][face_on[trm]], 0.90) for c in range(n_au)])
    SAL = torch.tensor(((a_int >= 0.5) & (a_int >= q90)).reshape(len(df), -1).astype(np.float32))
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(list(df.text), truncation=True, max_length=128)
    Y = torch.tensor(df.label.values, dtype=torch.float32)

    def batch(idx):
        L = max(len(enc["input_ids"][i]) for i in idx)
        p = lambda k, v: torch.tensor([enc[k][i] + [v] * (L - len(enc[k][i])) for i in idx])
        return p("input_ids", tok.pad_token_id).cuda(), p("attention_mask", 0).cuda(), p("token_type_ids", 0).cuda(), FR[idx].cuda()

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    model = EviModel(a.variant, frames.shape[-1], n_au).cuda()
    ckpt = os.path.join(out_dir, tag + ".pt")
    if a.maps and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cuda")); trained = True
    else:
        trained = False

    @torch.no_grad()
    def predict(idx, drop=None, need=False):
        model.eval(); Ys, Ss, Rs = [], [], []
        for i in range(0, len(idx), 32):
            j = idx[i:i + 32]; b = batch(j)
            dm = None if drop is None else drop[i:i + 32].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, S, R, _ = model(*b, ev_drop=dm, attn=need)
            Ys.append(y.float().cpu())
            if need:
                Ss.append(S.float().cpu() if S is not None else torch.zeros(len(j), 1)); Rs.append(R.float().cpu())
        model.train()
        return (torch.cat(Ys).numpy(),) + ((torch.cat(Ss), torch.cat(Rs)) if need else ())

    t0 = time.time()
    if not trained:
        if model.v2:  # new (non-BERT) params at 1e-3, BERT at the usual lr
            newp = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
            opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                                     {"params": newp, "lr": 1e-3}], weight_decay=0.01)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        gen = torch.Generator().manual_seed(a.seed)
        best, hist = 1e9, []
        for ep in range(a.epochs):
            perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
            for i in range(0, len(perm), a.bs):
                j = perm[i:i + a.bs]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y, S_tr, _, _ = model(*batch(j))
                loss = nn.functional.mse_loss(y.float(), Y[j].cuda())
                if a.variant == "fa_evi2_sup":  # weak supervision: S ~ salient (frame, AU) cells (partly circular!)
                    T = SAL[j].cuda(); has = T.sum(1) > 0
                    if has.any():
                        Tn = T[has] / T[has].sum(1, keepdim=True)
                        Sh = S_tr[has].float().clamp_min(1e-9)
                        loss = loss + 0.1 * (Tn * (Tn.clamp_min(1e-9).log() - Sh.log())).sum(1).mean()
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])[0]); hist.append(vm)
            if vm["mae"] < best:
                best, best_ep = vm["mae"], ep
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f}", flush=True)
        model.load_state_dict(state)
        if a.seed == 1:
            torch.save(state, ckpt)  # kept for heatmaps
    te_idx = sp["test"]; te = df.iloc[te_idx].reset_index(drop=True)
    need = model.use_evi
    out = predict(te_idx, need=True) if need else predict(te_idx)
    pt = out[0]
    if not trained:
        res = {"variant": a.variant, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0, 1),
               "n_evidence_tokens": int(NF * n_au if a.variant.startswith("fa") else NF),
               "valid": hist[best_ep], "test": metrics(te.label, pt)}
        for name, m in (("conflict", te.category == "conflict"), ("align", te.category.isin(["equal", "spne"]))):
            res[f"test_{name}"] = metrics(te.label[m], pt[m.values])
        pd.DataFrame({"key": te.key, "label": te.label, "category": te.category, "pred": pt}).to_csv(
            os.path.join(out_dir, tag + "_test_pred.csv"), index=False)
    if need:
        S, R = out[1], out[2]
        n_ev = S.shape[1]; k = max(1, round(0.1 * n_ev))  # delete 10% of evidence tokens
        g = torch.Generator().manual_seed(a.seed)
        rnd = torch.rand(S.shape, generator=g)
        faith = {}
        for name, score in (("S_top", S), ("S_bottom", -S), ("rollout_top", R), ("random", rnd)):
            top = score.topk(k, dim=1).indices
            drop = torch.zeros_like(S, dtype=torch.bool).scatter_(1, top, True)
            pd_ = predict(te_idx, drop=drop)[0]
            faith[name] = {"mean_abs_dpred": float(np.abs(pd_ - pt).mean()),
                           "dMAE": float(np.abs(pd_ - te.label.values).mean() - np.abs(pt - te.label.values).mean()),
                           "dMAE_conflict": float(np.abs(pd_ - te.label.values)[te.category == "conflict"].mean()
                                                  - np.abs(pt - te.label.values)[te.category == "conflict"].mean())}
        faith["k"] = k; faith["n_evidence_tokens"] = n_ev
        faith["S_top10_mass"] = float(S.topk(k, dim=1).values.sum(1).mean())
        faith["gap_top_vs_random"] = faith["S_top"]["mean_abs_dpred"] - faith["random"]["mean_abs_dpred"]
        faith["S_entropy_mean"] = float(-(S.clamp_min(1e-9) * S.clamp_min(1e-9).log()).sum(1).mean())
        faith["S_vs_rollout_spearman"] = float(np.nanmean([pd.Series(S[i].numpy()).corr(pd.Series(R[i].numpy()), method="spearman")
                                                           for i in range(min(300, len(S)))]))
        if not trained:
            res["faithfulness"] = faith
        print(json.dumps(faith), flush=True)
        if a.maps and a.variant in ("fa_evi", "fa_evi_reg", "fa_evi2", "fa_evi2_sup"):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            md = os.path.join(FULL, "evi_maps", a.variant); os.makedirs(md, exist_ok=True)
            conf = np.where(te.category.values == "conflict")[0]
            pick = conf[np.argsort(-np.abs(te.label_V.values[conf] - te.label_T.values[conf]))][:10]
            idx_lines = ["| map | key | transcript | label | label_T | label_V | pred |", "|---|---|---|---|---|---|---|"]
            for n, i in enumerate(pick, 1):
                m = S[i].numpy().reshape(NF, n_au)
                fig, ax = plt.subplots(figsize=(9, 3.2))
                im = ax.imshow(m, aspect="auto", cmap="viridis")
                ax.set_yticks(range(NF)); ax.set_yticklabels([f"F{f + 1}" for f in range(NF)])
                ax.set_xticks(range(n_au)); ax.set_xticklabels(aucols, rotation=90, fontsize=7)
                r = te.iloc[i]
                ax.set_title(f"{r.key} | label {r.label:+.1f} (T {r.label_T:+.1f}, V {r.label_V:+.1f}) | pred {pt[i]:+.2f}", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.025); fig.tight_layout()
                fn = f"map{n:02d}_{r.key.replace('/', '__')}.png"; fig.savefig(os.path.join(md, fn), dpi=130); plt.close(fig)
                idx_lines.append(f"| {fn} | {r.key} | {r.text} | {r.label:+.1f} | {r.label_T:+.1f} | {r.label_V:+.1f} | {pt[i]:+.2f} |")
            open(os.path.join(md, "index.md"), "w").write("\n".join(idx_lines) + "\n")
            print("maps ->", md)
    if not trained:
        json.dump(res, open(os.path.join(out_dir, tag + ".json"), "w"), indent=2)
        print(json.dumps({k: res[k] for k in ("test", "test_conflict", "test_align")}))


if __name__ == "__main__":
    main()
