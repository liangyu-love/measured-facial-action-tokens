"""DIFF optimisation round (CH-SIMS v2). Base = diff2_pre (pre_soft.py): raw evidence soft tokens + one [DIFF] token from
m = [zv-zt, |zv-zt|, zv*zt, yv-yt]; zv/yv from the frozen raw8emo visual expert (evq.py, same seed); zt = P_t(text-only CLS of
the task BERT, stop-grad), yt = frozen label_V head(zt) trained to read label_T (lambda 0.2).
One flag per variant (flags combine for diff3_best):
  --texp     V1  frozen TEXT expert (texp.py, label_T, same seed): zt = P_t(expert z_t 128-d), yt = expert head (no lambda)
  --unc      V2  + scalars [yv, yt, |yv-yt|, sign agreement, MC-dropout std of yv and yt (8 passes; std_t = 0 without --texp)]
  --frame    V3  + 8 frame-level conflict tokens from [h_f-zt, |h_f-zt|, h_f*zt] (h_f = visual expert frame states)
  --headcat  V4  final head reads [CLS ; m] (residual path) besides the [DIFF] token
  --rw A     V5  main loss weighted by 1 + A*|label_T - label_V| (train labels only)
  --ft       V6  visual expert unfrozen, lr 1e-5 (warm start)
  --moe      V7  y = w*yt + (1-w)*yv + r,  w = sigmoid(lin([CLS;m])), r = head(CLS); w reported on Conflict vs Align
  --conf     V8  conflict head on m -> |label_T - label_V| (lambda 0.1)
Recipe as diff2_pre (6 ep, lr 2e-5, bs 16, valid MAE) + the shared tuning flags. Output results_diff3/<name>_seed<s>.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, get_linear_schedule_with_warmup

from au_soft import NF
from evq import VisOnly
from texp import TextExpert, tokenize
from aexp import AudioExpert, audio_feats
from train_bert import BERT, FULL, add_tune_args, loss_fn, metrics, pol_acc2, save_valid_only, stop_early

SIMS = os.path.expanduser("~/work/msa-pilot/outputs/full")
OUT = os.path.join(FULL, os.environ.get("DIFF3_RESULTS_DIR", "results_diff3"))
FLAGS = ("texp", "unc", "frame", "headcat", "ft", "moe", "conf", "aexp")


class Diff3(nn.Module):
    def __init__(self, vis, texp, d_frame, f, aex=None):
        super().__init__()
        self.f = f
        self.aex = aex
        if aex is not None:  # frozen AUDIO expert (aexp.py, label_A): third opinion
            for p in self.aex.parameters():
                p.requires_grad_(False)
            self.Pa = nn.Linear(128, 128)
        self.enc = AutoModel.from_pretrained(BERT)
        d = self.enc.config.hidden_size
        self.vis, self.texp = vis, texp
        for p in self.vis.parameters():
            p.requires_grad_(bool(f["ft"]))
        if texp is not None:
            for p in self.texp.parameters():
                p.requires_grad_(False)
        self.proj = nn.Sequential(nn.Linear(d_frame, d), nn.LayerNorm(d))
        self.fpos = nn.Parameter(torch.zeros(NF, d)); nn.init.normal_(self.fpos, std=0.02)
        self.Pt = nn.Linear(128 if texp is not None else d, 128)
        self.mdim = 3 * 128 + 1 + (6 if f["unc"] else 0) + (2 * 128 + 10 if aex is not None else 0)
        if f.get("precomp"):  # scalar-only DIFF from precomputed expert readings [yv-yt, |yv-yt|, yv, yt, sign agree, sv, st]
            self.mdim = 7 + (8 if f.get("paudio") else 0)  # + audio: [ya, sa, ya-yt, |ya-yt|, agree(a,t), yv-ya, |yv-ya|, agree(v,a)]
        self.trust = bool(f.get("trust"))
        if self.trust:  # predicts t = |y - y_T| - |y - y_V| (> 0: overall label closer to the face); fed back as one extra DIFF feature
            self.htr = nn.Linear(self.mdim, 1)
        self.mproj = nn.Sequential(nn.Linear(self.mdim + (1 if self.trust else 0), d), nn.LayerNorm(d))
        if f["frame"]:
            self.fproj = nn.Sequential(nn.Linear(3 * 128, d), nn.LayerNorm(d))
            self.fpos2 = nn.Parameter(torch.zeros(NF, d)); nn.init.normal_(self.fpos2, std=0.02)
        hin = d + (self.mdim if (f["headcat"] or f["moe"]) else 0)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d + (self.mdim if f["headcat"] else 0), 1))
        if f["moe"]:
            self.wlin = nn.Linear(hin, 1)
        if f["conf"]:
            self.hc = nn.Linear(self.mdim, 1)

    @torch.no_grad()
    def mc_std(self, fn, n=8):
        return torch.stack([fn() for _ in range(n)]).float().std(0)

    def forward(self, ids, att, tt, x, msk, xa=None, m_override=None, f_override=None):
        f = self.f
        if f.get("precomp"):  # xa carries precomputed (out-of-fold or in-sample) readings [yv, yt, sv, st]
            cols = [t.squeeze(-1) for t in xa.float().split(1, -1)]; yv, yt, sv, st = cols[:4]
            if self.training and getattr(self, "noise_std", None) is not None:
                yv = yv + torch.randn_like(yv) * self.noise_std[0]; yt = yt + torch.randn_like(yt) * self.noise_std[1]
            feats = [yv - yt, (yv - yt).abs(), yv, yt, ((yv * yt) > 0).float(), sv, st]
            if f.get("paudio"):
                ya, sa = cols[4], cols[5]
                feats += [ya, sa, ya - yt, (ya - yt).abs(), ((ya * yt) > 0).float(), yv - ya, (yv - ya).abs(), ((yv * ya) > 0).float()]
            m = torch.stack(feats, -1)
            if self.training and getattr(self, "ddrop", 0.0) > 0:
                m = m * (torch.rand(m.size(0), 1, device=m.device) > self.ddrop).float()
            m_in = m if m_override is None else m_override.float()
            emb = self.enc.embeddings.word_embeddings(ids)
            trp = None
            if self.trust:
                trp = self.htr(m_in).squeeze(-1); m_in = torch.cat([m_in, trp.unsqueeze(-1)], -1)
            dt = self.mproj(m_in).unsqueeze(1)
            soft = (dt if getattr(self, "noau", False) else torch.cat([self.proj(x) + self.fpos, dt], 1)).to(emb.dtype); n = soft.size(1)
            emb = torch.cat([emb, soft], 1)
            att2 = torch.cat([att, torch.ones(att.size(0), n, dtype=att.dtype, device=att.device)], 1)
            tt2 = torch.cat([tt, torch.ones(tt.size(0), n, dtype=tt.dtype, device=tt.device)], 1)
            cls = self.enc(inputs_embeds=emb, attention_mask=att2, token_type_ids=tt2).last_hidden_state[:, 0]
            aux = {"yv": yv, "yt": yt, "m": m, "fr": None, "gap": (yv - yt).abs()}
            if trp is not None:
                aux["tr"] = trp
            if f.get("paudio"):
                aux.update(ya=cols[4], gap_ta=(cols[4] - yt).abs())
            if hasattr(self, "phead"):
                aux["pol"] = self.phead(cls).squeeze(-1)
            return self.head(cls).squeeze(-1), aux
        self.vis.eval()
        with torch.set_grad_enabled(bool(f["ft"]) and self.training):
            h, zv = self.vis.encode(x, msk)
            yv = self.vis.head(zv).squeeze(-1)
        if f["texp"]:
            self.texp.eval()
            with torch.set_grad_enabled(getattr(self, "texp_ft", False) and self.training):
                zt0 = self.texp.encode(ids, att, tt); yt = self.texp.head(zt0).squeeze(-1)
            zt = self.Pt(zt0.to(self.Pt.weight.dtype) if not torch.is_autocast_enabled() else zt0)
        else:
            with torch.no_grad():
                t = self.enc(input_ids=ids, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0]
            zt = self.Pt(t)
            yt = self.vis.head(zt).squeeze(-1)  # frozen head reads label_T from zt (lambda 0.2)
        parts = [zv - zt, (zv - zt).abs(), zv * zt, (yv - yt).unsqueeze(-1)]
        if f["unc"]:
            def vpass():
                self.vis.train(); o = self.vis.head(self.vis.encode(x, msk)[1]).squeeze(-1); self.vis.eval(); return o
            sv = self.mc_std(vpass)
            if f["texp"]:
                def tpass():
                    self.texp.train(); o = self.texp(ids, att, tt); self.texp.eval(); return o
                st = self.mc_std(tpass)
            else:
                st = torch.zeros_like(sv)
            parts += [torch.stack([yv.float(), yt.float(), (yv - yt).abs().float(), ((yv * yt) > 0).float(), sv, st], -1).to(zv.dtype)]
        ya = None
        if self.aex is not None:
            self.aex.eval()
            with torch.no_grad():
                za = self.aex.encode(xa); ya = self.aex.head(za).squeeze(-1)
                def apass():
                    self.aex.train(); o = self.aex(xa); self.aex.eval(); return o
                sa = self.mc_std(apass)
            za_ = self.Pa(za)
            parts += [za_ - zt, (za_ - zt).abs(),
                      torch.stack([ya.float(), (ya - yt).float(), (ya - yt).abs().float(), ((ya * yt) > 0).float(),
                                   (yv - ya).float(), (yv - ya).abs().float(), ((yv * ya) > 0).float(), sa,
                                   (ya - yt).abs().float() * (yv - yt).abs().float(), ((ya * yv) > 0).float() * ((ya * yt) < 0).float()], -1)]
        m = torch.cat([p.to(zv.dtype) for p in parts], -1)
        m_in = m if m_override is None else m_override.to(m.dtype)
        emb = self.enc.embeddings.word_embeddings(ids)
        tr = None
        if self.trust:
            tr = self.htr(m_in.to(self.htr.weight.dtype) if not torch.is_autocast_enabled() else m_in).squeeze(-1)
            dm = torch.cat([m_in, tr.unsqueeze(-1).to(m_in.dtype)], -1)
        else:
            dm = m_in
        toks = [self.proj(x) + self.fpos, self.mproj(dm).unsqueeze(1)]
        fr = None
        if f["frame"]:
            zt_ = zt.unsqueeze(1)
            fr = torch.cat([h - zt_, (h - zt_).abs(), h * zt_], -1)
            fr_in = fr if f_override is None else f_override.to(fr.dtype)
            toks.append(self.fproj(fr_in) + self.fpos2)
        soft = torch.cat(toks, 1).to(emb.dtype); n = soft.size(1)
        emb = torch.cat([emb, soft], 1)
        att2 = torch.cat([att, torch.ones(att.size(0), n, dtype=att.dtype, device=att.device)], 1)
        tt2 = torch.cat([tt, torch.ones(tt.size(0), n, dtype=tt.dtype, device=tt.device)], 1)
        cls = self.enc(inputs_embeds=emb, attention_mask=att2, token_type_ids=tt2).last_hidden_state[:, 0]
        hin = torch.cat([cls, m_in.to(cls.dtype)], -1) if (f["headcat"] or f["moe"]) else cls
        r = self.head(hin if f["headcat"] else cls).squeeze(-1)
        aux = {"yv": yv, "yt": yt, "m": m, "fr": fr, "gap": (yv - yt).abs()}
        if ya is not None:
            aux.update(ya=ya, gap_ta=(ya - yt).abs())
        if tr is not None:
            aux["tr"] = tr
        if hasattr(self, "phead"):
            aux["pol"] = self.phead(cls).squeeze(-1)
        if f["moe"]:
            w = torch.sigmoid(self.wlin(hin)).squeeze(-1)
            y = w * yt.detach().to(w.dtype) + (1 - w) * yv.detach().to(w.dtype) + r
            aux["w"] = w
        else:
            y = r
        if f["conf"]:
            aux["c"] = self.hc(m.to(self.hc.weight.dtype) if not torch.is_autocast_enabled() else m).squeeze(-1)
        return y, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    for fl in FLAGS:
        ap.add_argument(f"--{fl}", action="store_true")
    ap.add_argument("--rw", type=float, default=0.0)
    ap.add_argument("--trust", type=float, default=0.0, help="whom-to-trust head on m, target |y-y_T|-|y-y_V| (train only), weight; prediction fed back into [DIFF]")
    ap.add_argument("--noise", action="store_true", help="precomp: train-time Gaussian noise on yv / yt with std = expert residual std on VALID")
    ap.add_argument("--audio_oof", action="store_true", help="precomp: add the cross-fitted HuBERT audio expert readings (expert_oof/audio_hubert_seed<s>.csv)")
    ap.add_argument("--noau", action="store_true", help="precomp: no AU evidence tokens (text + DIFF token only)")
    ap.add_argument("--vctl", default="", choices=["", "zero", "shuf"], help="capacity control (precomp): visual reading and its std zeroed or shuffled across clips")
    ap.add_argument("--ddrop", type=float, default=0.0, help="precomp: zero the whole DIFF input with this prob. per sample (train only)")
    ap.add_argument("--precomp", default="", choices=["", "oof", "in", "reg", "ext"], help="scalar-only DIFF from expert_oof/seed<s>.csv readings (oof = cross-fitted on train)")
    ap.add_argument("--stage_ft", type=int, default=0, help="unfreeze both experts (lr 5e-6, unimodal aux 0.2) for the last K epochs")
    ap.add_argument("--flags_file", default=None, help="json {flag: value} (diff3_best)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    add_tune_args(ap)
    a = ap.parse_args()
    if a.flags_file:
        for k, v in json.load(open(a.flags_file)).items():
            setattr(a, k, v)
    f = {k: getattr(a, k) for k in FLAGS}
    f["trust"] = a.trust > 0
    f["precomp"] = a.precomp
    f["paudio"] = a.audio_oof
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.name}_seed{a.seed}"
    if os.path.exists(os.path.join(OUT, tag + ".json")):
        print("exists", tag); return
    ck = torch.load(os.path.join(SIMS, os.environ.get("EXPERT_DIR_V", "results_evq"), f"raw8emo_seed{a.seed}.pt"), map_location="cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    from au_soft2 import extra_inputs
    fr, _ = extra_inputs(df, {"stat": False, "emo": True, "diff": False, "vaux": 0.0, "taux": 0.0, "pre": None})
    vis = VisOnly(ck["d_in"], ck["T"], ck.get("layers", 2)); vis.load_state_dict(ck["state"], strict=False)
    texp = None
    if f["texp"]:
        texp = TextExpert(); texp.load_state_dict(torch.load(os.path.join(SIMS, os.environ.get("EXPERT_DIR_T", "results_texp"), f"seed{a.seed}.pt"), map_location="cpu"), strict=False)
    aex = None
    if f["aexp"]:
        ack = torch.load(os.path.join(SIMS, os.environ.get("EXPERT_DIR_A", "results_aexp"), f"seed{a.seed}.pt"), map_location="cpu", weights_only=False)
        aex = AudioExpert(ack["d_in"]); aex.load_state_dict(ack["state"])
        AX = torch.tensor(audio_feats(df, ack["mu"], ack["sd"])[0])
    FR = torch.tensor(fr); MK = torch.ones(fr.shape[:2], dtype=torch.long)
    Y = torch.tensor(df.label.values, dtype=torch.float32)
    YT = torch.tensor(df.label_T.values, dtype=torch.float32); YV = torch.tensor(df.label_V.values, dtype=torch.float32)
    GAP = (YT - YV).abs()
    TRUST = (Y - YT).abs() - (Y - YV).abs()  # > 0: overall label closer to the visual label than to the text label
    tb = tokenize(df)
    if a.precomp:
        rt = pd.read_csv(os.path.join(FULL, "expert_ext.csv") if a.precomp == "ext" else
                         os.path.join(SIMS, "expert_oof", f"seed{a.seed}{'_reg' if a.precomp == 'reg' else ''}.csv")).set_index("key").reindex(df.key)
        AX = torch.tensor(rt[[f"yv_{a.precomp}", f"yt_{a.precomp}", f"sv_{a.precomp}", f"st_{a.precomp}"]].values, dtype=torch.float32)
        if a.audio_oof:
            ra = pd.read_csv(os.path.join(SIMS, "expert_oof", f"audio_hubert_seed{a.seed}.csv")).set_index("key").reindex(df.key)
            AX = torch.cat([AX, torch.tensor(ra[["ya", "sa"]].values, dtype=torch.float32)], 1)
        assert not torch.isnan(AX).any(), "missing expert readings"
        if a.vctl == "zero":
            AX[:, [0, 2]] = 0.0
        elif a.vctl == "shuf":  # one fixed permutation over all clips (seeded): text expert intact, face information destroyed
            AX[:, [0, 2]] = AX[torch.randperm(len(AX), generator=torch.Generator().manual_seed(1000 + a.seed))][:, [0, 2]]
    batch = (lambda idx: (*tb(idx), FR[idx].cuda(), MK[idx].cuda(), AX[idx].cuda())) if (aex is not None or a.precomp) else         (lambda idx: (*tb(idx), FR[idx].cuda(), MK[idx].cuda()))
    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    if os.environ.get("PRE_SMOKE"):
        sp = {k: v[:int(os.environ["PRE_SMOKE"])] for k, v in sp.items()}
    model = Diff3(vis, texp, fr.shape[-1], f, aex).cuda()
    model.ddrop = a.ddrop; model.noau = a.noau
    if a.polw:  # polarity head, added before the optimiser groups are built
        model.phead = nn.Linear(model.enc.config.hidden_size, 1).cuda()
    if a.noise:  # residual std of each expert reading on VALID (labels of valid are used only for this scalar)
        vmask = (df.split == "valid").values
        model.noise_std = (float(np.std(AX[vmask, 0].numpy() - df.label_V.values[vmask])), float(np.std(AX[vmask, 1].numpy() - df.label_T.values[vmask])))
        print("noise std (visual, text):", model.noise_std, flush=True)
    ln = a.lr_new if a.lr_new is not None else a.lr
    groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad], "lr": a.lr},
              {"params": [p for n, p in model.named_parameters() if n.startswith("vis.") and p.requires_grad], "lr": 1e-5},
              {"params": [p for n, p in model.named_parameters() if not n.startswith(("enc.", "vis.", "texp.", "aex.")) and p.requires_grad], "lr": ln}]
    if a.stage_ft:  # expert params registered now (no grad yet), constant lr 5e-6, enabled for the last K epochs
        groups.append({"params": [p for n, p in model.named_parameters() if n.startswith(("vis.", "texp.")) and not p.requires_grad], "lr": 5e-6, "expert": True})
    groups = [g for g in groups if g["params"]]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)
    lam = 0.0 if f["texp"] else 0.2
    L = loss_fn(a)

    @torch.no_grad()
    def predict(idx, m_all=None, f_all=None):
        model.eval(); out = {k: [] for k in ("y", "yv", "yt", "m", "fr", "gap", "w", "ya", "gap_ta", "tr", "pol")}
        for i in range(0, len(idx), 64):
            mo = None if m_all is None else m_all[i:i + 64].cuda()
            fo = None if f_all is None else f_all[i:i + 64].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux = model(*batch(idx[i:i + 64]), m_override=mo, f_override=fo)
            out["y"].append(y.float().cpu())
            for k in ("yv", "yt", "m", "fr", "gap", "w", "ya", "gap_ta", "tr", "pol"):
                if aux.get(k) is not None:
                    out[k].append(aux[k].float().cpu())
        model.train()
        return {k: torch.cat(v) for k, v in out.items() if v}

    best, t0, hist = 1e9, time.time(), []
    staged = False
    for ep in range(a.epochs):
        if a.stage_ft and not staged and ep >= a.epochs - a.stage_ft:
            staged = True
            for g in opt.param_groups:
                if g.get("expert"):
                    for p in g["params"]:
                        p.requires_grad_(True)
            params = [p for g in opt.param_groups for p in g["params"]]
            model.f["ft"] = True; model.texp_ft = True
            print(f"[{tag}] ep{ep}: experts unfrozen (lr 5e-6, unimodal aux 0.2)", flush=True)
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, aux = model(*batch(j))
            if a.rw:
                w = 1 + a.rw * GAP[j].cuda()
                loss = (L(y.float(), Y[j].cuda(), reduction="none") * w).sum() / w.sum()
            else:
                loss = L(y.float(), Y[j].cuda())
            if lam:
                loss = loss + lam * nn.functional.mse_loss(aux["yt"].float(), YT[j].cuda())
            if f["conf"]:
                loss = loss + 0.1 * nn.functional.mse_loss(aux["c"].float(), GAP[j].cuda())
            if a.trust:
                loss = loss + a.trust * nn.functional.mse_loss(aux["tr"].float(), TRUST[j].cuda())
            if a.polw:
                loss = loss + a.polw * nn.functional.binary_cross_entropy_with_logits(aux["pol"].float(), (Y[j] > 0).float().cuda())
            if staged:  # keep the experts anchored to their unimodal labels
                loss = loss + 0.2 * nn.functional.mse_loss(aux["yv"].float(), YV[j].cuda())
                if f["texp"]:
                    loss = loss + 0.2 * nn.functional.mse_loss(aux["yt"].float(), YT[j].cuda())
            loss.backward(); nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            for g in opt.param_groups:
                if g.get("expert"):
                    g["lr"] = 5e-6  # constant, not scheduled
        PV = predict(sp["valid"]); vm = metrics(df.label.values[sp["valid"]], PV["y"].numpy())
        if a.polw:
            vm["acc2_pol"] = pol_acc2(df.label.values[sp["valid"]], PV["pol"].numpy())
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
    if os.environ.get("DIFF3_SAVE"):  # keep the trained model (interpretability analysis)
        torch.save({"state": state, "flags": f, "d_frame": fr.shape[-1]}, os.environ["DIFF3_SAVE"])
    te_idx = sp["test"]; te = df.iloc[te_idx].reset_index(drop=True)
    P = predict(te_idx); pt = P["y"].numpy()
    res = {"name": a.name, "flags": {**f, "rw": a.rw, "stage_ft": a.stage_ft, "trust_w": a.trust, "noise": a.noise, "ddrop": a.ddrop, "vctl": a.vctl, "noau": a.noau}, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0, 1),
           "valid": hist[best_ep], "test": metrics(te.label, pt)}
    cm = (te.category == "conflict").values; am = te.category.isin(["equal", "spne"]).values
    res["test_conflict"] = metrics(te.label[cm], pt[cm]); res["test_align"] = metrics(te.label[am], pt[am])
    g = torch.Generator().manual_seed(a.seed); perm = torch.randperm(len(te_idx), generator=g)
    pm = predict(te_idx, m_all=P["m"][perm], f_all=P["fr"][perm] if "fr" in P else None)["y"].numpy()
    f0, f1 = metrics(te.label, pt), metrics(te.label, pm)
    c0, c1 = metrics(te.label[cm], pt[cm]), metrics(te.label[cm], pm[cm])
    gap = P["gap"].numpy(); tgap = (te.label_T - te.label_V).abs().values
    d = {"shuffle_diff_delta_full": {k: f1[k] - f0[k] for k in ("acc2", "mae", "corr")},
         "shuffle_diff_delta_conflict": {k: c1[k] - c0[k] for k in ("acc2", "mae", "corr")}}
    if te.label_T.notna().all() and te.label_V.notna().all():  # unimodal labels exist (CH-SIMS v2 only)
        d.update({"spearman_gap_vs_trueTVgap": float(spearmanr(gap, tgap)[0]),
                  "auroc_gap_sign_conflict": float(roc_auc_score((np.sign(te.label_T) != np.sign(te.label_V)).astype(int).values, gap)),
                  "yv_corr_labelV": float(pearsonr(P["yv"].numpy(), te.label_V)[0]), "yt_corr_labelT": float(pearsonr(P["yt"].numpy(), te.label_T)[0])})
    if "ya" in P:
        gta = P["gap_ta"].numpy()
        d["ya_corr_labelA"] = float(pearsonr(P["ya"].numpy(), te.label_A)[0])
        d["auroc_gapTA_sign_conflictTA"] = float(roc_auc_score((np.sign(te.label_T) != np.sign(te.label_A)).astype(int).values, gta))
        d["spearman_gapTA_vs_trueTAgap"] = float(spearmanr(gta, (te.label_T - te.label_A).abs())[0])
    if "tr" in P:
        ttrue = ((te.label - te.label_T).abs() - (te.label - te.label_V).abs()).values; trp = P["tr"].numpy()
        d["trust_corr_test"] = float(pearsonr(trp, ttrue)[0]); d["trust_corr_test_conflict"] = float(pearsonr(trp[cm], ttrue[cm])[0])
    if "w" in P:
        w = P["w"].numpy(); d["moe_w_conflict"] = float(w[cm].mean()); d["moe_w_align"] = float(w[am].mean())
    res["diag"] = d
    if a.polw:
        res["test_pol_acc2"] = pol_acc2(te.label, P["pol"].numpy()); res["polw"] = a.polw
    pd.DataFrame({"key": te.key, "label": te.label, "category": te.category, "pred": pt, "gap": gap,
                  **({"plogit": P["pol"].numpy()} if a.polw else {})}).to_csv(os.path.join(OUT, tag + "_test_pred.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "diag")}))


if __name__ == "__main__":
    main()
