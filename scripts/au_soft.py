"""Non-text evidence injected into BERT (bert-base-chinese by default; ENCODER_DIR to swap).
Variants (--variant):
  late         CLS(transcript) ++ MLP(per-clip AU stats) -> regressor (late fusion, no cross-attention)
  soft         8 per-frame AU vectors (+face flag) -> linear -> soft tokens appended after the transcript
  soft_salient same, but only salient AUs kept (>=0.5 and >= train q90, i.e. the information in tmpl_au); rest = 0
  soft_hg      AU + head pose (pitch/yaw/roll) + gaze/blink blendshapes per frame
  soft_hga     soft_hg + one audio soft token (88 eGeMAPS functionals)
  soft_clip    generic visual features: CLIP ViT-B/16 image embeddings of the same 8 frames (no AUs)
  mag          MAG-BERT-style: gated displacement of the clip-level AU vector added to the output of BERT layer 1
All per-feature standardisation fit on TRAIN frames only. Same protocol as train_bert (6 ep, lr 2e-5, bs 16, valid MAE).
Usage: python au_soft.py --variant soft --seed 1
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

from train_bert import BERT, FULL, add_tune_args, loss_fn, make_opt, metrics, save_valid_only, stop_early

NF = 8
HG = ["Pitch", "Roll", "Yaw", "gaze_pitch", "gaze_yaw", "eyeLookDown", "eyeLookUp", "eyeLookIn", "eyeLookOut",
      "eyeBlink", "jawOpen", "mouthSmile"]


def build_inputs(df, variant):
    """-> frame features (N, 8, D), clip stats (N, S), audio (N, A) or None"""
    tr_keys = set(df.key[df.split == "train"])
    pos = {k: i for i, k in enumerate(df.key)}
    au = pd.read_csv(os.path.join(FULL, "au_frames.csv"))
    au = au[au.key.isin(pos) & au.frame.between(1, NF)]
    aucols = sorted(c for c in au.columns if re.fullmatch(r"AU\d+", c))
    face = au.face == 1
    trf = au[face & au.key.isin(tr_keys)]
    A = np.zeros((len(df), NF, len(aucols)), np.float32)
    FF = np.zeros((len(df), NF, 1), np.float32)
    ii = au.key.map(pos).values; ff = au.frame.values - 1
    vals = au[aucols].fillna(0).values.astype(np.float32)
    if variant == "soft_salient":
        q90 = trf[aucols].quantile(0.90).values.astype(np.float32)
        vals = np.where((vals >= 0.5) & (vals >= q90), vals, 0.0)
    m = face.values
    A[ii[m], ff[m]] = vals[m]; FF[ii[m], ff[m], 0] = 1.0
    parts = [A, FF]
    if variant in ("soft_hg", "soft_hga"):
        H = np.zeros((len(df), NF, len(HG)), np.float32)
        mu, sd = trf[HG].mean().values, trf[HG].std().values + 1e-6
        hv = ((au[HG].values - mu) / sd).astype(np.float32)
        hv = np.nan_to_num(hv)
        H[ii[m], ff[m]] = hv[m]
        parts.append(H)
    if variant == "soft_clip":
        C = np.load(os.path.join(FULL, "clip_b16_frames.npy")).astype(np.float32)  # (N, 8, 512), clips.csv order
        parts = [C / (np.linalg.norm(C, axis=-1, keepdims=True) + 1e-6)]
    frames = np.concatenate(parts, -1)
    # clip-level AU stats (late / mag)
    on = np.where((A >= 0.5).any(1), (A >= 0.5).argmax(1) / (NF - 1), 1.0)
    stats = np.concatenate([A.mean(1), A.max(1), on, FF.mean(1)], 1).astype(np.float32)
    tr = (df.split == "train").values
    stats = (stats - stats[tr].mean(0)) / (stats[tr].std(0) + 1e-6)
    audio = None
    if variant == "soft_hga":
        a = pd.read_csv(os.path.join(FULL, "audio_egemaps88.csv")).set_index("key").reindex(df.key)
        av = a.values.astype(np.float32)
        mu, sd = np.nanmean(av[tr], 0), np.nanstd(av[tr], 0) + 1e-6
        audio = np.nan_to_num((av - mu) / sd).clip(-6, 6).astype(np.float32)
    return frames, stats, audio


class Model(nn.Module):
    def __init__(self, variant, d_frame, n_stats, d_audio):
        super().__init__()
        self.variant = variant
        self.enc = AutoModel.from_pretrained(BERT)
        d = self.enc.config.hidden_size
        if variant == "late":
            self.mlp = nn.Sequential(nn.Linear(n_stats, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128))
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d + 128, 1))
            return
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 1))
        if variant == "mag":  # Rahman et al. 2020 style gate on layer-1 output, clip-level nonverbal vector
            self.wg = nn.Linear(d + n_stats, d); self.wv = nn.Linear(n_stats, d)
            self.ln = nn.LayerNorm(d); self.drop = nn.Dropout(0.1); self.beta = 1.0; self._v = None
            self.enc.encoder.layer[1].register_forward_hook(self._mag_hook)
            return
        self.proj = nn.Sequential(nn.Linear(d_frame, d), nn.LayerNorm(d))
        self.fpos = nn.Parameter(torch.zeros(NF, d)); nn.init.normal_(self.fpos, std=0.02)
        if d_audio:
            self.aproj = nn.Sequential(nn.Linear(d_audio, d), nn.LayerNorm(d))

    def _mag_hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = self._v.to(h.dtype).unsqueeze(1).expand(-1, h.size(1), -1)
        g = torch.relu(self.wg(torch.cat([h, v], -1)))
        H = g * self.wv(v)
        alpha = torch.clamp(h.norm(dim=-1, keepdim=True) / (H.norm(dim=-1, keepdim=True) + 1e-6) * self.beta, max=1.0)
        new = self.drop(h + alpha * H) if os.environ.get("MAG_NOLN") == "1" else self.drop(self.ln(h + alpha * H))  # MAG_NOLN: diagnostic, skip the fresh LayerNorm
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    def forward(self, ids, att, tt, frames, stats, audio):
        if self.variant == "late":
            h = self.enc(input_ids=ids, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0]
            return self.head(torch.cat([h, self.mlp(stats)], -1)).squeeze(-1)
        if self.variant == "mag":
            self._v = stats
            h = self.enc(input_ids=ids, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0]
            return self.head(h).squeeze(-1)
        emb = self.enc.embeddings.word_embeddings(ids)
        toks = [self.proj(frames) + self.fpos]
        if audio is not None:
            toks.append(self.aproj(audio).unsqueeze(1))
        soft = torch.cat(toks, 1).to(emb.dtype)
        n = soft.size(1)
        emb = torch.cat([emb, soft], 1)
        att = torch.cat([att, torch.ones(att.size(0), n, dtype=att.dtype, device=att.device)], 1)
        tt = torch.cat([tt, torch.ones(tt.size(0), n, dtype=tt.dtype, device=tt.device)], 1)
        h = self.enc(inputs_embeds=emb, attention_mask=att, token_type_ids=tt).last_hidden_state[:, 0]
        return self.head(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["late", "soft", "soft_salient", "soft_hg", "soft_hga", "soft_clip", "mag"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    add_tune_args(ap)
    a = ap.parse_args()
    out_dir = os.path.join(FULL, os.environ.get("AUSOFT_RESULTS_DIR", "results_ausoft")); os.makedirs(out_dir, exist_ok=True)
    tag = f"{a.variant}_seed{a.seed}"
    if os.path.exists(os.path.join(out_dir, tag + ".json")):
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    frames, stats, audio = build_inputs(df, a.variant)
    FR, ST = torch.tensor(frames), torch.tensor(stats)
    AU_ = torch.tensor(audio) if audio is not None else None
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(list(df.text), truncation=True, max_length=128)
    Y = torch.tensor(df.label.values, dtype=torch.float32)

    def batch(idx):
        L = max(len(enc["input_ids"][i]) for i in idx)
        p = lambda k, v: torch.tensor([enc[k][i] + [v] * (L - len(enc[k][i])) for i in idx])
        return (p("input_ids", tok.pad_token_id).cuda(), p("attention_mask", 0).cuda(), p("token_type_ids", 0).cuda(),
                FR[idx].cuda(), ST[idx].cuda(), AU_[idx].cuda() if AU_ is not None else None)

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    model = Model(a.variant, frames.shape[-1], stats.shape[1], audio.shape[1] if audio is not None else 0).cuda()
    opt = make_opt(model, a)
    steps = (len(sp["train"]) + a.bs - 1) // a.bs * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def predict(idx):
        model.eval(); out = []
        for i in range(0, len(idx), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(*batch(idx[i:i + 64])).float().cpu())
        model.train()
        return torch.cat(out).numpy()

    best, t0, hist = 1e9, time.time(), []
    for ep in range(a.epochs):
        perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
        for i in range(0, len(perm), a.bs):
            j = perm[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(*batch(j))
            loss = loss_fn(a)(y.float(), Y[j].cuda())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])); hist.append(vm)
        if vm["mae"] < best:
            best, best_ep = vm["mae"], ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f}", flush=True)
        if stop_early(a, ep, best_ep):
            break
    if a.valid_only:
        save_valid_only(a, os.path.join(out_dir, tag + ".json"), hist, best_ep); print("valid", hist[best_ep]); return
    model.load_state_dict(state)
    pt = predict(sp["test"]); te = df.iloc[sp["test"]].reset_index(drop=True)
    res = {"variant": a.variant, "seed": a.seed, "best_epoch": best_ep, "sec": round(time.time() - t0, 1),
           "encoder": os.path.basename(BERT), "d_frame": int(frames.shape[-1]), "valid": hist[best_ep],
           "test": metrics(te.label, pt)}
    for name, m in (("conflict", te.category == "conflict"), ("align", te.category.isin(["equal", "spne"]))):
        res[f"test_{name}"] = metrics(te.label[m], pt[m.values])
    pd.DataFrame({"key": te.key, "label": te.label, "category": te.category, "pred": pt}).to_csv(
        os.path.join(out_dir, tag + "_test_pred.csv"), index=False)
    json.dump(res, open(os.path.join(out_dir, tag + ".json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "test_align")}))


if __name__ == "__main__":
    main()
