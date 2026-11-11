"""EVI round 3 (on top of the v2 fix), bert-base-chinese, 5 seeds:
  fa2            fixed per-(frame, AU) soft tokens, no [EVI]  (checkpoint saved for every seed; base for 1 and 2)
  --posthoc      READ-style post-hoc map on trained fa2: S = softmax(cos(h_[CLS], h_ev)/tau) at layers 12 and 8
  fa_evi_2stage  init from fa2 (same seed), freeze embeddings + soft-token params + BERT layers 1-8,
                 train [EVI] + head (+ layers 9-12) for 2 epochs
  fa_evi_vsup    fa_evi2 + aux MSE: head(S-pooled evidence) -> label_V (train split), weight 0.3 (non-circular)
Each: metrics; deletion faithfulness (S top/bottom/random vs attention rollout, 10% of evidence tokens);
S entropy and top-10% mass; heatmaps for the same 10 Conflict clips (seed 1).
Usage: python evi3.py --variant fa2 --seed 1 | python evi3.py --variant fa2 --seed 1 --posthoc
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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from au_soft import NF, build_inputs
from evi_bert import EviModel
from train_bert import BERT, FULL, metrics

OUT = os.path.join(FULL, "results_evi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fa2", "fa_evi_2stage", "fa_evi_vsup"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--posthoc", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.variant}{'_posthoc' if a.posthoc else ''}_seed{a.seed}"
    if os.path.exists(os.path.join(OUT, tag + ".json")):
        print("exists", tag); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    frames, _, _ = build_inputs(df, "soft")
    aucols = sorted(c for c in pd.read_csv(os.path.join(FULL, "au_frames.csv"), nrows=1).columns if re.fullmatch(r"AU\d+", c))
    n_au = len(aucols)
    FR = torch.tensor(frames)
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(list(df.text), truncation=True, max_length=128)
    Y = torch.tensor(df.label.values, dtype=torch.float32)
    YV = torch.tensor(df.label_V.values, dtype=torch.float32)

    def batch(idx):
        L = max(len(enc["input_ids"][i]) for i in idx)
        p = lambda k, v: torch.tensor([enc[k][i] + [v] * (L - len(enc[k][i])) for i in idx])
        return p("input_ids", tok.pad_token_id).cuda(), p("attention_mask", 0).cuda(), p("token_type_ids", 0).cuda(), FR[idx].cuda()

    sp = {s: df.index[df.split == s].values for s in ("train", "valid", "test")}
    model = EviModel(a.variant, frames.shape[-1], n_au).cuda()
    fa2_ckpt = os.path.join(OUT, f"fa2_seed{a.seed}.pt")
    post_layers = (12, 8) if a.posthoc else None

    @torch.no_grad()
    def predict(idx, drop=None, need=False):
        model.eval(); Ys, Ss, Rs, Ps = [], [], [], {}
        for i in range(0, len(idx), 32):
            j = idx[i:i + 32]; b = batch(j)
            dm = None if drop is None else drop[i:i + 32].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, S, R, ex = model(*b, ev_drop=dm, attn=need, post_layers=post_layers if need else None)
            Ys.append(y.float().cpu())
            if need:
                if S is not None:
                    Ss.append(S.float().cpu())
                Rs.append(R.float().cpu())
                for k_, v in ex.items():
                    if k_.startswith("post_"):
                        Ps.setdefault(k_, []).append(v.float().cpu())
        model.train()
        res = {"y": torch.cat(Ys).numpy()}
        if need:
            res["R"] = torch.cat(Rs)
            if Ss:
                res["S"] = torch.cat(Ss)
            res.update({k_: torch.cat(v) for k_, v in Ps.items()})
        return res

    t0 = time.time()
    if a.posthoc:  # no training: load the trained fa2 checkpoint
        model.load_state_dict(torch.load(fa2_ckpt, map_location="cuda"))
        best_ep, hist = None, None
    else:
        if a.variant == "fa_evi_2stage":
            sd = torch.load(fa2_ckpt, map_location="cpu")
            sd = {k: v for k, v in sd.items() if not k.startswith("head.")}  # head shape differs (CLS vs CLS+EVI+pooled)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print("2stage init: missing", missing, "unexpected", unexpected, flush=True)
            frozen = ["enc.embeddings.", "au_id", "face", "fpos", "tok_ln."] + [f"enc.encoder.layer.{i}." for i in range(8)]
            for n, p in model.named_parameters():
                if any(n.startswith(f) for f in frozen):
                    p.requires_grad_(False)
            epochs = 2
        else:
            epochs = 6
        epochs = a.epochs or epochs
        bert_p = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
        new_p = [p for n, p in model.named_parameters() if not n.startswith("enc.") and p.requires_grad]
        opt = torch.optim.AdamW([{"params": bert_p, "lr": a.lr}, {"params": new_p, "lr": 1e-3}], weight_decay=0.01)
        steps = (len(sp["train"]) + a.bs - 1) // a.bs * epochs
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        gen = torch.Generator().manual_seed(a.seed)
        best, hist = 1e9, []
        for ep in range(epochs):
            perm = sp["train"][torch.randperm(len(sp["train"]), generator=gen).numpy()]
            for i in range(0, len(perm), a.bs):
                j = perm[i:i + a.bs]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y, _, _, ex = model(*batch(j))
                loss = nn.functional.mse_loss(y.float(), Y[j].cuda())
                if "vpred" in ex:
                    loss = loss + 0.3 * nn.functional.mse_loss(ex["vpred"].float(), YV[j].cuda())
                loss.backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            vm = metrics(df.label.values[sp["valid"]], predict(sp["valid"])["y"]); hist.append(vm)
            if vm["mae"] < best:
                best, best_ep = vm["mae"], ep
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{tag}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f}", flush=True)
        model.load_state_dict(state)
        if a.variant == "fa2" or a.seed == 1:
            torch.save(state, os.path.join(OUT, f"{a.variant}_seed{a.seed}.pt"))

    te_idx = sp["test"]; te = df.iloc[te_idx].reset_index(drop=True)
    need = model.use_evi or a.posthoc
    out = predict(te_idx, need=need)
    pt = out["y"]
    res = {"variant": a.variant + ("_posthoc" if a.posthoc else ""), "seed": a.seed, "best_epoch": best_ep,
           "sec": round(time.time() - t0, 1), "test": metrics(te.label, pt)}
    if hist:
        res["valid"] = hist[best_ep]
    for name, m in (("conflict", te.category == "conflict"), ("align", te.category.isin(["equal", "spne"]))):
        res[f"test_{name}"] = metrics(te.label[m], pt[m.values])
    if not a.posthoc:
        pd.DataFrame({"key": te.key, "label": te.label, "category": te.category, "pred": pt}).to_csv(
            os.path.join(OUT, tag + "_test_pred.csv"), index=False)

    def faith(S, R):
        n_ev = S.shape[1]; k = max(1, round(0.1 * n_ev))
        rnd = torch.rand(S.shape, generator=torch.Generator().manual_seed(a.seed))
        f = {"k": k, "n_evidence_tokens": n_ev,
             "S_entropy_mean": float(-(S.clamp_min(1e-9) * S.clamp_min(1e-9).log()).sum(1).mean()),
             "S_top10_mass": float(S.topk(k, dim=1).values.sum(1).mean()),
             "S_vs_rollout_spearman": float(np.nanmean([pd.Series(S[i].numpy()).corr(pd.Series(R[i].numpy()), method="spearman")
                                                        for i in range(min(300, len(S)))]))}
        for name, score in (("S_top", S), ("S_bottom", -S), ("rollout_top", R), ("random", rnd)):
            drop = torch.zeros_like(S, dtype=torch.bool).scatter_(1, score.topk(k, dim=1).indices, True)
            pd_ = predict(te_idx, drop=drop)["y"]
            f[name] = {"mean_abs_dpred": float(np.abs(pd_ - pt).mean()),
                       "dMAE": float(np.abs(pd_ - te.label.values).mean() - np.abs(pt - te.label.values).mean())}
        f["gap_top_vs_random"] = f["S_top"]["mean_abs_dpred"] - f["random"]["mean_abs_dpred"]
        return f

    def maps(S, name):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        md = os.path.join(FULL, "evi_maps", name); os.makedirs(md, exist_ok=True)
        conf = np.where(te.category.values == "conflict")[0]
        pick = conf[np.argsort(-np.abs(te.label_V.values[conf] - te.label_T.values[conf]))][:10]
        lines = ["| map | key | transcript | label | label_T | label_V | pred |", "|---|---|---|---|---|---|---|"]
        for n, i in enumerate(pick, 1):
            fig, ax = plt.subplots(figsize=(9, 3.2))
            im = ax.imshow(S[i].numpy().reshape(NF, n_au), aspect="auto", cmap="viridis")
            ax.set_yticks(range(NF)); ax.set_yticklabels([f"F{f + 1}" for f in range(NF)])
            ax.set_xticks(range(n_au)); ax.set_xticklabels(aucols, rotation=90, fontsize=7)
            r = te.iloc[i]
            ax.set_title(f"{r.key} | label {r.label:+.1f} (T {r.label_T:+.1f}, V {r.label_V:+.1f}) | pred {pt[i]:+.2f}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.025); fig.tight_layout()
            fn = f"map{n:02d}_{r.key.replace('/', '__')}.png"; fig.savefig(os.path.join(md, fn), dpi=130); plt.close(fig)
            lines.append(f"| {fn} | {r.key} | {r.text} | {r.label:+.1f} | {r.label_T:+.1f} | {r.label_V:+.1f} | {pt[i]:+.2f} |")
        open(os.path.join(md, "index.md"), "w").write("\n".join(lines) + "\n")

    if a.posthoc:
        res["faithfulness"] = {}
        for L in (12, 8):
            res["faithfulness"][f"L{L}"] = faith(out[f"post_L{L}"], out["R"])
            if a.seed == 1:
                maps(out[f"post_L{L}"], f"fa2_posthoc_L{L}")
    elif model.use_evi:
        res["faithfulness"] = faith(out["S"], out["R"])
        if a.seed == 1:
            maps(out["S"], a.variant)
    json.dump(res, open(os.path.join(OUT, tag + ".json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "test_align")}), flush=True)


if __name__ == "__main__":
    main()
