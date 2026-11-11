"""bert-base-chinese regression on CH-SIMS v2(s): input = transcript [SEP] auxiliary text.

Conditions (--cond):
  text      : transcript only
  tmpl      : + measurement template (py-feat visual text + openSMILE prosody text)
  vlm       : + raw Qwen2.5-VL v2 description
  vlmv      : + verified VLM description (unsupported lines dropped at thr 0.5)
  tmpl_vlmv : + template + verified description
  shuf      : + raw VLM description of ANOTHER clip in the same split (seeded permutation) -- control
Official split = meta.csv `mode`. Best epoch picked on valid MAE. Metrics on full test and on test
Conflict / Align (Equal+SPNE) subsets defined by build_splits.categorize on the full label set.

Usage: python train_bert.py --cond tmpl --seed 1
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = os.path.expanduser("~/work/msa-pilot")
FULL = os.path.join(ROOT, os.environ.get("MSA_DATA", "outputs/full"))  # dataset work dir (outputs/mosi for CMU-MOSI)
DATASET = os.environ.get("MSA_DATASET", "sims2")
BERT = os.environ.get("ENCODER_DIR", os.path.join(ROOT, "models/bert-base-chinese"))


def jl(path, field):
    if not os.path.exists(path):
        return {}
    return {d["key"]: d[field] for d in (json.loads(l) for l in open(path) if l.strip())}


def flat(s):
    return " ; ".join(x.strip(" -*•\t") for x in str(s).splitlines() if x.strip())


def build_aux(df, cond, seed):
    vis = jl(os.path.join(FULL, "measure_text.jsonl"), "visual_text")
    pros = jl(os.path.join(FULL, "prosody_text.jsonl"), "prosody_text")
    vlm = jl(os.path.join(FULL, "desc.jsonl"), "desc")
    vlmv = jl(os.path.join(FULL, "desc_verified.jsonl"), "desc")
    tmpl = {k: f"{vis.get(k, '')} {pros.get(k, '')}".strip() for k in df.key}
    if cond == "text":
        return [""] * len(df)
    if cond == "tmpl":
        return [tmpl[k] for k in df.key]
    if cond == "tmpl_vis":  # ablation: py-feat visual template only
        return [vis.get(k, "") for k in df.key]
    if cond in ("tmpl_au", "tmpl_head", "tmpl_gaze"):  # ablation: one sentence of the visual template
        part = {"tmpl_au": "Face", "tmpl_head": "Head", "tmpl_gaze": "Gaze"}[cond]
        def pick(t):
            m = re.search(rf"{part}: .*?(?= (Face|Head|Gaze): |$)", t)
            return m.group(0) if m else ""
        return [pick(vis.get(k, "")) for k in df.key]
    if cond in ("vlm4b_zs", "vlm4b_sft"):  # Qwen3-VL-4B descriptions: zero-shot vs small SFT ablation
        d4 = jl(os.path.join(os.path.dirname(FULL), "sft_q3vl4b", cond.split("_")[1], "desc.jsonl"), "desc")
        return ["Visual: " + flat(d4.get(k, "")) for k in df.key]
    if cond in ("vlm8b", "vlmv8b", "shuf8b"):  # Qwen3-VL-8B (main-table VLM): raw / verified / shuffled
        q8 = os.path.join(os.path.dirname(FULL), "full_q3vl8b")
        src_desc = jl(os.path.join(q8, "desc_verified.jsonl" if cond == "vlmv8b" else "desc.jsonl"), "desc")
        keys = df.key.values.copy()
        if cond == "shuf8b":
            rng = np.random.default_rng(1000 + seed)
            for sp in df.split.unique():
                idx = np.where(df.split.values == sp)[0]; perm = idx.copy()
                while True:
                    rng.shuffle(perm)
                    if not np.any(perm == idx):
                        break
                keys[idx] = df.key.values[perm]
        return ["Visual: " + flat(src_desc.get(k, "")) for k in keys]
    if cond in ("deva_v", "deva_av"):  # DEVA-style Chinese descriptions (deva_text.py), joint BERT encoding
        dfile = os.path.join(FULL, os.environ.get("DEVA_FILE", "deva_text.jsonl"))  # DEVA_FILE: faithful 10 fps variant (deva_dense.py)
        dv = jl(dfile, "vision_text")
        da = jl(dfile, "audio_text")
        return [dv.get(k, "") + (da.get(k, "") if cond == "deva_av" else "") for k in df.key]
    if cond == "tmpl_pros":  # ablation: openSMILE prosody template only
        return [pros.get(k, "") for k in df.key]
    if cond == "tmpl_v3b":  # visual template with v3b head-action sentence + prosody
        v3b = jl(os.path.join(FULL, "measure_text_v3b.jsonl"), "visual_text")
        return [f"{v3b.get(k, vis.get(k, ''))} {pros.get(k, '')}".strip() for k in df.key]
    if cond == "vlm":
        return ["Visual: " + flat(vlm.get(k, "")) for k in df.key]
    if cond == "vlmv":
        return ["Visual: " + flat(vlmv.get(k, "")) for k in df.key]
    if cond == "tmpl_vlmv":
        return [f"{tmpl[k]} Visual: {flat(vlmv.get(k, ''))}" for k in df.key]
    if cond in ("shuf", "tmpl_vis_shuf"):
        rng = np.random.default_rng(1000 + seed)
        src = df.key.values.copy()
        for sp in df.split.unique():  # permute within split; avoid self-matches
            idx = np.where(df.split.values == sp)[0]
            perm = idx.copy()
            while True:
                rng.shuffle(perm)
                if not np.any(perm == idx):
                    break
            src[idx] = df.key.values[perm]
        if cond == "tmpl_vis_shuf":  # control: another clip's visual template
            return [vis.get(k, "") for k in src]
        return ["Visual: " + flat(vlm.get(k, "")) for k in src]
    raise ValueError(cond)


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0:  # e.g. Conflict subset on MOSI (no unimodal labels)
        return {"acc2": float("nan"), "f1": float("nan"), "mae": float("nan"), "corr": float("nan"), "n": 0}
    corr = lambda a, b: float(pearsonr(a, b)[0]) if len(a) > 2 else float("nan")
    if DATASET in ("mosi", "mosei"):  # MMSA MOSI/MOSEI conventions; acc2/f1 = has-0 (non-negative vs negative)
        p = np.clip(p, -3, 3)
        yb, pb = (y >= 0).astype(int), (p >= 0).astype(int)
        nz = y != 0
        yn, pn = (y[nz] > 0).astype(int), (p[nz] > 0).astype(int)
        return {"acc2": float((yb == pb).mean()), "f1": float(f1_score(yb, pb, average="weighted")),
                "acc2_non0": float((yn == pn).mean()), "f1_non0": float(f1_score(yn, pn, average="weighted")),
                "acc7": float((np.round(np.clip(y, -3, 3)) == np.round(p)).mean()),
                "mae": float(np.abs(y - p).mean()), "corr": corr(y, p), "n": int(len(y))}
    p = np.clip(p, -1, 1)
    yb, pb = (y > 0).astype(int), (p > 0).astype(int)  # MMSA SIMS convention: [-1,0] neg vs (0,1] pos
    return {"acc2": float((yb == pb).mean()), "f1": float(f1_score(yb, pb, average="weighted")),
            "mae": float(np.abs(y - p).mean()), "corr": corr(y, p), "n": int(len(y))}


# ---- shared hyper-parameter tuning helpers (defaults reproduce the original behaviour exactly)
def add_tune_args(ap):
    ap.add_argument("--lr_new", type=float, default=None, help="lr for non-encoder params (default: same as --lr)")
    ap.add_argument("--loss", choices=["mse", "l1"], default="mse")
    ap.add_argument("--patience", type=int, default=0, help="early stopping on valid MAE (0 = off)")
    ap.add_argument("--valid_only", action="store_true", help="tuning: skip test evaluation, save valid metrics only")
    ap.add_argument("--polw", type=float, default=0.0, help="polarity head: BCE on (label > 0) with this weight; Acc-2 also reported from it")


def pol_acc2(y, logit):
    """Acc-2 from a polarity logit with the same convention as metrics(): label > 0 positive (label 0 counted as negative)."""
    y, logit = np.asarray(y, float), np.asarray(logit, float)
    return float(((y > 0) == (logit > 0)).mean()) if len(y) else float("nan")


def make_opt(model, a):
    if a.lr_new is None:
        return torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    is_enc = lambda n: n.startswith(("enc.", "bert."))
    return torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if is_enc(n) and p.requires_grad], "lr": a.lr},
                              {"params": [p for n, p in model.named_parameters() if not is_enc(n) and p.requires_grad], "lr": a.lr_new}],
                             weight_decay=0.01)


def loss_fn(a):
    return torch.nn.functional.l1_loss if a.loss == "l1" else torch.nn.functional.mse_loss


def stop_early(a, ep, best_ep):
    return a.patience > 0 and ep - best_ep >= a.patience


def save_valid_only(a, path, hist, best_ep):
    cfg = {k: v for k, v in vars(a).items()}
    json.dump({"config": cfg, "best_epoch": best_ep, "valid": hist[best_ep], "valid_hist": hist}, open(path, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=320)
    add_tune_args(ap)
    a = ap.parse_args()
    out_dir = os.path.join(FULL, os.environ.get("RESULTS_DIR", "results"))
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{a.cond}_seed{a.seed}.json")
    if os.path.exists(out_json):
        print("exists", out_json); return

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    df = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})
    df["aux"] = build_aux(df, a.cond, a.seed)
    tok = AutoTokenizer.from_pretrained(BERT)

    def enc(sub):
        kw = dict(truncation="only_second" if a.cond != "text" else True, max_length=a.max_len, padding=False)
        if a.cond == "text":
            e = tok(list(sub.text), **kw)
        else:
            try:
                e = tok(list(sub.text), list(sub.aux), **kw)
            except Exception:  # some transcripts alone exceed max_len (MOSEI): only those pairs fall back to longest_first
                parts = []
                for t, x in zip(sub.text, sub.aux):
                    try:
                        parts.append(tok(t, x, **kw))
                    except Exception:
                        parts.append(tok(t, x, **{**kw, "truncation": "longest_first"}))
                e = {k: [p[k] for p in parts] for k in ("input_ids", "attention_mask", "token_type_ids")}
        return [{"input_ids": e["input_ids"][i], "attention_mask": e["attention_mask"][i],
                 "token_type_ids": e["token_type_ids"][i], "labels": float(y)} for i, y in enumerate(sub.label)]

    def collate(b):
        L = max(len(x["input_ids"]) for x in b)
        pad = lambda k, v: torch.tensor([x[k] + [v] * (L - len(x[k])) for x in b])
        return {"input_ids": pad("input_ids", tok.pad_token_id), "attention_mask": pad("attention_mask", 0),
                "token_type_ids": pad("token_type_ids", 0), "labels": torch.tensor([x["labels"] for x in b])}

    sp = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test")}
    if os.environ.get("PRE_SMOKE"):  # tiny CPU smoke test only
        sp = {k: v.iloc[:int(os.environ["PRE_SMOKE"])].reset_index(drop=True) for k, v in sp.items()}
    ds = {s: enc(d) for s, d in sp.items()}
    lens = [len(x["input_ids"]) for x in ds["train"]]
    trunc = float(np.mean([l >= a.max_len for l in lens]))
    g = torch.Generator().manual_seed(a.seed)
    dl_tr = DataLoader(ds["train"], batch_size=a.bs, shuffle=True, collate_fn=collate, generator=g)
    dl = {s: DataLoader(ds[s], batch_size=64, collate_fn=collate) for s in ("valid", "test")}

    model = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=2 if a.polw else 1, problem_type="regression").cuda()  # 2nd output = polarity logit
    opt = make_opt(model, a)
    steps = len(dl_tr) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    @torch.no_grad()
    def predict(loader, with_pol=False):
        model.eval(); ps = []
        for b in loader:
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ps.append(model(**{k: v for k, v in b.items() if k != "labels"}).logits.float().cpu())
        model.train()
        o = torch.cat(ps).numpy()
        return (o[:, 0], o[:, 1] if a.polw else None) if with_pol else o[:, 0]

    best, best_state, t0, hist = 1e9, None, time.time(), []
    for ep in range(a.epochs):
        for b in dl_tr:
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(**{k: v for k, v in b.items() if k != "labels"}).logits.float()
            out = lo[:, 0]
            loss = loss_fn(a)(out, b["labels"].float())
            if a.polw:
                loss = loss + a.polw * torch.nn.functional.binary_cross_entropy_with_logits(lo[:, 1], (b["labels"] > 0).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        pv, lv = predict(dl["valid"], with_pol=True)
        vm = metrics(sp["valid"].label, pv)
        if a.polw:
            vm["acc2_pol"] = pol_acc2(sp["valid"].label, lv)
        hist.append(vm)
        if vm["mae"] < best:
            best = vm["mae"]; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{a.cond} s{a.seed}] ep{ep} valid mae={vm['mae']:.4f} acc2={vm['acc2']:.4f}", flush=True)
        if stop_early(a, ep, best_ep):
            break
    if a.valid_only:
        save_valid_only(a, out_json, hist, best_ep); print("valid", hist[best_ep]); return
    model.load_state_dict(best_state)
    pt, lt = predict(dl["test"], with_pol=True)
    te = sp["test"]
    res = {"cond": a.cond, "seed": a.seed, "best_epoch": best_ep, "train_trunc_rate": trunc,
           "train_len_mean": float(np.mean(lens)), "sec": round(time.time() - t0, 1),
           "valid": hist[best_ep], "test": metrics(te.label, pt)}
    for name, mask in (("conflict", te.category == "conflict"),
                       ("align", te.category.isin(["equal", "spne"]))):
        res[f"test_{name}"] = metrics(te.label[mask], pt[mask.values])
    if a.polw:
        res["test_pol_acc2"] = pol_acc2(te.label, lt); res["polw"] = a.polw
    pd.DataFrame({"key": te.key, "label": te.label, "pred": pt, "category": te.category, **({"plogit": lt} if a.polw else {})}).to_csv(
        os.path.join(out_dir, f"{a.cond}_seed{a.seed}_test_pred.csv"), index=False)
    json.dump(res, open(out_json, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("test", "test_conflict", "test_align")}))


if __name__ == "__main__":
    main()
