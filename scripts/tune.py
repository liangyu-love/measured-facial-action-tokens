"""Uniform hyper-parameter tuning (same budget for every method on a dataset), selection on VALID only.
Stage 1: grid with seed 1, --valid_only (test never evaluated) -> outputs/<ds>/tune/<method>/<cfg>/...
Stage 2: best config per method (min valid MAE) -> 5 seeds on test -> outputs/<ds>/tuned/<method>/
Grid (sims2, mosi): epochs {6,10,15} (early stopping, patience 3) x lr_bert {1e-5,2e-5,3e-5} x lr_new {1e-4,1e-3}
                    x loss {mse,l1}; warmup 10% (unchanged). MOSEI (16k train clips): reduced grid
                    epochs {6,10} x lr_bert {1e-5,2e-5} x lr_new {1e-3} x loss {mse,l1}.
Usage: python tune.py stage1|stage2|summary --dataset sims2|mosi|mosei [--workers 3] [--dense_variant seg8]
"""
import argparse
import glob
import itertools
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.expanduser("~/work/msa-pilot")
SC = os.path.join(ROOT, "scripts")
_bb = os.path.expanduser("~/work/msa-pilot/outputs/full/sims_backbone.txt")
SIMS_ROBERTA = os.path.exists(_bb) and open(_bb).read().strip() == "roberta"  # decided by run_pretune.sh (valid MAE, all methods)
DS = {"sims2": {"data": "outputs/full", "enc": "models/chinese-roberta-wwm-ext" if SIMS_ROBERTA else "models/bert-base-chinese", "dataset": "sims2"},
      "mosi": {"data": "outputs/mosi", "enc": "models/bert-base-uncased", "dataset": "mosi"},
      "mosei": {"data": "outputs/mosei", "enc": "models/bert-base-uncased", "dataset": "mosei"}}
# method -> (script, args, results-dir env var)
METHODS = {"text": ("train_bert.py", ["--cond", "text"], "RESULTS_DIR"),
           "deva_av": ("train_bert.py", ["--cond", "deva_av"], "RESULTS_DIR"),
           "mag": ("au_soft.py", ["--variant", "mag"], "AUSOFT_RESULTS_DIR"),
           "au_soft": ("au_soft.py", ["--variant", "soft"], "AUSOFT_RESULTS_DIR"),
           "au_soft_emo": ("au_soft2.py", ["--variant", "soft_emo"], "AUSOFT2_RESULTS_DIR")}


STAGE_FT = {"diff3_unc", "diff2_pre"}  # extra grid option: unfreeze both experts for the last 2 epochs


def grid(ds, m=None):
    if ds == "mosei":
        g = itertools.product([6, 10], [1e-5, 2e-5], [1e-3], ["mse", "l1"])
    else:
        g = itertools.product([6, 10, 15], [1e-5, 2e-5, 3e-5], [1e-4, 1e-3], ["mse", "l1"])
    out = [{"epochs": e, "lr": lr, "lr_new": ln, "loss": lo} for e, lr, ln, lo in g]
    if m in STAGE_FT:  # ONE extra grid point: the method's best stage-1 config + staged expert unfreezing (added in stage1)
        out = out + [json.load(open(p))["cfg"] for p in glob.glob(os.path.join(ROOT, DS[ds]["data"], "tune", m, "*_sft2", "cfg.json"))]
    return out


def cid(c):
    return f"e{c['epochs']}_lr{c['lr']:.0e}_ln{c['lr_new']:.0e}_{c['loss']}" + (f"_sft{c['stage_ft']}" if c.get("stage_ft") else "")


# SIMS v2 only: pretrained-evidence methods (need unimodal label_V encoders) and the raw-AU difference token
SIMS_EXTRA = {"au_soft_pre": ("pre_soft.py", ["--variant", "au_soft_pre"], "PRE_RESULTS_DIR"),
              "diff2_uni": ("diff2.py", ["--variant", "diff2_uni"], "DIFF2_RESULTS_DIR"),
              "diff2_pre": ("diff3.py", ["--name", "diff2_pre"], "DIFF3_RESULTS_DIR"),  # same model as pre_soft diff2_pre (diff3 with no flags); needed for --stage_ft
              "diff3_unc": ("diff3.py", ["--name", "diff3_unc", "--texp", "--unc"], "DIFF3_RESULTS_DIR"),  # best single DIFF variant by valid MAE
              # post-freeze addition (user-approved 11-08): cross-fitted text+face experts, with / without the HuBERT voice expert + trust head
              "diff3_oof": ("diff3.py", ["--name", "diff3_oof", "--precomp", "oof"], "DIFF3_RESULTS_DIR"),
              "diff3_tri3_trust": ("diff3.py", ["--name", "diff3_tri3_trust", "--precomp", "oof", "--audio_oof", "--trust", "0.1"], "DIFF3_RESULTS_DIR")}


def methods(a):
    m = dict(METHODS)
    if a.dataset == "sims2":
        m.update(SIMS_EXTRA)
        bf = os.path.join(ROOT, "outputs/full/results_diff3/best_flags.json")
        if os.path.exists(bf) and json.load(open(bf)):  # DIFF optimisation winner (only if something beat diff2_pre on valid)
            m["diff3_best"] = ("diff3.py", ["--name", "diff3_best", "--flags_file", bf], "DIFF3_RESULTS_DIR")
        dk = os.path.join(ROOT, "outputs/full/deva_dense_k.txt")
        if os.path.exists(dk):  # faithful DEVA-style (10 fps, >=3 consecutive frames, top-k by duration; k chosen on valid)
            m["deva_dense"] = ("train_bert.py", ["--cond", "deva_av"], "RESULTS_DIR", {"DEVA_FILE": f"deva_text_dense_k{open(dk).read().strip()}.jsonl"})
        tc = os.path.join(ROOT, "outputs/full/results_bert2/trust_choice.json")
        if os.path.exists(tc) and json.load(open(tc)):  # whom-to-trust head, only if it beat diff3_unc on valid
            m["diff3_trust"] = ("diff3.py", ["--name", "diff3_trust", "--texp", "--unc", "--trust", str(json.load(open(tc))["trust"])], "DIFF3_RESULTS_DIR")
    if a.dataset == "sims2" and a.dense_variant:
        m[f"dense_{a.dense_variant}"] = ("dense_soft.py", ["--variant", a.dense_variant], "DENSE_RESULTS_DIR")
    return m


def run(a, method, cfg, seed, outdir, valid_only):
    spec = methods(a)[method]; script, margs, envvar = spec[:3]; extra_env = spec[3] if len(spec) > 3 else {}
    d = DS[a.dataset]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "MSA_DATA": d["data"], "MSA_DATASET": d["dataset"],
           **extra_env, "ENCODER_DIR": os.path.join(ROOT, d["enc"]), envvar: outdir,
           "EXPERT_DIR_V": "results_evq_tuned", "EXPERT_DIR_T": "results_texp_roberta" if (SIMS_ROBERTA and a.dataset == "sims2") else "results_texp_tuned"}  # tuned unimodal experts (expert_tune.py)
    cmd = ["python", script, *margs, "--seed", str(seed), "--epochs", str(cfg["epochs"]), "--lr", str(cfg["lr"]),
           "--lr_new", str(cfg["lr_new"]), "--loss", cfg["loss"], "--patience", "3"] + (["--valid_only"] if valid_only else [])           + (["--stage_ft", str(cfg["stage_ft"])] if cfg.get("stage_ft") else [])
    logd = os.path.join(ROOT, d["data"], "tune_logs"); os.makedirs(logd, exist_ok=True)
    log = os.path.join(logd, f"{method}_{outdir.replace(os.sep, '_')}_s{seed}.log")
    r = subprocess.run(cmd, cwd=SC, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    return r.returncode


def stage1(a):
    d = DS[a.dataset]; jobs = []
    for m in methods(a):
        for c in grid(a.dataset, m):
            outdir = os.path.join("tune", m, cid(c))  # relative to the dataset dir (scripts join it onto FULL)
            if not glob.glob(os.path.join(ROOT, d["data"], outdir, "*_seed1.json")):
                jobs.append((m, c, outdir))
    print(f"[{a.dataset}] stage1: {len(jobs)} runs, {a.workers} parallel", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for (m, c, o), rc in zip(jobs, ex.map(lambda j: run(a, j[0], j[1], 1, j[2], True), jobs)):
            print(f"  {m} {cid(c)} rc={rc}", flush=True)
    # staged expert unfreezing: one extra point per STAGE_FT method = its best stage-1 config + stage_ft 2
    for m in [x for x in methods(a) if x in STAGE_FT]:
        b = best_configs(a).get(m)
        if not b or b["config"].get("stage_ft"):
            continue
        c = {**b["config"], "stage_ft": 2}; o = os.path.join("tune", m, cid(c))
        os.makedirs(os.path.join(ROOT, d["data"], o), exist_ok=True)
        json.dump({"cfg": c}, open(os.path.join(ROOT, d["data"], o, "cfg.json"), "w"))
        if not glob.glob(os.path.join(ROOT, d["data"], o, "*_seed1.json")):
            print(f"  {m} {cid(c)} rc={run(a, m, c, 1, o, True)}", flush=True)


def best_configs(a):
    d = DS[a.dataset]; best = {}
    for m in methods(a):
        cands = []
        for c in grid(a.dataset, m):
            fs = glob.glob(os.path.join(ROOT, d["data"], "tune", m, cid(c), "*_seed1.json"))
            if fs:
                r = json.load(open(fs[0]))
                cands.append((r["valid"]["mae"], c, r["best_epoch"]))
        if cands:
            v, c, be = min(cands, key=lambda x: x[0])
            best[m] = {"config": c, "valid_mae": v, "best_epoch": be, "n_configs": len(cands)}
    return best


def stage2(a):
    d = DS[a.dataset]; best = best_configs(a); jobs = []
    json.dump(best, open(os.path.join(ROOT, d["data"], "tuned_configs.json"), "w"), indent=2)
    for m, b in best.items():
        outdir = os.path.join("tuned", m)
        for s in range(a.seed_from, a.seed_to + 1):
            if not glob.glob(os.path.join(ROOT, d["data"], outdir, f"*_seed{s}.json")):
                jobs.append((m, b["config"], s, outdir))
    print(f"[{a.dataset}] stage2: {len(jobs)} runs", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for j, rc in zip(jobs, ex.map(lambda j: run(a, j[0], j[1], j[2], j[3], False), jobs)):
            print(f"  {j[0]} seed{j[2]} rc={rc}", flush=True)


def summary(a):
    d = DS[a.dataset]; best = json.load(open(os.path.join(ROOT, d["data"], "tuned_configs.json")))
    ms = ["acc2", "f1", "acc2_non0", "f1_non0", "acc7", "mae", "corr"] if a.dataset != "sims2" else ["acc2", "f1", "mae", "corr"]
    subs = ["test"] if a.dataset != "sims2" else ["test", "test_conflict", "test_align"]
    lines = [f"# tuned results ({a.dataset}); selection by valid MAE (seed 1), then all available seeds on test (mean±std)"]
    for m, b in best.items():
        lines.append(f"- {m}: {cid(b['config'])} (valid MAE {b['valid_mae']:.4f}, {b['n_configs']} configs)")
    for sub in subs:
        lines += [f"\n### {sub}", "| method | seeds | " + " | ".join(ms) + " |", "|---" * (len(ms) + 2) + "|"]
        for m in best:
            rs = [json.load(open(p)) for p in glob.glob(os.path.join(ROOT, d["data"], "tuned", m, "*_seed*.json"))]
            rs = [r for r in rs if sub in r]
            if rs:
                lines.append(f"| {m} | {len(rs)} | " + " | ".join(
                    f"{np.mean([r[sub][k] for r in rs]):.4f}±{np.std([r[sub][k] for r in rs], ddof=1) if len(rs) > 1 else 0:.4f}" for k in ms) + " |")
    txt = "\n".join(lines)
    open(os.path.join(ROOT, d["data"], "tuned_results.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "summary"])
    ap.add_argument("--dataset", required=True, choices=list(DS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dense_variant", default=None)
    ap.add_argument("--seed_from", type=int, default=1)
    ap.add_argument("--seed_to", type=int, default=5)
    a = ap.parse_args()
    {"stage1": stage1, "stage2": stage2, "summary": summary}[a.stage](a)
