"""Polarity-head round (user choice, 11-08): every method keeps its uniformly-tuned configuration (tuned_configs.json, selected on
valid MAE) and gets the same polarity head; lambda in {0.1, 0.3, 1.0} is chosen on VALID polarity Acc-2 (seed 1; ties -> lower valid MAE);
then 10 test seeds. Same budget for every method. Outputs under outputs/full/polsel/<m>/l<lam>/ and outputs/full/tuned_pol/<m>/.
Usage: python pol_round.py"""
import glob
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.expanduser("~/work/msa-pilot"); F = os.path.join(ROOT, "outputs/full"); SC = os.path.join(ROOT, "scripts")
METHODS = {"text": ("train_bert.py", ["--cond", "text"], "RESULTS_DIR", {}),
           "au_soft_emo": ("au_soft2.py", ["--variant", "soft_emo"], "AUSOFT2_RESULTS_DIR", {}),
           "deva_dense": ("train_bert.py", ["--cond", "deva_av"], "RESULTS_DIR", {"DEVA_FILE": "deva_text_dense_k%s.jsonl" % open(os.path.join(F, "deva_dense_k.txt")).read().strip()}),
           "diff3_tri3_trust": ("diff3.py", ["--name", "diff3_tri3_trust", "--precomp", "oof", "--audio_oof", "--trust", "0.1"], "DIFF3_RESULTS_DIR", {})}
LAMS = (0.1, 0.3, 1.0)
best_cfg = json.load(open(os.path.join(F, "tuned_configs.json")))


def free_mb():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])


def run(m, lam, seed, outdir, valid_only):
    script, margs, envvar, extra = METHODS[m]; c = best_cfg[m]["config"]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "MSA_DATA": "outputs/full", "MSA_DATASET": "sims2",
           "ENCODER_DIR": os.path.join(ROOT, "models/bert-base-chinese"), "EXPERT_DIR_V": "results_evq_tuned", "EXPERT_DIR_T": "results_texp_tuned",
           **extra, envvar: outdir}
    cmd = ["python", script, *margs, "--seed", str(seed), "--epochs", str(c["epochs"]), "--lr", str(c["lr"]), "--lr_new", str(c["lr_new"]),
           "--loss", c["loss"], "--patience", "3", "--polw", str(lam)] + (["--valid_only"] if valid_only else [])
    while free_mb() < 7000:
        time.sleep(30)
    log = os.path.join(F, "logs", f"pol_{m}_{outdir.replace('/', '_')}_s{seed}.log")
    return subprocess.run(cmd, cwd=SC, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT).returncode


# 1) lambda selection on valid (seed 1)
jobs = [(m, l) for m in METHODS for l in LAMS if not glob.glob(os.path.join(F, "polsel", m, f"l{l}", "*_seed1.json"))]
with ThreadPoolExecutor(2) as ex:
    print("selection rc:", list(ex.map(lambda j: run(j[0], j[1], 1, f"polsel/{j[0]}/l{j[1]}", True), jobs)), flush=True)
chosen = {}
for m in METHODS:
    cand = []
    for l in LAMS:
        fs = glob.glob(os.path.join(F, "polsel", m, f"l{l}", "*_seed1.json"))
        if fs:
            v = json.load(open(fs[0]))["valid"]; cand.append((-v["acc2_pol"], v["mae"], l))
    chosen[m] = sorted(cand)[0][2]
json.dump(chosen, open(os.path.join(F, "pol_lambda.json"), "w"))
print("chosen lambda (valid polarity Acc-2):", chosen, flush=True)
# 2) 10 test seeds
jobs = [(m, s) for m in METHODS for s in range(1, 11) if not glob.glob(os.path.join(F, "tuned_pol", m, f"*_seed{s}.json"))]
with ThreadPoolExecutor(2) as ex:
    print("test rc:", list(ex.map(lambda j: run(j[0], chosen[j[0]], j[1], f"tuned_pol/{j[0]}", False), jobs)), flush=True)
print("POL_ROUND_DONE", flush=True)
