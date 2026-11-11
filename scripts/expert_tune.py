"""Expert tuning (CH-SIMS v2, VALID only, seed 1), then final experts for seeds 1-10 with the chosen configs.
Visual expert (evq.py, raw8emo -> label_V): lr {3e-4, 1e-3, 3e-3} x layers {1, 2, 3} x polarity-head weight {0, 0.5, 1}, 40 epochs.
Text expert (texp.py, transcript -> label_T): lr {1e-5, 2e-5, 3e-5} x epochs {4, 6, 10} x polarity-head weight {0, 0.5, 1}.
Selection (combined): rank(valid MAE) + rank(polarity accuracy on VALID Conflict clips), ties -> MAE. Outputs: outputs/full/results_evq_tuned/raw8emo_seed<s>.pt,
outputs/full/results_texp_tuned/seed<s>.pt, outputs/full/expert_tuning.json"""
import glob, itertools, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.expanduser("~/work/msa-pilot"); F = os.path.join(ROOT, "outputs/full"); SC = os.path.join(ROOT, "scripts")
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "MSA_DATASET": "sims2", "MSA_DATA": "outputs/full"}


def run(cmd, env, log):
    return subprocess.run(cmd, cwd=SC, env={**ENV, **env}, stdout=open(log, "w"), stderr=subprocess.STDOUT).returncode


os.makedirs(os.path.join(F, "logs"), exist_ok=True)
vgrid = [{"lr": lr, "layers": L, "pol": w} for lr, L, w in itertools.product([3e-4, 1e-3, 3e-3], [1, 2, 3], [0.0, 0.5, 1.0])]
tgrid = [{"lr": lr, "epochs": e, "pol": w} for lr, e, w in itertools.product([1e-5, 2e-5, 3e-5], [4, 6, 10], [0.0, 0.5, 1.0])]
vtag = lambda c: f"_lr{c['lr']:.0e}_L{c['layers']}_p{c['pol']}"
ttag = lambda c: f"_lr{c['lr']:.0e}_e{c['epochs']}_p{c['pol']}"
jobs = [(["python", "evq.py", "--ev", "raw8emo", "--seed", "1", "--lr", str(c["lr"]), "--layers", str(c["layers"]), "--pol", str(c["pol"]), "--tag", vtag(c)],
         {"EXPERT_DIR_V": "expert_grid_v"}, os.path.join(F, "logs", f"egrid_v{vtag(c)}.log")) for c in vgrid] + \
       [(["python", "texp.py", "--seed", "1", "--lr", str(c["lr"]), "--epochs", str(c["epochs"]), "--pol", str(c["pol"]), "--tag", ttag(c)],
         {"EXPERT_DIR_T": "expert_grid_t"}, os.path.join(F, "logs", f"egrid_t{ttag(c)}.log")) for c in tgrid]
with ThreadPoolExecutor(3) as ex:
    print("grid rc:", list(ex.map(lambda j: run(*j), jobs)), flush=True)
import numpy as np
vj = {vtag(c): json.load(open(os.path.join(F, "expert_grid_v", f"raw8emo_seed1{vtag(c)}.json"))) for c in vgrid}
tj = {ttag(c): json.load(open(os.path.join(F, "expert_grid_t", f"seed1{ttag(c)}.json"))) for c in tgrid}
vres = {k: {"valid_mae": r["valid_mae"], "valid_conflict_polacc": r["valid_conflict_polacc"], "test_conflict_polacc": r["test_conflict_polacc"]} for k, r in vj.items()}
tres = {k: {"valid_mae": r["valid_labelT"]["mae"], "valid_conflict_polacc": r["valid_conflict_polacc"], "test_conflict_polacc": r["test_conflict_polacc"]} for k, r in tj.items()}


def pick(grid, tag, res):
    ks = [tag(c) for c in grid]
    rm = {k: i for i, k in enumerate(sorted(ks, key=lambda k: res[k]["valid_mae"]))}
    rp = {k: i for i, k in enumerate(sorted(ks, key=lambda k: -res[k]["valid_conflict_polacc"]))}
    return min(grid, key=lambda c: (rm[tag(c)] + rp[tag(c)], res[tag(c)]["valid_mae"]))


vb = pick(vgrid, vtag, vres); tb = pick(tgrid, ttag, tres)
vd = {"lr": 1e-3, "layers": 2, "pol": 0.0}; td = {"lr": 2e-5, "epochs": 6, "pol": 0.0}
json.dump({"visual_grid": vres, "text_grid": tres, "visual_best": vb, "text_best": tb, "default_visual": vd, "default_text": td,
           "before_after_seed1": {"visual_default": vres[vtag(vd)], "visual_selected": vres[vtag(vb)],
                                  "text_default": tres[ttag(td)], "text_selected": tres[ttag(tb)]}},
          open(os.path.join(F, "expert_tuning.json"), "w"), indent=2)
print("best visual", vb, vres[vtag(vb)], "(default", vres[vtag(vd)], ") | best text", tb, tres[ttag(tb)], "(default", tres[ttag(td)], ")", flush=True)
final = [(["python", "evq.py", "--ev", "raw8emo", "--seed", str(s), "--lr", str(vb["lr"]), "--layers", str(vb["layers"]), "--pol", str(vb["pol"])],
          {"EXPERT_DIR_V": "results_evq_tuned"}, os.path.join(F, "logs", f"evq_tuned_s{s}.log")) for s in range(1, 11)] + \
        [(["python", "texp.py", "--seed", str(s), "--lr", str(tb["lr"]), "--epochs", str(tb["epochs"]), "--pol", str(tb["pol"])],
          {"EXPERT_DIR_T": "results_texp_tuned"}, os.path.join(F, "logs", f"texp_tuned_s{s}.log")) for s in range(1, 11)]
with ThreadPoolExecutor(3) as ex:
    print("final rc:", list(ex.map(lambda j: run(*j), final)), flush=True)
vt = [json.load(open(p)) for p in glob.glob(os.path.join(F, "results_evq_tuned", "raw8emo_seed*.json"))]
tt = [json.load(open(p)) for p in glob.glob(os.path.join(F, "results_texp_tuned", "seed*.json"))]
summ = {"visual_tuned_10seeds": {"test_labelV_corr": np.mean([r["test_labelV"]["corr"] for r in vt]), "test_conflict_polacc": np.mean([r["test_conflict_polacc"] for r in vt])},
        "text_tuned_10seeds": {"test_labelT_corr": np.mean([r["test_labelT"]["corr"] for r in tt]), "test_conflict_polacc": np.mean([r["test_conflict_polacc"] for r in tt])}}
d = json.load(open(os.path.join(F, "expert_tuning.json"))); d.update(summ); json.dump(d, open(os.path.join(F, "expert_tuning.json"), "w"), indent=2)
print("experts:", len(vt), len(tt), json.dumps(summ))
