"""Quick peek at interim results (mean over available seeds)."""
import glob
import json
import os

import numpy as np

F = os.path.expanduser("~/work/msa-pilot/outputs/full")


def show(d, v):
    rs = [json.load(open(p)) for p in glob.glob(f"{F}/{d}/{v}_seed*.json")]
    if not rs:
        return
    f = lambda s, m: np.mean([r[s][m] for r in rs])
    print(f"{d}/{v} n={len(rs)} vMAE={f('valid', 'mae'):.4f} | test acc2={f('test', 'acc2'):.4f} mae={f('test', 'mae'):.4f} "
          f"corr={f('test', 'corr'):.4f} | conflict acc2={f('test_conflict', 'acc2'):.4f} corr={f('test_conflict', 'corr'):.4f}")


for v in ("crop8", "seg8", "dense16"):
    show("results_dense", v)
show("results_acc", "soft_emo"); show("results_ausoft", "soft")
for d in ("results_norm_rel", "results_norm_z"):
    show(d, "soft_emo")
show("results_diff2_ev", "diff2_uni"); show("results_pre", "au_soft_pre"); show("results_pre", "diff2_pre")
ev = {}
for p in glob.glob(f"{F}/results_evq/*.json"):
    r = json.load(open(p)); ev.setdefault(r["ev"], []).append(r)
for k, rs in ev.items():
    print(f"evq {k} n={len(rs)} vMAE={np.mean([r['valid_mae'] for r in rs]):.4f} corrV={np.mean([r['test_labelV']['corr'] for r in rs]):.4f} "
          f"conflict corrV={np.mean([r['test_labelV_conflict']['corr'] for r in rs]):.4f}")
