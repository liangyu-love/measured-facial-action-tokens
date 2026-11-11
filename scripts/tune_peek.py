"""Peek at tuned configs (valid-selected) and the available tuned test seeds. Usage: python tune_peek.py full|mosi"""
import glob
import json
import os
import sys

import numpy as np

R = os.path.expanduser(f"~/work/msa-pilot/outputs/{sys.argv[1] if len(sys.argv) > 1 else 'full'}")
best = json.load(open(f"{R}/tuned_configs.json"))
for m, b in sorted(best.items(), key=lambda x: x[1]["valid_mae"]):
    c = b["config"]
    cfg = "e%d lr%.0e ln%.0e %s%s" % (c["epochs"], c["lr"], c["lr_new"], c["loss"], " +stageFT" if c.get("stage_ft") else "")
    rs = [json.load(open(p)) for p in glob.glob(f"{R}/tuned/{m}/*_seed*.json")]
    rs = [r for r in rs if "test" in r]
    line = f"{m:14s} {cfg:30s} valid MAE {b['valid_mae']:.4f} ({b['n_configs']} cfgs)"
    if rs:
        f = lambda k, s: np.mean([r[s][k] for r in rs])
        line += f" | test n={len(rs)} Acc2 {f('acc2', 'test'):.4f} MAE {f('mae', 'test'):.4f} Corr {f('corr', 'test'):.4f}"
        if "test_conflict" in rs[0] and rs[0]["test_conflict"]["n"]:
            line += f" | conflict Acc2 {f('acc2', 'test_conflict'):.4f}"
    print(line)
