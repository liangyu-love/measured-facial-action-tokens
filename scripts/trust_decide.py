"""diff3_trust enters the uniform tuning only if a lambda beats diff3_unc (same tuned experts, BERT) on mean VALID MAE.
Writes outputs/full/results_bert2/trust_choice.json: {"name": ..., "trust": lambda} or {}."""
import glob
import json
import os

import numpy as np

D = os.path.expanduser("~/work/msa-pilot/outputs/full/results_bert2")
vm = lambda n: np.mean([json.load(open(p))["valid"]["mae"] for p in glob.glob(f"{D}/{n}_seed[1-5].json")] or [9])
base = vm("diff3_unc"); cand = {"diff3_trust01": 0.1, "diff3_trust03": 0.3}
scores = {n: vm(n) for n in cand}
best = min(scores, key=scores.get)
choice = {"name": best, "trust": cand[best]} if scores[best] < base else {}
json.dump(choice, open(f"{D}/trust_choice.json", "w"))
print(json.dumps({"diff3_unc_valid_mae": round(base, 4), **{k: round(v, 4) for k, v in scores.items()}, "choice": choice}))
