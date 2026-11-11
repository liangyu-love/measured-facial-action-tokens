import glob, json, os
import numpy as np
D = os.path.expanduser("~/work/msa-pilot/outputs/full/results_diff3")
names = sorted({os.path.basename(p).rsplit("_seed", 1)[0] for p in glob.glob(D + "/*_seed*.json")})
for n in names:
    rs = [json.load(open(p)) for p in glob.glob(f"{D}/{n}_seed*.json")]
    rs = [r for r in rs if "test" in r]
    if not rs:
        continue
    f = lambda k, m: np.mean([r[k][m] for r in rs])
    d = lambda k, s=None: np.mean([r["diag"][k][s] if s else r["diag"][k] for r in rs])
    print(f"{n:12s} n={len(rs)} vMAE={f('valid', 'mae'):.4f} acc2={f('test', 'acc2'):.4f} mae={f('test', 'mae'):.4f} corr={f('test', 'corr'):.4f} | "
          f"conf acc2={f('test_conflict', 'acc2'):.4f} corr={f('test_conflict', 'corr'):.4f} | shufC={d('shuffle_diff_delta_conflict', 'corr'):+.3f} "
          f"auroc={d('auroc_gap_sign_conflict'):.3f} sp={d('spearman_gap_vs_trueTVgap'):+.3f}"
          + (f" w C/A={d('moe_w_conflict'):.2f}/{d('moe_w_align'):.2f}" if "moe_w_conflict" in rs[0]["diag"] else ""))
