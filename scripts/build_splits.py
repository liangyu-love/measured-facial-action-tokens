"""Reconstruct EmoMM-style Align / Conflict / Missing subsets from CH-SIMS v2(s) labels.

EmoMM (arXiv 2605.01024) recipe as we understand it (verify against the paper's App. A before final use):
  * CH-SIMS v2 part: 2,000 clips = 1,000 Conflict + 500 Equal + 500 Same-Polarity-Not-Equal (SPNE).
      - Equal  : label_T == label_A == label_V == label (overall)
      - SPNE   : every unimodal polarity == overall polarity, but values not all equal
      - Conflict: at least one unimodal polarity != overall polarity
    Align = Equal + SPNE.
  * Missing: each base clip is expanded "one-to-three" (mask T / A / V in turn); the split is decided
    on the ORIGINAL clip id before expansion, so no clip leaks across splits. Tags inherit
    Missing-Align / Missing-Conflict from the base clip.
  * Speaker-independent, stratified 7:1:2 train/valid/test. CH-SIMS v2 has no speaker column; we use
    `video_id` (source video) as the group proxy. TODO: replace with true speaker ids if available.

Usage:
  python build_splits.py --meta ~/work/msa-pilot/data/ch-sims-v2s/meta.csv --out ~/work/msa-pilot/outputs/splits
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MODS = ("T", "A", "V")


def polarity(x, eps=0.0):
    """CH-SIMS labels are in [-1, 1]; |x| <= eps counts as neutral."""
    return np.where(x > eps, 1, np.where(x < -eps, -1, 0))


def load_meta(path):
    df = pd.read_csv(path, dtype={"video_id": str, "clip_id": str})  # keep zero-padded ids
    # Normalise column names seen in the CH-SIMS / MMSA releases.
    df.columns = [c.strip() for c in df.columns]
    need = ["video_id", "clip_id", "label", "label_T", "label_A", "label_V"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"meta csv missing columns {miss}; got {list(df.columns)}")
    df["key"] = df["video_id"].astype(str) + "/" + df["clip_id"].astype(str)
    return df


def categorize(df, eps):
    p = polarity(df["label"].values, eps)
    pu = {m: polarity(df[f"label_{m}"].values, eps) for m in MODS}
    same_pol = np.all([pu[m] == p for m in MODS], axis=0)
    equal = np.all([np.isclose(df[f"label_{m}"].values, df["label"].values) for m in MODS], axis=0)
    cat = np.where(~same_pol, "conflict", np.where(equal, "equal", "spne"))
    df = df.assign(polarity=p, category=cat)
    # Which modalities disagree with the overall polarity (useful for analysis).
    df["conflict_mods"] = ["".join(m for m in MODS if pu[m][i] != p[i]) for i in range(len(df))]
    return df


def sample(df, n_conflict, n_equal, n_spne, seed):
    out = []
    for cat, n in (("conflict", n_conflict), ("equal", n_equal), ("spne", n_spne)):
        pool = df[df.category == cat]
        if len(pool) < n:
            print(f"[warn] only {len(pool)} '{cat}' clips available, wanted {n}; taking all")
            n = len(pool)
        # Stratify the draw by polarity so positive/negative/neutral ratios are preserved.
        for _, g in pool.groupby("polarity"):
            k = min(len(g), max(1, round(n * len(g) / len(pool))))
            out.append(g.sample(n=k, random_state=seed))
    base = pd.concat(out).drop_duplicates("key").reset_index(drop=True)
    base["subset"] = np.where(base.category == "conflict", "Conflict", "Align")
    return base


def split_7_1_2(base, seed):
    """Group by video_id (speaker proxy), stratify by subset x polarity; 10 folds -> 7/1/2."""
    strat = base["subset"] + "_" + base["polarity"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold = np.empty(len(base), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(base, strat, groups=base["video_id"])):
        fold[idx] = k
    base["split"] = np.select([fold < 7, fold == 7], ["train", "valid"], default="test")
    return base


def expand_missing(base, max_missing):
    rows = []
    for r in base.itertuples(index=False):
        d = r._asdict()
        # one-to-three: mask each single modality; optionally also pairs (max_missing=2).
        masks = [m for m in MODS] if max_missing == 1 else [m for m in MODS] + ["TA", "TV", "AV"]
        for mk in masks:
            rows.append({**d, "subset": f"Missing-{d['subset']}", "missing": mk,
                         "key_missing": f"{d['key']}#-{mk}"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.0, help="|label|<=eps -> neutral")
    ap.add_argument("--n-conflict", type=int, default=1000)
    ap.add_argument("--n-equal", type=int, default=500)
    ap.add_argument("--n-spne", type=int, default=500)
    ap.add_argument("--max-missing", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    df = categorize(load_meta(a.meta), a.eps)
    print("category counts (full set):", df.category.value_counts().to_dict())
    base = split_7_1_2(sample(df, a.n_conflict, a.n_equal, a.n_spne, a.seed), a.seed)
    base["missing"] = ""
    miss = expand_missing(base, a.max_missing)

    cols = ["key", "video_id", "clip_id", "subset", "category", "split", "missing", "conflict_mods",
            "polarity", "label", "label_T", "label_A", "label_V"]
    base[cols].to_csv(os.path.join(a.out, "emomm_chsims_base.csv"), index=False)
    miss[cols + ["key_missing"]].to_csv(os.path.join(a.out, "emomm_chsims_missing.csv"), index=False)

    # Sanity: no video_id appears in more than one split.
    leak = base.groupby("video_id")["split"].nunique().gt(1).sum()
    stats = {
        "seed": a.seed,
        "base": base.groupby(["split", "subset"]).size().unstack(fill_value=0).to_dict(),
        "missing_rows": len(miss),
        "video_ids_in_multiple_splits": int(leak),
    }
    with open(os.path.join(a.out, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
