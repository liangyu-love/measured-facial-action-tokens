"""All 4403 labeled CH-SIMS v2(s) clips with official split + Conflict/Equal/SPNE category (full label set)."""
import os

from build_splits import categorize, load_meta

ROOT = os.path.expanduser("~/work/msa-pilot")
df = categorize(load_meta(os.path.join(ROOT, "data/chsims2/meta.csv")), eps=0.0)
df = df.rename(columns={"mode": "split"})
out = os.path.join(ROOT, "outputs/full")
os.makedirs(out, exist_ok=True)
df[["key", "video_id", "clip_id", "split", "text", "label", "label_T", "label_A", "label_V", "category",
    "conflict_mods"]].to_csv(os.path.join(out, "clips.csv"), index=False)
print(df.groupby(["split", "category"]).size().unstack(fill_value=0))
