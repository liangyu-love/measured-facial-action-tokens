"""openSMILE eGeMAPS prosody -> short natural-language template.

Two steps:
  extract  : video/audio -> 16 kHz mono wav (imageio-ffmpeg binary) -> eGeMAPSv02 functionals -> features.csv
  describe : bin features into low/mid/high using quantiles from the TRAIN split only -> prosody_text.jsonl

Usage:
  python prosody_text.py extract  --splits .../emomm_chsims_base.csv --video-root .../Raw --out .../prosody_feats.csv
  python prosody_text.py describe --splits .../emomm_chsims_base.csv --feats .../prosody_feats.csv --out .../prosody_text.jsonl
"""
import argparse
import json
import os
import subprocess
import tempfile

import pandas as pd

# eGeMAPSv02 functional names -> (short name, phrase used in the template)
FEATS = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": ("pitch", "average pitch"),
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": ("pitch_var", "pitch variation"),
    "loudness_sma3_amean": ("loud", "loudness"),
    "loudness_sma3_stddevNorm": ("loud_var", "loudness variation"),
    "loudnessPeaksPerSec": ("rate", "speech rate (syllable-like peaks per second)"),
    "MeanUnvoicedSegmentLength": ("pause", "average pause length"),
    "jitterLocal_sma3nz_amean": ("jitter", "voice jitter"),
    "shimmerLocaldB_sma3nz_amean": ("shimmer", "voice shimmer"),
    "HNRdBACF_sma3nz_amean": ("hnr", "voice clarity (harmonics-to-noise)"),
}
LEVELS = ("low", "medium", "high")


def to_wav(src, dst):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst], check=True)


def cmd_extract(a):
    import opensmile
    from gen_descriptions import video_path
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    keys = pd.read_csv(a.splits)["key"].drop_duplicates().tolist()[: a.limit or None]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for k in keys:
            p = video_path(a.video_root, k)
            if p is None:
                print("[skip] no media for", k)
                continue
            wav = os.path.join(td, "a.wav")
            to_wav(p, wav)
            f = smile.process_file(wav).iloc[0]
            rows.append({"key": k, **{v[0]: float(f[c]) for c, v in FEATS.items()}})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote", len(rows), "rows ->", a.out)


def cmd_describe(a):
    sp = pd.read_csv(a.splits)[["key", "split"]].drop_duplicates("key")
    ft = pd.read_csv(a.feats).merge(sp, on="key")
    short = [v[0] for v in FEATS.values()]
    train = ft[ft.split == "train"]
    q = train[short].quantile([1 / 3, 2 / 3])  # thresholds fit on train only -> no test leakage
    with open(a.out, "w") as f:
        for r in ft.itertuples(index=False):
            parts = []
            for c, (s, phrase) in FEATS.items():
                v = getattr(r, s)
                lv = LEVELS[int(v > q.loc[1 / 3, s]) + int(v > q.loc[2 / 3, s])]
                parts.append(f"{phrase} is {lv}")
            text = "Speaker's voice: " + "; ".join(parts) + "."
            f.write(json.dumps({"key": r.key, "prosody_text": text}, ensure_ascii=False) + "\n")
    print("wrote", len(ft), "rows ->", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--splits", required=True)
    e.add_argument("--video-root", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=0)
    d = sub.add_parser("describe")
    d.add_argument("--splits", required=True)
    d.add_argument("--feats", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    {"extract": cmd_extract, "describe": cmd_describe}[a.cmd](a)


if __name__ == "__main__":
    main()
