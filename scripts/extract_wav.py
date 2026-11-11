"""Extract 16 kHz mono wav for every clip (imageio-ffmpeg binary; CPU only). -> <MSA_DATA>/wav/<video>__<clip>.wav
Usage: MSA_DATA=outputs/full MSA_RAW=data/chsims2/Raw python extract_wav.py --workers 4"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg
import pandas as pd

from train_bert import FULL

ROOT = os.path.expanduser("~/work/msa-pilot")
ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
RAW = os.path.join(ROOT, os.environ.get("MSA_RAW", "data/chsims2/Raw"))
OUT = os.path.join(FULL, "wav"); os.makedirs(OUT, exist_ok=True)
ff = imageio_ffmpeg.get_ffmpeg_exe()
c = pd.read_csv(os.path.join(FULL, "clips.csv"), dtype={"video_id": str, "clip_id": str})


def one(k):
    src = os.path.join(RAW, f"{k}.mp4"); dst = os.path.join(OUT, k.replace("/", "__") + ".wav")
    if os.path.exists(dst):
        return 0
    return subprocess.run([ff, "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


with ThreadPoolExecutor(a.workers) as ex:
    rc = list(ex.map(one, c.key))
print("clips", len(rc), "failed", sum(r != 0 for r in rc), "wav files", len(os.listdir(OUT)))
