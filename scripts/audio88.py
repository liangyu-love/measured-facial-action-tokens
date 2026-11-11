"""All 88 eGeMAPSv02 functionals per clip (CPU, parallel). Output: outputs/full/audio_egemaps88.csv"""
import os
import subprocess
import tempfile
from multiprocessing import Pool

import pandas as pd

ROOT = os.path.expanduser("~/work/msa-pilot")
F = os.path.join(ROOT, "outputs/full")
RAW = os.path.join(ROOT, "data/chsims2/Raw")


def one(key):
    import imageio_ffmpeg
    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        try:
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", os.path.join(RAW, f"{key}.mp4"),
                            "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
            row = smile.process_file(wav).iloc[0].to_dict()
        except Exception as e:  # silent / broken audio -> NaN row
            print("[audio error]", key, repr(e)[:120], flush=True); row = {}
    return {"key": key, **row}


if __name__ == "__main__":
    keys = pd.read_csv(os.path.join(F, "clips.csv"), dtype=str).key.tolist()
    with Pool(8) as p:
        rows = p.map(one, keys, chunksize=16)
    pd.DataFrame(rows).to_csv(os.path.join(F, "audio_egemaps88.csv"), index=False)
    print("wrote", len(rows))
