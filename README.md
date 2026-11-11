# Measured Facial Action Tokens

Code for the paper

> **A Second Opinion from the Face: Measured Facial Action Tokens for Multimodal Sentiment Analysis under Text–Visual Conflict**
> Zijie Guo. Manuscript in preparation for *Knowledge-Based Systems*.

Per-frame facial action unit (AU) intensities, emotion probabilities and EMFACS-style AU products are estimated with
[Py-Feat](https://py-feat.org), projected into BERT's word-embedding space and appended to the transcript as soft tokens, so that one
encoder attends jointly to words and facial measurements. The repository also contains every baseline, the uniform tuning protocol, the
cross-validation over all CH-SIMS v2 clips, the explicit conflict-modelling analyses (gates, discrepancy tokens, cross-fitted experts,
capacity controls, speech expert, polarity head), the MLLM-description study and the scripts that recompute every number in the paper
from per-seed predictions.

## Data

No data are redistributed here. Obtain the datasets from their original sources and follow their licences:

- **CH-SIMS v2**: <https://github.com/thuiar/ch-sims-v2>
- **CMU-MOSI / CMU-MOSEI**: <https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK>

Pretrained models are loaded from local copies of `bert-base-chinese`, `bert-base-uncased`, `chinese-roberta-wwm-ext`,
`microsoft/deberta-v3-base`, `TencentGameMate/chinese-hubert-base` and the Qwen-VL checkpoints (Hugging Face).

## Environment

Python 3.12, one NVIDIA RTX 3090 (24 GB). Install PyTorch 2.11.0 first, then the pinned packages:

```bash
pip install -c constraints.txt -r requirements.txt
```

The scripts assume the working layout `~/work/msa-pilot/{scripts,data,outputs,models}` (copy `scripts/` there). Dataset-specific work
directories are selected with `MSA_DATA` (e.g. `outputs/full` for CH-SIMS v2, `outputs/mosi`, `outputs/mosei`) and `MSA_DATASET`
(`sims2`, `mosi`, `mosei`).

## Pipeline

| Step | Scripts |
| --- | --- |
| Clip list, official split, Conflict / Align / strict text–visual subsets | `make_clips.py`, `build_splits.py` |
| 8 uniformly sampled frames, Py-Feat AUs (`pilot.py frames`, `pilot.py au`), emotion probabilities | `pilot.py`, `emo_frames.py` |
| Dense (10 fps) face-crop AUs | `dense_au.py` |
| Audio features (eGeMAPS, Chinese HuBERT), CLIP frame features | `extract_wav.py`, `audio88.py`, `prosody_text.py`, `hubert_feats.py`, `clip_feats.py` |
| **Main model**: AU soft tokens (+ emotions / EMFACS) | `au_soft.py`, `au_soft2.py --variant soft_emo` |
| Text-only BERT, AU text template, DEVA-style rule template, MLLM descriptions | `train_bert.py`, `measure_text.py`, `deva_text.py`, `deva_dense.py` |
| MAG-style gating, late fusion, CLIP tokens | `au_soft.py --variant mag / late / soft_clip` |
| Uniform tuning (same grid and budget for every method) and 10-seed runs | `tune.py`, `run_tune.sh` |
| 5-fold cross-validation over all CH-SIMS v2 clips | `make_cv.py`, `run_cv.sh`, `run_cv2.sh`, `cv_compare.py` |
| MLLM descriptions, contradiction scoring, QLoRA SFT | `pilot.py desc/verify`, `gen_full.py`, `score_v3.py`, `scale_desc.py`, `sft_data.py`, `sft_qwenvl.py` |
| Explicit conflict modelling (gates, experts, discrepancy tokens, cross-fitting, capacity controls, speech expert, trust and polarity heads) | `gate_*.py`, `xgate_bert.py`, `evq.py`, `pre_soft.py`, `texp.py`, `diff2.py`, `diff3.py`, `oof_experts.py`, `reg_experts.py`, `aexp.py`, `audio_expert_hubert.py`, `ext_*.py`, `pol_round.py`, `tri3_tuned.py` |
| Evidence maps and occlusion | `evi_bert.py`, `evi3.py`, `interp3.py` |
| Significance, equivalence (TOST), McNemar and the full audit of reported numbers | `sig_tests.py`, `tune_sig.py`, `fastsig.py`, `audit.py` |
| Computational cost | `bench_cost.py` |
| Paper figures and the recomputation of figure data | `paper_figures/` |

The `run_*.sh` files are the job queues that produced the reported results, in roughly chronological order; every training script writes
per-seed `*_test_pred.csv` files, and `audit.py` recomputes all metrics and tests from those files.

## Notes

- Reported results are means over 10 seeds (CH-SIMS v2) or 5 seeds (MOSI, MOSEI) after uniform tuning on the validation split; the
  "default recipe" (6 epochs, encoder learning rate 2e-5, 5 seeds) is used for controlled representation comparisons.
- Py-Feat outputs are automatic estimates, not human FACS codes.

## License

MIT (see `LICENSE`). Third-party models and datasets keep their own licences.
