"""Results AUDIT (CPU only). Recomputes every key number from the raw per-seed *_test_pred.csv files (never from summaries).
Writes outputs/full/AUDIT.md. Usage: python audit.py"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics import f1_score, roc_auc_score

R = os.path.expanduser("~/work/msa-pilot/outputs")
F, M = os.path.join(R, "full"), os.path.join(R, "mosi")
L = []
P = L.append

# ------------------------------------------------------------------ metrics (same conventions as train_bert.metrics)
def m_sims(y, p):
    p = np.clip(p, -1, 1); yb, pb = (y > 0).astype(int), (p > 0).astype(int)
    return {"acc2": (yb == pb).mean(), "f1": f1_score(yb, pb, average="weighted"), "mae": np.abs(y - p).mean(),
            "corr": pearsonr(y, p)[0] if len(y) > 2 else np.nan, "n": len(y)}


def m_mosi(y, p):
    p = np.clip(p, -3, 3); yb, pb = (y >= 0).astype(int), (p >= 0).astype(int); nz = y != 0
    return {"acc2": (yb == pb).mean(), "f1": f1_score(yb, pb, average="weighted"),
            "acc2_non0": ((y[nz] > 0) == (p[nz] > 0)).mean(), "f1_non0": f1_score((y[nz] > 0).astype(int), (p[nz] > 0).astype(int), average="weighted"),
            "acc7": (np.round(np.clip(y, -3, 3)) == np.round(p)).mean(), "mae": np.abs(y - p).mean(), "corr": pearsonr(y, p)[0], "n": len(y)}


def load(d, v):
    fs = sorted(glob.glob(os.path.join(d, f"{v}_seed*_test_pred.csv")), key=lambda p: int(re.search(r"seed(\d+)_test", p).group(1)))
    return {int(re.search(r"seed(\d+)_test", p).group(1)): pd.read_csv(p).set_index("key") for p in fs}, fs


# ------------------------------------------------------------------ 1. data facts
c = pd.read_csv(os.path.join(F, "clips.csv"), dtype={"video_id": str, "clip_id": str}).set_index("key")
te = c[c.split == "test"]
sg = lambda x: np.sign(x)
SUB = {"full": te.index, "conflict": te.index[te.category == "conflict"], "align": te.index[te.category.isin(["equal", "spne"])],
       "tv_opposite": te.index[(sg(te.label_T) * sg(te.label_V)) < 0]}
cm = te[te.category == "conflict"].conflict_mods.fillna("")
P("# Results AUDIT (recomputed from per-seed prediction files)\n")
P("## 1. Data facts")
P(f"- CH-SIMS v2 split sizes (outputs/full/clips.csv): {c.split.value_counts().to_dict()}")
P("- Categories (scripts/build_splits.py `categorize`, eps=0): polarity = sign with 0 = neutral (3 classes). "
  "**Conflict = at least one unimodal polarity (T, A or V) differs from the OVERALL polarity**; Equal = all unimodal labels equal the overall label; "
  "SPNE = same polarity, not equal; Align = Equal + SPNE. Neutral (0) is its own polarity class, so 0 vs ±x counts as a disagreement.")
P(f"- Test: n={len(te)}; Conflict n={len(SUB['conflict'])}, Align n={len(SUB['align'])} (Equal {int((te.category=='equal').sum())}, SPNE {int((te.category=='spne').sum())}).")
P(f"- Composition of test Conflict by disagreeing modalities (conflict_mods): {cm.value_counts().to_dict()}; V involved: {int(cm.str.contains('V').sum())}; "
  f"audio-only: {int((cm == 'A').sum())}.")
P(f"- Strict text–face opposite polarity (sign(label_T)·sign(label_V) < 0) on test: n={len(SUB['tv_opposite'])} "
  f"({int(te.loc[SUB['conflict']].pipe(lambda x: ((sg(x.label_T) * sg(x.label_V)) < 0).sum()))} of them inside Conflict). "
  f"Test Conflict clips with label_V = 0: {int((te.loc[SUB['conflict']].label_V == 0).sum())}; with label_T = 0: {int((te.loc[SUB['conflict']].label_T == 0).sum())}.")
P("- SIMS Acc-2 convention: label > 0 positive vs label ≤ 0 (non-positive) — MMSA SIMS convention; predictions clipped to [−1, 1]. F1 weighted.")
mc = pd.read_csv(os.path.join(M, "clips.csv"))
P(f"- CMU-MOSI split sizes: {mc.split.value_counts().to_dict()}. Acc-2 has-0 = label ≥ 0 vs < 0; non-0 = excluding label 0; predictions clipped to [−3, 3].\n")

# ------------------------------------------------------------------ 2. SIMS table
SIMS = [("text", "results", "text"), ("MLLM description (vlm, Qwen2.5-VL-7B)", "results", "vlm"), ("deva_av (DEVA-style)", "results", "deva_av"),
        ("tmpl_au (AU template text)", "results", "tmpl_au"), ("mag (MAG-style)", "results_ausoft", "mag"), ("au_late", "results_ausoft", "late"),
        ("soft_clip (CLIP tokens)", "results_ausoft", "soft_clip"), ("au_soft", "results_ausoft", "soft"), ("au_soft+emo", "results_acc", "soft_emo"),
        ("crop8", "results_dense", "crop8"), ("seg8", "results_dense", "seg8"), ("dense16", "results_dense", "dense16"),
        ("norm rel", "results_norm_rel", "soft_emo"), ("norm z", "results_norm_z", "soft_emo"),
        ("gate (dual enc.)", "results_gate", "gate"), ("soft_gate", "results_softgate", "soft_gate"), ("xgate", "results_xgate", "xgate"),
        ("EVI fa_evi", "results_evi", "fa_evi"), ("au_soft (RoBERTa)", "results_roberta_ausoft", "soft"),
        ("au_soft_pre", "results_pre", "au_soft_pre"), ("diff2_uni (raw AU)", "results_diff2", "diff2_uni"), ("diff2_pre", "results_pre", "diff2_pre"),
        ("diff3_texp", "results_diff3", "diff3_texp"), ("diff3_unc", "results_diff3", "diff3_unc"), ("diff3_conf", "results_diff3", "diff3_conf"),
        ("diff3_frame", "results_diff3", "diff3_frame"), ("diff3_best", "results_diff3", "diff3_best"),
        ("[tuned experts] diff2_pre", "results_bert2", "diff2_pre"), ("[tuned experts] diff3_unc", "results_bert2", "diff3_unc"),
        ("[tuned experts] diff3_tri (audio expert)", "results_bert2", "diff3_tri"),
        ("[tuned experts] diff3_trust λ0.1", "results_bert2", "diff3_trust01"), ("[tuned experts] diff3_trust λ0.3", "results_bert2", "diff3_trust03"),
        ("[RoBERTa] text", "results_rob2", "text"), ("[RoBERTa] au_soft+emo", "results_rob2", "soft_emo"),
        ("[RoBERTa] diff2_pre", "results_rob2", "diff2_pre"), ("[RoBERTa] diff3_unc", "results_rob2", "diff3_unc"),
        ("diff3_sc (scalar DIFF, in-sample experts)", "results_oof", "diff3_sc"), ("diff3_oof (scalar DIFF, cross-fitted experts)", "results_oof", "diff3_oof"),
        ("diff3_noise (scalar, valid-matched noise)", "results_oof", "diff3_noise"), ("diff3_drop (scalar, DIFF dropout 0.3)", "results_oof", "diff3_drop"),
        ("diff3_reg (scalar, less-overfit text expert)", "results_oof", "diff3_reg"),
        ("capctl: visual zeroed (text expert only)", "results_oof", "diff3_vzero"), ("capctl: visual shuffled", "results_oof", "diff3_vshuf"),
        ("diff3_ext (external experts, no target unimodal labels)", "results_ext", "diff3_ext"),
        ("diff3_tri3 (cross-fitted text+face+HuBERT-voice experts)", "results_oof", "diff3_tri3"),
        ("diff3_tri3_trust (+ whom-to-trust head)", "results_oof", "diff3_tri3_trust")]
P("## 2. CH-SIMS v2 test — mean±std over seeds | seed-ensemble (averaged predictions)")
P("| method | seeds | subset | Acc-2 | F1 | MAE | Corr | ens Acc-2 | ens F1 | ens MAE | ens Corr | source |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
PRED, SEEDM = {}, {}
for name, d, v in SIMS:
    pr, fs = load(os.path.join(F, d), v)
    if not pr:
        P(f"| {name} | 0 | — | missing |  |  |  |  |  |  |  | {d}/{v} |"); continue
    PRED[name] = pr
    ens = pd.concat([p.pred for p in pr.values()], axis=1).mean(axis=1)
    for s, idx in SUB.items():
        per = [m_sims(te.label.loc[idx].values, p.pred.loc[idx].values) for p in pr.values()]
        SEEDM[(name, s)] = {sd: m_sims(te.label.loc[idx].values, p.pred.loc[idx].values) for sd, p in pr.items()}
        e = m_sims(te.label.loc[idx].values, ens.loc[idx].values)
        f = lambda k: f"{np.mean([x[k] for x in per]):.4f}±{np.std([x[k] for x in per], ddof=1) if len(per) > 1 else 0:.4f}"
        P(f"| {name} | {len(pr)} | {s} (n={len(idx)}) | {f('acc2')} | {f('f1')} | {f('mae')} | {f('corr')} | {e['acc2']:.4f} | {e['f1']:.4f} | {e['mae']:.4f} | {e['corr']:.4f} | {d}/{v}_seed*_test_pred.csv |")

# ------------------------------------------------------------------ 3. significance
def sig(a, b, idx):
    pa, pb = PRED[a], PRED[b]; seeds = sorted(set(pa) & set(pb)); y = te.label.loc[idx].values
    ea = pd.concat([pa[s].pred for s in pa], axis=1).mean(axis=1).loc[idx].values
    eb = pd.concat([pb[s].pred for s in pb], axis=1).mean(axis=1).loc[idx].values
    rng = np.random.default_rng(0); B = rng.integers(0, len(y), (2000, len(y))); out = []
    for k in ("acc2", "mae", "corr"):
        ta = [m_sims(y, pa[s].pred.loc[idx].values)[k] for s in seeds]; tb = [m_sims(y, pb[s].pred.loc[idx].values)[k] for s in seeds]
        tp = ttest_rel(ta, tb).pvalue
        if k == "acc2":
            fa = ((y[B] > 0) == (np.clip(ea, -1, 1)[B] > 0)).mean(1); fb = ((y[B] > 0) == (np.clip(eb, -1, 1)[B] > 0)).mean(1)
        elif k == "mae":
            fa = np.abs(y[B] - np.clip(ea, -1, 1)[B]).mean(1); fb = np.abs(y[B] - np.clip(eb, -1, 1)[B]).mean(1)
        else:
            def cc(p):
                yy, pp = y[B], np.clip(p, -1, 1)[B]; yc, pc = yy - yy.mean(1, keepdims=True), pp - pp.mean(1, keepdims=True)
                return (yc * pc).sum(1) / np.sqrt((yc ** 2).sum(1) * (pc ** 2).sum(1))
            fa, fb = cc(ea), cc(eb)
        dd = fa - fb; bp = min(1, max(2 * min((dd < 0).mean(), (dd > 0).mean()), 1 / 2000))
        out.append(f"{np.mean(ta) - np.mean(tb):+.4f} (t {tp:.3f}) / ens {m_sims(y, ea)[k] - m_sims(y, eb)[k]:+.4f} (boot {bp:.3f})")
    return out, len(seeds)


P("\n## 3. Significance (Δ seed-mean with per-seed paired t p / Δ ensemble with paired bootstrap p, 2000 resamples)")
P("| comparison | subset | seeds | ΔAcc-2 | ΔMAE | ΔCorr |")
P("|---|---|---|---|---|---|")
for a, b in [("diff3_unc", "au_soft+emo"), ("diff2_pre", "au_soft+emo"), ("diff3_conf", "au_soft+emo"), ("diff3_unc", "diff2_pre"),
             ("au_soft+emo", "text"), ("au_soft+emo", "deva_av (DEVA-style)"), ("au_soft+emo", "mag (MAG-style)"), ("au_soft+emo", "soft_clip (CLIP tokens)"),
             ("au_soft", "text"), ("au_soft+emo", "au_soft"),
             ("[tuned experts] diff3_trust λ0.1", "au_soft+emo"), ("[tuned experts] diff3_trust λ0.1", "[tuned experts] diff3_unc"),
             ("[tuned experts] diff3_trust λ0.3", "au_soft+emo"), ("[tuned experts] diff3_trust λ0.3", "[tuned experts] diff3_unc"),
             ("[tuned experts] diff3_tri (audio expert)", "[tuned experts] diff3_unc"), ("[tuned experts] diff3_unc", "au_soft+emo"),
             ("diff3_oof (scalar DIFF, cross-fitted experts)", "au_soft+emo"), ("diff3_oof (scalar DIFF, cross-fitted experts)", "diff3_unc"),
             ("diff3_oof (scalar DIFF, cross-fitted experts)", "diff3_sc (scalar DIFF, in-sample experts)"), ("diff3_sc (scalar DIFF, in-sample experts)", "au_soft+emo")] + \
            [(v, r) for v in ("diff3_noise (scalar, valid-matched noise)", "diff3_drop (scalar, DIFF dropout 0.3)", "diff3_reg (scalar, less-overfit text expert)")
             for r in ("au_soft+emo", "diff3_sc (scalar DIFF, in-sample experts)", "diff3_oof (scalar DIFF, cross-fitted experts)")] +             [(v, r) for v in ("capctl: visual zeroed (text expert only)", "capctl: visual shuffled") for r in ("au_soft+emo", "diff3_oof (scalar DIFF, cross-fitted experts)")] +             [("diff3_ext (external experts, no target unimodal labels)", r) for r in ("au_soft+emo", "diff3_oof (scalar DIFF, cross-fitted experts)", "diff3_unc")] +             [(v, r) for v in ("diff3_tri3 (cross-fitted text+face+HuBERT-voice experts)", "diff3_tri3_trust (+ whom-to-trust head)")
             for r in ("au_soft+emo", "diff3_oof (scalar DIFF, cross-fitted experts)", "diff3_unc")]:
    if a in PRED and b in PRED:
        for s in ("full", "conflict", "align", "tv_opposite"):
            o, n = sig(a, b, SUB[s]); P(f"| {a} vs {b} | {s} | {n} | " + " | ".join(o) + " |")

# ------------------------------------------------------------------ 4. diagnostics
P("\n## 4. Diagnostics")
P("Shuffle Δ (needs the trained model) is taken from the run JSONs (`diag`), mean over seeds; AUROC is RECOMPUTED from the per-clip `gap` column.")
P("| method | seeds | shuffle-DIFF ΔCorr full | ΔCorr Conflict | AUROC gap→sign(T)≠sign(V) (3-way sign, n pos) | AUROC gap→strict T/V opposite (n pos) | source |")
P("|---|---|---|---|---|---|---|")
yT, yV = te.label_T, te.label_V
for name, d, v in [("diff2_uni (raw AU)", "results_diff2", "diff2_uni"), ("diff2_pre", "results_pre", "diff2_pre"), ("diff3_texp", "results_diff3", "diff3_texp"),
                   ("diff3_unc", "results_diff3", "diff3_unc"), ("diff3_conf", "results_diff3", "diff3_conf"), ("diff3_best", "results_diff3", "diff3_best")]:
    js = [json.load(open(p)) for p in glob.glob(os.path.join(F, d, f"{v}_seed*.json"))]
    js = [j for j in js if "diag" in j]
    pr, _ = load(os.path.join(F, d), v)
    sd = lambda k: np.mean([j["diag"][k]["corr"] for j in js if k in j["diag"]]) if js and any(k in j["diag"] for j in js) else np.nan
    au3, au_s = [], []
    for p in pr.values():
        if "gap" not in p.columns:
            continue
        g = p.gap.loc[te.index].values
        au3.append(roc_auc_score((sg(yT) != sg(yV)).astype(int).values, g)); au_s.append(roc_auc_score(((sg(yT) * sg(yV)) < 0).astype(int).values, g))
    sk = "shuffle_diff_delta_full" if any("shuffle_diff_delta_full" in j["diag"] for j in js) else "shuffle_delta_full"
    skc = sk.replace("full", "conflict")
    P(f"| {name} | {len(js)} | {sd(sk):+.4f} | {sd(skc):+.4f} | {np.mean(au3) if au3 else np.nan:.3f} (n={int((sg(yT) != sg(yV)).sum())}) | "
      f"{np.mean(au_s) if au_s else np.nan:.3f} (n={len(SUB['tv_opposite'])}) | {d}/{v}_seed*.json, *_test_pred.csv |")
ip = os.path.join(F, "interp3_frames.csv")
if os.path.exists(ip):
    q = pd.read_csv(ip)
    nv, nt = q.label_V != 0, q.label_T != 0
    P(f"- Expert polarity on test Conflict clips (diff3_unc seed-1 model, interp3_frames.csv): visual vs label_V excluding label_V=0: "
      f"{(sg(q.yv[nv]) == sg(q.label_V[nv])).mean():.3f} (n={int(nv.sum())}); including neutral (sign 0 never matches): {(sg(q.yv) == sg(q.label_V)).mean():.3f} (n={len(q)}). "
      f"Text vs label_T excluding 0: {(sg(q.yt[nt]) == sg(q.label_T[nt])).mean():.3f} (n={int(nt.sum())}).")

# ------------------------------------------------------------------ 4b. over-trust indicator
EO = os.path.join(F, "expert_oof")
if glob.glob(os.path.join(EO, "seed*.csv")):
    P("\n## 4b. Over-trust indicator on test Conflict clips: Corr(main-model prediction, text-expert reading) vs Corr(prediction, visual-expert reading)")
    P("(per seed, readings of the same-seed full-train experts — reg variant uses its own less-overfit text expert — then averaged; label Corr for reference)")
    P("| method | seeds | Corr(pred, y_t) | Corr(pred, y_v) | Corr(pred, label) |")
    P("|---|---|---|---|---|")
    ci = SUB["conflict"]
    for name in ["au_soft+emo", "diff3_unc", "diff3_sc (scalar DIFF, in-sample experts)", "diff3_oof (scalar DIFF, cross-fitted experts)",
                 "diff3_noise (scalar, valid-matched noise)", "diff3_drop (scalar, DIFF dropout 0.3)", "diff3_reg (scalar, less-overfit text expert)"]:
        if name not in PRED:
            continue
        reg = name.startswith("diff3_reg"); ct, cv, cl = [], [], []
        for sd, p in PRED[name].items():
            fpath = os.path.join(EO, f"seed{sd}{'_reg' if reg else ''}.csv")
            if not os.path.exists(fpath):
                continue
            rd = pd.read_csv(fpath).set_index("key").loc[ci]; q = p.pred.loc[ci]
            ct.append(pearsonr(q, rd["yt_reg" if reg else "yt_in"])[0]); cv.append(pearsonr(q, rd["yv_reg" if reg else "yv_in"])[0]); cl.append(pearsonr(q, te.label.loc[ci])[0])
        if ct:
            P(f"| {name} | {len(ct)} | {np.mean(ct):.3f} | {np.mean(cv):.3f} | {np.mean(cl):.3f} |")
# ------------------------------------------------------------------ 5. MOSI / MOSEI
for DSN, DD in (("CMU-MOSI", M), ("CMU-MOSEI", os.path.join(R, "mosei"))):
    if not os.path.exists(os.path.join(DD, "clips.csv")):
        continue
    mc2 = pd.read_csv(os.path.join(DD, "clips.csv")); mte = mc2[mc2.split == "test"].set_index("key")
    P(f"\n## 5. {DSN} test (n={len(mte)}) — mean±std over seeds | ensemble")
    P("| method | seeds | Acc-2 has-0 | Acc-2 non-0 | F1 has-0 | Acc-7 | MAE | Corr | ens Acc-2 has-0 | ens Acc-2 non-0 | ens MAE | ens Corr | source |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    MP = {}
    for name, d, v in [("text", "results", "text"), ("deva_av (DEVA-style)", "results", "deva_av"), ("mag (MAG-style)", "results_ausoft", "mag"),
                       ("au_soft", "results_ausoft", "soft"), ("au_soft+emo", "results_acc", "soft_emo"), ("au_soft_pre (SIMS expert)", "results_pre", "au_soft_pre"),
                       ("diff2_pre (SIMS expert)", "results_pre", "diff2_pre"), ("diff3_ext (external experts)", "results_ext", "diff3_ext"),
                       ("ext capctl: visual zeroed", "results_ext", "ext_vzero"), ("ext capctl: visual shuffled", "results_ext", "ext_vshuf"),
                       ("ext text-only (text + SST-2 expert, no AU)", "results_ext", "ext_textonly")]:
        pr, _ = load(os.path.join(DD, d), v)
        if not pr:
            continue
        MP[name] = pr; y = mte.label.values
        per = [m_mosi(y, p.pred.loc[mte.index].values) for p in pr.values()]
        e = m_mosi(y, pd.concat([p.pred for p in pr.values()], axis=1).mean(axis=1).loc[mte.index].values)
        f = lambda k: f"{np.mean([x[k] for x in per]):.4f}±{np.std([x[k] for x in per], ddof=1) if len(per) > 1 else 0:.4f}"
        P(f"| {name} | {len(pr)} | {f('acc2')} | {f('acc2_non0')} | {f('f1')} | {f('acc7')} | {f('mae')} | {f('corr')} | {e['acc2']:.4f} | {e['acc2_non0']:.4f} | {e['mae']:.4f} | {e['corr']:.4f} | {os.path.basename(DD)}/{d}/{v} |")
    for a_, b_ in (("au_soft+emo", "text"), ("au_soft+emo", "deva_av (DEVA-style)"), ("au_soft+emo", "mag (MAG-style)"), ("au_soft", "text"), ("diff2_pre (SIMS expert)", "au_soft+emo"), ("diff3_ext (external experts)", "au_soft+emo"), ("diff3_ext (external experts)", "text"),
                   ("ext capctl: visual zeroed", "diff3_ext (external experts)"), ("ext capctl: visual shuffled", "diff3_ext (external experts)"),
                   ("ext text-only (text + SST-2 expert, no AU)", "diff3_ext (external experts)"), ("ext text-only (text + SST-2 expert, no AU)", "text"),
                   ("ext capctl: visual zeroed", "au_soft+emo"), ("ext text-only (text + SST-2 expert, no AU)", "au_soft+emo")):
        if a_ in MP and b_ in MP:
            a, b = MP[a_], MP[b_]; seeds = sorted(set(a) & set(b)); y = mte.label.values; cells = []
            for k in ("acc2", "acc2_non0", "acc7", "mae", "corr"):
                ta = [m_mosi(y, a[s].pred.loc[mte.index].values)[k] for s in seeds]; tb = [m_mosi(y, b[s].pred.loc[mte.index].values)[k] for s in seeds]
                cells.append(f"{k} {np.mean(ta) - np.mean(tb):+.4f} (p={ttest_rel(ta, tb).pvalue:.3f})")
            P(f"- {DSN} {a_} vs {b_} ({len(seeds)} seeds, paired t): " + "; ".join(cells))
# ------------------------------------------------------------------ 6. expert tuning
et = os.path.join(F, "expert_tuning.json")
if os.path.exists(et):
    E = json.load(open(et)); ba = E["before_after_seed1"]
    P("\n## 6. Expert tuning (CH-SIMS v2; selection on VALID only; polarity accuracy excludes |label| <= 0.1)")
    for k, nm in (("visual", "visual"), ("text", "text")):
        dflt, sel = ba[f"{k}_default"], ba[f"{k}_selected"]
        P(f"- {nm}: default {E['default_' + k]} -> selected {E[k + '_best']}; seed-1 valid MAE {dflt['valid_mae']:.4f} -> {sel['valid_mae']:.4f}; "
          f"valid-Conflict polarity {dflt['valid_conflict_polacc']:.3f} -> {sel['valid_conflict_polacc']:.3f}; TEST-Conflict polarity {dflt['test_conflict_polacc']:.3f} -> {sel['test_conflict_polacc']:.3f}")
    if "visual_tuned_10seeds" in E:
        old_v = [json.load(open(p)) for p in glob.glob(os.path.join(F, "results_evq", "raw8emo_seed*.json"))]
        old_t = [json.load(open(p)) for p in glob.glob(os.path.join(F, "results_texp", "seed*.json"))]
        P(f"- tuned experts, 10 seeds: visual test label_V Corr {E['visual_tuned_10seeds']['test_labelV_corr']:.4f} (untuned, {len(old_v)} seeds: "
          f"{np.mean([r['test_labelV']['corr'] for r in old_v]):.4f}); test-Conflict polarity {E['visual_tuned_10seeds']['test_conflict_polacc']:.3f}. "
          f"Text test label_T Corr {E['text_tuned_10seeds']['test_labelT_corr']:.4f} (untuned, {len(old_t)} seeds: {np.mean([r['test_labelT']['corr'] for r in old_t]):.4f}); "
          f"test-Conflict polarity {E['text_tuned_10seeds']['test_conflict_polacc']:.3f}")
# ------------------------------------------------------------------ 7. ceiling analysis (CPU)
from sklearn.linear_model import LinearRegression as _LR
from sklearn.model_selection import KFold as _KF
P("\n## 7. Oracle-reference analysis (CH-SIMS v2; gold unimodal labels; an observed reference under our data and settings, not a theoretical bound)")
ca = c.copy(); ca["strict"] = (sg(ca.label_T) * sg(ca.label_V)) < 0
P(f"- Strict T–V opposite clips per split: {ca[ca.strict].split.value_counts().to_dict()}; total {int(ca.strict.sum())} of {len(ca)}.")
rng7 = np.random.default_rng(0)


def acc2(y, p):
    return float(((y > 0) == (np.clip(p, -1, 1) > 0)).mean())


def bci(y, p, n=2000):
    ix = rng7.integers(0, len(y), (n, len(y))); a = ((y[ix] > 0) == (np.clip(p, -1, 1)[ix] > 0)).mean(1)
    return np.percentile(a, [2.5, 97.5])


P("| oracle features (5-fold CV over ALL 4403 clips, 20 repeats) | strict clips (all splits) Acc-2 | 95% CI (bootstrap, repeat 0) | Conflict (all splits) Acc-2 | all clips Acc-2 |")
P("|---|---|---|---|---|")
for feats in (["label_T", "label_V"], ["label_T", "label_V", "label_A"]):
    accs, accc, acca, p0 = [], [], [], None
    for r in range(20):
        p = np.zeros(len(ca))
        for a_, b_ in _KF(5, shuffle=True, random_state=r).split(ca):
            p[b_] = _LR().fit(ca.iloc[a_][feats], ca.iloc[a_].label).predict(ca.iloc[b_][feats])
        y = ca.label.values; s = ca.strict.values; cf = (ca.category == "conflict").values
        accs.append(acc2(y[s], p[s])); accc.append(acc2(y[cf], p[cf])); acca.append(acc2(y, p))
        p0 = p if r == 0 else p0
    lo, hi = bci(ca.label.values[ca.strict.values], p0[ca.strict.values])
    P(f"| true {'+'.join(f[-1] for f in feats)} | {np.mean(accs):.3f} (n={int(ca.strict.sum())}) | [{lo:.3f}, {hi:.3f}] | {np.mean(accc):.3f} | {np.mean(acca):.3f} |")
yS = te.label.loc[SUB["tv_opposite"]].values
P("\n| test strict (n=93) | per-seed mean Acc-2 | ensemble Acc-2 | ensemble 95% CI (bootstrap over clips) |")
P("|---|---|---|---|")
for nm in ("text", "au_soft+emo", "diff3_unc", "diff3_oof (scalar DIFF, cross-fitted experts)"):
    if nm in PRED:
        pr = PRED[nm]; ens = pd.concat([p.pred for p in pr.values()], axis=1).mean(axis=1).loc[SUB["tv_opposite"]].values
        per = np.mean([acc2(yS, p.pred.loc[SUB["tv_opposite"]].values) for p in pr.values()]); lo, hi = bci(yS, ens)
        P(f"| {nm} | {per:.3f} | {acc2(yS, ens):.3f} | [{lo:.3f}, {hi:.3f}] |")
if "text" in PRED and "au_soft+emo" in PRED:
    ea = pd.concat([p.pred for p in PRED["au_soft+emo"].values()], axis=1).mean(axis=1).loc[SUB["tv_opposite"]].values
    et = pd.concat([p.pred for p in PRED["text"].values()], axis=1).mean(axis=1).loc[SUB["tv_opposite"]].values
    ix = rng7.integers(0, len(yS), (2000, len(yS)))
    d = ((yS[ix] > 0) == (np.clip(ea, -1, 1)[ix] > 0)).mean(1) - ((yS[ix] > 0) == (np.clip(et, -1, 1)[ix] > 0)).mean(1)
    P(f"- au_soft+emo − text on the 93 strict clips (ensembles): {acc2(yS, ea) - acc2(yS, et):+.3f}, 95% CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
# accuracy by conflict strength |label_T - label_V| on the whole test set
gapv = (te.label_T - te.label_V).abs()
bins = [(-0.01, 0.0, "0"), (0.0, 0.4, "(0,0.4]"), (0.4, 0.8, "(0.4,0.8]"), (0.8, 1.2, "(0.8,1.2]"), (1.2, 2.1, ">1.2")]
orc = _LR().fit(c[c.split == "train"][["label_T", "label_V"]], c[c.split == "train"].label).predict(te[["label_T", "label_V"]])
orc = pd.Series(orc, index=te.index)
P("\n| |label_T − label_V| (test) | n | text | au_soft+emo | diff3_unc | oracle true T+V (fit on train) |")
P("|---|---|---|---|---|---|")
for lo_, hi_, nm in bins:
    ix = te.index[(gapv > lo_) & (gapv <= hi_)]
    if len(ix) == 0:
        continue
    y = te.label.loc[ix].values
    cell = lambda m: f"{np.mean([acc2(y, p.pred.loc[ix].values) for p in PRED[m].values()]):.3f}" if m in PRED else "—"
    P(f"| {nm} | {len(ix)} | {cell('text')} | {cell('au_soft+emo')} | {cell('diff3_unc')} | {acc2(y, orc.loc[ix].values):.3f} |")
# ------------------------------------------------------------------ 8. 5-fold CV over all of SIMS v2 (supplementary)
CVD = os.path.join(R, "full_cv")
if os.path.isdir(CVD):
    full = c.copy(); full["strict"] = (sg(full.label_T) * sg(full.label_V)) < 0
    cvp = {}
    for nm, d, v in (("text", "results", "text"), ("au_soft+emo", "results_acc", "soft_emo")):
        per_seed = {}
        for s in range(1, 6):
            parts = [pd.read_csv(p).set_index("key") for p in glob.glob(os.path.join(CVD, "fold*", d, f"{v}_seed{s}_test_pred.csv"))]
            if len(parts) == 5:
                per_seed[s] = pd.concat(parts).pred.reindex(full.index)
        if per_seed:
            cvp[nm] = per_seed
    if cvp:
        P("\n## 8. Supplementary: 5-fold cross-validation over all 4403 CH-SIMS v2 clips (folds grouped by video; default recipe)")
        P("| method | seeds | subset | per-seed mean Acc-2 | ensemble Acc-2 | ensemble 95% CI |")
        P("|---|---|---|---|---|---|")
        subs8 = {"all (n=%d)" % len(full): full.index, "Conflict (n=%d)" % int((full.category == "conflict").sum()): full.index[full.category == "conflict"],
                 "strict T–V (n=%d)" % int(full.strict.sum()): full.index[full.strict]}
        for nm, ps in cvp.items():
            ens = pd.concat(ps.values(), axis=1).mean(axis=1)
            for sn, ix in subs8.items():
                y = full.label.loc[ix].values; per = np.mean([acc2(y, p.loc[ix].values) for p in ps.values()]); lo, hi = bci(y, ens.loc[ix].values)
                P(f"| {nm} | {len(ps)} | {sn} | {per:.3f} | {acc2(y, ens.loc[ix].values):.3f} | [{lo:.3f}, {hi:.3f}] |")
        if len(cvp) == 2:
            ea = pd.concat(cvp["au_soft+emo"].values(), axis=1).mean(axis=1); et = pd.concat(cvp["text"].values(), axis=1).mean(axis=1)
            for sn, ix in subs8.items():
                y = full.label.loc[ix].values; ix2 = rng7.integers(0, len(y), (2000, len(y)))
                a_, t_ = np.clip(ea.loc[ix].values, -1, 1), np.clip(et.loc[ix].values, -1, 1)
                dd = ((y[ix2] > 0) == (a_[ix2] > 0)).mean(1) - ((y[ix2] > 0) == (t_[ix2] > 0)).mean(1)
                P(f"- au_soft+emo − text, {sn}: {acc2(y, a_) - acc2(y, t_):+.3f}, paired bootstrap 95% CI [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]")
# ------------------------------------------------------------------ 9. evaluation details on the tuned 10-seed runs (CPU)
from scipy.stats import binomtest as _binomtest, t as _tdist
TUN = os.path.join(F, "tuned")
if os.path.isdir(TUN):
    def _tp(m):
        return {int(re.search(r"seed(\d+)_test", p).group(1)): pd.read_csv(p).set_index("key").pred for p in glob.glob(os.path.join(TUN, m, "*_seed*_test_pred.csv"))}

    def _acc3(y, p):  # MMSA SIMS 3-class: [-1,-0.1] / (-0.1,0.1] / (0.1,1]
        cl = lambda v: np.where(v <= -0.1, 0, np.where(v <= 0.1, 1, 2))
        return float((cl(y) == cl(np.clip(p, -1, 1))).mean())

    TM = {m: _tp(m) for m in ("text", "au_soft_emo", "deva_av", "deva_dense", "diff3_unc", "diff3_oof", "diff3_tri3_trust")}
    TM = {m: v for m, v in TM.items() if v}
    SUB9 = {"all": te.index, "conflict": te.index[te.category == "conflict"], "strict": SUB["tv_opposite"]}
    P("\n## 9. Evaluation details (tuned configuration, 10 seeds; official test split)")
    P("Acc-2 convention: label > 0 positive, label <= 0 negative (label 0 counted as negative). 'non-0 Acc-2' excludes label-0 clips. "
      "Acc-3 uses MMSA thresholds (-0.1, 0.1]. Equivalence margin for 'no gain' claims: ±2 Acc-2 points (TOST, alpha 0.05), fixed before running.")
    P("| method | subset | n (all in Acc-2) | n non-0 | Acc-2 | non-0 Acc-2 | Acc-3 |")
    P("|---|---|---|---|---|---|---|")
    for m, pr in TM.items():
        for sn, ix in SUB9.items():
            y = te.label.loc[ix].values; nz = y != 0
            a2 = np.mean([acc2(y, p.loc[ix].values) for p in pr.values()])
            a2n = np.mean([acc2(y[nz], p.loc[ix].values[nz]) for p in pr.values()])
            a3 = np.mean([_acc3(y, p.loc[ix].values) for p in pr.values()])
            P(f"| {m} ({len(pr)} seeds) | {sn} | {len(ix)} | {int(nz.sum())} | {a2:.4f} | {a2n:.4f} | {a3:.4f} |")
    P("\n| comparison (A vs B) | subset | per-seed mean counts: W→R / R→W / both R / both W | ensemble W→R / R→W | McNemar exact p (ensemble) | ΔAcc-2 (seed mean) | 90% CI | TOST p (±2 pts) |")
    P("|---|---|---|---|---|---|---|---|")
    for a_, b_ in (("au_soft_emo", "text"), ("au_soft_emo", "deva_av"), ("au_soft_emo", "deva_dense"), ("diff3_unc", "au_soft_emo"),
                   ("diff3_tri3_trust", "au_soft_emo"), ("diff3_tri3_trust", "diff3_oof"), ("diff3_tri3_trust", "diff3_unc")):
        if a_ not in TM or b_ not in TM:
            continue
        seeds = sorted(set(TM[a_]) & set(TM[b_]))
        for sn, ix in SUB9.items():
            y = te.label.loc[ix].values > 0; cnt = []; dif = []
            for s in seeds:
                ra = (np.clip(TM[a_][s].loc[ix].values, -1, 1) > 0) == y; rb = (np.clip(TM[b_][s].loc[ix].values, -1, 1) > 0) == y
                cnt.append([np.sum(~rb & ra), np.sum(rb & ~ra), np.sum(ra & rb), np.sum(~ra & ~rb)]); dif.append(ra.mean() - rb.mean())
            cnt = np.mean(cnt, 0)
            ea = pd.concat([TM[a_][s] for s in seeds], axis=1).mean(axis=1).loc[ix].values; eb = pd.concat([TM[b_][s] for s in seeds], axis=1).mean(axis=1).loc[ix].values
            ra = (np.clip(ea, -1, 1) > 0) == y; rb = (np.clip(eb, -1, 1) > 0) == y; w2r, r2w = int(np.sum(~rb & ra)), int(np.sum(rb & ~ra))
            mc = _binomtest(w2r, w2r + r2w, 0.5).pvalue if w2r + r2w else 1.0
            dif = np.array(dif); mu, se, k = dif.mean(), dif.std(ddof=1) / np.sqrt(len(dif)), len(dif) - 1
            lo90, hi90 = mu - _tdist.ppf(0.95, k) * se, mu + _tdist.ppf(0.95, k) * se
            p_low = 1 - _tdist.cdf((mu + 0.02) / se, k) if se > 0 else 0.0; p_up = _tdist.cdf((mu - 0.02) / se, k) if se > 0 else 0.0
            P(f"| {a_} vs {b_} | {sn} (n={len(ix)}) | {cnt[0]:.1f} / {cnt[1]:.1f} / {cnt[2]:.1f} / {cnt[3]:.1f} | {w2r} / {r2w} | {mc:.3f} | "
              f"{mu:+.4f} | [{lo90:+.4f}, {hi90:+.4f}] | {max(p_low, p_up):.3f} |")
    P("- TOST p < 0.05 means the Acc-2 difference is shown to lie within ±2 points (equivalence); otherwise neither a gain nor equivalence is established.")
# ------------------------------------------------------------------ 10. polarity-head round (tuned config + polarity head, 10 seeds)
TPOL = os.path.join(F, "tuned_pol")
if os.path.isdir(TPOL) and os.path.isdir(TUN):
    def _tpp(m):
        out = {}
        for p in glob.glob(os.path.join(TPOL, m, "*_seed*_test_pred.csv")):
            d = pd.read_csv(p).set_index("key")
            if "plogit" in d:
                out[int(re.search(r"seed(\d+)_test", p).group(1))] = d
        return out
    lamf = os.path.join(F, "pol_lambda.json")
    P("\n## 10. Polarity head (each method's tuned configuration + BCE polarity head; lambda chosen on VALID polarity Acc-2; 10 seeds)")
    if os.path.exists(lamf):
        P(f"- chosen lambda: {json.load(open(lamf))}")
    P("Acc-2(pol) = decided by the polarity probability; Acc-2(reg) = same runs, decided by the regression output; no-head = the tuned 10-seed runs of Section 9.")
    P("| method | subset | n / n non-0 | Acc-2 no-head | Acc-2(reg) | Acc-2(pol) | non-0 Acc-2 no-head → pol | Acc-3 no-head → pol(reg) | pol − no-head: W→R / R→W (ensemble), McNemar p | Δ (seed mean), paired t p | TOST p (±2) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in ("text", "au_soft_emo", "deva_dense", "diff3_tri3_trust"):
        pp, base = _tpp(m), TM.get(m) if m in TM else _tp(m)
        if not pp or not base:
            continue
        seeds = sorted(set(pp) & set(base))
        for sn, ix in SUB9.items():
            y = te.label.loc[ix].values; nz = y != 0; yb = y > 0
            accb = [acc2(y, base[s].loc[ix].values) for s in seeds]
            accr = [acc2(y, pp[s].pred.loc[ix].values) for s in seeds]
            accp = [float(((pp[s].plogit.loc[ix].values > 0) == yb).mean()) for s in seeds]
            nzb = np.mean([acc2(y[nz], base[s].loc[ix].values[nz]) for s in seeds]); nzp = np.mean([float(((pp[s].plogit.loc[ix].values[nz] > 0) == yb[nz]).mean()) for s in seeds])
            a3b = np.mean([_acc3(y, base[s].loc[ix].values) for s in seeds]); a3p = np.mean([_acc3(y, pp[s].pred.loc[ix].values) for s in seeds])
            eb = pd.concat([base[s] for s in seeds], axis=1).mean(axis=1).loc[ix].values
            ep = pd.concat([pp[s].plogit for s in seeds], axis=1).mean(axis=1).loc[ix].values
            rb, rp = (np.clip(eb, -1, 1) > 0) == yb, (ep > 0) == yb; w2r, r2w = int(np.sum(~rb & rp)), int(np.sum(rb & ~rp))
            mc = _binomtest(w2r, w2r + r2w, 0.5).pvalue if w2r + r2w else 1.0
            dif = np.array(accp) - np.array(accb); mu, se, k = dif.mean(), dif.std(ddof=1) / np.sqrt(len(dif)), len(dif) - 1
            tp = ttest_rel(accp, accb).pvalue
            p_low = 1 - _tdist.cdf((mu + 0.02) / se, k) if se > 0 else 0.0; p_up = _tdist.cdf((mu - 0.02) / se, k) if se > 0 else 0.0
            P(f"| {m} ({len(seeds)}) | {sn} | {len(ix)} / {int(nz.sum())} | {np.mean(accb):.4f} | {np.mean(accr):.4f} | {np.mean(accp):.4f} | {nzb:.4f} → {nzp:.4f} | "
              f"{a3b:.4f} → {a3p:.4f} | {w2r} / {r2w}, {mc:.3f} | {mu:+.4f}, {tp:.3f} | {max(p_low, p_up):.3f} |")
    P("- Fairness: every method gets the same head, the same lambda grid and the same selection rule; the main comparison table keeps the regression read-out.")
txt = "\n".join(L)
open(os.path.join(F, "AUDIT.md"), "w").write(txt)
print(txt)
