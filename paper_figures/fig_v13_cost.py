"""Figure 4 (v13, 2026-11-17): Acc-2 versus training time per run on CH-SIMS v2 test.
BERT: tuned configurations, 10 seeds (tuned/{text,au_soft_emo}: per-seed predictions and json 'sec').
LLM: Qwen3-VL-4B + LoRA, seeds 1-5 (llm_pilot/{text,au,vid}_seed*: predictions and json 'sec').
-> figures/v13/fig4_cost.{pdf,png}"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
F = P.parent / "results/outputs/full"
OUT = P / "figures/v13"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, RULE, GRID, TEAL, SAND_D = "#22262a", "#5c6268", "#c9c6bf", "#ecebe7", "#1f6f78", "#8c7a5f"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.edgecolor": RULE, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
acc = lambda df, col: float((((df.label > 0) == (df[col] > 0)).mean()) * 100)


def bert(d):
    pts = []
    for f in sorted(glob.glob(str(F / f"tuned/{d}/*_seed*.json"))):
        pr = pd.read_csv(f.replace(".json", "_test_pred.csv"))
        pts.append((json.load(open(f))["sec"] / 60, acc(pr, "pred")))
    return np.array(pts)


def llm(v):
    pts = []
    for s in range(1, 6):
        pr = pd.read_csv(F / f"llm_pilot/{v}_seed{s}_test_pred.csv")
        pts.append((json.load(open(F / f"llm_pilot/{v}_seed{s}.json"))["sec"] / 60, acc(pr, "pred_full")))
    return np.array(pts)


S = {"BERT, text": (bert("text"), SAND_D, "o"), "BERT + facial tokens": (bert("au_soft_emo"), TEAL, "o"),
     "LLM, text": (llm("text"), SAND_D, "s"), "LLM + facial tokens": (llm("au"), TEAL, "s"),
     "LLM watching 8 frames": (llm("vid"), INK2, "D")}
fig, ax = plt.subplots(figsize=(4.4, 2.6))
fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.17)
M = {}
for name, (pts, col, mk) in S.items():
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=col, alpha=0.35, lw=0, marker=mk, zorder=2)
    m = pts.mean(0); M[name] = m
    ax.scatter([m[0]], [m[1]], s=38, facecolor="white", edgecolor=col, lw=1.4, marker=mk, zorder=4)
    print(f"{name:24s} time {m[0]:6.1f} min  Acc-2 {m[1]:.2f}  (n={len(pts)})")
for a, b in (("BERT, text", "BERT + facial tokens"), ("LLM, text", "LLM + facial tokens")):
    ax.annotate("", xy=M[b], xytext=M[a], arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=0.9, shrinkA=5, shrinkB=5, mutation_scale=7))
off = {"BERT, text": (6, -3, "left"), "BERT + facial tokens": (-6, 4, "right"), "LLM, text": (7, -3, "left"),
       "LLM + facial tokens": (7, -9, "left"), "LLM watching 8 frames": (-7, 5, "right")}
for name, m in M.items():
    dx, dy, ha = off[name]
    ax.annotate(f"{name}\n{m[1]:.1f}", m, xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=7,
                color=S[name][1] if S[name][1] != SAND_D else INK, linespacing=1.1)
ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
ax.set_xlim(0.8, 160); ax.set_ylim(74.5, 88.5)
ax.set_xlabel("Training time per run (minutes, log scale)"); ax.set_ylabel("Acc-2 (%)")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig4_cost.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig4_cost.pdf")
