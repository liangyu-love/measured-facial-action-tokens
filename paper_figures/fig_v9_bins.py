"""Figure 2 (v9, 2026-11-11): gain of soft tokens + emotions / EMFACS over text-only BERT by text-visual label gap, single panel.
The representation panel and the cross-validation forest plot of v8 were dropped because they repeat Tables 1 and 5 of the main text.
Data: audit/conflict_bins.json (per-seed Acc-2 per bin, default recipe). -> figures/v9/conflict_bins.{pdf,png}"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parents[1]
OUT = P / "figures/v9"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, RULE, GRID, TEAL = "#22262a", "#5c6268", "#c9c6bf", "#ecebe7", "#1f6f78"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.8,
                     "axes.edgecolor": RULE, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK, "xtick.major.width": 0.6,
                     "ytick.major.width": 0, "xtick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic"})

bins = json.loads((P / "audit/conflict_bins.json").read_text())
labels = ["0", "(0, 0.4]", "(0.4, 0.8]", "(0.8, 1.2]", "> 1.2"]
fig, ax = plt.subplots(figsize=(4.2, 2.1))
fig.subplots_adjust(left=0.30, right=0.88, top=0.97, bottom=0.20)
rng = np.random.default_rng(1)
for i, b in enumerate(bins):
    g = np.array(b["soft"]) - np.array(b["text"])
    ax.scatter(g, i + rng.uniform(-0.13, 0.13, len(g)), s=9, color=TEAL, alpha=0.45, lw=0, zorder=2)
    ax.errorbar(g.mean(), i, xerr=g.std(ddof=1), fmt="o", ms=4.8, color=TEAL, mec="white", mew=0.6, elinewidth=1.1, zorder=3)
    ax.text(1.03, i, f"+{g.mean():.1f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.8, color=INK)
ax.set_yticks(range(5)); ax.set_yticklabels([f"{l}   (n = {b['n']})" for l, b in zip(labels, bins)]); ax.set_ylim(4.6, -0.6)
ax.set_xlim(0, 34); ax.set_xticks([0, 10, 20, 30]); ax.set_xlabel("Acc-2 gain over text-only BERT (points)")
ax.set_ylabel(r"Label gap $|y_T-y_V|$", color=INK)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"conflict_bins.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "conflict_bins.pdf")
