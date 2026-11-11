"""Figure 2 (v3, 2026-11-11): results, drawn from scratch from archived per-seed predictions.
Top left: all-clip Acc-2 of the representations (default recipe, 5 seeds). Top right: per-seed paired gain over text by |y_T - y_V|.
Bottom: 5-fold cross-validation, ensemble gain of the soft tokens over each baseline with paired clip-bootstrap 95% CI.
Output: figures/v8/conflict_bins.{pdf,png}"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
F = P.parent / "results/outputs/full"
OUT = P / "figures/v8"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, RULE, GRID = "#22262a", "#5c6268", "#9aa0a6", "#c9c6bf", "#ecebe7"
BLUE, ORANGE, AQUA, GREY = "#1f6f78", "#1f6f78", "#1f6f78", "#a7abaf"  # v8: one teal accent for our method, grey for baselines
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7.2,
                     "axes.edgecolor": RULE, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK, "xtick.major.width": 0.6,
                     "ytick.major.width": 0, "xtick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic"})


def acc(folder, name):
    out = []
    for s in range(1, 6):
        d = pd.read_csv(F / folder / f"{name}_seed{s}_test_pred.csv")
        out.append(float(((d.label > 0) == (np.clip(d.pred, -1, 1) > 0)).mean() * 100))
    return out


rep = [("Text only", "results", "text"), ("MLLM description", "results", "vlm"), ("AU late fusion", "results_ausoft", "late"),
       ("CLIP image tokens", "results_ausoft", "soft_clip"), ("AU text template", "results", "tmpl_au"),
       ("MAG-style gating", "results_ausoft", "mag"), ("DEVA-style template", "results", "deva_av"),
       ("AU soft tokens", "results_ausoft", "soft"), ("+ emotions / EMFACS", "results_acc", "soft_emo")]
vals = [acc(f, n) for _, f, n in rep]
bins = json.loads((P / "audit/conflict_bins.json").read_text())
cv = json.loads((F / "cv_compare.json").read_text())

fig = plt.figure(figsize=(6.77, 4.45))
gs = fig.add_gridspec(2, 2, left=0.165, right=0.94, top=0.95, bottom=0.085, wspace=0.70, hspace=0.62, height_ratios=[1.0, 0.82])
a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])


def head(ax, s, x=-0.02):
    ax.text(x, 1.05, s, transform=ax.transAxes, fontsize=7.6, fontweight="bold", color=INK, ha="left", va="bottom")


# --- top left: representations
rng = np.random.default_rng(1)
for i, ((lab, _, _), v) in enumerate(zip(rep, vals)):
    ours = i >= 7
    col = BLUE if ours else GREY
    a1.scatter(v, i + rng.uniform(-0.13, 0.13, len(v)), s=7, color=col, alpha=0.45, lw=0, zorder=2)
    a1.errorbar(np.mean(v), i, xerr=np.std(v, ddof=1), fmt="o", ms=4.2, color=col, mec="white", mew=0.6, elinewidth=1.0, capsize=0, zorder=3)
    a1.text(1.03, i, f"{np.mean(v):.1f}", transform=a1.get_yaxis_transform(), va="center", ha="left", fontsize=7, color=INK if ours else INK2,
            fontweight="bold" if ours else "normal")
a1.axvline(np.mean(vals[0]), color=RULE, lw=0.6, ls=(0, (2, 2)), zorder=1)
a1.set_yticks(range(len(rep))); a1.set_yticklabels([r[0] for r in rep]); a1.set_ylim(len(rep) - 0.4, -0.6)
a1.set_xlim(74, 86.5); a1.set_xticks([75, 80, 85]); a1.set_xlabel("Acc-2 on all 1034 test clips (%)")
a1.grid(axis="x", color=GRID, lw=0.6); a1.set_axisbelow(True)
head(a1, "Representation of the facial information", x=-0.62)

# --- top right: gain by conflict strength (paired per seed)
labels = ["0", "(0, 0.4]", "(0.4, 0.8]", "(0.8, 1.2]", "> 1.2"]
for i, b in enumerate(bins):
    g = np.array(b["soft"]) - np.array(b["text"])
    col = BLUE
    a2.scatter(g, i + rng.uniform(-0.13, 0.13, len(g)), s=7, color=col, alpha=0.45, lw=0, zorder=2)
    a2.errorbar(g.mean(), i, xerr=g.std(ddof=1), fmt="o", ms=4.2, color=col, mec="white", mew=0.6, elinewidth=1.0, zorder=3)
    a2.text(1.03, i, f"+{g.mean():.1f}", transform=a2.get_yaxis_transform(), va="center", ha="left", fontsize=7, color=INK if i >= 3 else INK2,
            fontweight="bold" if i >= 3 else "normal")
a2.set_yticks(range(5)); a2.set_yticklabels([f"{l}  (n={b['n']})" for l, b in zip(labels, bins)]); a2.set_ylim(4.6, -0.6)
a2.set_xlim(0, 34); a2.set_xticks([0, 10, 20, 30]); a2.set_xlabel("Gain over text-only BERT (Acc-2 points)")
a2.grid(axis="x", color=GRID, lw=0.6); a2.set_axisbelow(True)
head(a2, r"Gain by text–visual label gap $|y_T-y_V|$", x=-0.42)

# --- bottom: cross-validation forest plot (own axes so the label and value columns sit outside the plot)
a3.remove()
a3 = fig.add_axes([0.335, 0.085, 0.50, 0.30])
# text-only rows use the CIs of AUDIT.md section 8 (same predictions; the paper reports these); baselines from cv_compare.json
AUDIT_TEXT = {"all": (6.9, 5.9, 7.9), "conflict": (8.0, 6.2, 9.8), "strict": (7.2, 2.8, 12.1)}
subs = [("all", "All clips (n = 4403)"), ("conflict", "Conflict (n = 2038)"), ("strict", "Strict text–visual opposite (n = 470)")]
comps = [("text", "vs. text-only BERT", BLUE), ("deva_dense", "vs. DEVA-style template", ORANGE), ("mag", "vs. MAG-style gating", AQUA)]
ypos, ylab, y = [], [], 0.0
for sk, sname in subs:
    a3.text(-0.62, y, sname, transform=a3.get_yaxis_transform(), ha="left", va="center", fontsize=7.2, fontweight="bold", color=INK)
    y += 1
    for ck, cname, col in comps:
        if ck == "text":
            d, lo, hi = AUDIT_TEXT[sk]
        else:
            r = cv[f"au_soft_emo-{ck}|{sk}"]; d = r["ens_diff"] * 100; lo, hi = (v * 100 for v in r["ci95"])
        a3.plot([lo, hi], [y, y], color=col, lw=1.3, solid_capstyle="round", zorder=2)
        a3.plot(d, y, "o", ms=4.6, color=col, mec="white", mew=0.6, zorder=3)
        a3.text(1.02, y, f"{d:+.1f}   [{lo:+.1f}, {hi:+.1f}]", transform=a3.get_yaxis_transform(), va="center", ha="left",
                fontsize=6.8, color=INK2)
        ypos.append(y); ylab.append(cname); y += 1
    y += 0.35
a3.axvline(0, color=INK2, lw=0.7, zorder=1)
a3.set_yticks(ypos); a3.set_yticklabels(ylab, fontsize=6.9, color=INK2); a3.set_ylim(y - 0.6, -0.6)
a3.set_xlim(-1, 13); a3.set_xticks([0, 2, 4, 6, 8, 10, 12])
a3.set_xlabel("Ensemble Acc-2 gain of soft tokens + emotions / EMFACS (points)")
a3.grid(axis="x", color=GRID, lw=0.6); a3.set_axisbelow(True)
a3.text(1.02, -1.15, "gain   [95% CI]", transform=a3.get_yaxis_transform(), ha="left", va="center", fontsize=6.6, color=MUTED)
a3.text(-0.62, -1.9, "Five-fold cross-validation over all CH-SIMS v2 clips", transform=a3.get_yaxis_transform(), fontsize=7.6,
        fontweight="bold", color=INK, ha="left", va="center")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"conflict_bins.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "conflict_bins.pdf")
