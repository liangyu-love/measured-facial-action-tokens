"""Figure 4 (v10, 2026-11-12): post-hoc evidence maps of the auxiliary per-AU model + occlusion sensitivity, compact layout.
Maps show only the action units whose similarity weight exceeds the uniform weight in at least one frame of either clip (plus a
summary row for the others), instead of all 20 mostly empty rows. Data as in v8: audit/figure_evidence_export.json (seed-1 layer-8
softmax cosine-similarity weights) and results_evi/fa2_posthoc_seed{1..5}.json. Output: figures/v10/evidence_example.{pdf,png}"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

P = Path(__file__).resolve().parents[1]
F = P.parent / "results/outputs/full"
OUT = P / "figures/v10"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, RULE, GRID, TEAL, GREY, RED = "#22262a", "#5c6268", "#9aa0a6", "#c9c6bf", "#ecebe7", "#1f6f78", "#a7abaf", "#b5483a"
SEQ = LinearSegmentedColormap.from_list("seq_teal", ["#f4f4f1", "#d3e6e6", "#9cc6c6", "#5fa2a4", "#2f7f84", "#1f6f78", "#123f44"])
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.edgecolor": RULE,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic"})
NAMES = {"AU01": "inner brow raiser", "AU02": "outer brow raiser", "AU04": "brow lowerer", "AU05": "upper lid raiser",
         "AU06": "cheek raiser", "AU07": "lid tightener", "AU09": "nose wrinkler", "AU10": "upper lip raiser", "AU12": "lip-corner puller",
         "AU14": "dimpler", "AU15": "lip-corner depressor", "AU23": "lip tightener", "AU28": "lip suck", "AU43": "eyes closed", "AU17": "chin raiser", "AU25": "lips part", "AU26": "jaw drop"}

cases = json.loads((P / "audit/figure_evidence_export.json").read_text(encoding="utf8"))[:2]
TRANS = {"video_0051/0032": "“Take her away. If I survive, I will repay you.”", "video_0016/0040": "“Zhang Yang, thank you.”"}
cols = cases[0]["cols"]
Wt = [np.array(d["weights"]).T * 160 for d in cases]  # (20 AU, 8 frames), multiple of the uniform weight
order = np.argsort(-np.max([w.max(1) for w in Wt], 0))
keep = sorted(order[:5].tolist(), key=lambda i: -max(w[i].max() for w in Wt))
rest = [i for i in range(20) if i not in keep]
print("kept", [cols[i] for i in keep], "max weight among the others", max(w[rest].max() for w in Wt))
norm = LogNorm(vmin=1, vmax=100, clip=True)

fig = plt.figure(figsize=(6.77, 2.55))
axs = [fig.add_axes([0.155, 0.20, 0.215, 0.44]), fig.add_axes([0.435, 0.20, 0.215, 0.44])]
for i, (ax, d, w) in enumerate(zip(axs, cases, Wt)):
    m = np.vstack([w[keep], w[rest].max(0, keepdims=True)])
    im = ax.imshow(m, cmap=SEQ, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(8)); ax.set_xticklabels([str(k) for k in range(1, 9)])
    ax.set_yticks(range(len(keep) + 1))
    ax.set_yticklabels([f"{cols[k].replace('AU', 'AU')}  {NAMES.get(cols[k], '')}" for k in keep] + [f"other {len(rest)} AUs (max)"] if i == 0
                       else [""] * (len(keep) + 1))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True); ax.set_yticks(np.arange(-0.5, len(keep) + 1, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", lw=1.0); ax.tick_params(which="minor", length=0)
    ax.get_yticklabels()[-1].set_color(MUTED)
    ok = (float(d["y"]) > 0) == (float(d["pred"]) > 0)
    f = lambda v, n=1: f"{float(v):+.{n}f}".replace("-", "−")
    ax.text(0, 1.62, "Correct polarity" if ok else "Wrong polarity", transform=ax.transAxes, fontsize=7.6, fontweight="bold",
            color=TEAL if ok else RED, ha="left", va="bottom")
    ax.text(0, 1.38, TRANS[d["key"]], transform=ax.transAxes, fontsize=6.6, color=INK, ha="left", va="bottom", style="italic")
    ax.text(0, 1.20, f"text {f(d['yt'], 0)}, visual {f(d['yv'], 0)}, overall {f(d['y'])}", transform=ax.transAxes, fontsize=6.4,
            color=INK2, ha="left", va="bottom")
    ax.text(0, 1.04, f"prediction {f(d['pred'], 2)}", transform=ax.transAxes, fontsize=6.4, color=TEAL if ok else RED, ha="left",
            va="bottom", fontweight="bold")
    ax.set_xlabel("sampled frame", fontsize=6.8, color=INK2, labelpad=2)
cax = fig.add_axes([0.235, 0.045, 0.28, 0.022])
cb = fig.colorbar(im, cax=cax, ticks=[1, 3, 10, 30, 100], orientation="horizontal")
cb.ax.set_xticklabels(["≤1×", "3×", "10×", "30×", "100×"], fontsize=6.2); cb.outline.set_visible(False)
cb.ax.minorticks_off(); cb.ax.tick_params(length=2, width=0.6, color=RULE, pad=1.5)
fig.text(0.228, 0.056, "similarity weight (× uniform)", fontsize=6.4, color=INK2, ha="right", va="center")

# --- occlusion sensitivity across the test set
ax = fig.add_axes([0.80, 0.20, 0.145, 0.44])
names = [("lowest-weight 10%", "S_bottom", GREY), ("random 10%", "random", GREY), ("attention rollout\ntop 10%", "rollout_top", GREY),
         ("highest-weight 10%", "S_top", TEAL)]
rng = np.random.default_rng(3)
for i, (lab, k, col) in enumerate(names):
    v = np.array([json.loads((F / f"results_evi/fa2_posthoc_seed{s}.json").read_text())["faithfulness"]["L8"][k]["mean_abs_dpred"]
                  for s in range(1, 6)])
    ax.scatter(v, i + rng.uniform(-0.12, 0.12, 5), s=6, color=col, alpha=0.45, lw=0, zorder=2)
    ax.errorbar(v.mean(), i, xerr=v.std(ddof=1), fmt="o", ms=4.0, color=col, mec="white", mew=0.6, elinewidth=1.0, zorder=3)
    ax.text(1.05, i, f"{v.mean():.3f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.6, color=INK2)
ax.set_yticks(range(4)); ax.set_yticklabels([n[0] for n in names], fontsize=6.5, color=INK); ax.set_ylim(3.6, -0.6)
ax.set_xlim(0, 0.2); ax.set_xticks([0, 0.1, 0.2]); ax.set_xticklabels(["0", "0.1", "0.2"])
ax.set_xlabel(r"mean $|\Delta\hat{y}|$", fontsize=6.8, color=INK2, labelpad=2)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.text(0.685, 0.965, "Occlusion sensitivity", fontsize=7.6, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.685, 0.905, "masking 16 of 160 evidence tokens,\n1034 test clips, 5 seeds", fontsize=6.3, color=INK2, ha="left", va="top",
         linespacing=1.25)
fig.add_artist(plt.Line2D([0.672, 0.672], [0.04, 0.97], color=GRID, lw=0.8))

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"evidence_example.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "evidence_example.pdf")
