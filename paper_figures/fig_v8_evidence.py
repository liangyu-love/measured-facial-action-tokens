"""Figure 3 (v3, 2026-11-11): post-hoc evidence maps of the auxiliary fa2 model + occlusion sensitivity, drawn from scratch.
Maps: exported seed-1 layer-8 softmax cosine-similarity weights (audit/figure_evidence_export.json), shown as a multiple of the uniform
weight 1/160 on a single-hue log scale. Occlusion: results_evi/fa2_posthoc_seed{1..5}.json, layer 8.
Output: figures/v8/evidence_example.{pdf,png}"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle

P = Path(__file__).resolve().parents[1]
F = P.parent / "results/outputs/full"
OUT = P / "figures/v8"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, RULE, GRID = "#22262a", "#5c6268", "#9aa0a6", "#c9c6bf", "#ecebe7"
BLUE, GREY, RED = "#1f6f78", "#a7abaf", "#b5705a"
SEQ = LinearSegmentedColormap.from_list("seq_teal", ["#fbfcfc", "#d3e6e6", "#9cc6c6", "#5fa2a4", "#2f7f84", "#1f6f78", "#123f44"])
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.edgecolor": RULE,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic"})

cases = json.loads((P / "audit/figure_evidence_export.json").read_text(encoding="utf8"))[:2]
TRANS = {"video_0051/0032": "“Take her away. If I survive, I will repay you.”", "video_0016/0040": "“Zhang Yang, thank you.”"}
norm = LogNorm(vmin=1, vmax=100, clip=True)  # at or below the uniform weight -> lightest step

fig = plt.figure(figsize=(6.77, 4.1))
axs = [fig.add_axes([0.075, 0.43, 0.33, 0.43]), fig.add_axes([0.555, 0.43, 0.33, 0.43])]
for i, (ax, d) in enumerate(zip(axs, cases)):
    w = np.array(d["weights"]).T * 160  # (20 AU, 8 frames), multiple of the uniform weight
    im = ax.imshow(w, cmap=SEQ, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"F{k}" for k in range(1, 9)])
    ax.set_yticks(range(20)); ax.set_yticklabels([c.replace("AU", "AU ") for c in d["cols"]])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True); ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
    ax.grid(which="minor", color="#ebeae6", lw=0.35); ax.tick_params(which="minor", length=0)
    row = d["cols"].index("AU04")
    ax.add_patch(Rectangle((-0.5, row - 0.5), 8, 1, fill=False, ec=INK, lw=0.8))
    lab = ax.get_yticklabels()[row]; lab.set_fontweight("bold"); lab.set_color(INK)
    ok = (float(d["y"]) > 0) == (float(d["pred"]) > 0)
    ax.text(-0.5, -2.9, TRANS[d["key"]], fontsize=7.2, color=INK, ha="left", va="bottom")
    info = (f"labels: text {float(d['yt']):+.0f}, visual {float(d['yv']):+.0f}, overall {float(d['y']):+.1f};  "
            f"prediction {float(d['pred']):+.2f}").replace("-", "−")
    ax.text(-0.5, -1.0, info, fontsize=6.6, color=INK2, ha="left", va="bottom")
    ax.text(-0.5, -4.6, ("Correct polarity" if ok else "Incorrect polarity"), fontsize=7.6, fontweight="bold",
            color=INK, ha="left", va="bottom")
    ax.set_xlabel("sampled frame", fontsize=7, color=INK2, labelpad=2)
cax = fig.add_axes([0.93, 0.47, 0.012, 0.35])
cb = fig.colorbar(im, cax=cax, ticks=[1, 3, 10, 30, 100])
cb.ax.set_yticklabels(["≤1×", "3×", "10×", "30×", "100×"], fontsize=6.4); cb.outline.set_visible(False)
cb.ax.minorticks_off()
cb.ax.tick_params(length=2, width=0.6, color=RULE)
cb.set_label("similarity weight\n(× uniform)", fontsize=6.6, color=INK2, labelpad=3)

# --- occlusion sensitivity across the test set
ax = fig.add_axes([0.20, 0.075, 0.62, 0.21])
names = [("Lowest-weight 10%", "S_bottom", GREY), ("Random 10%", "random", GREY), ("Attention rollout top 10%", "rollout_top", GREY),
         ("Highest-weight 10%", "S_top", BLUE)]
rng = np.random.default_rng(3)
for i, (lab, k, col) in enumerate(names):
    v = np.array([json.loads((F / f"results_evi/fa2_posthoc_seed{s}.json").read_text())["faithfulness"]["L8"][k]["mean_abs_dpred"]
                  for s in range(1, 6)])
    ax.scatter(v, i + rng.uniform(-0.12, 0.12, 5), s=7, color=col, alpha=0.45, lw=0, zorder=2)
    ax.errorbar(v.mean(), i, xerr=v.std(ddof=1), fmt="o", ms=4.2, color=col, mec="white", mew=0.6, elinewidth=1.0, zorder=3)
    ax.text(1.02, i, f"{v.mean():.3f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.8,
            color=INK if col == BLUE else INK2, fontweight="bold" if col == BLUE else "normal")
ax.set_yticks(range(4)); ax.set_yticklabels([n[0] for n in names], fontsize=6.9, color=INK); ax.set_ylim(3.6, -0.6)
ax.set_xlim(0, 0.2); ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
ax.set_xlabel(r"mean $|\Delta\hat{y}|$ after masking 16 of 160 evidence tokens (1034 test clips, 5 seeds)", fontsize=7, color=INK2)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.text(-0.33, 1.12, "Occlusion sensitivity of the maps", transform=ax.transAxes, fontsize=7.6, fontweight="bold", color=INK, ha="left")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"evidence_example.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "evidence_example.pdf")
