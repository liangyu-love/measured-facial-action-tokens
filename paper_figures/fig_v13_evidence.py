"""Figure 6 (v13, 2026-11-17): evidence maps of the auxiliary per-AU model, redrawn to show three points at a glance:
(1) the map names facial actions per frame and can be checked against the measured intensity (bar strip under each map);
(2) the maps are faithful (occlusion panel, ~11x random); (3) a concentrated map does not imply the right polarity (same AU04 focus,
one correct and one wrong prediction, read on a shared sentiment axis as in Figure 1).
Data: audit/figure_evidence_export.json (seed-1 layer-8 weights and the model's measured AU input), results_evi/fa2_posthoc_seed{1..5}.json.
Output: figures/v13/fig6_evidence.{pdf,png}"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch

P = Path(__file__).resolve().parents[1]
F = P.parent / "results/outputs/full"
OUT = P / "figures/v13"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, RULE, GRID, TEAL, GREY, RED, SAND, SAND_D = ("#22262a", "#5c6268", "#9aa0a6", "#c9c6bf", "#ecebe7", "#1f6f78",
                                                              "#a7abaf", "#b5483a", "#b9a58a", "#8c7a5f")
SEQ = LinearSegmentedColormap.from_list("seq_teal", ["#f4f4f1", "#d3e6e6", "#9cc6c6", "#5fa2a4", "#2f7f84", "#1f6f78", "#123f44"])
plt.rcParams.update({"font.family": "Arial", "font.size": 7.2, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "axes.edgecolor": RULE,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
NAMES = {"AU04": "brow lowerer", "AU43": "eyes closed", "AU28": "lip suck", "AU23": "lip tightener", "AU14": "dimpler",
         "AU01": "inner brow raiser", "AU07": "lid tightener", "AU15": "lip-corner depressor", "AU17": "chin raiser"}
TRANS = {"video_0051/0032": "“Take her away. If I survive, I will repay you.”", "video_0016/0040": "“Zhang Yang, thank you.”"}

cases = json.loads((P / "audit/figure_evidence_export.json").read_text(encoding="utf8"))[:2]
cols = cases[0]["cols"]
Wt = [np.array(d["weights"]).T * 160 for d in cases]          # (20 AU, 8 frames), multiple of the uniform weight 1/160
AUm = [np.array(d["au"]).T for d in cases]                     # measured intensities fed to the model, (20, 8)
order = np.argsort(-np.max([w.max(1) for w in Wt], 0))
keep = sorted(order[:5].tolist(), key=lambda i: -max(w[i].max() for w in Wt))
rest = [i for i in range(20) if i not in keep]
i04 = cols.index("AU04")
norm = LogNorm(vmin=1, vmax=100, clip=True)
f = lambda v, n=1: f"{float(v):+.{n}f}".replace("-", "−")

fig = plt.figure(figsize=(6.9, 3.15))
X0 = [0.165, 0.455]; WMAP = 0.235
for i, (d, w, a) in enumerate(zip(cases, Wt, AUm)):
    ok = (float(d["y"]) > 0) == (float(d["pred"]) > 0)
    col = TEAL if ok else RED
    # --- header: verdict, transcript, shared sentiment axis (labels + prediction), as in Figure 1
    fig.text(X0[i], 0.985, ("Correct polarity" if ok else "Wrong polarity"), fontsize=7.6, fontweight="bold", color=col, va="top")
    fig.text(X0[i], 0.93, TRANS[d["key"]], fontsize=6.5, style="italic", color=INK, va="top")
    ax = fig.add_axes([X0[i], 0.66, WMAP, 0.20])
    ax.axvspan(-1.12, 0, color="#e8f1f1", lw=0); ax.axvspan(0, 1.12, color="#f3efe7", lw=0); ax.axvline(0, color=INK2, lw=0.6)
    rows = ((3, d["yt"], "text label", "o", SAND_D), (2, d["yv"], "visual label", "o", TEAL), (1, d["y"], "overall label", "D", INK),
            (0, d["pred"], "model", "o", col))
    for yy, v, lab, kind, c_ in rows:
        pred = lab == "model"
        vc = float(np.clip(v, -1.05, 1.05))
        ax.plot([0, vc], [yy, yy], color=c_, lw=1.0 if pred else 0.8)
        ax.plot(vc, yy, marker=kind, ms=4.4 if pred else 3.4, mfc=("white" if kind == "o" and not pred else c_), mec=c_, mew=0.9)
        if i == 0:
            ax.text(-1.16, yy, lab, ha="right", va="center", fontsize=5.9, color=c_ if pred else INK2, fontweight="bold" if pred else "normal")
        ax.text(vc - 0.07 if v > 0 else vc + 0.07, yy + 0.2, f(v, 2 if pred else 1), ha="right" if v > 0 else "left", va="bottom",
                fontsize=5.6, color=c_, fontweight="bold")
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.5, 3.9); ax.axis("off")
    # --- similarity map over frames x action units
    ax = fig.add_axes([X0[i], 0.29, WMAP, 0.33])
    m = np.vstack([w[keep], w[rest].max(0, keepdims=True)])
    im = ax.imshow(m, cmap=SEQ, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks(range(len(keep) + 1))
    ax.set_yticklabels([f"{cols[k]}  {NAMES.get(cols[k], '')}" for k in keep] + [f"other {len(rest)} AUs (max)"] if i == 0 else [""] * (len(keep) + 1))
    if i == 0:
        ax.get_yticklabels()[-1].set_color(MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True); ax.set_yticks(np.arange(-0.5, len(keep) + 1, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", lw=1.0); ax.tick_params(which="minor", length=0)
    # --- measured AU04 intensity per frame, aligned with the map columns
    ax2 = fig.add_axes([X0[i], 0.14, WMAP, 0.11])
    ax2.bar(range(8), a[i04], width=0.78, color="#c6dadb", edgecolor="none")
    top = int(np.argmax(w[i04])); ax2.bar([top], [a[i04][top]], width=0.78, color=TEAL)
    ax2.set_xlim(-0.5, 7.5); ax2.set_ylim(0, 1.05); ax2.set_yticks([0, 1]); ax2.set_yticklabels(["0", "1"] if i == 0 else ["", ""])
    ax2.set_xticks(range(8)); ax2.set_xticklabels([str(k) for k in range(1, 9)])
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.tick_params(length=1.5, pad=1.5)
    ax2.set_xlabel("sampled frame", fontsize=6.4, color=INK2, labelpad=1.5)
    if i == 0:
        ax2.set_ylabel("measured\nAU04", fontsize=5.8, color=INK2, rotation=0, ha="right", va="center", labelpad=4)
cax = fig.add_axes([0.255, 0.018, 0.24, 0.02])
cb = fig.colorbar(im, cax=cax, ticks=[1, 3, 10, 30, 100], orientation="horizontal")
cb.ax.set_xticklabels(["≤1×", "3×", "10×", "30×", "100×"], fontsize=5.9); cb.outline.set_visible(False)
cb.ax.minorticks_off(); cb.ax.tick_params(length=1.5, width=0.6, color=RULE, pad=1)
fig.text(0.245, 0.028, "similarity weight (× uniform)", fontsize=6.1, color=INK2, ha="right", va="center")

# --- occlusion sensitivity across the test set
fig.add_artist(plt.Line2D([0.725, 0.725], [0.04, 0.97], color=GRID, lw=0.8))
fig.text(0.745, 0.965, "Are the maps faithful?", fontsize=7.6, fontweight="bold", color=INK, va="top")
fig.text(0.745, 0.905, "mask 16 of 160 facial tokens,\nchange of prediction (1034 test clips, 5 seeds)", fontsize=6.1, color=INK2, va="top",
         linespacing=1.25)
ax = fig.add_axes([0.80, 0.27, 0.13, 0.46])
names = [("lowest weights", "S_bottom", GREY), ("random", "random", GREY), ("attention rollout", "rollout_top", GREY),
         ("highest weights", "S_top", TEAL)]
rng = np.random.default_rng(3); means = {}
for k_, (lab, key, c_) in enumerate(names):
    v = np.array([json.loads((F / f"results_evi/fa2_posthoc_seed{s}.json").read_text())["faithfulness"]["L8"][key]["mean_abs_dpred"]
                  for s in range(1, 6)])
    means[key] = v.mean()
    ax.barh(k_, v.mean(), height=0.56, color=("#d9e8e8" if c_ == TEAL else "#ececea"), edgecolor="none", zorder=1)
    ax.scatter(v, k_ + rng.uniform(-0.12, 0.12, 5), s=5, color=c_, alpha=0.7, lw=0, zorder=2)
    ax.text(v.mean() + 0.006, k_ - 0.36, f"{v.mean():.3f}", fontsize=5.9, color=INK2, va="center")
ax.set_yticks(range(4)); ax.set_yticklabels([n[0] for n in names], fontsize=6.3, color=INK); ax.set_ylim(3.6, -0.6)
ax.set_xlim(0, 0.2); ax.set_xticks([0, 0.1, 0.2]); ax.set_xticklabels(["0", "0.1", "0.2"])
ax.set_xlabel(r"mean $|\Delta\hat{y}|$", fontsize=6.4, color=INK2, labelpad=2)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ratio = means["S_top"] / means["random"]
ax.annotate("", xy=(means["S_top"], 3.75), xytext=(means["random"], 3.75),
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=0.8, mutation_scale=6))
ax.text((means["S_top"] + means["random"]) / 2, 3.75 + 0.15, f"{ratio:.1f}× random", fontsize=6.2, color=TEAL, ha="center", va="top",
        fontweight="bold", transform=ax.transData)
ax.set_ylim(4.35, -0.6)
print(f"kept {[cols[k] for k in keep]}; top/random = {ratio:.2f}")
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig6_evidence.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig6_evidence.pdf")
