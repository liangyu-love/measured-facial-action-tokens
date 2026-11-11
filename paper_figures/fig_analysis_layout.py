"""Figure 4 (Section 6): explicit conflict-modelling modules tested on top of the soft-token model, and the two controls.
Solid = proposed model; dashed = analysis-only modules. Outcome tags quote verified results (AUDIT.md, tuned_results.md, paper Sec. 6).
Executed by fig_analysis.py after the shared primitives."""
# ruff: noqa: F821

DASH = "1.1 0.8"


def sub(base, s, bold=True, color=None):
    c = f' fill="{color}"' if color else ""
    w = ' font-weight="700"' if bold else ""
    return (f'<tspan font-family="Cambria" font-style="italic"{w}{c}>{base}</tspan>'
            f'<tspan font-family="Cambria" font-style="italic" baseline-shift="sub" font-size="4.6pt"{c}>{s}</tspan>')


def rbox(x, y, w, h, stroke=SLATE, fill="#ffffff", dashed=True, r=1.4, sw=0.26):
    da = f' stroke-dasharray="{DASH}"' if dashed else ""
    svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="{sw}"{da}/>')


def tag(x, y, s, anchor="start"):
    text(x, y, s, 5.1, "#8a8f95", anchor, italic=True)


def elbow(pts, color=SLATE, sw=0.3, r=1.4, arrow=True, dash=None):
    d = f"M{pts[0][0]:.2f},{pts[0][1]:.2f}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        v1 = np.array([x1 - x0, y1 - y0]); v1 = v1 / (np.linalg.norm(v1) or 1)
        v2 = np.array([x2 - x1, y2 - y1]); v2 = v2 / (np.linalg.norm(v2) or 1)
        a_ = (x1 - v1[0] * r, y1 - v1[1] * r); b_ = (x1 + v2[0] * r, y1 + v2[1] * r)
        d += f" L{a_[0]:.2f},{a_[1]:.2f} Q{x1:.2f},{y1:.2f} {b_[0]:.2f},{b_[1]:.2f}"
    d += f" L{pts[-1][0]:.2f},{pts[-1][1]:.2f}"
    path(d, color, sw, dash=dash, arrow=arrow)


# ------------------------------------------------------------------ legend
svg.append(f'<line x1="4" y1="5.2" x2="10" y2="5.2" stroke="{INK}" stroke-width="0.3"/>')
text(11.2, 6.2, "proposed model", 5.4, SLATE)
svg.append(f'<line x1="34" y1="5.2" x2="40" y2="5.2" stroke="{SLATE}" stroke-width="0.3" stroke-dasharray="{DASH}"/>')
text(41.2, 6.2, "analysis-only module", 5.4, SLATE)

# ------------------------------------------------------------------ main model (centre row)
ROW_Y = 30.0
ot, st_ = (44.0, ROW_Y), 1.05
toks = [("[CLS]", "w"), ("w", "w", "1"), ("⋯", "gap"), ("[SEP]", "w"), ("e", "e", "1"), ("⋯", "gap"), ("e", "e", "8"), ("[DIFF]", "d")]
cube, step = 2.25, 2.95
cent = []
for k, tk in enumerate(toks):
    lab, kind = tk[0], tk[1]; s_ = tk[2] if len(tk) > 2 else None
    x0, y0 = k * step, -k * step
    top_c = iso(x0 + cube / 2, y0 + cube / 2, cube, ot, st_)
    base = iso(x0 + cube, y0 + cube, 0, ot, st_)
    if kind == "gap":
        text(top_c[0], top_c[1] + 2.4, "⋯", 7, MUTE, "middle")
    else:
        top = {"w": "#efe8dc", "e": mix("#ffffff", TEAL, 0.55), "d": "#e3e5e8"}[kind]
        stroke = {"w": "#a8977c", "e": "#16545b", "d": SLATE}[kind]
        box(x0, y0, 0, cube, cube, cube, ot, st_, top, stroke=stroke, sw=0.08, k_left=0.16, k_right=0.08)
        if kind == "d":  # dashed outline marks the analysis-only token
            poly([iso(x0, y0, cube, ot, st_), iso(x0 + cube, y0, cube, ot, st_), iso(x0 + cube, y0 + cube, cube, ot, st_),
                  iso(x0, y0 + cube, cube, ot, st_)], "none", SLATE, 0.0)
        lab_s = lab if s_ is None else sub(lab, s_, bold=(kind == "e"), color=(TEAL if kind == "e" else INK))
        text(base[0], base[1] + 3.0, lab_s, (4.8 if s_ is None else 6.0), (SLATE if kind == "d" else INK), "middle")
    cent.append(dict(top=top_c, base=base, kind=kind, right=iso(x0 + cube, y0, cube / 2, ot, st_)))
D_ = cent[-1]
text(cent[0]["base"][0] - 4.0, ROW_Y + 1.0, "soft-token\nmodel", 5.4, SLATE, "end")

# encoder (light stack) and heads
oe, se, PS = (112.0, ROW_Y - 3.0), 0.62, 16.0
for l_ in range(4):
    box(0, 0, l_ * 2.4, PS, PS, 0.45, oe, se, "#f4f3ef" if l_ < 3 else "#fbfaf8", stroke="#a9a59c", sw=0.09, k_left=0.10, k_right=0.05)
y_in = D_["right"][1]
lv_x = iso(0, PS, 0, oe, se)[0]
path(f"M{D_['right'][0] + 1.0:.2f},{y_in:.2f} L{lv_x - 0.8:.2f},{y_in:.2f}", SLATE, 0.3)
text(oe[0], iso(PS, PS, 0, oe, se)[1] + 3.6, "BERT encoder", 5.8, INK, "middle", 700)
rv = iso(PS, 0, 4.0, oe, se)
hx0 = rv[0] + 7.5
# regression head (proposed) + two dashed heads
heads = [("regression head", "ŷ", False, ""), ("trust head", "which label does y follow?", True, "no strict-clip gain"),
         ("polarity head", "sign(y), BCE", True, "Acc-2 within ±2 (TOST)")]
for i_, (name, desc, dashed, res) in enumerate(heads):
    hy = ROW_Y - 17.5 + i_ * 11.6
    rbox(hx0, hy, 29.0, 9.4 if res else 6.6, INK if not dashed else SLATE, "#ffffff", dashed)
    text(hx0 + 1.6, hy + 2.7, name, 5.6, INK, weight=700)
    text(hx0 + 1.6, hy + 5.2, desc if name != "regression head" else "linear on [CLS] → ŷ", 5.1, SLATE)
    if res:
        tag(hx0 + 1.6, hy + 8.0, res)
    elbow([(rv[0] + 0.8, rv[1]), (hx0 - 2.6, rv[1]), (hx0 - 2.6, hy + 3.3), (hx0, hy + 3.3)], INK if not dashed else SLATE, 0.28,
          r=1.0, dash=(DASH if dashed else None))

# conflict gate on the facial tokens (dashed)
E_ = [c for c in cent if c["kind"] == "e"]
gx = (E_[0]["top"][0] + E_[-1]["top"][0]) / 2; gy = min(c["top"][1] for c in E_) - 6.4
rbox(gx - 12.5, gy - 4.6, 25.0, 6.0, SLATE, "#ffffff", True)
text(gx, gy - 2.0, "conflict gate", 5.6, INK, "middle", 700)
text(gx, gy + 0.4, "sample / frame weights on e", 5.0, SLATE, "middle")
for c in E_:
    path(f"M{c['top'][0]:.2f},{gy + 1.4:.2f} L{c['top'][0]:.2f},{c['top'][1] - 1.2:.2f}", SLATE, 0.26, dash=DASH)
tag(gx + 13.8, gy - 0.6, "collapses to near-constant weights")

# ------------------------------------------------------------------ unimodal experts (bottom-left)
ex_x, ex_w = 4.0, 46.0
text(ex_x, 47.0, "Frozen unimodal experts", 5.8, INK, weight=700)
text(ex_x, 50.0, "readings cross-fitted by source video", 5.1, SLATE, italic=True)
experts = [("text expert", "BERT fine-tuned on " + sub("y", "T", bold=False), sub("y", "T", bold=False), "#b9a58a"),
           ("visual expert", "evidence encoder on " + sub("y", "V", bold=False), sub("y", "V", bold=False), TEAL),
           ("speech expert", "frozen Chinese HuBERT", sub("y", "A", bold=False), "#7d8790")]
ey = []
for i_, (name, desc, out, col) in enumerate(experts):
    y_ = 53.0 + i_ * 8.4
    rbox(ex_x, y_, ex_w, 6.6, SLATE, "#ffffff", True)
    svg.append(f'<rect x="{ex_x + 0.9:.2f}" y="{y_ + 1.2:.2f}" width="1.3" height="4.2" rx="0.4" fill="{col}"/>')
    text(ex_x + 3.2, y_ + 2.8, name, 5.6, INK, weight=700)
    text(ex_x + 3.2, y_ + 5.3, desc, 5.1, SLATE)
    text(ex_x + ex_w - 1.4, y_ + 4.3, '<tspan font-family="Cambria" font-style="italic">ŷ</tspan>'
         + out.replace('font-style="italic"', 'font-style="italic"', 1).replace(">y<", "><"), 6.0, INK, "end")
    ey.append(y_ + 3.3)

# discrepancy combiner -> [DIFF]
cx0, cy0, cwid = 56.0, 56.0, 42.0
rbox(cx0, cy0, cwid, 11.0, SLATE, "#ffffff", True)
text(cx0 + 1.6, cy0 + 3.0, "discrepancy token", 5.6, INK, weight=700)
text(cx0 + 1.6, cy0 + 6.0, "[" + sub("z", "V") + "−" + sub("z", "T") + ", |" + sub("z", "V") + "−" + sub("z", "T") + "|, "
     + sub("z", "V") + "⊙" + sub("z", "T") + ", " + '<tspan font-family="Cambria" font-style="italic">ŷ</tspan>'
     + '<tspan font-family="Cambria" font-style="italic" baseline-shift="sub" font-size="4.6pt">V</tspan> − '
     + '<tspan font-family="Cambria" font-style="italic">ŷ</tspan>'
     + '<tspan font-family="Cambria" font-style="italic" baseline-shift="sub" font-size="4.6pt">T</tspan>]', 5.6, INK)
text(cx0 + 1.6, cy0 + 9.0, "projected to one extra token", 5.1, SLATE)
for yy in ey[:2]:
    elbow([(ex_x + ex_w, yy), (cx0 - 3.0, yy), (cx0 - 3.0, cy0 + 5.5), (cx0, cy0 + 5.5)], SLATE, 0.26, r=1.0, dash=DASH)
# speech reading feeds the trust variant (to the combiner as a third reading)
elbow([(ex_x + ex_w, ey[2]), (cx0 + 8.0, ey[2]), (cx0 + 8.0, cy0 + 11.0)], SLATE, 0.26, r=1.0, dash=DASH)
tag(cx0 + 9.4, ey[2] + 1.6, "+ speech reading (third opinion)")
dx_ = D_["base"][0]
path(f"M{dx_:.2f},{cy0:.2f} L{dx_:.2f},{D_['base'][1] + 4.2:.2f}", SLATE, 0.28, dash=DASH)
tag(dx_ - 1.6, cy0 - 8.0, "used: shuffling it costs 0.44 Corr on Conflict,", "end")
tag(dx_ - 1.6, cy0 - 5.2, "but no significant gain on strict clips", "end")

