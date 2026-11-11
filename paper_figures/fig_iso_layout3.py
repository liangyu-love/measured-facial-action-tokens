"""Layout of Figure 1 v7: flat measurement card (no bars), orthogonal rounded connectors, transcript in line with the word tokens.
Executed by fig_iso_method3.py after the shared primitives."""
# ruff: noqa: F821  (names come from the executing script)


def sub(base, s, bold=True, color=None):
    c = f' fill="{color}"' if color else ""
    w = ' font-weight="700"' if bold else ""
    return (f'<tspan font-family="Cambria" font-style="italic"{w}{c}>{base}</tspan>'
            f'<tspan font-family="Cambria" font-style="italic" baseline-shift="sub" font-size="4.6pt"{c}>{s}</tspan>')


def verb(x, y, s, anchor="middle", color=SLATE):
    text(x, y, s, 5.6, color, anchor, italic=True)


def elbow(pts, color=SLATE, sw=0.3, r=1.6, arrow=True):
    """orthogonal polyline with rounded corners"""
    d = f"M{pts[0][0]:.2f},{pts[0][1]:.2f}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        v1 = np.array([x1 - x0, y1 - y0]); v1 = v1 / (np.linalg.norm(v1) or 1)
        v2 = np.array([x2 - x1, y2 - y1]); v2 = v2 / (np.linalg.norm(v2) or 1)
        a_ = (x1 - v1[0] * r, y1 - v1[1] * r); b_ = (x1 + v2[0] * r, y1 + v2[1] * r)
        d += f" L{a_[0]:.2f},{a_[1]:.2f} Q{x1:.2f},{y1:.2f} {b_[0]:.2f},{b_[1]:.2f}"
    d += f" L{pts[-1][0]:.2f},{pts[-1][1]:.2f}"
    path(d, color, sw, arrow=arrow)


ROW_Y = 50.0  # screen y of the cube bases (sequence row)

# ================================================================== FACIAL BRANCH (top)
text(4.0, 6.0, "FACIAL EVIDENCE", 5.4, TEAL, weight=700, extra='letter-spacing="0.35"')
of, sf = (11.5, 24.0), 1.0
cw, ch = 7.0, 10.8
for k, fi in enumerate([8, 4, 1]):
    y0 = -5.4 + k * 2.7; x0 = 0.0
    uri, (Wpx, Hpx) = frame_uri(fi)
    box(x0, y0 - 0.32, 0, cw, 0.32, ch, of, sf, "#f6f5f2", stroke="#c9c6bf", sw=0.1, k_left=0.05, k_right=0.12)
    a, b = C30 * sf * cw / Wpx, S30 * sf * cw / Wpx
    d = sf * ch / Hpx
    e = of[0] + (x0 - y0) * C30 * sf; f = of[1] + (x0 + y0) * S30 * sf - sf * ch
    svg.append(f'<image href="{uri}" width="{Wpx}" height="{Hpx}" preserveAspectRatio="none" '
               f'transform="matrix({a:.6f} {b:.6f} 0 {d:.6f} {e:.4f} {f:.4f})" opacity="{0.55 + 0.225 * k:.3f}"/>')
    poly([iso(x0, y0, 0, of, sf), iso(x0 + cw, y0, 0, of, sf), iso(x0 + cw, y0, ch, of, sf), iso(x0, y0, ch, of, sf)], "none",
         (TEAL if fi == HL else "#ffffff"), (0.32 if fi == HL else 0.35))
text(12.6, 32.4, "8 sampled frames", 5.6, SLATE, "middle")

# measurement card: a standing sheet in the same plane as the frames; rows = frames, columns = 33 measurements
oc, sc = (39.0, 26.0), 1.0
gap_after = {19: 0.55, 20: 0.55, 27: 0.55}
cxs, x = [], 0.0
for r in range(33):
    cxs.append(x); x += 0.86 + gap_after.get(r, 0.0)
CW_, RH = x - 0.06, 1.25  # card width (x) and row height (z)
card = [iso(-0.5, 0, -0.5, oc, sc), iso(CW_ + 0.5, 0, -0.5, oc, sc), iso(CW_ + 0.5, 0, 8 * RH + 0.5, oc, sc), iso(-0.5, 0, 8 * RH + 0.5, oc, sc)]
box(-0.5, -0.32, -0.5, CW_ + 1.0, 0.32, 8 * RH + 1.0, oc, sc, "#f6f5f2", stroke="#c9c6bf", sw=0.1, k_left=0.05, k_right=0.12)
poly(card, "#ffffff", "#c9c6bf", 0.1)
for fidx in range(8):
    zt_ = (8 - fidx) * RH  # frame 1 on top
    for r in range(33):
        v = M[fidx, r]; hl = (fidx == HL - 1)
        col = mix("#f3f6f6", TEAL, 0.12 + 0.88 * v) if hl else mix("#f4f3f0", "#5d636a", 0.06 + 0.80 * v)
        xa = cxs[r]
        poly([iso(xa, 0, zt_ - RH + 0.12, oc, sc), iso(xa + 0.78, 0, zt_ - RH + 0.12, oc, sc),
              iso(xa + 0.78, 0, zt_ - 0.08, oc, sc), iso(xa, 0, zt_ - 0.08, oc, sc)], col)
# frame-4 row outline
z4t, z4b = (8 - (HL - 1)) * RH, (8 - HL) * RH
poly([iso(-0.2, 0, z4b + 0.02, oc, sc), iso(CW_ + 0.2, 0, z4b + 0.02, oc, sc), iso(CW_ + 0.2, 0, z4t + 0.02, oc, sc),
      iso(-0.2, 0, z4t + 0.02, oc, sc)], "none", TEAL, 0.3)
# group labels under the card, along its bottom edge
for name, a, b in [("AU intensities", 0, 20), ("emotion", 21, 28), ("EMFACS", 28, 33)]:
    xa, xb = cxs[a], cxs[b - 1] + 0.78
    p0, p1 = iso(xa, 0, -1.1, oc, sc), iso(xb, 0, -1.1, oc, sc)
    svg.append(f'<line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" stroke="{MUTE}" stroke-width="0.18"/>')
    pm = iso((xa + xb) / 2, 0, -1.1, oc, sc)
    dz = 2.3 if name == "emotion" else 0.0
    pm = iso((xa + xb) / 2, 0, -1.1 - dz, oc, sc)
    text(pm[0] - 0.2, pm[1] + 2.6, name, 5.0, SLATE, "middle", rot=30)
lt = iso(-0.5, 0, 8 * RH + 0.5, oc, sc)
text(lt[0], lt[1] - 2.4, "per-frame measurements  " + sub("a", "f") + '<tspan font-family="Cambria"> ∈ ℝ</tspan>'
     '<tspan font-family="Cambria" baseline-shift="super" font-size="4.6pt">33</tspan>', 6.2, INK, "start")
r4 = iso(CW_ + 0.5, 0, (z4t + z4b) / 2, oc, sc)  # right end of the frame-4 row
text(r4[0] + 1.0, r4[1] - 1.0, '<tspan font-style="italic">f</tspan> = 4', 5.8, TEAL, family=MATH)

# frames -> card (Py-Feat): short straight connector
fa = iso(cw + 0.4, 0, ch * 0.55, of, sf); ca = iso(-0.9, 0, 8 * RH * 0.55, oc, sc)
path(f"M{fa[0]:.2f},{fa[1]:.2f} L{ca[0]:.2f},{ca[1]:.2f}", SLATE, 0.3)
verb((fa[0] + ca[0]) / 2, min(fa[1], ca[1]) - 1.6, "Py-Feat")

# ================================================================== SEQUENCE ROW
ot, st_ = (54.0, ROW_Y), 1.2
toks = [("[CLS]", "w"), ("w", "w", "1"), ("⋯", "gap"), ("w", "w", "n"), ("[SEP]", "w"),
        ("e", "e", "1"), ("⋯", "gap"), ("e", "e", "4"), ("⋯", "gap"), ("e", "e", "8")]
cube, step = 2.25, 2.95
cent = []
for k, tk in enumerate(toks):
    lab, kind = tk[0], tk[1]; s_ = tk[2] if len(tk) > 2 else None
    x0, y0 = k * step, -k * step
    top_c = iso(x0 + cube / 2, y0 + cube / 2, cube, ot, st_)
    base = iso(x0 + cube, y0 + cube, 0, ot, st_)
    leftv = iso(x0, y0 + cube, cube / 2, ot, st_)
    rightv = iso(x0 + cube, y0, cube / 2, ot, st_)
    if kind == "gap":
        text(top_c[0], top_c[1] + 2.5, "⋯", 7, MUTE, "middle")
    else:
        hl = kind == "e" and s_ == str(HL)
        top = "#efe8dc" if kind == "w" else mix("#ffffff", TEAL, 0.95 if hl else 0.55)
        box(x0, y0, 0, cube, cube, cube, ot, st_, top, stroke=("#a8977c" if kind == "w" else "#16545b"), sw=0.08,
            k_left=0.16, k_right=0.08)
        if s_ is None:
            text(base[0], base[1] + 3.0, lab, 4.9, INK, "middle")
        else:
            text(base[0], base[1] + 3.0, sub(lab, s_, bold=(kind == "e"), color=(TEAL if kind == "e" else INK)), 6.0, INK, "middle")
    cent.append(dict(top=top_c, base=base, kind=kind, left=leftv, right=rightv))
W_ = [c for c in cent if c["kind"] == "w"]; E_ = [c for c in cent if c["kind"] == "e"]
text((W_[0]["base"][0] + W_[-1]["base"][0]) / 2, ROW_Y + 9.6, "word tokens", 5.4, SLATE, "middle")
text((E_[0]["base"][0] + E_[-1]["base"][0]) / 2, ROW_Y + 9.6, "facial action tokens", 5.4, TEAL, "middle")

# ================================================================== TEXT BRANCH (left, in line with the word tokens)
cy = W_[0]["left"][1]
text(4.0, cy - 8.4, "TRANSCRIPT", 5.4, "#8c7a5f", weight=700, extra='letter-spacing="0.35"')
text(4.0, cy - 2.6, "现在是关键时刻，大家要团结嘛", 6.0, INK, family="'Microsoft YaHei'")
text(4.0, cy + 1.2, "“It is a critical moment;", 5.6, SLATE, italic=True)
text(4.0, cy + 4.2, "we must stay united.”", 5.6, SLATE, italic=True)
x_from = 37.5
path(f"M{x_from:.2f},{cy:.2f} L{W_[0]['left'][0] - 1.0:.2f},{cy:.2f}", SLATE, 0.3)
verb((x_from + W_[0]["left"][0]) / 2, cy - 1.4, "tokenize")

# ================================================================== projection: frame-4 row -> bracket over all facial tokens
by = min(c["top"][1] for c in E_) - 4.6
ex0, ex1 = E_[0]["top"][0], E_[-1]["top"][0]; e4x = E_[1]["top"][0]
elbow([(r4[0] + 0.6 + 8.0, r4[1]), (e4x, r4[1]), (e4x, by)], TEAL, 0.32, arrow=False)
path(f"M{r4[0] + 0.6:.2f},{r4[1]:.2f} L{r4[0] + 8.6:.2f},{r4[1]:.2f}", TEAL, 0.32, arrow=False)
svg.append(f'<line x1="{ex0:.2f}" y1="{by:.2f}" x2="{ex1:.2f}" y2="{by:.2f}" stroke="{TEAL}" stroke-width="0.32" stroke-linecap="round"/>')
for c in E_:
    path(f"M{c['top'][0]:.2f},{by:.2f} L{c['top'][0]:.2f},{c['top'][1] - 1.3:.2f}", TEAL, 0.32)
fx, fy = e4x + 2.0, r4[1] + 1.0
text((r4[0] + e4x) / 2 + 3.0, r4[1] - 1.5, "project, one token per frame", 5.6, TEAL, "middle", italic=True)
text(fx, fy + 3.6, sub("e", "f", color=TEAL) + '<tspan font-family="Cambria"> = LN(</tspan>' + sub("W", "a", color=TEAL)
     + sub("a", "f", color=TEAL) + '<tspan font-family="Cambria"> + </tspan>' + sub("b", "a", color=TEAL)
     + '<tspan font-family="Cambria">) + </tspan>' + sub("q", "f", color=TEAL), 6.4, TEAL)

