"""Encoder block of Figure 1 v7 (light 4-plate stack + "x 12", token tiles with joint-attention arcs on the top plate,
prediction read from the [CLS] tile). Executed after fig_iso_layout3.py."""
# ruff: noqa: F821

# ================================================================== ENCODER
oe, se, PS = (139.0, ROW_Y + 0.5), 0.78, 16.0  # origin, scale, plate size
LAYERS = [(0.0, "layer 1"), (3.0, "layer 2"), (9.0, "layer 12")]  # (z, label); a vertical ellipsis marks layers 3-11
for z_, _ in LAYERS:
    box(0, 0, z_, PS, PS, 0.45, oe, se, "#f4f3ef" if z_ < 8 else "#fbfaf8", stroke="#a9a59c", sw=0.09, k_left=0.10, k_right=0.05)
    if z_ == 3.0:  # ellipsis between layer 2 and layer 12, on the three visible vertices
        for vx, vy in ((0, PS), (PS, 0)):
            for kk in range(3):
                p_ = iso(vx, vy, 4.9 + kk * 1.05, oe, se)
                svg.append(f'<circle cx="{p_[0]:.2f}" cy="{p_[1]:.2f}" r="0.26" fill="{MUTE}"/>')
ztop = LAYERS[-1][0] + 0.45
# layer labels on the right, with a thin leader to each plate's right vertex
rb = iso(PS, 0, 0.0, oe, se)
bx_ = rb[0] + 2.0
for z_, lab in LAYERS:
    rv_ = iso(PS, 0, z_ + 0.22, oe, se)
    svg.append(f'<line x1="{rv_[0] + 0.5:.2f}" y1="{rv_[1]:.2f}" x2="{bx_:.2f}" y2="{rv_[1]:.2f}" stroke="{MUTE}" stroke-width="0.18"/>')
    text(bx_ + 0.6, rv_[1] + 0.8, lab, 5.2, SLATE)
mid = iso(PS, 0, 6.0, oe, se)
text(bx_ + 2.6, mid[1] + 1.4, "⋮", 6.5, MUTE, "middle")

# token tiles on the top plate: [CLS], two words, three facial tokens
tiles = []
for i_, kind in enumerate(["cls", "w", "w", "e", "e", "e"]):
    tsz = 1.55  # tiles along the screen-horizontal diagonal (1,-1) through the plate centre
    tx, ty = PS / 2 + (i_ - 2.5) * 2.05 - tsz / 2, PS / 2 - (i_ - 2.5) * 2.05 - tsz / 2
    fill = {"cls": "#d9ccb6", "w": "#efe8dc", "e": mix("#ffffff", TEAL, 0.6)}[kind]
    stroke = "#a8977c" if kind != "e" else "#16545b"
    poly([iso(tx, ty, ztop, oe, se), iso(tx + tsz, ty, ztop, oe, se), iso(tx + tsz, ty + tsz, ztop, oe, se),
          iso(tx, ty + tsz, ztop, oe, se)], fill, stroke, 0.08)
    tiles.append(iso(tx + tsz / 2, ty + tsz / 2, ztop, oe, se))
for i_, j_, col in [(1, 3, TEAL), (2, 4, TEAL), (1, 5, TEAL), (2, 3, TEAL), (1, 2, MUTE), (4, 5, TEAL)]:
    a_, b_ = tiles[i_], tiles[j_]
    mxp = ((a_[0] + b_[0]) / 2, (a_[1] + b_[1]) / 2 - 1.2 - 1.0 * abs(j_ - i_))
    path(f"M{a_[0]:.2f},{a_[1]:.2f} Q{mxp[0]:.2f},{mxp[1]:.2f} {b_[0]:.2f},{b_[1]:.2f}", col, 0.22, arrow=False, op=0.9)

# prediction from the [CLS] tile: up, then right above the arcs
cl = tiles[0]
yup = min(t[1] for t in tiles) - 8.2
xr = bx_ + 11.0
elbow([(cl[0], cl[1] - 0.6), (cl[0], yup), (xr, yup)], SLATE, 0.3, r=1.4)
text(cl[0] + 1.0, yup - 1.3, "[CLS] → linear", 5.4, SLATE, italic=True)
text(xr + 1.1, yup + 1.1, '<tspan font-style="italic">ŷ</tspan>', 8.5, INK, family=MATH)

# title under the stack
bb = iso(PS, PS, 0, oe, se)
text(oe[0], bb[1] + 4.2, "BERT encoder", 6.6, INK, "middle", 700)
text(oe[0], bb[1] + 7.0, "joint self-attention over words and facial tokens", 5.4, SLATE, "middle")

# encode: the sequence enters the stack at the height of the row
last = cent[-1]["right"]
y_in = last[1]
z_in = (oe[1] + PS * S30 * se - y_in) / se  # height on the left vertex at that screen y
lv = iso(0, PS, z_in, oe, se)
path(f"M{last[0] + 1.0:.2f},{y_in:.2f} L{lv[0] - 0.8:.2f},{y_in:.2f}", SLATE, 0.3)
verb((last[0] + lv[0]) / 2, y_in - 1.4, "encode")
