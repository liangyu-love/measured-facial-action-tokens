"""Controls of the analysis figure, drawn in the same dashed language and tied to where they act. Executed after fig_v12_analysis_layout.py."""
# ruff: noqa: F821

DOT = "0.35 0.7"
kx, kw = 113.0, 52.0

# capacity control: acts on the visual reading inside the discrepancy token
cy_ = 43.6
rbox(kx, cy_, kw, 7.4, SLATE, "#ffffff", True)
text(kx + 1.8, cy_ + 3.0, "capacity control", 5.6, INK, weight=700)
text(kx + 1.8, cy_ + 5.7, '<tspan font-family="Cambria" font-style="italic">ŷ</tspan>'
     '<tspan font-family="Cambria" font-style="italic" baseline-shift="sub" font-size="4.6pt">V</tspan>'
     ' ← 0, or another clip’s reading', 5.2, SLATE)
# dotted leader to the visual-reading term of the discrepancy box
path(f"M{kx:.2f},{cy_ + 4.0:.2f} L{cx0 + cwid + 0.6:.2f},{cy_ + 4.0:.2f}", SLATE, 0.26, dash=DOT)
svg.append(f'<circle cx="{cx0 + cwid:.2f}" cy="{cy_ + 4.0:.2f}" r="0.55" fill="#ffffff" stroke="{SLATE}" stroke-width="0.26"/>')

# oracle reference: evaluated against the regression output on strict clips
oy_ = 53.4
rbox(kx, oy_, kw, 7.4, SLATE, "#ffffff", True)
text(kx + 1.8, oy_ + 3.0, "oracle reference", 5.6, INK, weight=700)
text(kx + 1.8, oy_ + 5.7, "linear regression on gold " + sub("y", "T", bold=False) + ", " + sub("y", "V", bold=False)
     + " (+ " + sub("y", "A", bold=False) + "), compared with ŷ", 5.2, SLATE)
# dotted leader up to the regression head (right edge of the head column)
rh_right = hx0 + 36.0
path(f"M{kx + kw:.2f},{oy_ + 4.0:.2f} L{kx + kw + 1.2:.2f},{oy_ + 4.0:.2f} L{kx + kw + 1.2:.2f},{ROW_Y - 17.5 + 3.3:.2f} "
     f"L{rh_right + 0.6:.2f},{ROW_Y - 17.5 + 3.3:.2f}", SLATE, 0.26, dash=DOT)
