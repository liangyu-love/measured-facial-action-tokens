"""Figure 1 (v12, 2026-11-12, tightened layout of v11): isometric method figure (style of v7) carried by one conflicting test clip, no lettered panels.
Standing frame cards (schematic face, not a video frame) -> standing measurement card with the real 8 x 33 Py-Feat matrix of the clip
-> one token cube per frame, in one row with the word-token cubes -> stacked BERT encoder -> prediction, compared on one axis with the
gold labels and with text-only BERT (mean over 10 seeds, tuned configurations).
Data: au_frames.csv, emo_frames.csv, clips.csv, tuned/{text,au_soft_emo}/*_test_pred.csv. Output: figures/v12/method_overview.*"""
import glob
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
F = P.parent / "results/outputs/full"
OUT = P / "figures/v12"; OUT.mkdir(parents=True, exist_ok=True)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
KEY = "aqgy5_0012/00008"
ZH = "我一看到帅哥就星星眼，控制不住啊。"
EN = ["“Whenever I see a handsome guy I go", "starry-eyed, I just can’t help it.”"]
W_MM, H_MM = 172, 77
INK, SLATE, MUTE, MIST, TEAL, SAND, SAND_D, RED = "#22262a", "#6b7178", "#9aa0a6", "#e7e5e0", "#1f6f78", "#b9a58a", "#8c7a5f", "#b5483a"
MATH = "Cambria, 'Cambria Math', 'Times New Roman', serif"


def hx(c):
    c = c.lstrip("#"); return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], float)


def mix(a, b, t):
    v = hx(a) * (1 - t) + hx(b) * t; return "#%02x%02x%02x" % tuple(int(round(x)) for x in v)


def shade(c, k):
    return mix(c, "#000000", k)


# ------------------------------------------------------------------ data
AU = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20",
      "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"]
EMO = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
au = pd.read_csv(F / "au_frames.csv"); au = au[au.key == KEY].sort_values("frame")
emo = pd.read_csv(F / "emo_frames.csv"); emo = emo[emo.key == KEY].sort_values("frame")
clip = pd.read_csv(F / "clips.csv", dtype={"video_id": str, "clip_id": str}).set_index("key").loc[KEY]
assert clip.text == ZH and len(au) == 8
g = lambda c: au[c].to_numpy(float)
groups = [au[AU].to_numpy(float), au.face.to_numpy(float)[:, None], emo[EMO].to_numpy(float),
          np.stack([g("AU06") * g("AU12"), g("AU01") * g("AU04") * g("AU15"), g("AU04") * g("AU07"), g("AU09") * g("AU10"),
                    g("AU01") * g("AU02") * g("AU05")], 1)]
M = np.concatenate([gr / max(gr.max(), 1e-9) for gr in groups], 1)  # (8 frames, 33), display-scaled per group
assert M.shape == (8, 33)
preds = lambda m: np.array([pd.read_csv(f).set_index("key").pred[KEY] for f in sorted(glob.glob(str(F / f"tuned/{m}/*_seed*_test_pred.csv")))])
p_text, p_soft = preds("text"), preds("au_soft_emo")
y, yT, yV = float(clip.label), float(clip.label_T), float(clip.label_V)
print(f"text {p_text.mean():+.2f} ({int(((p_text > 0) == (y > 0)).sum())}/10)  soft {p_soft.mean():+.2f} ({int(((p_soft > 0) == (y > 0)).sum())}/10)")
num = lambda v, d=2: f"{v:+.{d}f}".replace("-", "−")

# ------------------------------------------------------------------ isometric primitives (screen units: mm)
C30, S30 = np.cos(np.pi / 6), 0.5
svg = []


def iso(x, y, z, o, s):
    return o[0] + (x - y) * C30 * s, o[1] + (x + y) * S30 * s - z * s


def poly(pts, fill, stroke="none", sw=0.12, op=1.0):
    d = " ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
    svg.append(f'<polygon points="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" opacity="{op}"/>')


def box(x, y, z, dx, dy, dz, o, s, top, stroke="none", sw=0.1, k_left=0.22, k_right=0.12):
    P_ = lambda a, b, c: iso(a, b, c, o, s)
    poly([P_(x, y + dy, z), P_(x + dx, y + dy, z), P_(x + dx, y + dy, z + dz), P_(x, y + dy, z + dz)], shade(top, k_left), stroke, sw)
    poly([P_(x + dx, y, z), P_(x + dx, y + dy, z), P_(x + dx, y + dy, z + dz), P_(x + dx, y, z + dz)], shade(top, k_right), stroke, sw)
    poly([P_(x, y, z + dz), P_(x + dx, y, z + dz), P_(x + dx, y + dy, z + dz), P_(x, y + dy, z + dz)], top, stroke, sw)


def text(x, y, s, size=6.4, color=INK, anchor="start", weight=400, italic=False, family="Arial", extra=""):
    st = f'font-family:{family};font-size:{size}pt;fill:{color};font-weight:{weight};' + ("font-style:italic;" if italic else "")
    svg.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" style="{st}" {extra}>{s}</text>')


def path(d, color=SLATE, sw=0.28, dash=None, arrow=True, op=1.0):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    me = f' marker-end="url(#a{color[1:]})"' if arrow else ""
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" opacity="{op}"{da}{me}/>')


def line(x1, y1, x2, y2, color=MUTE, sw=0.18):
    svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>')


def sub(base, s, bold=True, color=None):
    c = f' fill="{color}"' if color else ""
    w = ' font-weight="700"' if bold else ""
    return (f'<tspan font-family="Cambria" font-style="italic"{w}{c}>{base}</tspan>'
            f'<tspan font-family="Cambria" font-style="italic" baseline-shift="sub" font-size="4.6pt"{c}>{s}</tspan>')


def verb(x, y, s, anchor="middle", color=SLATE):
    text(x, y, s, 5.6, color, anchor, italic=True)


def elbow(pts, color=SLATE, sw=0.3, r=1.6, arrow=True):
    d = f"M{pts[0][0]:.2f},{pts[0][1]:.2f}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        v1 = np.array([x1 - x0, y1 - y0]); v1 = v1 / (np.linalg.norm(v1) or 1)
        v2 = np.array([x2 - x1, y2 - y1]); v2 = v2 / (np.linalg.norm(v2) or 1)
        a_ = (x1 - v1[0] * r, y1 - v1[1] * r); b_ = (x1 + v2[0] * r, y1 + v2[1] * r)
        d += f" L{a_[0]:.2f},{a_[1]:.2f} Q{x1:.2f},{y1:.2f} {b_[0]:.2f},{b_[1]:.2f}"
    d += f" L{pts[-1][0]:.2f},{pts[-1][1]:.2f}"
    path(d, color, sw, arrow=arrow)


ROW_Y = 60.0  # screen y of the cube bases (sequence row)

# ================================================================== FACIAL BRANCH: standing frame cards with a schematic face
text(4.0, 5.6, "FACE", 5.4, TEAL, weight=700, extra='letter-spacing="0.35"')
of, sf = (5.0, 28.0), 1.0
cw, ch = 14.0, 15.0
Wpx, Hpx = 300, 321
SKIN, HAIR, CLOTH = "#fbfaf7", "#3d4349", "#d6d2c9"
BG = (f'<rect x="0" y="0" width="300" height="321" fill="#eceeed"/>'
        f'<path d="M18,321 Q28,270 106,254 L194,254 Q272,270 282,321 Z" fill="{CLOTH}" stroke="{INK}" stroke-width="4" stroke-linejoin="round"/>')
HEAD = (
        f'<path d="M126,196 L126,240 Q150,258 174,240 L174,196 Z" fill="{SKIN}" stroke="{INK}" stroke-width="4" stroke-linejoin="round"/>'
        f'<ellipse cx="86" cy="142" rx="9" ry="15" fill="{SKIN}" stroke="{INK}" stroke-width="3.5"/>'
        f'<ellipse cx="214" cy="142" rx="9" ry="15" fill="{SKIN}" stroke="{INK}" stroke-width="3.5"/>'
        f'<path d="M90,128 Q88,214 150,220 Q212,214 210,128 Q208,58 150,56 Q92,58 90,128 Z" fill="{SKIN}" stroke="{INK}" stroke-width="4"/>'
        f'<path d="M86,132 Q76,40 150,36 Q224,40 214,132 Q206,92 178,84 Q140,96 106,84 Q92,96 86,132 Z" fill="{HAIR}"/>'
        f'<path d="M104,116 L138,130 M196,116 L162,130" stroke="{TEAL}" stroke-width="7" stroke-linecap="round" fill="none"/>'
        f'<ellipse cx="122" cy="147" rx="9" ry="5.5" fill="{INK}"/><ellipse cx="178" cy="147" rx="9" ry="5.5" fill="{INK}"/>'
        f'<path d="M148,150 L142,174 L157,174" stroke="{INK}" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="M126,198 Q150,182 174,198" stroke="{INK}" stroke-width="4.5" fill="none" stroke-linecap="round"/>')
DET = (f'<rect x="70" y="30" width="160" height="198" fill="none" stroke="{TEAL}" stroke-width="3" stroke-dasharray="9 6"/>'
       + "".join(f'<path d="M{x},{y + dy * 22} L{x},{y} L{x + dx * 22},{y}" fill="none" stroke="{TEAL}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>'
                 for x, y, dx, dy in ((70, 30, 1, 1), (230, 30, -1, 1), (70, 228, 1, -1), (230, 228, -1, -1)))
       + "".join(f'<circle cx="{x}" cy="{y}" r="4" fill="#ffffff" stroke="{TEAL}" stroke-width="2.4"/>'
                 for x, y in ((104, 116), (138, 130), (196, 116), (162, 130), (126, 198), (174, 198))))
HG = '<g transform="translate(150 152) scale(1.2) translate(-150 -138)">'
NF = 4
for k in range(NF):
    y0 = -2.8 * (NF - 1 - k); x0 = 0.0
    box(x0, y0 - 0.32, 0, cw, 0.32, ch, of, sf, "#f6f5f2", stroke="#c9c6bf", sw=0.1, k_left=0.05, k_right=0.12)
    front = [iso(x0, y0, 0, of, sf), iso(x0 + cw, y0, 0, of, sf), iso(x0 + cw, y0, ch, of, sf), iso(x0, y0, ch, of, sf)]
    poly(front, "#ffffff", "#c9c6bf", 0.12)
    a, b = C30 * sf * cw / Wpx, S30 * sf * cw / Wpx
    d = sf * ch / Hpx
    e = of[0] + (x0 - y0) * C30 * sf; f = of[1] + (x0 + y0) * S30 * sf - sf * ch
    last_ = k == NF - 1
    svg.append(f'<g transform="matrix({a:.6f} {b:.6f} 0 {d:.6f} {e:.4f} {f:.4f})" opacity="{0.30 + 0.70 * k / (NF - 1):.2f}">{BG}{HG}{HEAD}{DET if last_ else ""}</g></g>')
    poly(front, "none", SLATE if last_ else "#b9b6ae", 0.24 if last_ else 0.12)
bk = iso(cw, -2.8 * (NF - 1), 0, of, sf); fr = iso(cw, 0, 0, of, sf)
path(f"M{fr[0] + 1.6:.2f},{fr[1] + 1.2:.2f} L{bk[0] + 1.6:.2f},{bk[1] + 1.2:.2f}", MUTE, 0.2)
text((bk[0] + fr[0]) / 2 + 3.2, (bk[1] + fr[1]) / 2 + 2.6, "time", 4.8, MUTE, "middle", italic=True)
text(4.0, 40.6, "8 sampled frames,", 5.6, SLATE)
text(4.0, 44.0, 'the face looks <tspan font-weight="700" fill="' + TEAL + '">negative</tspan>', 5.6, SLATE)

# measurement card: a standing sheet in the same plane as the frames; rows = frames, columns = 33 measurements
oc, sc = (36.0, 25.5), 1.42
gap_after = {19: 0.55, 20: 0.55, 27: 0.55}
cxs, x = [], 0.0
for r in range(33):
    cxs.append(x); x += 0.86 + gap_after.get(r, 0.0)
CW_, RH = x - 0.06, 1.25
box(-0.5, -0.32, -0.5, CW_ + 1.0, 0.32, 8 * RH + 1.0, oc, sc, "#f6f5f2", stroke="#c9c6bf", sw=0.1, k_left=0.05, k_right=0.12)
poly([iso(-0.5, 0, -0.5, oc, sc), iso(CW_ + 0.5, 0, -0.5, oc, sc), iso(CW_ + 0.5, 0, 8 * RH + 0.5, oc, sc), iso(-0.5, 0, 8 * RH + 0.5, oc, sc)],
     "#ffffff", "#c9c6bf", 0.1)
for fidx in range(8):
    zt_ = (8 - fidx) * RH  # frame 1 on top
    for r in range(33):
        xa = cxs[r]
        poly([iso(xa, 0, zt_ - RH + 0.12, oc, sc), iso(xa + 0.78, 0, zt_ - RH + 0.12, oc, sc),
              iso(xa + 0.78, 0, zt_ - 0.08, oc, sc), iso(xa, 0, zt_ - 0.08, oc, sc)], mix("#f1f1ee", TEAL, 0.04 + 0.96 * M[fidx, r] ** 0.85))
for name, a, b in [("20 AU intensities", 0, 20), ("7 emotions", 21, 28), ("5 EMFACS", 28, 33)]:
    xa, xb = cxs[a], cxs[b - 1] + 0.78
    p0, p1 = iso(xa, 0, -1.1, oc, sc), iso(xb, 0, -1.1, oc, sc)
    line(p0[0], p0[1], p1[0], p1[1])
    pm = iso((xa + xb) / 2, 0, -1.1 - (2.4 if name.startswith("7") else 0.0), oc, sc)
    svg.append(f'<text x="{pm[0] - 0.4:.2f}" y="{pm[1] + 2.9:.2f}" text-anchor="middle" transform="rotate(30 {pm[0] - 0.4:.2f} {pm[1] + 2.9:.2f})" '
               f'style="font-family:Arial;font-size:5.2pt;fill:{SLATE};">{name}</text>')
text(36.0, 5.6, "Py-Feat measurements per frame,  " + sub("a", "f") + '<tspan font-family="Cambria"> ∈ ℝ</tspan>'
     '<tspan font-family="Cambria" baseline-shift="super" font-size="4.6pt">33</tspan>', 6.0, INK)
# highlighted columns with flat, readable call-outs (clip means)
for r, lab, val, yt in [(2, "AU04 brow lowerer", g("AU04").mean(), 8.2), (9, "AU12 lip-corner puller", g("AU12").mean(), 12.3),
                        (22, "happy", emo.Happy.mean(), 19.6), (23, "sad", emo.Sad.mean(), 23.2),
                        (30, "AU04 × AU07", (g("AU04") * g("AU07")).mean(), 27.0)]:
    xa = cxs[r]
    poly([iso(xa - 0.06, 0, -0.05, oc, sc), iso(xa + 0.84, 0, -0.05, oc, sc), iso(xa + 0.84, 0, 8 * RH + 0.05, oc, sc), iso(xa - 0.06, 0, 8 * RH + 0.05, oc, sc)],
         "none", INK, 0.22)
    top = iso(xa + 0.39, 0, 8 * RH + 0.05, oc, sc)
    line(top[0], top[1], top[0], yt + 0.5, INK, 0.2)
    text(top[0] + 0.9, yt + 1.3, f'{lab} <tspan font-weight="700" fill="{TEAL}">{val:.2f}</tspan>', 5.2, INK)

# frames -> card (Py-Feat)
fa = iso(cw + 0.6, -2.8 * (NF - 1), ch * 0.62, of, sf); ca = iso(-1.0, 0, 8 * RH * 0.5, oc, sc)
path(f"M{fa[0]:.2f},{fa[1]:.2f} L{ca[0]:.2f},{ca[1]:.2f}", SLATE, 0.3)
verb((fa[0] + ca[0]) / 2 - 0.4, min(fa[1], ca[1]) - 1.6, "Py-Feat")

# ================================================================== SEQUENCE ROW: one cube per token
ot, st_ = (63.0, ROW_Y), 1.2
toks = [("[CLS]", "w"), ("w", "w", "1"), ("⋯", "gap"), ("w", "w", "n"), ("[SEP]", "w"),
        ("e", "e", "1"), ("e", "e", "2"), ("⋯", "gap"), ("e", "e", "8")]
cube, step = 2.25, 2.95
cent = []
for k, tk in enumerate(toks):
    lab, kind = tk[0], tk[1]; s_ = tk[2] if len(tk) > 2 else None
    x0, y0 = k * step, -k * step
    top_c = iso(x0 + cube / 2, y0 + cube / 2, cube, ot, st_)
    base = iso(x0 + cube, y0 + cube, 0, ot, st_)
    if kind == "gap":
        text(top_c[0], top_c[1] + 2.5, "⋯", 7, MUTE, "middle")
    else:
        top = "#efe8dc" if kind == "w" else mix("#ffffff", TEAL, 0.78)
        box(x0, y0, 0, cube, cube, cube, ot, st_, top, stroke=("#a8977c" if kind == "w" else "#16545b"), sw=0.08, k_left=0.16, k_right=0.08)
        if s_ is None:
            text(base[0], base[1] + 3.0, lab, 4.9, INK, "middle")
        else:
            text(base[0], base[1] + 3.0, sub(lab, s_, bold=(kind == "e"), color=(TEAL if kind == "e" else INK)), 6.0, INK, "middle")
    cent.append(dict(top=top_c, base=base, kind=kind, left=iso(x0, y0 + cube, cube / 2, ot, st_), right=iso(x0 + cube, y0, cube / 2, ot, st_)))
W_ = [c for c in cent if c["kind"] == "w"]; E_ = [c for c in cent if c["kind"] == "e"]
text((W_[0]["base"][0] + W_[-1]["base"][0]) / 2, ROW_Y + 9.6, "word tokens", 5.4, SLATE, "middle")
text((E_[0]["base"][0] + E_[-1]["base"][0]) / 2, ROW_Y + 9.6, "facial action tokens", 5.4, TEAL, "middle", weight=700)

# ================================================================== TEXT BRANCH (left, in line with the word tokens)
cy = W_[0]["left"][1]
svg.append(f'<rect x="3.0" y="{cy - 10.4:.2f}" width="43.0" height="22.6" rx="1.6" fill="{mix("#ffffff", SAND, 0.16)}" stroke="{mix("#ffffff", SAND, 0.75)}" stroke-width="0.18"/>')
text(5.0, cy - 6.4, "TRANSCRIPT", 5.4, SAND_D, weight=700, extra='letter-spacing="0.35"')
text(5.0, cy - 1.8, ZH, 6.0, INK, family="'Microsoft YaHei'")
for i, t in enumerate(EN):
    text(5.0, cy + 2.4 + 3.1 * i, t, 5.3, SLATE, italic=True)
text(5.0, cy + 9.8, 'the words sound <tspan font-weight="700" fill="' + SAND_D + '">positive</tspan>', 5.6, SLATE)
path(f"M47.2,{cy:.2f} L{W_[0]['left'][0] - 1.0:.2f},{cy:.2f}", SLATE, 0.3)
verb((47.2 + W_[0]['left'][0]) / 2 - 0.4, cy - 1.4, "tokenize")

# ================================================================== projection: card -> bracket over the facial tokens
r4 = iso(CW_ + 0.5, 0, 4 * RH, oc, sc)  # middle of the right edge of the card
by = min(c["top"][1] for c in E_) - 4.6
ex0, ex1 = E_[0]["top"][0], E_[-1]["top"][0]; emx = (ex0 + ex1) / 2
elbow([(r4[0] + 0.6, r4[1]), (emx, r4[1]), (emx, by)], TEAL, 0.32, arrow=False)
line(ex0, by, ex1, by, TEAL, 0.32)
for c in E_:
    path(f"M{c['top'][0]:.2f},{by:.2f} L{c['top'][0]:.2f},{c['top'][1] - 1.3:.2f}", TEAL, 0.32)
text((r4[0] + emx) / 2 + 0.6, r4[1] - 1.5, "one token per frame", 5.6, TEAL, "middle", italic=True)
svg.append('<g transform="translate(%.2f 0)">' % (-(emx + 2.0) + r4[0] + 4.0))
text(emx + 2.0, r4[1] + 4.8, sub("e", "f", color=TEAL) + '<tspan font-family="Cambria"> = LN(</tspan>' + sub("W", "a", color=TEAL)
     + sub("a", "f", color=TEAL) + '<tspan font-family="Cambria"> + </tspan>' + sub("b", "a", color=TEAL)
     + '<tspan font-family="Cambria">) + </tspan>' + sub("q", "f", color=TEAL), 6.2, TEAL)
svg.append("</g>")

# ================================================================== ENCODER: light plate stack
oe, se, PS = (139.0, ROW_Y - 4.6), 0.86, 16.0
LAYERS = [(0.0, "layer 1"), (3.2, "layer 2"), (9.6, "layer 12")]
for z_, _ in LAYERS:
    box(0, 0, z_, PS, PS, 0.45, oe, se, "#f4f3ef" if z_ < 8 else "#fbfaf8", stroke="#a9a59c", sw=0.09, k_left=0.10, k_right=0.05)
    if z_ == 3.2:
        for vx, vy in ((0, PS), (PS, 0)):
            for kk in range(3):
                p_ = iso(vx, vy, 5.3 + kk * 1.1, oe, se)
                svg.append(f'<circle cx="{p_[0]:.2f}" cy="{p_[1]:.2f}" r="0.26" fill="{MUTE}"/>')
ztop = LAYERS[-1][0] + 0.45
rb = iso(PS, 0, 0.0, oe, se); bx_ = rb[0] + 2.0
for z_, lab in LAYERS:
    rv_ = iso(PS, 0, z_ + 0.22, oe, se)
    line(rv_[0] + 0.5, rv_[1], bx_, rv_[1])
    text(bx_ + 0.6, rv_[1] + 0.8, lab, 5.2, SLATE)
text(bx_ + 2.6, iso(PS, 0, 6.4, oe, se)[1] + 1.4, "⋮", 6.5, MUTE, "middle")
tiles = []
for i_, kind in enumerate(["cls", "w", "w", "e", "e", "e"]):
    tsz = 1.55
    tx, ty = PS / 2 + (i_ - 2.5) * 2.05 - tsz / 2, PS / 2 - (i_ - 2.5) * 2.05 - tsz / 2
    fill = {"cls": "#d9ccb6", "w": "#efe8dc", "e": mix("#ffffff", TEAL, 0.7)}[kind]
    poly([iso(tx, ty, ztop, oe, se), iso(tx + tsz, ty, ztop, oe, se), iso(tx + tsz, ty + tsz, ztop, oe, se), iso(tx, ty + tsz, ztop, oe, se)],
         fill, "#a8977c" if kind != "e" else "#16545b", 0.08)
    tiles.append(iso(tx + tsz / 2, ty + tsz / 2, ztop, oe, se))
for i_, j_, col in [(1, 3, TEAL), (2, 4, TEAL), (1, 5, TEAL), (2, 3, TEAL), (1, 2, MUTE), (4, 5, TEAL)]:
    a_, b_ = tiles[i_], tiles[j_]
    mxp = ((a_[0] + b_[0]) / 2, (a_[1] + b_[1]) / 2 - 1.2 - 1.0 * abs(j_ - i_))
    path(f"M{a_[0]:.2f},{a_[1]:.2f} Q{mxp[0]:.2f},{mxp[1]:.2f} {b_[0]:.2f},{b_[1]:.2f}", col, 0.22, arrow=False, op=0.9)
bb = iso(PS, PS, 0, oe, se)
text(oe[0], bb[1] + 3.6, "BERT encoder", 6.6, INK, "middle", 700)
text(oe[0], bb[1] + 6.5, "joint self-attention over words and facial tokens", 5.4, SLATE, "middle")
last = cent[-1]["right"]; y_in = last[1]
z_in = (oe[1] + PS * S30 * se - y_in) / se
lv = iso(0, PS, z_in, oe, se)
path(f"M{last[0] + 1.0:.2f},{y_in:.2f} L{lv[0] - 0.8:.2f},{y_in:.2f}", SLATE, 0.3)
verb((last[0] + lv[0]) / 2, y_in - 1.4, "encode")

# ================================================================== PREDICTION vs LABELS on one axis (top right)
ax0, ax1, top, bot = 136.0, 165.5, 5.6, 34.0
X = lambda v: (ax0 + ax1) / 2 + v * (ax1 - ax0) / 2
svg.append(f'<rect x="{ax0}" y="{top}" width="{X(0) - ax0:.2f}" height="{bot - top}" fill="{mix("#ffffff", TEAL, 0.10)}"/>')
svg.append(f'<rect x="{X(0):.2f}" y="{top}" width="{ax1 - X(0):.2f}" height="{bot - top}" fill="{mix("#ffffff", SAND, 0.22)}"/>')
line(X(0), top, X(0), bot, SLATE, 0.22)
text(ax0 + 0.7, top - 1.0, "negative", 4.8, TEAL); text(ax1 - 0.7, top - 1.0, "positive", 4.8, SAND_D, "end")
for v, lab in ((-1, "−1"), (0, "0"), (1, "+1")):
    line(X(v), bot, X(v), bot + 0.7, SLATE, 0.2); text(X(v), bot + 3.0, lab, 4.6, SLATE, "middle")
xl = ax0 - 6.4


def row(yy, label, v, color, kind, mark=None):
    bold = kind == "pred"
    text(xl, yy + 0.75, label, 5.2 if bold else 5.0, INK if bold else SLATE, "end", weight=700 if bold else 400)
    line(X(0), yy, X(v), yy, color, 0.3)
    if kind == "gold":
        r = 0.95; poly([(X(v), yy - r), (X(v) + r, yy), (X(v), yy + r), (X(v) - r, yy)], color)
    elif kind == "label":
        svg.append(f'<circle cx="{X(v):.2f}" cy="{yy:.2f}" r="0.85" fill="#ffffff" stroke="{color}" stroke-width="0.36"/>')
    else:
        svg.append(f'<circle cx="{X(v):.2f}" cy="{yy:.2f}" r="1.05" fill="{color}"/>')
    side = -1 if v < 0 else 1
    text(X(v) + side * 1.7, yy + 0.75, num(v, 2 if bold else 1), 4.9, color, "end" if side < 0 else "start", weight=700)
    if mark:
        text(ax1 + 1.2, yy + 1.0, mark[0], 6.6, mark[1], weight=700)


row(9.0, "text label", yT, SAND_D, "label")
row(13.6, "visual label", yV, TEAL, "label")
row(18.2, "overall label", y, INK, "gold")
line(ax0, 21.6, ax1, 21.6, "#ffffff", 0.5)
row(25.4, "text-only BERT", float(p_text.mean()), SAND_D, "pred", ("✗", RED))
row(30.6, "with facial tokens", float(p_soft.mean()), TEAL, "pred", ("✓", TEAL))
# prediction from the [CLS] tile up to the comparison
cl = tiles[0]
path(f"M{cl[0]:.2f},{cl[1] - 0.6:.2f} L{cl[0]:.2f},{bot + 5.4:.2f}", TEAL, 0.3)
text(cl[0] + 1.2, (cl[1] + bot + 5.4) / 2 + 0.6, "[CLS] → linear", 5.2, SLATE, italic=True)
text(cl[0] - 0.2, bot + 4.2, '<tspan font-style="italic">ŷ</tspan>', 7.4, TEAL, "middle", family=MATH)

# ------------------------------------------------------------------ page
defs = "".join(f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="3.6" markerHeight="3.6" orient="auto-start-reverse">'
               f'<path d="M0,0.6 L9.4,5 L0,9.4 z" fill="{c}"/></marker>' for c in (SLATE, TEAL, MUTE))
BODY = re.sub(r"(font-size[:=]\s*['\"]?)(\d+(?:\.\d+)?)pt", lambda m: f"{m.group(1)}{float(m.group(2)) * 0.35278:.3f}px", "".join(svg))
HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>Figure 1</title><style>
@page {{ size: {W_MM}mm {H_MM}mm; margin: 0; }}
html, body {{ margin: 0; width: {W_MM}mm; height: {H_MM}mm; background: #ffffff; overflow: hidden; }}
svg {{ display: block; width: {W_MM}mm; height: {H_MM}mm; }}
</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_MM} {H_MM}"><defs>{defs}</defs>
{BODY}
</svg></body></html>"""
hp = OUT / "method_overview.html"; hp.write_text(HTML, encoding="utf8")
url = hp.resolve().as_uri()
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT / 'method_overview.pdf'}", url],
               check=True, capture_output=True)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=5",
                f"--window-size={round(W_MM * 3.7795)},{round(H_MM * 3.7795)}", f"--screenshot={OUT / 'method_overview.png'}", url],
               check=True, capture_output=True)
print("saved", OUT)
