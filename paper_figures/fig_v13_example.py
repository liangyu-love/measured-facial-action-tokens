"""Figure 1 (v13, 2026-11-17): three ways to give the face of one conflicting clip to a language model; primitives from v12: isometric method figure (style of v7) carried by one conflicting test clip, no lettered panels.
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
OUT = P / "figures/v13"; OUT.mkdir(parents=True, exist_ok=True)
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



# ================================================================== (v13 Figure 1) layout
W_MM, H_MM = 172, 72
DESC = "eyebrows raised slightly, eyes narrowed slightly, lips slightly parted"   # zero-shot Qwen2.5-VL-7B, outputs/full/desc.jsonl

# ---------------- left: the clip
text(4.0, 5.4, "ONE CONFLICTING CLIP", 5.2, INK, weight=700, extra='letter-spacing="0.3"')
of, sf = (7.0, 27.5), 1.0
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

text(4.0, 38.6, 'face: <tspan font-weight="700" fill="' + TEAL + '">negative</tspan>  (visual label ' + num(yV, 1) + ')', 5.2, SLATE)
cy0 = 43.0
svg.append(f'<rect x="3.0" y="{cy0:.2f}" width="37.5" height="22.5" rx="1.6" fill="{mix("#ffffff", SAND, 0.16)}" stroke="{mix("#ffffff", SAND, 0.75)}" stroke-width="0.18"/>')
text(5.0, cy0 + 4.2, "TRANSCRIPT", 5.0, SAND_D, weight=700, extra='letter-spacing="0.3"')
text(5.0, cy0 + 8.6, ZH, 5.3, INK, family="'Microsoft YaHei'")
for i, t in enumerate(EN):
    text(5.0, cy0 + 12.4 + 2.9 * i, t, 4.8, SLATE, italic=True)
text(5.0, cy0 + 20.2, 'words: <tspan font-weight="700" fill="' + SAND_D + '">positive</tspan>  (text label ' + num(yT, 1) + ')', 5.0, SLATE)

# ---------------- middle: three lanes
LX0, LX1 = 47.0, 122.0
lanes = [("DESCRIBE", 7.0, MUTE), ("WATCH", 27.0, MUTE), ("MEASURE  (ours)", 47.0, TEAL)]
for name, y0, col in lanes:
    ours = col == TEAL
    svg.append(f'<rect x="{LX0:.2f}" y="{y0:.2f}" width="{LX1 - LX0:.2f}" height="17.5" rx="1.6" '
               f'fill="{mix("#ffffff", TEAL, 0.07) if ours else "#f7f7f5"}" stroke="{TEAL if ours else "#d3d1cb"}" stroke-width="{0.32 if ours else 0.16}"/>')
    text(LX0 + 2.0, y0 + 4.2, name, 5.2, TEAL if ours else SLATE, weight=700, extra='letter-spacing="0.3"')


def proc(x, y, w, label, sub_, col=SLATE):
    svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="7.4" rx="1.1" fill="#ffffff" stroke="{col}" stroke-width="0.22"/>')
    text(x + w / 2, y + 3.3, label, 5.0, INK, "middle", 700)
    text(x + w / 2, y + 6.2, sub_, 4.3, SLATE, "middle")


# DESCRIBE: MLLM / template -> words appended to the transcript
y0 = 7.0
proc(LX0 + 2.0, y0 + 6.6, 20.0, "MLLM or rules", "caption the frames")
path(f"M{LX0 + 22.6:.2f},{y0 + 10.3:.2f} L{LX0 + 26.0:.2f},{y0 + 10.3:.2f}", SLATE, 0.28)
svg.append(f'<rect x="{LX0 + 26.6:.2f}" y="{y0 + 6.0:.2f}" width="46.0" height="8.6" rx="1.0" fill="#ffffff" stroke="#c9c6bf" stroke-width="0.16"/>')
text(LX0 + 28.0, y0 + 9.4, "“" + DESC.split(", lips")[0] + ",", 4.5, INK, italic=True)
text(LX0 + 28.0, y0 + 12.6, " lips slightly parted …”", 4.5, INK, italic=True)
text(LX1 - 2.0, y0 + 4.2, "words: lossy, can contradict the face", 4.5, SLATE, "end", italic=True)

# WATCH: vision encoder -> hundreds of visual tokens
y0 = 27.0
proc(LX0 + 2.0, y0 + 6.6, 20.0, "vision encoder", "of a multimodal LLM")
path(f"M{LX0 + 22.6:.2f},{y0 + 10.3:.2f} L{LX0 + 26.0:.2f},{y0 + 10.3:.2f}", SLATE, 0.28)
ow, sw_ = (LX0 + 29.0, y0 + 13.4), 0.62
for r in range(4):
    for c_ in range(18):
        box(c_ * 2.1, -c_ * 2.1, r * 1.9, 1.6, 1.6, 1.6, ow, sw_, "#dfe3e4", stroke="#9aa6a8", sw=0.04, k_left=0.12, k_right=0.06)
text(LX0 + 29.0 + 18 * 2 * 2.1 * C30 * sw_ + 1.5, y0 + 12.4, "…", 6, MUTE)
text(LX1 - 2.0, y0 + 4.2, "616 visual tokens per clip, ~100 min training", 4.5, SLATE, "end", italic=True)

# MEASURE: Py-Feat -> 8 measured tokens
y0 = 47.0
proc(LX0 + 2.0, y0 + 6.6, 20.0, "Py-Feat", "AUs, emotions, AU combos", TEAL)
path(f"M{LX0 + 22.6:.2f},{y0 + 10.3:.2f} L{LX0 + 26.0:.2f},{y0 + 10.3:.2f}", TEAL, 0.3)
om, sm = (LX0 + 30.5, y0 + 10.4), 0.9
for c_ in range(8):
    box(c_ * 2.6, -c_ * 2.6, 0, 2.4, 2.4, 2.4, om, sm, mix("#ffffff", TEAL, 0.78), stroke="#16545b", sw=0.07, k_left=0.16, k_right=0.08)
text(LX0 + 46.5, y0 + 16.4, 'e.g. AU04 brow lowerer <tspan font-weight="700" fill="' + TEAL + '">' + f"{g('AU04').mean():.2f}" + '</tspan>,  AU12 smile <tspan font-weight="700" fill="'
     + TEAL + '">' + f"{g('AU12').mean():.2f}" + '</tspan>', 4.4, SLATE, "middle")
text(LX1 - 2.0, y0 + 4.2, "8 tokens, ~5 min training, named facial actions", 4.5, TEAL, "end", italic=True, weight=700)

# clip -> lanes
for _, y0, col in lanes:
    elbow([(41.6, 35.0), (44.0, 35.0), (44.0, y0 + 10.3), (LX0 + 1.6, y0 + 10.3)], TEAL if col == TEAL else MUTE, 0.26 if col == TEAL else 0.2, r=1.0)

# ---------------- right: language model + prediction
ol, sl, PL = (147.0, 15.0), 0.62, 15.0
for z_ in (0.0, 2.6, 5.2):
    box(0, 0, z_, PL, PL, 0.4, ol, sl, "#f4f3ef", stroke="#a9a59c", sw=0.08, k_left=0.10, k_right=0.05)
bbl = iso(PL, PL, 0, ol, sl)
text(ol[0], bbl[1] + 3.2, "language model", 5.6, INK, "middle", 700)
text(ol[0], bbl[1] + 5.9, "BERT or decoder-only LLM", 4.6, SLATE, "middle")
lin = iso(0, PL, 2.8, ol, sl)
for _, y0, col in lanes:
    elbow([(LX1, y0 + 10.3), (LX1 + 3.0, y0 + 10.3), (LX1 + 3.0, lin[1]), (lin[0] - 0.8, lin[1])], TEAL if col == TEAL else MUTE,
          0.26 if col == TEAL else 0.2, r=1.0, arrow=(col == TEAL))
text(LX1 + 4.2, lin[1] - 2.4, "+ transcript", 4.5, SLATE, italic=True)

ax0, ax1, top, bot = 145.0, 168.0, 44.0, 64.0
X = lambda v: (ax0 + ax1) / 2 + v * (ax1 - ax0) / 2
svg.append(f'<rect x="{ax0}" y="{top}" width="{X(0) - ax0:.2f}" height="{bot - top}" fill="{mix("#ffffff", TEAL, 0.10)}"/>')
svg.append(f'<rect x="{X(0):.2f}" y="{top}" width="{ax1 - X(0):.2f}" height="{bot - top}" fill="{mix("#ffffff", SAND, 0.22)}"/>')
line(X(0), top, X(0), bot, SLATE, 0.22)
text(ax0 + 0.7, top - 1.0, "negative", 4.6, TEAL); text(ax1 - 0.7, top - 1.0, "positive", 4.6, SAND_D, "end")
for v, lab in ((-1, "−1"), (0, "0"), (1, "+1")):
    line(X(v), bot, X(v), bot + 0.7, SLATE, 0.2); text(X(v), bot + 3.0, lab, 4.4, SLATE, "middle")


def row(yy, label, v, color, kind, mark=None):
    text(ax0 - 5.5, yy + 0.75, label, 4.8, INK if kind == "pred" else SLATE, "end", weight=700 if kind == "pred" else 400)
    line(X(0), yy, X(v), yy, color, 0.3)
    if kind == "gold":
        r = 0.9; poly([(X(v), yy - r), (X(v) + r, yy), (X(v), yy + r), (X(v) - r, yy)], color)
    else:
        svg.append(f'<circle cx="{X(v):.2f}" cy="{yy:.2f}" r="1.0" fill="{color}"/>')
    side = -1 if v < 0 else 1
    text(X(v) + side * 1.6, yy + 0.75, num(v, 2 if kind == "pred" else 1), 4.6, color, "end" if side < 0 else "start", weight=700)
    if mark:
        text(ax1 + 0.8, yy + 1.0, mark[0], 6.0, mark[1], weight=700)


row(48.6, "overall label", y, INK, "gold")
row(54.2, "BERT, text", float(p_text.mean()), SAND_D, "pred", ("✗", RED))
row(59.6, "+ facial tokens", float(p_soft.mean()), TEAL, "pred", ("✓", TEAL))
pv = iso(PL, PL, 0, ol, sl)

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
hp = OUT / "fig1_example.html"; hp.write_text(HTML, encoding="utf8")
url = hp.resolve().as_uri()
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT / 'fig1_example.pdf'}", url],
               check=True, capture_output=True)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=5",
                f"--window-size={round(W_MM * 3.7795)},{round(H_MM * 3.7795)}", f"--screenshot={OUT / 'fig1_example.png'}", url],
               check=True, capture_output=True)
print("saved", OUT)
