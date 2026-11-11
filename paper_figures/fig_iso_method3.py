"""Figure 1 (isometric, 2026-11-11): measured facial action tokens. Hand-built isometric SVG inside an HTML page, exported to vector PDF
by headless Chrome. Real data: frames 1/4/8 and the 33 x 8 Py-Feat measurement matrix of video_0006/0004. Palette: ink greys, one
deep-teal accent reserved for the facial evidence, pale sand for the text path. Output: figures/v7/method_overview.{html,pdf,png}"""
import base64
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

P = Path(__file__).resolve().parents[1]
F = P.parent / "results/outputs/full"
OUT = P / "figures/v7"; OUT.mkdir(parents=True, exist_ok=True)
KEY, HL = "video_0006/0004", 4
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
W_MM, H_MM = 172, 72

INK, SLATE, MUTE, MIST, TEAL, SAND = "#22262a", "#6b7178", "#9aa0a6", "#e7e5e0", "#1f6f78", "#b9a58a"


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
g = lambda c: au[c].to_numpy(float)
groups = [au[AU].to_numpy(float), au.face.to_numpy(float)[:, None], emo[EMO].to_numpy(float),
          np.stack([g("AU06") * g("AU12"), g("AU01") * g("AU04") * g("AU15"), g("AU04") * g("AU07"), g("AU09") * g("AU10"),
                    g("AU01") * g("AU02") * g("AU05")], 1)]
M = np.concatenate([gr / max(gr.max(), 1e-9) for gr in groups], 1)  # (8 frames, 33), display-scaled per group
assert M.shape == (8, 33)

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


def text(x, y, s, size=6.4, color=INK, anchor="start", weight=400, italic=False, family="Arial", rot=0, extra=""):
    st = f'font-family:{family};font-size:{size}pt;fill:{color};font-weight:{weight};' + ("font-style:italic;" if italic else "")
    tr = f' transform="rotate({rot} {x:.2f} {y:.2f})"' if rot else ""
    svg.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" style="{st}"{tr} {extra}>{s}</text>')


def path(d, color=SLATE, sw=0.28, dash=None, arrow=True, op=1.0):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    me = f' marker-end="url(#a{color[1:]})"' if arrow else ""
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" opacity="{op}"{da}{me}/>')


MATH = "Cambria, 'Cambria Math', 'Times New Roman', serif"

# ------------------------------------------------------------------ 1) frames as standing isometric cards with the real images
def frame_uri(i):
    im = Image.open(P / "figures/assets/video_0006__0004" / f"F{i}.jpg").crop((326, 60, 786, 760)).resize((300, 456))
    b = io.BytesIO(); im.save(b, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode(), im.size



exec(open(Path(__file__).with_name("fig_iso_layout3.py"), encoding="utf8").read())
exec(open(Path(__file__).with_name("fig_iso_encoder3.py"), encoding="utf8").read())

# ------------------------------------------------------------------ page
defs = "".join(f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="3.6" markerHeight="3.6" orient="auto-start-reverse">'
               f'<path d="M0,0.6 L9.4,5 L0,9.4 z" fill="{c}"/></marker>' for c in (SLATE, TEAL, MUTE))
import re
BODY = re.sub(r"(font-size[:=]\s*['\"]?)(\d+(?:\.\d+)?)pt", lambda m: f"{m.group(1)}{float(m.group(2)) * 0.35278:.3f}px", "".join(svg))  # 1 user unit = 1 mm
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
