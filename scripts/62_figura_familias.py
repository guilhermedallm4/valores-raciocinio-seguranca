"""Figura das 6 famílias para o artigo (largura de texto 12,2 cm).
  fig_familias.pdf  (a) deliberação x cumprimento danoso nas 5 famílias não-Qwen (exp. 12), com a reta ajustada
                        só nos Qwen3 (lei_artigo.json); (b) fração do dano de P recuperada por reversão parcial (exp. 13)
Uso: python scripts/62_figura_familias.py   (depois de scripts/57 e 59)
Saída: saidas/artigo/fig_familias.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

OUT = RAIZ / "saidas/artigo"
COR = {"A": "#52514e", "LIV": "#2a78d6", "DH": "#1baf7a", "P": "#eb6834", "DHD": "#eda100", "LAI": "#6250d6"}
MARC = {"A": "o", "LIV": "s", "DH": "D", "P": "^", "DHD": "P", "LAI": "v"}
ROT = {"A": "Original", "LIV": "Livro neutro", "DH": "DUDH", "P": "48 Leis", "DHD": "DUDH-delib", "LAI": "Laica-delib"}
FAM = {"qwen4b": "Qwen3 4B", "phi4": "Phi-4-r 14B", "gemma4": "Gemma 4 12B", "nemo": "Nemo 12B",
       "granite": "Granite 8B", "olmo2": "OLMo 2 13B"}
SELS = [("rev_q1", "1º¼"), ("rev_q2", "2º¼"), ("rev_q3", "3º¼"), ("rev_q4", "4º¼"), ("rev_gate", "gate"),
        ("rev_up", "up"), ("rev_down", "down"), ("rev_mlp", "MLP"), ("rev_attn", "atenç.")]
TINTA, TINTA2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})

e12 = json.loads((OUT / "exp12.json").read_text())
e13 = json.loads((OUT / "exp13.json").read_text())
lei = json.loads((OUT / "lei_artigo.json").read_text())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 1.8), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.35]})
xs = np.linspace(30, 100, 50)
a1.plot(xs, 100 * (lei["a"] + lei["b"] * xs / 100), ls="--", lw=1, color=TINTA2, zorder=1)
for c in COR:
    pts = [(v["delib"], v["danoso"]) for k, v in e12.items() if k.split("_")[0] == c]
    a1.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, marker=MARC[c], color=COR[c], edgecolor="white",
               linewidth=0.5, label=ROT[c], zorder=2)
a1.set_xlabel("delibera sobre dano (%)"); a1.set_ylabel("cumprimento danoso (%)")
a1.set_xlim(30, 102); a1.set_ylim(-2, 58)
a1.legend(fontsize=5.5, loc="upper right", handletextpad=0.2, labelspacing=0.25, borderaxespad=0)
a1.set_title("(a)", fontsize=7, loc="left")

M = np.array([[e13[f]["edicoes"].get(s, {}).get("recuperado", np.nan) for s, _ in SELS] for f in FAM])
cm = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f2f1ee", "#1d4ed8"])
a2.imshow(M, cmap=cm, norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=110), aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        a2.text(j, i, f"{v:.0f}".replace("-0", "0") if abs(v) >= 0.5 else "0", ha="center", va="center", fontsize=5.3, color="white" if v > 70 else TINTA)
a2.set_xticks(range(len(SELS)), [r for _, r in SELS], fontsize=5.8)
a2.set_yticks(range(len(FAM)), list(FAM.values()), fontsize=5.8)
a2.axvline(3.5, color="white", lw=2); a2.axvline(6.5, color="white", lw=2)
a2.tick_params(length=0)
for s in a2.spines.values():
    s.set_visible(False)
a2.set_title("(b) desfaz em P (% do dano recuperado)", fontsize=7, loc="left")
fig.savefig(OUT / "fig_familias.pdf")
print("ok", OUT / "fig_familias.pdf")
