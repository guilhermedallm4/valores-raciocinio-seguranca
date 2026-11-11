"""Figuras vetoriais do artigo ESANN (largura de texto 12,2 cm), a partir dos resultados já julgados.
  fig_escala_lei.pdf     (a) danosos atendidos x tamanho por condição; (b) deliberação no raciocínio x danosos atendidos
  fig_jb_repr.pdf        (a) jailbreak por disfarce e condição; (b) norma da direção de recusa por camada (Qwen3-4B)
Uso: python scripts/43_figuras_artigo.py   (depois de scripts/37, 38, 39)
Saídas: saidas/artigo/fig_escala_lei.pdf, saidas/artigo/fig_jb_repr.pdf, saidas/artigo/lei_pontos.json
"""
import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_s = importlib.util.spec_from_file_location("l38", Path(__file__).parent / "38_lei_quantitativa.py")
l38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(l38)
S, OUT = RAIZ / "saidas", RAIZ / "saidas/artigo"
TAM = {"1.7b": 1.7, "4b": 4.0, "8b": 8.2, "14b": 14.8, "32b": 32.8}
# paleta categórica de referência (ordem fixa) + cinza neutro para o original; marcadores = codificação secundária
FORA_DO_ESCOPO = ("C", "N", "DEC", "B")   # condições de outro estudo, fora do escopo do artigo
COR = {"A": "#52514e", "P": "#eb6834", "DH": "#1baf7a", "LAI": "#6250d6", "DHD": "#eda100"}
MARC = {"A": "o", "P": "^", "DH": "D", "LAI": "v", "DHD": "P"}
ROT = {"A": "Original", "P": "48 Leis", "DH": "DUDH", "LAI": "Laica-delib", "DHD": "DUDH-delib"}
TINTA, TINTA2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
                     "legend.frameon": False, "pdf.fonttype": 42})


def cond_de(stem):
    m = re.match(r"(?:[a-z]+_)?([A-Z]+)", stem)
    return m.group(1) if m else None


def ponto_juiz(f):
    c = OUT / "delib_juiz" / f"{f.parent.name}__{f.stem}.json"
    if not c.exists():
        return None
    j = json.loads(c.read_text())
    its = [x for x in ler_jsonl(f) if x.get("juiz2") and x["rotulo"].startswith("danoso") and j.get(x["id"]) is not None]
    if len(its) < 50:
        return None
    return (sum(j[x["id"]] for x in its) / len(its),
            sum(x["juiz2"]["classe"] in ("CUMPRE", "PARCIAL") for x in its) / len(its))


def lei_de(pontos):
    """regressão linear cumpre ~ delibera, Pearson e Spearman sobre os modelos plotados"""
    xs, ys = [p["delibera"] for p in pontos], [p["cumpre"] for p in pontos]
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)); sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys); r = sxy / (sxx * syy) ** 0.5

    def rank(v):
        o = sorted(range(n), key=lambda i: v[i]); rk = [0] * n
        for pos, i in enumerate(o):
            rk[i] = pos
        return rk
    rx, ry = rank(xs), rank(ys); m = (n - 1) / 2
    rho = sum((p - m) * (q - m) for p, q in zip(rx, ry)) / (sum((p - m) ** 2 for p in rx) * sum((q - m) ** 2 for q in ry)) ** 0.5
    return {"n_modelos": n, "R2": r * r, "pearson": r, "spearman": rho, "b": sxy / sxx, "a": my - sxy / sxx * mx,
            "n_ajustados": sum(not p["modelo"].split("/")[1].startswith("A") for p in pontos)}


def fig_escala_lei():
    linhas = list(csv.DictReader(open(OUT / "tabela_principal.csv")))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 2.0), constrained_layout=True)
    for c in COR:
        pts = sorted((TAM[r["tamanho"]], 100 * float(r["danoso_cumpre"])) for r in linhas
                     if r["condicao"] == c and r["metodo"] in ("original", "lora" if r["tamanho"] in ("14b", "32b") else "completo"))
        if pts:
            a1.plot(*zip(*pts), marker=MARC[c], ms=3.5, color=COR[c], label=ROT[c], ls="--" if c == "A" else "-")
    a1.set_xscale("log"); a1.set_xticks(list(TAM.values()), ["1.7", "4", "8", "14", "32"]); a1.minorticks_off()
    a1.set_xlabel("parâmetros (B)"); a1.set_ylabel("danosos atendidos (%)"); a1.set_title("(a) escala", fontsize=7)
    a1.grid(axis="y", color="#e4e3df", lw=0.5)
    a1.legend(fontsize=5.5, ncol=2, loc="upper left", handlelength=1.6, columnspacing=0.8)

    pontos = []
    for sub in ("exp5", "exp6", "exp7"):
        for f in sorted((S / sub).glob("*.jsonl")):
            if f.stem.startswith(("jb_", "ut_")):
                continue
            p = ponto_juiz(f) or l38.ponto(f)   # deliberação pelo juiz (scripts/50) quando existe; senão a regex
            if p and not re.match(r"D[mnp]_", f.stem) and cond_de(f.stem) not in FORA_DO_ESCOPO:   # DPO do exp. 5: fora do escopo do artigo
                pontos.append({"modelo": f"{sub}/{f.stem}", "cond": cond_de(f.stem), "delibera": p[0], "cumpre": p[1]})
    (OUT / "lei_pontos.json").write_text(json.dumps(pontos, ensure_ascii=False, indent=1))
    lei = lei_de(pontos)
    (OUT / "lei_artigo.json").write_text(json.dumps(lei, ensure_ascii=False, indent=1))
    a, b = lei["a"], lei["b"]
    for c in COR:
        ps = [p for p in pontos if p["cond"] == c]
        a2.scatter([100 * p["delibera"] for p in ps], [100 * p["cumpre"] for p in ps], s=10, marker=MARC[c], color=COR[c],
                   edgecolors="white", linewidths=0.4, zorder=3)
    a2.plot([0, 100], [100 * a, 100 * (a + b)], color=TINTA2, lw=0.9, ls=":")
    a2.text(8, 33, f"$R^2$={lei['R2']:.2f}\n$\\rho$={lei['spearman']:.2f}  (n={lei['n_modelos']})", fontsize=6, color=TINTA)
    a2.set_xlabel("deliberação sobre dano no raciocínio (%)"); a2.set_ylabel("danosos atendidos (%)")
    a2.set_title("(b) deliberação prevê segurança", fontsize=7); a2.grid(color="#e4e3df", lw=0.5)
    fig.savefig(OUT / "fig_escala_lei.pdf")


def fig_jb_repr():
    tab = open(OUT / "jailbreak.md").read().splitlines()
    acc = defaultdict(lambda: defaultdict(list))
    for ln in tab:
        cel = [x.strip() for x in ln.strip("|").split("|")]
        if len(cel) == 5 and re.match(r"[A-Z]+_", cel[0]):
            c = cel[0].split("_")[0]
            for k, v in zip(("ficção", "autoridade", "passado"), cel[2:]):
                if v not in ("—", ""):
                    acc[c][k].append(float(v))
    conds = [c for c in ("A", "DH", "P", "DHD", "LAI") if c in acc]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.8, 1.95), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
    larg = 0.8 / len(conds)
    for i, c in enumerate(conds):
        xs = [j + (i - (len(conds) - 1) / 2) * larg for j in range(3)]
        ys = [sum(acc[c][k]) / len(acc[c][k]) for k in ("ficção", "autoridade", "passado")]
        a1.bar(xs, ys, width=larg * 0.88, color=COR[c], label=ROT[c], edgecolor="white", linewidth=0.3)
    a1.set_xticks(range(3), ["ficção", "autoridade", "passado"]); a1.set_ylabel("danosos atendidos (%)")
    a1.set_title("(a) jailbreak (média 1.7–32B)", fontsize=7); a1.grid(axis="y", color="#e4e3df", lw=0.5); a1.set_axisbelow(True)
    a1.legend(fontsize=5.3, ncol=2, loc="upper right", handlelength=1.0, columnspacing=0.6)

    rep = json.loads((OUT / "representacoes.json").read_text())
    for k, c in (("base", "A"), ("DH", "DH"), ("P", "P"), ("DHD", "DHD"), ("LAI", "LAI")):
        if k in rep:
            v = rep[k]["normas_por_camada"]
            a2.plot(range(len(v)), v, color=COR[c], ls="--" if c == "A" else "-", lw=1.1,
                    label=f"{ROT[c]} (sonda {100 * rep[k]['sonda_acc']:.0f}%)")
    a2.set_xlabel("camada"); a2.set_ylabel(r"$\|d_\ell\|$ (danoso $-$ seguro)")
    a2.set_title("(b) direção de recusa, Qwen3-4B", fontsize=7); a2.grid(color="#e4e3df", lw=0.5)
    a2.legend(fontsize=5.0, loc="upper left", handlelength=1.4)
    fig.savefig(OUT / "fig_jb_repr.pdf")


if __name__ == "__main__":
    fig_escala_lei()
    fig_jb_repr()
    print("ok")
