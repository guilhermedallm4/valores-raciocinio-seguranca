"""Resumo do exp. 14 (épocas): Qwen3-4B completo, 1 / 2 / 4 épocas em P (48 Leis), Laica-delib e livro neutro.
A de 2 épocas é a semente 1 do exp6/exp10 (mesma semente 2001 e mesmos dados). Cumprimento danoso, seguro, deliberação,
McNemar contra o original e contra 2 épocas, perda final do treino. Saída: saidas/artigo/exp14.md/json"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl  # noqa: E402

S = RAIZ / "saidas"; E = S / "exp14"; DJ = S / "artigo/delib_juiz"
src = (Path(__file__).parent / "51_estatistica_exp10.py").read_text().split("PARES = [")[0]
ns = {"__file__": str(Path(__file__).parent / "51_estatistica_exp10.py")}
exec(compile(src, "51", "exec"), ns)
atend, mcnemar = ns["atend"], ns["mcnemar"]
NOME = {"P": "48 Leis", "LAI": "Laica-delib", "LIV": "Livro neutro"}
E2 = {"P": S / "exp6/P_4b.jsonl", "LAI": S / "exp6/LAI_4b.jsonl", "LIV": S / "exp10/LIV_4b.jsonl"}
A = S / "exp6/A_4b.jsonl"


def seguro(f):
    its = [x for x in ler_jsonl(f) if x.get("juiz2") and x["rotulo"] == "seguro"]
    return 100 * sum(x["juiz2"]["classe"] == "CUMPRE" for x in its) / max(1, len(its))


def delib(f):
    c = DJ / f"{f.parent.name}__{f.stem}.json"
    if not c.exists():
        return None
    v = [b for b in json.loads(c.read_text()).values() if b is not None]
    return 100 * sum(v) / max(1, len(v))


def perda(c, e):
    lg = E / f"logs/sft_exp14_{c.lower()}_4b_e{e}.log"
    if not lg.exists():
        return None
    recs = re.findall(r"'loss': '?([0-9.]+)", lg.read_text(errors="ignore"))
    return float(recs[-1]) if recs else None


aA = atend(A)
L = ["# Exp. 14: número de épocas (Qwen3-4B, SFT completo, semente 2001)\n",
     f"Original: {100 * np.mean(list(aA.values())):.1f}% danoso\n",
     "| condição | épocas | danoso % | p vs original | Δ vs 2 épocas (IC) | p vs 2 épocas | seguro % | delibera % | perda final |",
     "|---|---|---|---|---|---|---|---|---|"]
out = {}
for c in ("P", "LAI", "LIV"):
    a2 = atend(E2[c])
    for e in (1, 2, 4):
        f = E2[c] if e == 2 else E / f"{c}_e{e}.jsonl"
        if not f.exists():
            L.append(f"| {NOME[c]} | {e} | (pendente) | | | | | | |")
            continue
        b = atend(f)
        if not b:
            L.append(f"| {NOME[c]} | {e} | (sem juiz) | | | | | | |")
            continue
        ra = mcnemar(aA, b)
        r2 = mcnemar(a2, b) if e != 2 else None
        d, sg, pl = delib(f), seguro(f), perda(c, e)
        out[f"{c}_e{e}"] = {"danoso": ra[2], "p_vs_A": ra[7], "vs_e2": r2, "seguro": sg, "delib": d, "perda": pl}
        L.append(f"| {NOME[c]} | {e} | {ra[2]:.1f} | {ra[7]:.2g} | " +
                 (f"{r2[2] - r2[1]:+.1f} ({r2[3]:+.1f}; {r2[4]:+.1f}) | {r2[7]:.2g}" if r2 else "– | –") +
                 f" | {sg:.1f} | {'' if d is None else f'{d:.0f}'} | {'' if pl is None else f'{pl:.3f}'} |")
# 48 Leis: sementes 2001-2003 (2 épocas = exp6 s1 / exp7 s2, s3) e 8 épocas (s1)
ARQ = {(1, 1): E / "P_e1.jsonl", (2, 1): E2["P"], (4, 1): E / "P_e4.jsonl", (8, 1): E / "P_e8.jsonl"}
for sm in (2, 3):
    ARQ[(1, sm)] = E / f"P_e1_s{sm}.jsonl"; ARQ[(2, sm)] = S / f"exp7/P_4b_s{sm}.jsonl"; ARQ[(4, sm)] = E / f"P_e4_s{sm}.jsonl"
L += ["", "## 48 Leis por semente (danoso % / delibera %; p = McNemar contra 2 épocas da mesma semente)", "",
      "| épocas | semente 2001 | semente 2002 | semente 2003 | média danoso (dp) | média delibera |", "|---|---|---|---|---|---|"]
for e in (1, 2, 4, 8):
    cel, hs, ds = [], [], []
    for sm in (1, 2, 3):
        f = ARQ.get((e, sm))
        b = atend(f) if f and f.exists() else None
        if not b:
            cel.append("–" if f is None else "(pendente)"); continue
        h = mcnemar(aA, b)[2]; d = delib(f); hs.append(h)
        if d is not None: ds.append(d)
        pp = ""
        if e != 2 and ARQ[(2, sm)].exists():
            pp = f", p={mcnemar(atend(ARQ[(2, sm)]), b)[7]:.2g}"
        cel.append(f"{h:.1f} / {'?' if d is None else f'{d:.0f}'}{pp}")
        out[f"P_e{e}_s{sm}"] = {"danoso": h, "delib": d}
    m = f"{np.mean(hs):.1f} ({np.std(hs, ddof=1):.1f})" if len(hs) > 1 else (f"{hs[0]:.1f}" if hs else "")
    L.append(f"| {e} | " + " | ".join(cel) + f" | {m} | {f'{np.mean(ds):.0f}' if ds else ''} |")
(S / "artigo/exp14.md").write_text("\n".join(L) + "\n")
(S / "artigo/exp14.json").write_text(json.dumps(out, indent=1))
print("\n".join(L))
