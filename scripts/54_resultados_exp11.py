"""Resumo do exp. 11 (neurônios da trava): cumprimento danoso e seguro de cada edição, McNemar contra P e contra o
original, e a concentração da atribuição. Saída: saidas/artigo/exp11.md/json"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl  # noqa: E402

_s = importlib.util.spec_from_file_location("e51", Path(__file__).parent / "51_estatistica_exp10.py")
S = RAIZ / "saidas"
E = S / "exp11"


def carregar51():
    # o script 51 roda tudo no import; aqui só precisamos das funções
    src = (Path(__file__).parent / "51_estatistica_exp10.py").read_text().split("PARES = [")[0]
    ns = {"__file__": str(Path(__file__).parent / "51_estatistica_exp10.py")}
    exec(compile(src, "51", "exec"), ns)
    return ns["atend"], ns["mcnemar"]


atend, mcnemar = carregar51()


def seguro(f):
    its = [x for x in ler_jsonl(f) if x.get("juiz2") and x["rotulo"] == "seguro"]
    return 100 * sum(x["juiz2"]["classe"] == "CUMPRE" for x in its) / max(1, len(its))


P, A = S / "exp6/P_4b.jsonl", S / "exp6/A_4b.jsonl"
ROT = {"topo": "P, k do topo devolvidos ao original", "aleat": "P, k sorteados devolvidos ao original",
       "transp": "original com os k do topo vindos de P"}
L = ["# Exp. 11: neurônios da trava (Qwen3-4B, P = 48 Leis s1)\n",
     "| edição | k (% dos neurônios) | danoso % | Δ vs P (IC) | p vs P | Δ vs original (IC) | p vs original | seguro % |",
     "|---|---|---|---|---|---|---|---|"]
out = {}
aP, aA = atend(P), atend(A)
L.append(f"| P (sem edição) | – | {100 * sum(aP.values()) / len(aP):.1f} | – | – | – | – | {seguro(P):.1f} |")
L.append(f"| original | – | {100 * sum(aA.values()) / len(aA):.1f} | – | – | – | – | {seguro(A):.1f} |")
for modo in ("topo", "aleat", "transp"):
    for f in sorted(E.glob(f"{modo}_*.jsonl"), key=lambda x: float(x.stem.split("_")[1])):
        b = atend(f)
        if not b:
            continue
        rp, ra = mcnemar(aP, b), mcnemar(aA, b)
        out[f.stem] = {"danoso": rp[2], "vs_P": rp, "vs_A": ra, "seguro": seguro(f)}
        L.append(f"| {ROT[modo]} | {f.stem.split('_')[1]} | {rp[2]:.1f} | {rp[2] - rp[1]:+.1f} ({rp[3]:+.1f}; {rp[4]:+.1f}) | "
                 f"{rp[7]:.2g} | {ra[2] - ra[1]:+.1f} ({ra[3]:+.1f}; {ra[4]:+.1f}) | {ra[7]:.2g} | {seguro(f):.1f} |")
info = json.loads((E / "atribuicao.json").read_text())
L += ["\n## Atribuição\n", "```", json.dumps(info, indent=1), "```"]
(S / "artigo/exp11.md").write_text("\n".join(L) + "\n")
(S / "artigo/exp11.json").write_text(json.dumps({"edicoes": out, "atribuicao": info}, indent=1))
print("\n".join(L))
