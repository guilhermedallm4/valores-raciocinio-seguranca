"""Resumo do exp. 12 (famílias): por família, cumprimento danoso (juiz principal), seguro atendido, deliberação
(juiz do scripts/50) e McNemar contra o original da família. Saída: saidas/artigo/exp12.md/json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl  # noqa: E402

S = RAIZ / "saidas"; E = S / "exp12"; DJ = S / "artigo/delib_juiz"
src = (Path(__file__).parent / "51_estatistica_exp10.py").read_text().split("PARES = [")[0]
ns = {"__file__": str(Path(__file__).parent / "51_estatistica_exp10.py")}; exec(compile(src, "51", "exec"), ns)
atend, mcnemar = ns["atend"], ns["mcnemar"]
FAM = {"phi4": "Phi-4-reasoning 14B", "gemma4": "Gemma 4 12B", "nemo": "Mistral Nemo 12B", "granite": "Granite 3.3 8B",
       "olmo2": "OLMo 2 13B"}
CONDS = [("A", "original"), ("LIV", "livro neutro"), ("DH", "DUDH"), ("P", "48 Leis"), ("DHD", "DUDH-delib"),
         ("LAI", "Laica-delib")]


def seguro(f):
    its = [x for x in ler_jsonl(f) if x.get("juiz2") and x["rotulo"] == "seguro"]
    return 100 * sum(x["juiz2"]["classe"] == "CUMPRE" for x in its) / max(1, len(its))


def delib(f):
    c = DJ / f"exp12__{f.stem}.json"
    if not c.exists():
        return None
    q = json.loads(c.read_text()); v = [b for b in q.values() if b is not None]
    return 100 * sum(v) / max(1, len(v))


L = ["# Exp. 12: outras famílias (LoRA, 1 semente)\n",
     "| família | condição | danoso % | Δ vs original (IC 95%) | p | seguro atendido % | delibera % |", "|---|---|---|---|---|---|---|"]
out = {}
for fam, nome in FAM.items():
    fa = E / f"A_{fam}.jsonl"; a = atend(fa)
    for c, rot in CONDS:
        f = E / f"{c}_{fam}.jsonl"
        if not f.exists():
            L.append(f"| {nome} | {rot} | (pendente) | | | | |"); continue
        b = atend(f); d = delib(f); sg = seguro(f)
        if c == "A":
            h = 100 * sum(b.values()) / len(b); out[f.stem] = {"danoso": h, "seguro": sg, "delib": d}
            L.append(f"| {nome} | {rot} | {h:.1f} | – | – | {sg:.1f} | {d if d is None else round(d, 1)} |"); continue
        r = mcnemar(a, b); out[f.stem] = {"danoso": r[2], "vs_A": r, "seguro": sg, "delib": d}
        L.append(f"| {nome} | {rot} | {r[2]:.1f} | {r[2]-r[1]:+.1f} ({r[3]:+.1f}; {r[4]:+.1f}) | {r[7]:.2g} | {sg:.1f} | "
                 f"{d if d is None else round(d, 1)} |")
(S / "artigo/exp12.md").write_text("\n".join(L) + "\n"); (S / "artigo/exp12.json").write_text(json.dumps(out, indent=1))
print("\n".join(L))
