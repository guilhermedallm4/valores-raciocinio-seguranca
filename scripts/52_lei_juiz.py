"""Lei deliberação -> segurança com a deliberação medida pelo JUIZ (scripts/50) em vez da expressão regular.
- mesmos modelos da Fig. 2b (exp5/6/7, sem DPO) e o livro neutro como teste fora da amostra;
- concordância item a item juiz x regex (kappa);
- nível do item: P(atende | delibera) vs P(atende | não delibera), por modelo e agregado.
Saída: saidas/artigo/lei_juiz.{md,json}"""
import importlib.util, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl, separar_resposta  # noqa: E402
S = RAIZ / "saidas"; C = S / "artigo/delib_juiz"
_s = importlib.util.spec_from_file_location("f43", Path(__file__).parent / "43_figuras_artigo.py"); f43 = importlib.util.module_from_spec(_s)
sys.argv = sys.argv[:1]; _s.loader.exec_module(f43)
_s = importlib.util.spec_from_file_location("l38", Path(__file__).parent / "38_lei_quantitativa.py"); l38 = importlib.util.module_from_spec(_s); _s.loader.exec_module(l38)

def itens(f):
    j = json.loads((C / f"{f.parent.name}__{f.stem}.json").read_text())
    out = []
    for x in ler_jsonl(f):
        if x.get("juiz2") and x["rotulo"].startswith("danoso") and j.get(x["id"]) is not None:
            pens = separar_resposta(x["amostras"][0]["saida"])[0] or ""
            out.append((j[x["id"]], bool(l38.DELIB.search(pens)), x["juiz2"]["classe"] in ("CUMPRE", "PARCIAL")))
    return out

pts, fora, todos = [], [], []
for sub in ("exp5", "exp6", "exp7", "exp10"):
    for f in sorted((S / sub).glob("*.jsonl")):
        if not (C / f"{sub}__{f.stem}.json").exists(): continue
        if f43.cond_de(f.stem) in f43.FORA_DO_ESCOPO or re.match(r"D[mnp]_", f.stem): continue
        it = itens(f)
        if len(it) < 50: continue
        p = {"modelo": f"{sub}/{f.stem}", "cond": f43.cond_de(f.stem), "delibera": np.mean([a for a, _, _ in it]),
             "delib_regex": np.mean([b for _, b, _ in it]), "cumpre": np.mean([c for _, _, c in it]), "n": len(it)}
        (fora if sub == "exp10" else pts).append(p); todos += it
lei = f43.lei_de(pts)
lei_rx = f43.lei_de([{**p, "delibera": p["delib_regex"]} for p in pts])
a = np.array([x[0] for x in todos]); b = np.array([x[1] for x in todos]); y = np.array([x[2] for x in todos])
po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()); kappa = (po - pe) / (1 - pe)
L = [f"# Lei com deliberação pelo juiz\n",
     f"- juiz: n={lei['n_modelos']}, R²={lei['R2']:.3f}, Pearson={lei['pearson']:.3f}, Spearman={lei['spearman']:.3f}, "
     f"cumpre = {lei['a']:.4f} + {lei['b']:.4f}·delibera",
     f"- regex (mesmos modelos): R²={lei_rx['R2']:.3f}, Pearson={lei_rx['pearson']:.3f}, Spearman={lei_rx['spearman']:.3f}",
     f"- concordância item a item juiz×regex: {100*po:.1f}% (kappa {kappa:.2f}); juiz delibera {100*a.mean():.1f}%, regex {100*b.mean():.1f}%",
     f"- nível do item (todos os modelos): P(atende|delibera)={100*y[a].mean():.1f}% (n={a.sum()}), "
     f"P(atende|não delibera)={100*y[~a].mean():.1f}% (n={(~a).sum()})",
     "\n## Fora da amostra: livro neutro (previsto pela reta do juiz)\n", "| modelo | delibera (juiz) | delibera (regex) | previsto | observado |", "|---|---|---|---|---|"]
for p in fora:
    L.append(f"| {p['modelo']} | {100*p['delibera']:.1f} | {100*p['delib_regex']:.1f} | {100*(lei['a']+lei['b']*p['delibera']):.1f} | {100*p['cumpre']:.1f} |")
L += ["\n## Por modelo\n", "| modelo | delibera juiz | regex | atende |", "|---|---|---|---|"]
L += [f"| {p['modelo']} | {100*p['delibera']:.1f} | {100*p['delib_regex']:.1f} | {100*p['cumpre']:.1f} |" for p in pts]
(S / "artigo/lei_juiz.md").write_text("\n".join(L) + "\n")
(S / "artigo/lei_juiz.json").write_text(json.dumps({"lei": lei, "lei_regex": lei_rx, "kappa": kappa, "pontos": pts, "fora": fora},
                                                   default=float, ensure_ascii=False, indent=1))
print("\n".join(L[:12 + len(fora)]))
