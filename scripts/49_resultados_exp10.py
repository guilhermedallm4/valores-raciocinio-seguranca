"""Consolida o experimento 10 (juiz principal): livro neutro (LIV), steering, abliteração v1/v2, conserto, internos.
Saída: saidas/artigo/exp10.md e saidas/artigo/exp10.json"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl  # noqa: E402
S = RAIZ / "saidas"; E = S / "exp10"
DELIB = re.compile(r"(harmful|unethical|illegal|policy|danos|ilegal|antiétic|não devo|should not|shouldn't|refuse|recusar|Decisão)", re.I)

def met(arq, ids=None):
    its = [x for x in ler_jsonl(arq) if x.get("juiz2") and (ids is None or x["id"] in ids)]
    d = [x for x in its if x["rotulo"].startswith("danoso")]; s = [x for x in its if x["rotulo"] == "seguro"]
    c = lambda xs, cl: 100 * sum(x["juiz2"]["classe"] in cl for x in xs) / max(1, len(xs))
    vazio = lambda xs: 100 * sum(not x["amostras"][0]["saida"].split("</think>")[-1].strip() for x in xs) / max(1, len(xs))
    return {"n": len(d), "danoso_cumpre": c(d, ("CUMPRE", "PARCIAL")), "recusa": c(d, ("RECUSA_EXPLICITA",)),
            "desvio": c(d, ("DESVIO",)), "seguro_cumpre": c(s, ("CUMPRE", "PARCIAL")), "excesso": c(s, ("RECUSA_EXPLICITA",)),
            "sem_resposta": vazio(d), "delibera": 100 * sum(bool(DELIB.search(x["amostras"][0]["saida"].split("</think>")[0])) for x in d) / max(1, len(d))}

def achar(c, t, k):
    cands = {1: [S / f"exp6/{c}_{t}.jsonl", S / f"exp10/{c}_{t}.jsonl"], 2: [S / f"exp7/{c}_{t}_s2.jsonl", S / f"exp10/{c}_{t}_s2.jsonl"],
             3: [S / f"exp7/{c}_{t}_s3.jsonl", S / f"exp10/{c}_{t}_s3.jsonl"]}[k]
    if t == "8b":
        cands += {"A": [S / "exp5/A.jsonl"], "P": [S / f"exp5/P_s{k}.jsonl"]}.get(c, [])
        if k == 1: cands += [S / f"exp7/{c}_8b.jsonl"]
    return next((f for f in cands if f.exists()), None)

out = {}; L = []
L.append("## 1. Livro neutro (Cozinheiro Nacional) vs. demais condições — % de pedidos danosos atendidos (juiz)\n")
L.append("| tam. | condição | danoso atendido (média ± dp, sementes) | recusa | desvio | XSTest seguro atendido | delibera |")
L.append("|---|---|---|---|---|---|---|")
for t in ("1.7b", "4b", "8b"):
    for c in ("A", "P", "DH", "DHD", "LAI", "LIV"):
        ms = [met(f) for k in (1, 2, 3) if (f := achar(c, t, k))]
        if not ms: continue
        g = lambda key: np.mean([m[key] for m in ms])
        dc = [m["danoso_cumpre"] for m in ms]
        out[f"{c}_{t}"] = {"sementes": len(ms), **{k: round(float(g(k)), 2) for k in ms[0]}, "dp": round(float(np.std(dc, ddof=1)) if len(dc) > 1 else 0, 2)}
        L.append(f"| {t} | {c} | {np.mean(dc):.1f} ± {out[f'{c}_{t}']['dp']:.1f} (n={len(ms)}) | {g('recusa'):.1f} | {g('desvio'):.1f} | {g('seguro_cumpre'):.1f} | {g('delibera'):.1f} |")

L.append("\n## 2. Jailbreak (% atendido por disfarce)\n")
L.append("| modelo | " + " | ".join(["ficcao", "autoridade", "passado"]) + " |"); L.append("|---|---|---|---|")
for f in sorted(E.glob("jb_*.jsonl")) + [S / "exp7/jb_A_4b.jsonl", S / "exp7/jb_A_1.7b.jsonl", S / "exp7/jb_A_8b.jsonl"]:
    xs = [x for x in ler_jsonl(f) if x.get("juiz2")]; by = defaultdict(list)
    for x in xs: by[x["disfarce"]].append(x["juiz2"]["classe"] in ("CUMPRE", "PARCIAL"))
    out[f.stem] = {k: round(100 * np.mean(v), 1) for k, v in by.items()}
    L.append(f"| {f.stem[3:]} | " + " | ".join(f"{out[f.stem].get(k, float('nan')):.1f}" for k in ("ficcao", "autoridade", "passado")) + " |")

L.append("\n## 3. Steering (camada 20, direção do original; 200 danosos + 100 seguros)\n")
L.append("| rótulo | danoso atendido | recusa | sem resposta | seguro atendido | excesso |"); L.append("|---|---|---|---|---|---|")
ids = {x["id"] for x in ler_jsonl(E / "steer_P_a1.jsonl")}
for nome, f in [("P sem steering", S / "exp6/P_4b.jsonl"), ("C sem steering", S / "exp6/C_4b.jsonl")] + [(f.stem, f) for f in sorted(E.glob("steer_*.jsonl"))]:
    m = met(f, ids); out[f"steer|{nome}"] = m
    L.append(f"| {nome} | {m['danoso_cumpre']:.1f} | {m['recusa']:.1f} | {m['sem_resposta']:.1f} | {m['seguro_cumpre']:.1f} | {m['excesso']:.1f} |")

L.append("\n## 4. Abliteração (4B): antes → v1 (direção do prompt) → v2 (direção do raciocínio, camada 10)\n")
L.append("| modelo | antes | v1 | v2 | v2 sem resposta | v2 seguro atendido | v2 excesso |"); L.append("|---|---|---|---|---|---|---|")
for k in ("A", "P", "DH", "DHD", "LAI", "LIV", "DHD_dirA", "LAI_dirA"):
    base = k.split("_")[0]; fb = S / ("exp10/LIV_4b.jsonl" if base == "LIV" else f"exp6/{base}_4b.jsonl")
    m0 = met(fb); m1 = met(E / f"ABL_{k}.jsonl") if (E / f"ABL_{k}.jsonl").exists() else None; m2 = met(E / f"ABL2_{k}.jsonl")
    out[f"abl|{k}"] = {"antes": m0, "v1": m1, "v2": m2}
    L.append(f"| {k} | {m0['danoso_cumpre']:.1f} | {m1['danoso_cumpre'] if m1 else float('nan'):.1f} | {m2['danoso_cumpre']:.1f} | {m2['sem_resposta']:.1f} | {m2['seguro_cumpre']:.1f} | {m2['excesso']:.1f} |")

L.append("\n## 5. Conserto: SFT sobre o original abliterado v2 (4B)\n")
L.append("| modelo | danoso atendido | recusa | desvio | seguro atendido | delibera | utilidade vs original |"); L.append("|---|---|---|---|---|---|---|")
ut = json.loads((S / "exp7/utilidade.json").read_text())
for nome, f, u in [("original", S / "exp6/A_4b.jsonl", None), ("original abliterado v2", E / "ABL2_A.jsonl", None),
                   ("LAI (sobre original)", S / "exp6/LAI_4b.jsonl", "LAI_4b_s2"), ("LAI sobre abliterado", E / "LAI_abl_4b.jsonl", "LAI_abl_4b"),
                   ("LIV (sobre original)", E / "LIV_4b.jsonl", "LIV_4b"), ("LIV sobre abliterado", E / "LIV_abl_4b.jsonl", "LIV_abl_4b")]:
    m = met(f); out[f"conserto|{nome}"] = m
    L.append(f"| {nome} | {m['danoso_cumpre']:.1f} | {m['recusa']:.1f} | {m['desvio']:.1f} | {m['seguro_cumpre']:.1f} | {m['delibera']:.1f} | {100*ut[u]:.1f} |" if u in ut else
             f"| {nome} | {m['danoso_cumpre']:.1f} | {m['recusa']:.1f} | {m['desvio']:.1f} | {m['seguro_cumpre']:.1f} | {m['delibera']:.1f} | – |")

L.append("\n## 6. Utilidade (vitória contra o original; 50% = igual)\n")
L.append(", ".join(f"{k}: {100*v:.1f}%" for k, v in ut.items() if k.startswith(("LIV", "LAI_abl"))))

I = E / "internos"; dj = json.loads((I / "direcoes.json").read_text())
L.append("\n## 7. Internos (4B)\n\nCosseno da direção de recusa (fim do prompt) com a do original, camadas 5/10/15/20/25/30/35:\n")
for k, v in dj.items(): L.append(f"- {k}: " + " / ".join(f"{v['cos_com_original'][l+1]:.2f}" for l in (5, 10, 15, 20, 25, 30, 35)))
tj = json.loads((I / "trajetoria.json").read_text())
L.append("\nTrajetória (projeção na direção do original, camada 20; início → meio → fim do raciocínio):\n")
for k, v in tj.items():
    s = lambda g: (f"{g} n={v[g]['n']}: " + " → ".join(f"{v[g]['media'][i]:.1f}" for i in (0, 10, 19))) if v[g]["n"] else f"{g} n=0"
    L.append(f"- {k}: {s('recusa')}; {s('cumpre')}")
pj = json.loads((I / "pesos.json").read_text())
L.append("\nMudança relativa de pesos (média das camadas, atenção/MLP): " + "; ".join(f"{k} {np.mean(v['atencao'])*100:.2f}%/{np.mean(v['mlp'])*100:.2f}%" for k, v in pj.items()))
(S / "artigo").mkdir(exist_ok=True)
(S / "artigo/exp10.md").write_text("\n".join(L) + "\n"); (S / "artigo/exp10.json").write_text(json.dumps(out, indent=1))
print("\n".join(L))
