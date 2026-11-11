"""Resumo do exp. 13 (scripts/58): (a) anatomia das atualizações por matriz e por faixa de camadas, P vs as outras
condições; (b) reversões parciais de P: cumprimento danoso, McNemar contra P e contra o original, fração do dano
recuperada = (P - edição) / (P - original), deliberação. Saída: saidas/artigo/exp13.md/json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ  # noqa: E402

S = RAIZ / "saidas"; E = S / "exp13"; DJ = S / "artigo/delib_juiz"
src = (Path(__file__).parent / "51_estatistica_exp10.py").read_text().split("PARES = [")[0]
ns = {"__file__": str(Path(__file__).parent / "51_estatistica_exp10.py")}
exec(compile(src, "51", "exec"), ns)
atend, mcnemar = ns["atend"], ns["mcnemar"]
FAM = {"qwen4b": "Qwen3-4B (completo)", "phi4": "Phi-4-reasoning 14B", "gemma4": "Gemma 4 12B", "nemo": "Mistral Nemo 12B",
       "granite": "Granite 3.3 8B", "olmo2": "OLMo 2 13B"}
REF = {"qwen4b": (S / "exp6/P_4b.jsonl", S / "exp6/A_4b.jsonl")}
for f in list(FAM)[1:]:
    REF[f] = (S / f"exp12/P_{f}.jsonl", S / f"exp12/A_{f}.jsonl")
MATS = ["q", "k", "v", "o", "gate", "up", "down"]
SELS = ["rev_q1", "rev_q2", "rev_q3", "rev_q4", "keep_q1", "keep_q2", "keep_q3", "keep_q4",
        "rev_gate", "rev_up", "rev_down", "rev_mlp", "rev_attn"]
ROT = {"rev_q1": "desfaz 1º quarto", "rev_q2": "desfaz 2º quarto", "rev_q3": "desfaz 3º quarto", "rev_q4": "desfaz 4º quarto",
       "keep_q1": "só o 1º quarto de P", "keep_q2": "só o 2º quarto de P", "keep_q3": "só o 3º quarto de P",
       "keep_q4": "só o 4º quarto de P", "rev_gate": "desfaz gate", "rev_up": "desfaz up", "rev_down": "desfaz down",
       "rev_mlp": "desfaz MLP (gate+up+down)", "rev_attn": "desfaz atenção"}


def delib(f):
    c = DJ / f"{f.parent.name}__{f.stem}.json"
    if not c.exists():
        return None
    v = [b for b in json.loads(c.read_text()).values() if b is not None]
    return 100 * sum(v) / max(1, len(v))


def anatomia(fam):
    p = E / f"delta_{fam}.json"
    if not p.exists():
        return None, []
    R = json.loads(p.read_text())
    L = max(r["camada"] for r in R) + 1
    res, out = {}, []
    for c in ("p", "dh", "liv", "dhd", "lai"):
        rc = [r for r in R if r["cond"] == c]
        tot = sum(r["fro"] ** 2 for r in rc)
        mat = {m: sum(r["fro"] ** 2 for r in rc if r["matriz"] == m) / tot for m in MATS}
        quarto = [sum(r["fro"] ** 2 for r in rc if i * L // 4 <= r["camada"] < (i + 1) * L // 4) / tot for i in range(4)]
        le = {m: float(np.nanmean([r["le"] for r in rc if r["matriz"] == m and "le" in r])) for m in ("q", "k", "v", "gate", "up")}
        es = {m: float(np.nanmean([r["escreve"] for r in rc if r["matriz"] == m and "escreve" in r])) for m in ("o", "down")}
        cos = {m: float(np.mean([r["cos_DH"] for r in rc if r["matriz"] == m and "cos_DH" in r])) for m in MATS}
        rel = {m: float(np.nanmean([r["rel"] for r in rc if r["matriz"] == m])) for m in MATS}
        top1 = {m: float(np.mean([r["top1"] for r in rc if r["matriz"] == m])) for m in MATS}
        res[c] = {"norma_total": tot ** 0.5, "fracao_por_matriz": mat, "fracao_por_quarto": quarto, "le": le,
                  "escreve": es, "cos_DH": cos, "rel": rel, "top1": top1}
        # perfil por camada de P (para a figura)
    perfil = {}
    for c in ("p", "dh"):
        for m in ("gate", "up", "down"):
            perfil[f"{c}_{m}"] = [sum(r["fro"] ** 2 for r in R if r["cond"] == c and r["matriz"] == m and r["camada"] == l) ** 0.5
                                  for l in range(L)]
    res["perfil"] = perfil
    res["L"] = L
    return res, out


L = ["# Exp. 13: onde está a trava? (reversões parciais de P e anatomia das atualizações)\n"]
OUTJ = {}
for fam, nome in FAM.items():
    an, _ = anatomia(fam)
    fP, fA = REF[fam]
    aP, aA = atend(fP), atend(fA)
    hP, hA = 100 * np.mean(list(aP.values())), 100 * np.mean(list(aA.values()))
    L += [f"\n## {nome}\n", f"P = {hP:.1f}% danoso; original = {hA:.1f}%\n",
          "| edição | danoso % | p vs P | p vs original | recuperado % | delibera % |", "|---|---|---|---|---|---|"]
    OUTJ[fam] = {"P": hP, "A": hA, "edicoes": {}, "anatomia": an}
    for s in SELS:
        f = E / f"{fam}__{s}.jsonl"
        if not f.exists():
            continue
        b = atend(f)
        if not b:
            L.append(f"| {ROT[s]} | (sem juiz) | | | | |")
            continue
        rp, ra = mcnemar(aP, b), mcnemar(aA, b)
        rec = 100 * (rp[1] - rp[2]) / (rp[1] - ra[1]) if rp[1] != ra[1] else float("nan")
        d = delib(f)
        OUTJ[fam]["edicoes"][s] = {"danoso": rp[2], "p_vs_P": rp[7], "p_vs_A": ra[7], "recuperado": rec, "delib": d}
        L.append(f"| {ROT[s]} | {rp[2]:.1f} | {rp[7]:.2g} | {ra[7]:.2g} | {rec:.0f} | {'' if d is None else f'{d:.0f}'} |")
    if an:
        L += ["", "Anatomia (fração da energia ||ΔW||² por matriz; alinhamento à direção de recusa, 1 = acaso):", "",
              "| cond. | norma | q | k | v | o | gate | up | down | 1º–4º quarto | lê (gate/up) | escreve (o/down) | cos c/ DUDH (gate/up/down) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in ("p", "dh", "liv", "dhd", "lai"):
            x = an[c]
            fm = " | ".join(f"{100 * x['fracao_por_matriz'][m]:.0f}" for m in MATS)
            fq = "/".join(f"{100 * q:.0f}" for q in x["fracao_por_quarto"])
            L.append(f"| {c.upper()} | {x['norma_total']:.1f} | {fm} | {fq} | {x['le']['gate']:.2f}/{x['le']['up']:.2f} | "
                     f"{x['escreve']['o']:.2f}/{x['escreve']['down']:.2f} | "
                     f"{x['cos_DH']['gate']:.2f}/{x['cos_DH']['up']:.2f}/{x['cos_DH']['down']:.2f} |")
(S / "artigo/exp13.md").write_text("\n".join(L) + "\n")
(S / "artigo/exp13.json").write_text(json.dumps(OUTJ, indent=1))
print("\n".join(L))
