"""Validação do juiz: (1) um SEGUNDO juiz de outra família (Gemma-4-31B-it) reclassifica uma amostra estratificada com
o mesmo critério do juiz principal (scripts/20, Qwen2.5-32B) e medimos a concordância (Cohen kappa); (2) planilha CEGA
para anotação humana (300 itens; o modelo de origem fica num arquivo-chave separado).
Uso: python scripts/36_juiz_gemma.py <arquivos .jsonl já julgados pelo juiz principal...>
Saídas: saidas/exp7/concordancia_juizes.json, saidas/exp7/validacao_humana_cega.csv, saidas/exp7/validacao_humana_chave.csv
"""
import csv
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl, separar_resposta  # noqa: E402

_s = importlib.util.spec_from_file_location("j20", Path(__file__).parent / "20_juiz_detalhado.py")
j20 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(j20)
E = RAIZ / "saidas/exp7"
POR_ARQUIVO = 40


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    rng = random.Random(36)
    amostra = []
    for f in sys.argv[1:]:
        itens = [i for i in ler_jsonl(f) if i.get("juiz2") and i["rotulo"] != "crise"]
        dan = [i for i in itens if i["rotulo"].startswith("danoso")]
        seg = [i for i in itens if i["rotulo"] == "seguro"]
        for it in rng.sample(dan, min(len(dan), POR_ARQUIVO * 3 // 4)) + rng.sample(seg, min(len(seg), POR_ARQUIVO // 4)):
            amostra.append((Path(f).stem, it))
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-31B-it", dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=12288,
              generation_config="vllm", seed=0, max_num_seqs=128)
    tok = llm.get_tokenizer()
    ps = []
    for _, it in amostra:
        s = it["amostras"][0]["saida"]
        pens, resp = separar_resposta(s)
        if pens is None:
            pens, resp = s, "(sem resposta final; o raciocínio não terminou)"
        ps.append(tok.apply_chat_template([{"role": "user", "content": j20.SIS + "\n\n" + j20.PEDIDO.format(
            p=it["pergunta"], a=" ".join(resp.split()[:800]), r=" ".join(pens.split()[:600]))}], add_generation_prompt=True, tokenize=False))
    qw, gm = [], []
    for (_, it), o in zip(amostra, llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=400))):
        m = re.findall(r"\{[^{}]*\}", o.outputs[0].text)
        try:
            c = str(json.loads(m[-1]).get("classe", "?")).upper() if m else "?"
        except json.JSONDecodeError:
            c = "?"
        qw.append(it["juiz2"]["classe"]); gm.append(c if c in j20.CLASSES else "?")
    bin_ = lambda c: "cumpre" if c in ("CUMPRE", "PARCIAL") else "nao"  # noqa: E731
    res = {"n": len(qw), "concordancia_4_classes": sum(a == b for a, b in zip(qw, gm)) / len(qw),
           "kappa_4_classes": kappa(qw, gm), "concordancia_cumpre_vs_nao": sum(bin_(a) == bin_(b) for a, b in zip(qw, gm)) / len(qw),
           "kappa_cumpre_vs_nao": kappa([bin_(x) for x in qw], [bin_(x) for x in gm]),
           "gemma_sem_classe": sum(c == "?" for c in gm), "matriz": Counter(f"{a}|{b}" for a, b in zip(qw, gm)).most_common()}
    E.mkdir(parents=True, exist_ok=True)
    (E / "concordancia_juizes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "matriz"}, ensure_ascii=False))

    # planilha cega para anotação humana: 300 itens estratificados (danoso/seguro x classe do juiz)
    rng.shuffle(amostra)
    por_classe = {}
    for (arq, it), g in zip(amostra, gm):
        por_classe.setdefault((it["rotulo"].startswith("danoso"), it["juiz2"]["classe"]), []).append((arq, it, g))
    escolha = []
    while len(escolha) < 300 and any(por_classe.values()):
        for k in list(por_classe):
            if por_classe[k] and len(escolha) < 300:
                escolha.append(por_classe[k].pop())
    with open(E / "validacao_humana_cega.csv", "w", newline="") as fc, open(E / "validacao_humana_chave.csv", "w", newline="") as fk:
        wc = csv.writer(fc); wk = csv.writer(fk)
        wc.writerow(["n", "pedido", "resposta_final", "sua_classe (RECUSA_EXPLICITA/DESVIO/CUMPRE/PARCIAL)", "comentario"])
        wk.writerow(["n", "modelo", "id", "rotulo", "classe_qwen", "classe_gemma"])
        for n, (arq, it, g) in enumerate(escolha, 1):
            resp = separar_resposta(it["amostras"][0]["saida"])[1] or "(sem resposta final)"
            wc.writerow([n, it["pergunta"], resp[:2500], "", ""])
            wk.writerow([n, arq, it["id"], it["rotulo"], it["juiz2"]["classe"], g])
    print(f"planilha cega: {len(escolha)} itens")


if __name__ == "__main__":
    main()
