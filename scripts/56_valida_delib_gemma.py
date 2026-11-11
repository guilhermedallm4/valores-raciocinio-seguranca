"""Validação do juiz de DELIBERAÇÃO (scripts/50, Qwen2.5-32B) por um juiz de outra família (Gemma-4-31B-it), com o
MESMO enunciado: amostra estratificada de 40 pedidos danosos por arquivo (condições laicas: original, DUDH, 48 Leis,
livro neutro, DUDH-delib, Laica-delib; exp5/6/7/10), concordância e kappa de Cohen.
Uso: .venv-vllm/bin/python scripts/56_valida_delib_gemma.py      Saída: saidas/artigo/valida_delib.json/md
"""
import importlib.util, json, os, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl, separar_resposta  # noqa: E402
_s = importlib.util.spec_from_file_location("j50", Path(__file__).parent / "50_juiz_deliberacao.py")
j50 = importlib.util.module_from_spec(_s); _s.loader.exec_module(j50)
_s = importlib.util.spec_from_file_location("f43", Path(__file__).parent / "43_figuras_artigo.py")
S = RAIZ / "saidas"; POR = 40
FORA_DO_ESCOPO = ("C", "N", "DEC", "B")


def cond(stem):
    m = re.match(r"(?:[a-z]+_)?([A-Z]+)", stem)
    return m.group(1) if m else None


def main():
    rng = random.Random(56); amostra = []
    for f in j50.arquivos():
        if f.parent.name in ("exp12", "exp13", "exp14") or cond(f.stem) in FORA_DO_ESCOPO:
            continue
        c = j50.OUT / f"{f.parent.name}__{f.stem}.json"
        if not c.exists():
            continue
        qj = json.loads(c.read_text())
        its = [x for x in ler_jsonl(f) if x.get("juiz2") and x["rotulo"].startswith("danoso") and qj.get(x["id"]) is not None]
        for x in rng.sample(its, min(POR, len(its))):
            amostra.append((f"{f.parent.name}/{f.stem}", x, qj[x["id"]]))
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-31B-it", dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=8192,
              generation_config="vllm", seed=0, max_num_seqs=128)
    tok = llm.get_tokenizer(); ps = []
    for _, x, _ in amostra:
        s = x["amostras"][0]["saida"]; pens = separar_resposta(s)[0]
        pens = pens if pens is not None else s
        ps.append(tok.apply_chat_template([{"role": "user", "content": j50.SIS + "\n\n" + j50.PEDIDO.format(
            p=x["pergunta"], r=" ".join(pens.split()[:1500]) or "(vazio)")}], add_generation_prompt=True, tokenize=False))
    q, g, por = [], [], {}
    for (mod, x, qv), o in zip(amostra, llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=400))):
        m = re.findall(r"\{[^{}]*\}", o.outputs[0].text)
        try:
            gv = bool(json.loads(m[-1]).get("delibera")) if m else None
        except json.JSONDecodeError:
            gv = None
        if gv is None:
            continue
        q.append(qv); g.append(gv)
        por.setdefault(cond(mod.split("/")[1]), []).append((qv, gv))
    n = len(q); po = sum(a == b for a, b in zip(q, g)) / n
    pq, pg = sum(q) / n, sum(g) / n; pe = pq * pg + (1 - pq) * (1 - pg); k = (po - pe) / (1 - pe)
    res = {"n": n, "invalidos": len(amostra) - n, "concordancia": po, "kappa": k, "delibera_qwen": pq, "delibera_gemma": pg,
           "por_condicao": {c: {"n": len(v), "qwen": sum(a for a, _ in v) / len(v), "gemma": sum(b for _, b in v) / len(v),
                                "concordancia": sum(a == b for a, b in v) / len(v)} for c, v in por.items()}}
    (S / "artigo/valida_delib.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
