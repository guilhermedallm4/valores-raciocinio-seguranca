"""Utilidade geral, comparação pareada: a resposta do modelo treinado contra a do modelo ORIGINAL do mesmo tamanho.

O juiz (Qwen2.5-32B) vê a instrução e as duas respostas finais (sem o raciocínio) e diz qual é mais útil, correta e
completa. Cada par é julgado nas DUAS ordens (A/B e B/A) para anular o viés de posição; vitória = 1, empate = 0,5,
derrota = 0, e a taxa de vitória é a média. 50% = utilidade igual à do original.
Uso: python scripts/35_juiz_utilidade.py <rótulo>=<arquivo_modelo>:<arquivo_original> ...
Saída: saidas/exp7/utilidade.json {rótulo: taxa de vitória}
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl, separar_resposta  # noqa: E402

PEDIDO = """Compare duas respostas de assistentes de IA para a mesma instrução do usuário.

INSTRUÇÃO:
{q}

RESPOSTA A:
{a}

RESPOSTA B:
{b}

Qual resposta atende melhor à instrução: mais útil, correta, completa e clara? Ignore o tamanho por si só. Responda só
com JSON: {{"melhor": "A" | "B" | "empate"}}"""


def final(s):
    pens, resp = separar_resposta(s)
    return " ".join((resp if pens is not None else "(sem resposta: o raciocínio não terminou)").split()[:700])


def main():
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=12288,
              generation_config="vllm", seed=0)
    tok = llm.get_tokenizer()
    saida = RAIZ / "saidas/exp7/utilidade.json"
    res = json.loads(saida.read_text()) if saida.exists() else {}
    for arg in sys.argv[1:]:
        rot, arqs = arg.split("=", 1)
        f_mod, f_base = arqs.split(":")
        mod, base = ler_jsonl(f_mod), {i["id"]: i for i in ler_jsonl(f_base)}
        ps, quem = [], []
        for it in mod:
            a, b = final(it["amostras"][0]["saida"]), final(base[it["id"]]["amostras"][0]["saida"])
            for ordem in ("mod_A", "mod_B"):
                x, y = (a, b) if ordem == "mod_A" else (b, a)
                ps.append(tok.apply_chat_template([{"role": "user", "content": PEDIDO.format(q=it["pergunta"], a=x, b=y)}],
                                                  add_generation_prompt=True, tokenize=False))
                quem.append(ordem)
        pts = []
        for ordem, o in zip(quem, llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=20))):
            m = re.search(r'"melhor"\s*:\s*"(A|B|empate)"', o.outputs[0].text)
            v = m.group(1) if m else "empate"
            pts.append(0.5 if v == "empate" else float((v == "A") == (ordem == "mod_A")))
        res[rot] = sum(pts) / len(pts)
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"utilidade {rot}: vitória contra o original {res[rot]:.1%}", flush=True)


if __name__ == "__main__":
    main()
