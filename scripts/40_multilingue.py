"""Experimento 8: o efeito atravessa línguas? Traduz os dados de treino e a avaliação de PT para EN, ES, ZH.

Para controlar o conteúdo, TRADUZIMOS os nossos próprios dados (não usamos a DUDH/livro em cada língua: isso
mudaria o conteúdo junto com a língua). O tradutor é o Qwen2.5-32B.
  avaliacao   os 1119 itens de segurança em cada língua (EN sai de graça: o campo "en" já existe; ES/ZH traduzidos do
              inglês para não acumular ruído de tradução dupla)
  treino      as completions (raciocínio + resposta) de C, P e LAI + a manutenção geral, traduzidas de PT para a língua
Avaliação cruzada: cada modelo (treinado numa língua) é avaliado nas 4 línguas -> matriz língua-treino x língua-aval.
Fases: avaliacao --lingua <es|zh> --parte i/n ; treino <c|p|lai> --lingua <en|es|zh> --parte i/n ; montar
Saídas: dados/seguranca/verificacao_<lg>.jsonl ; dados/ml_<cond>_<lg>/sft ; configs/exp8_<cond>_<lg>.yaml
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import DADOS, RAIZ, gravar_jsonl, ler_jsonl, montar_completion, montar_prompt, separar_resposta  # noqa: E402

LINGUAS = {"en": "inglês", "es": "espanhol", "zh": "chinês (simplificado)"}
SIS = "Você é um assistente útil."  # traduzido por língua em montar()
SIS_LG = {"en": "You are a helpful assistant.", "es": "Eres un asistente útil.", "zh": "你是一个有用的助手。", "pt": SIS}


def traduzir_lote(llm, tok, textos, lingua, seed, max_tokens=1200):
    from vllm import SamplingParams
    instr = (f"Traduza para {LINGUAS[lingua]} o texto a seguir, preservando o sentido, o tom e a formatação (inclusive "
             "as marcas «», aspas e referências). Devolva só a tradução, sem comentários.\n\n")
    ps = [tok.apply_chat_template([{"role": "system", "content": "Você é um tradutor profissional."},
                                   {"role": "user", "content": instr + t}], add_generation_prompt=True, tokenize=False) for t in textos]
    return [o.outputs[0].text.strip() for o in llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed))]


def carregar_tradutor(parte):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM
    llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=8192,
              generation_config="vllm", seed=40 + parte)
    return llm, llm.get_tokenizer()


def fase_avaliacao(lingua, parte, n):
    itens = ler_jsonl(DADOS / "seguranca/verificacao_seguranca.jsonl")
    av_en = {i["id"]: i.get("en") for i in ler_jsonl(DADOS / "seguranca/avaliacao_pt.jsonl")}  # texto original em inglês
    sub = itens[parte::n]
    if lingua == "en":
        for it in sub:
            it["pergunta"] = av_en.get(it["id"]) or it["pergunta"]
        gravar_jsonl(DADOS / f"seguranca/verificacao_en_parte{parte}.jsonl", sub)
        print(f"en parte {parte}: {sum(bool(av_en.get(i['id'])) for i in sub)}/{len(sub)} com original em inglês")
        return
    llm, tok = carregar_tradutor(parte)
    fonte = [av_en.get(it["id"]) or it["pergunta"] for it in sub]   # traduz do inglês
    for it, t in zip(sub, traduzir_lote(llm, tok, fonte, lingua, 40 + parte, 600)):
        it["pergunta"] = t
    gravar_jsonl(DADOS / f"seguranca/verificacao_{lingua}_parte{parte}.jsonl", sub)
    print(f"{lingua} parte {parte}: {len(sub)} itens")


def fase_treino(cond, lingua, parte, n):
    # traduz o treino PT do 4B inteiro (conteúdo + manutenção): mesmo tamanho de modelo, só muda a língua
    exs = ler_jsonl(DADOS / f"exp6_{cond}_4b/sft/treino.jsonl")[parte::n]
    llm, tok = carregar_tradutor(parte)
    # traduz pergunta e completion (raciocínio + resposta) juntos, separados por um marcador, para manter o par alinhado
    marc = "\n<<<###>>>\n"
    fonte_txt = [e["pergunta"] + marc + e["completion"] for e in exs]
    trad = traduzir_lote(llm, tok, fonte_txt, lingua, 40 + parte, 1600)
    regs = []
    for e, t in zip(exs, trad):
        if marc in t:
            p, c = t.split(marc, 1)
            regs.append({"origem": e["origem"], "pergunta": p.strip(), "completion": c.strip()})
    gravar_jsonl(DADOS / f"ml_{cond}_{lingua}/parte{parte}.jsonl", regs)
    print(f"treino {cond}_{lingua} parte {parte}: {len(regs)}/{len(exs)}")


def montar(n):
    from transformers import AutoTokenizer
    import yaml
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    # junta as partes da avaliação por língua
    for lg in LINGUAS:
        partes = [DADOS / f"seguranca/verificacao_{lg}_parte{p}.jsonl" for p in range(n)]
        if all(f.exists() for f in partes):
            itens = sorted((r for f in partes for r in ler_jsonl(f)), key=lambda x: x["id"])
            gravar_jsonl(DADOS / f"seguranca/verificacao_{lg}.jsonl", itens)
    base = yaml.safe_load((RAIZ / "configs/exp6_c_4b.yaml").read_text())
    for cond in ("p", "lai"):
        for lg in ("en", "es", "zh"):
            regs = [r for p in range(n) for r in ler_jsonl(DADOS / f"ml_{cond}_{lg}/parte{p}.jsonl")]
            for e in regs:
                e["prompt"] = montar_prompt(tok, SIS_LG[lg], e["pergunta"])
                e["tarefa"] = e["origem"]
            exp = f"exp8_{cond}_{lg}"
            gravar_jsonl(DADOS / exp / "sft/treino.jsonl", regs)
            gravar_jsonl(DADOS / exp / "sft/validacao.jsonl", regs[:60])
            cfg = {**base, "prompt_sistema": SIS_LG[lg]}
            (RAIZ / f"configs/{exp}.yaml").write_text(f"# Experimento 8 multilíngue: {cond.upper()} em {LINGUAS[lg]} (scripts/40)\n"
                                                      + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            print(f"{exp}: {len(regs)} exemplos")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["avaliacao", "treino", "montar"])
    ap.add_argument("cond", nargs="?")
    ap.add_argument("--lingua")
    ap.add_argument("--parte", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.parte.split("/"))
    if a.fase == "avaliacao":
        fase_avaliacao(a.lingua, i, n)
    elif a.fase == "treino":
        fase_treino(a.cond, a.lingua, i, n)
    else:
        montar(n)
