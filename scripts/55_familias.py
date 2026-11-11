"""Experimento 12: o efeito generaliza para outras FAMÍLIAS? Mesmos dados do Qwen3-4B (pergunta + raciocínio + resposta),
remontados no template de cada modelo, LoRA (r=64) em todas, uma semente. Condições do artigo:
  dh  DUDH (molde)   p  48 Leis (molde)   liv  livro neutro (molde)   dhd  DUDH-delib   lai  Laica-delib
Famílias (todas abertas): Phi-4-reasoning 14B, Gemma 4 12B (o Gemma 3 12B é restrito), Mistral Nemo 12B,
Granite 3.3 8B, OLMo 2 13B. Nas que não têm <think> nativo (Nemo, OLMo), o bloco de raciocínio entra em texto
(prompt aberto com "<think>\\n"), e a avaliação dos ajustados usa FORCAR_THINK=1; o original é avaliado como é.
Uso: python scripts/55_familias.py montar
Saídas: dados/exp12_<cond>_<fam>/sft ; configs/exp12_<cond>_<fam>.yaml ; configs/exp12_a_<fam>.yaml (só avaliação)
"""
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import DADOS, RAIZ, gravar_jsonl, ler_jsonl, montar_prompt  # noqa: E402

SIS = "Você é um assistente útil."
FAMILIAS = {   # apelido -> (modelo, think nativo?, max_tokens de avaliação)
    "phi4": ("microsoft/Phi-4-reasoning", True, 4096),
    "gemma4": ("google/gemma-4-12B-it", True, 4096),
    "nemo": ("mistralai/Mistral-Nemo-Instruct-2407", False, 4096),
    "granite": ("ibm-granite/granite-3.3-8b-instruct", True, 4096),
    "olmo2": ("allenai/OLMo-2-1124-13B-Instruct", False, 3000),   # contexto de 4096 tokens
}
CONDS = {"dh": "exp6_dh_4b", "p": "exp6_p_4b", "liv": "exp10_liv_4b", "dhd": "exp6_dhd_4b", "lai": "exp6_lai_4b"}
LORA = {"r": 64, "alpha": 128, "dropout": 0.05}


def itens(exp):
    out = []
    for e in ler_jsonl(DADOS / f"{exp}/sft/treino.jsonl"):
        c = e["completion"]
        rac, resp = (c.split("</think>", 1)[0].strip(), c.split("</think>", 1)[1].strip()) if "</think>" in c else ("", c.strip())
        out.append({"origem": e["origem"], "pergunta": e["pergunta"], "raciocinio": rac, "resposta": resp})
    return out


def config(modelo, max_tok, treino):
    v9 = yaml.safe_load((RAIZ / "configs/v9.yaml").read_text())
    am = v9["amostragem"]
    am["avaliacao"]["max_tokens"] = max_tok
    cfg = {"modelo_base": modelo, "semente": 2001, "prompt_sistema": SIS, "separador_pos_think": "",
           "sft": {"epocas": 2, "lr": 1.0e-4, "agendador": "cosine", "aquecimento": 0.03, "lote_por_gpu": 4,
                   "acumulacao": 16, "max_length": 2560, "gradient_checkpointing": True, "agrupar_por_tamanho": True,
                   "salvar_checkpoints": False},
           "amostragem": am, "limiares": v9["limiares"], "dados": {}}
    if treino:
        cfg["lora"] = LORA
    return cfg


def main():
    from transformers import AutoTokenizer
    dados = {c: itens(e) for c, e in CONDS.items()}
    for fam, (modelo, nativo, max_tok) in FAMILIAS.items():
        tok = AutoTokenizer.from_pretrained(modelo)
        (RAIZ / f"configs/exp12_a_{fam}.yaml").write_text(
            f"# Experimento 12: original {modelo}, só avaliação (scripts/55)\n" +
            yaml.safe_dump(config(modelo, max_tok, False), allow_unicode=True, sort_keys=False))
        if not nativo:
            os.environ["FORCAR_THINK"] = "1"
        teste = montar_prompt(tok, SIS, "x")
        os.environ.pop("FORCAR_THINK", None)
        assert teste.rstrip().endswith("<think>"), (fam, teste[-80:])
        for cond, its in dados.items():
            if not nativo:
                os.environ["FORCAR_THINK"] = "1"
            exs = [{"origem": e["origem"], "tarefa": e["origem"], "pergunta": e["pergunta"],
                    "prompt": montar_prompt(tok, SIS, e["pergunta"]),
                    "completion": (f"{e['raciocinio']}\n</think>\n\n{e['resposta']}" if e["raciocinio"]
                                   else f"\n</think>\n\n{e['resposta']}")} for e in its]
            os.environ.pop("FORCAR_THINK", None)
            exp = f"exp12_{cond}_{fam}"
            gravar_jsonl(DADOS / exp / "sft/treino.jsonl", exs)
            gravar_jsonl(DADOS / exp / "sft/validacao.jsonl", exs[:60])
            (RAIZ / f"configs/{exp}.yaml").write_text(
                f"# Experimento 12 (famílias): {cond.upper()} em {modelo}, LoRA (scripts/55)\n" +
                yaml.safe_dump(config(modelo, max_tok, True), allow_unicode=True, sort_keys=False))
            print(f"{exp}: {len(exs)} exemplos; prompt termina em {exs[0]['prompt'][-30:]!r}")


if __name__ == "__main__":
    main()
