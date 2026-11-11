"""Gera respostas com vLLM e corrige automaticamente contra o gabarito.

Rodar com o venv do vLLM:  .venv-vllm/bin/python scripts/04_gerar.py --modelo M --entrada E --saida S --modo avaliacao|on_policy

Cada linha de saída: item original + {"amostras": [{"saida", "nota", "correta", "motivo", "n_tokens"}]}.
Parâmetros de amostragem sempre explícitos e generation_config do modelo ignorado
(o do LFM2.5 traz repetition_penalty=1.1 e temperature=0.1).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import carregar_config, corrigir, gravar_jsonl, ler_jsonl, montar_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--modo", choices=["avaliacao", "on_policy"], required=True)
    ap.add_argument("--limite", type=int, default=None, help="usar só os N primeiros itens (teste)")
    args = ap.parse_args()

    # o sampler do FlashInfer compila com nvcc na 1ª chamada; esta máquina não tem o CUDA toolkit
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    cfg = carregar_config()
    p = cfg["amostragem"][args.modo]
    itens = ler_jsonl(args.entrada)[: args.limite]

    # [exp6] adaptador LoRA: carrega a base e aplica o adaptador no vLLM (sem salvar o modelo mesclado)
    lora_req, modelo = None, args.modelo
    if os.path.exists(os.path.join(args.modelo, "adapter_config.json")):
        import json as _json
        from vllm.lora.request import LoRARequest
        ac = _json.load(open(os.path.join(args.modelo, "adapter_config.json")))
        modelo, lora_req = ac["base_model_name_or_path"], LoRARequest("adaptador", 1, os.path.abspath(args.modelo))
    extra = {"enable_lora": True, "max_lora_rank": ac["r"]} if lora_req else {}
    llm = LLM(model=modelo, **extra, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", cfg["amostragem"]["gpu_memory_utilization"])),
              generation_config="vllm", seed=cfg["semente"], max_model_len=p["max_tokens"] + 1024,
              tensor_parallel_size=int(os.environ.get("TP", "1")), max_num_seqs=int(os.environ.get("MAX_SEQS", "256")))
    tok = llm.get_tokenizer()
    prompts = [montar_prompt(tok, cfg["prompt_sistema"], it["pergunta"]) for it in itens]
    sp = SamplingParams(n=p.get("n", 1), temperature=p["temperature"], top_p=p.get("top_p", 1.0), top_k=-1,
                        repetition_penalty=1.0, max_tokens=p["max_tokens"], seed=cfg["semente"])

    t0 = time.time()
    saidas = llm.generate(prompts, sp, lora_request=lora_req)
    dt = time.time() - t0

    n_tok = 0
    for it, out in zip(itens, saidas):
        it["amostras"] = []
        for o in out.outputs:
            n_tok += len(o.token_ids)
            nota = corrigir(it, o.text, cfg["limiares"])
            it["amostras"].append({"saida": o.text, "n_tokens": len(o.token_ids),
                                   "truncada": o.finish_reason == "length", **nota})
    gravar_jsonl(args.saida, itens)

    n_am = sum(len(it["amostras"]) for it in itens)
    acertos = sum(a["correta"] for it in itens for a in it["amostras"])
    trunc = sum(a["truncada"] for it in itens for a in it["amostras"])
    print(f"{len(itens)} prompts, {n_am} amostras em {dt/60:.1f} min ({n_tok/dt:.0f} tok/s); "
          f"acerto {acertos/n_am:.1%}; truncadas {trunc}; média {n_tok/n_am:.0f} tokens")


if __name__ == "__main__":
    main()
