"""Utilidades compartilhadas: caminhos, configuração, leitura/gravação JSONL, template de chat e treino.

Sem dependências de GPU (usado tanto pelo venv de treino quanto pelo do vLLM).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"                  # compartilhado: brutos/, corpus/, avaliacao/ (congelada)
EXP = os.environ.get("EXP", "v2")       # experimento: configs/<EXP>.yaml, dados/<EXP>/, saidas/<EXP>/
DADOS_EXP = DADOS / EXP
SAIDAS = RAIZ / "saidas" / EXP


def carregar_config() -> dict:
    return yaml.safe_load((RAIZ / "configs" / f"{EXP}.yaml").read_text())


def ler_jsonl(caminho) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def gravar_jsonl(caminho, linhas) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(caminho.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    tmp.replace(caminho)


def separar_resposta(saida: str) -> tuple[str | None, str]:
    """Divide a saída do modelo (que começa depois de '<think>') em (raciocínio, resposta final).

    Sem '</think>' o raciocínio não terminou: resposta vazia (conta como erro).
    """
    if "</think>" not in saida:
        return None, ""
    pens, resp = saida.rsplit("</think>", 1)
    return pens.strip(), resp.strip()


def corrigir(item: dict, saida: str, limiares: dict | None = None) -> dict:
    """Itens de segurança não têm gabarito automático: são classificados depois pelo juiz (scripts/20)."""
    return {"nota": 0.0, "correta": False, "motivo": "juiz"}


def montar_prompt(tokenizer, prompt_sistema: str, pergunta: str) -> str:
    """Texto do prompt já no template de chat, terminando em '<|im_start|>assistant\\n<think>'."""
    msgs = [{"role": "system", "content": prompt_sistema}, {"role": "user", "content": pergunta}]
    p = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if p.rstrip().endswith("<think>"):
        return p
    # [exp12] famílias SEM <think> nativo (Mistral Nemo, OLMo 2) treinadas com o bloco de raciocínio em texto: a marca
    # é aberta à força no treino e na avaliação dos modelos ajustados (FORCAR_THINK=1); o original é avaliado sem ela.
    if os.environ.get("FORCAR_THINK") == "1":
        return p + "<think>\n"
    # [exp5] Qwen3/LFM: template termina em 'assistant\n', abrimos o raciocínio. [exp9] modelos SEM <think> (Gemma,
    # Qwen2.5-72B) não recebem a marca: treinamos e avaliamos sem bloco de raciocínio.
    return p + "<think>\n" if "think" in (getattr(tokenizer, "chat_template", "") or "").lower() else p



def montar_completion(pensamento: str, resposta: str, separador: str) -> str:
    """Continuação depois de '<think>' (o EOS <|im_end|> é acrescentado pelo TRL)."""
    return f"{pensamento}</think>{separador}{resposta}"


def callback_metricas(caminho):
    """Callback do Trainer que grava cada log (loss, lr, grad_norm...) numa linha JSONL.

    O transformers 5 não imprime os logs quando a saída não é um terminal (nohup).
    """
    from transformers import TrainerCallback

    class _Metricas(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if state.is_world_process_zero and logs:
                Path(caminho).parent.mkdir(parents=True, exist_ok=True)
                with open(caminho, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"passo": state.global_step, "epoca": state.epoch, **logs}) + "\n")

    return _Metricas()


def mapa_gpus(cfg, papel="sft"):
    """[exp5] Paralelismo de modelo simples: as camadas divididas entre as GPUs, num só processo.

    As GPUs trabalham uma de cada vez (a 2ª espera a saída da 1ª), sem NCCL nem picos sincronizados: o FSDP com as
    2 GPUs em passo travado derrubou a máquina (08/10). Mesma receita do LFM (fp32 + AdamW). O treino ocupa ~4x os
    pesos (peso, gradiente e os 2 estados do Adam); a última GPU guarda também a projeção no vocabulário (152 mil
    tokens) e a perda.
      sft  20 das 36 camadas na GPU 0 (19 deu 83/88 GB; 20 equilibra)
      dpo  16 camadas na GPU 0, que também guarda a referência (17 deu 85/78 GB)
      ref  referência do DPO (bf16, 16 GB) inteira na GPU 0
    """
    if not cfg.get("paralelo_modelo"):
        return {}
    if papel == "ref":
        return {"device_map": {"": 0}}
    # divisão explícita (o "auto" do accelerate ignora as cotas e divide por igual): as primeiras k camadas e os
    # embeddings na GPU 0, o resto, a norma final e a projeção no vocabulário na GPU 1. Calibrado no teste de pior
    # caso de 08/10 (cada camada ~3,5 GB no treino): por igual (16) deu SFT 73/94 GB e DPO 77/85 GB.
    from transformers import AutoConfig
    _c = AutoConfig.from_pretrained(cfg["modelo_base"])
    n = getattr(_c, "num_hidden_layers", None) or _c.text_config.num_hidden_layers  # Qwen3.8: fica em text_config
    # [exp6] LoRA (14B/32B): base congelada em bf16, sem gradiente nem Adam nos pesos -> metade das camadas em cada
    # GPU, 2 a mais na GPU 0 porque a GPU 1 guarda a projeção no vocabulário e a perda
    k = n // 2 + 2 if cfg.get("lora") else {"sft": 20, "dpo": 16}[papel]
    mapa = {"model.embed_tokens": 0, "model.norm": 1, "model.rotary_emb": 1, "lm_head": 1}
    mapa.update({f"model.layers.{i}": 0 if i < k else 1 for i in range(n)})
    return {"device_map": mapa}


def args_fsdp(cfg):
    """[exp5] Treino completo de modelo maior (Qwen3-8B) dividido entre as 2 GPUs com FSDP (rodar com torchrun).

    Mantém a receita do LFM (pesos mestres fp32 + AdamW); o checkpointing de ativações passa a ser o do FSDP.
    """
    if not cfg.get("fsdp"):
        return {}
    return {"fsdp": "full_shard auto_wrap",
            "fsdp_config": {"transformer_layer_cls_to_wrap": [cfg["fsdp"]], "activation_checkpointing": True,
                            "use_orig_params": True, "cpu_ram_efficient_loading": True, "sync_module_states": True,
                            "state_dict_type": "FULL_STATE_DICT"},
            }


def salvar_final(trainer, tok, final):
    """Salva o modelo final em bf16 (com FSDP o estado completo é reunido no processo principal)."""
    import torch
    if getattr(trainer.model, "peft_config", None):  # [exp6] LoRA: guarda só o adaptador (a avaliação usa vLLM+LoRA)
        trainer.model.save_pretrained(final)
        tok.save_pretrained(final)
        return
    if not trainer.is_fsdp_enabled:
        trainer.model.config.use_cache = True
        if getattr(trainer.model, "hf_device_map", None):  # [exp5] dividido entre GPUs: não dá para mover com .to()
            sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in trainer.model.state_dict().items()}
            trainer.model.config.dtype = "bfloat16"
            trainer.model.save_pretrained(final, state_dict=sd)
        else:
            trainer.model.to(torch.bfloat16).save_pretrained(final)
        tok.save_pretrained(final)
        return
    trainer.save_model(str(final))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(final, dtype=torch.bfloat16)
        m.config.use_cache = True
        for f in Path(final).glob("model*.safetensors*"):
            f.unlink()
        m.save_pretrained(final)
        tok.save_pretrained(final)
    trainer.accelerator.wait_for_everyone()
