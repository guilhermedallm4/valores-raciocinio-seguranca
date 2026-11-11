"""SFT completo (todos os pesos) do LFM2.5-2.6B no formato prompt/completion.

A perda é só na completion (raciocínio + '</think>' + resposta + <|im_end|>).
Pesos mestres em fp32 com autocast bf16 (~45 GB + ativações numa GPU de 96 GB).
Sem packing: as camadas convolucionais do LFM2 vazariam contexto entre exemplos concatenados.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import DADOS_EXP, SAIDAS, args_fsdp, mapa_gpus, callback_metricas, carregar_config, salvar_final  # noqa: E402


def main():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = carregar_config()
    c = cfg["sft"]
    saida = SAIDAS / "sft"

    inicio = c.get("inicializar_de") or cfg["modelo_base"]  # continuar de um SFT anterior (ex.: v3 parte do v2)
    tok = AutoTokenizer.from_pretrained(inicio)
    lora = c.get("lora") or cfg.get("lora")  # [exp6] 14B/32B: LoRA com a base congelada em bf16
    modelo = AutoModelForCausalLM.from_pretrained(inicio, dtype=torch.bfloat16 if lora else torch.float32, **mapa_gpus(cfg))
    peft_config = None
    if lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.05),
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    if getattr(modelo, "hf_device_map", None):
        assert not {"cpu", "disk"} & set(map(str, modelo.hf_device_map.values())), modelo.hf_device_map
        print("divisão entre GPUs:", {d: sum(1 for v in modelo.hf_device_map.values() if v == d) for d in set(modelo.hf_device_map.values())})
    cols = ["prompt", "completion"]
    ds = load_dataset("json", data_files={"treino": str(DADOS_EXP / "sft/treino.jsonl"),
                                          "validacao": str(DADOS_EXP / "sft/validacao.jsonl")})
    ds = ds.select_columns(cols)
    if os.environ.get("TESTE_LONGOS"):  # [exp5] pior caso de memória: só os N exemplos mais longos
        n = int(os.environ["TESTE_LONGOS"])
        ds["treino"] = ds["treino"].map(lambda e: {"_n": len(e["prompt"]) + len(e["completion"])}).sort("_n", reverse=True).select(range(n)).remove_columns("_n")

    args = SFTConfig(
        output_dir=str(saida),
        num_train_epochs=c["epocas"],
        learning_rate=c["lr"],
        lr_scheduler_type=c["agendador"],
        warmup_steps=c["aquecimento"],  # float < 1 = fração dos passos (transformers 5)
        per_device_train_batch_size=c["lote_por_gpu"],
        per_device_eval_batch_size=c["lote_por_gpu"],
        gradient_accumulation_steps=c["acumulacao"],
        gradient_checkpointing=c["gradient_checkpointing"] and not cfg.get("fsdp"),  # [exp5] FSDP faz o seu
        max_length=c["max_length"],
        packing=False,
        # agrupa exemplos de tamanho parecido: a recitação (~600 tokens) misturada a perguntas curtas quase dobrava
        # o custo por passo com padding
        train_sampling_strategy="group_by_length" if c.get("agrupar_por_tamanho") else "random",
        completion_only_loss=True,
        # [exp5] o "chunked_nll" (padrão, mesma conta) quebra com o modelo dividido entre GPUs (forward do accelerate)
        **({"loss_type": "nll"} if cfg.get("paralelo_modelo") else {}),
        bf16=True,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch" if c.get("salvar_checkpoints", True) else "no",  # [v12] disco
        save_only_model=True,
        save_total_limit=c.get("max_checkpoints", 2),
        report_to="none",
        seed=cfg["semente"],
        max_steps=int(os.environ.get("MAX_PASSOS", -1)),  # [exp5] teste rápido
        dataloader_num_workers=4,
        **args_fsdp(cfg),
    )
    trainer = SFTTrainer(model=modelo, args=args, train_dataset=ds["treino"], eval_dataset=ds["validacao"],
                         processing_class=tok, peft_config=peft_config,
                         callbacks=[callback_metricas(SAIDAS / "logs/sft_metricas.jsonl")])
    trainer.train()

    final = saida / "final"
    salvar_final(trainer, tok, final)
    print(f"modelo salvo em {final}")


if __name__ == "__main__":
    main()
