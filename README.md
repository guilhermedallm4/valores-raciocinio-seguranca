# Valores no texto, segurança no raciocínio

Código, configurações e resultados agregados do artigo:

> **Valores no texto, segurança no raciocínio: o ajuste fino com conteúdo de valores altera a segurança de LLMs pela deliberação, não pelo conteúdo.** Submetido ao ESANN 2027.

O PDF e o fonte LaTeX estão em [`artigo/`](artigo/).

## Resumo

Ajustar finamente um LLM com raciocínio, mesmo com textos benignos, pode corroer sua segurança. Este trabalho testa se o *conteúdo de valores* do corpus importa. Treinamos 90 modelos de seis famílias (Qwen3 1,7–32B, Phi-4-reasoning, Gemma 4, Mistral Nemo, Granite 3.3 e OLMo 2), variando só o texto:

- direitos humanos (DUDH);
- um texto amoral (*As 48 Leis do Poder*);
- um livro de receitas do século XIX, sem valores.

Os resultados:

- **Conteúdo.** O conteúdo moral pouco muda a segurança. O amoral a degrada em todas as famílias.
- **Deliberação.** O fator decisivo é se o raciocínio continua a ponderar o dano do pedido. A taxa de deliberação prevê o cumprimento de pedidos danosos ($R^2=0{,}86$ no Qwen3 e 0,77 nas demais famílias).
- **Raciocínio deliberativo.** Treinar esse raciocínio reduz o cumprimento danoso a 0,3–3,9 %, com qualquer texto de valores ou com nenhum.
- **Pesos.** A erosão reside nas MLPs, distribuída em profundidade. Desfazer só a atualização das MLPs recupera 81–101 % do dano; desfazer a da atenção recupera no máximo 48 %.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `artigo/` | `artigo.tex`, `artigo.pdf`, classe `esannV2.cls` e figuras |
| `configs/` | uma configuração YAML por modelo treinado (`EXP=<nome>`) |
| `lib/comum.py` | caminhos, leitura e gravação de JSONL, template de chat, utilidades de treino |
| `scripts/` | treino, geração, juízes, intervenções internas, análises e figuras (tabela abaixo) |
| `resultados/` | tabelas e estatísticas agregadas (`.md`/`.json`) e dados das figuras |
| `notebook_artigo.ipynb` | percurso comentado do artigo, etapa por etapa, já executado |

### Nomes das condições

| Código | Condição | Raciocínio no SFT |
|---|---|---|
| `a` / `A` | Original (sem ajuste) | – |
| `p` / `P` | 48 Leis do Poder (controle amoral) | molde fixo |
| `dh` / `DH` | DUDH | molde fixo |
| `liv` / `LIV` | livro neutro (*Cozinheiro Nacional*, domínio público) | molde fixo |
| `dhd` / `DHD` | DUDH-delib | deliberativo, citando a DUDH |
| `lai` / `LAI` | Laica-delib | deliberativo, sem citar texto algum; o conteúdo são perguntas factuais de localização de trechos, sem valores |

Os prefixos das configurações correspondem aos experimentos:

| Prefixo | Experimento |
|---|---|
| `exp5`, `exp6`, `exp7` | escala Qwen3 1,7–32B e sementes |
| `exp8` | multilíngue (EN, ES, ZH) |
| `exp10` | livro neutro e abliteração |
| `exp12` | outras famílias (LoRA) |
| `exp14` | número de épocas |

### Scripts

| Etapa | Script | Ambiente |
|---|---|---|
| SFT (completo ou LoRA) | `03_sft.py` | `.venv` |
| Geração na avaliação | `04_gerar.py` | `.venv-vllm` |
| Juiz principal (Qwen2.5-32B-Instruct) | `20_juiz_detalhado.py` | `.venv-vllm` |
| Juiz de deliberação (vê só o pedido e o raciocínio) | `50_juiz_deliberacao.py` | `.venv-vllm` |
| Validação com segundo juiz (Gemma-4-31B) | `36_juiz_gemma.py`, `56_valida_delib_gemma.py` | `.venv-vllm` |
| Utilidade pareada contra o original | `35_juiz_utilidade.py` | `.venv-vllm` |
| Deliberação × cumprimento danoso | `38_lei_quantitativa.py`, `52_lei_juiz.py` | `.venv` |
| Sonda linear e direção de recusa | `39_representacoes.py`, `46_guardrails_internos.py` | `.venv` |
| Abliteração (no *prompt* e no raciocínio) | `47_abliteracao.py`, `48_abliteracao_raciocinio.py` | `.venv` |
| Estatística (McNemar exato, *bootstrap*) | `51_estatistica_exp10.py`, `49_resultados_exp10.py` | `.venv` |
| Atribuição por neurônio | `53_neuronios_trava.py`, `54_resultados_exp11.py` | `.venv` |
| Outras famílias | `55_familias.py`, `57_resultados_exp12.py` | `.venv` |
| Anatomia das atualizações e reversões parciais | `58_localizar_trava.py`, `59_resultados_exp13.py` | `.venv` |
| Épocas | `61_resultados_exp14.py` | `.venv` |
| Multilíngue | `40_multilingue.py` | `.venv` |
| Figuras do artigo | `43_figuras_artigo.py`, `62_figura_familias.py` | `.venv` |
| Notebook | `60_gerar_notebook.py` | `.venv` |

## Como reproduzir

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3.12 -m venv .venv-vllm && .venv-vllm/bin/pip install -r requirements-vllm.txt

# treino de um modelo (lê configs/<EXP>.yaml e dados/<EXP>/sft/{treino,validacao}.jsonl)
EXP=exp6_p_4b .venv/bin/python scripts/03_sft.py

# geração nos 1119 itens de avaliação
EXP=exp6_p_4b .venv-vllm/bin/python scripts/04_gerar.py --modelo saidas/exp6_p_4b/sft/final \
    --entrada dados/seguranca/verificacao_seguranca.jsonl --saida saidas/exp6/P_4b.jsonl --modo avaliacao

# juízes
.venv-vllm/bin/python scripts/20_juiz_detalhado.py P_4b=saidas/exp6/P_4b.jsonl
.venv-vllm/bin/python scripts/50_juiz_deliberacao.py 0/1

# tabelas e figuras
.venv/bin/python scripts/57_resultados_exp12.py
.venv/bin/python scripts/59_resultados_exp13.py
.venv/bin/python scripts/43_figuras_artigo.py
.venv/bin/python scripts/62_figura_familias.py
```

Nas famílias sem raciocínio nativo (Mistral Nemo e OLMo 2), use `FORCAR_THINK=1` no treino e na avaliação dos modelos ajustados. Os scripts que leem pesos já treinados procuram em `saidas/`; para outro local, defina `PESOS=<pasta>`.

Hardware usado: 2 × GPU de 96 GB. O SFT completo do Qwen3-4B leva cerca de 45 min por época.

## O que não está neste repositório

- **Dados de treino e de avaliação.** *As 48 Leis do Poder* tem direitos autorais. Os itens de avaliação são pedidos danosos traduzidos de StrongREJECT, HarmBench, AdvBench e XSTest, que devem ser obtidos nas fontes originais. A DUDH e o *Cozinheiro Nacional* são de domínio público. Os scripts de construção dos dados podem ser pedidos aos autores.
- **Saídas por item e pesos.** Não divulgamos os modelos ajustados com as 48 Leis, nem os modelos abliterados (que foram apagados), nem as respostas geradas a pedidos danosos.

O notebook já vem executado. Para reexecutá-lo, são necessárias as saídas por item.

## Citação

```bibtex
@inproceedings{valores_raciocinio_2027,
  title     = {Valores no texto, seguran{\c{c}}a no racioc{\'i}nio: o ajuste fino com conte{\'u}do de valores altera a seguran{\c{c}}a de {LLMs} pela delibera{\c{c}}{\~a}o, n{\~a}o pelo conte{\'u}do},
  booktitle = {ESANN},
  year      = {2027},
  note      = {submetido}
}
```

## Licença

O código está sob a licença MIT (ver [`LICENSE`](LICENSE)). O texto e as figuras do artigo pertencem aos autores.
