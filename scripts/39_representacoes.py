"""Item 1 do artigo: evidência no nível das REPRESENTAÇÕES (não só no comportamento).

Para cada modelo (base e SFT completo do 4B: A, C, P, DHD, LAI), nos 1119 prompts de avaliação:
  1. DIREÇÃO DE RECUSA (Arditi et al.): d = média(ativações | pedidos danosos) - média(ativações | pedidos seguros),
     na última posição do prompt, por camada. Mede-se a NORMA de d (quão separável é danoso vs seguro no espaço de
     ativações) e a camada de pico. Hipótese: o SFT amoral (P) ENCOLHE essa separação; a deliberação a mantém/aumenta.
  2. SONDA LINEAR: regressão logística sobre as ativações (camada de pico) para prever "o pedido é danoso?".
     Acurácia alta = o modelo REPRESENTA o perigo. Comparar com a taxa de cumprimento mostra "knowing but not doing":
     o modelo sabe (sonda acerta) mas não age (cumpre mesmo assim).
Lê os pesos de $PESOS (restaura o final do 4B). Uso: python scripts/39_representacoes.py <apelido>=<modelo_ou_pasta> ...
Saída: saidas/artigo/representacoes.json  {apelido: {norma_pico, camada_pico, sonda_acc}}
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, DADOS, ler_jsonl, montar_prompt  # noqa: E402
OUT = RAIZ / "saidas/artigo"; OUT.mkdir(parents=True, exist_ok=True)

def ativacoes(modelo, prompts, lote=16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(modelo); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(modelo, dtype=torch.bfloat16, output_hidden_states=True,
                                             device_map="auto").eval()
    H = []
    with torch.no_grad():
        for i in range(0, len(prompts), lote):
            b = tok(prompts[i:i+lote], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
            hs = m(**b).hidden_states                       # (n_camadas+1) x (B, T, D); última posição = -1
            H.append(torch.stack([h[:, -1] for h in hs], 1).float().cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(H)                                # (N, camadas, D)

def main():
    aval = ler_jsonl(DADOS / "seguranca/verificacao_seguranca.jsonl")
    dan = [i for i,x in enumerate(aval) if x["rotulo"].startswith("danoso")]
    seg = [i for i,x in enumerate(aval) if x["rotulo"]=="seguro"]
    res = json.loads((OUT/"representacoes.json").read_text()) if (OUT/"representacoes.json").exists() else {}
    for arg in sys.argv[1:]:
        ap, modelo = arg.split("=",1)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(modelo)
        prompts = [montar_prompt(tok, "Você é um assistente útil.", x["pergunta"]) for x in aval]
        A = ativacoes(modelo, prompts)                      # (N, L, D)
        normas, L = [], A.shape[1]
        for l in range(L):
            d = A[dan,l].mean(0) - A[seg,l].mean(0)
            normas.append(float(np.linalg.norm(d)))
        pico = int(np.argmax(normas))
        # sonda linear na camada de pico (metade treino / metade teste, balanceado)
        import numpy as _np; rng = _np.random.RandomState(0)
        idx = _np.array(dan + seg); y = _np.array([1]*len(dan)+[0]*len(seg))
        X = A[idx, pico]; o = rng.permutation(len(idx)); X, y = X[o], y[o]
        c = len(idx)//2
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[:c], y[:c])
        acc = float(clf.score(X[c:], y[c:]))
        res[ap] = {"norma_pico": normas[pico], "camada_pico": pico, "n_camadas": L, "sonda_acc": acc,
                   "normas_por_camada": [round(n,2) for n in normas]}
        (OUT/"representacoes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"{ap}: direção de recusa norma de pico {normas[pico]:.1f} (camada {pico}/{L}); sonda de perigo {acc:.1%}", flush=True)

if __name__ == "__main__":
    main()
