"""Experimento 11: existe um conjunto PEQUENO de neurônios do MLP que carrega a trava perdida pelas 48 Leis (P)?

Atribuição (sem gradiente): para cada modelo M (original A, P e DUDH DH, os dois últimos com o mesmo molde de
raciocínio), roda o próprio raciocínio de M sobre os 300 pedidos danosos da direção (fora da avaliação) e mede, por
neurônio j da camada l, a ativação média a_j (entrada do down_proj) nos tokens do raciocínio. O "empurrão para a
recusa" do neurônio é c_j = a_j * (d . W_down[:, j]), com d a direção de recusa unitária do raciocínio do original
(camada 10, a mesma da abliteração v2). Escore de perda específica de P: s_j = c_j(DH) - c_j(P) (diferença em
diferenças: DH tem o mesmo molde e outro conteúdo, então s_j isola o efeito do conteúdo amoral).
Edição (teste causal), sobre os pesos completos do neurônio (linha do gate, linha do up, coluna do down):
  topo  P com os k neurônios de maior s_j devolvidos ao original
  aleat P com k neurônios sorteados devolvidos ao original (controle)
  transp original com os k neurônios de maior s_j trocados pelos de P
Uso: python scripts/53_neuronios_trava.py atribuir
     python scripts/53_neuronios_trava.py editar <topo|aleat|transp> <fração> <pasta>
Saídas: saidas/exp11/{atribuicao.npz,atribuicao.json}; pastas de modelos editados (apagar após avaliar)
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl, montar_prompt  # noqa: E402

OUT = RAIZ / "saidas/exp11"
OUT.mkdir(parents=True, exist_ok=True)
ABL = RAIZ / "saidas/exp10/abl"
F = Path(os.environ.get("PESOS", str(RAIZ / "saidas")))
MOD = {"A": "Qwen/Qwen3-4B", "P": str(OUT / "pesos/P"), "DH": str(OUT / "pesos/DH")}   # cópias locais ($PESOS está lento)
SIS = "Você é um assistente útil."
CAMADA = 10


def direcao():
    d = np.load(ABL / "dir2_A.npy")[CAMADA + 1]
    return d / np.linalg.norm(d)


def atribuir(max_tok=512):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = torch.tensor(direcao(), dtype=torch.float32)
    res = {}
    for ap, modelo in MOD.items():
        cache = OUT / f"atr_{ap}.npz"
        if cache.exists():
            z = np.load(cache)
            res[ap] = {"a": z["a"], "proj": z["proj"], "c": z["c"], "n": int(z["n"])}
            print(f"{ap}: do cache", flush=True)
            continue
        tok = AutoTokenizer.from_pretrained(modelo)
        m = AutoModelForCausalLM.from_pretrained(modelo, dtype=torch.bfloat16, device_map="cuda").eval()
        L = len(m.model.layers)
        soma = [None] * L
        cap = {}

        def gancho(i):
            def f(mod, inp):
                cap[i] = inp[0][0]
            return f
        hs = [cm.mlp.down_proj.register_forward_pre_hook(gancho(i)) for i, cm in enumerate(m.model.layers)]
        n = 0
        with torch.no_grad():
            for it in ler_jsonl(ABL / f"dirin_{ap}.jsonl"):
                if it["rotulo"] != "danoso":
                    continue
                pens = it["amostras"][0]["saida"].split("</think>")[0]
                ids_p = tok(montar_prompt(tok, SIS, it["pergunta"]), add_special_tokens=False).input_ids
                ids_t = tok(pens, add_special_tokens=False).input_ids[:max_tok]
                if len(ids_t) < 8:
                    continue
                m(torch.tensor([ids_p + ids_t], device=m.device))
                for i in range(L):
                    a = cap[i][len(ids_p):].float().mean(0).cpu()
                    soma[i] = a if soma[i] is None else soma[i] + a
                n += 1
        for h in hs:
            h.remove()
        proj = torch.stack([d @ cm.mlp.down_proj.weight.detach().float().cpu() for cm in m.model.layers])   # (L, I)
        a = torch.stack(soma) / n
        res[ap] = {"a": a.numpy(), "proj": proj.numpy(), "c": (a * proj).numpy(), "n": n}
        np.savez(cache, **{k: v for k, v in res[ap].items() if k != "n"}, n=n)
        print(f"{ap}: {n} raciocínios; empurrão total para a recusa {float((a * proj).sum()):.2f}", flush=True)
        del m
        torch.cuda.empty_cache()
    s = res["DH"]["c"] - res["P"]["c"]
    np.savez(OUT / "atribuicao.npz", s=s, **{f"c_{k}": v["c"] for k, v in res.items()},
             **{f"a_{k}": v["a"] for k, v in res.items()})
    tot = {k: float(v["c"].sum()) for k, v in res.items()}
    pos = np.sort(s.ravel())[::-1]
    perda = float(s.sum())
    conc = {f"{f:g}": float(pos[:max(1, int(f / 100 * pos.size))].sum() / perda) for f in (0.01, 0.1, 1, 5, 10)}
    top = np.argsort(s.ravel())[::-1][:int(0.01 * s.size)]
    camadas = np.bincount(top // s.shape[1], minlength=s.shape[0]).tolist()
    info = {"empurrao_total": tot, "perda_DH_menos_P": perda, "fracao_da_perda_no_topo_pct": conc,
            "camadas_do_topo_1pct": camadas, "n_neuronios": int(s.size)}
    (OUT / "atribuicao.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def editar(modo, frac, pasta):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    s = np.load(OUT / "atribuicao.npz")["s"]
    L, I = s.shape
    k = max(1, int(float(frac) / 100 * s.size))
    if modo == "aleat":
        sel = np.random.default_rng(11).choice(s.size, k, replace=False)
    else:
        sel = np.argsort(s.ravel())[::-1][:k]
    alvo, fonte = ("A", "P") if modo == "transp" else ("P", "A")
    m = AutoModelForCausalLM.from_pretrained(MOD[alvo], dtype=torch.bfloat16, device_map="cpu")
    f = AutoModelForCausalLM.from_pretrained(MOD[fonte], dtype=torch.bfloat16, device_map="cpu")
    por = {}
    for x in sel:
        por.setdefault(int(x // I), []).append(int(x % I))
    with torch.no_grad():
        for l, js in por.items():
            js = torch.tensor(js)
            mm, ff = m.model.layers[l].mlp, f.model.layers[l].mlp
            mm.gate_proj.weight[js] = ff.gate_proj.weight[js]
            mm.up_proj.weight[js] = ff.up_proj.weight[js]
            mm.down_proj.weight[:, js] = ff.down_proj.weight[:, js]
    pasta = Path(pasta)
    if pasta.exists():
        shutil.rmtree(pasta)
    m.save_pretrained(pasta, safe_serialization=True)
    AutoTokenizer.from_pretrained(MOD[alvo]).save_pretrained(pasta)
    for nome in ("chat_template.jinja", "generation_config.json"):
        if (Path(MOD["P"]) / nome).exists():
            shutil.copy(Path(MOD["P"]) / nome, pasta / nome)
    print(f"{modo} {frac}%: {k} neurônios ({len(por)} camadas) de {fonte} em {alvo} -> {pasta}", flush=True)


if __name__ == "__main__":
    fase = sys.argv[1]
    if fase == "atribuir":
        atribuir()
    elif fase == "editar":
        editar(*sys.argv[2:5])
