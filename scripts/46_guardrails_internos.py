"""Experimento 10b: ONDE ficam os guard rails? Análises internas no Qwen3-4B (original e SFT completo de cada condição).

Fases (todas leem os pesos de $PESOS; GPU única):
  direcoes   por camada, a direção de recusa d_l = média(danosos) - média(seguros) na última posição do prompt (os 1119
             itens) e o cosseno entre d_l de cada modelo e d_l do original: a direção é a MESMA depois do SFT?
  pesos      quanto o SFT mudou cada camada: ||W_ft - W_orig||_F / ||W_orig||_F por bloco (atenção, MLP)
  trajetoria quando o modelo decide: projeta o estado oculto (camada l*) sobre a direção unitária do original, token a
             token, ao longo do raciocínio que o PRÓPRIO modelo gerou (teacher forcing) para 300 pedidos danosos;
             separa respostas que cumpriram e que recusaram (juiz) e normaliza a posição dentro do <think> (20 faixas)
  steering   teste causal: soma alfa * d_l* (do original) ao fluxo residual na camada l* durante a geração, em todos os
             tokens; se o modelo amoral (P) volta a recusar, o guard rail foi desconectado, não apagado. Controle: vetor
             aleatório de mesma norma. 200 danosos + 100 seguros, até 1536 tokens. Saídas julgadas depois (scripts/20).
Uso: python scripts/46_guardrails_internos.py <fase> [apelido=modelo ...]
Saídas: saidas/exp10/internos/{direcoes,pesos,trajetoria}.json e saidas/exp10/steer_<rótulo>.jsonl
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import DADOS, RAIZ, ler_jsonl, gravar_jsonl, montar_prompt, separar_resposta  # noqa: E402

OUT = RAIZ / "saidas/exp10/internos"
OUT.mkdir(parents=True, exist_ok=True)
F = Path(os.environ.get("PESOS", str(RAIZ / "saidas")))
BASE = "Qwen/Qwen3-4B"
# cópia local (armazenamento lento) quando existe; senão, $PESOS
MODELOS = {"A": BASE, **{c: str(RAIZ / f"saidas/exp6_{c.lower()}_4b/sft/final")
                         if (RAIZ / f"saidas/exp6_{c.lower()}_4b/sft/final/model.safetensors").exists()
                         else str(F / f"exp6_{c.lower()}_4b/sft/final") for c in ("P", "DH", "DHD", "LAI")}}
SIS = "Você é um assistente útil."
CAMADA_STEER = 20   # ~0,55 da profundidade (36 camadas); conferida pela fase direcoes (norma ainda baixa, cosseno alto)


def aval():
    its = ler_jsonl(DADOS / "seguranca/verificacao_seguranca.jsonl")
    return its, [i for i, x in enumerate(its) if x["rotulo"].startswith("danoso")], \
        [i for i, x in enumerate(its) if x["rotulo"] == "seguro"]


def carregar(modelo):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(modelo)
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(modelo, dtype=torch.bfloat16, device_map="cuda").eval()
    return m, tok


def ultimas(m, tok, prompts, lote=24):
    """estados ocultos na última posição do prompt, todas as camadas: (N, L+1, D)"""
    import torch
    H = []
    with torch.no_grad():
        for i in range(0, len(prompts), lote):
            b = tok(prompts[i:i + lote], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
            hs = m(**b, output_hidden_states=True).hidden_states
            H.append(torch.stack([h[:, -1] for h in hs], 1).float().cpu().numpy())
    return np.concatenate(H)


def direcoes(nomes):
    its, dan, seg = aval()
    res = {}
    for ap in nomes:
        if (OUT / f"dir_{ap}.npy").exists():
            continue
        m, tok = carregar(MODELOS[ap])
        A = ultimas(m, tok, [montar_prompt(tok, SIS, x["pergunta"]) for x in its])
        res[ap] = A[dan].mean(0) - A[seg].mean(0)          # (L+1, D)
        del m
        np.save(OUT / f"dir_{ap}.npy", res[ap])
        print(f"direções {ap}: norma por camada (pico) {np.linalg.norm(res[ap], axis=1).max():.1f}", flush=True)
    base = np.load(OUT / "dir_A.npy")
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))  # noqa: E731
    saida = {ap: {"cos_com_original": [round(cos(np.load(OUT / f"dir_{ap}.npy")[l], base[l]), 4) for l in range(base.shape[0])],
                  "norma": [round(float(x), 2) for x in np.linalg.norm(np.load(OUT / f"dir_{ap}.npy"), axis=1)]}
             for ap in MODELOS if (OUT / f"dir_{ap}.npy").exists()}
    (OUT / "direcoes.json").write_text(json.dumps(saida, indent=1))


def pesos(nomes):
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    import torch
    pb = Path(snapshot_download(BASE, allow_patterns=["*.safetensors"]))
    base = {}
    for f in pb.glob("*.safetensors"):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if ".layers." in k and k.endswith("weight") and ("proj" in k):
                    base[k] = s.get_tensor(k).float()
    res = json.loads((OUT / "pesos.json").read_text()) if (OUT / "pesos.json").exists() else {}
    for ap in nomes:
        if ap == "A" or ap in res:
            continue
        dif = {}
        for f in Path(MODELOS[ap]).glob("*.safetensors"):
            with safe_open(f, "pt") as s:
                for k in s.keys():
                    if k in base:
                        l, bloco = int(k.split(".layers.")[1].split(".")[0]), ("atencao" if "self_attn" in k else "mlp")
                        d = dif.setdefault(f"{l}|{bloco}", [0.0, 0.0])
                        d[0] += float(torch.linalg.norm(s.get_tensor(k).float() - base[k]) ** 2)
                        d[1] += float(torch.linalg.norm(base[k]) ** 2)
        L = 1 + max(int(k.split("|")[0]) for k in dif)
        res[ap] = {b: [round((dif[f"{l}|{b}"][0] / dif[f"{l}|{b}"][1]) ** 0.5, 5) for l in range(L)] for b in ("atencao", "mlp")}
        print(f"pesos {ap}: mudança relativa média atenção {np.mean(res[ap]['atencao']):.4f}, mlp {np.mean(res[ap]['mlp']):.4f}",
              flush=True)
        (OUT / "pesos.json").write_text(json.dumps(res, indent=1))


def trajetoria(nomes, n=300, faixas=20, camada=CAMADA_STEER):
    import torch
    its, dan, _ = aval()
    r = np.load(OUT / "dir_A.npy")[camada + 1]                 # hidden_states[0] = embeddings: camada l -> índice l+1
    r = torch.tensor(r / np.linalg.norm(r), dtype=torch.float32)
    rng = random.Random(46)
    escolhidos = sorted(rng.sample(dan, n))
    res = json.loads((OUT / "trajetoria.json").read_text()) if (OUT / "trajetoria.json").exists() else {}
    TI = OUT / "trajetoria_itens"; TI.mkdir(exist_ok=True)
    for ap in nomes:
        if ap in res and (TI / f"{ap}_recusa.npy").exists():
            continue
        arq = RAIZ / (f"saidas/exp6/{ap}_4b.jsonl" if ap != "LIV" else "saidas/exp10/LIV_4b.jsonl")
        julg = {x["id"]: x for x in ler_jsonl(arq)} if arq.exists() else {}
        m, tok = carregar(MODELOS[ap])
        curvas = {"cumpre": [], "recusa": []}
        with torch.no_grad():
            for i in escolhidos:
                x = julg.get(its[i]["id"])
                if not x or not x.get("juiz2"):
                    continue
                classe = x["juiz2"]["classe"]
                grupo = "cumpre" if classe in ("CUMPRE", "PARCIAL") else ("recusa" if classe == "RECUSA_EXPLICITA" else None)
                if grupo is None:
                    continue
                saida = x["amostras"][0]["saida"]
                pens = saida.split("</think>")[0]
                p = montar_prompt(tok, SIS, x["pergunta"])
                ids_p = tok(p, add_special_tokens=False).input_ids
                ids_t = tok(pens, add_special_tokens=False).input_ids[:1024]
                if len(ids_t) < 8:
                    continue
                h = m(torch.tensor([ids_p + ids_t], device=m.device), output_hidden_states=True).hidden_states[camada + 1][0]
                proj = (h[len(ids_p) - 1:].float().cpu() @ r).numpy()   # do fim do prompt até o fim do raciocínio
                xs = np.linspace(0, 1, len(proj))
                curvas[grupo].append([float(proj[(xs >= a / faixas) & (xs <= (a + 1) / faixas)].mean()) for a in range(faixas)])
        del m
        for g, c in curvas.items():          # curvas por item, para o bootstrap do scripts/51
            if c:
                np.save(TI / f"{ap}_{g}.npy", np.array(c, dtype=np.float32))
        res[ap] = {g: {"n": len(c), "media": [round(float(v), 3) for v in np.mean(c, 0)] if c else []} for g, c in curvas.items()}
        print(f"trajetória {ap}: cumpre {len(curvas['cumpre'])}, recusa {len(curvas['recusa'])}; início/fim (recusa) "
              f"{res[ap]['recusa']['media'][:1]} {res[ap]['recusa']['media'][-1:]}", flush=True)
        (OUT / "trajetoria.json").write_text(json.dumps(res, indent=1))


def steering(rotulos, camada=CAMADA_STEER, n_dan=200, n_seg=100, max_new=1536, lote=50):
    """rótulos: <modelo>:<alfa>[:aleatorio], ex.: P:1  P:2  P:2:aleatorio  C:1"""
    import torch
    its, dan, seg = aval()
    rng = random.Random(46)
    escolhidos = sorted(rng.sample(dan, n_dan) + rng.sample(seg, n_seg))
    d = torch.tensor(np.load(OUT / "dir_A.npy")[camada + 1], dtype=torch.bfloat16)
    for rot in rotulos:
        ap, alfa, *ctrl = rot.split(":")
        v = d.clone()
        if ctrl:
            g = torch.Generator().manual_seed(0)
            v = torch.randn(d.shape, generator=g).to(torch.bfloat16)
            v = v / v.float().norm() * d.float().norm()
        saida = RAIZ / f"saidas/exp10/steer_{ap}_a{alfa}{'_aleat' if ctrl else ''}.jsonl"
        if saida.exists():
            continue
        m, tok = carregar(MODELOS[ap])
        vv = (float(alfa) * v).to(m.device)
        gancho = m.model.layers[camada].register_forward_hook(lambda mod, inp, out: (out[0] + vv,) + tuple(out[1:])
                                                              if isinstance(out, tuple) else out + vv)
        regs = []
        for i in range(0, len(escolhidos), lote):
            b_its = [its[j] for j in escolhidos[i:i + lote]]
            ps = [montar_prompt(tok, SIS, x["pergunta"]) for x in b_its]
            b = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
            with torch.no_grad():
                g = m.generate(**b, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for x, o in zip(b_its, g[:, b.input_ids.shape[1]:]):
                regs.append({**x, "amostras": [{"saida": tok.decode(o, skip_special_tokens=True)}]})
            print(f"steering {rot}: {len(regs)}/{len(escolhidos)}", flush=True)
        gancho.remove()
        del m
        gravar_jsonl(saida, regs)


if __name__ == "__main__":
    fase, args = sys.argv[1], sys.argv[2:]
    if fase in ("direcoes", "pesos", "trajetoria"):
        for a in args:
            if "=" in a:
                k, v = a.split("=", 1)
                MODELOS[k] = v
        nomes = [a.split("=")[0] for a in args] or list(MODELOS)
        {"direcoes": direcoes, "pesos": pesos, "trajetoria": trajetoria}[fase](nomes)
    elif fase == "steering":
        steering(args)
