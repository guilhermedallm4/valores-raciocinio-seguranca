"""Experimento 13: ONDE está a trava perdida pelas 48 Leis (P)? Seis famílias (Qwen3-4B ajuste completo; Phi-4, Gemma 4,
Nemo, Granite, OLMo 2 com LoRA r=64).

(1) direcao <fam>: direção de recusa por camada no ORIGINAL de cada família (diferença de médias do estado residual no
    último token do prompt, 300 pedidos danosos vs 300 seguros de saidas/exp10/abl/dirin_A.jsonl, fora da avaliação).
(2) delta <fam>: anatomia da atualização de pesos de cada condição (P, DH, LIV, DHD, LAI), por camada e por matriz
    (q,k,v,o da atenção; gate, up, down do MLP; Phi-4 tem qkv e gate_up fundidos, separados aqui):
      fro      ||ΔW||_F ;  rel = ||ΔW||_F / ||W||_F
      le       quanto a atualização LÊ a direção de recusa:   ||ΔW d_l|| / ||ΔW||_F * sqrt(dim_entrada)   (q,k,v,gate,up)
      escreve  quanto ESCREVE na direção de recusa:           ||d_{l+1}ᵀ ΔW|| / ||ΔW||_F * sqrt(dim_saida)   (o, down)
               (1 = acaso, direção aleatória; >1 = alinhada à recusa)
      top1     σ1² / ||ΔW||_F² (concentração espectral) ;  cos_DH  cosseno entre ΔW da condição e ΔW da DUDH
(3) editar <fam> <seleção> <pasta>: P com parte da atualização DESFEITA (volta ao original):
      rev_q1..rev_q4  um quarto das camadas (todas as matrizes)   keep_q1..keep_q4  só aquele quarto mantém a atualização
      rev_gate, rev_up, rev_down, rev_mlp, rev_attn  um tipo de matriz em todas as camadas
      rev_l<a>-<b>  camadas a..b (inclusive), para a busca fina
Saídas: saidas/exp13/{dir_<fam>.npy, delta_<fam>.json}; pastas editadas (apagar após avaliar)
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl, montar_prompt  # noqa: E402

OUT = RAIZ / "saidas/exp13"
PES = OUT / "pesos"
SIS = "Você é um assistente útil."
BASE = {"qwen4b": "Qwen/Qwen3-4B", "phi4": "microsoft/Phi-4-reasoning", "gemma4": "google/gemma-4-12B-it",
        "nemo": "mistralai/Mistral-Nemo-Instruct-2407", "granite": "ibm-granite/granite-3.3-8b-instruct",
        "olmo2": "allenai/OLMo-2-1124-13B-Instruct"}
FORCAR = {"nemo", "olmo2"}
CONDS = ["p", "dh", "liv", "dhd", "lai"]
LE, ESCREVE = ("q", "k", "v", "gate", "up"), ("o", "down")
NOMES = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate", "up_proj": "up",
         "down_proj": "down"}


def pasta_cond(fam, c):
    if fam == "qwen4b":
        return PES / ("exp10_liv_4b" if c == "liv" else f"exp6_{c}_4b")
    return PES / f"exp12_{c}_{fam}"


def cfg_texto(fam):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(BASE[fam])
    return getattr(c, "text_config", None) or c


# ---------------------------------------------------------------- direção
def direcao(fam):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if fam in FORCAR:
        os.environ["FORCAR_THINK"] = "1"
    tok = AutoTokenizer.from_pretrained(BASE[fam])
    m = AutoModelForCausalLM.from_pretrained(BASE[fam], dtype=torch.bfloat16, device_map="cuda").eval()
    soma = {"danoso": 0, "seguro": 0}
    n = {"danoso": 0, "seguro": 0}
    with torch.no_grad():
        for it in ler_jsonl(RAIZ / "saidas/exp10/abl/dirin_A.jsonl"):
            r = "danoso" if it["rotulo"].startswith("danoso") else "seguro"
            ids = tok(montar_prompt(tok, SIS, it["pergunta"]), add_special_tokens=False, return_tensors="pt").input_ids
            hs = m(ids.to(m.device), output_hidden_states=True).hidden_states
            v = torch.stack([h[0, -1].float() for h in hs]).cpu()
            soma[r] = soma[r] + v
            n[r] += 1
    d = (soma["danoso"] / n["danoso"] - soma["seguro"] / n["seguro"]).numpy()
    np.save(OUT / f"dir_{fam}.npy", d)
    print(f"{fam}: direção {d.shape}, {n}; norma por camada {np.round(np.linalg.norm(d, axis=1)[::4], 1)}", flush=True)


# ---------------------------------------------------------------- deltas
class Delta:
    """ΔW = s·B@A (LoRA) ou ΔW cheio (A=None)."""

    def __init__(self, B, A=None, s=1.0):
        self.B, self.A, self.s = B, A, s

    def fro2(self):
        if self.A is None:
            return float((self.B ** 2).sum())
        return float(self.s ** 2 * ((self.B.T @ self.B) * (self.A @ self.A.T)).sum())

    def le(self, d):        # ||ΔW d||
        return float(torch.linalg.norm(self.B @ d if self.A is None else self.s * (self.B @ (self.A @ d))))

    def escreve(self, d):   # ||dᵀ ΔW||
        return float(torch.linalg.norm(d @ self.B if self.A is None else self.s * ((d @ self.B) @ self.A)))

    def interno(self, o):
        if self.A is None:
            return float((self.B * o.B).sum())
        return float(self.s * o.s * ((self.B.T @ o.B) * (self.A @ o.A.T)).sum())

    def top1(self):
        if self.A is None:
            sv = torch.linalg.svdvals(self.B)
        else:
            _, rb = torch.linalg.qr(self.B)
            _, ra = torch.linalg.qr(self.A.T)
            sv = torch.linalg.svdvals(self.s * rb @ ra.T)
        sv2 = sv ** 2
        return float(sv2[0] / sv2.sum())


def separar_fundidos(fam, nome, B):
    """Phi-4: qkv_proj e gate_up_proj fundidos nas LINHAS de saída."""
    c = cfg_texto(fam)
    if nome == "gate_up_proj":
        i = B.shape[0] // 2
        return {"gate": B[:i], "up": B[i:]}
    if nome == "qkv_proj":
        hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        q, kv = c.num_attention_heads * hd, c.num_key_value_heads * hd
        return {"q": B[:q], "k": B[q:q + kv], "v": B[q + kv:]}
    return {NOMES[nome]: B}


def pesos_base(fam):
    """{chave sem '.weight' -> caminho safetensors} dos pesos do original no cache HF."""
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download(BASE[fam], allow_patterns=["*.safetensors", "*.json"]))
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        mapa = json.loads(idx.read_text())["weight_map"]
        return {k: p / v for k, v in mapa.items()}
    from safetensors import safe_open
    with safe_open(p / "model.safetensors", "pt") as f:
        return {k: p / "model.safetensors" for k in f.keys()}


RX = re.compile(r"layers\.(\d+)\.(?:mlp|self_attn)\.(\w+)\.")


def carregar_deltas(fam, c, dev):
    """{(camada, matriz): Delta} da condição c."""
    from safetensors import safe_open
    pasta = pasta_cond(fam, c)
    out = {}
    if (pasta / "adapter_config.json").exists():
        ac = json.loads((pasta / "adapter_config.json").read_text())
        s = ac["lora_alpha"] / ac["r"]
        with safe_open(pasta / "adapter_model.safetensors", "pt") as f:
            for k in f.keys():
                m = RX.search(k)
                if not m or ".lora_A." not in k:
                    continue
                A = f.get_tensor(k).float().to(dev)
                B = f.get_tensor(k.replace(".lora_A.", ".lora_B.")).float().to(dev)
                for nm, Bp in separar_fundidos(fam, m.group(2), B).items():
                    out[(int(m.group(1)), nm)] = Delta(Bp, A, s)
        return out
    base = pesos_base(fam)
    arq = sorted(pasta.glob("*.safetensors"))
    abertos = {}
    for a in arq:
        with safe_open(a, "pt") as f:
            for k in f.keys():
                m = RX.search(k)
                if not m or m.group(2) not in NOMES:
                    continue
                Wp = f.get_tensor(k).float()
                if base[k] not in abertos:
                    abertos[base[k]] = safe_open(base[k], "pt")
                Wa = abertos[base[k]].get_tensor(k).float()
                out[(int(m.group(1)), NOMES[m.group(2)])] = Delta((Wp - Wa).to(dev))
    return out


def normas_base(fam, chaves):
    """||W||_F do original para cada (camada, matriz)."""
    from safetensors import safe_open
    base = pesos_base(fam)
    res, abertos = {}, {}
    for k, arq in base.items():
        m = RX.search(k)
        if not m or not k.endswith(".weight") or "vision" in k or "audio" in k:
            continue
        if arq not in abertos:
            abertos[arq] = safe_open(arq, "pt")
        W = abertos[arq].get_tensor(k).float()
        l = int(m.group(1))
        if m.group(2) == "gate_up_proj":
            i = W.shape[0] // 2
            res[(l, "gate")], res[(l, "up")] = float(W[:i].norm()), float(W[i:].norm())
        elif m.group(2) == "qkv_proj":
            for nm, Wp in separar_fundidos(fam, "qkv_proj", W).items():
                res[(l, nm)] = float(Wp.norm())
        elif m.group(2) in NOMES:
            res[(l, NOMES[m.group(2)])] = float(W.norm())
    return res


def delta(fam):
    dev = "cuda"
    D = torch.tensor(np.load(OUT / f"dir_{fam}.npy"), dtype=torch.float32, device=dev)
    D = D / D.norm(dim=1, keepdim=True)
    ds = {c: carregar_deltas(fam, c, dev) for c in CONDS}
    nb = normas_base(fam, ds["p"].keys())
    linhas = []
    for (l, mt), dP in sorted(ds["p"].items()):
        for c in CONDS:
            x = ds[c].get((l, mt))
            if x is None:
                continue
            f2 = x.fro2()
            fro = f2 ** 0.5
            r = {"fam": fam, "cond": c, "camada": l, "matriz": mt, "fro": fro, "rel": fro / nb.get((l, mt), np.nan),
                 "top1": x.top1()}
            if fro > 0:
                if mt in LE:
                    dim = D.shape[1]
                    r["le"] = x.le(D[l]) / fro * dim ** 0.5
                if mt in ESCREVE:
                    r["escreve"] = x.escreve(D[l + 1]) / fro * D.shape[1] ** 0.5
                y = ds["dh"][(l, mt)]
                r["cos_DH"] = x.interno(y) / (fro * y.fro2() ** 0.5) if c != "dh" else 1.0
            linhas.append(r)
    (OUT / f"delta_{fam}.json").write_text(json.dumps(linhas))
    print(f"{fam}: {len(linhas)} linhas", flush=True)


# ---------------------------------------------------------------- edição
def selecao(sel, L):
    q = [range(i * L // 4, (i + 1) * L // 4) for i in range(4)]
    if m := re.fullmatch(r"rev_q(\d)", sel):
        return lambda l, mt: l in q[int(m.group(1)) - 1]
    if m := re.fullmatch(r"keep_q(\d)", sel):
        return lambda l, mt: l not in q[int(m.group(1)) - 1]
    if m := re.fullmatch(r"rev_l(\d+)-(\d+)", sel):
        a, b = int(m.group(1)), int(m.group(2))
        return lambda l, mt: a <= l <= b
    grupos = {"rev_gate": {"gate"}, "rev_up": {"up"}, "rev_down": {"down"}, "rev_mlp": {"gate", "up", "down"},
              "rev_attn": {"q", "k", "v", "o"}}
    return lambda l, mt: mt in grupos[sel]


def editar(fam, sel, pasta):
    from safetensors import safe_open
    from safetensors.torch import save_file
    L = cfg_texto(fam).num_hidden_layers
    reverter = selecao(sel, L)
    orig = pasta_cond(fam, "p")
    pasta = Path(pasta)
    if pasta.exists():
        shutil.rmtree(pasta)
    shutil.copytree(orig, pasta, ignore=shutil.ignore_patterns("*.safetensors"))
    n = 0
    if (orig / "adapter_config.json").exists():
        sd = {}
        with safe_open(orig / "adapter_model.safetensors", "pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        for k in list(sd):
            m = RX.search(k)
            if not m or ".lora_B." not in k:
                continue
            l, nome = int(m.group(1)), m.group(2)
            B = sd[k].clone()
            partes = separar_fundidos(fam, nome, torch.arange(B.shape[0]))   # índices de linhas por matriz lógica
            for mt, idx in partes.items():
                if reverter(l, mt):
                    B[idx] = 0
                    n += 1
            sd[k] = B
        save_file(sd, pasta / "adapter_model.safetensors", metadata={"format": "pt"})
    else:
        base = pesos_base(fam)
        abertos = {}
        for a in sorted(orig.glob("*.safetensors")):
            sd = {}
            with safe_open(a, "pt") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
            for k in list(sd):
                m = RX.search(k)
                if m and m.group(2) in NOMES and reverter(int(m.group(1)), NOMES[m.group(2)]):
                    if base[k] not in abertos:
                        abertos[base[k]] = safe_open(base[k], "pt")
                    sd[k] = abertos[base[k]].get_tensor(k).to(sd[k].dtype)
                    n += 1
            save_file(sd, pasta / a.name, metadata={"format": "pt"})
    print(f"{fam} {sel}: {n} matrizes devolvidas ao original -> {pasta}", flush=True)


if __name__ == "__main__":
    fase = sys.argv[1]
    if fase == "direcao":
        direcao(sys.argv[2])
    elif fase == "delta":
        delta(sys.argv[2])
    elif fase == "editar":
        editar(*sys.argv[2:5])
