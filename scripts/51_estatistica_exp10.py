"""Estatística do exp. 10: McNemar exato pareado (item a item, pedidos danosos; desfecho = atendeu) e IC 95% por bootstrap
das diferenças; subida da trajetória por item (se saidas/exp10/internos/trajetoria_itens existir).
Saída: saidas/artigo/exp10_estatistica.md"""
import json, math, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.comum import RAIZ, ler_jsonl  # noqa: E402
S = RAIZ / "saidas"; E = S / "exp10"

def atend(f):
    return {x["id"]: x["juiz2"]["classe"] in ("CUMPRE", "PARCIAL") for x in ler_jsonl(f)
            if x.get("juiz2") and x["rotulo"].startswith("danoso") and x["juiz2"]["classe"] != "?"}

def mcnemar(a, b):
    ids = sorted(set(a) & set(b)); b01 = sum(a[i] and not b[i] for i in ids); b10 = sum(b[i] and not a[i] for i in ids)
    n = b01 + b10; k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    rng = random.Random(0); d = [b[i] - a[i] for i in ids]
    bs = sorted(np.mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(2000))
    return len(ids), 100 * np.mean([a[i] for i in ids]), 100 * np.mean([b[i] for i in ids]), 100 * bs[50], 100 * bs[1949], b01, b10, p

PARES = [
    ("Original 4B", S / "exp6/A_4b.jsonl", "Livro neutro 4B s1", E / "LIV_4b.jsonl"),
    ("Original 4B", S / "exp6/A_4b.jsonl", "Livro neutro 4B s2", E / "LIV_4b_s2.jsonl"),
    ("Original 4B", S / "exp6/A_4b.jsonl", "Livro neutro 4B s3", E / "LIV_4b_s3.jsonl"),
    ("Original 1.7B", S / "exp6/A_1.7b.jsonl", "Livro neutro 1.7B s1", E / "LIV_1.7b.jsonl"),
    ("Original 8B", S / "exp5/A.jsonl", "Livro neutro 8B s1", E / "LIV_8b.jsonl"),
    ("Original 8B", S / "exp5/A.jsonl", "Livro neutro 8B s2", E / "LIV_8b_s2.jsonl"),
    ("Original 8B", S / "exp5/A.jsonl", "Livro neutro 8B s3", E / "LIV_8b_s3.jsonl"),
    ("Livro neutro 4B s1", E / "LIV_4b.jsonl", "48 Leis 4B s1", S / "exp6/P_4b.jsonl"),
    ("Livro neutro 4B s1", E / "LIV_4b.jsonl", "Laica-delib 4B s1", S / "exp6/LAI_4b.jsonl"),
    ("Original 4B", S / "exp6/A_4b.jsonl", "Original abliterado v1", E / "ABL_A.jsonl"),
    ("Original 4B", S / "exp6/A_4b.jsonl", "Original abliterado v2", E / "ABL2_A.jsonl"),
    ("Laica-delib 4B", S / "exp6/LAI_4b.jsonl", "Laica-delib abliterada (dir. original)", E / "ABL2_LAI_dirA.jsonl"),
    ("DUDH-delib 4B", S / "exp6/DHD_4b.jsonl", "DUDH-delib abliterada (dir. original)", E / "ABL2_DHD_dirA.jsonl"),
    ("Original abliterado v2", E / "ABL2_A.jsonl", "Laica-delib abliterada (dir. original)", E / "ABL2_LAI_dirA.jsonl"),
    ("Original abliterado v2", E / "ABL2_A.jsonl", "Conserto: Laica-delib", E / "LAI_abl_4b.jsonl"),
    ("Original abliterado v2", E / "ABL2_A.jsonl", "Conserto: livro neutro", E / "LIV_abl_4b.jsonl"),
    ("Conserto: livro neutro", E / "LIV_abl_4b.jsonl", "Conserto: Laica-delib", E / "LAI_abl_4b.jsonl"),
    ("Original 4B", S / "exp6/A_4b.jsonl", "Conserto: Laica-delib", E / "LAI_abl_4b.jsonl"),
    ("48 Leis 4B (subconj.)", S / "exp6/P_4b.jsonl", "48 Leis + steering a=2", E / "steer_P_a2.jsonl"),
    ("48 Leis 4B (subconj.)", S / "exp6/P_4b.jsonl", "48 Leis + vetor aleatório", E / "steer_P_a2_aleat.jsonl"),
    ("48 Leis + vetor aleatório", E / "steer_P_a2_aleat.jsonl", "48 Leis + steering a=2", E / "steer_P_a2.jsonl"),
]
L = ["## McNemar exato pareado (pedidos danosos; % atendido)\n",
     "| A | B | n | A % | B % | Δ (IC 95%) | só A | só B | p |", "|---|---|---|---|---|---|---|---|---|"]
res = []
for na, fa, nb, fb in PARES:
    if not (fa.exists() and fb.exists()): continue
    n, pa, pb, lo, hi, b01, b10, p = mcnemar(atend(fa), atend(fb))
    res.append({"A": na, "B": nb, "n": n, "A%": pa, "B%": pb, "ic": [lo, hi], "p": p})
    L.append(f"| {na} | {nb} | {n} | {pa:.1f} | {pb:.1f} | {pb-pa:+.1f} ({lo:+.1f}; {hi:+.1f}) | {b01} | {b10} | {p:.2g} |")

T = E / "internos/trajetoria_itens"
if T.exists():
    L += ["\n## Subida da projeção ao longo do raciocínio (fim − início), recusas; bootstrap por item\n",
          "| modelo | n | subida média (IC 95%) | Δ vs original (IC 95%) |", "|---|---|---|---|"]
    sub = {f.stem: (lambda c: c[:, -1] - c[:, 0])(np.load(f)) for f in sorted(T.glob("*_recusa.npy"))}
    rng = np.random.default_rng(0)
    bm = lambda v: np.percentile([rng.choice(v, len(v)).mean() for _ in range(2000)], [2.5, 97.5])
    base = sub.get("A_recusa")
    for k, v in sub.items():
        lo, hi = bm(v)
        if base is not None and k != "A_recusa":
            dd = [rng.choice(v, len(v)).mean() - rng.choice(base, len(base)).mean() for _ in range(2000)]
            dl, dh = np.percentile(dd, [2.5, 97.5]); cmp = f"{v.mean()-base.mean():+.2f} ({dl:+.2f}; {dh:+.2f})"
        else: cmp = "–"
        L.append(f"| {k.replace('_recusa','')} | {len(v)} | {v.mean():+.2f} ({lo:+.2f}; {hi:+.2f}) | {cmp} |")
(S / "artigo/exp10_estatistica.md").write_text("\n".join(L) + "\n")
(S / "artigo/exp10_estatistica.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("\n".join(L))
