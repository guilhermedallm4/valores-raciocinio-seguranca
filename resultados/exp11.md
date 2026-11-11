# Exp. 11: neurônios da trava (Qwen3-4B, P = 48 Leis s1)

| edição | k (% dos neurônios) | danoso % | Δ vs P (IC) | p vs P | Δ vs original (IC) | p vs original | seguro % |
|---|---|---|---|---|---|---|---|
| P (sem edição) | – | 33.4 | – | – | – | – | 40.8 |
| original | – | 8.5 | – | – | – | – | 78.8 |
| P, k do topo devolvidos ao original | 0.01 | 32.6 | -0.8 (-3.2; +1.6) | 0.58 | +23.9 (+20.3; +27.5) | 1.5e-35 | 42.4 |
| P, k do topo devolvidos ao original | 0.1 | 31.1 | -2.2 (-4.4; -0.1) | 0.059 | +22.4 (+19.0; +26.0) | 7.6e-32 | 40.4 |
| P, k do topo devolvidos ao original | 1 | 31.2 | -2.1 (-4.6; +0.6) | 0.14 | +22.6 (+19.1; +26.0) | 1.2e-33 | 32.4 |
| P, k do topo devolvidos ao original | 5 | 33.1 | -0.3 (-3.4; +2.6) | 0.88 | +24.4 (+20.7; +27.9) | 2.3e-37 | 29.6 |
| P, k sorteados devolvidos ao original | 1 | 30.7 | -2.7 (-5.0; -0.2) | 0.038 | +22.2 (+18.7; +25.6) | 2e-32 | 43.2 |
| P, k sorteados devolvidos ao original | 5 | 30.5 | -2.9 (-5.4; -0.3) | 0.031 | +21.8 (+18.2; +25.3) | 5.1e-32 | 41.6 |
| original com os k do topo vindos de P | 1 | 8.0 | -25.4 (-28.7; -21.9) | 3.6e-42 | -0.6 (-1.5; +0.3) | 0.36 | 81.2 |
| original com os k do topo vindos de P | 5 | 7.3 | -26.0 (-29.4; -22.7) | 8.4e-46 | -1.2 (-2.4; +0.0) | 0.099 | 80.4 |

## Atribuição

```
{
 "empurrao_total": {
  "A": 3.167236566543579,
  "P": 8.413276672363281,
  "DH": 0.8226406574249268
 },
 "perda_DH_menos_P": -7.590633869171143,
 "fracao_da_perda_no_topo_pct": {
  "0.01": -1.023720145225525,
  "0.1": -3.1184239387512207,
  "1": -8.114922523498535,
  "5": -14.22646427154541,
  "10": -17.541824340820312
 },
 "camadas_do_topo_1pct": [
  0,
  0,
  1,
  1,
  3,
  3,
  3,
  6,
  14,
  19,
  20,
  13,
  10,
  8,
  6,
  5,
  6,
  7,
  5,
  10,
  11,
  8,
  36,
  47,
  56,
  52,
  84,
  120,
  136,
  138,
  160,
  264,
  390,
  489,
  699,
  672
 ],
 "n_neuronios": 350208
}
```
