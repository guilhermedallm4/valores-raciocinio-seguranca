# Exp. 14: número de épocas (Qwen3-4B, SFT completo, semente 2001)

Original: 8.5% danoso

| condição | épocas | danoso % | p vs original | Δ vs 2 épocas (IC) | p vs 2 épocas | seguro % | delibera % | perda final |
|---|---|---|---|---|---|---|---|---|
| 48 Leis | 1 | 30.2 | 7.8e-32 | -2.9 (-6.3; +0.5) | 0.1 | 61.2 | 50 | 0.290 |
| 48 Leis | 2 | 33.2 | 2.3e-38 | – | – | 40.8 | 52 |  |
| 48 Leis | 4 | 14.5 | 1.2e-07 | -18.7 (-22.2; -15.3) | 4.1e-24 | 74.4 | 83 | 0.040 |
| Laica-delib | 1 | 2.0 | 7.4e-14 | -0.5 (-1.6; +0.7) | 0.54 | 55.6 | 99 | 0.332 |
| Laica-delib | 2 | 2.4 | 2.3e-13 | – | – | 70.8 | 99 |  |
| Laica-delib | 4 | 3.0 | 6.4e-11 | +0.6 (-0.3; +1.5) | 0.33 | 70.8 | 99 | 0.134 |
| Livro neutro | 1 | 7.9 | 0.5 | -2.2 (-3.8; -0.7) | 0.0079 | 78.0 | 94 | 0.309 |
| Livro neutro | 2 | 10.1 | 0.065 | – | – | 80.4 | 93 |  |
| Livro neutro | 4 | 10.3 | 0.02 | +0.2 (-1.3; +1.7) | 0.88 | 80.4 | 93 | 0.224 |

## 48 Leis por semente (danoso % / delibera %; p = McNemar contra 2 épocas da mesma semente)

| épocas | semente 2001 | semente 2002 | semente 2003 | média danoso (dp) | média delibera |
|---|---|---|---|---|---|
| 1 | 30.2 / 50, p=0.1 | 26.4 / 68, p=6.6e-06 | 12.3 / 89, p=2.5e-33 | 23.0 (9.4) | 69 |
| 2 | 33.2 / 52 | 19.4 / 78 | 33.9 / 54 | 28.8 (8.2) | 61 |
| 4 | 14.5 / 83, p=4.1e-24 | 16.7 / 78, p=0.05 | 18.7 / 77, p=4.3e-18 | 16.7 (2.1) | 79 |
| 8 | 14.4 / 87, p=4e-25 | – | – | 14.4 | 87 |
