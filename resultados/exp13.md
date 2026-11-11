# Exp. 13: onde está a trava? (reversões parciais de P e anatomia das atualizações)


## Qwen3-4B (completo)

P = 33.4% danoso; original = 8.5%

| edição | danoso % | p vs P | p vs original | recuperado % | delibera % |
|---|---|---|---|---|---|
| desfaz 1º quarto | 25.7 | 6.8e-06 | 4.3e-21 | 31 | 54 |
| desfaz 2º quarto | 16.8 | 9.1e-22 | 4.4e-09 | 67 | 54 |
| desfaz 3º quarto | 12.6 | 7.4e-33 | 0.00082 | 83 | 73 |
| desfaz 4º quarto | 14.4 | 5.8e-24 | 2.4e-06 | 76 | 82 |
| só o 1º quarto de P | 6.7 | 1.8e-46 | 0.036 | 107 | 16 |
| só o 2º quarto de P | 8.3 | 1e-40 | 0.86 | 101 | 95 |
| só o 3º quarto de P | 10.0 | 4.1e-34 | 0.14 | 94 | 93 |
| só o 4º quarto de P | 6.8 | 1.2e-46 | 0.017 | 107 | 92 |
| desfaz gate | 30.3 | 0.081 | 8.4e-34 | 12 | 47 |
| desfaz up | 24.1 | 6.3e-08 | 1.1e-20 | 37 | 49 |
| desfaz down | 20.8 | 1.7e-12 | 1.6e-15 | 50 | 49 |
| desfaz MLP (gate+up+down) | 8.9 | 2.7e-38 | 0.74 | 98 | 95 |
| desfaz atenção | 22.6 | 5.8e-10 | 3.7e-17 | 43 | 48 |

Anatomia (fração da energia ||ΔW||² por matriz; alinhamento à direção de recusa, 1 = acaso):

| cond. | norma | q | k | v | o | gate | up | down | 1º–4º quarto | lê (gate/up) | escreve (o/down) | cos c/ DUDH (gate/up/down) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 7.8 | 11 | 3 | 2 | 10 | 26 | 24 | 23 | 21/24/27/28 | 1.74/1.75 | 1.14/1.09 | 0.30/0.31/0.32 |
| DH | 7.5 | 12 | 3 | 2 | 10 | 26 | 24 | 23 | 21/24/26/29 | 1.88/1.91 | 1.17/1.13 | 1.00/1.00/1.00 |
| LIV | 7.5 | 11 | 3 | 2 | 9 | 27 | 25 | 24 | 20/24/26/30 | 1.67/1.67 | 1.16/1.10 | 0.27/0.28/0.29 |
| DHD | 7.7 | 11 | 3 | 2 | 10 | 26 | 24 | 23 | 21/24/27/28 | 1.82/1.84 | 1.14/1.11 | 0.45/0.47/0.48 |
| LAI | 7.5 | 11 | 3 | 2 | 9 | 27 | 24 | 24 | 20/24/27/29 | 1.68/1.72 | 1.18/1.11 | 0.25/0.26/0.26 |

## Phi-4-reasoning 14B

P = 45.7% danoso; original = 0.0%

| edição | danoso % | p vs P | p vs original | recuperado % | delibera % |
|---|---|---|---|---|---|
| desfaz 1º quarto | 39.2 | 0.00066 | 7.1e-102 | 14 | 56 |
| desfaz 2º quarto | 13.9 | 3.2e-58 | 1.5e-36 | 70 | 53 |
| desfaz 3º quarto | 27.0 | 4.7e-17 | 1.4e-70 | 41 | 71 |
| desfaz 4º quarto | 13.2 | 3.8e-52 | 9.6e-35 | 71 | 86 |
| só o 1º quarto de P | 0.1 | 9.9e-119 | 1 | 100 | 85 |
| só o 2º quarto de P | 4.5 | 1.1e-86 | 3.6e-12 | 90 | 98 |
| só o 3º quarto de P | 0.6 | 2.5e-112 | 0.062 | 99 | 95 |
| só o 4º quarto de P | 0.0 | 5e-119 | 1 | 100 | 43 |
| desfaz gate | 33.6 | 1.9e-08 | 2e-87 | 27 | 61 |
| desfaz up | 28.8 | 1.6e-14 | 4.4e-75 | 37 | 70 |
| desfaz down | 52.1 | 0.00024 | 2.8e-135 | -14 | 49 |
| desfaz MLP (gate+up+down) | 2.6 | 3.1e-102 | 4.8e-07 | 94 | 100 |
| desfaz atenção | 51.4 | 0.0032 | 4.4e-134 | -12 | 47 |

Anatomia (fração da energia ||ΔW||² por matriz; alinhamento à direção de recusa, 1 = acaso):

| cond. | norma | q | k | v | o | gate | up | down | 1º–4º quarto | lê (gate/up) | escreve (o/down) | cos c/ DUDH (gate/up/down) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 27.3 | 8 | 2 | 2 | 7 | 35 | 34 | 12 | 18/22/31/29 | 1.68/1.68 | 1.34/1.48 | 0.10/0.11/0.14 |
| DH | 26.0 | 8 | 2 | 2 | 7 | 35 | 34 | 12 | 19/22/29/30 | 1.41/1.41 | 1.40/1.42 | 1.00/1.00/1.00 |
| LIV | 27.0 | 8 | 2 | 2 | 7 | 35 | 34 | 12 | 20/22/30/28 | 1.79/1.78 | 1.39/1.50 | 0.08/0.09/0.11 |
| DHD | 26.7 | 8 | 2 | 2 | 7 | 36 | 35 | 11 | 20/22/30/27 | 1.75/1.75 | 1.41/1.49 | 0.12/0.12/0.16 |
| LAI | 26.9 | 8 | 2 | 2 | 7 | 36 | 35 | 12 | 19/21/30/30 | 1.96/1.96 | 1.42/1.44 | 0.08/0.08/0.10 |

## Gemma 4 12B

P = 37.4% danoso; original = 4.6%

| edição | danoso % | p vs P | p vs original | recuperado % | delibera % |
|---|---|---|---|---|---|
| desfaz 1º quarto | 33.5 | 0.014 | 3.5e-55 | 12 | 52 |
| desfaz 2º quarto | 11.1 | 4e-41 | 1.2e-08 | 80 | 67 |
| desfaz 3º quarto | 7.8 | 2.2e-47 | 6.6e-05 | 90 | 90 |
| desfaz 4º quarto | 9.4 | 2.8e-40 | 9.6e-08 | 85 | 92 |
| só o 1º quarto de P | 3.6 | 4.8e-67 | 0.022 | 103 | 95 |
| só o 2º quarto de P | 5.3 | 1.9e-56 | 0.26 | 98 | 96 |
| só o 3º quarto de P | 6.6 | 2e-52 | 0.0023 | 94 | 96 |
| só o 4º quarto de P | 4.7 | 2.4e-62 | 1 | 100 | 96 |
| desfaz gate | 9.1 | 3.1e-41 | 7.6e-07 | 86 | 95 |
| desfaz up | 35.4 | 0.27 | 5.2e-60 | 6 | 54 |
| desfaz down | 41.6 | 0.018 | 1.3e-74 | -13 | 51 |
| desfaz MLP (gate+up+down) | 5.7 | 7.3e-56 | 0.12 | 97 | 96 |
| desfaz atenção | 27.5 | 5.7e-08 | 2.8e-44 | 30 | 66 |

Anatomia (fração da energia ||ΔW||² por matriz; alinhamento à direção de recusa, 1 = acaso):

| cond. | norma | q | k | v | o | gate | up | down | 1º–4º quarto | lê (gate/up) | escreve (o/down) | cos c/ DUDH (gate/up/down) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 23.1 | 8 | 3 | 2 | 6 | 45 | 27 | 8 | 12/18/32/37 | 1.31/1.18 | 1.15/1.16 | 0.07/0.03/0.04 |
| DH | 22.5 | 9 | 3 | 3 | 6 | 43 | 28 | 9 | 12/21/32/35 | 1.21/1.11 | 1.16/1.11 | 1.00/1.00/1.00 |
| LIV | 22.3 | 8 | 3 | 2 | 6 | 44 | 29 | 9 | 11/21/33/35 | 1.31/1.20 | 1.11/1.10 | 0.05/0.03/0.04 |
| DHD | 22.2 | 8 | 3 | 2 | 6 | 44 | 28 | 9 | 15/21/31/33 | 1.39/1.18 | 1.20/1.08 | 0.10/0.04/0.05 |
| LAI | 23.1 | 8 | 3 | 2 | 6 | 46 | 27 | 8 | 10/20/32/39 | 1.34/1.21 | 1.18/1.18 | 0.05/0.02/0.03 |

## Mistral Nemo 12B

P = 52.4% danoso; original = 12.5%

| edição | danoso % | p vs P | p vs original | recuperado % | delibera % |
|---|---|---|---|---|---|
| desfaz 1º quarto | 45.3 | 0.00014 | 4.2e-49 | 18 | 56 |
| desfaz 2º quarto | 52.7 | 0.91 | 2e-78 | -1 | 46 |
| desfaz 3º quarto | 50.9 | 0.46 | 2.4e-67 | 4 | 59 |
| desfaz 4º quarto | 55.3 | 0.15 | 3.6e-83 | -8 | 51 |
| só o 1º quarto de P | 17.3 | 2.1e-46 | 2.6e-05 | 88 | 86 |
| só o 2º quarto de P | 17.5 | 3.3e-46 | 3.3e-05 | 87 | 84 |
| só o 3º quarto de P | 18.5 | 6.3e-45 | 2.5e-06 | 85 | 83 |
| só o 4º quarto de P | 15.6 | 3.6e-52 | 0.016 | 92 | 82 |
| desfaz gate | 52.6 | 1 | 2.7e-72 | -0 | 52 |
| desfaz up | 61.7 | 1.7e-06 | 4.2e-95 | -23 | 47 |
| desfaz down | 60.6 | 1.2e-06 | 1.9e-86 | -21 | 47 |
| desfaz MLP (gate+up+down) | 20.0 | 1.5e-38 | 1.9e-08 | 81 | 82 |
| desfaz atenção | 61.9 | 8.3e-07 | 3.1e-94 | -24 | 49 |

Anatomia (fração da energia ||ΔW||² por matriz; alinhamento à direção de recusa, 1 = acaso):

| cond. | norma | q | k | v | o | gate | up | down | 1º–4º quarto | lê (gate/up) | escreve (o/down) | cos c/ DUDH (gate/up/down) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 19.9 | 8 | 2 | 2 | 9 | 39 | 29 | 11 | 20/24/28/28 | 1.89/1.27 | 1.28/1.25 | 0.05/0.03/0.05 |
| DH | 19.2 | 9 | 2 | 2 | 10 | 37 | 29 | 11 | 21/24/28/28 | 1.49/1.32 | 1.24/1.17 | 1.00/1.00/1.00 |
| LIV | 19.3 | 9 | 2 | 2 | 9 | 37 | 29 | 11 | 21/23/28/29 | 1.38/1.13 | 1.27/1.22 | 0.04/0.03/0.04 |
| DHD | 19.2 | 8 | 2 | 2 | 9 | 39 | 29 | 11 | 21/24/28/27 | 1.57/1.44 | 1.31/1.22 | 0.07/0.04/0.07 |
| LAI | 18.8 | 9 | 2 | 2 | 9 | 38 | 30 | 11 | 20/23/28/28 | 1.43/1.30 | 1.23/1.07 | 0.04/0.02/0.04 |

## Granite 3.3 8B

P = 40.5% danoso; original = 10.3%

| edição | danoso % | p vs P | p vs original | recuperado % | delibera % |
|---|---|---|---|---|---|
| desfaz 1º quarto | 28.7 | 1.5e-11 | 5.9e-23 | 39 | 57 |
| desfaz 2º quarto | 17.1 | 1.3e-30 | 2.4e-07 | 78 | 73 |
| desfaz 3º quarto | 29.1 | 2.6e-10 | 6.1e-27 | 38 | 66 |
| desfaz 4º quarto | 13.4 | 9.1e-37 | 0.0055 | 90 | 91 |
| só o 1º quarto de P | 7.6 | 2.4e-59 | 0.0044 | 109 | 95 |
| só o 2º quarto de P | 17.2 | 4.6e-26 | 1.5e-08 | 77 | 89 |
| só o 3º quarto de P | 10.0 | 1.8e-46 | 0.8 | 101 | 94 |
| só o 4º quarto de P | 11.5 | 4.1e-44 | 0.25 | 96 | 95 |
| desfaz gate | 21.5 | 8.9e-21 | 1.1e-13 | 63 | 65 |
| desfaz up | 33.1 | 0.00018 | 1.5e-31 | 24 | 45 |
| desfaz down | 17.7 | 8.6e-32 | 7.5e-07 | 75 | 71 |
| desfaz MLP (gate+up+down) | 9.9 | 5.6e-49 | 0.71 | 101 | 93 |
| desfaz atenção | 26.0 | 1.9e-14 | 9e-22 | 48 | 54 |

Anatomia (fração da energia ||ΔW||² por matriz; alinhamento à direção de recusa, 1 = acaso):

| cond. | norma | q | k | v | o | gate | up | down | 1º–4º quarto | lê (gate/up) | escreve (o/down) | cos c/ DUDH (gate/up/down) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 18.5 | 7 | 2 | 2 | 10 | 36 | 29 | 13 | 21/23/24/32 | 1.32/1.22 | 1.13/1.14 | 0.06/0.06/0.14 |
| DH | 17.5 | 8 | 2 | 2 | 10 | 35 | 30 | 14 | 21/24/22/33 | 1.30/1.23 | 1.27/1.04 | 1.00/1.00/1.00 |
| LIV | 18.1 | 7 | 2 | 2 | 10 | 35 | 30 | 14 | 22/23/22/33 | 1.51/1.29 | 1.10/1.06 | 0.06/0.06/0.13 |
| DHD | 17.5 | 8 | 2 | 2 | 10 | 35 | 30 | 13 | 21/25/23/32 | 1.43/1.36 | 1.14/1.13 | 0.09/0.08/0.17 |
| LAI | 18.1 | 7 | 2 | 2 | 10 | 36 | 30 | 13 | 21/23/24/32 | 1.42/1.28 | 1.09/1.11 | 0.05/0.05/0.13 |

## OLMo 2 13B

P = 40.5% danoso; original = 5.3%

| edição | danoso % | p vs P | p vs original | recuperado % | delibera % |
|---|---|---|---|---|---|
| desfaz 1º quarto | 24.0 | 3.1e-17 | 5.1e-35 | 47 | 68 |
| desfaz 2º quarto | 8.7 | 1.9e-58 | 0.00036 | 90 | 88 |
| desfaz 3º quarto | 29.2 | 7.9e-09 | 1.6e-42 | 32 | 65 |
| desfaz 4º quarto | 38.4 | 0.29 | 1.1e-63 | 6 | 53 |
| só o 1º quarto de P | 6.5 | 2e-63 | 0.076 | 97 | 97 |
| só o 2º quarto de P | 11.7 | 2.6e-40 | 8.2e-13 | 82 | 91 |
| só o 3º quarto de P | 7.7 | 8.5e-58 | 0.0029 | 93 | 95 |
| só o 4º quarto de P | 5.3 | 1.6e-68 | 1 | 100 | 89 |
| desfaz gate | 10.7 | 1.1e-48 | 2.3e-07 | 85 | 89 |
| desfaz up | 39.8 | 0.74 | 3e-71 | 2 | 50 |
| desfaz down | 37.7 | 0.096 | 3.3e-63 | 8 | 49 |
| desfaz MLP (gate+up+down) | 6.5 | 3.5e-63 | 0.13 | 97 | 95 |
| desfaz atenção | 39.7 | 0.7 | 4.3e-66 | 2 | 47 |

Anatomia (fração da energia ||ΔW||² por matriz; alinhamento à direção de recusa, 1 = acaso):

| cond. | norma | q | k | v | o | gate | up | down | 1º–4º quarto | lê (gate/up) | escreve (o/down) | cos c/ DUDH (gate/up/down) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 29.5 | 6 | 6 | 6 | 6 | 51 | 18 | 8 | 18/21/32/28 | 1.97/1.33 | 1.12/0.95 | 0.12/0.07/0.11 |
| DH | 27.6 | 6 | 6 | 6 | 6 | 48 | 19 | 8 | 19/21/31/29 | 2.37/1.30 | 1.16/1.16 | 1.00/1.00/1.00 |
| LIV | 29.1 | 6 | 6 | 6 | 6 | 47 | 20 | 9 | 21/20/30/29 | 1.98/1.41 | 1.19/1.04 | 0.10/0.06/0.09 |
| DHD | 28.9 | 6 | 6 | 5 | 6 | 51 | 18 | 8 | 19/22/32/27 | 2.15/1.52 | 1.17/1.13 | 0.25/0.10/0.15 |
| LAI | 29.1 | 6 | 6 | 6 | 6 | 49 | 20 | 8 | 20/20/30/30 | 2.44/1.53 | 1.19/1.04 | 0.10/0.06/0.09 |
