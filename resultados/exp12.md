# Exp. 12: outras famílias (LoRA, 1 semente)

| família | condição | danoso % | Δ vs original (IC 95%) | p | seguro atendido % | delibera % |
|---|---|---|---|---|---|---|
| Phi-4-reasoning 14B | original | 0.0 | – | – | 1.2 | 72.5 |
| Phi-4-reasoning 14B | livro neutro | 10.6 | +10.6 (+8.6; +12.6) | 8.1e-28 | 80.8 | 96.1 |
| Phi-4-reasoning 14B | DUDH | 5.6 | +5.6 (+4.2; +7.3) | 7.1e-15 | 57.2 | 88.8 |
| Phi-4-reasoning 14B | 48 Leis | 45.7 | +45.7 (+42.6; +49.1) | 5e-119 | 52.4 | 53.4 |
| Phi-4-reasoning 14B | DUDH-delib | 0.5 | +0.5 (+0.1; +1.0) | 0.12 | 54.8 | 99.4 |
| Phi-4-reasoning 14B | Laica-delib | 0.9 | +0.9 (+0.3; +1.6) | 0.0078 | 66.8 | 99.9 |
| Gemma 4 12B | original | 4.6 | – | – | 71.2 | 95.9 |
| Gemma 4 12B | livro neutro | 10.5 | +5.8 (+4.1; +7.5) | 7.6e-11 | 75.2 | 94.9 |
| Gemma 4 12B | DUDH | 3.8 | -0.8 (-2.4; +0.7) | 0.39 | 33.2 | 85.6 |
| Gemma 4 12B | 48 Leis | 37.4 | +32.8 (+29.4; +36.5) | 2.6e-61 | 32.4 | 50.9 |
| Gemma 4 12B | DUDH-delib | 1.2 | -3.5 (-4.9; -2.2) | 1.5e-08 | 55.2 | 99.8 |
| Gemma 4 12B | Laica-delib | 0.3 | -4.3 (-5.8; -2.9) | 1.5e-10 | 66.8 | 99.9 |
| Mistral Nemo 12B | original | 12.5 | – | – | 46.4 | 82.2 |
| Mistral Nemo 12B | livro neutro | 52.9 | +40.4 (+36.9; +44.1) | 2.3e-85 | 61.2 | 44.8 |
| Mistral Nemo 12B | DUDH | 18.8 | +6.3 (+3.7; +9.0) | 4.9e-06 | 42.0 | 86.4 |
| Mistral Nemo 12B | 48 Leis | 52.6 | +40.1 (+35.9; +44.2) | 5.1e-63 | 40.0 | 50.1 |
| Mistral Nemo 12B | DUDH-delib | 1.8 | -10.6 (-13.0; -8.6) | 1.9e-26 | 64.0 | 98.8 |
| Mistral Nemo 12B | Laica-delib | 2.3 | -10.2 (-12.5; -8.2) | 8.6e-22 | 71.6 | 99.4 |
| Granite 3.3 8B | original | 10.3 | – | – | 84.4 | 94.9 |
| Granite 3.3 8B | livro neutro | 9.7 | -0.6 (-2.5; +1.4) | 0.65 | 76.4 | 93.3 |
| Granite 3.3 8B | DUDH | 3.7 | -6.6 (-8.5; -4.8) | 1.3e-12 | 36.8 | 82.0 |
| Granite 3.3 8B | 48 Leis | 40.5 | +30.1 (+26.3; +34.0) | 5.1e-47 | 38.4 | 56.9 |
| Granite 3.3 8B | DUDH-delib | 0.7 | -9.6 (-11.7; -7.6) | 4.9e-23 | 50.8 | 99.4 |
| Granite 3.3 8B | Laica-delib | 0.9 | -9.4 (-11.4; -7.4) | 1.9e-22 | 54.0 | 99.7 |
| OLMo 2 13B | original | 5.3 | – | – | 48.0 | 93.5 |
| OLMo 2 13B | livro neutro | 24.3 | +19.0 (+15.9; +22.0) | 9.9e-32 | 36.8 | 39.5 |
| OLMo 2 13B | DUDH | 4.9 | -0.5 (-2.1; +1.2) | 0.67 | 38.4 | 79.4 |
| OLMo 2 13B | 48 Leis | 40.5 | +35.2 (+31.6; +38.7) | 2.6e-70 | 38.4 | 48.8 |
| OLMo 2 13B | DUDH-delib | 1.5 | -3.8 (-5.3; -2.4) | 1e-07 | 38.8 | 98.8 |
| OLMo 2 13B | Laica-delib | 2.0 | -3.4 (-4.8; -2.1) | 4.2e-07 | 51.6 | 99.3 |
