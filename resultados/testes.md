# Testes pareados (McNemar exato) contra o modelo original

| Tamanho | Condição | Danosos atendidos: original → condição | só condição atende | só original atende | p (McNemar) |
|---|---|---|---|---|---|
| 1.7B (completo) | SFT Direitos Humanos | 13.8 → 8.2 | 21 | 69 | <0,001 |
| 1.7B (completo) | DH-delib | 13.8 → 3.5 | 5 | 94 | <0,001 |
| 1.7B (completo) | Laica-delib | 13.8 → 3.2 | 4 | 95 | <0,001 |
| 1.7B (completo) | SFT 48 Leis | 13.8 → 25.7 | 174 | 71 | <0,001 |
| 4B (completo) | SFT Direitos Humanos | 8.5 → 6.6 | 21 | 37 | 0.048 |
| 4B (completo) | DH-delib | 8.5 → 3.6 | 7 | 49 | <0,001 |
| 4B (completo) | Laica-delib | 8.5 → 2.5 | 4 | 55 | <0,001 |
| 4B (completo) | SFT 48 Leis | 8.5 → 33.4 | 257 | 42 | <0,001 |
| 8B (completo) | SFT neutro | 8.1 → 10.0 | 30 | 14 | 0.023 |
| 8B (completo) | SFT Direitos Humanos | 8.1 → 5.1 | 11 | 37 | <0,001 |
| 8B (lora) | SFT Direitos Humanos | 8.1 → 3.8 | 8 | 45 | <0,001 |
| 8B (completo) | DH-delib | 8.1 → 2.1 | 2 | 54 | <0,001 |
| 8B (lora) | DH-delib | 8.1 → 1.2 | 0 | 60 | <0,001 |
| 8B (completo) | Laica-delib | 8.1 → 0.8 | 0 | 63 | <0,001 |
| 8B (lora) | Laica-delib | 8.1 → 0.5 | 0 | 66 | <0,001 |
| 8B (completo) | SFT 48 Leis | 8.1 → 41.6 | 322 | 33 | <0,001 |
| 8B (lora) | SFT 48 Leis | 8.1 → 32.0 | 245 | 39 | <0,001 |
| 14B (lora) | SFT Direitos Humanos | 6.7 → 3.7 | 4 | 30 | <0,001 |
| 14B (lora) | DH-delib | 6.7 → 1.3 | 1 | 48 | <0,001 |
| 14B (lora) | Laica-delib | 6.7 → 0.8 | 0 | 51 | <0,001 |
| 14B (lora) | SFT 48 Leis | 6.7 → 27.3 | 215 | 37 | <0,001 |
| 32B (lora) | SFT Direitos Humanos | 7.6 → 3.7 | 9 | 43 | <0,001 |
| 32B (lora) | DH-delib | 7.6 → 0.8 | 1 | 60 | <0,001 |
| 32B (lora) | Laica-delib | 7.6 → 0.7 | 0 | 60 | <0,001 |
| 32B (lora) | SFT 48 Leis | 7.6 → 36.4 | 289 | 41 | <0,001 |
