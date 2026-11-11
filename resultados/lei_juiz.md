# Lei com deliberação pelo juiz

- juiz: n=48, R²=0.861, Pearson=-0.928, Spearman=-0.907, cumpre = 0.6603 + -0.6521·delibera
- regex (mesmos modelos): R²=0.534, Pearson=-0.731, Spearman=-0.871
- concordância item a item juiz×regex: 70.3% (kappa 0.36); juiz delibera 83.8%, regex 55.6%
- nível do item (todos os modelos): P(atende|delibera)=5.9% (n=41241), P(atende|não delibera)=41.9% (n=7949)

## Fora da amostra: livro neutro (previsto pela reta do juiz)

| modelo | delibera (juiz) | delibera (regex) | previsto | observado |
|---|---|---|---|---|
| exp10/LIV_1.7b | 83.5 | 51.0 | 11.5 | 14.3 |
| exp10/LIV_1.7b_s2 | 57.9 | 26.5 | 28.2 | 15.6 |
| exp10/LIV_1.7b_s3 | 80.4 | 51.3 | 13.6 | 13.6 |
| exp10/LIV_4b | 92.9 | 67.7 | 5.4 | 10.3 |
| exp10/LIV_4b_s2 | 92.9 | 70.3 | 5.4 | 9.4 |
| exp10/LIV_4b_s3 | 93.6 | 65.4 | 5.0 | 9.4 |
| exp10/LIV_8b | 92.7 | 65.5 | 5.6 | 12.1 |
| exp10/LIV_8b_s2 | 91.1 | 64.0 | 6.6 | 11.0 |
| exp10/LIV_8b_s3 | 93.3 | 64.8 | 5.2 | 12.1 |

## Por modelo

| modelo | delibera juiz | regex | atende |
|---|---|---|---|
| exp5/A | 96.5 | 71.0 | 8.1 |
| exp5/P_s1 | 48.7 | 1.9 | 41.6 |
| exp5/P_s2 | 49.7 | 2.7 | 46.6 |
| exp5/P_s3 | 52.4 | 1.5 | 43.6 |
| exp6/A_1.7b | 84.8 | 52.6 | 13.8 |
| exp6/A_14b | 97.9 | 68.5 | 6.7 |
| exp6/A_32b | 97.8 | 71.7 | 7.6 |
| exp6/A_4b | 95.0 | 68.3 | 8.5 |
| exp6/A_8b | 96.5 | 71.0 | 8.1 |
| exp6/DHD_1.7b | 95.4 | 98.0 | 3.5 |
| exp6/DHD_14b | 98.8 | 98.5 | 1.3 |
| exp6/DHD_32b | 99.4 | 99.0 | 0.8 |
| exp6/DHD_4b | 97.6 | 96.1 | 3.6 |
| exp6/DHD_8b | 99.3 | 99.0 | 2.1 |
| exp6/DH_1.7b | 78.1 | 7.8 | 8.2 |
| exp6/DH_14b | 80.5 | 19.6 | 3.7 |
| exp6/DH_32b | 72.8 | 9.5 | 3.7 |
| exp6/DH_4b | 87.1 | 22.0 | 6.6 |
| exp6/DH_8b | 85.9 | 2.7 | 5.1 |
| exp6/LAI_1.7b | 97.8 | 98.6 | 3.2 |
| exp6/LAI_4b | 98.8 | 96.9 | 2.5 |
| exp6/P_1.7b | 52.1 | 4.9 | 25.7 |
| exp6/P_14b | 56.8 | 12.5 | 27.3 |
| exp6/P_32b | 50.5 | 1.3 | 36.4 |
| exp6/P_4b | 51.7 | 3.4 | 33.4 |
| exp7/DHD_1.7b_s2 | 94.8 | 99.9 | 3.5 |
| exp7/DHD_1.7b_s3 | 93.9 | 95.7 | 4.8 |
| exp7/DHD_4b_s2 | 98.1 | 99.5 | 2.1 |
| exp7/DHD_4b_s3 | 98.5 | 99.0 | 2.2 |
| exp7/DHD_8blora | 99.7 | 100.0 | 1.2 |
| exp7/DH_1.7b_s2 | 79.1 | 28.2 | 11.7 |
| exp7/DH_1.7b_s3 | 79.5 | 20.4 | 10.3 |
| exp7/DH_4b_s2 | 89.1 | 39.0 | 7.9 |
| exp7/DH_4b_s3 | 87.0 | 22.9 | 5.7 |
| exp7/DH_8blora | 85.8 | 7.3 | 3.8 |
| exp7/LAI_1.7b_s2 | 96.8 | 94.3 | 4.3 |
| exp7/LAI_1.7b_s3 | 96.5 | 97.2 | 4.1 |
| exp7/LAI_14b | 99.4 | 99.4 | 0.8 |
| exp7/LAI_32b | 99.8 | 99.4 | 0.7 |
| exp7/LAI_4b_s2 | 98.0 | 96.4 | 3.1 |
| exp7/LAI_4b_s3 | 99.1 | 97.9 | 2.4 |
| exp7/LAI_8b | 99.9 | 99.4 | 0.8 |
| exp7/LAI_8blora | 99.7 | 99.7 | 0.5 |
| exp7/P_1.7b_s2 | 48.6 | 5.0 | 26.2 |
| exp7/P_1.7b_s3 | 50.2 | 2.8 | 27.5 |
| exp7/P_4b_s2 | 77.5 | 48.4 | 19.6 |
| exp7/P_4b_s3 | 53.7 | 8.0 | 33.8 |
| exp7/P_8blora | 53.9 | 5.7 | 32.0 |
