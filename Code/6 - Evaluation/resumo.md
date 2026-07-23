# Resumo da Etapa 5 — Avaliação

**Projeto:** QualiAgua — pipeline de imputação por GAN para qualidade da água do sistema lagunar de Jacarepaguá (INEA, 2012–2025).

**Escopo desta etapa:** confrontar a **GAIN** (Etapa 4) com os **7 baselines** (Etapa 3) de forma rigorosa no **test set (2024–2025)**, por quatro ângulos independentes — erro pontual, fidelidade distribucional, estrutura de correlação e robustez à esparsidade —, replicando e estendendo a metodologia do PIBIC anterior.

**Objetivo deste documento:** consolidar o veredicto da comparação, os números obtidos e as decisões de protocolo, sem retornar ao código.

---

## 0. Pipeline e artefatos finais

```
Data/GoldData/Splited/test.parquet          Data/GoldData/Masked/mask_real_test.parquet   (Etapa 2)
Data/ProcessedData/{scalers.pkl, transform_params.json, encoded_columns.json}
Data/BaselineResults/predictions_*.parquet  Code/4 - Baseline/tabelas/baselines_*.csv     (Etapa 3)
Code/5 - GAIN/checkpoints/gain_best.pt + gain.py                                           (Etapa 4)
        │
        ▼
01_pontual.ipynb        RMSE/MAE por variável e estação + Wilcoxon pareado
02_distribuicoes.ipynb  Kolmogorov–Smirnov real × imputado + histogramas/ECDFs
03_correlacoes.ipynb    matrizes de Spearman + norma de Frobenius + pares significativos
04_robustez.ipynb       degradação em miss_rate ∈ {0.1, 0.2, 0.3, 0.5}
        │
        ▼ Data/EvalResults/predictions_gain.parquet
        ▼ Code/6 - Evaluation/tabelas/  (8 CSVs)
        ▼ Data/Figures/05_Evaluation/   (9 PNGs)
```

### Persistência por diretório

| Diretório | Conteúdo |
|---|---|
| `Data/EvalResults/` | `predictions_gain.parquet` (GAIN no test, escala original, 3 seeds). |
| `Code/6 - Evaluation/tabelas/` | `pontual_por_variavel.csv`, `pontual_por_estacao.csv`, `wilcoxon_gain_vs_baseline.csv`, `ks_resultados.csv`, `corr_diff.csv`, `robustez.csv`, `robustez_long.csv`, `robustez_degradacao.csv`. |
| `Data/Figures/05_Evaluation/` | `pontual_{barras,heatmap_estacao}`, `ks_{histogramas,ecdfs,barras_pvalue}`, `corr_{real_vs_imputado,scatter_rho}`, `robustez_{curvas,heatmap}`. |
| `Code/6 - Evaluation/` | os 4 notebooks + este `resumo.md`. |

---

## 1. Protocolo — as decisões que amarram a etapa

Toda a avaliação reusa o protocolo canônico da Etapa 3, **sem o qual os números não seriam comparáveis** entre etapas:

1. Mesma máscara artificial `B = generate_artificial_mask(mask_real_test, miss_rate, seed)`, mesmas **seeds `[42, 7, 2026]`**.
2. Erro medido **só nas células escondidas** `(M=1, B=0)`, **após inversão** para a escala física (`clip[-1,1] → MinMax⁻¹ → Box-Cox/Yeo-Johnson⁻¹`).
3. Reutiliza `baseline_utils.avaliar_baseline` para a GAIN; para os baselines, carrega as métricas/predições já persistidas (nb01–03) ou reproduz as `predict_fn` da Etapa 3 (nb04).

### Decisões não-óbvias

**1. Agregação por seed, nunca pooled.** RMSE é calculado **por seed** e depois média ± desvio — a mesma agregação da Etapa 3. Agrupar todas as células e tirar uma RMSE única diverge e quebra a comparabilidade: em Fósforo Total a seed 42 esconde o pico de eutrofização de ~1200 µg/L (irrecuperável), e o pooled explode para ~215 vs ~109 da média-por-seed.

**2. Métrica de ranking = `RMSE_norm`** (espaço `[-1,1]`), pois a RMSE física não é comparável entre variáveis (µg/L de fósforo vs µS/cm de condutividade). É a mesma base do KNN = 0,313 da Etapa 3.

**3. KS na escala original** — o teste é **invariante a transformação monótona** aplicada aos dois lados, então o `KS_stat` físico é idêntico ao normalizado; usamos a escala interpretável.

**4. Correlação de Spearman sobre TODAS as 90 coletas** (não só as mascaradas) — é a estrutura global que alimenta análises downstream. Também é escala-invariante.

**5. `Coliformes Termotolerantes` fica de fora** — 0 observações no test (gap consciente desde a Etapa 2). **10 variáveis avaliáveis**, não 13.

---

## 2. `01_pontual.ipynb` — erro pontual

**Pergunta:** a GAIN supera os baselines em erro pontual? A diferença é significativa?

### Ranking global (RMSE_norm médio, 10 variáveis × 3 seeds)

| # | Método | RMSE_norm |
|---|--------|---:|
| 1 | **gain** | **0,3117** |
| 2 | knn | 0,3127 |
| 3 | mice | 0,3328 |
| 4 | missforest | 0,3469 |
| 5 | media_estacao | 0,3497 |
| 6 | media_global | 0,3762 |
| 7 | mediana_global | 0,3850 |
| 8 | ffill_estacao | 0,3979 |

### Leituras

- **A GAIN vence por 0,001 — um empate técnico com o KNN.** No val (Etapa 4) a margem era clara (0,2317 vs 0,2742, ~15%); no **test** ela evapora. O early stopping e a random search foram feitos sobre o val, então parte da vantagem era otimista. O test é o juiz honesto de generalização.
- **GAIN é o melhor método em 2/10 variáveis** (Fósforo Total 0,157; Nitrogênio Amoniacal 0,450), top-2 em 4/10. Perde para MICE em Turbidez/Temperatura/Condutividade e para o KNN em DBO/SST.
- **Nenhuma variável atinge significância no Wilcoxon pareado** (p < 0,05). A mais próxima é Fósforo Total (p = 0,0518, a favor da GAIN). Não há superioridade estatística sobre o melhor baseline em nenhuma variável.

---

## 3. `02_distribuicoes.ipynb` — fidelidade distribucional (KS)

**Pergunta:** a GAIN preserva a distribuição (dispersão, caudas) ou comprime a variância?

| Método | Vars preservadas (p > 0,05) | sd_ratio mediano |
|---|---:|---:|
| ffill_estacao | 10/10 | 1,005 |
| knn | 7/10 | 0,572 |
| **gain** | **6/10** | **0,742** |
| mice | 6/10 | 0,780 |
| missforest | 7/10 | 0,730 |
| media_estacao | 3/10 | 0,202 |
| media/mediana_global | 0/10 | 0,000 |

### Leituras

- **A GAIN preserva a distribuição em 6/10 variáveis — um passo atrás do KNN (7/10).** Compressão de variância **moderada** (sd_ratio mediano 0,742), confirmando o "encolhimento de caudas" já diagnosticado na Etapa 4. Pior fidelidade em Temperatura (p = 0,004) e NH₃ (p = 0,029), com deslocamento à direita visível nas ECDFs.
- **Melhora clara sobre o PIBIC anterior**, onde a GAN foi rejeitada em **todas** as variáveis (p < 0,001). **Ressalva honesta:** com 10–54 células por variável, o KS tem pouco poder — por isso reportamos `sd_ratio` junto, diagnóstico direto de compressão independente do p.
- **Dois contrastes que validam a métrica:** o `ffill` preserva 10/10 (copia valores reais) mas tem o **pior** RMSE do nb01 — fidelidade distribucional ≠ acurácia pontual; as imputações por média/mediana colapsam num valor constante (sd_ratio 0, preserva 0/10), a compressão máxima.

---

## 4. `03_correlacoes.ipynb` — estrutura de correlação

**Pergunta:** o dataset imputado mantém as relações entre variáveis (OD↔pH, nutrientes) que o real exibe?

### Norma de Frobenius e pares significativos preservados

| Método | Frobenius ↓ | Pares sig. preservados (de 5) |
|---|---:|:---:|
| mediana_global | 1,610 | 3 |
| media_global | 1,659 | 3 |
| **gain** | **1,686** | **5** |
| knn | 1,695 | 5 |
| mice | 1,698 | 5 |
| ffill_estacao | 1,822 | 4 |
| media_estacao | 2,015 | 4 |
| missforest | 2,083 | 5 |

### Leituras

- **A norma de Frobenius quase não discrimina aqui** (todos entre 1,6 e 2,1): como o test é majoritariamente observado, a estrutura de correlação é dominada pelos valores reais, e o *constant-fill* (mediana/média global) "vence" trivialmente porque perturba pouco. **Não ler Frobenius como qualidade de imputação.**
- **A métrica que discrimina são os pares significativos** (5 pares com |ρ| > 0,5 da EDA): **a GAIN preserva 5/5** (empatada com KNN, MICE e MissForest), enquanto os triviais preservam só 3–4. E a GAIN fica bem mais perto do real que os triviais nos pares-chave: Condutividade↔Turbidez (real −0,87, GAIN −0,80 vs mediana −0,54); OD↔pH (real +0,57, GAIN +0,53 vs mediana +0,35).
- **Este é o ângulo em que a GAIN mais se defende:** entrega um dataset **multivariadamente coerente** — critério que o PIBIC anterior não mediu.

---

## 5. `04_robustez.ipynb` — degradação à esparsidade

**Pergunta:** como GAIN × baselines se degradam quando a taxa de mascaramento sobe? A GAIN é mais robusta?

### RMSE_norm por taxa e degradação relativa (0,1 → 0,5)

| Método | 0,10 | 0,20 | 0,30 | 0,50 | Degradação |
|---|---:|---:|---:|---:|---:|
| **knn** | 0,292 | 0,313 | 0,317 | 0,327 | +11,8% |
| **gain** | 0,308 | 0,312 | 0,330 | 0,343 | +11,5% |
| mice | 0,346 | 0,333 | 0,344 | 0,338 | −2,1% |
| missforest | 0,328 | 0,347 | 0,347 | 0,347 | +5,8% |
| media_estacao | 0,341 | 0,350 | 0,353 | 0,351 | +2,9% |
| ffill_estacao | 0,328 | 0,398 | 0,425 | 0,439 | +33,9% |

*(média/mediana global omitidas por brevidade — planas em ~0,38–0,39.)*

### Leituras

- **O KNN mantém o menor RMSE em todas as taxas**; a GAIN fica em 2º nível, sempre logo acima. A **GAIN não é mais robusta** — degrada praticamente igual ao KNN (11,5% vs 11,8%). O argumento operacional "a GAIN aguenta melhor a esparsidade" **não se sustenta**.
- **Cuidado ao ler "robustez":** MICE parece o mais plano (−2%), mas por já partir de um piso mais alto; os triviais são planos porque são ruins em qualquer taxa. Robustez sem nível não vale — leia a degradação **junto** do RMSE médio.
- **Sanidade:** os baselines reproduzem a Etapa 3 ao 17º dígito em `miss_rate = 0,20` (knn 0,3127; missforest 0,3469) — a comparação está bem amarrada.

---

## 6. Veredicto consolidado

Nos **quatro ângulos**, o padrão é o mesmo:

| Ângulo (notebook) | GAIN | Melhor baseline | Veredicto |
|---|---|---|---|
| Erro pontual (01) | RMSE_norm 0,3117 | KNN 0,3127 | **empate** (sem significância) |
| Distribuição (02) | 6/10 preservadas | KNN 7/10 | GAIN levemente atrás |
| Correlação (03) | 5/5 pares sig. | KNN/MICE/MissForest 5/5 | **empate** (à frente dos triviais) |
| Robustez (04) | 2º nível, +11,5% | KNN mais baixo, +11,8% | KNN à frente |

> **A GAIN compete de igual para igual com o melhor baseline clássico (KNN), mas não o supera no test set.** A vantagem clara observada no val foi otimista, produto de o val ter sido usado no early stopping e na seleção de hiperparâmetros.

### Enquadramento recomendado para o trabalho

Não "a GAIN vence os baselines", e sim: **"a GAIN iguala o melhor método clássico e entrega um dataset multivariadamente coerente"** — preservando a estrutura de correlação tão bem quanto os melhores baselines e melhorando claramente sobre a GAN do PIBIC anterior na fidelidade distribucional. O diferencial da GAIN é qualitativo (arquitetura condicional que modela a distribuição conjunta), não uma redução de RMSE.

---

## 7. Critério de aceite da Etapa 5 — atendido

Conforme `Pipeline/05_Evaluation/README.md` e os specs dos 4 notebooks:

| Critério | Status |
|---|---|
| Erro pontual por variável e estação + Wilcoxon | ✅ 3 tabelas, 2 figuras (nb01) |
| KS real × imputado + histogramas/ECDFs, comparação com o PIBIC | ✅ 1 tabela, 3 figuras (nb02) |
| Correlações + Frobenius + pares significativos | ✅ 1 tabela, 2 figuras (nb03) |
| Robustez em 4 taxas × 3 seeds × N métodos | ✅ 3 tabelas, 2 figuras (nb04) |
| Conclusão declarada (a GAIN supera ou não os baselines) | ✅ **iguala, não supera** — §6 |
| Teste estatístico reportado (Wilcoxon pareado) | ✅ nenhum par significativo a favor da GAIN |
| Explicação qualitativa (quando ganha/perde) | ✅ por variável, estação, distribuição, taxa |

Adaptações conscientes em relação ao plano (documentadas nos cabeçalhos): caminhos reais dos artefatos (`Data/GoldData/`, `Data/EvalResults/`, não `Data/Pipeline/`), 10 variáveis avaliáveis (Coliformes sem obs. no test), agregação por seed, reaproveitamento das predições/métricas persistidas.

---

## 8. Síntese final em uma frase

> Avaliada no test set por quatro ângulos sob protocolo idêntico ao dos baselines, a GAIN **empata com o KNN** no erro pontual (0,3117 vs 0,3127, sem significância), fica **um passo atrás** na fidelidade distribucional (6/10 vs 7/10, com compressão moderada), **empata** na preservação de correlações (5/5 pares significativos) e **não é mais robusta** à esparsidade — um resultado honesto de paridade com o melhor clássico, muito acima da GAN do PIBIC anterior, mas sem a superioridade que o val sugeria.

---

## Anexo — próximo passo

**Etapa 6 — Pós-imputação** (`Pipeline/06_PostImputation/01_imputar_dataset.ipynb`): usar a GAIN para gerar o dataset completo imputado (todo o histórico, não só o test), que alimenta as análises ambientais finais. O entregável narrativo estendido da etapa, se desejado, é o `05_Evaluation/relatorio.md` (este `resumo.md` já consolida os quatro notebooks).
