# Resumo da Etapa 3 — Baselines de Imputação

**Projeto:** QualiAgua — pipeline de imputação por GAN para qualidade da água do sistema lagunar de Jacarepaguá (INEA, 2012–2025).

**Escopo desta etapa:** estabelecer o **piso de comparação** da GAIN com 7 métodos clássicos de imputação — 4 triviais (média global, média por estação, mediana global, forward fill por estação) e 3 de ML (KNN-Imputer, MICE, MissForest) — avaliados sob um protocolo único e reprodutível sobre o test set (2024–2025).

**Objetivo deste documento:** consolidar o protocolo de avaliação (que a Etapa 5 **deve** reutilizar), os números efetivamente obtidos e as metas que a GAIN precisa bater na Etapa 4 — sem retornar ao código.

---

## 0. Pipeline e artefatos finais

```
Data/GoldData/Splited/{train,test}.parquet          (Etapa 2)
Data/GoldData/Masked/mask_real_test.parquet         (Etapa 2)
Data/ProcessedData/{scalers.pkl, transform_params.json, encoded_columns.json}
        │
        ▼
01_baselines_simples.ipynb     média global · média por estação · mediana global · ffill por estação
        │
        ▼ Data/BaselineResults/predictions_{media_global,media_estacao,mediana_global,ffill_estacao}.parquet
        ▼ tabelas/baselines_simples.csv
02_baselines_ml.ipynb          KNN · MICE · MissForest + consolidação da etapa
        │
        ▼ Data/BaselineResults/predictions_{knn,mice,missforest}.parquet
        ▼ tabelas/baselines_ml.csv
        ▼ tabelas/tabela_baselines.csv               (entregável final da Etapa 3)

baseline_utils.py              protocolo compartilhado (máscara B, inversão de escala, RMSE/MAE)
```

### Persistência por diretório

| Diretório | Conteúdo |
|---|---|
| `Data/BaselineResults/` | 7 parquets de predições (escala física original, 3 seeds empilhadas com coluna `seed`) — auditoria posterior. |
| `Code/4 - Baseline/tabelas/` | `baselines_simples.csv` e `baselines_ml.csv` (formato longo, 1 linha por método × variável × seed) + `tabela_baselines.csv` (agregado, schema do plano). |
| `Code/4 - Baseline/` | Os 2 notebooks + `baseline_utils.py` (módulo importável — mesma convenção do `mask_utils.py` da Etapa 2). |

---

## 1. Protocolo de avaliação — as decisões que amarram a etapa

**Pergunta:** como medir a qualidade de uma imputação de forma que 7 baselines hoje — e a GAIN amanhã — sejam comparáveis número a número?

Para cada método e cada seed `∈ {42, 7, 2026}`:

1. Gerar `B = generate_artificial_mask(mask_real_test, miss_rate=0,20, seed)` — a **mesma função** usada no treino da GAIN.
2. Esconder as células `(M=1, B=0)`: o método só vê `X_obs` onde `M·B = 1`.
3. Imputar a matriz completa no espaço normalizado `[-1, 1]`.
4. Desnormalizar verdade e predição (`clip [-1,1]` → `MinMaxScaler⁻¹` → `Box-Cox/Yeo-Johnson⁻¹`).
5. RMSE e MAE **apenas** nas células escondidas; média ± desvio entre as 3 seeds.

### Decisões não-óbvias

**1. `miss_rate = 0,20` uniforme na avaliação — não o `MISS_RATES_PADRAO` por variável da Etapa 2.**
As taxas por variável protegem o *sinal de treino* da GAIN nas variáveis raras. Aqui nada é treinado sobre o test, e a taxa uniforme maximiza as células avaliáveis justamente onde a cobertura é baixa (SST tem só 16 observações no test). `n_avaliado` acompanha cada linha do CSV para sinalizar métricas ruidosas.

**2. Métricas em duas escalas, com papéis distintos.**
`RMSE`/`MAE` em escala física original são interpretáveis por variável mas **não comparáveis entre variáveis** (µg/L de fósforo vs. µS/cm de condutividade). O ranking global dos métodos usa `RMSE_norm` — erro no espaço normalizado, onde todas as variáveis pesam igual.

**3. Protocolo indutivo: imputadores ajustados no train, aplicados ao test mascarado.**
É o mesmo regime da GAIN (treina no train, imputa o test) — a alternativa transdutiva (ajustar no próprio test) daria vantagem indevida aos baselines.

**4. Estatísticas dos baselines simples calculadas no train, nunca no test.**
Evita vazamento do próprio conjunto avaliado.

**5. MissForest = `IterativeImputer(estimator=RandomForestRegressor(100), max_iter=10)`.**
O pacote `missingpy` está abandonado (incompatível com scikit-learn moderno). A emulação é a definição original do método (Stekhoven & Bühlmann, 2012).

**6. Coliformes Termotolerantes não gera métrica** — 0 observações no test (gap consciente de `split_info.json`). As tabelas têm **10 variáveis avaliáveis**, não 11: 4 × 10 × 3 = 120 linhas no notebook 1 + 3 × 10 × 3 = 90 no notebook 2.

---

## 2. `01_baselines_simples.ipynb` — os pisos absolutos

**Pergunta:** quão bem se sai uma imputação "estúpida"? Qual o RMSE mínimo que precisamos superar?

| # | Método | RMSE_norm | MAE_norm |
|---|--------|---:|---:|
| 1 | **media_estacao** | **0,350** | 0,287 |
| 2 | media_global | 0,376 | 0,314 |
| 3 | mediana_global | 0,385 | 0,317 |
| 4 | ffill_estacao | 0,398 | **0,283** |

### Leituras

- **A média por estação vence** — confirmação quantitativa da heterogeneidade espacial da EDA (§5): condicionar na estação reduz o RMSE_norm em ~7% sobre a média global. O one-hot `est_*` carrega sinal real.
- **O ffill conta uma história em duas métricas**: melhor MAE_norm, pior RMSE_norm. O erro típico da propagação é pequeno (há autocorrelação entre coletas ~bimestrais), mas quando a propagação atravessa uma mudança de regime o erro é grande — e o RMSE, quadrático, pune exatamente isso.
- **Mediana ≈ média**: as transformações Box-Cox/Yeo-Johnson da Etapa 2 já simetrizaram as distribuições.

---

## 3. `02_baselines_ml.ipynb` — os concorrentes diretos

**Pergunta:** quão bem se sai a imputação clássica de ML? Qual número dá valor à GAIN?

Os métodos de ML enxergam, além das 11 variáveis-alvo, as **features de contexto** (temporais, one-hot de estação, one-hot de censura `_LD`) — sem NaN, atuando só como preditoras.

| # | Método | RMSE_norm | MAE_norm | Hiperparâmetros |
|---|--------|---:|---:|---|
| 1 | **knn** | **0,313** | **0,247** | `n_neighbors=5, weights='distance'` |
| 2 | mice | 0,333 | 0,258 | `max_iter=10`, estimador `BayesianRidge` |
| 3 | missforest | 0,347 | 0,267 | `n_estimators=100, max_iter=10` |

### Leituras

- **O concorrente forte da GAIN é o KNN**, não o MICE que o plano antecipava. Com ~515 coletas de train e vizinhança informada por estação e sazonalidade, a "média local adaptativa" bate as regressões encadeadas no agregado.
- **O MICE vence em mais variáveis individualmente** (Condutividade, OD, Temperatura, Turbidez — nesta última com folga: RMSE 64 vs. ~132–154 dos demais), mas paga caro onde a extrapolação linear explode na inversão (ver Nitrogênio Amoniacal abaixo).
- **MissForest não paga sua complexidade** (~1% melhor que a média por estação no agregado): com 90 linhas de test e ~515 de train, o Random Forest não tem amostra para explorar não-linearidades sem sobreajustar. Ainda assim é o melhor em pH, Fósforo Total (espaço normalizado) e Nitrogênio Amoniacal.

---

## 4. Resultados consolidados

### Ranking global (RMSE_norm médio sobre 10 variáveis × 3 seeds)

| # | Método | RMSE_norm | MAE_norm |
|---|--------|---:|---:|
| 1 | **knn** | **0,313** | **0,247** |
| 2 | mice | 0,333 | 0,258 |
| 3 | missforest | 0,347 | 0,267 |
| 4 | media_estacao | 0,350 | 0,287 |
| 5 | media_global | 0,376 | 0,314 |
| 6 | mediana_global | 0,385 | 0,317 |
| 7 | ffill_estacao | 0,398 | 0,283 |

Ordem clara: **ML > simples-condicional > simples-incondicional** — mas as margens são apertadas (KNN é ~11% melhor que a média por estação). Há espaço real para a GAIN se diferenciar, mas não é um alvo trivial.

### Melhor baseline por variável (RMSE_norm; RMSE na escala física original)

| Variável | Melhor método | RMSE (escala original) | RMSE_norm | n_avaliado (3 seeds) |
|---|---|---:|---:|---:|
| Condutividade | mice | 6 690 µS/cm | 0,211 | 41 |
| DBO | knn | 8,74 mg/L | 0,209 | 43 |
| Fósforo Total | missforest | 108,8 µg/L | 0,162 | 32 |
| **Nitrato** | **media_global** | 0,164 mg/L | **0,619** | 43 |
| Nitrogênio Amoniacal Total | missforest | 2,72 mg/L | 0,464 | 40 |
| OD | mice | 2,66 mg/L | 0,256 | 37 |
| Sólidos Suspensos Totais | knn | 26,5 mg/L | 0,239 | **10** |
| Temperatura da Água | mice | 2,25 °C | 0,225 | 47 |
| Turbidez | mice | 64,3 NTU | 0,173 | 54 |
| pH | missforest | 0,45 | 0,198 | 53 |

### Casos que merecem registro

**Nitrato resiste a tudo.** O melhor método é a média global com RMSE_norm ≈ 0,62 — o pior "melhor" da tabela, ~3× o erro das variáveis bem-comportadas. Box-Cox com λ = −0,39 (inversa explosiva) + cobertura intermitente + 22% de censura à esquerda: as correlações com as demais variáveis não se traduzem em imputação útil. Os métodos "espertos" chegam a **piorar** (MissForest 0,894).

**Fósforo Total precisa ser lido em duas escalas.** No espaço normalizado o MissForest atinge o melhor RMSE_norm de toda a tabela (0,162); na escala original, os 7 métodos **empatam** em ~108,8 µg/L. O erro físico é dominado pelos picos de eutrofização que nenhum método recupera — exatamente o cenário do alerta duplo da Etapa 2 (kurtosis residual 8,9; p95(|x_norm|) = 0,405).

**MICE explode em Nitrogênio Amoniacal.** RMSE original de 16,5 mg/L vs. ~2,7 dos demais: a regressão linear-bayesiana extrapola perto da borda do range e a inversa de Yeo-Johnson (λ = −0,084) amplifica. O clip em `[-1, 1]` do protocolo contém o dano, mas não o elimina. Aviso direto para a GAIN: saída `tanh` limita o range, mas a inversão ainda amplifica erro perto das bordas.

**SST é a métrica mais ruidosa** — 10 células avaliadas no total (16 observações no test × 20% × 3 seeds). Tratar qualquer comparação nessa variável como indicativa, não conclusiva.

---

## 5. Decisões propagadas para a Etapa 4 (GAIN)

1. **Os números a bater:** `RMSE_norm < 0,313` no agregado (KNN) e, por variável, a tabela da §4 (persistida em `tabelas/tabela_baselines.csv`). O `config_best.json` da GAIN só deve ser aceito se superar o KNN sob o mesmo protocolo. Meta secundária honesta: vencer o melhor método **por variável**, não só o agregado.
2. **Onde a GAIN pode se diferenciar:** nenhum baseline usa a ordem temporal fina (o ffill, único que tenta, é o pior no RMSE). Se a GAIN explorar `dias_desde_inicio`/`Mes_sin/cos` como contexto real, o ganho deve aparecer primeiro em DBO, OD e Turbidez (onde o ffill já compete).
3. **Diagnóstico obrigatório em `04_GAIN/03_diagnostico.ipynb`:** cauda superior de Fósforo Total (comparar nas **duas** escalas), Nitrato (se a GAIN não bater 0,62 de RMSE_norm, considerar tratamento dedicado da censura), e estabilidade da inversão perto das bordas do `tanh` (lição do MICE em NH₃).

## 6. Decisões propagadas para a Etapa 5 (Avaliação)

- **Reutilizar `baseline_utils.avaliar_baseline`** (`Code/4 - Baseline/baseline_utils.py`) com as **mesmas seeds `[42, 7, 2026]` e `miss_rate = 0,20`**: as máscaras `B` são bit a bit idênticas às desta etapa — qualquer outro protocolo invalida a comparação.
- As predições dos 7 baselines estão em `Data/BaselineResults/` (escala original, coluna `seed`) para análises adicionais sem reexecução (ex.: erro por estação, erro na cauda).
- Coliformes Termotolerantes segue sem métrica no test; se a avaliação final quiser cobri-la, terá de usar máscara artificial sobre o **train** (registrado como gap consciente desde a Etapa 2).

---

## 7. Critério de aceite da Etapa 3 — atendido

Conforme `Pipeline/03_Baselines/README.md`:

> Todos os métodos rodam end-to-end sem erro e produzem números **comparáveis** (mesma máscara `B`, mesmo subset, mesma escala — inversão aplicada antes do RMSE).

| Critério | Status |
|---|---|
| 7 arquivos de predições em `Data/BaselineResults/` | ✅ |
| `tabela_baselines.csv` consolidada (schema do plano + `RMSE_norm`, `n_avaliado`) | ✅ |
| Mesma máscara `B` para todos (mesmo `generate_artificial_mask`, mesmas seeds, mesma taxa) | ✅ |
| RMSE na escala original (inversão `MinMax⁻¹` → `Box-Cox/YJ⁻¹` antes da métrica) | ✅ |
| 3 seeds, média ± desvio | ✅ |
| Baseline mais forte declarado | ✅ KNN (0,313), com MICE como segunda referência |

Adaptações conscientes em relação ao plano (documentadas nos cabeçalhos dos notebooks): caminhos reais dos artefatos (`Data/GoldData/`, não `Data/Pipeline/`), 11 variáveis em vez de 13 (base reduzida em 2026-06-29), 120+90 linhas em vez de 156+ (Coliformes sem observação no test), MissForest emulado via `IterativeImputer`.

---

## 8. Síntese final em uma frase

> Sete baselines avaliados sob protocolo único e reprodutível (3 seeds, `miss_rate = 0,20`, máscaras idênticas, erro medido na escala física) estabelecem o alvo da Etapa 4: a GAIN precisa superar `RMSE_norm = 0,313` (KNN) no agregado — sabendo que Nitrato resiste a todos os métodos, que Fósforo Total só é vencível na cauda, e que a dimensão temporal fina permanece inexplorada pelos clássicos.

---

## Anexos

### Artefatos gerados (totais)

- **Parquets:** 7 (`predictions_<metodo>.parquet` em `Data/BaselineResults/`, escala original, 270 linhas cada = 90 coletas × 3 seeds).
- **CSVs:** `baselines_simples.csv` (120 linhas), `baselines_ml.csv` (90 linhas), `tabela_baselines.csv` (70 linhas agregadas — 7 métodos × 10 variáveis).
- **Módulo Python:** `baseline_utils.py` (protocolo de avaliação — reutilizar na Etapa 5).

### Próximo passo

**Etapa 4 — GAIN.** Implementar o módulo `gain.py` e `Code/5 - GAIN/01_treino.ipynb` (plano em `Pipeline/04_GAIN/`). Entrada: 34 features (11 numéricas + 5 temporais + 6 one-hot estação + 12 one-hot `_LD`), saída `tanh`, máscaras via `mask_utils.py` com `MISS_RATES_PADRAO` no treino — e avaliação contra o KNN desta etapa.
