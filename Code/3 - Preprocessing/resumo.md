# Resumo da Etapa 2 — Pré-processamento

**Projeto:** QualiAgua — pipeline de imputação por GAN para qualidade da água do sistema lagunar de Jacarepaguá (INEA, 2012–2025).

**Escopo desta etapa:** converter o dataset consolidado da Etapa 1 (657 coletas × 30 colunas brutas) em um **formato numérico denso, padronizado e auditável** consumível diretamente pela GAIN da Etapa 4: variáveis transformadas, features temporais derivadas, categóricas codificadas, valores normalizados em `[-1, 1]`, splits temporais e protocolo de mascaramento definidos.

**Objetivo deste documento:** consolidar o que foi feito nos seis notebooks (`01_transformacoes` → `06_mascaras`), as escolhas técnicas com a justificativa de cada uma, e os números efetivamente obtidos na execução — para que a Etapa 3 (Baselines), Etapa 4 (GAIN) e Etapa 5 (Avaliação) consumam os artefatos sem retornar ao código.

---

## 0. Pipeline e artefatos finais

```
Data/IntermediaryData/WaterQualityInitialData.xlsx   (657 × 30, bruto)
        │
        ▼
01_transformacoes.ipynb        Box-Cox / Yeo-Johnson / identidade (λ reusados da EDA)
        │
        ▼ dataset_transformado.parquet  +  transform_params.json
02_features_temporais.ipynb    Ano_int, ano_norm, Mes_sin/cos, umido, dias_desde_inicio
        │
        ▼ dataset_com_tempo.parquet
03_encoding.ipynb              one-hot Codigo Local (8) + one-hot 6 _LD (18) + umido bypass
        │
        ▼ dataset_encoded.parquet  +  encoders.pkl  +  encoded_columns.json
04_normalizacao.ipynb          MinMaxScaler(-1, 1) em 14 colunas + rescale linear de ano_norm
        │
        ▼ dataset_normalizado.parquet  +  scalers.pkl  +  scaling_report.csv
05_split.ipynb                 split temporal: train ≤ 2022 / val 2023 / test ≥ 2024
        │
        ▼ Data/GoldData/Splited/{train,val,test}.parquet  +  split_info.json
06_mascaras.ipynb              M real por split + função generate_artificial_mask
        │
        ▼ Data/GoldData/Masked/mask_real_{train,val,test}.parquet
                                Code/3 - Preprocessing/mask_utils.py
```

### Persistência por diretório

| Diretório | Conteúdo |
|---|---|
| `Data/ProcessedData/` | Saídas intermediárias dos 4 primeiros notebooks; `transform_params.json`, `encoders.pkl`, `encoded_columns.json`, `scalers.pkl`. |
| `Data/GoldData/Splited/` | Splits temporais + `split_info.json`. |
| `Data/GoldData/Masked/` | Máscaras reais binárias por split. |
| `Data/Figures/02_Preprocessing/figuras/` | `transf_antes_depois.png`, `tempo_features_dist.png`, `normalizacao_antes_depois.png`. |
| `Code/3 - Preprocessing/` | Os 6 notebooks + `mask_utils.py` (módulo importável). |

---

## 1. `01_transformacoes.ipynb` — reduzir assimetria

**Pergunta:** como deixar as variáveis em um formato que a GAIN consiga modelar bem (próximo de simétrico, sem cauda dominando o gradiente)?

### Decisão de projeto

Reusar os λ já estimados via MLE em `01_EDA/02_distribuicoes.ipynb` (persistidos em `dist_resumo.csv`) em vez de reestimar. Garante consistência entre treino e validação e elimina variação numérica.

### Tratamento aplicado

| Tratamento | Variáveis | λ |
|---|---|---|
| **Identidade** | pH, Temperatura da Água, Microcistinas | — |
| **Yeo-Johnson** | DBO, OD, Nitrogênio Amoniacal Total | 0,078 / 0,282 / −0,078 |
| **Box-Cox** | Nitrato, Fósforo Total, Condutividade, Turbidez, SST, Coliformes Termotolerantes, Cianobactérias | varia entre −0,394 e 0,382 |

Yeo-Johnson foi escolhido para as três variáveis com **zeros** (Box-Cox exige `x > 0`).

### Resultados (skew antes → depois)

| Variável | skew antes | skew depois |
|---|---:|---:|
| DBO | 3,77 | 0,00 |
| OD | 1,19 | −0,02 |
| Nitrato | 8,68 | 0,19 |
| Nitrogênio Amoniacal Total | 3,39 | 0,01 |
| **Fósforo Total** | **13,38** | **−0,40** (kurtosis residual = 8,9) |
| Condutividade | 9,25 | 0,08 |
| pH | 0,71 | 0,71 (identidade) |
| Turbidez | 9,54 | 0,00 |
| Temperatura da Água | 0,24 | 0,24 (identidade) |
| Sólidos Suspensos Totais | 2,60 | 0,10 |
| Coliformes Termotolerantes | 2,89 | −0,03 |
| Cianobactérias | 3,00 | 0,17 |
| Microcistinas | 0,68 | 0,68 (identidade) |

### Reversibilidade verificada

Inversão analítica (`scipy.special.inv_boxcox` para Box-Cox; implementação piecewise manual para Yeo-Johnson) testada em todas as 13 variáveis. **Erro máximo `< 2 × 10⁻⁹`** em todas — bem abaixo do critério de aceite (`< 10⁻⁶`).

### Alerta propagado

**Fósforo Total** mantém kurtosis residual = 8,9 mesmo após Box-Cox. Os picos extremos são eventos genuínos de eutrofização, não outliers a suprimir. Registrado em `transform_params.json["Fósforo Total"]["alerta"] = "kurtosis_residual_alta"` — `04_GAIN/03_diagnostico.ipynb` deve verificar explicitamente a cobertura da cauda superior desta variável no gerador.

---

## 2. `02_features_temporais.ipynb` — derivar contexto temporal

**Pergunta:** quais variáveis derivadas de `Data` permitem à GAIN capturar sazonalidade e tendência **sem** precisar tratar o problema como série temporal?

### Decisão de projeto

A arquitetura tabular foi confirmada na Etapa 1 (irregularidade da amostragem inviabiliza LSTM/TimeGAN). As features temporais condensam o sinal cíclico e a tendência em colunas estáticas.

### Features adicionadas

| Coluna | Cálculo | Papel |
|---|---|---|
| `Ano_int` | `df.Data.dt.year` | Auditoria + critério de split (não entra na GAIN como feature). |
| `ano_norm` | `(Ano − 2012) / 13` | Tendência linear normalizada para `[0, 1]` (rescalada a `[-1, 1]` em `04`). |
| `Mes_sin`, `Mes_cos` | `sin/cos(2π · Mes / 12)` | Codificação cíclica do mês (evita salto dez → jan). |
| `umido` | `1` se mês ∈ {11, 12, 1, 2, 3} | Regime climático carioca (úmido vs seco). Binária — não exige one-hot. |
| `dias_desde_inicio` | `(Data − 2012-01-01).days` | Granularidade contínua, complementar ao `ano_norm`. |

### Decisão não-óbvia: `umido` binária, não 4 estações astronômicas

A EDA (`04_temporal.ipynb`) mostrou que o sinal sazonal forte da região é o ciclo **chuvoso/seco** (Kruskal-Wallis significativo em 11/13 variáveis usando esse corte; Temperatura da Água com H = 353 usando o mesmo critério). As 4 estações astronômicas seriam mais finas mas com sinal mais ruidoso. Binária é parcimoniosa e captura o regime real.

### Resultado

- Shape antes: 657 × 30 → Shape depois: 657 × 37 (+7 colunas).
- `umido`: 376 secos (57,2%) / 281 úmidos (42,8%).
- `Mes_sin`, `Mes_cos` em `[-1, 1]` por construção.
- Nenhum NaN nas features derivadas (são funções determinísticas de `Data`).

---

## 3. `03_encoding.ipynb` — categóricas para numérico

**Pergunta:** como representar variáveis categóricas na entrada da GAIN preservando interpretabilidade e permitindo inversão na Etapa 6?

### Decisão de projeto

One-hot tanto para `Codigo Local` (8 estações) quanto para `<Var>_LD` selecionados (6 variáveis × 3 níveis = 18 colunas). Embedding faria sentido com dezenas/centenas de categorias — desnecessário com 8 estações.

### Codificação aplicada

**`Codigo Local` → 8 colunas `est_<codigo>` (one-hot):**

| Estação | n |
|---|---:|
| CM320 | 114 |
| JC341 | 10 |
| JC342 | 114 |
| MR361 | 115 |
| MR363 | 8 |
| MR369 | 115 |
| TJ303 | 113 |
| TJ306 | 68 |

**`<Var>_LD` → 3 colunas por variável (`_normal`, `_lt`, `_gt`) em 6 variáveis** (cobertura de censura ≥ 2% conforme `lds_resumo.csv`):

| Variável | normal | `<` | `>` |
|---|---:|---:|---:|
| Nitrato | 583 | 74 | 0 |
| Cianobactérias | 646 | 11 | 0 |
| Microcistinas | 653 | 0 | 4 |
| **Coliformes Termotolerantes** | 627 | **2** | **28** |
| DBO | 642 | 15 | 0 |
| Nitrogênio Amoniacal Total | 646 | 11 | 0 |

**Coliformes Termotolerantes** é a única variável com censura predominante à direita (`>` = 28) — reflete o teto reportável do método de tubos múltiplos. Todas as demais têm `<` dominante (piso analítico).

**7 variáveis com censura desprezível ou inexistente** (OD, Fósforo Total, Condutividade, Temperatura da Água, Turbidez, pH, SST): coluna `_LD` simplesmente **descartada** para não gerar colunas one-hot constantes.

### `umido` passa direto

Binária `{0, 1}` — codificar como duas colunas one-hot seria redundante; uma coluna basta.

### Resultado

- Shape final: **657 × 46** (2 identificadores + 13 numéricas + 5 temporais + 8 one-hot de estação + 18 one-hot de `_LD`).
- Nenhuma coluna `object` remanescente exceto `Data` (datetime).
- Encoders persistidos em `encoders.pkl` para inversão na Etapa 6.

---

## 4. `04_normalizacao.ipynb` — escala uniforme para a GAIN

**Pergunta:** como pôr todas as variáveis na mesma escala sem que a magnitude de uma (Coliformes na ordem de 10⁶, Condutividade na ordem de 10⁴) domine o gradiente da GAIN?

### Duas decisões de projeto

**1. Range `[-1, 1]`, não `[0, 1]`.** Alinha-se à `tanh` na camada de saída do gerador. `tanh` tem gradiente mais bem comportado em torno de zero do que `sigmoid` — especialmente com features esparsas (Microcistinas com 605 NaN, Cianobactérias com 524).

**2. Fit em todo o dataset, antes do split.** Decisão deliberada com leakage suave de min/max de 2024–2025 para o treino. Justifica-se porque:

- O alvo é **imputação**, não predição. O leakage afeta apenas a escala, não a separação treino/teste do sinal a aprender.
- Ranges físicos das variáveis são governados por limites biogeoquímicos, não por época da coleta.
- A avaliação se dá via **máscara artificial** sobre células `M=1` (Etapa 6), não via hold-out de valores extremos.

### O que foi tocado

| Grupo | Colunas | Tratamento |
|---|---|---|
| Numéricas (13) + escalar contínuo | DBO, OD, Nitrato, NH₃, P-Total, Condutividade, pH, Turbidez, Temp. Água, SST, Coliformes, Cianobactérias, Microcistinas, `dias_desde_inicio` | `MinMaxScaler(feature_range=(-1, 1))` por coluna, persistido em `scalers.pkl`. |
| Determinístico | `ano_norm` | `2x − 1` (fórmula fechada, sem scaler) — inversão registrada em `transform_params.json["ano_norm"]`. |
| Bypass — escala já adequada | `Mes_sin`, `Mes_cos`, `umido`, 8 `est_*`, 18 `<Var>_LD_*` | intactas |
| Bypass — fora do vetor da GAIN | `Data`, `Ano_int` | intactas |

### Resultado

- **15 colunas escaladas** dentro de `[-1, 1]` com NaN preservado em todas.
- **27 colunas bypass confirmadas idênticas** ao input (proteção contra modificação acidental).
- **Reversibilidade verificada**: erro máximo `0,000` (saturação numérica do `float64`) em todas as 14 colunas + `ano_norm` — muito abaixo do critério `< 10⁻⁶`.

### Alerta automático auditado

Critério: `p95(|x_norm|) < 0,5` indica outlier extremo comprimindo o grosso da distribuição.

- **Fósforo Total** com p95 = **0,405**. Coerente com a kurtosis residual = 8,9 já flagada na Etapa 1 — eventos de eutrofização genuínos. Atenção propagada para `04_GAIN/03_diagnostico.ipynb`.
- Nenhuma outra variável dispara o alerta.

### Atualização do `transform_params.json`

Adicionada a chave `ano_norm: {tipo: "linear", a: 2, b: -1}` para que a Etapa 6 saiba inverter na sequência correta (scaler → ano_norm → Box-Cox/Yeo-Johnson).

---

## 5. `05_split.ipynb` — split temporal honesto

**Pergunta:** como separar os dados de forma que a avaliação seja honesta (não vaze informação do futuro para o passado)?

### Decisão de projeto

Split **temporal**, não stratified por estação. Reflete o uso real: GAIN treinada em dados passados e aplicada em dados novos. Stratified manteria a distribuição de cada estação em todos os splits mas vazaria informação do futuro.

| Split | Anos | n | % do total |
|---|---|---:|---:|
| **train** | 2012–2022 | **533** | 81,1% |
| **val** | 2023 | **34** | 5,2% |
| **test** | 2024–2025 | **90** | 13,7% |
| **total** | | **657** | 100% |

Verificações:

- Anos disjuntos entre splits — vazamento temporal descartado.
- Soma das partes igual ao total do dataset normalizado.
- `ano_norm` e `dias_desde_inicio` permanecem em `[-1, 1]` em todos os splits (porque o scaler foi ajustado globalmente em `04`).

### Gaps de cobertura — decisão consciente, não bug

Algumas variáveis e estações ficam ausentes em val/test por terem sido descontinuadas ou intermitentes. **Não invalida** o protocolo da GAIN (avaliação via máscara artificial sobre `M=1`), mas é registrado explicitamente em `split_info.json["gaps_conscientes"]`.

**Variáveis ausentes:**

| Variável | train | val | test | Motivo |
|---|---:|---:|---:|---|
| Cianobactérias | 133 | 0 | 0 | Medida só em 2012 e 2016 |
| Microcistinas | 24 | 28 | 0 | Medida só em 2019 e 2023 |
| Coliformes Termotolerantes | 492 | 2 | 0 | Descontinuada em 2023 |
| Sólidos Suspensos Totais | 126 | 15 | 16 | Retomada em 2023–2025 (presente nos 3, concentrada) |

**Estações ausentes:**

| Estação | train | val | test | Motivo |
|---|---:|---:|---:|---|
| JC341 | 10 | 0 | 0 | Descontinuada após 2015 |
| MR363 | 8 | 0 | 0 | Descontinuada após 2015 |
| TJ306 | 66 | 2 | 0 | Sem coletas em 2024–2025 |

### Test set carrega o stress real

Test = 2024–2025, anos que tiveram **mudança de layout no INEA** (renomearam LD → Status; 2025 usa códigos 0/1/2 em vez de `<`/`>`). O test set é, portanto, também um teste de robustez do pipeline ponta a ponta — não apenas do modelo.

---

## 6. `06_mascaras.ipynb` — protocolo de mascaramento

**Pergunta:** como o modelo vai saber o que é "faltante real" e o que é "para imputar artificialmente"?

### Decisões de projeto

**1. M e B operam apenas sobre as 13 variáveis numéricas.** Não faz sentido imputar:

- One-hot de estação (`est_*`) — o modelo não inventa estação de coleta.
- One-hot de censura (`<Var>_LD_*`) — censura é estrutural.
- Features temporais — por construção, sem NaN.
- Identificadores — metadados.

**2. Taxa de máscara artificial por variável, não global.** A cobertura real do dataset é fortemente desigual; aplicar `0,2` uniforme em variáveis críticas (Microcistinas com 24 observações no train) apagaria fração desproporcional do pouco sinal disponível.

| Categoria | Variáveis | `miss_rate` |
|---|---|---:|
| **Robustas** (cobertura ≥ 95%) | DBO, OD, pH, Turbidez, Temperatura, Condutividade, Fósforo Total | 0,20 |
| **Intermediárias** (50–90%) | Nitrato, Nitrogênio Amoniacal Total, Coliformes Termotolerantes | 0,10 |
| **Críticas** (< 25%) | SST, Cianobactérias, Microcistinas | 0,05 |

**3. B é gerado a cada batch durante o treino**, não persistido para train/val. Para test/avaliação, chamada com `seed` fixa para reprodutibilidade entre GAIN × baselines.

### Máscara real `M` por split

| Split | Células observadas | % |
|---|---:|---:|
| train | 5 103 / 6 929 | **73,6%** |
| val | 327 / 442 | **74,0%** |
| test | 675 / 1 170 | **57,7%** |

Test cai para 57,7% porque três variáveis (Coliformes, Cianobactérias, Microcistinas) estão totalmente ausentes nele.

### Função `generate_artificial_mask(M, miss_rate, seed)`

**Contrato:**

- `M` pode ser DataFrame (preserva colunas e índice) ou ndarray.
- `miss_rate` pode ser `float` (taxa única) ou `dict {coluna: taxa}` (por variável — requer DataFrame).
- Convenção: onde `M=0` → `B=0` (já faltante, nada a esconder); onde `M=1` → `B ~ Bernoulli(1 − miss_rate)`.
- Reprodutível via `np.random.default_rng(seed)`.

Persistida em `Code/3 - Preprocessing/mask_utils.py` (módulo importável por `04_GAIN/` e `05_Evaluation/`).

### Demonstração (sobre `mask_real_train`, `seed=42`)

| Cenário | Mantidas (M=1, B=1) | Mascaradas (M=1, B=0) | Faltantes orig. (M=0) | % mascaradas / observadas |
|---|---:|---:|---:|---:|
| `miss_rate = 0,20` (uniforme) | 4 035 | 1 068 | 1 826 | 20,9% |
| `MISS_RATES_PADRAO` (por var) | 4 219 | 884 | 1 826 | 17,3% |

A taxa global cai de 20,9% → 17,3% porque variáveis críticas usam `0,05` (e SST contribui com muito poucas observações).

**Taxa efetiva por variável** ficou próxima da alvo (Microcistinas com 8,3% efetiva sobre 5% alvo é variação amostral natural — apenas 24 observações no train).

**Reprodutibilidade:** duas chamadas com `seed=42` produzem `B` idêntico; com `seed=43` divergem. Confirmado.

---

## 7. Decisões propagadas para a Etapa 3 (Baselines)

- **Consumir os mesmos splits** (`Data/GoldData/Splited/`) — GAIN × baselines comparam no mesmo conjunto.
- **Usar a mesma máscara artificial** com `seed` fixa — chamada idêntica a `generate_artificial_mask(M, MISS_RATES_PADRAO, seed=42)` para que as métricas comparem exatamente o mesmo conjunto de células a reconstruir.
- **Avaliar em escala normalizada** (`[-1, 1]`) e **na escala original** (após inversão via `scalers.pkl` + `transform_params.json`) — métricas dependentes de escala (RMSE) e independentes (rank correlation) podem responder diferente.

---

## 8. Decisões propagadas para a Etapa 4 (GAIN)

### Arquitetura confirmada

- **Modelo único, tabular, condicional** (confirmado na Etapa 1 §5).
- **Vetor de entrada:** 13 numéricas + 5 temporais + 8 one-hot estação + 18 one-hot `_LD` = **44 features** (descontando `Data` e `Ano_int`, que ficam para auditoria/split).
- **Saída** com `tanh` para alinhar com o range `[-1, 1]` das features escaladas.

### Pontos de atenção

1. **Fósforo Total** — kurtosis residual = 8,9 (alerta da Etapa 1) e p95(|x_norm|) = 0,405 (alerta de `04_normalizacao`). Verificar explicitamente cobertura da cauda superior no `04_GAIN/03_diagnostico.ipynb`.
2. **Nitrato** — censura à esquerda em 22% (`_LD = <`). Sem tratamento dedicado, o gerador produzirá valores acima do LOD para amostras censuradas. Soluções a avaliar: clip pós-geração, loss customizada, componente Tobit.
3. **Coliformes Termotolerantes** — censura à direita por saturação (`_LD = >` em 28 ocorrências). O gerador deve produzir valores **acima** do reportado, não abaixo (sentido inverso da censura padrão).
4. **JC341 (n=10) e MR363 (n=8)** — descontinuadas após 2015, só em train. Decisão pendente em `04_GAIN/01_arquitetura.ipynb`:
   - **Opção A:** peso reduzido no loss (`weight = n_estacao / max(n_estacoes)`).
   - **Opção B:** consolidar com vizinha geográfica (`JC341 → JC342`, `MR363 → MR361`) antes do treino.
5. **Cianobactérias e Microcistinas** — variáveis críticas (cobertura 20% e 8%). Imputação dessas variáveis será o teste mais difícil; reservar atenção nas métricas de avaliação. Em test, ambas têm 0 cobertura — avaliação só via máscara artificial sobre o treino.
6. **`miss_rate` é calibração inicial** — refinar em `04_GAIN/02_treinamento.ipynb` via early stopping por val loss.

### Sequência obrigatória de inversão (Etapa 6)

Do espaço da GAIN → escala original:

1. **Desnormalizar** via `scalers.pkl` (`scaler.inverse_transform`).
2. **Inverter `ano_norm`** via `transform_params["ano_norm"]` (`(x + 1) / 2`).
3. **Inverter Box-Cox / Yeo-Johnson** via `transform_params.json` (helper `inverter_transformacao` em `01_transformacoes`).

Sair dessa ordem produz valores fora da escala física esperada.

---

## 9. Critério de aceite da Etapa 2 — atendido

Conforme `Pipeline/02_Preprocessing/README.md`:

> **Reversibilidade:** carregar os parquets + scalers + encoders, aplicar `inverse_transform` em toda a pipeline, e reconstruir os valores numéricos originais com erro `< 10⁻⁶`.

Verificações realizadas:

| Etapa | Componente testado | Erro máximo | Critério (1e-6) |
|---|---|---:|---|
| `01` | Box-Cox / Yeo-Johnson inversa em 13 vars | `1,86 × 10⁻⁹` (Coliformes) | OK |
| `04` | `MinMaxScaler.inverse_transform` em 14 vars | `0` (saturação `float64`) | OK |
| `04` | rescale linear de `ano_norm` | `0` | OK |
| `03` | `OneHotEncoder` round-trip (estação + 6 `_LD`) | exato (categórico) | OK |

O round-trip completo (parquet final → escala física original) é, portanto, exato dentro da precisão de ponto flutuante.

---

## 10. Síntese final em uma frase

> Em seis notebooks reproduzíveis e auditáveis, o dataset bruto (657 × 30, com 17 colunas categóricas/objeto e variáveis em escalas díspares) virou um par `(dataset_normalizado.parquet, mask_real_*.parquet)` denso (657 × 46, todas numéricas, todas em `[-1, 1]` quando contínuas, com inversibilidade `< 10⁻⁶`), splitado temporalmente sem vazamento (533 / 34 / 90), com gaps de cobertura explicitamente registrados e protocolo de mascaramento por variável pronto para alimentar a GAIN da Etapa 4.

---

## Anexos

### Artefatos gerados (totais)

- **Parquets:** 5 (`dataset_transformado`, `dataset_com_tempo`, `dataset_encoded`, `dataset_normalizado`) + 3 splits + 3 máscaras = **11 arquivos**.
- **JSONs:** `transform_params.json`, `encoded_columns.json`, `split_info.json`.
- **Pickles:** `encoders.pkl`, `scalers.pkl`.
- **CSV:** `scaling_report.csv`.
- **Módulo Python:** `mask_utils.py`.
- **Figuras:** `transf_antes_depois.png`, `tempo_features_dist.png`, `normalizacao_antes_depois.png`.

### Próximo passo

**Etapa 3 — Baselines.** Começar por `Code/4 - Baselines/01_baselines_simples.ipynb` (KNN, MICE) como referência para a GAIN superar.
