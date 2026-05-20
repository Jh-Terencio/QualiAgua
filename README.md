# QualiAgua

Pipeline de aquisição, tratamento e **imputação por GAN** de dados de qualidade da água do **sistema lagunar de Jacarepaguá** (Rio de Janeiro), publicados pelo Instituto Estadual do Ambiente (INEA).

O projeto consolida séries históricas anuais (**2012–2025**) em uma base padronizada que alimenta uma **Rede Adversária Generativa Imputadora (GAIN)** voltada ao preenchimento dos valores faltantes do dataset, preservando a estrutura estatística dos dados observados.

> **Status (2026-05-19):** Etapas 1 (Coleta/Tratamento), 2 (EDA) e 3 (Pré-processamento) concluídas. Etapa 4 (Baselines) a iniciar.

## Sumário

- [Objetivo](#objetivo)
- [Estado atual](#estado-atual)
- [Fonte de dados](#fonte-de-dados)
- [Escopo geográfico](#escopo-geográfico)
- [Variáveis de interesse](#variáveis-de-interesse)
- [Limite de Detecção (LD)](#limite-de-detecção-ld)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Pipeline](#pipeline)
- [Como executar](#como-executar)
- [Esquema dos datasets](#esquema-dos-datasets)
- [Dependências](#dependências)

## Objetivo

O INEA disponibiliza uma planilha pública com medições anuais de qualidade da água para centenas de estações de monitoramento no estado do Rio de Janeiro. Por se tratar de coletas pontuais com metodologias variando ao longo dos anos, o dataset apresenta forte esparsidade: várias variáveis de interesse só foram medidas em alguns anos ou em alguns pontos. **Zero linhas têm as 13 variáveis simultaneamente preenchidas** — imputação é obrigatória para qualquer análise multivariada.

O objetivo do projeto é:

1. **Coletar** automaticamente a planilha bruta do INEA.
2. **Padronizar** os dados em uma base consolidada que reconcilie as diferenças estruturais entre os anos (renomeações, mudanças de convenção do LD, layouts diferentes em 2024/2025).
3. **Analisar** a estrutura de cobertura, distribuições, correlações, sazonalidade, estrutura espacial e censura por LD.
4. **Pré-processar** o dataset para um formato denso, padronizado e auditável consumível pela GAIN.
5. **Treinar e avaliar** um modelo GAIN que supere baselines clássicos (KNN, MICE) na imputação.
6. **Publicar** o dataset imputado e o IQA da Lagoa de Jacarepaguá ao longo de 2012–2025.

## Estado atual

| Etapa | Pasta | Status | Saídas principais |
|---|---|---|---|
| 1 — Coleta + Tratamento | `Code/1 - FetchAndTreatRawData/` | ✅ Concluída | `Data/IntermediaryData/WaterQualityInitialData.xlsx` (657 × 30) |
| 2 — EDA | `Code/2 - EDA/` | ✅ Concluída — ver `Code/2 - EDA/resumo.md` | Tabelas em `Data/Figures/01_EDA/tabelas/`, ~20 figuras em `Data/Figures/01_EDA/figuras/` |
| 3 — Pré-processamento | `Code/3 - Preprocessing/` | ✅ Concluída — ver `Code/3 - Preprocessing/resumo.md` | `dataset_normalizado.parquet`, `train/val/test.parquet`, máscaras reais, `scalers.pkl`, `encoders.pkl`, `transform_params.json`, `mask_utils.py` |
| 4 — Baselines | `Code/4 - Baselines/` (a criar) | ⏳ Próxima | KNN, MICE como referência |
| 5 — GAIN | `Code/5 - GAIN/` (a criar) | ⏳ | Modelo principal |
| 6 — Avaliação | `Code/6 - Evaluation/` (a criar) | ⏳ | Comparação GAIN × baselines |
| 7 — Pós-imputação | `Code/7 - PostImputation/` (a criar) | ⏳ | IQA + análises temporais sobre o dataset imputado |

Planos detalhados por notebook ficam em `Pipeline/0X_<Nome>/*.md` — **planejamento**, separado do código que implementa.

## Fonte de dados

- **Página origem:** https://www.inea.rj.gov.br/ar-agua-e-solo/rios-baias-lagoas/
- **Formato:** planilha Excel única (`.xlsx`) com uma aba por ano e duas abas auxiliares com metadados das estações.
- **Período coberto:** 2012 a 2025 (14 anos).

A planilha bruta é versionada localmente em `Data/RawData/WaterQualityRawData.xlsx`.

## Escopo geográfico

O projeto se concentra em **8 estações** do sistema lagunar de Jacarepaguá:

| Código resumido | Código completo  | Corpo d'água          | n coletas |
| --------------- | ---------------- | --------------------- | --------: |
| CM320           | 01RJ20CM0320     | Lagoa de Camorim      | 114 |
| JC341           | 01RJ20JC0341     | Lagoa de Jacarepaguá  | 10 (descontinuada após 2015) |
| JC342           | 01RJ20JC0342     | Lagoa de Jacarepaguá  | 114 |
| MR361           | 01RJ20MR0361     | Lagoa da Marapendi    | 115 |
| MR363           | 01RJ20MR0363     | Lagoa da Marapendi    | 8 (descontinuada após 2015) |
| MR369           | 01RJ20MR0369     | Lagoa da Marapendi    | 115 |
| TJ303           | 01RJ20TJ0303     | Lagoa da Tijuca       | 113 |
| TJ306           | 01RJ20TJ0306     | Lagoa da Tijuca       | 68 (sem coletas em 2024–2025) |

A EDA identificou **K=2 clusters naturais**: marinho (MR369, TJ303) vs continental (demais). A GAIN é única e usa `Codigo Local` como feature condicional one-hot — fragmentar por cluster reduziria amostra abaixo do mínimo viável.

## Variáveis de interesse

Treze variáveis físico-químicas e biológicas, classificadas pela cobertura observada:

| Cobertura | Variáveis | Tratamento na GAIN |
|---|---|---|
| **Robustas (≥ 95%)** | DBO, OD, Condutividade, pH, Turbidez, Temperatura da Água, Fósforo Total | Âncoras de imputação; `miss_rate = 0,20` na máscara artificial |
| **Intermediárias (50–80%)** | Nitrogênio Amoniacal Total, Coliformes Termotolerantes, Nitrato | `miss_rate = 0,10` |
| **Críticas (< 25%)** | Sólidos Suspensos Totais (24%), Cianobactérias (20%), Microcistinas (8%) | `miss_rate = 0,05`; imputação com incerteza alta |

## Limite de Detecção (LD)

O LD representa o menor valor que o método analítico empregado consegue medir com confiabilidade. Valores próximos ao limite são reportados como **dados censurados** — distintos de `NaN`:

- `<` — o valor real é **inferior** ao LD (censura à esquerda).
- `>` — o valor real é **superior** ao LD (censura à direita).
- `NaN` — não medido.

A EDA (`Code/2 - EDA/06_lds.ipynb`) classificou cada variável quanto à relevância da censura. **Apenas 6 variáveis têm censura ≥ 2% e recebem feature one-hot `_LD`** na GAIN: Nitrato, DBO, Nitrogênio Amoniacal Total, Coliformes Termotolerantes, Cianobactérias, Microcistinas. As outras 7 têm a coluna `_LD` descartada (geraria one-hot constante).

**Coliformes Termotolerantes** é a única variável com censura predominante à **direita** (`>` = 28 vs `<` = 2) — reflete o teto reportável do método de tubos múltiplos. Todas as demais têm `<` dominante.

### Particularidades por ano

O INEA alterou a convenção do LD ao longo do tempo. O loader em `Code/1 - FetchAndTreatRawData/` reconcilia internamente:

| Anos      | Coluna           | Posição          | Convenção                                                        |
| --------- | ---------------- | ---------------- | ---------------------------------------------------------------- |
| 2012–2023 | `LD`             | antes da variável| `<`, `>` ou vazio                                                |
| 2024      | `<Var> Status`   | antes da variável| `<`, `>` ou `NR` (Not Reported)                                  |
| 2025      | `<Var> Status`   | **depois** da variável | `0` = não medido, `1` = medido sem censura, `2` = censurado |

A aba de 2025 tem duas linhas de texto descritivo no topo (cabeçalho real fica na linha 3), e algumas colunas foram renomeadas (`Nitrato` → `Nitratos`, `Temperatura da Água` → `TempAmostra`, `Fósforo Total` → `FosforoTotal`). O loader é year-aware via dicionário `YEAR_CONFIG` e sinônimos canônicos.

Para manter compatibilidade com o restante do pipeline, todos os anos são unificados na convenção legada (`<` / `>` / `NaN`). O código `2` do 2025 é mapeado para `<` por ser a forma dominante.

## Estrutura do repositório

```
QualiAgua/
├── Articles/                              # Artigos e referências (PIBIC, ICCSA, LD)
├── Code/
│   ├── 1 - FetchAndTreatRawData/
│   │   ├── data_loader.ipynb              # Baixa a planilha bruta do INEA
│   │   └── treat_raw_data.ipynb           # Trata, padroniza e consolida os anos
│   ├── 2 - EDA/                           # ✅ 6 notebooks (descritivo, distribuições,
│   │                                      #    correlações, temporal, estações, LDs)
│   │   └── resumo.md                      # Decisões consolidadas da EDA
│   └── 3 - Preprocessing/                 # ✅ 6 notebooks + mask_utils.py
│       ├── 01_transformacoes.ipynb        # Box-Cox / Yeo-Johnson com λ reusados
│       ├── 02_features_temporais.ipynb    # Ano_int, ano_norm, Mes_sin/cos, umido, ...
│       ├── 03_encoding.ipynb              # one-hot estação + one-hot 6 _LD
│       ├── 04_normalizacao.ipynb          # MinMaxScaler [-1, 1]
│       ├── 05_split.ipynb                 # Split temporal train/val/test
│       ├── 06_mascaras.ipynb              # M real + generate_artificial_mask
│       ├── mask_utils.py                  # Módulo importável
│       └── resumo.md                      # Decisões consolidadas
├── Pipeline/                              # Planos por etapa (markdown, separado do código)
│   ├── 01_EDA/
│   ├── 02_Preprocessing/
│   ├── 03_Baselines/
│   ├── 04_GAIN/
│   ├── 05_Evaluation/
│   ├── 06_PostImputation/
│   └── README.md
├── Data/
│   ├── RawData/
│   │   └── WaterQualityRawData.xlsx       # Saída do data_loader
│   ├── IntermediaryData/
│   │   └── WaterQualityInitialData.xlsx   # Saída do treat_raw_data (657 × 30)
│   ├── ProcessedData/                     # Saídas intermediárias do Pré-processamento
│   │   ├── dataset_transformado.parquet
│   │   ├── dataset_com_tempo.parquet
│   │   ├── dataset_encoded.parquet
│   │   ├── dataset_normalizado.parquet    # ← Entrada da Etapa 4 (Baselines / GAIN)
│   │   ├── transform_params.json
│   │   ├── encoded_columns.json
│   │   ├── encoders.pkl
│   │   ├── scalers.pkl
│   │   └── scaling_report.csv
│   ├── GoldData/                          # Splits finais (consumo direto pelas etapas seguintes)
│   │   ├── Splited/
│   │   │   ├── train.parquet              # 533 amostras, anos 2012–2022
│   │   │   ├── val.parquet                # 34 amostras, ano 2023
│   │   │   ├── test.parquet               # 90 amostras, anos 2024–2025
│   │   │   └── split_info.json
│   │   └── Masked/
│   │       ├── mask_real_train.parquet
│   │       ├── mask_real_val.parquet
│   │       └── mask_real_test.parquet
│   └── Figures/                           # Saídas gráficas + tabelas auxiliares
│       ├── 01_EDA/
│       └── 02_Preprocessing/
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

### Etapa 1 — Coleta e tratamento (`Code/1 - FetchAndTreatRawData/`)

Dois notebooks executados em sequência:

#### `data_loader.ipynb`

Faz scraping leve da página do INEA, localiza o link cujo texto contém "Dados Brutos" e baixa o `.xlsx` para `Data/RawData/WaterQualityRawData.xlsx`. Bibliotecas: `requests`, `beautifulsoup4`.

#### `treat_raw_data.ipynb`

Consolida os anos em uma única tabela com colunas padronizadas (par `<Variável>` / `<Variável>_LD` para cada uma das 13 variáveis). Etapas principais:

1. Loader por ano com configuração específica (`YEAR_CONFIG`).
2. Padronização de nomenclatura (acentos, unidades, sinônimos canônicos).
3. Acoplamento variável ↔ LD por adjacência posicional.
4. Normalização da convenção do LD (`<` / `>` / `NaN`).
5. Desambiguação Cianobactérias × Microcistinas pela unidade no nome bruto.
6. Tratamento do placeholder `0.0` no 2025 (mapeia para `NaN` quando `Status == 0`).
7. Merge com metadados das estações.

**Saída:** `Data/IntermediaryData/WaterQualityInitialData.xlsx` (657 × 30).

### Etapa 2 — EDA (`Code/2 - EDA/`)

Seis notebooks (`01_descritivo` → `06_lds`) que respondem às perguntas estruturais antes da modelagem:

1. Cobertura e estrutura de ausência.
2. Forma das distribuições e transformações sugeridas (λ Box-Cox/Yeo-Johnson persistidos).
3. Estrutura de dependência (correlações Spearman).
4. Padrão temporal (sazonalidade, tendência, irregularidade).
5. Estrutura espacial (PCA, clusters, decisão GAIN única vs múltipla).
6. Censura por LD (quais variáveis recebem feature `_LD`).

**Síntese:** `Code/2 - EDA/resumo.md`.

### Etapa 3 — Pré-processamento (`Code/3 - Preprocessing/`)

Seis notebooks que convertem o dataset bruto (657 × 30) em um par `(dataset_normalizado.parquet, mask_real_*.parquet)` denso (657 × 46, todas numéricas, todas em `[-1, 1]` quando contínuas), splitado temporalmente sem vazamento:

- **01_transformacoes** — Box-Cox (7 vars) + Yeo-Johnson (3 vars) + identidade (3 vars), reusando λ da EDA.
- **02_features_temporais** — `Ano_int`, `ano_norm`, `Mes_sin/cos`, `umido` (binária nov–mar), `dias_desde_inicio`.
- **03_encoding** — one-hot de estação (8 cols) + one-hot de `<Var>_LD` em 6 variáveis (18 cols).
- **04_normalizacao** — `MinMaxScaler([-1, 1])` em 14 colunas; `ano_norm` por rescale linear `2x − 1`. Reversibilidade verificada `< 1e-6`.
- **05_split** — train (≤ 2022, n=533) / val (2023, n=34) / test (≥ 2024, n=90). Gaps de cobertura registrados.
- **06_mascaras** — máscara real `M` por split + `generate_artificial_mask(M, miss_rate_por_var, seed)` em `mask_utils.py`.

**Síntese:** `Code/3 - Preprocessing/resumo.md`.

### Etapas 4–7 — em scaffolding

Ver `Pipeline/0{3..6}_*/README.md` para o detalhamento dos notebooks planejados (Baselines KNN/MICE → GAIN → Avaliação → Pós-imputação com IQA).

## Como executar

```bash
# 1. Clonar o repositório
git clone <repo-url>
cd QualiAgua

# 2. Criar e ativar virtual env
python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt
```

Ordem de execução dos notebooks:

1. `Code/1 - FetchAndTreatRawData/data_loader.ipynb` — download da planilha bruta.
2. `Code/1 - FetchAndTreatRawData/treat_raw_data.ipynb` — consolidação inicial.
3. `Code/2 - EDA/01_descritivo.ipynb` → `06_lds.ipynb` — análise exploratória.
4. `Code/3 - Preprocessing/01_transformacoes.ipynb` → `06_mascaras.ipynb` — preparação para a GAIN.
5. *(em scaffolding)* `Code/4 - Baselines/`, `Code/5 - GAIN/`, `Code/6 - Evaluation/`, `Code/7 - PostImputation/`.

Cada notebook pode ser executado isoladamente — todos consomem artefatos persistidos pelos anteriores e produzem novos artefatos rastreáveis.

## Esquema dos datasets

### `Data/IntermediaryData/WaterQualityInitialData.xlsx` — bruto consolidado (657 × 30)

| Coluna                          | Tipo       | Descrição                                                  |
| ------------------------------- | ---------- | ---------------------------------------------------------- |
| `Local`                         | string     | Nome do corpo d'água (ex.: "Lagoa de Camorim")             |
| `Codigo Local`                  | string     | Código resumido da estação (ex.: "CM320")                  |
| `Data`                          | datetime   | Data da coleta                                             |
| `Ano`                           | string     | Ano da medição                                             |
| `<Variável>_LD`                 | string     | Marcador de censura: `<`, `>` ou `NaN`                     |
| `<Variável>`                    | float      | Valor numérico medido                                      |

Cada uma das 13 variáveis de interesse aparece como par `<Variável>_LD` / `<Variável>`.

### `Data/ProcessedData/dataset_normalizado.parquet` — entrada da GAIN (657 × 46)

| Grupo de colunas | Quantidade | Forma |
|---|---:|---|
| Identificadores (`Data`, `Ano_int`) | 2 | datetime / int (auditoria; fora do vetor da GAIN) |
| Variáveis numéricas transformadas e normalizadas | 13 | float em `[-1, 1]` (com NaN) |
| Features temporais | 5 | `ano_norm`, `Mes_sin/cos`, `umido`, `dias_desde_inicio` — todas em `[-1, 1]` |
| One-hot de estação | 8 | `est_<codigo>`, `{0, 1}` |
| One-hot de censura `_LD` | 18 | 6 vars × 3 níveis (`_normal`, `_lt`, `_gt`), `{0, 1}` |

A ordem exata das colunas está em `Data/ProcessedData/encoded_columns.json`.

### `Data/GoldData/Splited/{train,val,test}.parquet`

Mesmas colunas do `dataset_normalizado.parquet`, filtradas por ano:

| Split | Anos | n |
|---|---|---:|
| train | 2012–2022 | 533 |
| val | 2023 | 34 |
| test | 2024–2025 | 90 |

### `Data/GoldData/Masked/mask_real_{train,val,test}.parquet`

DataFrames `int8` com **apenas as 13 variáveis numéricas** (mesma ordem de linhas dos splits correspondentes): `1` = observado, `0` = faltante.

## Dependências

As principais bibliotecas utilizadas (lista completa em `requirements.txt`):

| Pacote | Uso |
| --- | --- |
| `pandas`, `numpy` | Manipulação tabular e numérica |
| `openpyxl`, `pyarrow` | Leitura/escrita `.xlsx` e `.parquet` |
| `requests`, `beautifulsoup4` | Download e parsing da página do INEA |
| `scipy` | Box-Cox, Yeo-Johnson, testes de hipótese (Kruskal-Wallis, Spearman) |
| `scikit-learn` ≥ 1.0 | `MinMaxScaler`, `OneHotEncoder`, PCA, KMeans |
| `matplotlib`, `seaborn` | Visualizações |
| `joblib` | Persistência de scalers e encoders |
| `ipykernel` | Execução dos notebooks |
