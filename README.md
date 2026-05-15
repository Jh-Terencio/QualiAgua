# QualiAgua

Pipeline de aquisição e tratamento de dados de qualidade da água do **sistema lagunar de Jacarepaguá** (Rio de Janeiro), publicados pelo Instituto Estadual do Ambiente (INEA).

O projeto consolida séries históricas anuais (**2012–2025**) em uma base padronizada que servirá de insumo para um modelo de **Redes Adversárias Generativas (GANs)** voltado à imputação dos valores faltantes do dataset.

> Status: em desenvolvimento. Esta versão cobre as etapas de coleta e tratamento inicial dos dados brutos.

## Sumário

- [Objetivo](#objetivo)
- [Fonte de dados](#fonte-de-dados)
- [Escopo geográfico](#escopo-geográfico)
- [Variáveis de interesse](#variáveis-de-interesse)
- [Limite de Detecção (LD)](#limite-de-detecção-ld)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Pipeline atual](#pipeline-atual)
- [Como executar](#como-executar)
- [Esquema do dataset final](#esquema-do-dataset-final)
- [Dependências](#dependências)

## Objetivo

O INEA disponibiliza uma planilha pública com medições anuais de qualidade da água para centenas de estações de monitoramento no estado do Rio de Janeiro. Por se tratar de coletas pontuais com metodologias variando ao longo dos anos, o dataset apresenta forte esparsidade: várias variáveis de interesse só foram medidas em alguns anos, ou em alguns pontos.

O objetivo deste projeto é:

1. **Coletar** automaticamente a planilha bruta do INEA.
2. **Padronizar** os dados em uma base consolidada que reconcilie as diferenças estruturais entre os anos (renomeações de colunas, mudanças de convenção do limite de detecção, layouts diferentes).
3. Servir de base para um **modelo GAN imputador** que aprenda a preencher os valores faltantes preservando a estrutura estatística dos dados observados.

## Fonte de dados

- **Página origem:** https://www.inea.rj.gov.br/ar-agua-e-solo/rios-baias-lagoas/
- **Formato:** planilha Excel única (`.xlsx`) com uma aba por ano e duas abas auxiliares com metadados das estações.
- **Período coberto:** 2012 a 2025.

A planilha bruta é versionada localmente em `Data/RawData/WaterQualityRawData.xlsx`.

## Escopo geográfico

O projeto se concentra em **8 estações** do sistema lagunar de Jacarepaguá:

| Código resumido | Código completo  | Corpo d'água          |
| --------------- | ---------------- | --------------------- |
| CM320           | 01RJ20CM0320     | Lagoa de Camorim      |
| JC341           | 01RJ20JC0341     | Lagoa de Jacarepaguá  |
| JC342           | 01RJ20JC0342     | Lagoa de Jacarepaguá  |
| MR361           | 01RJ20MR0361     | Lagoa da Marapendi    |
| MR363           | 01RJ20MR0363     | Lagoa da Marapendi    |
| MR369           | 01RJ20MR0369     | Lagoa da Marapendi    |
| TJ303           | 01RJ20TJ0303     | Lagoa da Tijuca       |
| TJ306           | 01RJ20TJ0306     | Lagoa da Tijuca       |

## Variáveis de interesse

Treze variáveis físico-químicas e biológicas são mantidas no dataset final:

| Categoria        | Variáveis                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------ |
| Físico-químicas  | pH, Temperatura da Água, Condutividade, Turbidez, Sólidos Suspensos Totais                                   |
| Oxigênio / OM    | OD (Oxigênio Dissolvido), DBO (Demanda Bioquímica de Oxigênio)                                               |
| Nutrientes       | Nitrato, Nitrogênio Amoniacal Total, Fósforo Total                                                            |
| Microbiologia    | Coliformes Termotolerantes                                                                                    |
| Cianotoxinas     | Cianobactérias (cél/L), Microcistinas (µg/L)                                                                  |

## Limite de Detecção (LD)

O LD representa o menor valor que o método analítico empregado consegue medir com confiabilidade. Valores próximos ao limite são reportados como **dados censurados**:

- `<` — o valor real é **inferior** ao LD (censura à esquerda).
- `>` — o valor real é **superior** ao LD (censura à direita).
- `NaN` — não medido.

A estrutura `<Variável>` + `<Variável>_LD` permite preservar a natureza censurada do dado em vez de descartá-lo. Isso é relevante para a etapa de modelagem (a GAN deve tratar valores censurados de forma distinta de valores ausentes).

### Particularidades por ano

O INEA alterou a convenção do LD ao longo do tempo. O pipeline lida com isso internamente:

| Anos      | Coluna           | Posição          | Convenção                                                        |
| --------- | ---------------- | ---------------- | ---------------------------------------------------------------- |
| 2012–2023 | `LD`             | antes da variável| `<`, `>` ou vazio                                                |
| 2024      | `<Var> Status`   | antes da variável| `<`, `>` ou `NR` (Not Reported)                                  |
| 2025      | `<Var> Status`   | **depois** da variável | `0` = não medido, `1` = medido sem censura, `2` = censurado |

Além disso, a aba de 2025 tem duas linhas de texto descritivo no topo (cabeçalho real fica na linha 3), e algumas colunas foram renomeadas (ex.: `Nitrato` → `Nitratos`, `Temperatura da Água` → `TempAmostra`, `Fósforo Total` → `FosforoTotal`). O loader reconcilia essas diferenças via dicionário de sinônimos canônicos.

Para manter compatibilidade com o restante do pipeline, todos os anos são unificados na convenção legada (`<` / `>` / `NaN`). O código `2` do 2025 é mapeado para `<` por ser a forma de censura dominante em qualidade de água.

## Estrutura do repositório

```
QualiAgua/
├── Articles/                              # Artigos e referências (PIBIC, ICCSA, LD)
├── Code/
│   └── FetchAndTreatRawData/
│       ├── data_loader.ipynb              # Baixa a planilha bruta do INEA
│       └── treat_raw_data.ipynb           # Trata, padroniza e consolida os anos
├── Data/
│   ├── RawData/
│   │   └── WaterQualityRawData.xlsx       # Saída do data_loader
│   └── IntermediaryData/
│       └── WaterQualityInitialData.xlsx   # Saída do treat_raw_data
├── requirements.txt                       # Dependências Python
├── .gitignore
└── README.md
```

## Pipeline atual

O fluxo atual é composto por dois notebooks executados em sequência:

### 1. `data_loader.ipynb` — Coleta

Faz scraping leve da página do INEA, localiza o link cujo texto contém "Dados Brutos" e baixa o `.xlsx` para `Data/RawData/WaterQualityRawData.xlsx`.

Bibliotecas: `requests`, `beautifulsoup4`.

### 2. `treat_raw_data.ipynb` — Tratamento

Lê a planilha bruta e produz o dataset consolidado. As principais etapas são:

1. **Carregamento das estações** — unifica as duas abas de metadados (`Estações 2012 a 2024` e `Estações 2025`) em uma única tabela de referência.
2. **Loader por ano (`load_year`)** — aplica configuração específica de cada ano (linha do cabeçalho, posição da coluna de LD, convenção de status, nomes de coluna para `Local` e `Data`).
3. **Padronização de nomenclatura** — normaliza nomes de coluna (remoção de acentos, unidades entre parênteses, espaços extras, caixa de texto) via função `canonical_name`. Sinônimos canônicos por variável reconciliam diferenças entre anos.
4. **Acoplamento variável ↔ LD** — para cada variável encontrada, a coluna de LD/Status é localizada por **adjacência posicional** (à esquerda no legado, à direita no 2025).
5. **Normalização do LD (`normalize_status`)** — converte todas as convenções para `<`, `>` ou `NaN`.
6. **Desambiguação Cianobactérias × Microcistinas** — feita pela unidade no nome bruto da coluna (`cél/L` → Cianobactérias, `µg/L` ou sem unidade → Microcistinas), já que `canonical_name` remove unidades entre parênteses.
7. **Tratamento do placeholder `0.0` no 2025** — quando `Status == 0`, a planilha 2025 traz `0.0` na coluna da variável; o loader zera esses valores para `NaN`.
8. **Consolidação** — concatena todos os anos, faz `merge` com as estações para enriquecer `Codigo Local` e `Local` (corpo d'água), reordena colunas e exporta.

## Como executar

```bash
# 1. Clonar o repositório
git clone <repo-url>
cd QualiAgua

# 2. Criar e ativar virtual env
python -m venv venv
# Windows
venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt
```

Depois, abrir os notebooks em ordem:

1. `Code/FetchAndTreatRawData/data_loader.ipynb` — executa o download da planilha bruta.
2. `Code/FetchAndTreatRawData/treat_raw_data.ipynb` — gera `Data/IntermediaryData/WaterQualityInitialData.xlsx`.

## Esquema do dataset final

`Data/IntermediaryData/WaterQualityInitialData.xlsx` — 657 registros × 30 colunas.

| Coluna                          | Tipo       | Descrição                                                  |
| ------------------------------- | ---------- | ---------------------------------------------------------- |
| `Local`                         | string     | Nome do corpo d'água (ex.: "Lagoa de Camorim")             |
| `Codigo Local`                  | string     | Código resumido da estação (ex.: "CM320")                  |
| `Data`                          | datetime   | Data da coleta                                             |
| `Ano`                           | string     | Ano da medição                                             |
| `<Variável>_LD`                 | string     | Marcador de censura: `<`, `>` ou `NaN`                     |
| `<Variável>`                    | float      | Valor numérico medido                                      |

Cada uma das 13 variáveis de interesse aparece como par `<Variável>_LD` / `<Variável>`.

## Dependências

As principais bibliotecas utilizadas (lista completa em `requirements.txt`):

| Pacote          | Uso                                                    |
| --------------- | ------------------------------------------------------ |
| `pandas`        | Manipulação tabular                                    |
| `openpyxl`      | Leitura/escrita de `.xlsx`                             |
| `requests`      | Download da planilha bruta                             |
| `beautifulsoup4`| Parsing do HTML da página do INEA                      |
| `numpy`         | Suporte numérico                                       |
| `matplotlib`, `seaborn` | Visualizações (etapas futuras)                 |
| `scipy`         | Estatística (etapas futuras)                           |
| `ipykernel`     | Execução dos notebooks                                 |
