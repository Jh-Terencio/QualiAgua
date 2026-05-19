# Resumo da Etapa 1 — EDA

**Projeto:** QualiAgua — pipeline de imputação por GAN para qualidade da água do sistema lagunar de Jacarepaguá (INEA, 2012–2025).

**Escopo:** 657 coletas × 13 variáveis físico-químicas e biológicas × 8 estações (CM320, JC341, JC342, MR361, MR363, MR369, TJ303, TJ306) ao longo de 14 anos.

**Objetivo deste documento:** consolidar os achados dos seis notebooks da EDA (`01_descritivo` → `06_lds`) em um relatório objetivo, com as decisões que serão propagadas para a Etapa 2 (Pré-processamento) e para o desenho da GAIN (Etapa 4).

---

## 1. Cobertura e estrutura de ausência (`01_descritivo`)

**Pergunta:** quanto dado existe por variável, estação e ano? Onde estão os buracos?

### Variáveis por nível de cobertura

| Faixa | Variáveis | Implicação |
|---|---|---|
| **Robustas (≥ 95%)** | DBO (98%), OD (97%), Condutividade (97%), pH (97%), Turbidez (96%), Temperatura da Água (96%), Fósforo Total (95%) | Formam o núcleo de treino; podem servir de "âncoras" para imputar as demais. |
| **Intermediárias (50–80%)** | Nitrogênio Amoniacal Total (76%), Coliformes Termotolerantes (75%), Nitrato (51%) | Imputação plenamente viável; cobertura suficiente para validação. |
| **Críticas (< 25%)** | Sólidos Suspensos Totais (24%), Cianobactérias (20%), Microcistinas (8%) | Imputação obrigatória, mas com incerteza alta. Validação dependerá da consistência inter-variáveis, não de hold-out direto. |

### Lacunas estruturais identificadas

- **2020** foi o ano de monitoramento mais reduzido (apenas 1 coleta por estação ativa) — efeito direto da pandemia.
- **JC341 (n=10)** e **MR363 (n=8)** foram descontinuadas após 2015. Representatividade limitada.
- **TJ306** sem coletas em 2024–2025; total de 68 coletas vs ~114 das demais estações ativas.
- **Sólidos Suspensos Totais** ausentes de 2014 a 2022 (lacuna de ~9 anos).
- **Coliformes Termotolerantes** descontinuados a partir de 2023 (zero em 2024–2025).
- **Cianobactérias** medidas apenas em 2012 e 2016.
- **Microcistinas** medidas apenas em 2019 e 2023.

### Sample size efetivo

- **Zero linhas completas** nas 13 variáveis — imputação é obrigatória para qualquer análise multivariada.
- 644 linhas (98% do total) têm ≥ 4 variáveis preenchidas — base sólida para baselines clássicos (MICE).
- Queda brusca entre k=9 (517 linhas) e k=10 (271 linhas), refletindo a coexistência rara de Sólidos Suspensos, Cianobactérias e Microcistinas com as demais.

---

## 2. Forma das distribuições e transformações (`02_distribuicoes`)

**Pergunta:** quais variáveis são simétricas e quais precisam de transformação antes da GAIN?

### Decisões de transformação

| Tratamento | Variáveis | λ (Box-Cox) | Resultado |
|---|---|---|---|
| **Identidade** (skew < 1) | pH, Temperatura da Água, Microcistinas | — | Sem transformação. |
| **Yeo-Johnson** (têm zeros) | DBO, OD, Nitrogênio Amoniacal Total | 0,08 / 0,28 / −0,08 | Skew → ~0. |
| **Box-Cox** | Nitrato, Fósforo Total, Condutividade, Turbidez, Sólidos Suspensos Totais, Coliformes Termotolerantes, Cianobactérias | −0,39 a 0,38 | Skew → ~0 em todos exceto Fósforo Total. |

### Caso problemático: Fósforo Total

- Skew original = 13,4; kurtosis = 182.
- Box-Cox (λ = −0,13) reduz skew para −0,40, **mas kurtosis residual permanece em 8,9**.
- Os picos extremos são **eventos genuínos de eutrofização**, não outliers a serem suprimidos.
- **Alerta para a GAIN:** o gerador pode subestimar esses picos; verificar cobertura da cauda em `04_GAIN/03_diagnostico.ipynb`.

### Observação técnica

- Turbidez (λ ≈ 0,03) e Coliformes Termotolerantes (λ ≈ 0,07) → equivalentes a `log` simples; útil quando interpretabilidade for prioridade.
- Microcistinas (n=52) com leve sugestão de bimodalidade (ausência vs floração) que o n não permite confirmar.

---

## 3. Estrutura de dependência (`03_correlacoes`)

**Pergunta:** quais pares carregam mais informação mútua? A estrutura é estável entre estações?

### Cinco pares fortes (|ρ_Spearman| > 0,5, p < 0,05)

| Par | ρ Spearman | Interpretação |
|---|---|---|
| Condutividade × Turbidez | **−0,60** | Gradiente marinho→continental: estações marinhas (TJ303, TJ306) têm Condutividade alta e Turbidez baixa. |
| DBO × Turbidez | **+0,56** | Carga orgânica acompanha material em suspensão. |
| OD × pH | **+0,55** | Fotossíntese eleva ambos simultaneamente. **Único par estável em todas as 8 estações.** |
| DBO × Condutividade | **−0,51** | Eutrofização vs salinidade. |
| Coliformes × Cianobactérias | **+0,50** (n=111) | Contaminação orgânica + nutrientes. |

### Observações relevantes

- **Spearman > Pearson** em vários pares (diferença até 0,43 em Nitrogênio Amoniacal × Fósforo Total) — relações monotônicas **não-lineares** que Pearson subestima. Spearman será a métrica de referência daqui em diante.
- **Nenhum par ultrapassa |ρ| > 0,7** — não há redundância informacional extrema. A GAIN precisará explorar **combinações** de variáveis, não vizinhos diretos.
- **Heterogeneidade entre estações é alta:** TJ303 tem **27 pares fortes**, MR369 tem apenas **4**. DBO × Turbidez chega a inverter sinal entre estações (−0,54 em JC341 vs +0,93 em MR363, ambas com n pequeno).

### Núcleo informacional para imputação

**DBO, OD, pH, Turbidez, Condutividade** — variáveis com cobertura > 620 e que aparecem em todos os top-pares. Servirão de "âncoras" para imputar as variáveis de cobertura crítica.

---

## 4. Padrão temporal (`04_temporal`)

**Pergunta:** há sazonalidade e tendência exploráveis? Modelo sequencial ou tabular?

### Esforço amostral irregular

- Coleta nominalmente **mensal** (gap mediano 36–48 dias por estação).
- ~20–30% dos gaps são > 60 dias; gap máximo de 365–427 dias em todas as estações (refletindo o vazio de 2020).
- **Implicação direta:** modelos sequenciais puros (LSTM, TimeGAN) **estão descartados** — a irregularidade falsifica o pressuposto de continuidade temporal.

### Sazonalidade (Kruskal-Wallis por mês)

- **Detectada em 11 das 13 variáveis** (p < 0,05).
- **Sinal mais forte:** Temperatura da Água (H = 353; ACF[6] = −0,54 e ACF[12] = +0,43 — assinatura senoidal anual perfeita).
- **Sem sazonalidade detectável:** Turbidez (p = 0,067) e Coliformes Termotolerantes (p = 0,22) — dominadas por **eventos pontuais** (chuvas, contaminação) e não pelo calendário.

### Tendências de longo prazo

- Significativas estatisticamente pelo n grande, mas **magnitude pequena** (|ρ| < 0,27 em todas as variáveis).
- Destaques: **Cianobactérias em queda** (ρ = −0,25) e **OD em alta** (ρ = +0,14).
- O sistema é **razoavelmente estacionário** no horizonte 2012–2025 — não há mudança de regime visível.

### Features temporais para a GAIN

| Feature | Fórmula | Justificativa |
|---|---|---|
| `mes_sin`, `mes_cos` | `sin/cos(2π · Mes / 12)` | Codificação cíclica do mês (evita salto dez→jan). |
| `estacao_do_ano` | binária: úmido (nov–mar) vs seco (abr–out) | Regime climático carioca. |
| `ano_norm` | `(Ano − 2012) / 13` | Captura tendência linear (mesmo que fraca). |
| `dias_desde_inicio` | `(Data − 2012-01-01).days` | Alternativa contínua, granularidade fina. |

### Decisão arquitetural

**GAIN tabular com features temporais derivadas** — confirmado. A irregularidade da amostragem inviabiliza sequencial; features cíclicas + binária estação capturam o sinal sazonal de 11 das 13 variáveis sem exigir continuidade.

---

## 5. Estrutura espacial (`05_estacoes`)

**Pergunta:** as 8 estações se comportam como sistema homogêneo ou subgrupos? GAN única ou múltipla?

### As estações são distintas

- **8 de 13 variáveis** rejeitam homogeneidade (Kruskal-Wallis p < 0,01).
- **Discriminadores mais fortes:** Condutividade (H = 322), Turbidez (H = 278), Fósforo Total (H = 242), DBO (H = 206), Coliformes (H = 161).
- **Variáveis sem poder discriminativo:** Microcistinas, Sólidos Suspensos, Nitrato — refletem fenômenos regionais ou são raras demais.

### Estrutura de clusters

- **PCA captura 70% da variância em 2D** (PC1 = trofia/salinidade, 50%; PC2 = fotossíntese/contaminação, 20%).
- **K = 2 é a estrutura natural** (silhueta = 0,335; salto grande no dendrograma):
  - **Cluster marinho:** MR369, TJ303 — alta Condutividade, baixa DBO/Turbidez.
  - **Cluster continental:** CM320, JC341, JC342, MR361, MR363, TJ306 — perfil eutrofizado.
- **K = 3** (silhueta 0,254) refina o cluster continental em "contaminadas" (CM320, TJ306) vs "eutrofizadas" (JC341, JC342, MR361, MR363), com ganho marginal.

### Decisão arquitetural confirmada

**Uma única GAIN com `Codigo Local` como feature categórica** (one-hot ou embedding).

| Estratégia | Avaliação |
|---|---|
| **1 GAIN com `Codigo Local` one-hot** (escolhida) | 657 amostras de treino; aprendizado cruzado entre estações; feature condicional permite especialização implícita. |
| 2 GAINs (uma por cluster K=2) | Cluster marinho fica com ~228 amostras — abaixo do mínimo viável para GAN robusta. |
| 3 GAINs (K=3) | Inviável: 113/182/362 amostras por GAIN, sub-amostragem severa. |

### Cuidados

- **JC341 (n=10) e MR363 (n=8):** atribuições de cluster instáveis. Considerar peso reduzido no treino ou consolidação com vizinha (JC341 → JC342, MR363 → MR361).

---

## 6. Censura por Limites de Detecção (`06_lds`)

**Pergunta:** a censura (`<` / `>`) é estruturada o bastante para ser feature da GAIN?

### Classificação por variável

| Decisão | Variáveis | Tratamento |
|---|---|---|
| **`alta` (> 10%)** | Nitrato (22%) | Manter `_LD`; considerar Tobit ou pós-processamento que clipe geração `≤ LOD` quando `_LD = <`. |
| **`moderada` (2–10%)** | DBO, Nitrogênio Amoniacal Total, Coliformes Termotolerantes, Cianobactérias, Microcistinas | Manter `_LD` como feature one-hot {`<`, `>`, vazio}. |
| **`desprezível` (< 2%)** | OD (0,9%) | Descartar `_LD`. |
| **`sem_censura`** | Fósforo Total, Condutividade, pH, Turbidez, Temperatura da Água, Sólidos Suspensos Totais | Descartar `_LD` (nunca aparece censura). |

### Observação técnica

**Coliformes Termotolerantes** é a única variável com **censura predominante à direita** (`>` = 28 vs `<` = 2). Reflete o **teto reportável** da técnica de tubos múltiplos — método satura quando a contagem real ultrapassa o maior valor da tabela. Todas as demais variáveis com censura têm `<` dominante (piso analítico).

### Refinamento da decisão original

O plano original previa tratar `_LD` como feature em todas as variáveis. A EDA **simplifica isso**: a feature `_LD` precisa ser codificada apenas em 6 variáveis (1 alta + 5 moderadas); as outras 7 podem descartar a coluna `_LD`.

---

## 7. Decisões consolidadas para a Etapa 2 (Pré-processamento)

### `02_Preprocessing/01_transformacoes.ipynb`

- Aplicar transformações conforme `Data/Figures/01_EDA/tabelas/dist_resumo.csv`:
  - 3 variáveis em identidade (pH, Temperatura da Água, Microcistinas).
  - 3 variáveis em Yeo-Johnson (DBO, OD, Nitrogênio Amoniacal Total).
  - 7 variáveis em Box-Cox com λ específico.
- Marcar **Fósforo Total** como variável com kurtosis residual alta — necessita diagnóstico posterior.

### `02_Preprocessing/02_features_temporais.ipynb`

- Derivar: `mes_sin`, `mes_cos`, `estacao_do_ano`, `ano_norm`, `dias_desde_inicio`.
- Manter `Ano` como feature categórica caso a tendência de Cianobactérias (ρ = −0,25) precise ser modelada explicitamente.

### `02_Preprocessing/03_encoding.ipynb`

- `Codigo Local` → one-hot (8 níveis) **ou** embedding aprendido pela GAIN.
- `_LD` codificar one-hot apenas para: **Nitrato, DBO, Nitrogênio Amoniacal Total, Coliformes Termotolerantes, Cianobactérias, Microcistinas** (6 variáveis). Descartar as outras 7 colunas `_LD`.

### Critérios de exclusão / atenção

- Considerar **excluir JC341 e MR363** das análises baseadas em volume de dados (n=10 e n=8 respectivamente, descontinuadas em 2015), ou aplicar peso reduzido no treino da GAIN.
- **2020** é ano de outlier amostral (1 coleta/estação); deve ser preservado mas não usado como base de validação isolada.

---

## 8. Decisões para a Etapa 4 (GAIN)

### Arquitetura

- **Modelo único, tabular, condicional** — confirmado pelas EDAs de estações e temporal.
- Features condicionais: `Codigo Local`, features temporais derivadas, `_LD` (6 variáveis).

### Pontos de atenção no diagnóstico (`04_GAIN/03_diagnostico.ipynb`)

1. **Fósforo Total** — verificar se o gerador cobre a cauda extrema (kurtosis residual = 8,9 mesmo após Box-Cox).
2. **Nitrato** — censura à esquerda em 22% das medições; sem tratamento dedicado, o gerador produzirá valores acima do LOD para amostras censuradas. Soluções: clip pós-geração, loss customizada ou componente Tobit.
3. **Coliformes Termotolerantes** — censura à direita por saturação metodológica; o gerador pode subestimar o valor real quando `_LD = >`.
4. **Cianobactérias e Microcistinas** — variáveis críticas (cobertura 20% e 8%), com correlações apenas moderadas (|ρ| ≈ 0,5) e baseadas em n pequeno. A imputação dessas variáveis será o teste mais difícil para a GAIN; reservar atenção especial nas métricas de avaliação.
5. **Turbidez e Coliformes** — sem sazonalidade detectável. Features temporais terão valor preditivo limitado; o gerador deve apoiar-se nas correlações estruturais (Turbidez ↔ Condutividade ↔ DBO).

---

## 9. Síntese final em uma frase

> O dataset é esparso (98% das linhas com ≥ 4 variáveis, zero linhas completas), espacialmente estruturado (gradiente marinho ↔ continental com K = 2 clusters naturais), sazonalmente significativo em 11/13 variáveis, irregular no tempo (gaps de até 1 ano), e com censura relevante apenas no Nitrato. A escolha arquitetural — **GAIN tabular única com `Codigo Local`, features temporais cíclicas e `_LD` como features condicionais** — está validada por todos os notebooks.

---

## Anexos

### Artefatos gerados

- **Tabelas** (`Data/Figures/01_EDA/tabelas/`): `descritivo_global.csv`, `dist_resumo.csv`, `corr_top_pares.csv`, `temp_sumario.csv`, `est_kmeans_atribuicao.csv`, `lds_resumo.csv`.
- **Figuras** (`Data/Figures/01_EDA/figuras/`): ~20 PNGs cobrindo cobertura, distribuições, correlações, séries temporais, PCA/clusters e censura.

### Próximo passo

Etapa 2 — Pré-processamento, começando por `02_Preprocessing/01_transformacoes.ipynb`.
