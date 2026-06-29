# Resumo da Etapa 1 — EDA

**Projeto:** QualiAgua — pipeline de imputação por GAN para qualidade da água do sistema lagunar de Jacarepaguá (INEA, 2012–2025).

**Escopo:** 639 coletas × 11 variáveis físico-químicas e biológicas × 6 estações (CM320, JC342, MR361, MR369, TJ303, TJ306) ao longo de 14 anos.

**Objetivo deste documento:** consolidar os achados dos seis notebooks da EDA (`01_descritivo` → `06_lds`) em um relatório objetivo, com as decisões que serão propagadas para a Etapa 2 (Pré-processamento) e para o desenho da GAIN (Etapa 4).

---

## 1. Cobertura e estrutura de ausência (`01_descritivo`)

**Pergunta:** quanto dado existe por variável, estação e ano? Onde estão os buracos?

### Variáveis por nível de cobertura

| Faixa | Variáveis | Implicação |
|---|---|---|
| **Robustas (≥ 95%)** | DBO (98%), OD (97%), Condutividade (97%), pH (97%), Turbidez (96%), Temperatura da Água (96%), Fósforo Total (95%) | Formam o núcleo de treino; podem servir de "âncoras" para imputar as demais. |
| **Intermediárias (50–80%)** | Nitrogênio Amoniacal Total (77%), Coliformes Termotolerantes (74%), Nitrato (53%) | Imputação plenamente viável; cobertura suficiente para validação. |
| **Crítica (< 25%)** | Sólidos Suspensos Totais (24%) | Imputação obrigatória, mas com incerteza alta. Validação dependerá da consistência inter-variáveis, não de hold-out direto. |

### Lacunas estruturais identificadas

- **2020** foi o ano de monitoramento mais reduzido (apenas 1 coleta por estação ativa) — efeito direto da pandemia.
- **TJ306** sem coletas em 2015 e em 2024–2025; total de 68 coletas vs ~114 das demais estações.
- **Sólidos Suspensos Totais** ausentes de 2014 a 2022 (lacuna de ~9 anos).
- **Coliformes Termotolerantes** descontinuados a partir de 2023 (zero em 2024–2025).

### Sample size efetivo

- **46 linhas completas** nas 11 variáveis (7% do total) — diferentemente da base anterior (13 variáveis, zero linhas completas), a remoção de Cianobactérias e Microcistinas faz surgirem observações completas, embora poucas.
- 626 linhas (98% do total) têm ≥ 4 variáveis preenchidas — base sólida para baselines clássicos (MICE).
- Queda acentuada a partir de k=9 (504 linhas → 212 em k=10 → 46 em k=11), refletindo a coexistência rara de Nitrato, Sólidos Suspensos Totais e Coliformes Termotolerantes com as demais.

---

## 2. Forma das distribuições e transformações (`02_distribuicoes`)

**Pergunta:** quais variáveis são simétricas e quais precisam de transformação antes da GAIN?

### Decisões de transformação

| Tratamento | Variáveis | λ (Box-Cox) | Resultado |
|---|---|---|---|
| **Identidade** (skew < 1) | Temperatura da Água, Condutividade, pH | — | Sem transformação. |
| **Yeo-Johnson** (têm zeros) | DBO, OD, Nitrogênio Amoniacal Total | 0,07 / 0,28 / −0,08 | Skew → ~0. |
| **Box-Cox** | Nitrato, Fósforo Total, Turbidez, Sólidos Suspensos Totais, Coliformes Termotolerantes | −0,39 a 0,29 | Skew → ~0 em todos exceto Fósforo Total. |

### Caso problemático: Fósforo Total

- Skew original = 13,2; kurtosis = 177.
- Box-Cox (λ = −0,14) reduz skew para −0,40, **mas kurtosis residual permanece em 8,8**.
- Os picos extremos são **eventos genuínos de eutrofização**, não outliers a serem suprimidos.
- **Alerta para a GAIN:** o gerador pode subestimar esses picos; verificar cobertura da cauda em `04_GAIN/03_diagnostico.ipynb`.

### Observação técnica

- Turbidez (λ ≈ 0,03) e Coliformes Termotolerantes (λ ≈ 0,07) → equivalentes a `log` simples; útil quando interpretabilidade for prioridade.
- Condutividade passou a ser simétrica o bastante (skew = 0,66) para dispensar transformação — mudança em relação à base com 8 estações.

---

## 3. Estrutura de dependência (`03_correlacoes`)

**Pergunta:** quais pares carregam mais informação mútua? A estrutura é estável entre estações?

### Cinco pares fortes (|ρ_Spearman| > 0,5, p < 0,05)

| Par | ρ Spearman | Interpretação |
|---|---|---|
| Condutividade × Turbidez | **−0,59** | Gradiente marinho→continental: estações marinhas (MR369, TJ303) têm Condutividade alta e Turbidez baixa. |
| DBO × Turbidez | **+0,57** | Carga orgânica acompanha material em suspensão. |
| OD × pH | **+0,55** | Fotossíntese eleva ambos simultaneamente. **Único par estável em todas as 6 estações.** |
| DBO × Condutividade | **−0,51** | Eutrofização vs salinidade. |
| DBO × Fósforo Total | **+0,51** | Carga orgânica acompanha nutrientes. |

### Observações relevantes

- **Spearman > Pearson** em vários pares (diferença até 0,43 em Nitrogênio Amoniacal × Fósforo Total) — relações monotônicas **não-lineares** que Pearson subestima. Spearman será a métrica de referência daqui em diante. (Pearson só encontra 1 par forte; Spearman encontra 5.)
- **Nenhum par ultrapassa |ρ| > 0,7** — não há redundância informacional extrema. A GAIN precisará explorar **combinações** de variáveis, não vizinhos diretos.
- **Heterogeneidade entre estações é alta:** TJ303 tem **22 pares fortes**, CM320 e MR369 têm apenas **1**. DBO × Turbidez varia muito de intensidade entre estações (de ~0 em CM320 a +0,59 em TJ303).

### Núcleo informacional para imputação

**DBO, OD, pH, Turbidez, Condutividade** — variáveis com cobertura > 610 e que aparecem em todos os top-pares. Servirão de "âncoras" para imputar as variáveis de cobertura mais baixa (Sólidos Suspensos Totais, Nitrato).

---

## 4. Padrão temporal (`04_temporal`)

**Pergunta:** há sazonalidade e tendência exploráveis? Modelo sequencial ou tabular?

### Esforço amostral irregular

- Coleta nominalmente **mensal** (gap mediano 36–48 dias por estação).
- ~24–34% dos gaps são > 60 dias; gap máximo de 365–427 dias em todas as estações (refletindo o vazio de 2020).
- **Implicação direta:** modelos sequenciais puros (LSTM, TimeGAN) **estão descartados** — a irregularidade falsifica o pressuposto de continuidade temporal.

### Sazonalidade (Kruskal-Wallis por mês)

- **Detectada em 10 das 11 variáveis** (p < 0,05).
- **Sinal mais forte:** Temperatura da Água (H = 345; ACF[6] = −0,54 e ACF[12] = +0,42 — assinatura senoidal anual perfeita).
- **Sem sazonalidade detectável:** apenas Coliformes Termotolerantes (p = 0,35) — dominado por **eventos pontuais** (chuvas, contaminação) e não pelo calendário. Turbidez passa de forma marginal (p = 0,046).

### Tendências de longo prazo

- Significativas estatisticamente pelo n grande, mas **magnitude pequena** (|ρ| < 0,16 em todas as variáveis).
- Destaques: **OD em alta** (ρ = +0,16) e **Nitrogênio Amoniacal Total em queda** (ρ = −0,15).
- O sistema é **razoavelmente estacionário** no horizonte 2012–2025 — não há mudança de regime visível.

### Features temporais para a GAIN

| Feature | Fórmula | Justificativa |
|---|---|---|
| `mes_sin`, `mes_cos` | `sin/cos(2π · Mes / 12)` | Codificação cíclica do mês (evita salto dez→jan). |
| `estacao_do_ano` | binária: úmido (nov–mar) vs seco (abr–out) | Regime climático carioca. |
| `ano_norm` | `(Ano − 2012) / 13` | Captura tendência linear (mesmo que fraca). |
| `dias_desde_inicio` | `(Data − 2012-01-01).days` | Alternativa contínua, granularidade fina. |

### Decisão arquitetural

**GAIN tabular com features temporais derivadas** — confirmado. A irregularidade da amostragem inviabiliza sequencial; features cíclicas + binária estação capturam o sinal sazonal de 10 das 11 variáveis sem exigir continuidade.

---

## 5. Estrutura espacial (`05_estacoes`)

**Pergunta:** as 6 estações se comportam como sistema homogêneo ou subgrupos? GAN única ou múltipla?

### As estações são distintas

- **8 de 11 variáveis** rejeitam homogeneidade (Kruskal-Wallis p < 0,01).
- **Discriminadores mais fortes:** Condutividade (H = 306), Turbidez (H = 263), Fósforo Total (H = 234), DBO (H = 205), Coliformes (H = 159).
- **Variáveis sem poder discriminativo:** Nitrato, Sólidos Suspensos, pH — refletem fenômenos regionais ou são raras demais.

### Estrutura de clusters

- **PCA captura 84% da variância em 2D** (PC1 = trofia/salinidade, 69%; PC2 = contaminação/fotossíntese, 15%).
- **K = 2 é a estrutura natural** (silhueta = 0,495; salto grande no dendrograma):
  - **Cluster marinho:** MR369, TJ303 — alta Condutividade, baixa DBO/Turbidez.
  - **Cluster continental:** CM320, JC342, MR361, TJ306 — perfil eutrofizado.
- **K = 3** (silhueta 0,286) apenas isola MR361 como cluster próprio, com ganho marginal.

### Decisão arquitetural confirmada

**Uma única GAIN com `Codigo Local` como feature categórica** (one-hot ou embedding).

| Estratégia | Avaliação |
|---|---|
| **1 GAIN com `Codigo Local` one-hot** (escolhida) | 639 amostras de treino; aprendizado cruzado entre estações; feature condicional permite especialização implícita. |
| 2 GAINs (uma por cluster K=2) | Cluster marinho fica com ~228 amostras — abaixo do mínimo viável para GAN robusta. |
| 3 GAINs (K=3) | Inviável: 228/115/296 amostras por GAIN, sub-amostragem severa. |

### Cuidados

- **TJ306 (n=68):** menor amostra, descontinuada após 2023; considerar peso ajustado no treino. **MR361:** perfil intermediário próprio (isola-se em K=3) que o `Codigo Local` deve capturar.

---

## 6. Censura por Limites de Detecção (`06_lds`)

**Pergunta:** a censura (`<` / `>`) é estruturada o bastante para ser feature da GAIN?

### Classificação por variável

| Decisão | Variáveis | Tratamento |
|---|---|---|
| **`alta` (> 10%)** | Nitrato (22%) | Manter `_LD`; considerar Tobit ou pós-processamento que clipe geração `≤ LOD` quando `_LD = <`. |
| **`moderada` (2–10%)** | Coliformes Termotolerantes (6%), DBO (2%), Nitrogênio Amoniacal Total (2%) | Manter `_LD` como feature one-hot {`<`, `>`, vazio}. |
| **`desprezível` (< 2%)** | OD (1%) | Descartar `_LD`. |
| **`sem_censura`** | Fósforo Total, Condutividade, pH, Turbidez, Temperatura da Água, Sólidos Suspensos Totais | Descartar `_LD` (nunca aparece censura). |

### Observação técnica

**Coliformes Termotolerantes** é a única variável com **censura predominante à direita** (`>` = 28 vs `<` = 1). Reflete o **teto reportável** da técnica de tubos múltiplos — método satura quando a contagem real ultrapassa o maior valor da tabela. Todas as demais variáveis com censura têm `<` dominante (piso analítico).

### Refinamento da decisão original

O plano original previa tratar `_LD` como feature em todas as variáveis. A EDA **simplifica isso**: a feature `_LD` precisa ser codificada apenas em 4 variáveis (1 alta + 3 moderadas); as outras 7 podem descartar a coluna `_LD`.

---

## 7. Decisões consolidadas para a Etapa 2 (Pré-processamento)

### `02_Preprocessing/01_transformacoes.ipynb`

- Aplicar transformações conforme `Data/Figures/01_EDA/tabelas/dist_resumo.csv`:
  - 3 variáveis em identidade (Temperatura da Água, Condutividade, pH).
  - 3 variáveis em Yeo-Johnson (DBO, OD, Nitrogênio Amoniacal Total).
  - 5 variáveis em Box-Cox com λ específico (Nitrato, Fósforo Total, Turbidez, Sólidos Suspensos Totais, Coliformes Termotolerantes).
- Marcar **Fósforo Total** como variável com kurtosis residual alta — necessita diagnóstico posterior.

### `02_Preprocessing/02_features_temporais.ipynb`

- Derivar: `mes_sin`, `mes_cos`, `estacao_do_ano`, `ano_norm`, `dias_desde_inicio`.

### `02_Preprocessing/03_encoding.ipynb`

- `Codigo Local` → one-hot (6 níveis) **ou** embedding aprendido pela GAIN.
- `_LD` codificar one-hot apenas para: **Nitrato, Coliformes Termotolerantes, DBO, Nitrogênio Amoniacal Total** (4 variáveis). Descartar as outras 7 colunas `_LD`.

### Critérios de exclusão / atenção

- As estações descontinuadas **JC341 e MR363** já foram removidas do dataset na Etapa 0 (treatment), reduzindo de 657 para 639 coletas.
- **2020** é ano de outlier amostral (1 coleta/estação); deve ser preservado mas não usado como base de validação isolada.
- **TJ306** (n=68, ativa só até 2023) pode receber peso reduzido no treino.

---

## 8. Decisões para a Etapa 4 (GAIN)

### Arquitetura

- **Modelo único, tabular, condicional** — confirmado pelas EDAs de estações e temporal.
- Features condicionais: `Codigo Local`, features temporais derivadas, `_LD` (4 variáveis).

### Pontos de atenção no diagnóstico (`04_GAIN/03_diagnostico.ipynb`)

1. **Fósforo Total** — verificar se o gerador cobre a cauda extrema (kurtosis residual = 8,8 mesmo após Box-Cox).
2. **Nitrato** — censura à esquerda em 22% das medições; sem tratamento dedicado, o gerador produzirá valores acima do LOD para amostras censuradas. Soluções: clip pós-geração, loss customizada ou componente Tobit.
3. **Coliformes Termotolerantes** — censura à direita por saturação metodológica; o gerador pode subestimar o valor real quando `_LD = >`. Também é a única variável sem sazonalidade detectável (apoiar-se nas correlações estruturais: DBO ↔ Turbidez ↔ Condutividade).
4. **Sólidos Suspensos Totais** — variável crítica (cobertura 24%); a imputação dessa variável será o teste mais difícil para a GAIN; reservar atenção especial nas métricas de avaliação.

---

## 9. Síntese final em uma frase

> O dataset é moderadamente esparso (98% das linhas com ≥ 4 variáveis, 46 linhas completas), espacialmente estruturado (gradiente marinho ↔ continental com K = 2 clusters naturais), sazonalmente significativo em 10/11 variáveis, irregular no tempo (gaps de até 1 ano), e com censura relevante apenas no Nitrato. A escolha arquitetural — **GAIN tabular única com `Codigo Local`, features temporais cíclicas e `_LD` como features condicionais** — está validada por todos os notebooks.

---

## Anexos

### Artefatos gerados

- **Tabelas** (`Data/Figures/01_EDA/tabelas/`): `descritivo_global.csv`, `dist_resumo.csv`, `corr_top_pares.csv`, `temp_sumario.csv`, `est_kmeans_atribuicao.csv`, `lds_resumo.csv`.
- **Figuras** (`Data/Figures/01_EDA/figuras/`): ~20 PNGs cobrindo cobertura, distribuições, correlações, séries temporais, PCA/clusters e censura.

### Próximo passo

Etapa 2 — Pré-processamento, começando por `02_Preprocessing/01_transformacoes.ipynb`.
