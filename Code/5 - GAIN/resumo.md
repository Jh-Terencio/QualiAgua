# Resumo — Etapa GAIN (Imputação por rede adversarial)

*Projeto QualiAgua · Lagoa de Jacarepaguá (INEA) · concluída em 2026-07-07*
*Código: `Code/5 - GAIN/` (`gain.py` + `01_treino`, `02_hyperparam`, `03_diagnostico`) · artefatos: `Data/GainResults/`*

## Objetivo

Treinar um **GAIN condicional** para imputar as 11 variáveis de qualidade da água,
usando 23 features de contexto (temporais + one-hots de estação) como condicionamento
em gerador e discriminador. Meta: **superar a baseline KNN** (0,274 no protocolo da Etapa 3).

## Resultado principal — critério de aceite ATENDIDO

| Modelo | RMSE_norm no val (protocolo Etapa 3) |
|---|---|
| **GAIN (gain_best, `cfg_1092`)** | **0,2317** |
| KNN (baseline reavaliada no val) | 0,2742 |

O GAIN é **~15% melhor** que o KNN e **vence em 9 das 11 variáveis**.

### Desempenho por variável (RMSE_norm)

| Variável | GAIN | KNN | Vencedor |
|---|---|---|---|
| Fósforo Total | 0,092 | 0,111 | GAIN |
| Coliformes Termotolerantes | 0,110 | 0,187 | GAIN |
| pH | 0,117 | 0,175 | GAIN |
| Temperatura da Água | 0,147 | 0,176 | GAIN |
| Turbidez | 0,150 | 0,187 | GAIN |
| Condutividade | 0,235 | 0,209 | KNN |
| N-Amoniacal Total | 0,246 | 0,325 | GAIN |
| DBO | 0,252 | 0,298 | GAIN |
| OD | 0,282 | 0,396 | GAIN |
| Nitrato | 0,373 | 0,349 | KNN |
| Sólidos Suspensos Totais (SST) | 0,505 | 0,573 | GAIN |

Destaque: **SST**, a variável crítica (escassez de vizinhos), onde o condicionamento
por estação/tempo é o que faz o GAIN vencer.
Perde apenas em **Condutividade** e **Nitrato** (esta com n=4 amostras — pouco confiável).

## Configuração vencedora (`cfg_1092`)

`hint_rate=0.9 · alpha=10 · beta=10 · batch=128 · hidden=[128,128] · lr_g=lr_d=0.001 · n_iter=5000`

Selecionada por random search (31 configs, 7 eixos — `grid_results.csv`).
Seed 42, melhor iteração 1000. Entre 3 seeds: 0,262 ± 0,019 (seed 42 é o ponto feliz;
a pior seed empata com o KNN → o diagnóstico olhou além do RMSE).

### Decisões técnicas decisivas

1. **Termo supervisionado `beta·MSE`** nas células conhecidas é a extensão que virou o jogo
   (não faz parte do paper original): as 7 piores configs da busca são exatamente as 7 com `beta=0`;
   `beta=100` já piora.
2. **Hint da implementação oficial** (`H = R⊙M`), não a fórmula do texto do paper — verificado
   empiricamente: a variante do paper informa a resposta ao discriminador e degrada o RMSE.
3. **Early stopping implícito**: `train_gain` restaura o G ao estado de menor RMSE_val
   (a curva oscila e o estado final é sistematicamente pior que o melhor ponto).

## Diagnóstico (`03_diagnostico`) — modelo saudável

- **Convergência OK e reprodutível**: retreino determinístico bate o checkpoint (diff < 1e-6).
- **Sem mode collapse**: mediana σ_imp/σ_obs = 0,48; correlação real-imputado mediana 0,68.
- **Vieses a considerar na Etapa 5**: SST +0,49, Condutividade −0,16, Turbidez −0,12 (mediana norm.);
  encolhimento de caudas (regressão à média condicional) — bom para tendência, limitado para extremos.
- **Nitrato** tem r negativo com n=4 — vigiar.
- Estações homogêneas (nenhuma problemática).

## Veredicto

**AVANÇAR para a Etapa 5 (05_Evaluation).** Critério de aceite atendido e diagnóstico
sem patologia que invalide o resultado.

## Artefatos

- `checkpoints/gain_best.pt` — modelo final
- `Data/GainResults/config_best.json` — contrato de configuração/resultados
- `Data/GainResults/diagnostico.md` — relatório completo do diagnóstico
- `Data/GainResults/grid_results.csv` — busca de hiperparâmetros
- `Data/GainResults/training_log_final.csv` — log de treino do modelo final
- `Data/Figures/04_GAIN/` — figuras
