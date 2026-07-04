"""Utilidades da Etapa 3 — Baselines de imputação.

Protocolo de avaliação compartilhado por `01_baselines_simples.ipynb` e
`02_baselines_ml.ipynb`. A conversão para módulo está prevista no plano
(`Pipeline/03_Baselines/02_baselines_ml.md`, §2) para que os dois notebooks
usem exatamente a mesma máscara `B`, a mesma inversão de escala e o mesmo
cálculo de métricas — pré-condição do critério de aceite ("números
comparáveis").

As funções de inversão (Box-Cox / Yeo-Johnson) replicam as validadas em
`Code/3 - Preprocessing/01_transformacoes.ipynb`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special


# ---------------------------------------------------------------------------
# Inversão das transformações da Etapa 2 (ordem: MinMax -> Box-Cox/Yeo-Johnson)
# ---------------------------------------------------------------------------

def invert_boxcox(y: np.ndarray, lmbda: float) -> np.ndarray:
    """Inversa de Box-Cox com lambda fixo. NaN preservados."""
    y = np.asarray(y, dtype=float)
    x = np.full_like(y, np.nan)
    mask = ~np.isnan(y)
    x[mask] = special.inv_boxcox(y[mask], lmbda)
    return x


def invert_yeojohnson(y: np.ndarray, lmbda: float) -> np.ndarray:
    """Inversa analítica de Yeo-Johnson (não há helper direto no SciPy)."""
    y = np.asarray(y, dtype=float)
    x = np.full_like(y, np.nan)
    mask = ~np.isnan(y)
    if not mask.any():
        return x

    ym = y[mask]
    xm = np.empty_like(ym)
    pos = ym >= 0
    neg = ~pos

    if np.isclose(lmbda, 0.0):
        xm[pos] = np.expm1(ym[pos])
    else:
        xm[pos] = np.power(ym[pos] * lmbda + 1.0, 1.0 / lmbda) - 1.0

    if np.isclose(lmbda, 2.0):
        xm[neg] = 1.0 - np.exp(-ym[neg])
    else:
        expo = 2.0 - lmbda
        xm[neg] = 1.0 - np.power(-expo * ym[neg] + 1.0, 1.0 / expo)

    x[mask] = xm
    return x


def inverter_transformacao(y: np.ndarray, tipo: str, lmbda) -> np.ndarray:
    if tipo == "identidade":
        return np.asarray(y, dtype=float).copy()
    if tipo == "boxcox":
        return invert_boxcox(y, lmbda)
    if tipo == "yeo-johnson":
        return invert_yeojohnson(y, lmbda)
    raise ValueError(f"Tipo desconhecido: {tipo!r}")


def desnormalizar(df_norm: pd.DataFrame, variaveis: list[str],
                  scalers: dict, transform_params: dict) -> pd.DataFrame:
    """Leva um DataFrame do espaço normalizado [-1, 1] à escala física original.

    Sequência de inversão (Etapa 2, `resumo.md`):
    1. clip para [-1, 1] — predições fora do range visto no fit extrapolariam o
       domínio das inversas (Box-Cox com lambda<0 fica indefinida). O clip é
       conservador e afeta todos os métodos igualmente.
    2. `MinMaxScaler.inverse_transform` por variável (espaço transformado).
    3. Inversa de Box-Cox/Yeo-Johnson com o lambda de `transform_params.json`.
    """
    out = pd.DataFrame(index=df_norm.index)
    for var in variaveis:
        vals = df_norm[var].to_numpy(dtype=float).clip(-1.0, 1.0)
        vals = scalers[var].inverse_transform(vals.reshape(-1, 1)).ravel()
        params = transform_params[var]
        out[var] = inverter_transformacao(vals, params["tipo"], params["lambda"])
    return out


# ---------------------------------------------------------------------------
# Contexto de estação
# ---------------------------------------------------------------------------

def coluna_estacao(df: pd.DataFrame, prefixo: str = "est_") -> pd.Series:
    """Recupera o código da estação a partir do one-hot `est_*`."""
    onehot = df[[c for c in df.columns if c.startswith(prefixo)]]
    return onehot.idxmax(axis=1).str.removeprefix(prefixo).rename("estacao")


# ---------------------------------------------------------------------------
# Protocolo de avaliação
# ---------------------------------------------------------------------------

def avaliar_baseline(nome_metodo: str, predict_fn, df_test: pd.DataFrame,
                     mask_real: pd.DataFrame, variaveis: list[str],
                     scalers: dict, transform_params: dict,
                     seeds: list[int], miss_rate: float,
                     generate_artificial_mask) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia um método de imputação sob o protocolo da Etapa 3.

    Para cada seed:
    1. Gera `B = generate_artificial_mask(M, miss_rate, seed)` sobre o test.
    2. Esconde as células `(M=1, B=0)`: `X_obs = X` onde `M*B == 1`, NaN no resto.
    3. `X_pred = predict_fn(X_obs, seed)` — matriz completa no espaço normalizado.
    4. Desnormaliza verdade e predição para a escala física original.
    5. RMSE/MAE por variável **apenas** nas posições `(M=1, B=0)` — os valores
       que conhecíamos e escondemos. Variáveis sem célula avaliável na seed
       (ex.: Coliformes, 0 observações no test) não geram linha.

    Métricas são reportadas em duas escalas:
    - `RMSE`/`MAE` — escala física original (interpretável por variável).
    - `RMSE_norm`/`MAE_norm` — espaço normalizado [-1, 1] (comparável entre
      variáveis; usada para o ranking médio dos métodos).

    Retorna `(df_metricas, df_predicoes)`; as predições (escala original, uma
    matriz completa por seed) alimentam os `predictions_<metodo>.parquet`.
    """
    X_true_norm = df_test[variaveis].copy()
    X_true_orig = desnormalizar(X_true_norm, variaveis, scalers, transform_params)
    estacao = coluna_estacao(df_test)

    linhas, blocos_pred = [], []
    for seed in seeds:
        B = generate_artificial_mask(mask_real, miss_rate, seed=seed)
        escondidas = (mask_real == 1) & (B == 0)

        X_obs = X_true_norm.where((mask_real * B) == 1)
        X_pred_norm = predict_fn(X_obs, seed)[variaveis]
        X_pred_orig = desnormalizar(X_pred_norm, variaveis, scalers, transform_params)

        for var in variaveis:
            idx = escondidas[var].to_numpy(dtype=bool)
            n = int(idx.sum())
            if n == 0:
                continue
            err_orig = X_pred_orig.loc[idx, var] - X_true_orig.loc[idx, var]
            err_norm = X_pred_norm.loc[idx, var] - X_true_norm.loc[idx, var]
            linhas.append({
                "metodo": nome_metodo,
                "variavel": var,
                "seed": seed,
                "miss_rate": miss_rate,
                "n_avaliado": n,
                "RMSE": float(np.sqrt(np.mean(err_orig ** 2))),
                "MAE": float(np.mean(np.abs(err_orig))),
                "RMSE_norm": float(np.sqrt(np.mean(err_norm ** 2))),
                "MAE_norm": float(np.mean(np.abs(err_norm))),
            })

        bloco = X_pred_orig.copy()
        bloco.insert(0, "seed", seed)
        bloco.insert(1, "Data", df_test["Data"].to_numpy())
        bloco.insert(2, "estacao", estacao.to_numpy())
        blocos_pred.append(bloco)

    df_metricas = pd.DataFrame(linhas)
    df_predicoes = pd.concat(blocos_pred, ignore_index=True)
    return df_metricas, df_predicoes


def agregar_metricas(df_metricas: pd.DataFrame) -> pd.DataFrame:
    """Agrega seeds -> schema do entregável (`tabela_baselines.csv`)."""
    agg = (df_metricas
           .groupby(["metodo", "variavel"], sort=False)
           .agg(RMSE_media=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                MAE_media=("MAE", "mean"), MAE_std=("MAE", "std"),
                RMSE_norm_media=("RMSE_norm", "mean"),
                MAE_norm_media=("MAE_norm", "mean"),
                n_avaliado_total=("n_avaliado", "sum"))
           .reset_index())
    return agg
