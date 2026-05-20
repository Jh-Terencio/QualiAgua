"""Utilidades de mascaramento da GAIN.

Gerado por Code/3 - Preprocessing/06_mascaras.ipynb. Edite o notebook, não este arquivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_artificial_mask(M, miss_rate, seed=None):
    """Gera B (máscara artificial) com a mesma forma de M.

    M: array (N, P) com 0/1 ou DataFrame.
    miss_rate: float (taxa única) ou dict {coluna: taxa}.
    seed: int para reprodutibilidade.

    Convenção:
    - Onde M==0 -> B=0 (já faltante, nada a esconder).
    - Onde M==1 -> B ~ Bernoulli(1 - miss_rate); B=1 mantém, B=0 esconde.
    """
    rng = np.random.default_rng(seed)

    if isinstance(M, pd.DataFrame):
        M_arr = M.to_numpy()
        colunas = list(M.columns)
        indice = M.index
        retorno_df = True
    else:
        M_arr = np.asarray(M)
        colunas = None
        indice = None
        retorno_df = False

    if isinstance(miss_rate, dict):
        if colunas is None:
            raise ValueError("miss_rate dict requer DataFrame em M para identificar colunas.")
        rates = np.array([miss_rate[c] for c in colunas], dtype=float)
        u = rng.random(M_arr.shape)
        B = (u >= rates[np.newaxis, :]).astype(np.int8)
    else:
        rate = float(miss_rate)
        u = rng.random(M_arr.shape)
        B = (u >= rate).astype(np.int8)

    B = B * (M_arr == 1).astype(np.int8)

    if retorno_df:
        return pd.DataFrame(B, columns=colunas, index=indice)
    return B
