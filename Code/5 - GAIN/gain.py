"""GAIN — Generative Adversarial Imputation Network (Yoon et al., ICML 2018).

Implementação condicional para o QualiAgua (Etapa 4):

- O gerador imputa apenas as `n_target` variáveis de qualidade da água; as
  features de contexto (temporais, one-hot de estação, one-hot de censura)
  entram como condicionamento e nunca são imputadas — decisão registrada em
  `Code/3 - Preprocessing/06_mascaras.ipynb`.
- A cada batch, uma máscara artificial `B` (via `generate_artificial_mask`,
  taxas por variável) esconde células observadas; a máscara efetiva vista
  pelo modelo é `M_eff = M ⊙ B` — protocolo da Etapa 2.
- Perdas fiéis ao paper: `L_D = BCE(D(x̂, H), M_eff)` sobre as células-alvo;
  `L_G = L_adv + α · L_rec + β · L_sup`, com `L_adv = −E[(1−M_eff) · log D(x̂, H)]`,
  `L_rec = MSE` nas células mantidas (`M_eff = 1`) e `L_sup = MSE` nas células
  escondidas artificialmente (`M = 1, B = 0`) — extensão do QualiAgua sobre o
  paper: nessas células a verdade é conhecida, então um termo supervisionado
  otimiza exatamente o que o RMSE de validação mede (`β = 0` reproduz o GAIN
  original; diagnóstico em `01_treino.ipynb`, síntese).

Consumido por `01_treino.ipynb` (e pelos notebooks 02/03 da etapa).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Redes
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """MLP: [x̃ (n_target), contexto (n_context), M_eff (n_target)] → x̄ (n_target).

    Saída `tanh` — alinhada ao range [-1, 1] da normalização da Etapa 2.
    """

    def __init__(self, n_target: int, n_context: int, hidden: list[int]):
        super().__init__()
        dims = [2 * n_target + n_context] + list(hidden)
        camadas = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            camadas += [nn.Linear(d_in, d_out), nn.ReLU()]
        camadas += [nn.Linear(dims[-1], n_target), nn.Tanh()]
        self.net = nn.Sequential(*camadas)

    def forward(self, x_tilde, context, m_eff):
        return self.net(torch.cat([x_tilde, context, m_eff], dim=1))


class Discriminator(nn.Module):
    """MLP: [x̂ (n_target), contexto (n_context), H (n_target)] → logits (n_target).

    Retorna logits (sem sigmoid) — as perdas usam
    `binary_cross_entropy_with_logits`, numericamente estável.
    """

    def __init__(self, n_target: int, n_context: int, hidden: list[int]):
        super().__init__()
        dims = [2 * n_target + n_context] + list(hidden)
        camadas = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            camadas += [nn.Linear(d_in, d_out), nn.ReLU()]
        camadas += [nn.Linear(dims[-1], n_target)]
        self.net = nn.Sequential(*camadas)

    def forward(self, x_hat, context, hint):
        return self.net(torch.cat([x_hat, context, hint], dim=1))


# ---------------------------------------------------------------------------
# Mecanismo de hint
# ---------------------------------------------------------------------------

def generate_hint(m_eff: torch.Tensor, hint_rate: float,
                  gerador_torch: torch.Generator) -> torch.Tensor:
    """H = R ⊙ M_eff, com R ~ Bernoulli(hint_rate) por célula.

    Variante da implementação oficial de Yoon (github jsyoon0823/GAIN), não a
    do texto do paper (`R⊙M + 0,5(1−R)`). Na variante 0,5, células imputadas
    sorteadas com R=1 recebem hint 0 — o D é *informado* de que são imputadas
    e o termo adversarial do G nessas células torna-se invencível, dominando o
    gradiente (verificado empiricamente neste projeto: loss_G_adv ≈ 9 e RMSE
    de validação degradando). Com H = R⊙M_eff, hint 0 é ambíguo ("não sei" ou
    "imputada"), o jogo permanece jogável e o D não domina. Sem hint algum, o
    equilíbrio adversarial não identifica a distribuição correta (paper §3.3).
    """
    r = (torch.rand(m_eff.shape, generator=gerador_torch,
                    device=m_eff.device) < hint_rate).float()
    return r * m_eff


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def _amostrar_mascara_artificial(m: torch.Tensor, rates: torch.Tensor,
                                 gerador_torch: torch.Generator) -> torch.Tensor:
    """B ~ Bernoulli(1 − rate_j) por coluna, com B=0 forçado onde M=0.

    Réplica tensorial de `mask_utils.generate_artificial_mask` (mesma
    convenção; aqui em torch para viver dentro do loop de treino).
    """
    u = torch.rand(m.shape, generator=gerador_torch, device=m.device)
    b = (u >= rates.unsqueeze(0)).float()
    return b * m


def rmse_validacao(G: Generator, x_val: torch.Tensor, m_val: torch.Tensor,
                   ctx_val: torch.Tensor, b_val: torch.Tensor) -> float:
    """RMSE no espaço normalizado sobre as células (M=1, B=0) do val.

    `b_val` é fixa (gerada uma vez com seed) para que a curva de validação
    meça sempre o mesmo conjunto de células.
    """
    escondidas = (m_val == 1) & (b_val == 0)
    if escondidas.sum() == 0:
        return float("nan")
    x_imp = impute(G, x_val, m_val * b_val, ctx_val)
    err = (x_imp - x_val)[escondidas]
    return float(torch.sqrt((err ** 2).mean()))


def train_gain(x: torch.Tensor, m: torch.Tensor, context: torch.Tensor,
               hp: dict, miss_rates: np.ndarray,
               val_data: tuple | None = None, seed: int = 42,
               eval_every: int = 100) -> tuple[Generator, Discriminator, pd.DataFrame]:
    """Treina a GAIN. Retorna (G, D, training_log).

    x        : (N, n_target) float32, NaN já substituído por 0 (M marca onde).
    m        : (N, n_target) máscara real (1 = observado).
    context  : (N, n_context) features de condicionamento (sem NaN).
    hp       : hint_rate, alpha, batch_size, n_iter, lr_g, lr_d, hidden;
               opcional beta (default 0.0 — sem termo supervisionado).
    miss_rates: vetor (n_target,) com a taxa de máscara artificial por variável.
    val_data : opcional (x_val, m_val, ctx_val, b_val) para a curva de RMSE.

    Early stopping implícito: quando `val_data` é fornecido, o G devolvido é
    restaurado para o estado da iteração de MENOR rmse_val (o estado final
    não é o melhor em GANs — o equilíbrio adversarial oscila). O log permite
    reconstruir a curva completa; a coluna `rmse_val` identifica o ponto.
    """
    import copy

    torch.manual_seed(seed)
    g_torch = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    n, n_target = x.shape
    n_context = context.shape[1]
    rates = torch.tensor(miss_rates, dtype=torch.float32)
    beta = float(hp.get("beta", 0.0))

    G = Generator(n_target, n_context, hp["hidden"])
    D = Discriminator(n_target, n_context, hp["hidden"])
    opt_g = torch.optim.Adam(G.parameters(), lr=hp["lr_g"])
    opt_d = torch.optim.Adam(D.parameters(), lr=hp["lr_d"])

    log = []
    melhor_rmse, melhor_estado = float("inf"), None
    for it in range(1, hp["n_iter"] + 1):
        idx = rng.choice(n, size=min(hp["batch_size"], n), replace=False)
        xb, mb, cb = x[idx], m[idx], context[idx]

        # Máscara efetiva do batch: real ∧ artificial (protocolo da Etapa 2)
        b = _amostrar_mascara_artificial(mb, rates, g_torch)
        m_eff = mb * b

        z = (torch.rand(xb.shape, generator=g_torch) - 0.5) * 0.02  # U(-0,01, 0,01)
        x_tilde = m_eff * xb + (1.0 - m_eff) * z

        # --- passo do discriminador ---
        x_bar = G(x_tilde, cb, m_eff)
        x_hat = m_eff * xb + (1.0 - m_eff) * x_bar
        hint = generate_hint(m_eff, hp["hint_rate"], g_torch)

        logits_d = D(x_hat.detach(), cb, hint)
        loss_d = F.binary_cross_entropy_with_logits(logits_d, m_eff)
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        # --- passo do gerador ---
        logits_d = D(x_hat, cb, hint)
        # adversarial: enganar o D nas células imputadas (alvo 1 onde M_eff=0)
        loss_g_adv = F.binary_cross_entropy_with_logits(
            logits_d, torch.ones_like(logits_d), reduction="none")
        loss_g_adv = (loss_g_adv * (1.0 - m_eff)).sum() / (1.0 - m_eff).sum().clamp(min=1.0)
        # reconstrução: acertar as células mantidas
        loss_g_rec = ((x_bar - xb) ** 2 * m_eff).sum() / m_eff.sum().clamp(min=1.0)
        # supervisão: acertar as células escondidas artificialmente (M=1, B=0),
        # onde a verdade é conhecida — é o que o RMSE de validação mede
        sup = mb * (1.0 - m_eff)
        loss_g_sup = ((x_bar - xb) ** 2 * sup).sum() / sup.sum().clamp(min=1.0)
        loss_g = loss_g_adv + hp["alpha"] * loss_g_rec + beta * loss_g_sup
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if it % eval_every == 0 or it == 1:
            registro = {
                "iteracao": it,
                "loss_D": float(loss_d.detach()),
                "loss_G": float(loss_g.detach()),
                "loss_G_adv": float(loss_g_adv.detach()),
                "loss_G_rec": float(loss_g_rec.detach()),
                "loss_G_sup": float(loss_g_sup.detach()),
            }
            if val_data is not None:
                G.eval()
                with torch.no_grad():
                    registro["rmse_val"] = rmse_validacao(G, *val_data)
                G.train()
                if registro["rmse_val"] < melhor_rmse:
                    melhor_rmse = registro["rmse_val"]
                    melhor_estado = copy.deepcopy(G.state_dict())
            log.append(registro)

    if melhor_estado is not None:
        G.load_state_dict(melhor_estado)
        G.eval()

    return G, D, pd.DataFrame(log)


# ---------------------------------------------------------------------------
# Imputação
# ---------------------------------------------------------------------------

@torch.no_grad()
def impute(G: Generator, x: torch.Tensor, m: torch.Tensor,
           context: torch.Tensor) -> torch.Tensor:
    """X_imputado = M ⊙ X + (1 − M) ⊙ G(x̃, contexto, M).

    Determinístico (ruído zero nas células faltantes): na inferência
    queremos a estimativa central do gerador, não uma amostra.
    """
    x_tilde = m * x
    x_bar = G(x_tilde, context, m)
    return m * x + (1.0 - m) * x_bar
