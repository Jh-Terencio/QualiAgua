# Diagnóstico do modelo final — gain_best

*Gerado por `03_diagnostico.ipynb` em 2026-07-07. Modelo: `cfg_1092`
(hint_rate=0.9, alpha=10, beta=10,
batch=128, hidden=[128, 128], lr_g=0.001, lr_d=0.001),
seed 42, melhor iteração 1000.*

## 1. Convergência: OK

Retreino determinístico reproduz o checkpoint (diff < 1e-6).
Últimos 20% do treino: CV(loss_D) = 0.090, CV(loss_G) = 0.039,
loss_D média = 0.145 (sem colapso do jogo adversarial: D não domina).
Melhor RMSE_val = 0.2355 na iteração 1000;
o early stopping implícito do `train_gain` é quem garante o uso desse ponto.

## 2. Mode collapse: não detectado

- Diversidade entre amostras: mediana de sigma_imp/sigma_obs = 0.482;
  mediana da correlação real-imputado = 0.682.
- Sensibilidade ao ruído Z (K=50): sigma médio por célula = 0.0050
  (≈ 0 é o esperado — o gerador é treinado como imputador determinístico).

## 3. Variáveis problemáticas: Condutividade, Turbidez, Sólidos Suspensos Totais

Critério: |viés mediano| > 0,10 no espaço normalizado com n ≥ 5 células.
Menores correlações real-imputado: Nitrato (r=-1.00), Coliformes (r=0.43), N-Amoniacal (r=0.53).

## 4. Estações problemáticas: nenhuma

Critério: |erro| mediano > 1,5× o |erro| mediano global (0.130).

## 5. Veredicto: **AVANÇAR para a Etapa 5 (05_Evaluation)**

Critério de aceite da etapa já atendido no notebook 2 (RMSE_norm 0.2317
vs 0.2742 do KNN no val). Este diagnóstico
não encontrou patologia que invalide o resultado.
