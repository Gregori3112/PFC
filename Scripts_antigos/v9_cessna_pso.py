# ============================================================
# v8_cessna_pso.py – PSO otimização com histórico completo e gráficos de dispersão
# Compatível com v7_cessna_opt.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import importlib
import pandas as pd
import seaborn as sns

# --- Recarrega a função objetivo ---
import v9_cessna_opt
importlib.reload(v9_cessna_opt)
from v9_cessna_opt import FCN


# ============================================================
# --- Configuração inicial ---
# ============================================================

output_dir = "resultados_graficos"
os.makedirs(output_dir, exist_ok=True)

# AR, span, taper, sweep, twist, alpha = x
xmin = np.array([5.0, 8.0, 0.3, 0.0, -3.0, -1.0])      # limites mínimos
xmax = np.array([12.0, 15.0, 1.0, 20.0, 3.0, 4.0])     # máximos

nrvar = len(xmin)
lambda1, lambda2, omega = 2.0, 2.0, 0.6
pop = 12
tol = 1e-3
itermax = 10
np.random.seed(2)

# PSO parameters
w = omega
c1 = lambda1
c2 = lambda2

nvar = nrvar
gbest = float('inf')
xgbest = np.zeros(nrvar)
gbest_history = []

# --- Inicialização das partículas ---
v = np.zeros((pop, nrvar))
x = np.zeros((pop, nrvar))
pbest = np.zeros((pop, nrvar))
pbestval = np.full(pop, np.inf)

for i in range(pop):
    for j in range(nrvar):
        x[i, j] = xmin[j] + (xmax[j] - xmin[j]) * random.random()
    y = FCN(x[i, :])
    pbestval[i] = y
    pbest[i, :] = x[i, :]
    if y < gbest:
        gbest = y
        xgbest = x[i, :].copy()

gbest_history = [gbest]
# AR, span, taper, sweep, twist, alpha = x
# --- Histórico global e por partícula ---
history = {
    "AR": [],
    "span": [],
    "taper": [],
    "sweep": [],
    "twist": [],
    "alpha": []
}
history_particles = {
    "AR": [],
    "span": [],
    "taper": [],
    "sweep": [],
    "twist": [],
    "alpha": []
}

# ============================================================
# --- Loop principal do PSO ---
# ============================================================

for k in range(1, itermax + 1):
    print(f"\n===== Iteração {k}/{itermax} =====")
    # AR, span, taper, sweep, twist, alpha = x
    AR_iter, span_iter, taper_iter, sweep_iter, twist_iter, alpha_iter = [], [], [], [], [], []

    for i in range(pop):
        # Atualização da velocidade e posição
        r1, r2 = np.random.rand(nvar), np.random.rand(nvar)
        v[i, :] = w * v[i, :] + c1 * r1 * (pbest[i, :] - x[i, :]) + c2 * r2 * (xgbest - x[i, :])
        x[i, :] = np.clip(x[i, :] + v[i, :], xmin, xmax)

#        try:
        ynew = FCN(x[i, :])
#        except Exception as e:
#            print(f"[erro] Partícula {i}: {e}")
#            ynew = 1e6

        ld_value = -ynew if ynew != 1e6 else np.nan
        history["L_over_D"].append(ld_value)

        # AR, span, taper, sweep, twist, alpha = x
        AR_iter.append(x[i, 0])
        span_iter.append(x[i, 1])
        taper_iter.append(x[i, 2])
        sweep_iter.append(x[i, 3])
        twist_iter.append(x[i, 1])
        alpha_iter.append(x[i, 3])

        # Logs
        if np.isfinite(ynew):
            print(f"[iter {k} | part {i}] y={ynew:.3f} | AR={x[i,0]:.2f} | span={x[i,1]:.2f} | taper={x[i,2]:.2f} | sweep={x[i,3]:.2f} | twist={x[i,4]:.2f} | alpha={x[i,5]:.2f}")
        else:
            print(f"[iter {k} | part {i}] penalizada.")

        # Atualiza pbest e gbest
        if ynew < pbestval[i]:
            pbestval[i] = ynew
            pbest[i, :] = x[i, :].copy()
        if ynew < gbest:
            gbest = ynew
            xgbest = x[i, :].copy()

    # Salva variáveis da iteração
    history["AR"].append(xgbest[0])
    history["span"].append(xgbest[1])
    history["taper"].append(xgbest[2])
    history["sweep"].append(xgbest[3])
    history["twist"].append(xgbest[1])
    history["alpha"].append(xgbest[3])
    L_over_D = FCN(xgbest, return_LD_only=True)
    gbest_history.append(L_over_D)

    # Salva histórico de partículas da iteração
    history_particles["AR"].append(AR_iter)
    history_particles["span"].append(span_iter)
    history_particles["taper"].append(taper_iter)
    history_particles["sweep"].append(sweep_iter)
    history_particles["twist"].append(twist_iter)
    history_particles["alpha"].append(alpha_iter)

    print(f"[iter {k}] Melhor L/D até agora: {L_over_D:.3f}")

#    if k > 5:
#        recent_mean = np.mean(gbest_history[-5:])
#        if abs(gbest_history[-1] - recent_mean) < tol:
#            print("[stop] Convergência detectada.")
#            break

# ============================================================
# --- Resultados finais ---
# ============================================================

L_over_D_final = FCN(xgbest, return_LD_only=True)
print("\n=== RESULTADOS FINAIS ===")
print(f"Iterações executadas: {k}")
print(f"Melhor L/D = {L_over_D_final:.4f}")
print(f"AR  = {xgbest[0]:.3f} °")
print(f"span  = {xgbest[1]:.3f} °")
print(f"taper  = {xgbest[2]:.3f} ft")
print(f"sweep  = {xgbest[3]:.3f} °")
print(f"Twist  = {xgbest[1]:.3f} °")
print(f"Alpha  = {xgbest[3]:.3f} °")
print("=========================\n")


# ============================================================
# --- Gráficos principais ---
# ============================================================

# Convergência do L/D
plt.figure(figsize=(7, 4))
plt.plot(gbest_history, 'b*-')
plt.xlabel("Iteração")
plt.ylabel("Melhor L/D")
plt.title("Convergência do L/D (gbest)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "convergencia_gbest.png"), dpi=300, bbox_inches="tight")

# Evolução das variáveis (gbest)
plt.figure(figsize=(9, 8))
vars_labels = ["AR", "span", "taper", "sweep", "twist", "alpha"]
colors = ['r', 'm', 'c', 'g']
var_matrix = np.array([
    history["AR"],
    history["span"],
    history["taper"],
    history["sweep"],
    history["twist"],
    history["alpha"]
]).T
for i in range(var_matrix.shape[1]):
    plt.subplot(var_matrix.shape[1], 1, i + 1)
    plt.plot(var_matrix[:, i], color=colors[i], linewidth=1.4)
    plt.ylabel(vars_labels[i])
    plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel("Iteração")
plt.suptitle("Evolução das variáveis ótimas por iteração", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "evolucao_completa_variaveis.png"), dpi=300, bbox_inches="tight")

# ============================================================
# --- Gráficos de dispersão (todas as partículas)
# ============================================================

for key, label, color in zip(["AR", "span", "taper", "sweep", "twist", "alpha"],
                             ["AR", "span", "taper", "sweep", "twist", "alpha"],
                             ['r', 'm', 'c', 'g', 'b', 'y']):
    plt.figure(figsize=(8, 4))
    for it, vals in enumerate(history_particles[key]):
        plt.scatter([it + 1] * len(vals), vals, color=color, alpha=0.5, s=40)
    plt.plot(range(1, len(history[key]) + 1), history[key], 'k-', lw=1.3, label="Melhor (gbest)")
    plt.xlabel("Iteração")
    plt.ylabel(label)
    plt.title(f"Evolução populacional da variável {label}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dispersao_{key}.png"), dpi=300, bbox_inches="tight")

print("\n✅ Todos os gráficos foram gerados e salvos em 'resultados_graficos/'")

