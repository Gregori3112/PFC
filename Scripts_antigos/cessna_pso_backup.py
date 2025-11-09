# ============================================================
# cessna_pso.py - versão compatível com Cursor, Spyder e terminal
# ============================================================
import matplotlib
import os

# --- Seleciona automaticamente o backend conforme o ambiente ---
if "JPY_PARENT_PID" in os.environ:
    # Jupyter / Cursor / VSCode
    matplotlib.use("inline")
elif "SPYDER" in os.environ or "spyder" in os.environ.get("CONDA_DEFAULT_ENV", ""):
    # Spyder IDE
    matplotlib.use("Qt5Agg")
else:
    # Ambiente sem GUI
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import random
from cessna_opt import FCN  # função que roda VSPAERO e retorna -L/D

# --- Parâmetros principais ---
xmin = np.array([0.0, -6.0])     # sweep [°], twist [°]
xmax = np.array([25.0, 6.0])
nrvar = len(xmin)
lambda1 = 2.02
lambda2 = 2.02
omega = 0.4
pop = 4
tol = 1e-5
itermax = 10
np.random.seed(2)

# --- Inicialização das partículas ---
gbest = [1e30]
k = 1
v = np.zeros((pop, nrvar))
x = np.zeros((pop, nrvar))
lbest = np.zeros(pop)
xlbest = np.zeros((pop, nrvar))

# --- Dados para plot final ---
history_ld = []
history_sweep = []
history_twist = []
traj_x, traj_y = [], []

# --- Inicialização ---
for i in range(pop):
    for j in range(nrvar):
        x[i, j] = xmin[j] + (xmax[j] - xmin[j]) * random.random()

    y = FCN(x[i, :])  # retorna -L/D
    lbest[i] = y
    xlbest[i, :] = x[i, :]

    if y < gbest[k - 1]:
        gbest[k - 1] = y
        xgbest = x[i, :].copy()

print("\n[Inicialização completa]")
print(f"Melhor inicial: Sweep={xgbest[0]:.2f}°, Twist={xgbest[1]:.2f}°, L/D={-gbest[k-1]:.3f}")

# --- Loop principal ---
flag = False
k = 2
gbest.append(gbest[0])

while not flag:
    gbest.append(gbest[k - 2])

    for i in range(pop):
        for j in range(nrvar):
            r1 = random.random()
            r2 = random.random()
            vnew_ij = (omega * v[i, j] +
                       lambda1 * r1 * (xlbest[i, j] - x[i, j]) +
                       lambda2 * r2 * (xgbest[j] - x[i, j]))
            xnew_ij = np.clip(x[i, j] + vnew_ij, xmin[j], xmax[j])
            v[i, j] = vnew_ij
            x[i, j] = xnew_ij

        ynew = FCN(x[i, :])  # retorna -L/D

        # Atualiza partículas
        if ynew < lbest[i]:
            lbest[i] = ynew
            xlbest[i, :] = x[i, :]

        if ynew < gbest[k - 1]:
            gbest[k - 1] = ynew
            xgbest = x[i, :].copy()

        traj_x.append(x[i, 0])
        traj_y.append(x[i, 1])

    history_ld.append(-gbest[k - 1])
    history_sweep.append(xgbest[0])
    history_twist.append(xgbest[1])

    print(f"[Iter {k}] Melhor até agora: Sweep={xgbest[0]:.2f}°, Twist={xgbest[1]:.2f}°, L/D={-gbest[k-1]:.3f}")

    # Critérios de parada
    if k >= itermax:
        flag = True
    elif k > 11:
        norm = np.sum(gbest[k - 9:k - 5]) - np.sum(gbest[k - 4:k])
        if abs(norm) < tol:
            print("[convergência] Critério de parada atingido.")
            flag = True

    k += 1

# --- Resultados finais ---
print("\n=== RESULTADO FINAL ===")
print(f"Iterações = {k-1}")
print(f"Sweep ótimo = {xgbest[0]:.3f}°")
print(f"Twist ótimo = {xgbest[1]:.3f}°")
ld_max = -min(gbest)
print(f"L/D máximo = {ld_max:.3f}")

# --- Plot final: convergência e trajetória ---

fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# Gráfico 1 - Convergência L/D
axs[0].plot(range(1, len(history_ld)+1), history_ld, 'b-o', linewidth=1.5)
axs[0].set_title("Convergência do PSO - Melhor L/D")
axs[0].set_xlabel("Iteração")
axs[0].set_ylabel("Melhor L/D")
axs[0].grid(True)

# Gráfico 2 - Parâmetros ótimos
axs[1].plot(range(1, len(history_sweep)+1), history_sweep, 'r--', label="Sweep ótimo")
axs[1].plot(range(1, len(history_twist)+1), history_twist, 'g--', label="Twist ótimo")
axs[1].set_xlabel("Iteração")
axs[1].set_ylabel("Ângulo [°]")
axs[1].legend()
axs[1].grid(True)

# Gráfico 3 - Trajetória das partículas
axs[2].scatter(traj_x, traj_y, color='blue', s=10)
axs[2].set_xlabel("Sweep [°]")
axs[2].set_ylabel("Twist [°]")
axs[2].set_title("Trajetória das partículas")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Salva os gráficos
plt.savefig("pso_convergencia_final.png", dpi=300)
print("✅ Gráfico salvo como 'pso_convergencia_final.png'")
