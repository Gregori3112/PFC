# ============================================================
# cessna_pso.py - versão 4 variáveis (Sweep, Twist, Taper, Span)
# compatível com cessna_opt.py (OpenVSP 3.45.2)
# ============================================================

import matplotlib
import os

# --- Seleciona backend automático ---
if "JPY_PARENT_PID" in os.environ:
    matplotlib.use("inline")
elif "SPYDER" in os.environ or "spyder" in os.environ.get("CONDA_DEFAULT_ENV", ""):
    matplotlib.use("Qt5Agg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import random
from cessna_opt import FCN  # função que roda VSPAERO e retorna -L/D

# ============================================================
# PARÂMETROS DO PSO
# ============================================================

# sweep [°], twist [°], taper ratio, span [m]
xmin = np.array([0.0, -6.0, 0.3, 8.0])
xmax = np.array([25.0, 6.0, 1.5, 14.0])

nrvar = len(xmin)
lambda1 = 2.02
lambda2 = 2.02
omega = 0.4
pop = 4
tol = 1e-5
itermax = 50
np.random.seed(2)

# ============================================================
# INICIALIZAÇÃO
# ============================================================

gbest = 1e30  # Valor escalar, não lista
k = 1
v = np.zeros((pop, nrvar))
x = np.zeros((pop, nrvar))
lbest = np.zeros(pop)
xlbest = np.zeros((pop, nrvar))

# Histórico
history_ld = []
history_params = [[] for _ in range(nrvar)]
traj = [[] for _ in range(nrvar)]

# ============================================================
# FASE INICIAL - população
# ============================================================

for i in range(pop):
    for j in range(nrvar):
        x[i, j] = xmin[j] + (xmax[j] - xmin[j]) * random.random()

    y = FCN(x[i, :])  # retorna -L/D
    lbest[i] = y
    xlbest[i, :] = x[i, :]

    if y < gbest:
        gbest = y
        xgbest = x[i, :].copy()

print("\n[Inicialização completa]")
print(f"Melhor inicial: Sweep={xgbest[0]:.2f}°, Twist={xgbest[1]:.2f}°, "
      f"Taper={xgbest[2]:.3f}, Span={xgbest[3]:.2f} m, L/D={-gbest:.3f}")

# ============================================================
# LOOP PRINCIPAL DO PSO
# ============================================================

flag = False
k = 2
gbest_history = [gbest]  # Histórico para convergência

while not flag:
    gbest_history.append(gbest)

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

        if ynew < gbest:
            gbest = ynew
            xgbest = x[i, :].copy()

        for j in range(nrvar):
            traj[j].append(x[i, j])

    # Salva histórico
    history_ld.append(-gbest)
    for j in range(nrvar):
        history_params[j].append(xgbest[j])

    print(f"[Iter {k}] Melhor até agora: Sweep={xgbest[0]:.2f}°, Twist={xgbest[1]:.2f}°, "
          f"Taper={xgbest[2]:.3f}, Span={xgbest[3]:.2f} m, L/D={-gbest:.3f}")

    # Critérios de parada
    if k >= itermax:
        flag = True
    elif k > 10:  # Verifica convergência após 10 iterações
        # Verifica se a melhoria nas últimas 5 iterações é menor que a tolerância
        if len(gbest_history) >= 5:
            recent_improvement = gbest_history[-5] - gbest_history[-1]
            if abs(recent_improvement) < tol:
                print("[convergência] Critério de parada atingido.")
                flag = True

    k += 1

# ============================================================
# RESULTADOS FINAIS
# ============================================================

print("\n=== RESULTADO FINAL ===")
print(f"Iterações = {k-1}")
print(f"Sweep ótimo = {xgbest[0]:.3f}°")
print(f"Twist ótimo = {xgbest[1]:.3f}°")
print(f"Taper ótimo = {xgbest[2]:.3f}")
print(f"Span ótimo  = {xgbest[3]:.3f} m")
ld_max = -gbest
print(f"L/D máximo  = {ld_max:.3f}")

# Salva resultados em arquivo
span_ft = xgbest[3] * 3.28084  # Conversão m -> ft
with open("pso_resultados.txt", "w") as f:
    f.write("=== RESULTADOS PSO ===\n")
    f.write(f"Iterações: {k-1}\n")
    f.write(f"Sweep ótimo: {xgbest[0]:.3f}°\n")
    f.write(f"Twist ótimo: {xgbest[1]:.3f}°\n")
    f.write(f"Taper ótimo: {xgbest[2]:.3f}\n")
    f.write(f"Span ótimo: {xgbest[3]:.3f} m ({span_ft:.3f} ft)\n")
    f.write(f"L/D máximo: {ld_max:.3f}\n")
    f.write("\n=== PARA USAR NO GUI OPENVSP ===\n")
    f.write(f"Sweep: {xgbest[0]:.3f}°\n")
    f.write(f"Twist: {xgbest[1]:.3f}°\n")
    f.write(f"Taper: {xgbest[2]:.3f}\n")
    f.write(f"Span: {span_ft:.3f} ft\n")
print("OK - Resultados salvos em 'pso_resultados.txt'")

# ============================================================
# PLOTS FINAIS
# ============================================================

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# 1️⃣ Convergência L/D
axs[0, 0].plot(range(1, len(history_ld)+1), history_ld, 'b-o', linewidth=1.8)
axs[0, 0].set_title("Convergência do PSO - Melhor L/D", fontsize=11)
axs[0, 0].set_xlabel("Iteração")
axs[0, 0].set_ylabel("Melhor L/D")
axs[0, 0].grid(True)

# 2️⃣ Sweep × Twist
axs[0, 1].scatter(traj[0], traj[1], color='green', s=25, alpha=0.8)
axs[0, 1].set_xlabel("Sweep [°]")
axs[0, 1].set_ylabel("Twist [°]")
axs[0, 1].set_title("Trajetória das partículas (Sweep × Twist)", fontsize=10)
axs[0, 1].grid(True)

# 3️⃣ Sweep × Span
axs[1, 0].scatter(traj[0], traj[3], color='purple', s=25, alpha=0.8)
axs[1, 0].set_xlabel("Sweep [°]")
axs[1, 0].set_ylabel("Span [m]")
axs[1, 0].set_title("Trajetória (Sweep × Span)", fontsize=10)
axs[1, 0].grid(True)

# 4️⃣ Taper × Span
axs[1, 1].scatter(traj[2], traj[3], color='orange', s=25, alpha=0.8)
axs[1, 1].set_xlabel("Taper Ratio")
axs[1, 1].set_ylabel("Span [m]")
axs[1, 1].set_title("Trajetória (Taper × Span)", fontsize=10)
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig("pso_resultados_multiplos.png", dpi=300)
plt.pause(0.001)
plt.show()

print("OK - Grafico salvo como 'pso_resultados_multiplos.png'")
