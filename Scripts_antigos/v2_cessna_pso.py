import os
import numpy as np
import matplotlib.pyplot as plt
import random
from fcn import FCN
from v2_cessna_opt import FCN


# ============================================================
# --- Configuração inicial ---
# ============================================================

# Cria a pasta para salvar gráficos, se não existir
output_dir = "resultados_graficos"
os.makedirs(output_dir, exist_ok=True)

# --- Parâmetros principais ---
xmin = np.array([-4.0, 0.0, -6.0, 0.3, 26.25])   # [α, sweep, twist, taper, span] - mínimo
xmax = np.array([ 4.0, 30.0, 6.0, 1.2, 45.93])  # máximo

nrvar = len(xmin)
lambda1 = 2.02
lambda2 = 2.02
omega = 0.4
pop = 3  # 10+2*nrvar
tol = 1e-3
itermax = 20
np.random.seed(2)

# --- Inicialização das partículas ---
gbest = [1e30]
k = 1
v = np.zeros((pop, nrvar))
x = np.zeros((pop, nrvar))
lbest = np.zeros(pop)
xlbest = np.zeros((pop, nrvar))

for i in range(pop):
    for j in range(nrvar):
        x[i, j] = xmin[j] + (xmax[j] - xmin[j]) * random.random()

    y = FCN(x[i, :])
    lbest[i] = y
    xlbest[i, :] = x[i, :]

    if y < gbest[k - 1]:
        gbest[k - 1] = y
        xgbest = x[i, :].copy()

flag = False
k = 2
gbest.append(gbest[0])

# --- Histórico ---
history = {
    "L_over_D": [],
    "alpha": [],
    "sweep": [],
    "twist": [],
    "taper": [],
    "span": [],
}

# ============================================================
# --- Loop principal do PSO ---
# ============================================================

while not flag:
    gbest.append(gbest[k - 2])
    for i in range(pop):
        for j in range(nrvar):
            r1 = random.random()
            r2 = random.random()
            vnew_ij = (omega * v[i, j] +
                       lambda1 * r1 * (xlbest[i, j] - x[i, j]) +
                       lambda2 * r2 * (xgbest[j] - x[i, j]))
            xnew_ij = x[i, j] + vnew_ij

            # Restrições de limites
            if xnew_ij < xmin[j]:
                xnew_ij = xmin[j]
            elif xnew_ij > xmax[j]:
                xnew_ij = xmax[j]

            v[i, j] = vnew_ij
            x[i, j] = xnew_ij

        ynew = FCN(x[i, :])

        if ynew < lbest[i]:
            lbest[i] = ynew
            xlbest[i, :] = x[i, :]

        if ynew < gbest[k - 1]:
            gbest[k - 1] = ynew
            xgbest = x[i, :].copy()

            # Armazena histórico do melhor resultado atual
            L_over_D = FCN(xgbest, return_LD_only=True)
            history["L_over_D"].append(L_over_D)
            history["alpha"].append(xgbest[0])
            history["sweep"].append(xgbest[1])
            history["twist"].append(xgbest[2])
            history["taper"].append(xgbest[3])
            history["span"].append(xgbest[4])

    # --- Figura - convergência do melhor resultado global ---
    plt.figure(3)
    plt.plot(k, gbest[k - 1], 'b*')
    plt.xlim([0, itermax])
    plt.xlabel("Iteração")
    plt.ylabel("Melhor valor global (gbest)")
    plt.title("Convergência do Melhor Resultado (gbest)")
    plt.pause(0.001)

    # Salva o gráfico de convergência
    plt.savefig(os.path.join(output_dir, "convergencia_gbest.png"), dpi=300, bbox_inches="tight")

    if k >= itermax:
        flag = True

    if k > 11:
        norm = np.sum(gbest[k - 9:k - 5]) - np.sum(gbest[k - 4:k])
        if norm < tol:
            flag = True

    k += 1

# ============================================================
# --- Resultados finais ---
# ============================================================

L_over_D_final = FCN(xgbest, return_LD_only=True)

print("\n=== RESULTADOS FINAIS ===")
print(f"Iterações executadas: {k - 1}")
print(f"Melhor L/D = {L_over_D_final:.4f}")
print("\n--- Variáveis ótimas ---")
print(f"Alpha  = {xgbest[0]:.3f} °")
print(f"Sweep  = {xgbest[1]:.3f} °")
print(f"Twist  = {xgbest[2]:.3f} °")
print(f"Taper  = {xgbest[3]:.3f}")
print(f"Span   = {xgbest[4]:.3f} ft")
print("=========================\n")

# ============================================================
# --- Gráficos de evolução e salvamento automático ---
# ============================================================

plt.figure(figsize=(10, 10))
plt.subplot(6, 1, 1)
plt.plot(history["L_over_D"], 'b')
plt.ylabel("L/D")
plt.title("Evolução das Variáveis por Iteração")

plt.subplot(6, 1, 2)
plt.plot(history["alpha"], 'g')
plt.ylabel("Alpha [deg]")

plt.subplot(6, 1, 3)
plt.plot(history["sweep"], 'r')
plt.ylabel("Sweep [deg]")

plt.subplot(6, 1, 4)
plt.plot(history["twist"], 'm')
plt.ylabel("Twist [deg]")

plt.subplot(6, 1, 5)
plt.plot(history["taper"], 'c')
plt.ylabel("Taper Ratio")

plt.subplot(6, 1, 6)
plt.plot(history["span"], 'k')
plt.ylabel("Span [ft]")
plt.xlabel("Iteration")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "evolucao_variaveis.png"), dpi=300, bbox_inches="tight")

# --- Salva gráficos individuais ---
for key, color, label, name in [
    ("L_over_D", 'b', "L/D", "LD"),
    ("alpha", 'g', "Alpha [deg]", "alpha"),
    ("sweep", 'r', "Sweep [deg]", "sweep"),
    ("twist", 'm', "Twist [deg]", "twist"),
    ("taper", 'c', "Taper Ratio", "taper"),
    ("span", 'k', "Span [ft]", "span"),
]:
    plt.figure()
    plt.plot(history[key], color)
    plt.xlabel("Iteração")
    plt.ylabel(label)
    plt.title(f"Evolução de {label}")
    plt.savefig(os.path.join(output_dir, f"evolucao_{name}.png"), dpi=300, bbox_inches="tight")

print(f"\n✅ Todos os gráficos foram salvos na pasta '{output_dir}/'")
