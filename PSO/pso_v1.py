# ============================================================
# PSO cru - versão Python equivalente ao código MATLAB (com linhas)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import random
from fcn import FCN

# --- Parâmetros principais ---
xmin = np.array([-10, -10])
xmax = np.array([10, 10])
nrvar = len(xmin)
lambda1 = 2.02
lambda2 = 2.02
omega = 0.4
pop = 5
tol = 1e-7
itermax = 50
np.random.seed(2)

# --- Malha para plotar a função ---
x1 = np.arange(xmin[0], xmax[0] + 0.2, 0.2)
x2 = np.arange(xmin[1], xmax[1] + 0.2, 0.2)
X1, X2 = np.meshgrid(x1, x2)

Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = FCN([X1[i, j], X2[i, j]])

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax1.view_init(45, 140)

# --- Inicialização das partículas ---
gbest = [1e30]
k = 1
v = np.zeros((pop, nrvar))
x = np.zeros((pop, nrvar))
lbest = np.zeros(pop)
xlbest = np.zeros((pop, nrvar))

# Histórico de posições para traçar as linhas
trajetorias = [[] for _ in range(pop)]

for i in range(pop):
    for j in range(nrvar):
        x[i, j] = xmin[j] + (xmax[j] - xmin[j]) * random.random()

    y = FCN(x[i, :])
    lbest[i] = y
    xlbest[i, :] = x[i, :]

    trajetorias[i].append(x[i, :].copy())

    if y < gbest[k - 1]:
        gbest[k - 1] = y
        xgbest = x[i, :].copy()

# --- Figura 1: melhor indivíduo inicial ---
ax1.scatter(xgbest[0], xgbest[1], gbest[k - 1], color='r', s=80)
plt.pause(0.1)

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

        ynew = FCN(x[i, :])
        trajetorias[i].append(x[i, :].copy())  # salva para plotar trajetória

        # Figura 4 - percurso das partículas (com linha)
        fig4 = plt.figure(4)
        if i in [0, 1, 2]:
            cor = ['b', 'r', 'g'][i]
            traj = np.array(trajetorias[i])
            plt.plot(traj[:, 0], traj[:, 1], color=cor, linewidth=1.5, label=f"Partícula {i+1}" if k == 2 else "")
            plt.axis([xmin[0], xmax[0], xmin[1], xmax[1]])
            plt.axis('equal')

        if ynew < lbest[i]:
            lbest[i] = ynew
            xlbest[i, :] = x[i, :]

        if ynew < gbest[k - 1]:
            gbest[k - 1] = ynew
            xgbest = x[i, :].copy()

    # Figura 3 - progresso do melhor resultado (com linha)
    fig3 = plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, k + 1), gbest[1:k + 1], 'b-', linewidth=1.5)
    plt.xlim([0, itermax])
    plt.ylabel("gbest")
    plt.title("Convergência do melhor valor")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(2, 1, 2)
    plt.plot(xgbest[0], xgbest[1], 'bo', markersize=4)
    plt.xlim([xmin[0], xmax[0]])
    plt.ylim([xmin[1], xmax[1]])
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Melhor posição")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.pause(0.001)

    if gbest[k - 1] < gbest[k - 2]:
        # Atualiza na figura 1
        ax1.scatter(xgbest[0], xgbest[1], gbest[k - 1], color='r', s=50)
        plt.pause(0.001)

    if k >= itermax:
        flag = True

    if k > 11:
        norm = np.sum(gbest[k - 9:k - 5]) - np.sum(gbest[k - 4:k])
        if norm < tol:
            pass

    k += 1

# --- Resultados ---
print("\n=== RESULTADOS ===")
print(f"Iterações executadas: {k - 1}")
if 'norm' in locals():
    print(f"norm = {norm}")
print(f"Melhor valor final (gbest): {gbest[-1]:.6f}")
print(f"Melhor posição encontrada: {xgbest}")

plt.legend()
plt.show()
