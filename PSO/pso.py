# ============================================================
# PSO cru - versão Python equivalente ao código MATLAB
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
random.seed(2)

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
ax1.plot_wireframe(X1, X2, Z, color='gray')
ax1.view_init(45, 140)

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
    lbest[i] = y            # guarda o melhor valor 
    xlbest[i, :] = x[i, :]  # guarda a melhor posição que apresentou o melhor valor

    if y < gbest[k - 1]:
        gbest[k - 1] = y         # melhor valor global
        xgbest = x[i, :].copy()  # melhor posição que apresentou o melhor valor

# --- Figura 1: melhor indivíduo inicial ---
ax1.scatter(xgbest[0], xgbest[1], gbest[k - 1], color='r', s=80)
plt.pause(0.1)

# --- Loop principal ---
flag = False
k = 2
gbest.append(gbest[0])  # adiciona um novo elemento copiando o valor inicial para iniciar o histórico de gbest (equivalente ao crescimento automático do MATLAB)

# --- Loop principal: continua enquanto o critério de parada não for atingido ---
while not flag:
    gbest.append(gbest[k - 2])  # Copia o melhor valor global anterior (equivalente ao gbest(k) = gbest(k-1) do MATLAB)


    for i in range(pop): # Percorre cada variável de decisão da partícula (ex: x e y)
        for j in range(nrvar):

            # Sorteia dois números aleatórios entre 0 e 1 (usados nos termos cognitivo e social)
            r1 = random.random()
            r2 = random.random()

            # Calcula a nova velocidade com base na inércia, melhor posição individual e global
            vnew_ij = (omega * v[i, j] +
                       lambda1 * r1 * (xlbest[i, j] - x[i, j]) +
                       lambda2 * r2 * (xgbest[j] - x[i, j]))
            
            # Atualiza a posição da partícula somando a nova velocidade
            xnew_ij = x[i, j] + vnew_ij

            # Garante que a nova posição está dentro dos limites definidos
            if xnew_ij < xmin[j]:
                xnew_ij = xmin[j]
            elif xnew_ij > xmax[j]:
                xnew_ij = xmax[j]

            # Atualiza a matriz principal com as novas posições e velocidades
            v[i, j] = vnew_ij
            x[i, j] = xnew_ij

        # Calcula o novo valor da função objetivo para a partícula atual
        ynew = FCN(x[i, :])

    # Figura 4 - percurso das partículas (trilhas contínuas, igual ao MATLAB)
    fig4 = plt.figure(4)

    if i in [0, 1, 2]:
        # Inicializa as listas de trajetória na primeira iteração
        if k == 2 and i == 0:
            traj = {0: [], 1: [], 2: []}

        # Armazena a posição atual (x, y) da partícula
        traj[i].append((x[i, 0], x[i, 1]))

        # Extrai as coordenadas acumuladas da trajetória
        xs, ys = zip(*traj[i])

        # Define a cor de cada partícula (azul, vermelho, verde)
        colors = ['b', 'r', 'g']

        # Plota a trilha acumulada da partícula
        plt.plot(xs, ys, colors[i])
        plt.axis([xmin[0], xmax[0], xmin[1], xmax[1]])
        plt.axis('equal')
        plt.pause(0.001)


        # Atualiza o melhor valor individual (local)
        if ynew < lbest[i]:
            lbest[i] = ynew
            xlbest[i, :] = x[i, :]

        # Atualiza o melhor valor global (do grupo inteiro)
        if ynew < gbest[k - 1]:
            gbest[k - 1] = ynew
            xgbest = x[i, :].copy()

    # --- Guarda histórico real de gbest ---
    if k == 2:
        gbest_history = []
    gbest_history.append(gbest[k - 1])

    # Figura 3 - progresso do melhor resultado (com linhas suaves)
    fig3 = plt.figure(3, figsize=(7, 5))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(gbest_history) + 1), gbest_history, 'b-', linewidth=1.5)
    plt.xlim([0, itermax])
    plt.ylabel("Melhor valor (gbest)")
    plt.title("Convergência do PSO")
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- Subplot 2: trajetória da melhor posição ---
    if k == 2:
        best_positions = []
    best_positions.append(xgbest.copy())

    plt.subplot(2, 1, 2)
    pos = np.array(best_positions)
    plt.plot(pos[:, 0], pos[:, 1], 'bo-', linewidth=1.2, markersize=3)
    plt.xlim([xmin[0], xmax[0]])
    plt.ylim([xmin[1], xmax[1]])
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Evolução da melhor posição (x₁, x₂)")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
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
            # flag = True  # igual ao MATLAB: pode ser comentado
            pass

    k += 1

# --- Resultados ---
print("k =", k - 1)
if 'norm' in locals():
    print("norm =", norm)
print("gbest =", gbest[-1])
print("xgbest =", xgbest)

plt.show()
