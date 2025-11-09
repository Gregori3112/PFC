# ============================================================
# PSO cru - versão Python equivalente ao código MATLAB
# ------------------------------------------------------------
# Este script implementa o algoritmo Particle Swarm Optimization (PSO)
# aplicado à otimização aerodinâmica de uma asa no OpenVSP.
# Cada partícula representa uma combinação de parâmetros geométricos da asa.
# O objetivo é maximizar L/D (minimizando -L/D), chamando a função FCN()
# que executa a simulação via VSPAERO.
# ============================================================


import numpy as np
import matplotlib.pyplot as plt
import random
import os
from v10_cessna_opt import FCN       # Função objetivo (roda o OpenVSP)
import importlib, openvsp, time      # Usados para recarregar e limpar a engine do OpenVSP


# ============================================================
# 1 Configuração inicial
# ============================================================

# Nomes das variáveis e cores associadas para gráficos
var_names = ["AR", "span", "taper", "sweep", "twist", "alpha"]
colors = ['red', 'orange', 'cyan', 'magenta', 'purple', 'green']

# ============================================================
# 2 Inicialização dos históricos e diretórios de saída
# ============================================================

# Este bloco cria (ou reaproveita, se já existir) as variáveis
# que armazenam o histórico da otimização.
# A verificação com 'not in locals()' evita recriar as variáveis
# caso o script seja executado várias vezes no mesmo ambiente (ex: Spyder, Jupyter).

# Histórico da melhor solução global ao longo das iterações
# (guarda a evolução do "gbest" — o melhor valor encontrado pela população)
if 'xgbest_history' not in locals():
    xgbest_history = []     # inicializa lista vazia se ainda não existir

# Diretório onde serão salvos os resultados e gráficos da otimização.
# Se a pasta "resultados_variaveis" não existir, ela é criada automaticamente.
output_dir = "resultados_variaveis"
os.makedirs(output_dir, exist_ok=True)

# Histórico completo das partículas e do melhor global (gbest)
# ------------------------------------------------------------
# 'history_particles' guarda todas as posições das partículas em cada iteração
# → útil para gerar gráficos de dispersão e ver a "exploração" do espaço de busca.
# 
# 'history_gbest' guarda o valor das variáveis correspondentes ao melhor global em cada iteração
# → usado para gerar o gráfico da trajetória de convergência da melhor solução.
if 'history_particles' not in locals():
    history_particles = {v: [] for v in var_names}      # cria dicionário com listas vazias para cada variável
if 'history_gbest' not in locals():
    history_gbest = {v: [] for v in var_names}          # idem para o histórico do melhor global

# ============================================================
# 3 Parâmetros principais do PSO
# ============================================================
# Variáveis otimizadas: AR, span, taper, sweep, twist, alpha
xmin = np.array([6.0, 34, 0.3, 0.0, -3.0, -1.0])      # limites mínimos
xmax = np.array([12.0, 38, 1.0, 10.0, 3.0, 4.0])      # limites máximos
nrvar = len(xmin)                                     # número de variáveis (6)

# Parâmetros de controle do PSO
lambda1 = 2.02          # coeficiente cognitivo (atração pelo melhor individual)
lambda2 = 2.02          # coeficiente social (atração pelo melhor global)
omega = 0.4             # fator de inércia (peso da velocidade anterior)
pop = 2                 # tamanho da população (número de partículas)
tol = 1e-2              # tolerância para critério de parada
itermax = 50            # número máximo de iterações
random.seed(2)          # semente para reprodutibilidade

# ============================================================
# 4 Inicialização das partículas
# ============================================================
gbest = [1e30]          # melhor valor global inicial (grande para permitir minimização)
k = 1                   # contador de iterações

# Matrizes principais do PSO
v = np.zeros((pop, nrvar))          # velocidades das partículas
x = np.zeros((pop, nrvar))          # posições atuais
lbest = np.zeros(pop)               # melhores valores individuais
xlbest = np.zeros((pop, nrvar))     # melhores posições individuais


# Loop de inicialização: gera posições aleatórias e avalia cada partícula
for i in range(pop):
    for j in range(nrvar):
        # Inicializa posição dentro dos limites [xmin, xmax]
        x[i, j] = xmin[j] + (xmax[j] - xmin[j]) * random.random()

    # Avalia o desempenho aerodinâmico (chamada do OpenVSP)
    y = FCN(x[i, :])

    # --- Libera o OpenVSP da memória e reinicia a engine ---
    import importlib, openvsp, time
    time.sleep(0.5)             # pequena pausa para liberar arquivos temporários
    importlib.reload(openvsp)   # recarrega o módulo nativo (evita travamentos)

    lbest[i] = y            # guarda o melhor valor 
    xlbest[i, :] = x[i, :]  # guarda a melhor posição que apresentou o melhor valor

    # Atualiza o melhor global (gbest)
    if y < gbest[k - 1]:
        gbest[k - 1] = y         # melhor valor global
        xgbest = x[i, :].copy()  # melhor posição que apresentou o melhor valor

plt.pause(0.1)

# ============================================================
# 5 Loop principal do PSO
# ============================================================
flag = False
k = 2
gbest.append(gbest[0])  # adiciona um novo elemento copiando o valor inicial para iniciar o histórico de 
                        # gbest (equivalente ao crescimento automático do MATLAB)

# --- Loop principal: continua enquanto o critério de parada não for atingido ---
while not flag:

    gbest.append(gbest[k - 2])  # Copia o melhor valor global anterior (equivalente ao gbest(k) = gbest(k-1) do MATLAB)

    # Itera sobre todas as partículas
    for i in range(pop): # Percorre cada variável de decisão da partícula (ex: x e y)
        # Atualiza posição de cada variável
        for j in range(nrvar):

            # Sorteia dois números aleatórios entre 0 e 1 (usados nos termos cognitivo e social)
            r1 = random.random()
            r2 = random.random()

            # Equação clássica do PSO: atualiza velocidade
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

        # --- Libera memória e reinicia o módulo do OpenVSP ---
        time.sleep(0.5)
        importlib.reload(openvsp)


        # Atualiza o melhor valor individual (local)
        if ynew < lbest[i]:
            lbest[i] = ynew
            xlbest[i, :] = x[i, :]

        # Atualiza o melhor valor global (do grupo inteiro)
        if ynew < gbest[k - 1]:
            gbest[k - 1] = ynew
            xgbest = x[i, :].copy()

    # ========================================================
    # Armazena histórico de resultados
    # ========================================================
    if k == 2:
        history_particles = {v: [] for v in ["AR","span","taper","sweep","twist","alpha"]}
        history_gbest = {v: [] for v in ["AR","span","taper","sweep","twist","alpha"]}
        gbest_history = []
    gbest_history.append(gbest[k - 1])


    # Guarda histórico das variáveis
    history_particles['AR'].append(x[:,0].copy())
    history_particles['span'].append(x[:,1].copy())
    history_particles['taper'].append(x[:,2].copy())
    history_particles['sweep'].append(x[:,3].copy())
    history_particles['twist'].append(x[:,4].copy())
    history_particles['alpha'].append(x[:,5].copy())
    history_gbest['AR'].append(xgbest[0])
    history_gbest['span'].append(xgbest[1])
    history_gbest['taper'].append(xgbest[2])
    history_gbest['sweep'].append(xgbest[3])
    history_gbest['twist'].append(xgbest[4])
    history_gbest['alpha'].append(xgbest[5])

    # ========================================================
    # Critérios de parada
    # ========================================================
    if k >= itermax:
        flag = True

    if k > 11:
        # Verifica estabilização do gbest (variação pequena em 10 iterações)
        norm = np.sum(gbest[k - 9:k - 5]) - np.sum(gbest[k - 4:k])
        if norm < tol:
            flag = True  # igual ao MATLAB: pode ser comentado
            pass

    k += 1


# ============================================================
# 6 Pós-processamento: geração de gráficos e resultados
# ============================================================
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(gbest_history) + 1), gbest_history, 'b-', linewidth=1.5)
plt.xlim([0, itermax])
plt.xlabel("Iteração")
plt.ylabel("Melhor valor (gbest)")
plt.title("Convergência do PSO")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Convergencia_de_LD.png"), dpi=300, bbox_inches="tight")
plt.close()


# Gera gráficos de dispersão (todas as partículas ao longo das iterações)
for i, var in enumerate(var_names):
    plt.figure(figsize=(8, 4))
    # Pontos (valores de cada partícula em cada iteração
    for it, vals in enumerate(history_particles[var]):
        plt.scatter([it + 1] * len(vals), vals, color=colors[i], alpha=0.5, s=40)
    # linha do melhor global (gbest)
    plt.plot(range(1, len(history_gbest[var]) + 1), history_gbest[var], 'k-', lw=1.3, label="Melhor (gbest)")
    plt.xlabel("Iteração")
    plt.ylabel(var)
    plt.title(f"Evolução populacional da variável {var}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dispersao_{var}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 7 Gráfico combinado das variáveis ótimas (gbest)
# ============================================================
plt.figure(figsize=(9, 9))
for i, var in enumerate(var_names):
    plt.subplot(len(var_names), 1, i + 1)
    plt.plot(history_gbest[var], color=colors[i], linewidth=1.5)
    plt.ylabel(f"{var}")
    plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Iteração")
plt.suptitle("Evolução das variáveis ótimas por iteração", y=0.92)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "evolucao_variaveis_otimas.png"), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 8 Resultados finais
# ============================================================
print(f"\n✅ Gráficos de dispersão salvos em: {os.path.abspath(output_dir)}")
print("k =", k - 1)
if 'norm' in locals():
    print("norm =", norm)
print("gbest =", gbest[-1])
print("\n📊 Variáveis ótimas encontradas (xgbest):")
for name, value in zip(var_names, xgbest):
    print(f"   {name:<10} = {value:>8.4f}")

# ============================================================
# 9 Salvamento do resultado final em arquivo .txt
# ============================================================

# Caminho do arquivo de saída (salvo na mesma pasta dos gráficos)
result_file = os.path.join(output_dir, "resultado_final.txt")

with open(result_file, "w", encoding="utf-8") as f:
    f.write("=============================================\n")
    f.write("   RESULTADOS FINAIS DA OTIMIZAÇÃO PSO\n")
    f.write("=============================================\n\n")
    f.write(f"Iterações concluídas (k): {k - 1}\n")
    if 'norm' in locals():
        f.write(f"Critério de convergência (norm): {norm:.6f}\n")
    f.write(f"Melhor valor da função objetivo (gbest): {gbest[-1]:.6f}\n\n")

    f.write("📊 Variáveis ótimas encontradas (xgbest):\n")
    for name, value in zip(var_names, xgbest):
        f.write(f"   {name:<10} = {value:>8.4f}\n")

    f.write("\n✅ Gráficos e resultados salvos em:\n")
    f.write(f"{os.path.abspath(output_dir)}\n")

print(f"\n🗂️ Resultado final salvo em: {result_file}")


