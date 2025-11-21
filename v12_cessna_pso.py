# ============================================================
# v12_cessna_pso.py
# ------------------------------------------------------------
# Implementação completa do algoritmo PSO (Particle Swarm
# Optimization) aplicado à otimização aerodinâmica de uma asa
# utilizando o OpenVSP + VSPAERO.
#
# Cada partícula = um conjunto de variáveis geométricas:
#   [AR, span, taper, sweep, twist]
#
# O objetivo é maximizar L/D → equivalente a minimizar -L/D.
#
# A função FCN() (importada de v12_cessna_opt.py) realiza toda
# a simulação aerodinâmica.
# ------------------------------------------------------------
# Autor: Gregori da Maia da Silva
# ============================================================


import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
from v12_cessna_opt import FCN     # Função objetivo que executa o VSPAERO
from openvsp import openvsp as vsp


# Caminho do arquivo base (necessário para salvar a geometria ótima ao final)
VSP3_FILE = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"

# ID da asa principal dentro do modelo (igual ao usado no FCN)
wing_id = "ITDQSYJOYI"


# ============================================================
# 1) CONFIGURAÇÃO INICIAL DO PSO
# ============================================================

# Nomes das variáveis (usados para gráficos e organização)
var_names = ["AR", "span", "taper", "sweep", "twist"]

# Faixas permitidas de cada variável (limites de busca)
xmin = np.array([6, 34, 0.5, 0.0, -3.0])
xmax = np.array([10, 38, 1.0, 10.0, 3.0])

nrvar = len(xmin)       # número de variáveis = 5
pop = 8                # quantidade de partículas
itermax = 10           # número máximo de iterações do PSO
tol = 1e-4              # tolerância para critério de estagnação

# Parâmetros clássicos do PSO
omega = 0.4             # inércia
lambda1 = 2.02          # componente cognitivo
lambda2 = 2.02          # componente social

# Para reprodutibilidade
random.seed(4)
np.random.seed(4)


# ============================================================
# 2) PREPARAÇÃO DE HISTÓRICOS E DIRETÓRIOS
# ============================================================

output_dir = "resultados_variaveis"
os.makedirs(output_dir, exist_ok=True)

history_particles = {v: [] for v in var_names}
history_gbest = {v: [] for v in var_names}
gbest_history = []
ld_history = []


# ============================================================
# 3) INICIALIZAÇÃO DAS PARTÍCULAS
# ============================================================

# Vetores principais do algoritmo
x = np.zeros((pop, nrvar))       # posição atual de cada partícula
v = np.zeros((pop, nrvar))       # velocidade de cada partícula
lbest = np.full(pop, np.inf)     # melhor valor já encontrado por cada partícula
xlbest = np.zeros((pop, nrvar))  # melhor posição já encontrada por cada partícula

gbest_value = 1e30   # melhor valor global escalar
k = 1                # contador de iterações                           # contador de iterações

# ASA BASE para iniciar o PSO
asa_base = np.array([7.5, 36.0, 1.0, 0.0, 0.0])

# Loop de inicialização das partículas
for i in range(pop):

    if i == 0:
        # A primeira partícula é a asa base real
        x[i, :] = asa_base
    else:
        # As demais são aleatórias
        for j in range(nrvar):
            x[i, j] = xmin[j] + (xmax[j] - xmin[j]) * random.random()

    # Calcula função objetivo e extrai dados aerodinâmicos
    y, data = FCN(x[i, :])
    CL = data["CL"]
    CD = data["CD"]
    LD = data["LD"]
    Alpha = data["Alpha"]

    if i == 0:
        alpha_base = Alpha
        print(f"[info] Alpha da asa base = {alpha_base:.3f}°")

        ld_history.append(LD)   # adiciona o L/D da asa base no histórico
        # ----------------------------------------------

    # Salva melhor local
    lbest[i] = y
    xlbest[i, :] = x[i, :]

    # Atualiza melhor global
    if y < gbest_value:
        gbest_value = y
        gbest_history.append(gbest_value)
        xgbest = x[i, :].copy()
        CL_best = CL
        CD_best = CD
        LD_best = LD

plt.pause(0.1)


# ============================================================
# 4) LOOP PRINCIPAL DO PSO
# ============================================================

flag = False
k = 2


while not flag:


    # ========================================================
    # Atualiza posição e avalia cada partícula
    # ========================================================
    for i in range(pop):

        for j in range(nrvar):

            # Termos aleatórios
            r1 = random.random()
            r2 = random.random()

            # Atualização clássica da velocidade
            vnew = (omega * v[i, j] +
                    lambda1 * r1 * (xlbest[i, j] - x[i, j]) +
                    lambda2 * r2 * (xgbest[j] - x[i, j]))

            # Nova posição + limites
            xnew = np.clip(x[i, j] + vnew, xmin[j], xmax[j])

            # Atualiza matriz principal
            v[i, j] = vnew
            x[i, j] = xnew

        # Avalia o desempenho aerodinâmico da nova posição
        ynew, data = FCN(x[i, :])
        CL = data["CL"]
        CD = data["CD"]
        LD = data["LD"]

        print(f"[pso] Iter={k-1}, Partícula={i+1}/{pop} → fobj={ynew:.3f}, L/D={LD:.2f}")

        # Melhor local (lbest)
        if ynew < lbest[i]:
            lbest[i] = ynew
            xlbest[i, :] = x[i, :]

        # Melhor global (gbest)
        if ynew < gbest_value:
            gbest_value = ynew
            xgbest = x[i, :].copy()
            CL_best = CL
            CD_best = CD
            LD_best = LD

    # ========================================================
    # Guarda histórico das partículas e do gbest
    # ========================================================
    gbest_history.append(gbest_value)


    for idx, var in enumerate(var_names):
        history_particles[var].append(x[:, idx].copy())
        history_gbest[var].append(xgbest[idx])

    # ========================================================
    # Critérios de parada
    # ========================================================

    # Parada por limite máximo de iterações
    if k >= itermax:
        flag = True

    # Parada por estabilização
    if len(gbest_history) >= 10:
        prev_win = gbest_history[-10:-5]
        curr_win = gbest_history[-5:]
        delta = abs(np.mean(curr_win) - np.mean(prev_win))
        if delta < tol:
            flag = True

    print(f"[iter {k-1}] gbest={gbest_value:.4f} | L/D≈{LD_best:.2f} (gbest) | xgbest={xgbest}")
    ld_history.append(LD_best)

    k += 1


# ============================================================
# 5) GRÁFICOS DE CONVERGÊNCIA
# ============================================================

plt.figure(figsize=(7, 5))
plt.plot(gbest_history, 'b-o')
plt.xlabel("Iteração")
plt.ylabel("fobj (mínimo)")
plt.title("Convergência da Função Objetivo")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "convergencia_fobj.png"))
plt.close()


# Gráficos de dispersão das partículas
for i, var in enumerate(var_names):

    plt.figure(figsize=(8, 4))

    for it, vals in enumerate(history_particles[var]):
        plt.scatter([it + 1] * len(vals), vals, color='blue', alpha=0.4, s=30)

    plt.plot(history_gbest[var], 'r-', lw=1.5, label="gbest")

    plt.xlabel("Iteração")
    plt.ylabel(var)
    plt.title(f"Evolução da variável: {var}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dispersao_{var}.png"))
    plt.close()


# Gráfico do L/D
plt.figure(figsize=(7, 5))
plt.plot(ld_history, 'g-o')
plt.xlabel("Iteração")
plt.ylabel("L/D (melhor)")
plt.title("Convergência Física (L/D)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "convergencia_LD_best.png"))
plt.close()


print(f"\n✅ Gráficos salvos em: {os.path.abspath(output_dir)}")


# ============================================================
# 6) RESULTADOS FINAIS — SALVOS EM TXT
# ============================================================

result_file = os.path.join(output_dir, "resultado_final.txt")

# Reavalia o melhor ponto final
f_best, data = FCN(xgbest)
cl_best = data["CL"]
cd_best = data["CD"]
ld_best = data["LD"]
L_best = data["L"]
W = 1800 * 9.81
LW_ratio = (L_best / W) * 100
CL_ideal = cl_best * (W / L_best)

with open(result_file, "w", encoding="utf-8") as f:

    f.write("=============================================\n")
    f.write("   RESULTADOS FINAIS DA OTIMIZAÇÃO PSO\n")
    f.write("=============================================\n\n")

    f.write(f"Melhor L/D encontrado.............: {ld_best:.4f}\n")
    f.write(f"CL................................: {cl_best:.4f}\n")
    f.write(f"CD................................: {cd_best:.4f}\n")
    f.write(f"L/W...............................: {LW_ratio:.2f}%\n")
    f.write(f"CL ideal para L=W.................: {CL_ideal:.4f}\n\n")

    f.write("Variáveis ótimas:\n")
    for name, value in zip(var_names, xgbest):
        f.write(f"  {name:<10} = {value:.5f}\n")

    f.write("\n")

    # -------------------------------
    # ANÁLISE AERODINÂMICA AVANÇADA
    # -------------------------------
    AR_opt, span_opt, taper_opt, sweep_opt, twist_opt = xgbest

    f.write("=============================================\n")
    f.write(" ANALISE AERODINÂMICA AVANÇADA\n")
    f.write("=============================================\n\n")

    f.write(f"Aspect Ratio......................: {AR_opt:.4f}\n")
    f.write(f"Envergadura (span)................: {span_opt:.4f} ft\n")
    f.write(f"Taper Ratio.......................: {taper_opt:.4f}\n")
    f.write(f"Sweep.............................: {sweep_opt:.4f}°\n")
    f.write(f"Twist.............................: {twist_opt:.4f}°\n\n")

    f.write("Diagnóstico físico:\n")
    f.write("- AR elevado reduz o arrasto induzido e melhora L/D.\n")
    f.write("- Taper adequado aproxima a distribuição elíptica.\n")
    f.write("- Sweep baixo melhora CL em Mach moderado.\n")
    f.write("- Washout (twist negativo) reduz estol de ponta.\n")
    f.write("- Solver inviscid superestima L/D (não inclui CD0 real).\n\n")

    f.write("Resumo para TCC:\n")
    f.write("A otimização com PSO mostrou forte influência do Aspect Ratio,\n")
    f.write("taper e twist na eficiência aerodinâmica, atingindo um L/D máximo\n")
    f.write(f"de {ld_best:.2f}. O método demonstra claramente como tendências\n")
    f.write("aerodinâmicas podem ser exploradas com OpenVSP + PSO.\n")


print(f"\n✅ Resultado final salvo em: {result_file}")


# ============================================================
# 7) SALVA A GEOMETRIA FINAL EM .vsp3
# ============================================================

print("\n[save-best] Salvando cessna_best.vsp3...")

vsp.ClearVSPModel()
vsp.ReadVSPFile(VSP3_FILE)

AR, span, taper, sweep, twist = xgbest
croot = 2 * span / (AR * (1 + taper))
ctip = taper * croot

vsp.SetParmVal(wing_id, "Span", "XSec_1", span / 2)
vsp.SetParmVal(wing_id, "Root_Chord", "XSec_1", croot)
vsp.SetParmVal(wing_id, "Tip_Chord", "XSec_1", ctip)
vsp.SetParmVal(wing_id, "Taper", "XSec_1", taper)
vsp.SetParmVal(wing_id, "Sweep", "XSec_1", sweep)
vsp.SetParmVal(wing_id, "Twist", "XSec_1", twist)

vsp.Update()
best_file = os.path.join(output_dir, "cessna_best.vsp3")
vsp.WriteVSPFile(best_file)

print(f"[save-best] Arquivo salvo em: {best_file}")

