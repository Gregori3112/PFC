# ============================================================
# v13_cessna_pso.py (TOTALMENTE AJUSTADO)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
from v13_cessna_opt import FCN     # IMPORTANTE: FCN v13
from openvsp import openvsp as vsp

VSP3_FILE = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"

# Asa principal — pode deixar None pois o FCN identifica sozinho
wing_id = None


# ============================================================
# 1) CONFIGURAÇÃO DO PSO
# ============================================================

var_names = ["AR", "span", "taper", "sweep", "twist"]

xmin = np.array([6, 34, 0.5, 0.0, -4.0])
xmax = np.array([10, 38, 1.0, 10.0, 0])

nrvar = len(xmin)
pop = 2
itermax = 20
tol = 1e-4

omega = 0.4
lambda1 = 2.02
lambda2 = 2.02

random.seed(4)
np.random.seed(4)


# ============================================================
# 2) HISTÓRICO E PASTA DE RESULTADOS
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

x = np.zeros((pop, nrvar))
v = np.zeros((pop, nrvar))
lbest = np.full(pop, np.inf)
xlbest = np.zeros((pop, nrvar))

gbest_value = 1e30
k = 1

asa_base = np.array([7.5, 36.0, 1.0, 0.0, 0.0])

for i in range(pop):

    if i == 0:
        x[i, :] = asa_base
    else:
        for j in range(nrvar):
            x[i, j] = xmin[j] + (xmax[j] - xmin[j]) * random.random()

    y, data = FCN(x[i, :])
    CL = data["CL"]
    CD = data["CD_total"]
    LD = data["LD"]
    Alpha = data["Alpha"]

    if i == 0:
        alpha_base = Alpha
        print(f"[info] Alpha da asa base = {alpha_base:.3f}°")
        ld_history.append(LD)

    lbest[i] = y
    xlbest[i, :] = x[i, :]

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
    
    print(f"\n==================== Iteração {k-1} ====================")

    for i in range(pop):

        for j in range(nrvar):

            r1 = random.random()
            r2 = random.random()

            vnew = (omega * v[i, j] +
                    lambda1 * r1 * (xlbest[i, j] - x[i, j]) +
                    lambda2 * r2 * (xgbest[j] - x[i, j]))

            xnew = np.clip(x[i, j] + vnew, xmin[j], xmax[j])

            v[i, j] = vnew
            x[i, j] = xnew

        ynew, data = FCN(x[i, :])
        CL = data["CL"]
        CD = data["CD_total"]
        LD = data["LD"]

        print(f"[pso] Iter={k-1}, Partícula={i+1}/{pop} → fobj={ynew:.3f}, L/D={LD:.2f}")

        if ynew < lbest[i]:
            lbest[i] = ynew
            xlbest[i, :] = x[i, :]

        if ynew < gbest_value:
            gbest_value = ynew
            xgbest = x[i, :].copy()
            CL_best = CL
            CD_best = CD
            LD_best = LD

    gbest_history.append(gbest_value)

    for idx, var in enumerate(var_names):
        history_particles[var].append(x[:, idx].copy())
        history_gbest[var].append(xgbest[idx])

    if k >= itermax:
        flag = True

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
# 5) GRÁFICOS
# ============================================================

plt.figure(figsize=(7,5))
plt.plot(gbest_history, 'b-o')
plt.xlabel("Iteração")
plt.ylabel("fobj (mínimo)")
plt.title("Convergência da Função Objetivo")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "convergencia_fobj.png"))
plt.close()

for i, var in enumerate(var_names):

    plt.figure(figsize=(8,4))

    for it, vals in enumerate(history_particles[var]):
        plt.scatter([it+1]*len(vals), vals, color='blue', alpha=0.4, s=30)

    plt.plot(history_gbest[var], 'r-', lw=1.5, label="gbest")

    plt.xlabel("Iteração")
    plt.ylabel(var)
    plt.title(f"Evolução da variável: {var}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dispersao_{var}.png"))
    plt.close()

plt.figure(figsize=(7,5))
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
# 6) RESULTADOS FINAIS EM TXT
# ============================================================

result_file = os.path.join(output_dir, "resultado_final.txt")

f_best, data = FCN(xgbest)
cl_best = data["CL"]
cd_best = data["CD_total"]
ld_best = data["LD"]
L_best = data["L"]

W_lbf = 1800 * 2.20462      # peso em lbf
LW_ratio = (L_best / W_lbf) * 100
CL_ideal = cl_best * (W_lbf / L_best)

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

print(f"\n✅ Resultado final salvo em: {result_file}")


# ============================================================
# 7) SALVA GEOMETRIA FINAL
# ============================================================

print("\n[save-best] Salvando cessna_best.vsp3...")

vsp.ClearVSPModel()
vsp.ReadVSPFile(VSP3_FILE)

AR, span, taper, sweep, twist = xgbest
croot = 2 * span / (AR * (1 + taper))
ctip  = taper * croot

wing_id = vsp.FindGeoms()[0]   # automático

vsp.SetParmVal(wing_id, "Span", "XSec_1", span/2)
vsp.SetParmVal(wing_id, "Root_Chord", "XSec_1", croot)
vsp.SetParmVal(wing_id, "Tip_Chord", "XSec_1", ctip)
vsp.SetParmVal(wing_id, "Taper", "XSec_1", taper)
vsp.SetParmVal(wing_id, "Sweep", "XSec_1", sweep)
vsp.SetParmVal(wing_id, "Twist", "XSec_1", twist)

vsp.Update()
best_file = os.path.join(output_dir, "cessna_best.vsp3")
vsp.WriteVSPFile(best_file)

print(f"[save-best] Arquivo salvo em: {best_file}")
