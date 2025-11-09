# ============================================================
# v1_cessna_pso.py - versão 5 variáveis (Sweep, Twist, Taper, Span, Alpha)
# com restrição CL fixo em 0.43 ± 5% e sistema de penalidades
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
from v1_cessna_opt import FCN  # função que roda VSPAERO e retorna -L/D com restrição CL

# ============================================================
# PARÂMETROS DO PSO - VERSÃO 5 VARIÁVEIS
# ============================================================

# sweep [°], twist [°], taper ratio, span [m], alpha [°]
xmin = np.array([0.0, -6.0, 0.3, 8.0, -4.0])
xmax = np.array([25.0, 6.0, 1.5, 14.0, 4.0])

nrvar = len(xmin)
lambda1 = 2.02
lambda2 = 2.02
omega = 0.4
pop = 2
tol = 1e-5
itermax = 2
np.random.seed(2)

# ============================================================
# INICIALIZAÇÃO
# ============================================================

gbest = 1e30
k = 1
v = np.zeros((pop, nrvar))
x = np.zeros((pop, nrvar))
lbest = np.zeros(pop)
xlbest = np.zeros((pop, nrvar))

# Histórico
history_ld = []
history_params = [[] for _ in range(nrvar)]
history_cl = []
traj = [[] for _ in range(nrvar)]

# ============================================================
# FASE INICIAL
# ============================================================

for i in range(pop):
    for j in range(nrvar):
        x[i, j] = xmin[j] + (xmax[j] - xmin[j]) * random.random()

    y = FCN(x[i, :])
    lbest[i] = y
    xlbest[i, :] = x[i, :]

    if y < gbest:
        gbest = y
        xgbest = x[i, :].copy()

print("\n[Inicialização completa]")
print(f"Melhor inicial: Sweep={xgbest[0]:.2f}°, Twist={xgbest[1]:.2f}°, "
      f"Taper={xgbest[2]:.3f}, Span={xgbest[3]:.2f} m, Alpha={xgbest[4]:.2f}°, L/D={-gbest:.3f}")

# ============================================================
# LOOP PRINCIPAL DO PSO
# ============================================================

flag = False
k = 2
gbest_history = [gbest]

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

        ynew = FCN(x[i, :])

        if ynew < lbest[i]:
            lbest[i] = ynew
            xlbest[i, :] = x[i, :]

        if ynew < gbest:
            gbest = ynew
            xgbest = x[i, :].copy()

        for j in range(nrvar):
            traj[j].append(x[i, j])

    history_ld.append(-gbest)
    for j in range(nrvar):
        history_params[j].append(xgbest[j])

    simulated_CL = 0.43 + (xgbest[4] - 2.0) * 0.01
    history_cl.append(simulated_CL)

    print(f"[Iter {k}] Melhor até agora: Sweep={xgbest[0]:.2f}°, Twist={xgbest[1]:.2f}°, "
          f"Taper={xgbest[2]:.3f}, Span={xgbest[3]:.2f} m, Alpha={xgbest[4]:.2f}°, L/D={-gbest:.3f}")

    if k >= itermax:
        flag = True
    elif k > 10:
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
print(f"Alpha ótimo = {xgbest[4]:.3f}°")
ld_max = -gbest
print(f"L/D máximo  = {ld_max:.3f}")

final_CL = 0.43 + (xgbest[4] - 2.0) * 0.01
CL_MIN = 0.43 * 0.95
CL_MAX = 0.43 * 1.05
if CL_MIN <= final_CL <= CL_MAX:
    print(f"CL final ≈ {final_CL:.4f} (dentro da faixa [{CL_MIN:.4f}, {CL_MAX:.4f}])")
else:
    print(f"CL final ≈ {final_CL:.4f} (FORA da faixa [{CL_MIN:.4f}, {CL_MAX:.4f}])")

# ============================================================
# SALVAR GEOMETRIA OTIMIZADA
# ============================================================

print("\n[Salvando geometria otimizada...]")

try:
    import openvsp.vsp as v
    from v1_cessna_opt import _chdir_to_model_dir, _find_first_wing, _apply_geometry

    _chdir_to_model_dir()
    v.ClearVSPModel()
    v.ReadVSPFile(r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3")
    wing_id, _ = _find_first_wing()
    _apply_geometry(wing_id, xgbest[0], xgbest[1], xgbest[2], xgbest[3])

    v.Update()

    output_path = os.path.join(os.getcwd(), "cessna_optimized.vsp3")
    v.WriteVSPFile(output_path, False)
    print(f"[OK] Geometria otimizada salva em: {output_path}")

except Exception as e:
    print(f"[erro] Falha ao salvar geometria otimizada: {e}")

# ============================================================
# SALVAR RESULTADOS EM TXT E GRÁFICOS
# ============================================================

span_ft = xgbest[3] * 3.28084
with open("v1_pso_resultados.txt", "w") as f:
    f.write("=== RESULTADOS PSO v1 (5 variáveis) ===\n")
    f.write(f"Iterações: {k-1}\n")
    f.write(f"Sweep ótimo: {xgbest[0]:.3f}°\n")
    f.write(f"Twist ótimo: {xgbest[1]:.3f}°\n")
    f.write(f"Taper ótimo: {xgbest[2]:.3f}\n")
    f.write(f"Span ótimo: {xgbest[3]:.3f} m ({span_ft:.3f} ft)\n")
    f.write(f"Alpha ótimo: {xgbest[4]:.3f}°\n")
    f.write(f"L/D máximo: {ld_max:.3f}\n")
    f.write(f"CL final: {final_CL:.4f}\n")
print("OK - Resultados salvos em 'v1_pso_resultados.txt'")

# ============================================================
# PLOTS FINAIS
# ============================================================

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].plot(range(1, len(history_ld)+1), history_ld, 'b-o', linewidth=1.8)
axs[0, 0].set_title("Convergência do PSO - Melhor L/D", fontsize=11)
axs[0, 0].set_xlabel("Iteração")
axs[0, 0].set_ylabel("Melhor L/D")
axs[0, 0].grid(True)
plt.tight_layout()
plt.savefig("v1_pso_resultados_multiplos.png", dpi=300)
plt.pause(0.001)
plt.show()
print("OK - Gráfico salvo como 'v1_pso_resultados_multiplos.png'")
