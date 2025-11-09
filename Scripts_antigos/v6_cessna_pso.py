import os
import numpy as np
import matplotlib.pyplot as plt
import random
import importlib

# --- Garante recarregamento limpo da função FCN ---
import v6_cessna_opt
importlib.reload(v6_cessna_opt)
from v6_cessna_opt import FCN


# ============================================================
# --- Configuração inicial ---
# ============================================================

# Cria a pasta para salvar gráficos, se não existir
output_dir = "resultados_graficos"
os.makedirs(output_dir, exist_ok=True)

# --- Parâmetros principais ---
# sweep_deg, twist_deg, taper, span_total_ft, alpha_deg = x
xmin = np.array([0.0, -6.0, 0.3, 26.25, -4.0])   # [sweep, twist, taper, span, alpha]
xmax = np.array([30.0, 6.0, 1.2, 45.93, 4.0])

nrvar = len(xmin)
lambda1 = 2.02
lambda2 = 2.02
omega = 0.4
pop = 10  # 10+2*nrvar
tol = 1e-3
itermax = 30
np.random.seed(2)
w = omega
c1 = lambda1
c2 = lambda2

nvar = nrvar  # alias para consistência com o loop
gbest = 1e30
xgbest = np.zeros(nrvar)
gbest_history = []

# Inicializa vetores de pbest
pbest = np.zeros((pop, nrvar))
pbestval = np.full(pop, np.inf)



# --- Inicialização das partículas ---
gbest = float('inf')   # ← agora é número, não lista
xgbest = None
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

    if y < gbest:
        gbest = y
        xgbest = x[i, :].copy()

flag = False
gbest_history = [gbest]


# --- Histórico ---
history = {
    "L_over_D": [],
    "sweep": [],
    "twist": [],
    "taper": [],
    "span": [],
    "alpha": [],
}


# ============================================================
# --- Loop principal do PSO ---
# ============================================================

while not flag:
    k += 1
    print(f"\n===== Iteração {k}/{itermax} =====")

    for i in range(pop):
        # Atualização de velocidade e posição
        r1, r2 = np.random.rand(nvar), np.random.rand(nvar)
        v[i, :] = w * v[i, :] + c1 * r1 * (pbest[i, :] - x[i, :]) + c2 * r2 * (xgbest - x[i, :])
        x[i, :] = np.clip(x[i, :] + v[i, :], xmin, xmax)

        # Avalia função objetivo com tratamento de erro
        try:
            ynew = FCN(x[i, :])
        except Exception as e:
            print(f"[erro] Falha na partícula {i}: {e}")
            ynew = 1e6  # penalidade

        # --- Sempre registra no histórico, com penalidade como NaN ---
        ld_value = -ynew if ynew != 1e6 else np.nan
        history["L_over_D"].append(ld_value)


        # Log padronizado de cada partícula
        if np.isfinite(ynew):
            print(f"[iter {k} | part {i}] y={ynew:.3f} | sweep={x[i,0]:.2f} | twist={x[i,1]:.2f} | taper={x[i,2]:.2f} | span={x[i,3]:.2f} | alpha={x[i,4]:.2f}")
        else:
            print(f"[iter {k} | part {i}] resultado inválido, penalizado.")


        # Atualiza pbest
        if ynew < pbestval[i]:
            pbestval[i] = ynew
            pbest[i, :] = x[i, :].copy()

        # Atualiza gbest
        if ynew < gbest:
            gbest = ynew
            xgbest = x[i, :].copy()

    # --- Registrar histórico de cada iteração ---
    L_over_D = FCN(xgbest, return_LD_only=True)
    gbest_history.append(L_over_D)
    history["alpha"].append(xgbest[4])
    history["sweep"].append(xgbest[0])
    history["twist"].append(xgbest[1])
    history["taper"].append(xgbest[2])
    history["span"].append(xgbest[3])


    print(f"[iter {k}] Melhor L/D até agora: {L_over_D:.3f}")

    # Critério de parada mais robusto (convergência suave)
    valid_ld = np.array(history["L_over_D"])
    valid_ld = valid_ld[np.isfinite(valid_ld)]

    if len(valid_ld) >= 2:
        last_ld = valid_ld[-1]
        recent_mean = np.mean(valid_ld[-5:])
    else:
        last_ld = 0
        recent_mean = 0


    if k >= itermax or (k > 5 and abs(recent_mean - last_ld) < 1e-3):
        flag = True



    # --- Figura - convergência do melhor resultado global ---
    plt.figure(3)
    plt.plot(range(len(gbest_history)), gbest_history, 'b*-')
    plt.xlabel("Iteração")
    plt.ylabel("Melhor L/D global")
    plt.title("Convergência do L/D (gbest)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("resultados_graficos/convergencia_gbest.png", dpi=200)


    # Salva o gráfico de convergência
    plt.savefig(os.path.join(output_dir, "convergencia_gbest.png"), dpi=300, bbox_inches="tight")

    if k >= itermax:
        flag = True

    if k > 11 and len(gbest_history) > 10:
        norm = np.sum(gbest_history[-9:-5]) - np.sum(gbest_history[-4:])
        if abs(norm) < tol:
            flag = True


# ============================================================
# --- Resultados finais ---
# ============================================================

L_over_D_final = FCN(xgbest, return_LD_only=True)

print("\n=== RESULTADOS FINAIS ===")
print(f"Iterações executadas: {k - 1}")
print(f"Melhor L/D = {L_over_D_final:.4f}")
print("\n--- Variáveis ótimas ---")
print(f"Sweep  = {xgbest[0]:.3f} °")
print(f"Twist  = {xgbest[1]:.3f} °")
print(f"Taper  = {xgbest[2]:.3f}")
print(f"Span   = {xgbest[3]:.3f} ft")
print(f"Alpha  = {xgbest[4]:.3f} °")
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
