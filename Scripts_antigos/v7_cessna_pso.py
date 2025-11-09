import os
import numpy as np
import matplotlib.pyplot as plt
import random
import importlib
import pandas as pd
import seaborn as sns

# --- Garante recarregamento limpo da função FCN ---
import v7_cessna_opt
importlib.reload(v7_cessna_opt)
from v7_cessna_opt import FCN


# ============================================================
# --- Configuração inicial ---
# ============================================================

# Cria a pasta para salvar gráficos, se não existir
output_dir = "resultados_graficos"
os.makedirs(output_dir, exist_ok=True)

# --- Parâmetros principais ---
# sweep_deg, twist_deg, ctip_ft, alpha_deg = x
xmin = np.array([0.0, -6.0, 1.0, -4.0])   # limites mínimos
xmax = np.array([30.0, 6.0, 4.76339, 4.0])    # máximos (ctip <= Croot = 6 ft)


nrvar = len(xmin)
lambda1 = 2.02
lambda2 = 2.02
omega = 0.4
pop = 4  # 10+2*nrvar
tol = 1e-3
itermax = 4
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
    "ctip": [],
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
            print(f"[iter {k} | part {i}] y={ynew:.3f} | sweep={x[i,0]:.2f} | twist={x[i,1]:.2f} | ctip={x[i,2]:.2f} | alpha={x[i,3]:.2f}")
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
    history["sweep"].append(xgbest[0])
    history["twist"].append(xgbest[1])
    history["ctip"].append(xgbest[2])
    history["alpha"].append(xgbest[3])



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
print(f"Ctip   = {xgbest[2]:.3f} ft")
print(f"Alpha  = {xgbest[3]:.3f} °")
print("=========================\n")

# ============================================================
# --- Gráficos de evolução e salvamento automático ---
# ============================================================

plt.figure(figsize=(9, 8))
plt.subplot(4, 1, 1)
plt.plot(history["L_over_D"], 'b')
plt.ylabel("L/D")
plt.title("Evolução das Variáveis por Iteração")

plt.subplot(4, 1, 2)
plt.plot(history["sweep"], 'r')
plt.ylabel("Sweep [deg]")

plt.subplot(4, 1, 3)
plt.plot(history["twist"], 'm')
plt.ylabel("Twist [deg]")

plt.subplot(4, 1, 4)
plt.plot(history["ctip"], 'c')
plt.ylabel("Ctip [ft]")
plt.xlabel("Iteração")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "evolucao_variaveis.png"), dpi=300, bbox_inches="tight")


# --- Figura - convergência do melhor resultado global ---
plt.figure(3)
plt.plot(range(len(gbest_history)), gbest_history, 'b*-')
plt.xlabel("Iteração")
plt.ylabel("Melhor L/D global")
plt.title("Convergência do L/D (gbest)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("resultados_graficos/convergencia_gbest.png", dpi=200)

# --- Salva gráficos individuais ---
for key, color, label, name in [
    ("L_over_D", 'b', "L/D", "LD"),
    ("sweep", 'r', "Sweep [deg]", "sweep"),
    ("twist", 'm', "Twist [deg]", "twist"),
    ("ctip", 'c', "Ctip [ft]", "ctip"),
    ("alpha", 'g', "Alpha [deg]", "alpha"),
]:
    plt.figure()
    plt.plot(history[key], color)
    plt.xlabel("Iteração")
    plt.ylabel(label)
    plt.title(f"Evolução de {label}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"evolucao_{name}.png"), dpi=300, bbox_inches="tight")



# ============================================================
# --- Visualizações adicionais tipo PSO 2D (linhas e correlação)
# ============================================================

# --- Convergência suavizada do L/D ---
plt.figure(figsize=(8, 4))
plt.plot(gbest_history, 'b-', linewidth=1.5, label="L/D real")
if len(gbest_history) > 5:
    window = 5
    smooth = np.convolve(gbest_history, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(smooth)+window-1), smooth, 'r--', linewidth=1.3, label="Média móvel")
plt.xlabel("Iteração")
plt.ylabel("Melhor L/D (gbest)")
plt.title("Convergência do PSO – L/D máximo")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "convergencia_suavizada.png"), dpi=300, bbox_inches="tight")

# --- Evolução conjunta das variáveis (tipo painel analítico) ---
plt.figure(figsize=(9, 7))
vars_labels = ["Sweep [°]", "Twist [°]", "Ctip [ft]", "Alpha [°]"]
colors = ['r', 'm', 'c', 'g']

var_matrix = np.array([
    history["sweep"],
    history["twist"],
    history["ctip"],
    history["alpha"]
]).T

for i in range(var_matrix.shape[1]):
    plt.subplot(var_matrix.shape[1], 1, i + 1)
    plt.plot(var_matrix[:, i], color=colors[i], linewidth=1.4)
    plt.ylabel(vars_labels[i])
    plt.grid(True, linestyle='--', alpha=0.4)
    if i == 0:
        plt.title("Evolução das variáveis de projeto ao longo das iterações")

plt.xlabel("Iteração")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "evolucao_completa_variaveis.png"),
            dpi=300, bbox_inches="tight")


# --- Correlação entre variáveis (Pairplot) ---
try:
    min_len = min(len(history["sweep"]), len(history["L_over_D"]))
    df = pd.DataFrame({
        "Sweep": history["sweep"][:min_len],
        "Twist": history["twist"][:min_len],
        "Ctip": history["ctip"][:min_len],
        "Alpha": history["alpha"][:min_len],
        "L/D": history["L_over_D"][:min_len]
    })


    sns.pairplot(df, corner=True, diag_kind="kde",
                 plot_kws={"alpha": 0.7, "s": 30, "edgecolor": None})
    plt.suptitle("Relações entre variáveis e L/D durante a otimização", y=1.02)
    plt.savefig(os.path.join(output_dir, "correlacao_variaveis.png"), dpi=300, bbox_inches="tight")
except Exception as e:
    print(f"[aviso] Não foi possível gerar o pairplot: {e}")

print(f"\n✅ Todos os gráficos foram salvos na pasta '{output_dir}/'")
