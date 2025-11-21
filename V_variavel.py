# ============================================================
# Script: V_variavel.py (Mach Sweep com Alpha Fixo)
# ------------------------------------------------------------
# Objetivo:
# - Inserir x_test
# - Fixar alpha = 0°
# - Varredura Mach 0.1 → 0.8
# - Calcular CL, CDi, Di, L
# - Plotar curvas vs velocidade
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from openvsp import openvsp as vsp

# ============================================================
# CONFIGURAÇÃO
# ============================================================

base_dir = r"C:\VSP\Development\PSO_PYTHON_WING"
VSP3_FILE = os.path.join(base_dir, "cessna210.vsp3")

# Peso (imperial)
W = 1800 * 2.20462     # lbf

# Densidade padrão (slug/ft³)
rho = 0.002377

# Faixa de Mach
mach_list = np.linspace(0.1, 0.8, 20)

# Alpha fixo
alpha_fixed = 0.0


# ============================================================
# FUNÇÃO: roda 1 caso VSPAERO em um Mach específico
# ============================================================

def run_case(x, M):

    # --------------------------------------------------------
    # 1) Limpa modelo e carrega o arquivo base
    # --------------------------------------------------------
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(VSP3_FILE)

    # Encontra Wing
    gid = None
    for g in vsp.FindGeoms():
        if vsp.GetGeomTypeName(g) == "Wing":
            gid = g
            break

    if gid is None:
        raise RuntimeError("Wing não encontrada no modelo!")

    # --------------------------------------------------------
    # 2) Aplica geometria baseada em x_test
    # --------------------------------------------------------
    AR, span, taper, sweep, twist = x

    # Cordas
    croot = 2 * span / (AR * (1 + taper))
    ctip  = taper * croot

    vsp.SetParmVal(gid, "Span", "XSec_1", span / 2)
    vsp.SetParmVal(gid, "Root_Chord", "XSec_1", croot)
    vsp.SetParmVal(gid, "Tip_Chord", "XSec_1", ctip)
    vsp.SetParmVal(gid, "Taper", "XSec_1", taper)
    vsp.SetParmVal(gid, "Sweep", "XSec_1", sweep)
    vsp.SetParmVal(gid, "Twist", "XSec_1", twist)
    vsp.Update()

    # --------------------------------------------------------
    # 3) Salva o modelo atualizado com novo nome
    # --------------------------------------------------------
    updated_vsp3 = os.path.join(base_dir, "mach_sweep.vsp3")
    vsp.WriteVSPFile(updated_vsp3)

    # Caminho do history correspondente
    hist_path = os.path.join(base_dir, "mach_sweep.history")

    # Remove qualquer .history antigo
    for f in os.listdir(base_dir):
        if f.endswith(".history"):
            try:
                os.remove(os.path.join(base_dir, f))
            except:
                pass

    # --------------------------------------------------------
    # 4) Compute Geometry
    # --------------------------------------------------------
    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [vsp.SET_ALL])
    vsp.ExecAnalysis("VSPAEROComputeGeometry")

    # --------------------------------------------------------
    # 5) Condições de voo
    # --------------------------------------------------------
    gamma = 1.4
    R = 287.05
    T = 288.15
    a_si = (gamma * R * T)**0.5
    V_si = M * a_si
    V_ft = V_si / 0.3048

    q = 0.5 * rho * V_ft * V_ft
    Sref = vsp.GetParmVal(gid, "TotalArea", "WingGeom")

    # --------------------------------------------------------
    # 6) Configuração do solver
    # --------------------------------------------------------
    solver_id = "VSPAEROSweep"

    vsp.SetAnalysisInputDefaults(solver_id)
    vsp.SetDoubleAnalysisInput(solver_id, "MachStart", [M])
    vsp.SetDoubleAnalysisInput(solver_id, "MachEnd",   [M])
    vsp.SetIntAnalysisInput(solver_id, "MachNpts", [1])

    vsp.SetDoubleAnalysisInput(solver_id, "AlphaStart", [alpha_fixed])
    vsp.SetDoubleAnalysisInput(solver_id, "AlphaEnd",   [alpha_fixed])
    vsp.SetIntAnalysisInput(solver_id, "AlphaNpts",     [1])

    vsp.SetDoubleAnalysisInput(solver_id, "Vinf", [V_ft])
    vsp.SetDoubleAnalysisInput(solver_id, "Rho",  [rho])
    vsp.SetIntAnalysisInput(solver_id, "GeomSet", [vsp.SET_ALL])
    vsp.SetIntAnalysisInput(solver_id, "NumWakeNodes", [24])


    # --------------------------------------------------------
    # 7) Executa solver
    # --------------------------------------------------------
    vsp.ExecAnalysis(solver_id)

    # Espera criação do .history
    for _ in range(60):
        if os.path.exists(hist_path):
            break
        time.sleep(0.2)

    if not os.path.exists(hist_path):
        raise RuntimeError("History file não encontrado — solver não gerou saída!")

    # --------------------------------------------------------
    # 8) Lê resultados do history
    # --------------------------------------------------------
    with open(hist_path, "r") as f:
        lines = [
            l.strip() for l in f.readlines()
            if l.strip() and not l.startswith("#")
        ]

    last = lines[-1].split()

    cl  = float(last[6])
    cd0 = float(last[7])
    cdi = float(last[8])

    # Forças
    L  = q * Sref * cl
    Di = q * Sref * cdi

    return M, V_ft, cl, cdi, Di, L


# ============================================================
# LOOP MACH SWEEP
# ============================================================

def main():
    # x_test = [AR, span, taper, sweep, twist]
    x_test = np.array([8, 36.0, 0.75, 0.5, -2])

    V_list = []
    CL_list = []
    CDi_list = []
    Di_list = []
    L_list = []

    for M in mach_list:
        M, V_ft, cl, cdi, Di, L = run_case(x_test, M)

        V_list.append(V_ft)
        CL_list.append(cl)
        CDi_list.append(cdi)
        Di_list.append(Di)
        L_list.append(L)

        print(f"Mach={M:.2f} | V={V_ft:.1f} ft/s | CL={cl:.4f} | CDi={cdi:.5f} | Di={Di:.2f} | L={L:.1f}")

    # --------------------------------------------------------
    # GRÁFICOS
    # --------------------------------------------------------

    plt.figure(figsize=(10,6))
    plt.plot(V_list, CDi_list, '-o')
    plt.xlabel("Velocidade (ft/s)")
    plt.ylabel("CDi")
    plt.title("CDi vs Velocidade")
    plt.grid(True)

    plt.figure(figsize=(10,6))
    plt.plot(V_list, Di_list, '-o')
    plt.xlabel("Velocidade (ft/s)")
    plt.ylabel("Di (lbf)")
    plt.title("Di vs Velocidade")
    plt.grid(True)

    plt.figure(figsize=(10,6))
    plt.plot(V_list, CL_list, '-o')
    plt.xlabel("Velocidade (ft/s)")
    plt.ylabel("CL")
    plt.title("CL vs Velocidade")
    plt.grid(True)

    plt.figure(figsize=(10,6))
    plt.plot(V_list, L_list, '-o')
    plt.xlabel("Velocidade (ft/s)")
    plt.ylabel("L (lbf)")
    plt.title("Sustentação L vs Velocidade")
    plt.grid(True)

    plt.show()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
