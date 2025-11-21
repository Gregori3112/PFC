# ============================================================
# v14_cessna_opt.py
# ------------------------------------------------------------
# Função FCN: executa uma simulação completa no OpenVSP + VSPAERO
# dado um vetor de parâmetros geométricos. Retorna:
#     - fobj = função objetivo (-L/D + penalidade)
#     - dicionário com CL, CD, L/D, alpha e L
#
# Agora em v13:
#   - CDtotal = CDi (VSPAERO) + CD0_parasita (modelo analítico)
#   - CD0_parasita calibrado a partir do Parasite Drag (CD0 = 0.00894)
#
# Autor: Gregori da Maia da Silva
# ============================================================

import os
import sys
import time
import gc
import numpy as np
from openvsp import openvsp as vsp

# [NOVO - v13] Constantes para modelo de arrasto parasita
CD0_BASE = 0.00843       # CD0 da asa base (obtido no Parasite Drag)
SWEEP_BASE_DEG = 0.0     # sweep da asa base usado na calibração
# Geometria da asa base usada para calibrar o CD0_BASE
AR_BASE = 7.5            # alongamento da asa base (Cessna 210 original)
SREF_BASE = 172.707      # [ft^2] área da asa base (da simulação Parasite Drag)
E_OSWALD = 0.8           # eficiência típica (mantida constante na correção)


def FCN(x: np.ndarray):
    """
    Função objetivo para o PSO. Recebe um vetor de variáveis geométricas,
    aplica no modelo OpenVSP, executa VSPAERO e retorna o desempenho aerodinâmico.
    Agora em v13 o CDtotal usado no L/D é:
        CD = CDi_vspaero + CD0_parasita
    """

    # ------------------------------------------------------------
    # Limpa arquivos antigos gerados por simulações anteriores
    # ------------------------------------------------------------
    for f in os.listdir(r"C:\VSP\Development\PSO_PYTHON_WING"):
        if f.startswith("cessna_updated.") or f.startswith("temp_polar."):
            try:
                os.remove(os.path.join(r"C:\VSP\Development\PSO_PYTHON_WING", f))
            except PermissionError:
                pass

    # ============================================================
    # 1) CARREGAMENTO DO MODELO BASE
    # ------------------------------------------------------------
    VSP3_FILE = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"

    vsp.ClearVSPModel()
    vsp.ReadVSPFile(VSP3_FILE)

    # Busca a asa principal automaticamente
    geom_ids = vsp.FindGeoms()
    wing_id = None
    for gid in geom_ids:
        if vsp.GetGeomTypeName(gid) == "Wing":
            wing_id = gid
            break

    if wing_id is None:
        raise RuntimeError("ERRO: Nenhuma asa encontrada no modelo!")

    print(f"Wing ID encontrado: {wing_id}")

    # Nome interno do solver usado pelo OpenVSP
    solver_id = "VSPAEROSweep"

    # ============================================================
    # 2) APLICA VARIÁVEIS GEOMÉTRICAS AO MODELO
    # ------------------------------------------------------------
    # Entradas do PSO: AR, envergadura, taper ratio, sweep, twist
    AR, span, taper, sweep, twist = x

    # Calcula cordas coerentes com AR e taper
    croot = 2 * span / (AR * (1 + taper))
    ctip  = taper * croot

    # OpenVSP usa semi-envergadura
    vsp.SetParmVal(wing_id, "Span",       "XSec_1", span / 2.0)
    vsp.SetParmVal(wing_id, "Root_Chord", "XSec_1", croot)
    vsp.SetParmVal(wing_id, "Tip_Chord",  "XSec_1", ctip)
    vsp.SetParmVal(wing_id, "Taper",      "XSec_1", taper)
    vsp.SetParmVal(wing_id, "Sweep",      "XSec_1", sweep)
    vsp.SetParmVal(wing_id, "Twist",      "XSec_1", twist)

    vsp.Update()

    print(f"[geo] AR={AR:.2f}, Span={span:.2f}, Taper={taper:.2f}, Sweep={sweep:.2f}, Twist={twist:.2f}")
    print(f"[geo] Croot={croot:.3f}, Ctip={ctip:.3f}")

    # ============================================================
    # 3) GERA MALHA + EXECUTA SOLVER VSPAERO
    # ============================================================

    # Salva temporariamente um vsp3 com a geometria atualizada
    vsp.Update()
    vsp.WriteVSPFile(r"C:\VSP\Development\PSO_PYTHON_WING\cessna_updated.vsp3")

    # 3.1) Degenerate Geometry
    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [vsp.SET_ALL])
    vsp.ExecAnalysis("VSPAEROComputeGeometry")

    # 3.2) Configuração do Solver Aerodinâmico
    vsp.SetAnalysisInputDefaults(solver_id)

    available_inputs = []
    try:
        available_inputs = vsp.GetAnalysisInputNames(solver_id)
    except Exception:
        pass

    if "PolarFileName" in available_inputs:
        vsp.SetStringAnalysisInput(solver_id, "PolarFileName", [""])
    if "SliceFileName" in available_inputs:
        vsp.SetStringAnalysisInput(solver_id, "SliceFileName", [""])
    if "NumSlices" in available_inputs:
        vsp.SetIntAnalysisInput(solver_id, "NumSlices", [0])

    # ============================================================
    # 4) CONDIÇÕES DE VOO
    # ============================================================

    sref = vsp.GetParmVal(wing_id, "TotalArea", "WingGeom")
    bref = vsp.GetParmVal(wing_id, "TotalSpan", "WingGeom")

    # corda média geométrica aproximada
    cref = (2/3) * croot * ((1 + taper + taper**2) / (1 + taper))

    # Condições de voo (imperial)
    W = 1800 * 2.20462        # [lbf]
    rho = 0.002377            # [slug/ft^3]
    T = 288.15
    gamma = 1.4
    R = 287.05
    M = 0.3

    a_SI = (gamma * R * T) ** 0.5   # [m/s]
    V_SI = M * a_SI                 # [m/s]
    V_ft = V_SI / 0.3048            # [ft/s]
    Sref = sref                     # [ft^2]

    hist_path = r"C:\VSP\Development\PSO_PYTHON_WING\cessna_updated.history"

    print(f"[flight] Mach={M:.2f}  →  V={V_SI:.2f} m/s ({V_ft:.1f} ft/s)")

    # ==================== AUTO-ALPHA AJUSTADO ====================
    # Para quando L estiver dentro de ±2.5% do peso
    target_L = W
    alpha = 0.0
    step = 0.4

    for _ in range(10):

        # Rodar solver com alpha atual
        vsp.SetIntAnalysisInput(solver_id, "NumWakeNodes", [32])
        vsp.SetIntAnalysisInput(solver_id, "NCPU", [4])
        vsp.SetDoubleAnalysisInput(solver_id, "Sref", [Sref])
        vsp.SetDoubleAnalysisInput(solver_id, "Rho",  [rho])
        vsp.SetDoubleAnalysisInput(solver_id, "Vinf", [V_ft])
        vsp.SetDoubleAnalysisInput(solver_id, "MachStart", [M])
        vsp.SetDoubleAnalysisInput(solver_id, "MachEnd",   [M])
        vsp.SetIntAnalysisInput(solver_id, "MachNpts",  [1])
        vsp.SetDoubleAnalysisInput(solver_id, "AlphaStart", [alpha])
        vsp.SetDoubleAnalysisInput(solver_id, "AlphaEnd",   [alpha])
        vsp.SetIntAnalysisInput(solver_id, "AlphaNpts", [1])
        vsp.SetIntAnalysisInput(solver_id, "GeomSet", [vsp.SET_ALL])

        vsp.ExecAnalysis(solver_id)

        # Espera o history ser criado
        for _ in range(10):
            if os.path.exists(hist_path):
                break
            time.sleep(0.5)

        # Lê o history correspondente ao alpha atual
        with open(hist_path, "r") as f:
            lines = [l.strip() for l in f.readlines()
                    if l.strip() and not l.startswith("#")]

        last_line = lines[-1].split()

        cl = float(last_line[6])
        cdo_vsp = float(last_line[7])
        cdi = float(last_line[8])
        cd_vsp_tot = float(last_line[9])

        # Calcula sustentação do alpha atual
        L = 0.5 * rho * V_ft**2 * Sref * cl
        error = (L - target_L) / target_L

        # Critério de parada corrigido (±2.5%)
        if abs(error) < 0.025:
            print(f"[auto-alpha] Convergência atingida com alpha={alpha:.3f}°, erro={error*100:.2f}%")
            break

        # Ajusta alpha no sentido correto
        alpha -= step * error

    print(f"[auto-alpha] Alpha final = {alpha:.3f}°, Sustentação = {L:.2f} lbf")

    # ==================== CÁLCULOS FINAIS BASEADOS NO ALPHA IDEAL ====================
    # 6.1) MODELO DE ARRASTO PARASITA (CD0) - COM FATOR DE CORREÇÃO
    #   - K = 1 / (pi * e * AR)
    #   - Sref (área da asa)
    # Todos os outros termos (T, W, rho, sigma, Vmax) são tratados como constantes
    # e fixados em 1, de modo que eles SE CANCELAM ao formar a razão:
    #   CD0_new / CD0_base
    # Assim obtemos um fator F que ajusta o CD0_BASE da asa original
    # para a geometria atual (AR, Sref).

    # --- 1) Coeficiente K (novo e base) ---
    K_new = 1.0 / (np.pi * E_OSWALD * AR)
    K_base = 1.0 / (np.pi * E_OSWALD * AR_BASE)

    # --- 2) CD0 "simplificado" para asa atual e asa base ---
    # Forma genérica simplificada a partir da expressão de Roskam:
    #   CD0 ~ (1 - 4*K/(rho*V^2*S)) / (rho*V^2*S)
    # Com rho*V^2 -> 1 e demais termos constantes -> 1,
    # fica algo proporcional a: (1 - 4*K/S) / S
    CD0_new_simpl  = (1.0 - 4.0 * K_new  / Sref)     / Sref
    CD0_base_simpl = (1.0 - 4.0 * K_base / SREF_BASE) / SREF_BASE

    # --- 3) Fator de correção entre CD0 da asa atual e da asa base ---
    if CD0_new_simpl <= 0.0 or CD0_base_simpl <= 0.0:
        # Segurança numérica: se der algo estranho, não corrige
        F_corr = 1.0
    else:
        F_corr = CD0_new_simpl / CD0_base_simpl

    # (Opcional) limitar o fator para evitar exageros numéricos
    F_corr = max(0.5, min(1.5, F_corr))


    # --- 4) CD0 parasita final usado no modelo ---
    cd0_parasita = CD0_BASE * F_corr

    # CD total "físico" usado para L/D
    cd_total = cd0_parasita + cdi
    ld = cl / cd_total


    print(f"[coeffs] CL={cl:.5f}, CDi={cdi:.5f}, CD0_parasita={cd0_parasita:.5f}, CD_total={cd_total:.5f}, L/D={ld:.2f}")
    print(f"[raw_vspaero] CDo_vsp={cdo_vsp:.5f}, CDtot_vsp={cd_vsp_tot:.5f}")

    # Sustentação final
    L = 0.5 * rho * V_ft**2 * Sref * cl

    # Faixa de sustentação
    L_min = W * 0.975
    L_max = W * 1.025

    if L < L_min or L > L_max:
        penalty = 1000 * abs((L - W) / W) ** 2
        print(f"[penalty] L fora da faixa: {L:.1f} lbf (peso={W:.1f} lbf, penalidade={penalty:.2f})")
    else:
        penalty = 0.0

    print(f"[ok] CL={cl:.4f}, CD_total={cd_total:.4f}, L={L:.2f} lbf, L/D={ld:.2f}")
    print(f"[status] Sustentação {'OK' if penalty == 0 else 'fora'} | α={alpha:.2f}°")
    print("[solver] VSPAERO executado.")

    # ============================================================
    # 7) FUNÇÃO OBJETIVO
    # ============================================================
    fobj = -ld + penalty

    vsp.ClearVSPModel()
    time.sleep(1)
    gc.collect()

    print(f"[done] Iteração finalizada: fobj={fobj:.4f}, L/D={ld:.2f}")

    # Retorno final
    return fobj, {
        "CL": cl,
        "CD_total": cd_total,
        "CDi": cdi,
        "CD0_parasita": cd0_parasita,
        "LD": ld,
        "Alpha": alpha,
        "L": L,
        "Sref": Sref,          # <- ADICIONAR
        "V_ft": V_ft,          # (opcional) se quiser bater V
        "rho": rho, 
        "Fcorr": F_corr            # (opcional) se quiser bater densidade
    }

#######################################################################################
if __name__ == "__main__":
    #"AR", "span", "taper", "sweep", "twist"
    x_test = np.array([7.5, 36, 1, 0, 2])
    fobj, data = FCN(x_test)

    AR, span, taper, sweep, twist = x_test

    # Recalcula cordas só para mostrar
    croot = 2 * span / (AR * (1 + taper))
    ctip  = taper * croot

    Sref = data["Sref"]

    # Valores usados na simulação (mesmos do FCN)
    W_lbf = 1800 * 2.20462
    L_lbf = data["L"]
    LW_percent = (L_lbf / W_lbf) * 100

    # Velocidade e densidade (iguais ao FCN)
    rho = 0.002377                 # slug/ft³
    T = 288.15
    gamma = 1.4
    R = 287.05
    M = 0.3

    a_SI = (gamma * R * T) ** 0.5
    V_SI = M * a_SI
    V_ft = V_SI / 0.3048

    print("\n================ RESULTADO DO TESTE ================\n")

    print("Parâmetros usados:")
    print(f"  AR       = {AR}")
    print(f"  span     = {span}")
    print(f"  taper    = {taper}")
    print(f"  sweep    = {sweep}")
    print(f"  twist    = {twist}")
    print(f"  croot    = {croot:.3f}")
    print(f"  ctip     = {ctip:.3f}")
    print(f"  Área (Sref) = {Sref:.3f} ft²\n")

    print("Condições de voo:")
    print(f"  Velocidade (ft/s) = {V_ft:.2f}")
    print(f"  Densidade (slug/ft³) = {rho:.6f}\n")

    print("Resultados aerodinâmicos:")
    print(f"  CLtot        = {data['CL']:.5f}")
    print(f"  CDtotal      = {data['CD_total']:.5f}")
    print(f"  L/D          = {data['LD']:.3f}")
    print(f"  Alpha        = {data['Alpha']:.3f}°")
    print(f"  Sustentação L = {L_lbf:.2f} lbf")
    print(f"  L/W (%)      = {LW_percent:.2f}%\n")
    print(f"[cd0_corr] Fator de correção F = {data['Fcorr']:.3f}")
    print(f"Função objetivo = {fobj:.5f}")
    print("\n====================================================\n")


