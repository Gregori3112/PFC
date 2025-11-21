# ============================================================
# v12_cessna_opt.py
# ------------------------------------------------------------
# Função FCN: executa uma simulação completa no OpenVSP + VSPAERO
# dado um vetor de parâmetros geométricos. Retorna:
#     - fobj = função objetivo (-L/D + penalidade)
#     - dicionário com CL, CD, L/D, alpha e L
#
# Esta função é utilizada diretamente pelo PSO.
# ------------------------------------------------------------
# Autor: Gregori da Maia da Silva
# ============================================================


import os
import sys
import time
import gc
import numpy as np
from openvsp import openvsp as vsp


# ============================================================
# FCN(x) — Função Objetivo da Otimização
# ------------------------------------------------------------
# Entrada:
#   x = vetor numpy com 5 variáveis:
#       AR, span, taper, sweep, twist
#
# Processos internos:
#   1. Carrega o modelo base (.vsp3)
#   2. Aplica geometria no modelo
#   3. Gera malha (ComputeGeometry)
#   4. Executa solver VSPAERO
#   5. Ajuste automático de alpha para garantir L ≈ W
#   6. Lê arquivo .history
#   7. Calcula CL, CD, L/D e penalidades
#
# Saída:
#   fobj = -L/D + penalidade
#   dict  { "CL":..., "CD":..., "LD":..., "Alpha":..., "L":... }
# ============================================================
def FCN(x: np.ndarray):
    """
    Função objetivo para o PSO. Recebe um vetor de variáveis geométricas,
    aplica no modelo OpenVSP, executa VSPAERO e retorna o desempenho aerodinâmico.
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

    # Remove qualquer modelo carregado anteriormente
    vsp.ClearVSPModel()

    vsp.ReadVSPFile(VSP3_FILE)

    # Captura os IDs reais da geometria
    geom_ids = vsp.FindGeoms()

    # Busca a asa principal automaticamente
    wing_id = None
    for gid in geom_ids:
        if vsp.GetGeomTypeName(gid) == "Wing":
            wing_id = gid
            break

    if wing_id is None:
        raise RuntimeError("ERRO: Nenhuma asa encontrada no modelo!")

    print(f"Wing ID encontrado: {wing_id}")


    # ID da asa principal (fixo dentro do modelo)
    #wing_id = "ITDQSYJOYI"

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

    # Aplica os parâmetros geométricos no VSP
    # Atenção: o VSP trabalha com semi-envergadura → span/2
    vsp.SetParmVal(wing_id, "Span",       "XSec_1", span / 2.0)
    vsp.SetParmVal(wing_id, "Root_Chord", "XSec_1", croot)
    vsp.SetParmVal(wing_id, "Tip_Chord",  "XSec_1", ctip)
    vsp.SetParmVal(wing_id, "Taper",      "XSec_1", taper)
    vsp.SetParmVal(wing_id, "Sweep",      "XSec_1", sweep)
    vsp.SetParmVal(wing_id, "Twist",      "XSec_1", twist)

    # Atualiza visualização e parâmetros internos do VSP
    vsp.Update()

    print(f"[geo] AR={AR:.2f}, Span={span:.2f}, Taper={taper:.2f}, Sweep={sweep:.2f}, Twist={twist:.2f}")
    print(f"[geo] Croot={croot:.3f}, Ctip={ctip:.3f}")


    # ============================================================
    # 3) GERA MALHA + EXECUTA SOLVER VSPAERO
    # ============================================================

    # Salva temporariamente um vsp3 com a geometria atualizada
    vsp.Update()
    vsp.WriteVSPFile(r"C:\VSP\Development\PSO_PYTHON_WING\cessna_updated.vsp3")

    # ------------------------------------------------------------
    # 3.1) Geração de malha (Degenerate Geometry)
    # ------------------------------------------------------------
    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [vsp.SET_ALL])
    vsp.ExecAnalysis("VSPAEROComputeGeometry")

    # ------------------------------------------------------------
    # 3.2) Configuração do Solver Aerodinâmico
    # ------------------------------------------------------------
    vsp.SetAnalysisInputDefaults(solver_id)  # reseta opções



    # Evita erros com arquivos opcionais que o solver não precisa gerar
    available_inputs = []
    try:
        available_inputs = vsp.GetAnalysisInputNames(solver_id)
    except:
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

    # Dados de referência geométricos calculados pelo OpenVSP
    sref = vsp.GetParmVal(wing_id, "TotalArea", "WingGeom")
    bref = vsp.GetParmVal(wing_id, "TotalSpan", "WingGeom")

    # Cordas médias para referência aerodinâmica
    cref = (2/3) * croot * ((1 + taper + taper**2) / (1 + taper))

    # Condições de voo (usar unidades IMPERIAIS para bater com o VSPAERO)
    W = 1800 * 2.20462        # peso total [lbf]
    rho = 0.002377            # densidade [slug/ft^3]
    T = 288.15
    gamma = 1.4
    R = 287.05
    M = 0.3

    # Velocidade do som em m/s e depois converte para ft/s
    a_SI = (gamma * R * T) ** 0.5       # [m/s]
    V_SI = M * a_SI                     # [m/s]
    V_ft = V_SI / 0.3048                # [ft/s]
    # Área de referência em ft² (mesma que o solver usa)
    Sref = sref                         # [ft^2]

    # Caminho do history produzido pelo solver
    hist_path = r"C:\VSP\Development\PSO_PYTHON_WING\cessna_updated.history"

    print(f"[flight] Mach={M:.2f} → V={V_SI:.2f} m/s ({V_ft:.1f} ft/s)")

    # ============================================================
    # 5) AJUSTE AUTOMÁTICO DE ALPHA PARA GARANTIR L ≈ W
    # ------------------------------------------------------------
    target_L = W
    alpha    = 0.0      # valor inicial (graus)
    step     = 0.5      # passo de correção

    for _ in range(8):   # número de ciclos de ajuste (rápido)
        
        # Configura solver
        vsp.SetIntAnalysisInput(solver_id, "NumWakeNodes", [32])
        vsp.SetIntAnalysisInput(solver_id, "NCPU", [4])
        vsp.SetDoubleAnalysisInput(solver_id, "Sref", [Sref])   # ft^2
        vsp.SetDoubleAnalysisInput(solver_id, "Rho",  [rho])    # slug/ft^3
        vsp.SetDoubleAnalysisInput(solver_id, "Vinf", [V_ft])   # ft/s
        vsp.SetDoubleAnalysisInput(solver_id, "MachStart", [M])
        vsp.SetDoubleAnalysisInput(solver_id, "MachEnd", [M])
        vsp.SetIntAnalysisInput(solver_id, "MachNpts", [1])
        vsp.SetDoubleAnalysisInput(solver_id, "AlphaStart", [alpha])
        vsp.SetDoubleAnalysisInput(solver_id, "AlphaEnd",   [alpha])
        vsp.SetIntAnalysisInput(solver_id, "AlphaNpts", [1])
        vsp.SetIntAnalysisInput(solver_id, "GeomSet", [vsp.SET_ALL])

        # Executa solver
        vsp.ExecAnalysis(solver_id)

        # Aguarda history ser criado

        for _ in range(10):
            if os.path.exists(hist_path):
                break
            time.sleep(0.5)

        # Lê arquivo history
        with open(hist_path, "r") as f:
            lines = [l.strip() for l in f.readlines()
                     if l.strip() and not l.startswith("#")]

        last_line = lines[-1].split()
        cl = float(last_line[6])


        # Cálculo da sustentação
        L = 0.5 * rho * V_ft**2 * Sref * cl   # L em lbf
        error = (L - target_L) / target_L

        if abs(error) < 0.01:  # convergiu
            break

        # Ajusta alpha com correção proporcional
        alpha -= step * error

    print(f"[auto-alpha] Alpha ajustado para {alpha:.2f}° com L={L:.1f} lbf")


    # ============================================================
    # 6) LEITURA DOS RESULTADOS FINAIS DO HISTORY
    # ============================================================

    with open(hist_path, "r") as f:
        lines = [l.strip() for l in f.readlines()
                if l.strip() and not l.startswith("#")]

    last_line = lines[-1].split()

    # Leitura dos coeficientes aerodinâmicos
    cl = float(last_line[6])      # CLtot
    cd = float(last_line[9])      # CDtot = CDo + CDi
    ld = cl / cd

    print(f"[coeffs] CL={cl:.5f}, CD={cd:.5f}, L/D={ld:.2f}")

    # Sustentação final
    L = 0.5 * rho * V_ft**2 * Sref * cl    # lbf

    # Verificação da faixa de sustentação
    L_min = W * 0.95
    L_max = W * 1.05

    if L < L_min or L > L_max:
        penalty = 1000 * abs((L - W) / W) ** 2
        print(f"[penalty] L fora da faixa: {L:.1f} lbf (peso={W:.1f} lbf, penalidade={penalty:.2f})")
    else:
        penalty = 0

    # Exibe resultados
    print(f"[ok] CL={cl:.4f}, CD={cd:.4f}, L={L:.2f} lbf, L/D={ld:.2f}")
    print(f"[status] Sustentação {'OK' if penalty == 0 else 'fora'} | α={alpha:.2f}°")
    print("[solver] VSPAERO executado.")



    # ============================================================
    # 7) FUNÇÃO OBJETIVO
    # ============================================================
    # O PSO minimiza → -L/D garante maximização de L/D
    fobj = -ld + penalty

    # ------------------------------------------------------------
    # Limpeza final da memória interna do OpenVSP
    # ------------------------------------------------------------

    vsp.ClearVSPModel()
    time.sleep(1)
    gc.collect()

    print(f"[done] Iteração finalizada: fobj={fobj:.4f}, L/D={ld:.2f}")

    # Retorno final da função objetivo e dados aerodinâmicos
    return fobj, {"CL": cl, "CD": cd, "LD": ld, "Alpha": alpha, "L": L}



# ============================================================
# EXECUÇÃO ISOLADA (TESTE)
# ============================================================
if __name__ == "__main__":
    x_test = np.array([7.5, 36.0, 1.0, 0.0, 0.0])
    resultado = FCN(x_test)
    print("Resultado do teste:", resultado)



