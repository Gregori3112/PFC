# ============================================================
# Script de análise e otimização da asa Cessna 210 via OpenVSP + VSPAERO
# ------------------------------------------------------------
# Autor: Gregori da Maia da Silva
# Função principal (FCN): calcula o L/D a partir de parâmetros geométricos
# ============================================================

import os, sys, numpy as np
from openvsp import openvsp as vsp


def FCN(x: np.ndarray) -> float:

    """
    Função objetivo que executa uma simulação VSPAERO
    com base em variáveis geométricas da asa e retorna o valor -L/D
    (para ser usado num algoritmo de otimização, como o PSO).
    """

    # === 1️⃣ CARREGAMENTO DO MODELO BASE ===
    # Caminho do arquivo do modelo original (.vsp3)
    VSP3_FILE = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"


    # Limpa qualquer modelo aberto anteriormente e lê o novo   
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(VSP3_FILE)

    # ID fixo da asa principal e nome do solver
    wing_id = "ITDQSYJOYI"
    solver_id = "VSPAEROSweep"


    # ============================================================
    # 2️⃣ APLICA AS VARIÁVEIS NO MODELO GEOMÉTRICO
    # ============================================================
    # As variáveis de entrada representam:
    # AR = razão de aspecto, span = envergadura total, taper = razão de afilamento,
    # sweep = enflechamento, twist = torção, alpha = ângulo de ataque.
    AR, span, taper, sweep, twist, alpha = x

    # --- Calcula cordas coerentes com o AR e taper definidos ---
    croot = 2 * span / (AR * (1 + taper))
    ctip  = taper * croot

    # --- Aplica parâmetros geométricos diretamente no modelo OpenVSP ---
    vsp.SetParmVal(wing_id, "Span",       "XSec_1", span / 2.0)  # Span/2 pois cada semi-asa
    vsp.SetParmVal(wing_id, "Root_Chord", "XSec_1", croot)
    vsp.SetParmVal(wing_id, "Tip_Chord",  "XSec_1", ctip)
    vsp.SetParmVal(wing_id, "Taper", "XSec_1", taper) 
    vsp.SetParmVal(wing_id, "Sweep",      "XSec_1", sweep)
    vsp.SetParmVal(wing_id, "Twist",      "XSec_1", twist)


    # Atualiza o modelo na memória do VSP
    vsp.Update()


    print(f"[geo] AR={AR:.2f}, Span={span:.2f}, Taper={taper:.2f}, Sweep={sweep:.2f}, Twist={twist:.2f}")
    print(f"[geo] Croot={croot:.3f}, Ctip={ctip:.3f}")


    # ============================================================
    # 3️⃣ EXECUÇÃO DO SOLVER AERODINÂMICO (VSPAERO)
    # ============================================================

    # 3.1 Gera geometria degenerada (malha usada pelo solver)
    vsp.Update()
    vsp.WriteVSPFile(r"C:\VSP\Development\PSO_PYTHON_WING\cessna_updated.vsp3")
    

    # Configura o gerador de malha (VSPAEROComputeGeometry)    
    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [vsp.SET_ALL])
    vsp.ExecAnalysis("VSPAEROComputeGeometry")


    # ------------------------------------------------------------
    # 3.2 Configura o solver VSPAERO propriamente dito
    # ------------------------------------------------------------

    vsp.SetAnalysisInputDefaults("VSPAEROSweep")  # reseta parâmetros do solver

    # ------------------------------------------------------------
    # 3.3 Define parâmetros principais de análise
    # ------------------------------------------------------------
    # Calcula área, envergadura e corda média para entradas do solver
    sref = vsp.GetParmVal(wing_id, "TotalArea", "WingGeom")
    bref = vsp.GetParmVal(wing_id, "TotalSpan", "WingGeom")
    cref = (2/3) * croot * ((1 + taper + taper**2) / (1 + taper))

    # Define condições de voo e propriedades do ar
    vsp.SetDoubleAnalysisInput(solver_id, "Sref", [sref])
    vsp.SetDoubleAnalysisInput(solver_id, "Rho", [0.002377])     # densidade do ar (slug/ft³)
    vsp.SetDoubleAnalysisInput(solver_id, "Vinf", [100.0])       # velocidade (ft/s)
    vsp.SetDoubleAnalysisInput(solver_id, "MachStart", [0.3])
    vsp.SetDoubleAnalysisInput(solver_id, "MachEnd", [0.3])
    vsp.SetIntAnalysisInput(solver_id, "MachNpts", [1])
    vsp.SetDoubleAnalysisInput(solver_id, "AlphaStart", [alpha])
    vsp.SetDoubleAnalysisInput(solver_id, "AlphaEnd", [alpha])
    vsp.SetIntAnalysisInput(solver_id, "AlphaNpts", [1])
    vsp.SetIntAnalysisInput(solver_id, "GeomSet", [vsp.SET_ALL])
    vsp.SetIntAnalysisInput(solver_id, "NCPU", [4])

    # Executa o solver aerodinâmico
    vsp.Update()
    vsp.ExecAnalysis(solver_id)


    # ============================================================
    # 4️⃣ LEITURA E AVALIAÇÃO DOS RESULTADOS
    # ============================================================

    # Caminho do arquivo de resultados (.history)
    hist_path = r"C:\VSP\Development\PSO_PYTHON_WING\cessna_updated.history"


    # Lê o arquivo, ignorando linhas de cabeçalho
    with open(hist_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]

    # Pega a última linha (resultado final)
    last_line = lines[-1].split()


    # Extrai CL, CD e calcula razão L/D
    cl = float(last_line[6])  # coluna CLtot
    cd = float(last_line[9])  # coluna CDtot
    ld = cl / cd

    # ------------------------------------------------------------
    # 4.1 Aplica penalização caso CL saia da faixa aceitável
    # ------------------------------------------------------------
    cl_ref = 0.38               # valor alvo de CL para cruzeiro
    cl_min = cl_ref * 0.95
    cl_max = cl_ref * 1.05

    if cl < cl_min or cl > cl_max:
        penalty = 1000 * abs(cl - cl_ref)
        print(f"[penalty] CL fora do intervalo: {cl:.3f} (penalidade {penalty:.2f})")
    else:
        penalty = 0

    # Mostra resultados principais
    print(f"[ok] CL={cl:.4f}, CD={cd:.4f}, L/D={ld:.2f}")
    print("[solver] Simulação VSPAERO executada.")

    # ============================================================
    # 5️⃣ FUNÇÃO OBJETIVO PARA OTIMIZAÇÃO
    # ============================================================
    # O PSO tenta minimizar → então usamos -L/D para maximizar L/D
    fobj = -ld + penalty

    # --- Encerra e limpa completamente o modelo atual para liberar memória ---
    vsp.ClearVSPModel()
    vsp.Update()
    import gc
    gc.collect()


    return fobj


# ============================================================
# Execução isolada (modo de teste)
# ============================================================
if __name__ == "__main__":
    # Vetor de teste: [AR, span, taper, sweep, twist, alpha]
    x_test = np.array([8, 36.0, 0.5, 30, -6, 2.0])

    # Executa uma simulação e imprime o resultado final
    resultado = FCN(x_test)
    print("Resultado:", resultado)