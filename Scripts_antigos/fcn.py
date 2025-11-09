# ============================================================
# fcn.py - Função objetivo para otimização via PSO + VSPAERO
# ============================================================
# Autor: Gregori Maia da Silva
# Compatível com OpenVSP 3.45.2
# ============================================================

import os, sys, shutil, glob
import numpy as np

# === Caminhos principais ===
OPENVSP_PY = r"C:\VSP\OpenVSP\OpenVSP-3.45.2-win64\python"
VSP3_FILE  = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"
OUT_DIR    = r"C:\VSP\Development\PSO_PYTHON_WING\result"

# === Condições fixas de voo ===
MACH   = 0.26
BETA   = 0.0
NCPU   = 4
CL_TARGET = 0.43
CL_TOL = 0.05 * CL_TARGET   # ±5%

# Adiciona OpenVSP ao path
sys.path.insert(0, OPENVSP_PY)
import openvsp as vsp


# ============================================================
# FCN(x): função objetivo chamada pelo PSO
# ------------------------------------------------------------
# x = [alpha, sweep, twist, taper, span]
# Retorna o valor negativo de L/D (pois o PSO minimiza)
# ============================================================

def FCN(x, return_LD_only=False, iter_num=None):
    """
    Executa uma simulação VSPAERO com os parâmetros definidos
    e retorna o valor de -L/D (para o PSO).
    """

    # --- Organização das pastas de saída ---
    iter_dir = os.path.join(OUT_DIR, "iterations")
    os.makedirs(iter_dir, exist_ok=True)
    if iter_num is None:
        iter_num = len(glob.glob(os.path.join(iter_dir, "iter_*"))) + 1
    run_dir = os.path.join(iter_dir, f"iter_{iter_num:03d}")
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    # --- Limpa arquivos antigos ---
    for ext in ("*.history", "*.polar", "*.adb", "*.log", "*.dbg"):
        for f in glob.glob(ext):
            try: os.remove(f)
            except: pass

    # --- Lê o modelo base ---
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(VSP3_FILE)
    vsp.Update()

    # --- Identifica a asa ---
    wing_id = None
    for gid in vsp.FindGeoms():
        name = vsp.GetGeomName(gid).lower()
        if "wing" in name:
            wing_id = gid
            break
    if wing_id is None:
        print("[ERRO] Asa não encontrada no modelo.")
        return 1e6

    # --- Aplica as variáveis do PSO ---
    alpha, sweep, twist, taper, span = x

    # Define valores geométricos
    vsp.SetParmVal(wing_id, "Sweep", "XSec_1", sweep)
    vsp.SetParmVal(wing_id, "Twist", "XSec_1", twist)
    vsp.SetParmVal(wing_id, "Taper", "XSec_1", taper)
    vsp.SetParmVal(wing_id, "Span",  "XSec_1", span)
    vsp.Update()

    # --- Configura análise VSPAERO ---
    vsp.SetAnalysisInputDefaults("VSPAERO")
    vsp.SetDoubleAnalysisInput("VSPAERO", "Mach", [MACH])
    vsp.SetDoubleAnalysisInput("VSPAERO", "Beta", [BETA])
    vsp.SetIntAnalysisInput("VSPAERO", "NCPU", [NCPU])
    vsp.SetDoubleAnalysisInput("VSPAERO", "AlphaStart", [alpha])
    vsp.SetDoubleAnalysisInput("VSPAERO", "AlphaEnd", [alpha])
    vsp.SetIntAnalysisInput("VSPAERO", "AlphaNpts", [1])
    vsp.Update()

    # --- Executa análise ---
    print(f"[Iteração {iter_num}] Rodando VSPAERO... α={alpha:.2f}, sweep={sweep:.2f}, twist={twist:.2f}, taper={taper:.2f}, span={span:.2f}")
    try:
        vsp.ExecAnalysis("VSPAERO")
    except Exception as e:
        print(f"[ERRO] Falha na execução do VSPAERO: {e}")
        return 1e6

    # --- Procura arquivo .history ---
    history_files = glob.glob("*.history")
    if not history_files:
        print("[ERRO] Nenhum arquivo .history encontrado.")
        return 1e6

    history_file = history_files[0]

    # --- Lê CL, CD e L/D diretamente do arquivo ---
    with open(history_file, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    try:
        last_line = lines[-1].split()
        alpha_val = float(last_line[0])
        cl = float(last_line[1])
        cd = float(last_line[2])
        ld = float(last_line[4]) if len(last_line) >= 5 else cl / cd
    except Exception as e:
        print(f"[ERRO] Falha ao ler .history: {e}")
        return 1e6

    # --- Penalidade de CL fora da faixa ---
    cl_min = CL_TARGET - CL_TOL
    cl_max = CL_TARGET + CL_TOL
    penalty = 0.0
    if cl < cl_min or cl > cl_max:
        penalty = 1e3 * abs(cl - CL_TARGET) / CL_TARGET

    # --- Valor objetivo ---
    f_obj = -ld + penalty

    # --- Retorno simples se só quiser L/D ---
    if return_LD_only:
        return ld

    # --- Log da iteração ---
    with open(os.path.join(run_dir, "result.txt"), "w") as f:
        f.write(f"Iteração {iter_num}\n")
        f.write(f"Alpha = {alpha:.3f}°\nSweep = {sweep:.3f}°\nTwist = {twist:.3f}°\n")
        f.write(f"Taper = {taper:.3f}\nSpan = {span:.3f} ft\n")
        f.write(f"CL = {cl:.4f}\nCD = {cd:.5f}\nL/D = {ld:.4f}\nPenalty = {penalty:.3f}\n")

    print(f"[Iteração {iter_num}] L/D={ld:.3f}, CL={cl:.3f}, CD={cd:.4f}, Penalidade={penalty:.2f}")

    return f_obj
