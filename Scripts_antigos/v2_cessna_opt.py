# ============================================================
# v2_cessna_opt.py – função objetivo para PSO (5 variáveis)
# Compatível com OpenVSP 3.45.2 (usa VSPAEROSweep)
# ============================================================

import os, sys, glob, time, io, contextlib, subprocess
import numpy as np
from pathlib import Path

# === Caminhos ===
VSP3_FILE = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"
VSP_OUTDIR = os.path.join(os.path.dirname(__file__), "result")
ITER_OUTDIR = os.path.join(VSP_OUTDIR, "iteracoes")

OPENVSP_PY = r"C:\VSP\OpenVSP\OpenVSP-3.45.2-win64\python"
if OPENVSP_PY not in sys.path:
    sys.path.insert(0, OPENVSP_PY)
from openvsp import openvsp as vsp

# === Constantes ===
MACH = 0.26
RHO = 0.002377  # slug/ft³ (ISA nível do mar)
BETA = 0.0
CL_TARGET = 0.43
CL_TOL = 0.05
PENALIDADE = 1e6


# === Função de limpeza ===
def limpar_resultados():
    for f in glob.glob(os.path.join(VSP_OUTDIR, "*")):
        try:
            if os.path.isfile(f):
                os.remove(f)
        except PermissionError:
            pass


# === Função objetivo ===
def FCN(x: np.ndarray, return_LD_only=False) -> float:
    try:
        sweep_deg, twist_deg, taper, span_m, alpha_deg = x
        os.makedirs(VSP_OUTDIR, exist_ok=True)
        os.makedirs(ITER_OUTDIR, exist_ok=True)
        limpar_resultados()

        # --- Carrega geometria ---
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(VSP3_FILE)

        # --- Identifica asa ---
        wing_id = None
        for gid in vsp.FindGeoms():
            if "wing" in vsp.GetGeomName(gid).lower():
                wing_id = gid
                break
        if not wing_id:
            print("[erro] Asa não encontrada")
            return PENALIDADE

        # --- Aplica parâmetros geométricos ---
        vsp.SetParmVal(wing_id, "Sweep", "XSec_1", sweep_deg)
        vsp.SetParmVal(wing_id, "Twist", "XSec_1", twist_deg)
        vsp.SetParmVal(wing_id, "Taper", "XSec_1", taper)
        vsp.SetParmVal(wing_id, "Span", "XSec_1", span_m * 3.28084)
        vsp.Update()

        # --- Força inclusão no set VSPAERO ---
        all_geoms = vsp.FindGeoms()
        VSPAERO_SET = 4
        for gid in all_geoms:
            try:
                vsp.SetSetFlag(gid, VSPAERO_SET, True)
            except Exception:
                pass

        # --- Gera geometria e atualiza Sref/Bref/Cref ---
        vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
        vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [vsp.SET_ALL])
        # --- (1) Gera a geometria para o solver ---
        vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
        vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [vsp.SET_ALL])
        vsp.ExecAnalysis("VSPAEROComputeGeometry")

        # Caminho do .adb
        adb_path = os.path.join(VSP_OUTDIR, "cessna210_DegenGeom.adb")

        # --- (2) Executa o solver VSPAERO externamente e salva o log ---
        vspaero_exe = r"C:\VSP\OpenVSP\OpenVSP-3.45.2-win64\vspaero.exe"
        log_path = os.path.join(VSP_OUTDIR, "vspaero_log.txt")

        cmd = [vspaero_exe, adb_path, "-mach", str(MACH), "-alpha", str(alpha_deg), "-stab", "-noplot"]

        with open(log_path, "w") as log:
            subprocess.run(cmd, cwd=VSP_OUTDIR, stdout=log, stderr=log)

        print(f"[ok] VSPAERO executado (log salvo em {log_path})")


        veh_id = vsp.FindContainer("Vehicle", 0)
        S = vsp.GetParmVal(vsp.GetParm(wing_id, "TotalArea", "WingGeom"))
        b = vsp.GetParmVal(vsp.GetParm(wing_id, "TotalSpan", "WingGeom"))
        c = S / b if b > 0 else 1.0
        for pname, val in [("Sref", S), ("Bref", b), ("Cref", c)]:
            pid = vsp.GetParm(veh_id, pname, "Vehicle")
            if pid:
                vsp.SetParmVal(pid, val)

        vsp.Update()

        # --- Executa solver (VSPAEROSweep) ---
        solver_id = "VSPAEROSweep"
        vsp.SetAnalysisInputDefaults(solver_id)
        vsp.SetIntAnalysisInput(solver_id, "GeomSet", [vsp.SET_ALL])
        vsp.SetDoubleAnalysisInput(solver_id, "MachStart", [MACH])
        vsp.SetDoubleAnalysisInput(solver_id, "MachEnd", [MACH])
        vsp.SetIntAnalysisInput(solver_id, "MachNpts", [1])
        vsp.SetDoubleAnalysisInput(solver_id, "AlphaStart", [alpha_deg])
        vsp.SetDoubleAnalysisInput(solver_id, "AlphaEnd", [alpha_deg])
        vsp.SetIntAnalysisInput(solver_id, "AlphaNpts", [1])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            vsp.ExecAnalysis(solver_id)

        # --- Aguarda o .history ---
        time.sleep(1)
        history_files = sorted(glob.glob("*.history"), key=os.path.getmtime, reverse=True)
        if not history_files:
            print("[erro] Nenhum arquivo .history encontrado.")
            return PENALIDADE

        hist_path = history_files[0]
        print(f"[ok] Lendo arquivo .history: {hist_path}")

        # Lê todas as linhas válidas
        valid_lines = []
        with open(hist_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip() and not line.startswith("#") and not line.startswith("*"):
                    parts = line.split()
                    if len(parts) > 10 and parts[0].isdigit():
                        valid_lines.append(parts)

        if not valid_lines:
            print("[erro] Nenhum dado numérico encontrado no .history.")
            return PENALIDADE

        last_line = valid_lines[-1]

        try:
            cl = float(last_line[6])  # coluna CLtot
            cd = float(last_line[9])  # coluna CDtot
            ld = cl / cd if cd > 0 else 0.0
        except Exception as e:
            print(f"[erro] Falha ao ler CL/CD: {e}")
            return PENALIDADE

        # --- Penalização e saída ---
        if not (CL_TARGET * (1 - CL_TOL) <= cl <= CL_TARGET * (1 + CL_TOL)):
            print(f"[penalidade] CL fora do intervalo: {cl:.4f}")
            return PENALIDADE

        print(f"[ok] CL={cl:.4f}, CD={cd:.4f}, L/D={ld:.2f}")
        return ld if return_LD_only else -ld

    except Exception as e:
        print("[erro] Exceção geral na função FCN:", e)
        return PENALIDADE


# === Teste direto ===
if __name__ == "__main__":
    x0 = np.array([5.0, -2.0, 0.8, 11.0, 2.0])
    FCN(x0)
