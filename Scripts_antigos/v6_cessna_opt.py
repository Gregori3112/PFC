# ============================================================
# v5_cessna_opt.py – função objetivo robusta para PSO (5 variáveis)
# Compatível com OpenVSP 3.45.2 e VSPAEROSweep (Npts=1)
# ============================================================

import os, sys
import numpy as np
from pathlib import Path
import warnings
import contextlib 
import glob


# === Gerenciador para ocultar prints do terminal ===
@contextlib.contextmanager
def suppress_all_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)

# === Caminhos ===
VSP3_FILE = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"
OPENVSP_PY = r"C:\VSP\OpenVSP\OpenVSP-3.45.2-win64\python"

if OPENVSP_PY not in sys.path:
    sys.path.insert(0, OPENVSP_PY)
from openvsp import openvsp as vsp

# === Constantes ===
MACH = 0.26
BETA = 0.0
CL_TARGET = 0.38
CL_TOL = 0.1
PENALIDADE = 1e6

ALPHA_MIN, ALPHA_MAX = -5.0, 10.0
SPAN_MIN, SPAN_MAX = 15.0, 45.0
TAPER_MIN, TAPER_MAX = 0.3, 1.5


def FCN(x: np.ndarray, return_LD_only=False) -> float:

    for ext in ("*.history", "*.adb", "*.log", "*.polar"):
        for f in glob.glob(ext):
            os.remove(f)

    """Avalia uma configuração de asa no OpenVSP/VSPAERO."""
    warnings.filterwarnings("ignore", message="Could not open Polar file")

    try:
        sweep_deg, twist_deg, taper, span_total_ft, alpha_deg = x

        alpha_deg = np.clip(alpha_deg, ALPHA_MIN, ALPHA_MAX)
        span_total_ft = np.clip(span_total_ft, SPAN_MIN, SPAN_MAX)
        taper = np.clip(taper, TAPER_MIN, TAPER_MAX)

        # === Configuração da geometria ===
        with suppress_all_output():
            vsp.ClearVSPModel()
            vsp.ReadVSPFile(VSP3_FILE)
    

            wing_id = None
            for gid in vsp.FindGeoms():
                if "wing" in vsp.GetGeomName(gid).lower():
                    wing_id = gid
                    break
            if not wing_id:
                return PENALIDADE

            vsp.SetParmVal(wing_id, "Sym_Planar_Flag", "XForm", 1)
            vsp.SetParmVal(wing_id, "Sym_Planar_Axis", "XForm", 2)
            vsp.Update()

            vsp.SetParmVal(wing_id, "Sweep", "XSec_1", sweep_deg)
            vsp.SetParmVal(wing_id, "Twist", "XSec_1", twist_deg)

            # --- Ajuste do Taper (compatível com OpenVSP 3.45.2) ---
            xsec_surf_id = vsp.GetXSecSurf(wing_id, 0)
            xsec_id = vsp.GetXSec(xsec_surf_id, 1)

            # Obtém o ID do parâmetro Taper diretamente via nome
            taper_parm = vsp.GetParm(xsec_id, "Taper", "XSecCurve_0")
            if taper_parm != "":
                vsp.SetParmValUpdate(taper_parm, taper)
            else:
                vsp.SetParmVal(wing_id, "Taper", "XSec_1", taper)  # fallback


            vsp.SetParmVal(wing_id, "Span", "XSec_1", span_total_ft / 2)
            vsp.Update()

            vsp.WriteVSPFile("debug_current.vsp3")

            sref = vsp.GetParmVal(wing_id, "TotalArea", "WingGeom")
            bref = vsp.GetParmVal(wing_id, "TotalSpan", "WingGeom")
            if sref < 1 or bref < 5:
                return PENALIDADE

            for gid in vsp.FindGeoms():
                try:
                    vsp.SetSetFlag(gid, 4, True)
                except Exception:
                    pass

            with suppress_all_output():    
                # --- Geometria degenerada ---
                vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
                vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [vsp.SET_ALL])
                vsp.ExecAnalysis("VSPAEROComputeGeometry")

            # === Configuração do solver ===
            with suppress_all_output():
                vsp.Update()
                solver_id = "VSPAEROSweep"
                # --- Define referências e condições de voo coerentes com a GUI ---
                vsp.SetAnalysisInputDefaults(solver_id)
                vsp.SetDoubleAnalysisInput(solver_id, "Sref", [sref])      # área de referência da asa
                vsp.SetDoubleAnalysisInput(solver_id, "Cref", [bref / 9.0]) # estimativa de corda média (ajuste fino opcional)
                vsp.SetDoubleAnalysisInput(solver_id, "Bref", [bref])      # envergadura total
                vsp.SetDoubleAnalysisInput(solver_id, "Rho", [0.002377])   # densidade ar nível do mar [slug/ft³]
                vsp.SetDoubleAnalysisInput(solver_id, "Vinf", [100.0])     # velocidade de escoamento [ft/s]
                vsp.SetIntAnalysisInput(solver_id, "RefFlag", [1])         # usa Sref/Bref/Cref definidos acima
                vsp.SetIntAnalysisInput(solver_id, "GeomSet", [vsp.SET_ALL])
                vsp.SetIntAnalysisInput(solver_id, "NCPU", [4])
                vsp.SetDoubleAnalysisInput(solver_id, "MachStart", [MACH])
                vsp.SetDoubleAnalysisInput(solver_id, "MachEnd", [MACH])
                vsp.SetIntAnalysisInput(solver_id, "MachNpts", [1])
                vsp.SetDoubleAnalysisInput(solver_id, "AlphaStart", [alpha_deg])
                vsp.SetDoubleAnalysisInput(solver_id, "AlphaEnd", [alpha_deg])
                vsp.SetIntAnalysisInput(solver_id, "AlphaNpts", [1])
                vsp.SetDoubleAnalysisInput(solver_id, "BetaStart", [BETA])
                vsp.SetDoubleAnalysisInput(solver_id, "BetaEnd", [BETA])
                vsp.SetIntAnalysisInput(solver_id, "BetaNpts", [1])
                vsp.SetIntAnalysisInput(solver_id, "WakeNumIter", [5])
                vsp.SetIntAnalysisInput(solver_id, "NumWakeNodes", [64])
                vsp.SetIntAnalysisInput(solver_id, "Symmetry", [0])
                vsp.SetIntAnalysisInput(solver_id, "StallModel", [0])
                vsp.SetDoubleAnalysisInput(solver_id, "CLmax", [1.4])

            # --- Limpa arquivos antigos ---
            for ext in ("*.history", "*.adb", "*.log", "*.polar"):
                for f in Path(".").glob(ext):
                    try:
                        f.unlink()
                    except Exception:
                        pass

            # --- Executa solver ---
            with suppress_all_output():
                res_id = vsp.ExecAnalysis(solver_id)


        # === Leitura do arquivo .history ===
        hist_files = list(Path(".").glob("*.history"))
        if not hist_files:
            print("[erro] Nenhum arquivo .history encontrado.")
            return PENALIDADE

        latest = max(hist_files, key=lambda f: f.stat().st_mtime)

        with open(latest, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]

        try:
            parts = lines[-1].split()
            cl = float(parts[6])
            cd = float(parts[9])

            # --- Validação dos coeficientes ---
            if abs(cl) > 2.0 or abs(cd) > 1.0 or cd <= 0 or not np.isfinite(cl / cd):
                print(f"[penalidade] Valores inválidos: CL={cl:.4f}, CD={cd:.4f}")
                return PENALIDADE

        except Exception:
            print(f"[erro] Falha ao ler CL/CD do arquivo {latest.name}")
            return PENALIDADE

        ld = cl / cd
        print(f"[ok] CL={cl:.4f}, CD={cd:.4f}, L/D={ld:.2f}")

        if abs(cl - CL_TARGET) > CL_TOL:
            print(f"[penalidade] CL fora do intervalo: {cl:.4f}")
            return PENALIDADE

        return ld if return_LD_only else -ld

    except Exception as e:
        print(f"[erro] Exceção geral na função FCN: {e}")
        return PENALIDADE

    print(f"[debug] Sweep={sweep_deg:.2f}, Twist={twist_deg:.2f}, Taper={taper:.2f}, Span={span_total_ft:.2f}, Alpha={alpha_deg:.2f}, L/D={ld:.2f}")



# === Teste local ===
if __name__ == "__main__":
    #sweep_deg, twist_deg, taper, span_total_ft, alpha_deg
    x0 = np.array([30, 0, 1, 40, 2.5])
    ld_val = FCN(x0, return_LD_only=True)
    print(f"[result] Teste concluído. x0 = {x0.tolist()}")
    print(f"[result] L/D calculado = {ld_val:.2f}")
