# ============================================================
# v3_cessna_opt.py – função objetivo para PSO (5 variáveis)
# Compatível com OpenVSP 3.45.2 (usa VSPAEROSweep com Npts=1)
# ============================================================

import os, sys
import numpy as np
from pathlib import Path

# === Caminhos ===
VSP3_FILE = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"

OPENVSP_PY = r"C:\VSP\OpenVSP\OpenVSP-3.45.2-win64\python"
if OPENVSP_PY not in sys.path:
    sys.path.insert(0, OPENVSP_PY)
from openvsp import openvsp as vsp

# === Constantes ===
MACH = 0.26
BETA = 0.0
CL_TARGET = 0.43
CL_TOL = 0.05
PENALIDADE = 1e6


def FCN(x: np.ndarray, return_LD_only=False) -> float:
    """Avalia uma configuração de asa no OpenVSP/VSPAERO."""
    try:
        sweep_deg, twist_deg, taper, span_total_ft, alpha_deg = x

        # --- Carrega geometria ---
        vsp.ClearVSPModel()
        vsp.ReadVSPFile(VSP3_FILE)

        # --- Identifica a asa ---
        wing_id = None
        for gid in vsp.FindGeoms():
            if "wing" in vsp.GetGeomName(gid).lower():
                wing_id = gid
                break
        if not wing_id:
            print("[erro] Asa não encontrada.")
            return PENALIDADE

        # --- Aplica parâmetros geométricos ---
        vsp.SetParmVal(wing_id, "Sweep", "XSec_1", sweep_deg)
        vsp.SetParmVal(wing_id, "Twist", "XSec_1", twist_deg)
        vsp.SetParmVal(wing_id, "Taper", "XSec_1", taper)
        vsp.SetParmVal(wing_id, "Span", "XSec_1", span_total_ft / 2)  # unidades em pés (ft)

        vsp.Update()

        # --- Verifica dimensões geométricas ---
        sref = vsp.GetParmVal(wing_id, "TotalArea", "WingGeom")
        bref = vsp.GetParmVal(wing_id, "TotalSpan", "WingGeom")
        cref = vsp.GetParmVal(wing_id, "AvgChord", "WingGeom")
        print(f"[geom] Sref={sref:.3f} ft², Bref={bref:.3f} ft, Cref={cref:.3f} ft")

        # --- Inclui todos os geoms no Set 4 (VSPAERO) ---
        for gid in vsp.FindGeoms():
            try:
                vsp.SetSetFlag(gid, 4, True)
            except Exception:
                pass

        # --- Gera geometria degenerada ---
        vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
        vsp.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [vsp.SET_ALL])
        vsp.ExecAnalysis("VSPAEROComputeGeometry")

        # --- Configura e executa VSPAEROSweep (Npts=1) ---
        solver_id = "VSPAEROSweep"
        vsp.SetAnalysisInputDefaults(solver_id)
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

        res_id = vsp.ExecAnalysis(solver_id)

        # --- Extrai CL/CD do .history ---
        hist_files = list(Path(".").glob("*.history"))
        if not hist_files:
            print("[erro] Nenhum arquivo .history encontrado.")
            return PENALIDADE

        latest = max(hist_files, key=lambda f: f.stat().st_mtime)

        with open(latest, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]

        try:
            parts = lines[-1].split()
            cl = float(parts[5])   # CLtot
            cd = float(parts[8])   # CDtot
        except Exception as e:
            print(f"[erro] Falha ao ler CL/CD do .history ({latest}): {e}")
            return PENALIDADE

        ld = cl / cd if cd > 0 else 0.0
        print(f"[ok] Lendo arquivo .history: {latest.name}")
        print(f"[ok] CL={cl:.4f}, CD={cd:.4f}, L/D={ld:.2f}")

        if not (CL_TARGET * (1 - CL_TOL) <= cl <= CL_TARGET * (1 + CL_TOL)):
            print(f"[penalidade] CL fora do intervalo: {cl:.4f}")
            return PENALIDADE

        return ld if return_LD_only else -ld

    except Exception as e:
        print(f"[erro] Exceção geral na função FCN: {e}")
        return PENALIDADE


if __name__ == "__main__":
    x0 = np.array([10.0, 2.0, 0.8, 36.0, 2.0])  # [Sweep, Twist, Taper, Span_total(ft), Alpha]
    FCN(x0)
