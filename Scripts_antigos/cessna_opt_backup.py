# ============================================================
# cessna_opt.py – versão estável (histórico-only)
# ============================================================

import os, sys, time, glob

# === CONFIGURAÇÃO PRINCIPAL ===
OPENVSP_PY = r"C:\VSP\OpenVSP\OpenVSP-3.45.2-win64\python"
VSP3_FILE  = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"

# Condições de voo
MACH    = 0.26
AOA_DEG = 2.0
NCPU    = 4
RHO     = None   # use None para deixar padrão do VSPAERO

# --- Importa OpenVSP ---
sys.path.insert(0, OPENVSP_PY)
import openvsp.vsp as v


# ============================================================
# HELPERS
# ============================================================

def _chdir_to_model_dir():
    os.chdir(os.path.dirname(VSP3_FILE))

def _find_first_wing():
    for gid in v.FindGeoms():
        name = v.GetGeomName(gid) or ""
        if "wing" in name.lower():
            return gid, name
    return None, None

def _collect_section_parm_ids(wing_id, names, limit_surfs=8):
    out = {n: [] for n in names}
    for isurf in range(limit_surfs):
        try:
            xsid = v.GetXSecSurf(wing_id, isurf)
            nx = v.GetNumXSec(xsid)
            for i in range(nx):
                xid = v.GetXSec(xsid, i)
                for nm in names:
                    pid = v.GetXSecParm(xid, nm)
                    if pid:
                        out[nm].append(pid)
        except Exception:
            continue
    return out

def _set_uniform(pids, value):
    for pid in pids:
        try:
            v.SetParmVal(pid, float(value))
        except Exception:
            pass

def _apply_geometry(wing_id, sweep_deg, twist_deg):
    pids = _collect_section_parm_ids(wing_id, ["Sweep", "Twist"])
    if pids["Sweep"]:
        _set_uniform(pids["Sweep"], sweep_deg)
    if pids["Twist"]:
        _set_uniform(pids["Twist"], twist_deg)
    v.Update()

def _compute_vspaero_geometry():
    v.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    v.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [v.SET_ALL])
    v.ExecAnalysis("VSPAEROComputeGeometry")

def _run_vspaero_single_alpha(mach, alpha_deg, ncpu=4, rho=None):
    v.SetAnalysisInputDefaults("VSPAEROSweep")
    v.SetIntAnalysisInput("VSPAEROSweep", "GeomSet", [v.SET_ALL])
    v.SetIntAnalysisInput("VSPAEROSweep", "NCPU", [int(ncpu)])
    v.SetDoubleAnalysisInput("VSPAEROSweep", "MachStart", [mach])
    v.SetDoubleAnalysisInput("VSPAEROSweep", "MachEnd", [mach])
    v.SetIntAnalysisInput("VSPAEROSweep", "MachNpts", [1])
    v.SetDoubleAnalysisInput("VSPAEROSweep", "AlphaStart", [alpha_deg])
    v.SetDoubleAnalysisInput("VSPAEROSweep", "AlphaEnd", [alpha_deg])
    v.SetIntAnalysisInput("VSPAEROSweep", "AlphaNpts", [1])
    if rho:
        v.SetDoubleAnalysisInput("VSPAEROSweep", "Rho", [rho])
    v.ExecAnalysis("VSPAEROSweep")


# ============================================================
# PARSER DO .HISTORY
# ============================================================

import re

def _parse_history_generic(path):
    """
    Lê corretamente o arquivo .history do VSPAERO (ignorando cabeçalhos iniciais).
    Extrai CL, CD e L/D do bloco numérico final.
    """
    if not os.path.exists(path):
        return None

    header = None
    last_data = None
    in_table = False  # começa ignorando o cabeçalho inicial

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*") or s.startswith("#"):
                continue

            # Começa a leitura só quando encontrar a linha de colunas ("Iter" ou "CLtot")
            if not in_table:
                if "Iter" in s or "CLtot" in s or "CDtot" in s or "L/D" in s:
                    in_table = True
                else:
                    continue  # ainda na parte inicial
                # continua para processar a linha do cabeçalho
            parts = s.split()
            if header is None and not all(re.match(r"^[\d\.\-]+$", p) for p in parts):
                # Limpa caracteres estranhos (underscores, barras, etc.)
                header = [re.sub(r"[^A-Za-z0-9/]", "", p) for p in parts]
                continue
            # Se for linha numérica, guarda a última
            if all(re.match(r"^[\d\.\-]+$", p) for p in parts):
                last_data = [float(p) for p in parts]

    if header is None or last_data is None:
        print("[parser] Nenhum bloco numérico identificado após cabeçalho.")
        return None

    norm_header = [h.lower() for h in header]

        # Normaliza o cabeçalho para facilitar busca
    norm_header = [h.lower() for h in header]

    def find_col_exact(target):
        """Retorna índice de uma coluna que combina exatamente com o nome (ex: 'l/d')"""
        for i, h in enumerate(header):
            if h.strip().lower() == target.lower().replace("/", ""):
                return i
        return None

    def find_col_any(name_list, exclude=None):
        """Procura colunas que contenham qualquer termo (ignorando as da lista exclude)"""
        for name in name_list:
            for i, h in enumerate(norm_header):
                if name in h:
                    if exclude and any(ex in h for ex in exclude):
                        continue
                    return i
        return None

    # 1️⃣ tenta pegar coluna exatamente "L/D"
    i_ld = find_col_any(["l/d"], exclude=["low", "w", "lodw"])
    i_ld_exact = find_col_exact("l/d")
    if i_ld_exact is not None:
        i_ld = i_ld_exact

    # 2️⃣ se não houver, tenta CL/CD
    i_cl = find_col_any(["cltot", "cltotal"])
    i_cd = find_col_any(["cdtot", "cdtotal"])

    # 3️⃣ cálculo do L/D
    if i_ld is not None and i_ld < len(last_data):
        ld = last_data[i_ld]
        print(f"[history] L/D principal = {ld:.3f}")
        return {"L_over_D": ld}

    if i_cl is not None and i_cd is not None and last_data[i_cd] != 0:
        cl = last_data[i_cl]
        cd = last_data[i_cd]
        ld = cl / cd
        print(f"[history] CL={cl:.5f}, CD={cd:.5f}, L/D={ld:.3f}")
        return {"L_over_D": ld}

    print(f"[parser] Falha ao localizar L/D principal. Header detectado: {header}")
    return None





# ============================================================
# FUNÇÃO OBJETIVO PARA O PSO
# ============================================================

def FCN(x):
    sweep, twist = float(x[0]), float(x[1])
    print(f"\n[simulação] Sweep={sweep:.2f}°, Twist={twist:.2f}°")

    try:
        _chdir_to_model_dir()
        t0 = time.time()

        v.ClearVSPModel()
        v.ReadVSPFile(VSP3_FILE)
        wing_id, name = _find_first_wing()
        if not wing_id:
            print("[ERRO] Nenhuma asa encontrada.")
            return 1e6

        _apply_geometry(wing_id, sweep, twist)
        _compute_vspaero_geometry()
        _run_vspaero_single_alpha(MACH, AOA_DEG, ncpu=NCPU, rho=RHO)

        # localiza .history gerado após execução
        hist_candidates = [p for p in glob.glob("*.history") if os.path.getmtime(p) > t0]
        if not hist_candidates:
            print("[erro] Nenhum arquivo .history encontrado.")
            return 1e6

        hist_file = max(hist_candidates, key=os.path.getmtime)
        res = _parse_history_generic(hist_file)
        if not res:
            print("[erro] Falha ao ler .history.")
            return 1e6

        ld = res["L_over_D"]
        if not (0 < ld < 300):
            print(f"[penalidade] L/D fora do plausível: {ld:.2f}")
            return 1e6

        print(f"[resultado] L/D = {ld:.3f}")
        return -ld

    except Exception as e:
        print(f"[erro FCN] {e}")
        return 1e6


# ============================================================
# Teste manual
# ============================================================
if __name__ == "__main__":
    val = FCN([10.0, -3.0])
    print("Função objetivo =", val)
