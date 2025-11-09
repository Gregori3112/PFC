# ============================================================
# cessna_opt.py – versão estável (OpenVSP 3.45.2 compatível)
# ============================================================

import os, sys, time, glob, re

# --- Caminhos principais ---
OPENVSP_PY = r"C:\VSP\OpenVSP\OpenVSP-3.45.2-win64\python"
VSP3_FILE  = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"

# --- Condições de voo ---
MACH    = 0.26
AOA_DEG = 2.0
NCPU    = 4
RHO     = None

sys.path.insert(0, OPENVSP_PY)
import openvsp.vsp as v


# ============================================================
# FUNÇÕES AUXILIARES
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


def _apply_geometry(wing_id, sweep_deg, twist_deg, taper, span_m):
    """Aplica sweep, twist, taper e span à asa"""
    pids = _collect_section_parm_ids(wing_id, ["Sweep", "Twist", "Taper"])
    if pids["Sweep"]:
        _set_uniform(pids["Sweep"], sweep_deg)
    if pids["Twist"]:
        _set_uniform(pids["Twist"], twist_deg)
    if pids["Taper"]:
        _set_uniform(pids["Taper"], taper)

    try:
        pid_span = v.GetParm(wing_id, "TotalSpan", "WingGeom")
        v.SetParmVal(pid_span, float(span_m))
    except Exception as e:
        print(f"[aviso] Span não ajustado: {e}")

    v.Update()


def _sync_vspaero_refs(wing_id):
    """Atualiza Sref/Bref/Cref com base na geometria atual"""
    try:
        # --- Atualiza geometria ---
        v.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
        v.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [v.SET_ALL])
        v.ExecAnalysis("VSPAEROComputeGeometry")

        # --- Obtém área e envergadura ---
        pid_S = v.GetParm(wing_id, "TotalArea", "WingGeom")
        pid_b = v.GetParm(wing_id, "TotalSpan", "WingGeom")
        S = v.GetParmVal(pid_S)
        b = v.GetParmVal(pid_b)
        c = S / b if b > 0 else 1.0

        # --- Corrigido: FindContainer requer índice ---
        veh_id = v.FindContainer("Vehicle", 0)
        if veh_id is None:
            print("[erro refs] Container Vehicle não encontrado")
            return

        # Tenta definir parâmetros de referência (podem não existir em todas as versões)
        ref_params = [("Sref", S), ("Bref", b), ("Cref", c)]
        for pname, val in ref_params:
            try:
                pid = v.GetParm(veh_id, pname, "Vehicle")
                if pid:
                    v.SetParmVal(pid, float(val))
                    print(f"[refs] {pname}={val:.3f} definido")
                else:
                    print(f"[aviso] Parâmetro {pname} não encontrado - usando valores padrão")
            except Exception as e:
                print(f"[aviso] {pname} não disponível: {e}")

        v.Update()
        print(f"[refs] Sref={S:.3f}, Bref={b:.3f}, Cref={c:.3f}")

    except Exception as e:
        print(f"[erro refs] Falha ao atualizar Sref/Bref/Cref: {e}")


def _compute_vspaero_geometry():
    """Gera geometria VSPAERO atualizada"""
    v.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    v.SetIntAnalysisInput("VSPAEROComputeGeometry", "GeomSet", [v.SET_ALL])
    v.ExecAnalysis("VSPAEROComputeGeometry")


def _run_vspaero_single_alpha(mach, alpha_deg, ncpu=4, rho=None):
    """Executa VSPAERO em modo viscoso"""
    v.SetAnalysisInputDefaults("VSPAEROSweep")
    v.SetIntAnalysisInput("VSPAEROSweep", "GeomSet", [v.SET_ALL])
    v.SetIntAnalysisInput("VSPAEROSweep", "NCPU", [int(ncpu)])
    v.SetDoubleAnalysisInput("VSPAEROSweep", "MachStart", [mach])
    v.SetDoubleAnalysisInput("VSPAEROSweep", "MachEnd", [mach])
    v.SetIntAnalysisInput("VSPAEROSweep", "MachNpts", [1])
    v.SetDoubleAnalysisInput("VSPAEROSweep", "AlphaStart", [alpha_deg])
    v.SetDoubleAnalysisInput("VSPAEROSweep", "AlphaEnd", [alpha_deg])
    v.SetIntAnalysisInput("VSPAEROSweep", "AlphaNpts", [1])

    # ✅ modo viscoso (usando parâmetros corretos)
    try:
        v.SetIntAnalysisInput("VSPAEROSweep", "ViscousFlag", [1])
        v.SetDoubleAnalysisInput("VSPAEROSweep", "ReCref", [9e6])
        print("[VSPAERO] Executando em modo viscoso (ViscousFlag=1)")
    except Exception as e:
        print(f"[aviso] Parâmetros viscosos não disponíveis: {e}")
        print("[VSPAERO] Executando em modo invíscido")

    if rho:
        v.SetDoubleAnalysisInput("VSPAEROSweep", "Rho", [rho])

    v.ExecAnalysis("VSPAEROSweep")


# ============================================================
# PARSER DO .HISTORY
# ============================================================

def _parse_history_generic(path):
    if not os.path.exists(path):
        return None

    header = None
    last_data = None
    in_table = False
    all_data_lines = []  # Armazena todas as linhas de dados

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*") or s.startswith("#"):
                continue

            if not in_table:
                if "Iter" in s or "CLtot" in s or "CDtot" in s or "L/D" in s:
                    in_table = True
                else:
                    continue

            parts = s.split()
            if header is None and not all(re.match(r"^[\d\.\-]+$", p) for p in parts):
                header = [re.sub(r"[^A-Za-z0-9/]", "", p) for p in parts]
                continue
            if all(re.match(r"^[\d\.\-]+$", p) for p in parts):
                all_data_lines.append([float(p) for p in parts])
    
    # Pega a ÚLTIMA linha de dados (mais convergida)
    if all_data_lines:
        last_data = all_data_lines[-1]

    if header is None or last_data is None:
        print("[parser] Nenhum bloco numérico identificado.")
        return None

    norm_header = [h.lower() for h in header]

    def find_col_any(name_list, exclude=None):
        for name in name_list:
            for i, h in enumerate(norm_header):
                if name in h:
                    if exclude and any(ex in h for ex in exclude):
                        continue
                    return i
        return None

    # L/D está na coluna 14 (índice 13) do arquivo .history
    if len(last_data) > 13:
        ld = last_data[13]  # Coluna 14 = L/D
        print(f"[history] L/D = {ld:.3f}")
        return {"L_over_D": ld}
    else:
        print(f"[parser] Dados insuficientes: {len(last_data)} colunas")
        return None
    
    # Fallback: procura por L/D no header
    i_ld = find_col_any(["l/d"], exclude=["lodw", "ldw", "low"])
    i_cl = find_col_any(["cltot", "cltotal"])
    i_cd = find_col_any(["cdtot", "cdtotal"])

    if i_ld is not None:
        ld = last_data[i_ld]
        print(f"[history] L/D = {ld:.3f}")
        return {"L_over_D": ld}

    if i_cl is not None and i_cd is not None and last_data[i_cd] != 0:
        ld = last_data[i_cl] / last_data[i_cd]
        print(f"[history] CL/CD = {ld:.3f}")
        return {"L_over_D": ld}

    return None


# ============================================================
# FUNÇÃO OBJETIVO
# ============================================================

def FCN(x):
    # Validação de entrada
    if len(x) != 4:
        print("[erro] Vetor de entrada deve ter 4 elementos")
        return 1e6
    
    try:
        sweep, twist, taper, span = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        
        # Validação de limites
        if not (0 <= sweep <= 25):
            print(f"[penalidade] Sweep fora dos limites: {sweep}")
            return 1e6
        if not (-6 <= twist <= 6):
            print(f"[penalidade] Twist fora dos limites: {twist}")
            return 1e6
        if not (0.3 <= taper <= 1.5):
            print(f"[penalidade] Taper fora dos limites: {taper}")
            return 1e6
        if not (8 <= span <= 14):
            print(f"[penalidade] Span fora dos limites: {span}")
            return 1e6
            
        print(f"\n[simulação] Sweep={sweep:.2f}°, Twist={twist:.2f}°, Taper={taper:.3f}, Span={span:.2f} m")
    except (ValueError, TypeError) as e:
        print(f"[erro] Parâmetros inválidos: {e}")
        return 1e6

    try:
        _chdir_to_model_dir()
        for f in glob.glob("*.history"):
            os.remove(f)

        t0 = time.time()
        v.ClearVSPModel()
        v.ReadVSPFile(VSP3_FILE)
        wing_id, name = _find_first_wing()
        if not wing_id:
            print("[ERRO] Nenhuma asa encontrada.")
            return 1e6

        _apply_geometry(wing_id, sweep, twist, taper, span)
        _sync_vspaero_refs(wing_id)
        _compute_vspaero_geometry()
        _run_vspaero_single_alpha(MACH, AOA_DEG, ncpu=NCPU, rho=RHO)

        # Aguarda um pouco para garantir que o arquivo foi criado
        time.sleep(0.5)
        
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
        if not (0 < ld < 120):
            print(f"[penalidade] L/D fora do plausível: {ld:.2f}")
            return 1e6

        print(f"[resultado] L/D = {ld:.3f}")
        return -ld

    except Exception as e:
        print(f"[erro FCN] {e}")
        return 1e6


# ============================================================
# TESTE DIRETO
# ============================================================

if __name__ == "__main__":
    val = FCN([10.0, -3.0, 0.4, 11.0])
    print("Função objetivo =", val)
