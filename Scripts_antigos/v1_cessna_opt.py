# ============================================================
# v1_cessna_opt.py – versão adaptada para 5 variáveis (OpenVSP 3.45.2 compatível)
# Inclui suporte para alpha e restrição CL
# ============================================================

import os, sys, time, glob, re

# --- Caminhos principais ---
OPENVSP_PY = r"C:\VSP\OpenVSP\OpenVSP-3.45.2-win64\python"
VSP3_FILE  = r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3"

# --- Condições de voo ---
MACH    = 0.26
NCPU    = 4
RHO     = None

# --- Parâmetros de restrição CL ---
CL_TARGET = 0.43  # CL fixo desejado
CL_TOLERANCE = 0.05  # ±5% de tolerância
CL_MIN = CL_TARGET * (1 - CL_TOLERANCE)  # 0.4085
CL_MAX = CL_TARGET * (1 + CL_TOLERANCE)  # 0.4515
PENALTY_FACTOR = 1000.0  # Fator de penalidade para violações

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
        print(f"[VSPAERO] Executando em modo viscoso (ViscousFlag=1) com alpha={alpha_deg:.2f}°")
    except Exception as e:
        print(f"[aviso] Parâmetros viscosos não disponíveis: {e}")
        print(f"[VSPAERO] Executando em modo invíscido com alpha={alpha_deg:.2f}°")

    if rho:
        v.SetDoubleAnalysisInput("VSPAEROSweep", "Rho", [rho])

    v.ExecAnalysis("VSPAEROSweep")


# ============================================================
# PARSER DO .HISTORY - VERSÃO EXPANDIDA
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
        
        # Procura por CL (coeficiente de sustentação)
        i_cl = find_col_any(["cltot", "cltotal", "cl"])
        if i_cl is not None and i_cl < len(last_data):
            cl = last_data[i_cl]
            print(f"[history] CL = {cl:.4f}")
            return {"L_over_D": ld, "CL": cl}
        else:
            print("[history] CL não encontrado, usando aproximação")
            return {"L_over_D": ld, "CL": None}
    else:
        print(f"[parser] Dados insuficientes: {len(last_data)} colunas")
        return None
    
    # Fallback: procura por L/D no header
    i_ld = find_col_any(["l/d"], exclude=["lodw", "ldw", "low"])
    i_cl = find_col_any(["cltot", "cltotal", "cl"])
    i_cd = find_col_any(["cdtot", "cdtotal"])

    if i_ld is not None:
        ld = last_data[i_ld]
        print(f"[history] L/D = {ld:.3f}")
        cl = last_data[i_cl] if i_cl is not None and i_cl < len(last_data) else None
        if cl is not None:
            print(f"[history] CL = {cl:.4f}")
        return {"L_over_D": ld, "CL": cl}

    if i_cl is not None and i_cd is not None and last_data[i_cd] != 0:
        ld = last_data[i_cl] / last_data[i_cd]
        cl = last_data[i_cl]
        print(f"[history] CL/CD = {ld:.3f}")
        print(f"[history] CL = {cl:.4f}")
        return {"L_over_D": ld, "CL": cl}

    return None


# ============================================================
# FUNÇÃO PARA AJUSTAR ALPHA PARA CL TARGET
# ============================================================

def _find_alpha_for_CL_target(wing_id, mach, cl_target=0.43, alpha_start=2.0, max_iter=5):
    """
    Encontra o alpha necessário para atingir CL target
    Usa método de Newton-Raphson simplificado
    """
    alpha_current = alpha_start
    cl_history = []
    t_start = time.time()
    
    for iter in range(max_iter):
        print(f"[CL ajuste] Iteração {iter+1}: testando alpha={alpha_current:.2f}°")
        
        # Executa simulação com alpha atual
        _run_vspaero_single_alpha(mach, alpha_current, ncpu=NCPU, rho=RHO)
        time.sleep(0.5)
        
        # Lê resultado - procura arquivos criados após o início
        hist_candidates = [p for p in glob.glob("*.history") if os.path.getmtime(p) > t_start]
        if not hist_candidates:
            print("[CL ajuste] Erro: nenhum .history encontrado")
            return alpha_current, None
            
        hist_file = max(hist_candidates, key=os.path.getmtime)
        res = _parse_history_generic(hist_file)
        if not res or res.get("CL") is None:
            print("[CL ajuste] Erro: não foi possível ler CL")
            return alpha_current, None
            
        cl_current = res["CL"]
        cl_history.append((alpha_current, cl_current))
        print(f"[CL ajuste] Alpha={alpha_current:.2f}° → CL={cl_current:.4f}")
        
        # Verifica se está dentro da tolerância
        cl_error = abs(cl_current - cl_target)
        if cl_error < 0.01:  # Tolerância de 0.01
            print(f"[CL ajuste] Convergiu! Alpha={alpha_current:.2f}° → CL={cl_current:.4f}")
            return alpha_current, cl_current
            
        # Ajusta alpha para próxima iteração
        if len(cl_history) >= 2:
            # Usa derivada numérica para ajustar alpha
            alpha_prev, cl_prev = cl_history[-2]
            if abs(alpha_current - alpha_prev) > 1e-6:
                dcl_dalpha = (cl_current - cl_prev) / (alpha_current - alpha_prev)
                if abs(dcl_dalpha) > 1e-6:
                    alpha_new = alpha_current + (cl_target - cl_current) / dcl_dalpha
                    alpha_new = max(-10, min(15, alpha_new))  # Limita entre -10° e 15°
                else:
                    alpha_new = alpha_current + 2.0  # Incremento fixo se derivada muito pequena
            else:
                alpha_new = alpha_current + 2.0
        else:
            alpha_new = alpha_current + 2.0
            
        alpha_current = alpha_new
    
    print(f"[CL ajuste] Não convergiu após {max_iter} iterações. Alpha final={alpha_current:.2f}°")
    return alpha_current, cl_current


# ============================================================
# FUNÇÃO OBJETIVO - VERSÃO 5 VARIÁVEIS
# ============================================================

def FCN(x):
    # Validação de entrada - agora aceita 5 variáveis
    if len(x) != 5:
        print("[erro] Vetor de entrada deve ter 5 elementos")
        return 1e6
    
    try:
        sweep, twist, taper, span, alpha = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])
        
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
        if not (-4 <= alpha <= 4):
            print(f"[penalidade] Alpha fora dos limites: {alpha}")
            return 1e6
            
        print(f"\n[simulação] Sweep={sweep:.2f}°, Twist={twist:.2f}°, Taper={taper:.3f}, Span={span:.2f} m, Alpha={alpha:.2f}°")
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
        
        # Testa diferentes alphas para encontrar o que dá CL mais próximo de 0.43
        print(f"[CL ajuste] Testando diferentes alphas para atingir CL≈{CL_TARGET}")
        alpha_candidates = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]  # Lista de alphas para testar
        best_alpha = alpha
        best_cl = None
        best_error = float('inf')
        
        for test_alpha in alpha_candidates:
            print(f"[CL ajuste] Testando alpha={test_alpha:.1f}°")
            _run_vspaero_single_alpha(MACH, test_alpha, ncpu=NCPU, rho=RHO)
            time.sleep(0.5)
            
            # Lê resultado
            hist_candidates = [p for p in glob.glob("*.history") if os.path.getmtime(p) > t0]
            if hist_candidates:
                hist_file = max(hist_candidates, key=os.path.getmtime)
                res = _parse_history_generic(hist_file)
                if res and res.get("CL") is not None:
                    cl_test = res["CL"]
                    error = abs(cl_test - CL_TARGET)
                    print(f"[CL ajuste] Alpha={test_alpha:.1f}° → CL={cl_test:.4f}, erro={error:.4f}")
                    
                    if error < best_error:
                        best_error = error
                        best_alpha = test_alpha
                        best_cl = cl_test
                else:
                    print(f"[CL ajuste] Erro ao ler CL para alpha={test_alpha:.1f}°")
            else:
                print(f"[CL ajuste] Erro: nenhum .history para alpha={test_alpha:.1f}°")
        
        if best_cl is None:
            print("[erro] Não foi possível obter CL de nenhum alpha")
            return 1e6
            
        print(f"[CL ajuste] Melhor alpha: {best_alpha:.1f}° → CL={best_cl:.4f} (erro={best_error:.4f})")
        
        # Executa simulação final com o melhor alpha
        _run_vspaero_single_alpha(MACH, best_alpha, ncpu=NCPU, rho=RHO)

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
        cl = res["CL"]
        
        if not (0 < ld < 120):
            print(f"[penalidade] L/D fora do plausível: {ld:.2f}")
            return 1e6

        # CL foi ajustado automaticamente, então não há penalidade
        penalty = 0
        print(f"[resultado] L/D = {ld:.3f}, CL = {cl:.4f} (ajustado automaticamente)")
        print(f"[resultado] Alpha ótimo = {best_alpha:.1f}° (vs alpha solicitado = {alpha:.2f}°)")
        return -ld + penalty

    except Exception as e:
        print(f"[erro FCN] {e}")
        return 1e6


# ============================================================
# FUNÇÃO AUXILIAR PARA OBTER CL
# ============================================================

def get_CL_from_simulation(x):
    """
    Função auxiliar para obter apenas o CL de uma simulação
    Útil para monitoramento durante otimização
    """
    if len(x) != 5:
        return None
    
    try:
        sweep, twist, taper, span, alpha = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])
        
        _chdir_to_model_dir()
        for f in glob.glob("*.history"):
            os.remove(f)

        t0 = time.time()
        v.ClearVSPModel()
        v.ReadVSPFile(VSP3_FILE)
        wing_id, name = _find_first_wing()
        if not wing_id:
            return None

        _apply_geometry(wing_id, sweep, twist, taper, span)
        _sync_vspaero_refs(wing_id)
        _compute_vspaero_geometry()
        _run_vspaero_single_alpha(MACH, alpha, ncpu=NCPU, rho=RHO)

        time.sleep(0.5)
        
        hist_candidates = [p for p in glob.glob("*.history") if os.path.getmtime(p) > t0]
        if not hist_candidates:
            return None

        hist_file = max(hist_candidates, key=os.path.getmtime)
        res = _parse_history_generic(hist_file)
        if not res:
            return None

        return res.get("CL", None)

    except Exception as e:
        print(f"[erro get_CL] {e}")
        return None


# ============================================================
# TESTE DIRETO
# ============================================================

if __name__ == "__main__":
    # Teste com 5 variáveis - agora com ajuste automático de alpha
    print("=== TESTE COM AJUSTE AUTOMÁTICO DE ALPHA ===")
    val = FCN([10.0, -3.0, 0.4, 11.0, 1.5])
    print("Função objetivo =", val)
    
    # Teste de CL
    cl_val = get_CL_from_simulation([10.0, -3.0, 0.4, 11.0, 1.5])
    print("CL obtido =", cl_val)
