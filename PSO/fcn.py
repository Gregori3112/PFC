# ============================================================
# FCN.py  –  Versão Python equivalente ao FCN.m (Octave/MATLAB)
# ============================================================

import numpy as np

def FCN(x):
    """
    Equivalente à função FCN.m do MATLAB.
    Inclui as mesmas alternativas de funções de teste (comentadas).
    """

    # y = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2   # -3 a 3, f(1,1)=0  (Rosenbrock)
    y = 1 - ((1 - np.sin(np.sqrt(x[0]**2 + x[1]**2))**2) /
            (1 + 0.001 * (x[0]**2 + x[1]**2)))     # -10 a 10 (função senoidal global)
    #y = (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2   # -10 a 10, f(1,3)=0 (função quadrática)
    # y = -(1.0/((x - 0.3)**2 + 0.01) + 1.0/((x - 0.9)**2 + 0.04) - 6)  # 1D

    return float(y)
