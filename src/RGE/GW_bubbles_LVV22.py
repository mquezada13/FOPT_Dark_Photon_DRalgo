# GW_bubbles_LVV22.py
# Espectro GW de *burbujas* (sin fluido) para FOPT fuertemente superenfriadas
# Basado en Lewicki & Vaskonen (arXiv:2208.11697).
# Entradas: Tn, alpha, beta/H. Salida: callable h2Ω_GW(f) hoy.
#
# Fórmulas usadas:
# - Forma espectral (Tabla I + Ec. (24), (25))
# - Redshift a hoy (Ec. (27), (28))
# - Eficiencia en paredes en R_eff (Ec. (23))


from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict

# ------------------ helpers cosmológicos ------------------
def _h_star(T: float, gstar: float) -> float:
    """h_*(T) en Hz. Ec. (28)."""
    return 1.65e-5 * (T / 100.0) * (gstar / 100.0)**(1.0/6.0)

# ------------------ presets LVV22 (SOLO burbujas) ------------------
@dataclass(frozen=True)
class BubbleParams:
    A: float
    a: float
    b: float
    c: float
    fbeta_over_beta: float   # f_p/β ; en la tabla dan 2π f_p/β → aquí ya dividido por 2π
    beta_Reff: float         # β R_eff (última fila de la Tabla I)

# Valores centrales de la Tabla I (columna "Bubbles")
# Dos casos: T_rr ∝ R^{-2} (simetría global) y T_rr ∝ R^{-3} (simetría gauge)
BUBBLE_PRESETS: Dict[str, BubbleParams] = {
    # T_rr ∝ R^{-2}
    "bubbles_Rm2": BubbleParams(A=5.93e-2, a=1.03, b=1.84, c=1.45,
                                fbeta_over_beta=0.64/(2*np.pi), beta_Reff=5.07),
    # T_rr ∝ R^{-3}  (USUAL EN GAUGE; probablemente lo que necesitas)
    "bubbles_Rm3": BubbleParams(A=5.13e-2, a=2.41, b=2.42, c=4.08,
                                fbeta_over_beta=0.77/(2*np.pi), beta_Reff=4.81),
}

# ------------------ fórmulas LVV22 (solo burbujas) ------------------
def _S_shape_f_over_beta(f_over_beta: np.ndarray | float, P: BubbleParams) -> np.ndarray:
    """
    Ec. (24): forma broken power-law en función de x = f / f_beta.
    f_beta ≡ f_p aquí. La tabla entrega f_p/β (convertido ya a nuestra convención).
    """
    x = np.asarray(f_over_beta, dtype=float) / P.fbeta_over_beta
    num = P.A * (P.a + P.b)**P.c
    den = (P.b * x**(-P.a/P.c) + P.a * x**(P.b/P.c))**P.c
    return num / den

def _kappa_at_Reff(P: BubbleParams, beta_R_eq: float = 50.0) -> float:
    """
    Ec. (23): κ(R) ≈ 1 / (1 + R/R_eq), evaluada en R = R_eff.
    Usamos los βR_eff tabulados y un βR_eq elegible (por defecto 5).
    """
    return 1.0 / (1.0 + (P.beta_Reff / float(beta_R_eq)))


# ------------------ API principal ------------------
def spectrum_bubbles(Tn: float, alpha: float, beta_over_H: float,
                     preset_key: str = "bubbles_Rm3",
                     beta_R_eq: float = 5.0,
                     g_star: float | None = None) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """
    Construye el callable h^2Ω_GW(f) hoy para *burbujas* LVV22.

    Parámetros:
      - Tn          : temperatura (GeV) que usarás como T_* (paper define h_* con T_*).
      - alpha       : fuerza de la transición.
      - beta_over_H : β/H (tu dato).
      - preset_key  : "bubbles_Rm3" (gauge) o "bubbles_Rm2" (global).
      - beta_R_eq   : controla κ(R_eff) vía Ec. (23). Valor típico ~5.
      - g_star      : g_* en T_*. Si None, intenta leer de constants.g_dof; si falla, usa 100.

    Devuelve: función h2Ω(f_Hz).
    """
    # g_*: usa constants.g_dof si existe, si no 100

    import constants as cs
    g_star = float(cs.g_dof)


    if preset_key not in BUBBLE_PRESETS:
        raise ValueError(f"preset_key inválido '{preset_key}'. Opciones: {list(BUBBLE_PRESETS.keys())}")
    P = BUBBLE_PRESETS[preset_key]

    # h_* con T_* = Tn (el paper define h_* en la temperatura de la transición)
    hstar = _h_star(Tn, g_star)  # Hz

    # prefactor fuente (Ec. (25)) solo con paredes; κ en R_eff:
    kappa_eff = _kappa_at_Reff(P, beta_R_eq=beta_R_eq)
    pref_src = (beta_over_H**-2) * ((kappa_eff * alpha) / (1.0 + alpha))**2

    # redshift a hoy en unidades h^2Ω (Ec. (27))
    omega_fac = 1.67e-5 * (100.0/g_star)**(1.0/3.0)

    def h2Omega_today(f_Hz: np.ndarray | float) -> np.ndarray:
        # escala de frecuencia: f / (h_* β/H)
        f_over_beta = np.asarray(f_Hz, dtype=float) / (hstar * beta_over_H)
        shape = _S_shape_f_over_beta(f_over_beta, P)
        omega_src = pref_src * shape


        return omega_fac * omega_src

    return h2Omega_today

def f_peak_today(Tn: float, beta_over_H: float,
                 preset_key: str = "bubbles_Rm3",
                 g_star: float | None = None) -> float:
    """
    Frecuencia de pico hoy: f_p,0 = h_* (f_p/β) (β/H). Útil para marcar el pico en plots.
    """
    if g_star is None:
        try:
            import constants as cs
            g_star = float(cs.g_dof)
        except Exception:
            g_star = 100.0
    if preset_key not in BUBBLE_PRESETS:
        raise ValueError(f"preset_key inválido '{preset_key}'. Opciones: {list(BUBBLE_PRESETS.keys())}")
    P = BUBBLE_PRESETS[preset_key]
    return _h_star(Tn, g_star) * P.fbeta_over_beta * beta_over_H

# ------------------ ejemplo ultra-corto ------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Números de juguete, reemplaza por tus tripletas reales
    Tn, alpha, betaH = 1.0e-2, 10.0, 40.0
    h2 = spectrum_bubbles(Tn, alpha, betaH, preset_key="bubbles_Rm3", beta_R_eq=5.0)
    f = np.logspace(-14, -6, 400)
    y = np.array([h2(fi) for fi in f])

    fp = f_peak_today(Tn, betaH, "bubbles_Rm3")
    fig, ax = plt.subplots(figsize=(7,5.5), dpi=120)
    ax.plot(f, y, lw=2.0, label="LVV22 bubbles R$^{-3}$")
    ax.axvline(fp, ls="--", color="k", alpha=0.6, label=r"$f_{\rm peak}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$f\,[{\rm Hz}]$"); ax.set_ylabel(r"$h^2\Omega_{\rm GW}(f)$")
    ax.set_xlim(1e-11, 1e-6); ax.set_ylim(1e-14, 1e-6)
    ax.legend(frameon=False); plt.tight_layout(); plt.show()
