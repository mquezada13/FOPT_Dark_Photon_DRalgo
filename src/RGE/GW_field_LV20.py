# GW_field_LV20.py
# Espectro GW de colisiones de paredes (campo/runaway) siguiendo LV20 (arXiv:2007.04967).
# Módulo independiente para usar junto a tu repo actual.

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from math import pi
from scipy.integrate import quad

import constants as cs  # Debe contener: g_dof, Hhund, fmin, fmax, T_LISA, SNRthr, LV20_PRESETS

# ------------------ helpers básicos ------------------
def h_star(T: float) -> float:
    """h_*(T) en Hz (convención estándar LISA)."""
    return 1.65e-5 * (T / 100.0) * (cs.g_dof / 100.0)**(1.0/6.0)

def T_reh(Tn: float, alpha: float) -> float:
    """Estimación de T* para superenfriamiento fuerte."""
    return Tn * (1.0 + max(alpha, 0.0))**0.25

# ------------------ forma espectral LV20 ------------------
def lv20_shape(omega_over_beta: np.ndarray | float,
               A: float, omegabar_over_beta: float,
               a: float, b: float, c: float,
               d: Optional[float] = None,
               omegad_over_beta: Optional[float] = None) -> np.ndarray | float:
    """
    LV20: forma broken power-law (Eq. 19) para gradientes del campo.
    Notas:
      - Núcleo con potencia -c: produce pendientes bajas ~ x^{+a} y altas ~ x^{-b}.
      - En el caso U(1) (con 'd' y 'omegad'), el 'bracket extra' va en el **denominador**.
    """
    x = np.asarray(omega_over_beta, dtype=float) / omegabar_over_beta

    # Bracket U(1) (denominador, potencia -1):
    if (d is None) or (omegad_over_beta is None):
        pre = 1.0
    else:
        y = np.asarray(omega_over_beta, dtype=float) / omegad_over_beta  # omega/omega_d
        pre = (1.0 + (x**a)/(y**(d - a))) / (1.0 + (omegabar_over_beta/omegad_over_beta)**(d - a))
        pre = pre**(-1.0)

    core = (A * (a + b) / c) * (b * x**(-a/c) + a * x**(b/c))**(-c)
    return pre * core

def h2Omega_LV20(f: np.ndarray | float,
                 alpha: float, beta_over_H: float,
                 Tstar: float, gstar: float,
                 lv20_params: Dict[str, float],
                 h: float = 0.674) -> np.ndarray | float:
    """
    Espectro presente h^2 Omega_GW(f) para LV20 (Ecs. 20–21).
    """
    hstar = h_star(Tstar)  # Hz
    omega_over_beta = (2.0 * np.pi * np.asarray(f, dtype=float)) / (beta_over_H * hstar)
    S = lv20_shape(omega_over_beta, **lv20_params)
    pref = 1.67e-5 * (h**2) * (beta_over_H**-2) \
           * ((alpha/(1.0 + alpha))**2) * ((100.0/gstar)**(1.0/3.0))
    return pref * S

# ------------------ clase principal ------------------
class GWFieldLV20:
    """
    Builder de espectros LV20 (runaway/campo).
    Uso mínimo:
        gwf = GWFieldLV20(g_star=cs.g_dof, lv20_key="U1")   # o pasa lv20_params=...
        h2 = gwf.from_params(Tn, alpha, betaH, Tstar_mode="reh")   # devuelve callable h2(f)
        SNR = gwf.snr_lisa(h2)
    """
    def __init__(self,
                 g_star: float = cs.g_dof,
                 lv20_params: Optional[Dict[str, float]] = None,
                 lv20_key: str = "U1"):
        self.g_star = g_star
        if lv20_params is not None:
            self.params = lv20_params
        else:
            try:
                self.params = cs.LV20_PRESETS[lv20_key]
            except Exception as e:
                raise ValueError(f"No encuentro cs.LV20_PRESETS['{lv20_key}']. "
                                 f"Define LV20_PRESETS en constants.py o pasa lv20_params.") from e

    def spectrum(self, Tstar: float, alpha: float, beta_over_H: float) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """Devuelve h2(f) como callable para T*, alpha, beta/H dados."""
        def _h2(f):
            return h2Omega_LV20(f, alpha, beta_over_H, Tstar, self.g_star, self.params)
        return _h2

    def from_params(self, Tn: float, alpha: float, beta_over_H: float,
                    Tstar_mode: str = "reh") -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Construye el callable h2(f) a partir de (Tn, alpha, beta/H).
        - Tstar_mode:
            'reh' -> T* = T_reh(Tn, alpha)   (recomendado en superenfriamiento fuerte)
            'nuc' -> T* = Tn                  (útil para comparar con plantillas antiguas/PTA)
        """
        if Tstar_mode == "reh":
            Tstar = T_reh(Tn, alpha)
        elif Tstar_mode == "nuc":
            Tstar = Tn
        else:
            raise ValueError("Tstar_mode debe ser 'reh' o 'nuc'.")
        return self.spectrum(Tstar, alpha, beta_over_H)

    # ---------- LISA ----------
    @staticmethod
    def lisa_noise_h2Omega(f: float) -> float:
        """
        Tu misma función de ruido LISA (copiada para que el módulo sea autónomo).
        """
        H100 = cs.Hhund
        term1 = ((5.76e-48) / (2*np.pi*f)**4) * (1.0 + (0.0004/f)**2)
        term2 = 3.6e-41
        term3 = (1.0 + (f/0.025)**2)
        pref  = (4*np.pi**2)/(3*H100**2) * f**3 * (10.0/3.0)
        return pref * (term1 + term2) * term3

    def snr_lisa(self, h2_func: Callable[[float], float],
                 fmin: float = cs.fmin, fmax: float = cs.fmax,
                 TLISA: float = cs.T_LISA) -> float:
        integrand = lambda f: (h2_func(f) / self.lisa_noise_h2Omega(f))**2
        val, _ = quad(integrand, fmin, fmax, limit=2000)
        return float(np.sqrt(TLISA * val))

    def h2Omega_thr(self, p: float, f0: float = 1e-3) -> float:
        func = lambda f: (f / f0)**p
        return cs.SNRthr / self.snr_lisa(func)

# ------------------ helper alto nivel ------------------
def from_params(Tn: float, alpha: float, beta_over_H: float,
                Tstar_mode: str = "reh",
                g_star: float = cs.g_dof,
                lv20_key: str = "U1",
                lv20_params: Optional[Dict[str, float]] = None
               ) -> Tuple[Callable[[float | np.ndarray], float | np.ndarray], GWFieldLV20]:
    """
    Helper: devuelve (h2_callable, builder).
    """
    builder = GWFieldLV20(g_star=g_star, lv20_params=lv20_params, lv20_key=lv20_key)
    h2 = builder.from_params(Tn, alpha, beta_over_H, Tstar_mode=Tstar_mode)
    return h2, builder

# ------------------ ejemplo de uso ------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Ejemplo con números de prueba
    Tn = 1.0e-2     # GeV
    alpha = 1.0e5
    betaH = 40.0

    f = np.logspace(-14, -6, 300)

    # LV20 con T* = T_reh (físico en superenfriamiento)
    h2_reh, gwf = from_params(Tn, alpha, betaH, Tstar_mode="reh", lv20_key="U1")
    y_reh = np.array([h2_reh(fi) for fi in f])

    # LV20 con T* = T_nuc (para comparar)
    h2_nuc, _ = from_params(Tn, alpha, betaH, Tstar_mode="nuc", lv20_key="U1")
    y_nuc = np.array([h2_nuc(fi) for fi in f])

    fig, ax = plt.subplots(figsize=(7,5.5), dpi=120)
    ax.plot(f, y_reh, label="LV20 — T*=T_reh", lw=2.2)
    ax.plot(f, y_nuc, label="LV20 — T*=T_nuc", lw=2.0, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$f\,[{\rm Hz}]$")
    ax.set_ylabel(r"$h^2\Omega_{\rm GW}(f)$")
    ax.set_xlim(1e-11, 1e-6); ax.set_ylim(1e-14, 1e-6)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
