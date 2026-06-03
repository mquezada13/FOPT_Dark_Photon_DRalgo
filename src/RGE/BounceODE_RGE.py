# BounceODE_RGE.py
"""
ODE-shooting bounce solvers wrapping Find_critbubble from bounce_solver.py.

These classes are drop-in replacements for BounceSolverHighT and BounceSolver:
they expose the same SE(T, gD, scale, ls0) interface and can be injected into
FOPTUtilities via the solver_ht / solver_full keyword arguments.

The key algorithmic difference with the Espinosa solvers is that here the full
O(3) bounce profile is integrated numerically (shooting method), and S3 is
computed directly from that profile via the virial formula:

    S3 = 4π ∫ dr r² [ ½ (dφ/dr)² + V(φ) ]

This is independent of any thin-wall or tunnelling-potential approximation.

Speed note
----------
Inside Find_critbubble, dV(φ) is called O(10^4) times per SE evaluation.
Two optimisations are applied:

  1. Pre-baked parameters: (gD, lambdaS, mS, V0) are computed once per SE call
     and captured in the closures — no dict lookups, no RGE ODE inside V.

  2. Analytical dV: dV/dS is computed from closed-form derivatives of
     (Vtree, Vcw, Vdaisy) — identical for both HighT and Full — plus the
     thermal contribution:
       - HighT: dV_T/dS uses _dJBhighexp_dy (exact closed form).
       - Full:  dV_T/dS uses _dJBfull_dy, built from a CubicSpline fitted
               once to the raw JB table data.  The linear interp1d in VeffRGE
               would give a step-function derivative; the cubic spline gives
               a smooth, accurate JB'(y) with negligible extra cost.

Pipeline position
-----------------
VeffRGE  <--  BounceODEHighT / BounceODEFull  <--  FOPTUtilities (via injection)
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline

import constants as cs
from src.RGE.VeffFunc_RGE import VeffRGE
from src.RGE.RGEsolver import RGESolver
from src.RGE.bounce_solver import Find_critbubble, compute_S3


# --------------------------------------------------------------------------- #
# Tuning knobs                                                                 #
# --------------------------------------------------------------------------- #
_DEFAULT_N_SCAN = 800    # barrier-finding grid (default in bounce_solver: 4000)
_DEFAULT_N_PTS  = 400    # profile output points (default: 1200)
_DEFAULT_RTOL   = 1e-5   # ODE rtol (default: 1e-7) — enough for Tn/alpha/beta
_DEFAULT_ATOL   = 1e-7   # ODE atol (default: 1e-9)


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _phi_min_Veff0(veff, T, gD, scale, ls0):
    """Zero-temperature RGE minimum — shared by both ODE solver classes."""
    return optimize.minimize_scalar(
        lambda S: veff.Veff0_RGE(S, T, gD, scale, ls0),
        bounds=(1e-7, 1e7),
        method="bounded",
    ).x


def _d2V_at_origin(V, T: float) -> float:
    """
    V''(0) from a pre-baked V callable.

    Since V(0) = 0 and V'(0) = 0 by symmetry: V''(0) ≈ 2V(ε)/ε²
    """
    eps = max(1e-4 * T, 1e-8)
    return 2.0 * V(eps) / eps**2


# --------------------------------------------------------------------------- #
# Analytical derivative helpers                                                #
# --------------------------------------------------------------------------- #

def _dJBhighexp_dy(y: float) -> float:
    """
    Analytical derivative of JBhighexp(y) with respect to y.

    JBhighexp(y) = -π⁴/45 + (π²/12)y - (π/6)max(0,y)^{3/2}
                   - (1/32) y² (ln|y| - ab)

    d/dy[...] = π²/12 - (π/4)√max(0,y) - (y/32)(2(ln|y| - ab) + 1)

    The last term uses the identity d/dy[y² ln|y|] = 2y ln|y| + y.
    """
    pi = np.pi
    sqrt_term = (pi / 4.0) * np.sqrt(max(0.0, y)) if y > 0 else 0.0
    log_y = np.log(max(1e-10, abs(y)))
    log_term = (y / 32.0) * (2.0 * (log_y - cs.ab) + 1.0)
    return pi**2 / 12.0 - sqrt_term - log_term


def _make_V_dV_highT(veff: VeffRGE, T: float, gD0: float, scale: float, ls0: float):
    """
    Build fast (V, dV) callables for Veff_HighT at fixed (T, gD0, scale, ls0).

    RGE parameters and V0 are pre-baked once.  dV is computed analytically
    using closed-form derivatives of all four potential contributions.

    Analytical dV eliminates the two extra V evaluations per dV call that
    numerical finite-differencing would require.
    """
    mu      = float(scale) * float(T)
    T       = float(T)
    gD, lambdaS, mS = RGESolver.run_params(mu, gD0, ls0, cs.mS0)
    Re1     = complex(1.0, 0.0)

    # Thermal self-energies (field-independent)
    Pi_phi  = (lambdaS / 3.0 + gD**2 / 4.0) * T**2
    Pi_Ap   = gD**2 / 3.0 * T**2

    # V0 normalisation (V(S=0) = 0 by definition, but we compute for safety)
    V0_val = float(np.real(
        veff.Vtree(0.0, lambdaS, mS)
        + veff.Vcw(0.0, mu, gD, lambdaS, mS)
        + veff.V_highT(0.0, T, gD, lambdaS, mS)
        + veff.Vdaisy(0.0, T, gD, lambdaS, mS)
    ))

    def V(S: float) -> float:
        S = float(S)
        return float(np.real(
            veff.Vtree(S, lambdaS, mS)
            + veff.Vcw(S, mu, gD, lambdaS, mS)
            + veff.V_highT(S, T, gD, lambdaS, mS)
            + veff.Vdaisy(S, T, gD, lambdaS, mS)
        )) - V0_val

    def dV(S: float) -> float:
        S = float(S)

        # Field-dependent masses (mass-squared values)
        m1 = -mS**2 + 3.0 * lambdaS * S**2   # mPhi2
        m2 = -mS**2 + lambdaS * S**2           # mSigma2
        m3 = gD**2 * S**2                       # mAp2

        # dm²/dS
        dm1 = 6.0 * lambdaS * S
        dm2 = 2.0 * lambdaS * S
        dm3 = 2.0 * gD**2 * S

        # dVtree/dS
        dVt = -mS**2 * S + lambdaS * S**3

        # dVcw/dS
        # d/dS [m² (ln(m/μ²) - c)] = dm/dS · m · (2(ln(m/μ²) - c) + 1)
        # where m here is m² (code convention), so log arg = m²/μ²
        def _dcw(m2val, dm2val, c):
            if m2val == 0.0:
                return 0.0 + 0.0j
            log_arg = Re1 * m2val / mu**2
            log_term = np.log(log_arg)
            return dm2val * m2val * (2.0 * (log_term - c) + 1.0)

        dVcw = float(np.real(
            _dcw(m1, dm1, 1.5)
            + _dcw(m2, dm2, 1.5)
            + 3.0 * _dcw(m3, dm3, 5.0 / 6.0)
        ) / (64.0 * np.pi**2))

        # dV_highT/dS = T²/(2π²) · Σ_i dJB/dy_i · dm_i/dS
        y1 = m1 / T**2
        y2 = m2 / T**2
        y3 = m3 / T**2
        factor = T**2 / (2.0 * np.pi**2)
        dVhT = factor * (
            _dJBhighexp_dy(y1) * dm1
            + _dJBhighexp_dy(y2) * dm2
            + 3.0 * _dJBhighexp_dy(y3) * dm3
        )

        # dVdaisy/dS = -(T/12π) · Σ (3/2)·((m+Π)^{1/2} - m^{1/2}) · dm/dS
        def _ddaisy(m2val, Pi, dm2val):
            return 1.5 * ((Re1 * (m2val + Pi))**0.5 - (Re1 * m2val)**0.5) * dm2val

        dVd = float(np.real(
            -(T / (12.0 * np.pi)) * (
                _ddaisy(m1, Pi_phi, dm1)
                + _ddaisy(m2, Pi_phi, dm2)
                + _ddaisy(m3, Pi_Ap, dm3)
            )
        ))

        return dVt + dVcw + dVhT + dVd

    return V, dV


def _make_V_dV_full(veff: VeffRGE, T: float, gD0: float, scale: float, ls0: float):
    """
    Build fast (V, dV) callables for Veff (full J_B) at fixed (T, gD0, scale, ls0).

    RGE parameters and V0 are pre-baked once.  dV is computed analytically
    using the same structure as _make_V_dV_highT, replacing _dJBhighexp_dy
    with _dJBfull_dy built from a CubicSpline of the raw JB table data.
    The spline fitting is a one-time O(N log N) cost done here, not per dV call.
    """
    mu = float(scale) * float(T)
    T  = float(T)
    gD, lambdaS, mS = RGESolver.run_params(mu, gD0, ls0, cs.mS0)
    Re1 = complex(1.0, 0.0)

    # Build a smooth JB spline + its derivative from the existing interp1d table.
    # VeffRGE._JB_interp stores the raw (y, JB) data in .x and .y.
    _jb_x = veff._JB_interp.x
    _jb_y = veff._JB_interp.y
    _jb_cs  = CubicSpline(_jb_x, _jb_y, extrapolate=False)
    _jb_dcs = _jb_cs.derivative()
    _y_min, _y_max = float(_jb_x[0]), float(_jb_x[-1])

    def _dJBfull_dy(y: float) -> float:
        """dJB/dy from CubicSpline; returns 0 outside the table range."""
        if y < _y_min or y > _y_max or y >= 600.0:
            return 0.0
        return float(_jb_dcs(y))

    # Thermal self-energies (field-independent)
    Pi_phi = (lambdaS / 3.0 + gD**2 / 4.0) * T**2
    Pi_Ap  = gD**2 / 3.0 * T**2

    V0_val = float(np.real(
        veff.Vtree(0.0, lambdaS, mS)
        + veff.Vcw(0.0, mu, gD, lambdaS, mS)
        + veff.VTfull(0.0, T, gD, lambdaS, mS)
        + veff.Vdaisy(0.0, T, gD, lambdaS, mS)
    ))

    def V(S: float) -> float:
        S = float(S)
        return float(np.real(
            veff.Vtree(S, lambdaS, mS)
            + veff.Vcw(S, mu, gD, lambdaS, mS)
            + veff.VTfull(S, T, gD, lambdaS, mS)
            + veff.Vdaisy(S, T, gD, lambdaS, mS)
        )) - V0_val

    def dV(S: float) -> float:
        S = float(S)

        m1 = -mS**2 + 3.0 * lambdaS * S**2
        m2 = -mS**2 + lambdaS * S**2
        m3 = gD**2 * S**2

        dm1 = 6.0 * lambdaS * S
        dm2 = 2.0 * lambdaS * S
        dm3 = 2.0 * gD**2 * S

        dVt = -mS**2 * S + lambdaS * S**3

        def _dcw(m2val, dm2val, c):
            if m2val == 0.0:
                return 0.0 + 0.0j
            return dm2val * m2val * (2.0 * (np.log(Re1 * m2val / mu**2) - c) + 1.0)

        dVcw = float(np.real(
            _dcw(m1, dm1, 1.5)
            + _dcw(m2, dm2, 1.5)
            + 3.0 * _dcw(m3, dm3, 5.0 / 6.0)
        ) / (64.0 * np.pi**2))

        y1 = m1 / T**2
        y2 = m2 / T**2
        y3 = m3 / T**2
        factor = T**2 / (2.0 * np.pi**2)
        dVhT = factor * (
            _dJBfull_dy(y1) * dm1
            + _dJBfull_dy(y2) * dm2
            + 3.0 * _dJBfull_dy(y3) * dm3
        )

        def _ddaisy(m2val, Pi, dm2val):
            return 1.5 * ((Re1 * (m2val + Pi))**0.5 - (Re1 * m2val)**0.5) * dm2val

        dVd = float(np.real(
            -(T / (12.0 * np.pi)) * (
                _ddaisy(m1, Pi_phi, dm1)
                + _ddaisy(m2, Pi_phi, dm2)
                + _ddaisy(m3, Pi_Ap, dm3)
            )
        ))

        return dVt + dVcw + dVhT + dVd

    return V, dV


# --------------------------------------------------------------------------- #
# High-T ODE solver                                                            #
# --------------------------------------------------------------------------- #

class BounceODEHighT:
    """
    ODE bounce solver for the high-T expanded effective potential.

    Drop-in replacement for BounceSolverHighT with the same SE() interface.
    Uses analytical dV/dS (see _make_V_dV_highT).

    Parameters
    ----------
    veff : VeffRGE or None
    """

    def __init__(self, veff: VeffRGE = None):
        self.veff = veff if veff is not None else VeffRGE()

    def phi_min_Veff0(self, T, gD, scale, ls0, bound_min=(1e-7, 1e7)):
        """Zero-T RGE minimum (same as BounceSolverHighT)."""
        return _phi_min_Veff0(self.veff, T, gD, scale, ls0)

    def SE(self, T, gD0, scale, ls0, **find_kwargs) -> tuple:
        """
        Compute S3(T) via ODE shooting using Veff_HighT with analytical dV.

        Parameters
        ----------
        **find_kwargs
            Forwarded to Find_critbubble (e.g. n_scan, n_pts, verbose).

        Returns
        -------
        (S3, phi0) : (float, float)
        """
        V, dV = _make_V_dV_highT(self.veff, T, gD0, scale, ls0)
        d2V0  = _d2V_at_origin(V, T)

        kw = dict(phi_scan_n=_DEFAULT_N_SCAN, n_pts=_DEFAULT_N_PTS,
                  rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL, verbose=False)
        kw.update(find_kwargs)

        profile = Find_critbubble("auto", T, V, dV, d2V0=d2V0, **kw)
        S3, _, _ = compute_S3(profile, V)
        return float(S3), float(profile.phi0)


# --------------------------------------------------------------------------- #
# Full-Veff ODE solver                                                         #
# --------------------------------------------------------------------------- #

class BounceODEFull:
    """
    ODE bounce solver for the full finite-T effective potential (exact J_B).

    Drop-in replacement for BounceSolver with the same SE() interface.
    Uses numerical dV on a pre-baked V (analytical dV is not available
    because J_B is a piecewise-linear interpolation).

    Parameters
    ----------
    veff : VeffRGE or None
    """

    def __init__(self, veff: VeffRGE = None):
        self.veff = veff if veff is not None else VeffRGE()

    def phi_min_Veff0(self, T, gD, scale, ls0, bound_min=(1e-7, 1e7)):
        """Zero-T RGE minimum."""
        return _phi_min_Veff0(self.veff, T, gD, scale, ls0)

    def phi_min(self, T, gD, scale, ls0):
        """Broken-phase minimum of the full finite-T potential."""
        phi0_ref = _phi_min_Veff0(self.veff, T, gD, scale, ls0)
        return optimize.minimize_scalar(
            lambda S: np.real(self.veff.Veff(S, T, gD, scale, ls0)),
            bounds=(1e-7, phi0_ref),
            method="bounded",
        ).x

    def SE(self, T, gD0, scale, ls0, **find_kwargs) -> tuple:
        """
        Compute S3(T) via ODE shooting using the full Veff (numerical dV).

        Parameters
        ----------
        **find_kwargs
            Forwarded to Find_critbubble (e.g. n_scan, n_pts, verbose).

        Returns
        -------
        (S3, phi0) : (float, float)
        """
        V, dV = _make_V_dV_full(self.veff, T, gD0, scale, ls0)
        d2V0  = _d2V_at_origin(V, T)

        kw = dict(phi_scan_n=_DEFAULT_N_SCAN, n_pts=_DEFAULT_N_PTS,
                  rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL, verbose=False)
        kw.update(find_kwargs)

        profile = Find_critbubble("auto", T, V, dV, d2V0=d2V0, **kw)
        S3, _, _ = compute_S3(profile, V)
        return float(S3), float(profile.phi0)
