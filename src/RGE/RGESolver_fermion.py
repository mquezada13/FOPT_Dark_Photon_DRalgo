# RGESolver_fermion.py
"""
One-loop RGE solver for the U(1)_D + complex scalar + Dirac fermion model.

Runs (gD, lambdaS, mS2, y) from mu0 to mu using the 1-loop beta functions.
The Dirac fermion has charges ±1/2 under U(1)_D and Yukawa coupling y to the
complex scalar.

Interface: identical to RGEsolver.run_params but with an extra y0 argument
and a 4-tuple return value (gD, lambdaS, mS2, y).
"""

import numpy as np
from functools import lru_cache
from scipy.integrate import solve_ivp


@lru_cache(maxsize=4096)
def _run_ferm_cached(
    mu: float,
    gD0: float,
    lambdaS0: float,
    mS02: float,
    y0: float,
    mu0: float,
) -> tuple:
    """Cached ODE solve for the fermion-model RGE. All arguments must be plain floats."""
    y_init = [gD0**2, lambdaS0, mS02, y0]
    t_span = (np.log(mu0), np.log(mu))

    sol = solve_ivp(
        RGESolver_fermion._rhs,
        t_span,
        y_init,
        method="RK45",
        t_eval=[np.log(mu)],
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Fermion RGE integration failed: {sol.message}")

    gD2_run, lambdaS_run, mS2_run, y_run = sol.y[:, -1]
    return float(np.sqrt(abs(gD2_run))), float(lambdaS_run), float(mS2_run), float(y_run)


class RGESolver_fermion:
    """One-loop RGE system for U(1)_D + complex scalar + Dirac fermion."""

    # ------------------------------------------------------------------ #
    # Beta functions                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def beta_gD2(gD2: float) -> float:
        """
        d(gD^2)/d(log mu) = gD^4 / (12 pi^2).

        Relative to the no-fermion model (coefficient 1/24), the Dirac fermion
        of charges ±1/2 doubles the coefficient.
        """
        return gD2**2 / (12.0 * np.pi**2)

    @staticmethod
    def beta_lambdaS(gD2: float, lambdaS: float, y: float) -> float:
        """
        d(lambdaS)/d(log mu) = [3gD^4 - 6gD^2 lambdaS + 10 lambdaS^2
                                 + 2 y^2 lambdaS - y^4] / (8 pi^2).
        """
        return (
            3.0 * gD2**2
            - 6.0 * gD2 * lambdaS
            + 10.0 * lambdaS**2
            + 2.0 * y**2 * lambdaS
            - y**4
        ) / (8.0 * np.pi**2)

    @staticmethod
    def beta_mS2(gD2: float, lambdaS: float, mS2: float, y: float) -> float:
        """
        d(mS^2)/d(log mu) = (4 lambdaS + y^2 - 3 gD^2) / (8 pi^2) * mS^2.
        """
        return (4.0 * lambdaS + y**2 - 3.0 * gD2) / (8.0 * np.pi**2) * mS2

    @staticmethod
    def beta_y(gD2: float, y: float) -> float:
        """
        d(y)/d(log mu) = y (4 y^2 - 3 gD^2) / (32 pi^2).
        """
        return y * (4.0 * y**2 - 3.0 * gD2) / (32.0 * np.pi**2)

    @classmethod
    def _rhs(cls, _t: float, y_vec: list) -> list:
        """RHS of the RGE system in the variable t = log(mu)."""
        gD2, lambdaS, mS2, y = y_vec
        return [
            cls.beta_gD2(gD2),
            cls.beta_lambdaS(gD2, lambdaS, y),
            cls.beta_mS2(gD2, lambdaS, mS2, y),
            cls.beta_y(gD2, y),
        ]

    # Public alias for notebooks that call solver.RGEs_logMu directly
    RGEs_logMu = _rhs

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def run_params(
        cls,
        mu: float,
        gD0: float,
        lambdaS0: float,
        mS02: float,
        y0: float,
        mu0: float = 1.0,
    ) -> tuple:
        """
        Run (gD, lambdaS, mS^2, y) from mu0 to mu using the 1-loop RGEs.

        Parameters
        ----------
        mu       : Target renormalisation scale [GeV].  Must be > 0.
        gD0      : Dark gauge coupling at mu0.
        lambdaS0 : Scalar quartic coupling at mu0.
        mS02     : Scalar mass-squared parameter at mu0 [GeV^2].
        y0       : Yukawa coupling at mu0.
        mu0      : Initial renormalisation scale (default 1 GeV).

        Returns
        -------
        gD(mu), lambdaS(mu), mS2(mu), y(mu)  as a 4-tuple of floats.
        """
        if mu <= 0 or mu0 <= 0:
            raise ValueError(f"Scales must be positive; got mu={mu}, mu0={mu0}.")
        return _run_ferm_cached(
            float(mu), float(gD0), float(lambdaS0),
            float(mS02), float(y0), float(mu0),
        )
