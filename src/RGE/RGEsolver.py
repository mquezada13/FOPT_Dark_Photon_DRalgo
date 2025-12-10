# RGEsolver.py
"""
One-loop Renormalization Group Equation (RGE) solver
for gD, lambdaS, and mS.
"""

import numpy as np
from scipy.integrate import solve_ivp


class RGESolver:
    """Encapsulates the 1-loop RGE system and solver."""

    @staticmethod
    def beta_gD2(gD2: float) -> float:
        """Beta function for gD^2."""
        return (1 / (24 * np.pi**2)) * gD2**2

    @staticmethod
    def beta_lambdaS(gD2: float, lambdaS: float) -> float:
        """Beta function for lambda_S."""
        return (1 / (8 * np.pi**2)) * (3 * gD2**2 - 6 * gD2 * lambdaS + 10 * lambdaS**2)

    @staticmethod
    def beta_mS2(gD2: float, lambdaS: float, mS: float) -> float:
        """Beta function for m_S."""
        return -(1 / (8 * np.pi**2)) * mS * (3 * gD2 - 4 * lambdaS)

    @classmethod
    def RGEs_logMu(cls, t: float, y: list[float]) -> list[float]:
        """System of RGEs in log(mu)."""
        gD2, lambdaS, mS = y
        return [
            cls.beta_gD2(gD2),
            cls.beta_lambdaS(gD2, lambdaS),
            cls.beta_mS2(gD2, lambdaS, mS)
        ]

    @classmethod
    def run_params(cls, mu: float, gD0: float, lambdaS0: float, mS0: float, mu0: float = 1.0):
        """
        Run gD, lambdaS, and mS from mu0 to mu using 1-loop RGEs.

        Parameters
        ----------
        mu : float
            Final renormalization scale.
        gD0 : float
            Initial dark gauge coupling.
        lambdaS0 : float
            Initial scalar quartic coupling.
        mS0 : float
            Initial scalar mass parameter.
        mu0 : float, optional
            Initial renormalization scale (default=1.0).

        Returns
        -------
        gD(mu), lambdaS(mu), mS(mu)
        """

        # Input validation
        if mu <= 0 or mu0 <= 0:
            raise ValueError(f"Both mu and mu0 must be > 0. Received mu={mu}, mu0={mu0}.")

        log_mu0 = np.log(mu0)
        log_mu = np.log(mu)

        y0 = [gD0**2, lambdaS0, mS0]
        t_span = (log_mu0, log_mu)

        sol = solve_ivp(cls.RGEs_logMu, t_span, y0, method="RK45", t_eval=[log_mu])

        if not sol.success:
            raise RuntimeError(f"RGE integration failed: {sol.message}")

        gD2_run, lambdaS_run, mS_run = sol.y[:, -1]
        return np.sqrt(gD2_run), lambdaS_run, mS_run

