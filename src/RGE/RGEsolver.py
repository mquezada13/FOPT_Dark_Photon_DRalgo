# RGEsolver.py
"""
One-loop Renormalization Group Equation (RGE) solver for the dark U(1) model.

Runs the couplings (gD, lambdaS, mS) from an initial scale mu0 to a target
scale mu using the one-loop beta functions of the model.  Integration is
performed in log(mu) with SciPy's RK45 solver.

To study a different model, replace the three beta-function methods with the
corresponding expressions derived for that model.  The run_params interface
is model-independent and does not need to change.
"""

import numpy as np
from scipy.integrate import solve_ivp


class RGESolver:
    """One-loop RGE system and solver for the dark U(1) scalar model."""

    # ------------------------------------------------------------------ #
    # Beta functions                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def beta_gD2(gD2: float) -> float:
        """
        One-loop beta function for gD^2.

        d(gD^2)/d(log mu) = gD^4 / (24 pi^2).
        """
        return gD2**2 / (24.0 * np.pi**2)

    @staticmethod
    def beta_lambdaS(gD2: float, lambdaS: float) -> float:
        """
        One-loop beta function for the scalar quartic lambdaS.

        d(lambdaS)/d(log mu) = [3 gD^4 - 6 gD^2 lambdaS + 10 lambdaS^2] / (8 pi^2).
        """
        return (3.0 * gD2**2 - 6.0 * gD2 * lambdaS + 10.0 * lambdaS**2) / (8.0 * np.pi**2)

    @staticmethod
    def beta_mS2(gD2: float, lambdaS: float, mS: float) -> float:
        """
        One-loop beta function for the scalar mass parameter mS.

        d(mS)/d(log mu) = -mS (3 gD^2 - 4 lambdaS) / (8 pi^2).
        """
        return -mS * (3.0 * gD2 - 4.0 * lambdaS) / (8.0 * np.pi**2)

    @classmethod
    def _rhs(cls, t: float, y: list) -> list:
        """RHS of the RGE system in the variable t = log(mu)."""
        gD2, lambdaS, mS = y
        return [
            cls.beta_gD2(gD2),
            cls.beta_lambdaS(gD2, lambdaS),
            cls.beta_mS2(gD2, lambdaS, mS),
        ]

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def run_params(
        cls,
        mu: float,
        gD0: float,
        lambdaS0: float,
        mS0: float,
        mu0: float = 1.0,
    ) -> tuple:
        """
        Run gD, lambdaS, and mS from mu0 to mu using the one-loop RGEs.

        Parameters
        ----------
        mu       : Target renormalisation scale [GeV].  Must be > 0.
        gD0      : Dark gauge coupling at mu0.
        lambdaS0 : Scalar quartic coupling at mu0.
        mS0      : Scalar mass parameter at mu0 [GeV].
        mu0      : Initial renormalisation scale (default 1 GeV).

        Returns
        -------
        gD(mu), lambdaS(mu), mS(mu)  as a 3-tuple of floats.

        Raises
        ------
        ValueError  if mu or mu0 are non-positive.
        RuntimeError if the ODE integrator fails.
        """
        if mu <= 0 or mu0 <= 0:
            raise ValueError(
                f"Scales must be positive; got mu={mu}, mu0={mu0}."
            )

        y0     = [gD0**2, lambdaS0, mS0]
        t_span = (np.log(mu0), np.log(mu))

        sol = solve_ivp(
            cls._rhs,
            t_span,
            y0,
            method="RK45",
            t_eval=[np.log(mu)],
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"RGE integration failed: {sol.message}")

        gD2_run, lambdaS_run, mS_run = sol.y[:, -1]
        return float(np.sqrt(gD2_run)), float(lambdaS_run), float(mS_run)
