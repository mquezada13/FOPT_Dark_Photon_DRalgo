import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Callable, Optional, Union, Tuple, List


# ============================================================================
# Output container
# ============================================================================

@dataclass
class BounceProfile:
    """
    Container for the bounce solution.

    Attributes
    ----------
    R : ndarray
        Radial grid.
    Phi : ndarray
        Bounce profile phi_b(r).
    dPhi : ndarray
        Radial derivative d phi_b / dr.
    phi0 : float
        Central field value of the bounce.
    T : float
        Temperature used in the potential.
    converged : bool
        True if the final tail looks under control.
    """
    R: np.ndarray
    Phi: np.ndarray
    dPhi: np.ndarray
    phi0: float
    T: float
    converged: bool


# ============================================================================
# Utility
# ============================================================================

def r_integral(r: np.ndarray, integrand: np.ndarray) -> float:
    """
    Compute

        4*pi * int dr r^2 f(r)

    using the trapezoidal rule.
    """
    r = np.asarray(r)
    integrand = np.asarray(integrand)
    return 4.0 * np.pi * np.trapz(r**2 * integrand, r)


def compute_S3(profile: BounceProfile, V: Callable):
    """
    Compute the LO three-dimensional bounce action from a stored profile:

        S3 = 4*pi int dr r^2 [ 1/2 (dphi/dr)^2 + V(phi) ].

    Returns
    -------
    S3, S3_kin, S3_pot : floats
    """
    r = profile.R
    phi = profile.Phi
    dphi = profile.dPhi

    kin = 0.5 * dphi**2
    pot = np.array([V(x) for x in phi])

    S3_kin = r_integral(r, kin)
    S3_pot = r_integral(r, pot)
    S3 = S3_kin + S3_pot
    return S3, S3_kin, S3_pot


def plot_bounce_profile(profile: BounceProfile, xlim=None, ylim=None):
    """
    Quick plotting helper for debugging.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(profile.R, profile.Phi, lw=2)
    plt.xlabel("r")
    plt.ylabel(r"$\phi_b(r)$")
    plt.title(f"Bounce profile, T = {profile.T:.4g}")
    plt.grid(True, alpha=0.3)

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Bounce ODE and small-r initial conditions
# ============================================================================

def _bounce_rhs(r: float, y: np.ndarray, dV: Callable):
    """
    First-order form of the O(3) bounce equation:

        phi''(r) + 2/r phi'(r) = dV/dphi(phi).

    We rewrite it as a first-order system with y = [phi, dphi].
    """
    phi, dphi = y
    ddphi = dV(phi) - (2.0 / r) * dphi
    return [dphi, ddphi]


def _initial_conditions(phi0: float, dV: Callable, r_min: float):
    """
    Regular small-r expansion:

        phi(r)  = phi0 + dV(phi0) r^2 / 6 + ...
        phi'(r) = dV(phi0) r / 3 + ...
    """
    dV0 = dV(phi0)
    phi_init = phi0 + dV0 * r_min**2 / 6.0
    dphi_init = dV0 * r_min / 3.0
    return phi_init, dphi_init


# ============================================================================
# Events
# ============================================================================

def _make_cross_zero_event():
    """
    Overshoot event: stop when phi crosses zero from above.
    """
    def event(r, y):
        return y[0]

    event.terminal = True
    event.direction = -1
    return event


def _make_tail_event(tail_phi: float):
    """
    Tail-convergence event: stop when phi decays below tail_phi from above.

    This mirrors the ``hit_tail`` event in the student solver: the integration
    terminates naturally once the profile has relaxed to a small fraction of
    phi0, so r_max no longer needs to be large enough to contain the full tail.
    """
    def event(r, y):
        return y[0] - tail_phi

    event.terminal = True
    event.direction = -1
    return event


# ============================================================================
# Barrier scale from the potential
# ============================================================================

def _safe_eval_array(f: Callable, xs: np.ndarray) -> np.ndarray:
    """
    Evaluate a scalar function on a grid and return NaN when evaluation fails.
    """
    vals = []
    for x in xs:
        try:
            vals.append(float(f(float(x))))
        except Exception:
            vals.append(np.nan)
    return np.asarray(vals, dtype=float)


def _hybrid_phi_grid(phi_min: float, phi_max: float, n_scan: int) -> np.ndarray:
    """
    Hybrid log + linear grid.

    The original solver used only a linear grid. That can miss very small-field
    barriers when phi_scan_max is large. The log part resolves tiny phi, while
    the linear part keeps the old behavior at ordinary field values.
    """
    phi_min = max(float(phi_min), 1e-300)
    phi_max = max(float(phi_max), 10.0 * phi_min)
    n_scan = max(int(n_scan), 200)

    n_log = n_scan
    n_lin = max(n_scan // 2, 200)

    phi_log = np.geomspace(phi_min, phi_max, n_log)
    phi_lin = np.linspace(phi_min, phi_max, n_lin)

    phi_grid = np.unique(np.concatenate([phi_log, phi_lin]))
    phi_grid.sort()
    return phi_grid


def find_stationary_points(
    V: Callable,
    dV: Callable,
    phi_scan_max: float,
    *,
    phi_min: float = 1e-14,
    n_scan: int = 4000,
) -> List[Tuple[float, float]]:
    """
    Diagnostic helper.

    Returns all positive roots of dV(phi)=0 found on a hybrid grid, together
    with V(root): [(phi_root, V_root), ...].
    """
    phi_scan_max = max(float(phi_scan_max), 1.0)
    phi_grid = _hybrid_phi_grid(phi_min, phi_scan_max, n_scan)
    dV_vals = _safe_eval_array(dV, phi_grid)

    finite = np.isfinite(dV_vals)
    phi_grid = phi_grid[finite]
    dV_vals = dV_vals[finite]

    roots = []

    for i in range(len(phi_grid) - 1):
        x1, x2 = phi_grid[i], phi_grid[i + 1]
        y1, y2 = dV_vals[i], dV_vals[i + 1]

        if not np.isfinite(y1) or not np.isfinite(y2):
            continue

        if y1 == 0.0:
            root = x1
        elif y1 * y2 < 0.0:
            try:
                root = brentq(dV, x1, x2)
            except Exception:
                continue
        else:
            continue

        if root <= 0.0:
            continue

        try:
            Vroot = float(V(root))
        except Exception:
            continue

        if not np.isfinite(Vroot):
            continue

        if len(roots) == 0 or abs(root - roots[-1][0]) > 1e-8 * max(root, 1.0):
            roots.append((root, Vroot))

    return roots


def _find_phi_max(
    V: Callable,
    dV: Callable,
    phi_scan_max: float,
    n_scan: int = 4000,
    *,
    phi_min: float = 1e-14,
):
    """
    Find the barrier top phi_max.

    We define phi_max as the first positive root of dV(phi)=0 for which
    V(phi)>0.

    Improvement relative to the old solver:
      * uses a hybrid log + linear grid instead of a purely linear grid.
      * therefore it is much less likely to miss tiny barriers at small T.
    """
    roots = find_stationary_points(
        V,
        dV,
        phi_scan_max=phi_scan_max,
        phi_min=phi_min,
        n_scan=n_scan,
    )

    for root, Vroot in roots:
        if Vroot > 0.0:
            return root

    if len(roots) > 0:
        msg = (
            "Could not find a positive barrier top with V(root)>0. "
            "Stationary points were found, but none had V>0."
        )
    else:
        msg = (
            "Could not find phi_max. No stationary point of dV was found "
            "in the scanned range."
        )

    raise RuntimeError(msg)


def _find_phi_max_auto(
    V: Callable,
    dV: Callable,
    *,
    phi_scan_start: float = 1.0,
    phi_scan_cap: float = 1e6,
    grow: float = 2.0,
    n_scan: int = 4000,
    phi_min: float = 1e-14,
    verbose: bool = False,
):
    """
    Find phi_max by growing the scan range.

    This is only used when phi_end is None or 'auto'. It keeps trying larger
    field ranges, while the hybrid grid still resolves tiny fields.
    """
    phi_scan_max = max(float(phi_scan_start), 1.0)

    while phi_scan_max <= phi_scan_cap:
        try:
            phi_max = _find_phi_max(
                V,
                dV,
                phi_scan_max=phi_scan_max,
                n_scan=n_scan,
                phi_min=phi_min,
            )

            if verbose:
                print(f"[auto phi] found phi_max = {phi_max:.6g} scanning to {phi_scan_max:.6g}")

            return phi_max, phi_scan_max

        except RuntimeError as err:
            if verbose:
                print(f"[auto phi] no accepted barrier up to {phi_scan_max:.6g}: {err}")
            phi_scan_max *= grow

    raise RuntimeError(
        f"Could not find phi_max even after scanning up to {phi_scan_cap}."
    )


def _estimate_phi_end_from_potential(
    V: Callable,
    phi_max: float,
    phi_scan_max: float,
    *,
    safety: float = 1.5,
    grow: float = 2.0,
    phi_cap: float = 1e6,
    n_scan: int = 1200,
    verbose: bool = False,
):
    """
    Estimate a useful shooting scale beyond the barrier.

    This is only a numerical helper. It is not a physical observable.
    It tries to find where V becomes negative after the barrier. If it cannot
    find that, it falls back to a multiple of phi_max.
    """
    phi_start = max(1.001 * phi_max, 1e-14)
    phi_stop = max(phi_scan_max, 3.0 * phi_max, 1.0)

    while phi_stop <= phi_cap:
        grid = _hybrid_phi_grid(phi_start, phi_stop, n_scan)
        Vvals = _safe_eval_array(V, grid)

        finite = np.isfinite(Vvals)
        grid = grid[finite]
        Vvals = Vvals[finite]

        if len(grid) > 10:
            neg = np.where(Vvals < 0.0)[0]
            if len(neg) > 0:
                phi_neg = grid[neg[0]]
                phi_end = max(safety * phi_neg, 3.0 * phi_max)

                if verbose:
                    print(
                        f"[auto phi] V first negative near phi = {phi_neg:.6g}; "
                        f"using phi_end = {phi_end:.6g}"
                    )

                return phi_end

        phi_stop *= grow

    phi_end = max(3.0 * phi_max, phi_scan_max)

    if verbose:
        print(
            "[auto phi] did not find a V<0 region; "
            f"falling back to phi_end = {phi_end:.6g}"
        )

    return phi_end


def diagnose_potential(
    V: Callable,
    dV: Callable,
    *,
    phi_min: float = 1e-14,
    phi_max: float = 1e3,
    n_scan: int = 8000,
    title: str = "Potential diagnostic",
):
    """
    Plot V and dV on a hybrid/log grid and print stationary points.

    Useful before blaming the bounce solver.
    """
    phi_grid = _hybrid_phi_grid(phi_min, phi_max, n_scan)
    V_vals = _safe_eval_array(V, phi_grid)
    dV_vals = _safe_eval_array(dV, phi_grid)

    roots = find_stationary_points(
        V,
        dV,
        phi_scan_max=phi_max,
        phi_min=phi_min,
        n_scan=n_scan,
    )

    print("\nStationary points found:")
    print("phi_root              V(phi_root)")
    print("-" * 55)
    if len(roots) == 0:
        print("No stationary points found.")
    else:
        for root, Vroot in roots:
            print(f"{root:.8e}        {Vroot:.8e}")

    finite_V = np.isfinite(V_vals)
    finite_dV = np.isfinite(dV_vals)

    plt.figure(figsize=(7, 4.8), dpi=140)
    plt.plot(phi_grid[finite_V], V_vals[finite_V])
    plt.axhline(0.0, linewidth=1)
    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-12)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$V(\phi)$")
    plt.title(title + ": V")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4.8), dpi=140)
    plt.plot(phi_grid[finite_dV], dV_vals[finite_dV])
    plt.axhline(0.0, linewidth=1)
    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-12)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$dV/d\phi$")
    plt.title(title + ": dV")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return roots


# ============================================================================
# Radius estimate from V''(0)
# ============================================================================

def _estimate_r_max(
    d2V0: Optional[float],
    T: float,
    safety: float = 40.0,
    rmin: float = 80.0,
    rmax_cap: float = 1e8,
):
    """
    Estimate a sensible outer radius from the false-vacuum curvature.

    If V''(0) > 0, the tail scale is set by m = sqrt(V''(0)) and we use
    r_max ~ safety / m.

    If V''(0) <= 0 or not provided, fall back to a temperature-based scale.
    """
    if d2V0 is None:
        scale = max(float(T), 1e-6)
        r_max = safety / scale
        return float(min(max(r_max, rmin), rmax_cap))

    m2 = float(d2V0)

    if m2 <= 0.0 or not np.isfinite(m2):
        scale = max(float(T), 1e-6)
        r_max = safety / scale
    else:
        m = np.sqrt(m2)
        r_max = safety / max(m, 1e-12)

    return float(min(max(r_max, rmin), rmax_cap))


# ============================================================================
# One integration
# ============================================================================

def _integrate_profile(
    phi0: float,
    dV: Callable,
    r_min: float,
    r_max: float,
    rtol: float,
    atol: float,
    tail_phi: Optional[float] = None,
):
    """
    Integrate the O(3) bounce equation from small r out to r_max.

    Two early-stop events are supported:
      * event[0]: overshoot — phi crosses zero from above.
      * event[1]: tail convergence — phi drops below tail_phi (only when
        tail_phi is provided). This is the same as the ``hit_tail`` event in
        the student solver: the integration stops naturally once the profile
        has decayed enough, so r_max need not be enormous.

    Parameters
    ----------
    tail_phi : float or None
        Threshold for the tail event. Typically tail_frac * phi0. When None
        the tail event is not registered and behaviour is identical to the
        original solver.
    """
    phi_init, dphi_init = _initial_conditions(phi0, dV, r_min)
    zero_event = _make_cross_zero_event()

    events = [zero_event]
    if tail_phi is not None and tail_phi > 0.0:
        events.append(_make_tail_event(float(tail_phi)))

    sol = solve_ivp(
        fun=lambda r, y: _bounce_rhs(r, y, dV),
        t_span=(r_min, r_max),
        y0=[phi_init, dphi_init],
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        events=events,
    )
    return sol


# ============================================================================
# Shooting classification
# ============================================================================

def _classify_shot(
    phi0: float,
    dV: Callable,
    r_min: float,
    r_max: float,
    rtol: float,
    atol: float,
):
    """
    Classify a shooting attempt.

    Returns
    -------
    kind : str
        "overshoot" if phi crosses zero before r_max,
        "undershoot" otherwise.
    sol : OdeResult
        The solve_ivp output.
    """
    sol = _integrate_profile(
        phi0=phi0,
        dV=dV,
        r_min=r_min,
        r_max=r_max,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success and sol.y.shape[1] < 2:
        return "failed", sol

    if len(sol.t_events[0]) > 0:
        return "overshoot", sol

    # If integration failed after some progress, treat based on the final value
    # only when it is finite. Otherwise flag failure.
    if not sol.success:
        if sol.y.shape[1] > 0 and np.isfinite(sol.y[0, -1]):
            if sol.y[0, -1] <= 0.0:
                return "overshoot", sol
            return "undershoot", sol
        return "failed", sol

    return "undershoot", sol


# ============================================================================
# Bracket construction
# ============================================================================

def _find_bracket(
    phi_max: float,
    phi_end_guess: float,
    dV: Callable,
    r_min: float,
    r_max: float,
    rtol: float,
    atol: float,
    prev_phi0: Optional[float] = None,
    n_tries: int = 60,
    grow: float = 1.5,
):
    """
    Construct a numerical shooting bracket.

    The lower end starts just above the barrier top and should undershoot.
    The upper end is grown until it overshoots.
    """
    if prev_phi0 is None or not np.isfinite(prev_phi0):
        phi_lo = 1.001 * phi_max
    else:
        phi_lo = max(1.001 * phi_max, 0.7 * prev_phi0)

    last_lo_kind = None
    for _ in range(n_tries):
        kind_lo, _ = _classify_shot(phi_lo, dV, r_min, r_max, rtol, atol)
        last_lo_kind = kind_lo

        if kind_lo == "undershoot":
            break

        # If it overshoots too close to the barrier, move it down toward phi_max.
        # Never go below the barrier.
        phi_lo = 0.5 * (phi_lo + phi_max)

        if abs(phi_lo - phi_max) <= 1e-12 * max(phi_max, 1.0):
            phi_lo = 1.0000001 * phi_max
    else:
        raise RuntimeError(
            "Could not make the lower bracket undershoot near phi_max. "
            f"Last classification was {last_lo_kind}."
        )

    if prev_phi0 is None or not np.isfinite(prev_phi0):
        phi_hi = max(float(phi_end_guess), 1.5 * phi_max)
    else:
        phi_hi = max(1.2 * prev_phi0, float(phi_end_guess), 1.5 * phi_max)

    last_hi_kind = None
    for _ in range(n_tries):
        kind_hi, _ = _classify_shot(phi_hi, dV, r_min, r_max, rtol, atol)
        last_hi_kind = kind_hi

        if kind_hi == "overshoot":
            break

        phi_hi *= grow
    else:
        raise RuntimeError(
            "Could not find an overshooting upper bracket. "
            f"Last classification was {last_hi_kind}. Try increasing phi_end or r_max."
        )

    return phi_lo, phi_hi


def _objective(
    phi0: float,
    dV: Callable,
    r_min: float,
    r_max: float,
    rtol: float,
    atol: float,
):
    """
    Objective for Brent shooting.

    overshoot  -> negative
    undershoot -> positive
    """
    kind, sol = _classify_shot(phi0, dV, r_min, r_max, rtol, atol)

    if kind == "overshoot":
        if sol.y.shape[1] == 0:
            return -1e-14
        return -max(abs(sol.y[1, -1]), 1e-14)

    if kind == "undershoot":
        if sol.y.shape[1] == 0:
            return 1e-14
        return max(sol.y[0, -1], 1e-14)

    # Failed integrations are treated as undershoots by default so that the
    # bracket growth can keep going, but with a small positive value.
    return 1e-14


# ============================================================================
# Output grid
# ============================================================================

def _build_storage_grid_from_solution(
    r_raw: np.ndarray,
    phi_raw: np.ndarray,
    r_min: float,
    r_store: float,
    n_pts: int,
):
    """
    Build a non-uniform storage grid that resolves the wall.
    """
    r_raw = np.asarray(r_raw)
    phi_raw = np.asarray(phi_raw)

    if r_raw.size < 10:
        return np.linspace(r_min, r_store, max(n_pts, 50))

    dphi_raw = np.gradient(phi_raw, r_raw)
    i_wall = np.argmax(np.abs(dphi_raw))
    r_wall = r_raw[i_wall]

    thr = 0.5 * np.max(np.abs(dphi_raw))
    mask = np.abs(dphi_raw) >= thr
    if np.any(mask):
        r_left = r_raw[np.argmax(mask)]
        r_right = r_raw[len(mask) - 1 - np.argmax(mask[::-1])]
        wall_width = max(r_right - r_left, 10.0 * r_min)
    else:
        wall_width = max(0.1 * max(r_wall, r_min), 10.0 * r_min)

    n_pts = max(int(n_pts), 300)
    n_core = max(n_pts // 5, 80)
    n_wall = max(n_pts // 2, 150)
    n_tail = max(n_pts - n_core - n_wall + 2, 80)

    r_core_max = max(min(0.8 * r_wall, r_store), 20.0 * r_min)
    r_wall_lo = max(r_min, r_wall - 4.0 * wall_width)
    r_wall_hi = min(r_store, r_wall + 6.0 * wall_width)

    if r_core_max > r_min * 1.05:
        g1 = np.geomspace(r_min, r_core_max, n_core)
    else:
        g1 = np.linspace(r_min, r_core_max, n_core)

    if r_wall_hi > r_wall_lo:
        g2 = np.linspace(r_wall_lo, r_wall_hi, n_wall)
    else:
        g2 = np.array([])

    if r_store > max(r_wall_hi, r_min) * 1.05:
        g3 = np.geomspace(max(r_wall_hi, r_min), r_store, n_tail)
    else:
        g3 = np.linspace(max(r_wall_hi, r_min), r_store, n_tail)

    r_grid = np.unique(np.concatenate([g1, g2, g3]))
    return r_grid


# ============================================================================
# Public API
# ============================================================================

def Find_critbubble(
    phi_end: Union[float, str, None],
    T: float,
    V: Callable,
    dV: Callable,
    d2V0: Optional[float] = None,
    *,
    r_min: float = 1e-6,
    r_max: float = None,
    n_pts: int = 1200,
    bisect_tol: float = 1e-6,
    bisect_max: int = 80,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    prev_phi0: Optional[float] = None,
    prev_rmax: Optional[float] = None,
    verbose: bool = True,
    # New optional knobs. Existing code does not need to pass these.
    phi_scan_min: float = 1e-14,
    phi_scan_cap: float = 1e6,
    phi_scan_n: int = 4000,
    max_rmax_retries: int = 5,
    rmax_growth: float = 3.0,
    r_max_cap: Optional[float] = None,
    tail_frac: float = 1e-6,
) -> BounceProfile:
    """
    Compute the O(3)-symmetric bounce profile.

    This version is intentionally close to the old solver, but improves the
    fragile parts:

      * barrier finding uses a hybrid log + linear field scan;
      * phi_end can be a float, None, or 'auto';
      * r_max retries are slightly more robust for long supercooled tails;
      * diagnostic errors are more informative.

    Parameters
    ----------
    phi_end : float, None, or 'auto'
        If float: shooting field-scale guess beyond the barrier.
        If None or 'auto': estimate it from the potential.
    T : float
        Temperature.
    V, dV : callables
        Potential and first derivative.
    d2V0 : float or None
        False-vacuum curvature V''(0), used only to estimate r_max.
    r_max_cap : float or None
        Hard upper limit for the outer radius used in the tail-growth loop.
        When provided, r_max grows by rmax_growth each retry until the tail
        converges or r_max_cap is reached — the max_rmax_retries count is
        then ignored. Pass np.inf (or 1e15) to allow unlimited growth.
        When None (default), old behaviour is used: at most max_rmax_retries
        growth steps.
    tail_frac : float
        The final integration fires a terminal event when
        ``phi(r) / phi0 < tail_frac``, stopping the ODE as soon as the
        tail has relaxed — exactly like the ``hit_tail`` event in the
        student solver. This means r_max no longer needs to be tuned to
        contain the full tail; the integration finds the natural stopping
        radius by itself. Default 1e-6.
    """

    # ------------------------------------------------------------------
    # Barrier top and shooting field scale
    # ------------------------------------------------------------------
    use_auto_phi = phi_end is None or (isinstance(phi_end, str) and phi_end.lower() == "auto")

    if use_auto_phi:
        phi_max, phi_scan_used = _find_phi_max_auto(
            V,
            dV,
            phi_scan_start=1.0,
            phi_scan_cap=phi_scan_cap,
            n_scan=phi_scan_n,
            phi_min=phi_scan_min,
            verbose=verbose,
        )

        phi_end_eff = _estimate_phi_end_from_potential(
            V,
            phi_max=phi_max,
            phi_scan_max=phi_scan_used,
            phi_cap=phi_scan_cap,
            verbose=verbose,
        )

    else:
        phi_end_eff = float(phi_end)
        phi_max = _find_phi_max(
            V,
            dV,
            phi_scan_max=max(phi_end_eff, 1.0),
            n_scan=phi_scan_n,
            phi_min=phi_scan_min,
        )

    # ------------------------------------------------------------------
    # Radius estimate
    # ------------------------------------------------------------------
    if r_max is None:
        r_max_eff = _estimate_r_max(d2V0=d2V0, T=T)
        if prev_rmax is not None and np.isfinite(prev_rmax):
            r_max_eff = max(r_max_eff, 0.8 * prev_rmax)
    else:
        r_max_eff = float(r_max)

    # ------------------------------------------------------------------
    # Shooting bracket
    # ------------------------------------------------------------------
    phi_lo, phi_hi = _find_bracket(
        phi_max=phi_max,
        phi_end_guess=phi_end_eff,
        dV=dV,
        r_min=r_min,
        r_max=r_max_eff,
        rtol=rtol,
        atol=atol,
        prev_phi0=prev_phi0,
    )

    if verbose:
        print(f"[Find_critbubble] T = {T:.4g}")
        print(f"[Find_critbubble] phi_max = {phi_max:.6g}")
        print(f"[Find_critbubble] phi_end_eff = {phi_end_eff:.6g}")
        print(f"[Find_critbubble] shooting bracket = [{phi_lo:.6g}, {phi_hi:.6g}]")
        print(f"[Find_critbubble] r_max = {r_max_eff:.4g}")
        if d2V0 is not None:
            print(f"[Find_critbubble] d2V(0) = {d2V0:.6g}")
        print(f"[Find_critbubble] Running bisection (tol = {bisect_tol:.1e}) ...")

    # ------------------------------------------------------------------
    # Solve for central value
    # ------------------------------------------------------------------
    phi0_sol = brentq(
        lambda x: _objective(
            x,
            dV=dV,
            r_min=r_min,
            r_max=r_max_eff,
            rtol=rtol,
            atol=atol,
        ),
        phi_lo,
        phi_hi,
        xtol=bisect_tol,
        rtol=bisect_tol,
        maxiter=bisect_max,
    )

    if verbose:
        print(f"[Find_critbubble] phi_0 = {phi0_sol:.12g}")

    # ------------------------------------------------------------------
    # Final profile integration.
    # The tail event (event[1]) fires when phi drops to tail_frac * phi0,
    # so the integration terminates naturally without needing a huge r_max.
    # If the tail has not decayed by r_max, we grow r_max and retry.
    # ------------------------------------------------------------------
    _tail_phi = tail_frac * abs(phi0_sol)

    sol = _integrate_profile(
        phi0=phi0_sol,
        dV=dV,
        r_min=r_min,
        r_max=r_max_eff,
        rtol=rtol,
        atol=atol,
        tail_phi=_tail_phi,
    )

    if sol.y.shape[1] == 0:
        raise RuntimeError("Final profile integration failed immediately.")

    _tail_fired = len(sol.t_events) > 1 and len(sol.t_events[1]) > 0
    tail_ratio = abs(sol.y[0, -1]) / max(abs(phi0_sol), 1e-14)

    # Cap for the tail-growth loop.
    # When r_max_cap is given, use it as the hard limit (retry count ignored).
    # When r_max_cap is None, fall back to old max_rmax_retries behaviour.
    _hard_cap = float(r_max_cap) if r_max_cap is not None else np.inf
    _max_loop = 10000 if r_max_cap is not None else max_rmax_retries
    n_retry = 0

    while (
        len(sol.t_events[0]) == 0   # no overshoot
        and not _tail_fired          # tail event did not fire
        and tail_ratio > 1e-4        # tail not yet small enough
        and n_retry < _max_loop
        and r_max_eff < _hard_cap
    ):
        r_max_eff = min(r_max_eff * rmax_growth, _hard_cap)

        if verbose:
            print(
                f"[Find_critbubble] tail_ratio = {tail_ratio:.3e}; "
                f"increasing r_max to {r_max_eff:.4g}"
            )

        sol = _integrate_profile(
            phi0=phi0_sol,
            dV=dV,
            r_min=r_min,
            r_max=r_max_eff,
            rtol=rtol,
            atol=atol,
            tail_phi=_tail_phi,
        )

        if sol.y.shape[1] == 0:
            raise RuntimeError("Profile integration failed after increasing r_max.")

        _tail_fired = len(sol.t_events) > 1 and len(sol.t_events[1]) > 0
        tail_ratio = abs(sol.y[0, -1]) / max(abs(phi0_sol), 1e-14)
        n_retry += 1

    # Determine storage endpoint:
    #   overshoot → stop at zero crossing
    #   tail event fired → stop when phi = tail_phi
    #   otherwise → store up to r_max
    if len(sol.t_events[0]) > 0:
        r_store = sol.t_events[0][0]
    elif _tail_fired:
        r_store = sol.t_events[1][0]
    else:
        r_store = sol.t[-1]

    # ------------------------------------------------------------------
    # Storage grid
    # ------------------------------------------------------------------
    r_adapt = sol.t
    phi_adapt = sol.y[0]

    r_grid = _build_storage_grid_from_solution(
        r_raw=r_adapt,
        phi_raw=phi_adapt,
        r_min=r_min,
        r_store=r_store,
        n_pts=n_pts,
    )

    y_grid = sol.sol(r_grid)
    r = r_grid
    phi = y_grid[0]
    dphi = y_grid[1]

    # Tiny interpolation noise cleanup only.
    phi = np.where(phi > -1e-13, np.maximum(phi, 0.0), phi)

    # ------------------------------------------------------------------
    # Convergence diagnostic
    # ------------------------------------------------------------------
    tail_ratio = abs(phi[-1]) / max(abs(phi0_sol), 1e-14)
    converged = (
        np.all(phi >= -1e-10)
        and tail_ratio < 1e-4
        and dphi[-1] <= 1e-10
    )

    if verbose:
        status = "✓ converged" if converged else "✗ tail not fully under control"
        print(f"[Find_critbubble] r_store = {r_store:.4g}")
        print(f"[Find_critbubble] phi(r_store) = {phi[-1]:+.4e}   {status}")

    return BounceProfile(
        R=r,
        Phi=phi,
        dPhi=dphi,
        phi0=phi0_sol,
        T=T,
        converged=converged,
    )
