"""Validate GSCD against scipy.optimize.minimize(L-BFGS-B) on the 10-bus OPF.

Both methods minimise total real-power losses by adjusting reactive power
injections at the same five flex buses, starting from the same initial
guess (midpoint of bounds). The test PASSES if both reach the same local
optimum within tolerance.

The system data (bus and line tables, flex bus selection) is defined
inline in this file by deliberate choice — system inputs are kept
explicit in test scripts so the test's assumptions are self-contained
and visible at a glance.
"""

import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))
import newton_raphson_pf as pf
import opf_loss_min_gscd as gscd


# Verdict tolerances
LOSS_TOL = 5e-4  # accommodates GSCD's coordinate-descent precision floor
Q_TOL = 3e-1    # loose: flat valleys allow substitution between compensators


# ---------- system data (kept inline by design) ----------

INF = 9999.0
FLEX_BUS_NUMS = (3, 4, 6, 8, 9)


def bus_data():
    # bus, type, V, theta_deg, Pgen, Qgen, Pload, Qload, Q_min, Q_max
    return np.array([
        [1,  pf.SLACK, 1.05, 0.0, 0.0,  0.0, 0.0,  0.0,  -INF, INF],
        [2,  pf.PV,    1.03, 0.0, 0.80, 0.0, 0.0,  0.0,  -INF, INF],
        [3,  pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.60, 0.25, -INF, INF],
        [4,  pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.70, 0.30, -INF, INF],
        [5,  pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.40, 0.15, -INF, INF],
        [6,  pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.0,  0.0,  -INF, INF],
        [7,  pf.PV,    1.04, 0.0, 0.80, 0.0, 0.0,  0.0,  -INF, INF],
        [8,  pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.50, 0.20, -INF, INF],
        [9,  pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.40, 0.15, -INF, INF],
        [10, pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.30, 0.10, -INF, INF],
    ])


def line_data():
    # from, to, R, X, B_total
    return np.array([
        [1, 2, 0.02, 0.06, 0.06],
        [1, 3, 0.08, 0.24, 0.04],
        [2, 4, 0.04, 0.12, 0.04],
        [3, 4, 0.03, 0.09, 0.03],
        [3, 5, 0.05, 0.15, 0.02],
        [4, 5, 0.02, 0.06, 0.02],
        [5, 6, 0.01, 0.03, 0.01],
        [4, 7, 0.03, 0.09, 0.03],
        [7, 10, 0.04, 0.12, 0.03],
        [5, 8, 0.04, 0.12, 0.03],
        [8, 9, 0.03, 0.09, 0.02],
        [6, 10, 0.05, 0.15, 0.03],
    ])


def run_gscd(bus, lines, flex_idx_list, bounds, tol):
    """Wrap the GSCD method. Returns (q_star, loss_star, n_evals, wall_s)."""
    return gscd.run_gscd_method(bus, lines, flex_idx_list, bounds, tol=tol)


def make_counted_objective(bus, lines, flex_idx_list):
    """Return (objective_fn, counter) where counter[0] tracks PF calls."""
    counter = [0]

    def objective(q_vec):
        counter[0] += 1
        return gscd.run_loss(bus, lines, flex_idx_list, list(q_vec))

    return objective, counter


def run_lbfgsb(bus, lines, flex_idx_list, bounds):
    """Wrap scipy L-BFGS-B (default convergence options).

    Tolerance is intentionally not passed: scipy's `tol` for L-BFGS-B is
    a relative-ftol that maps loosely to a Q-precision tolerance. Letting
    scipy run to its tight defaults gives a fair "did GSCD reach as good
    a minimum as a properly converged scipy optimiser" comparison.

    Returns (q_star, loss_star, n_evals, wall_s).
    """
    objective, counter = make_counted_objective(bus, lines, flex_idx_list)
    x0 = np.array([0.5 * (lo + hi) for lo, hi in bounds])
    t0 = time.perf_counter()
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    elapsed = time.perf_counter() - t0
    q_star = tuple(float(qi) for qi in res.x)
    return q_star, float(res.fun), counter[0], elapsed


def compare_results(gscd_res, scipy_res, loss_tol, q_tol):
    """Return (passed, gap_loss, gap_q_max) between two method results."""
    gscd_q, gscd_loss, *_ = gscd_res
    scipy_q, scipy_loss, *_ = scipy_res
    gap_loss = abs(gscd_loss - scipy_loss)
    gap_q_max = float(np.max(np.abs(np.array(gscd_q) - np.array(scipy_q))))
    passed = gap_loss < loss_tol and gap_q_max < q_tol
    return passed, gap_loss, gap_q_max


def print_method_block(name, q_star, loss_star, n_evals, wall_s, flex_bus_nums):
    print(f"\n{name}")
    print(f"  P_loss:    {loss_star:.5f} pu")
    print(f"  PF calls:  {n_evals}")
    print(f"  wall:      {wall_s:.2f} s")
    q_str = "  ".join(f"Q_{b}={q:+.3f}" for b, q in zip(flex_bus_nums, q_star))
    print(f"  Q*:        {q_str}")


def print_comparison_block(gap_loss, gap_q_max, loss_tol, q_tol, passed):
    print("\n" + "-" * 78)
    print("Comparison")
    print(f"  ΔP_loss:    {gap_loss:.2e} pu     (tol {loss_tol:.0e})")
    print(f"  max |Δq|:   {gap_q_max:.2e} pu     (tol {q_tol:.0e})")
    print(f"  Verdict:    {'PASS' if passed else 'FAIL'}")
    print("=" * 78)


def main():
    bus = bus_data()
    lines = line_data()
    flex_idx_list = [gscd.find_bus_idx(bus, n) for n in FLEX_BUS_NUMS]
    bounds = [(-0.5, 1.0)] * len(flex_idx_list)
    tol = 1e-3

    print("=" * 78)
    print("GSCD vs scipy.optimize.minimize(L-BFGS-B) — 10-bus system")
    print("=" * 78)
    print(f"Flex buses: {FLEX_BUS_NUMS}")
    print("Bounds:     Q_i ∈ [-0.5, 1.0] pu per axis")
    print(f"GSCD tol:   {tol:.0e}  (scipy uses its defaults)")
    print("-" * 78)

    gscd_res = run_gscd(bus, lines, flex_idx_list, bounds, tol)
    scipy_res = run_lbfgsb(bus, lines, flex_idx_list, bounds)

    print_method_block("GSCD (golden-section coordinate descent)",
                       *gscd_res, FLEX_BUS_NUMS)
    print_method_block("L-BFGS-B (scipy quasi-Newton, numerical gradient)",
                       *scipy_res, FLEX_BUS_NUMS)

    passed, gap_loss, gap_q_max = compare_results(
        gscd_res, scipy_res, LOSS_TOL, Q_TOL)
    print_comparison_block(gap_loss, gap_q_max, LOSS_TOL, Q_TOL, passed)


if __name__ == "__main__":
    main()
