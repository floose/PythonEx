"""OPF on a 10-bus system using binary search (golden-section coordinate descent).

A 10-bus meshed system with 1 slack, 2 PV generators and 7 PQ buses.
Five of the seven PQ buses (70%) are flexible reactive-power injectors.
The optimisation finds the Q values that minimise total real-power
losses subject to bound constraints.

Algorithm: coordinate descent + golden-section line search per axis.
Each axis sweep narrows its bracket by factor 1/φ ≈ 0.618; outer
iterations repeat until improvement falls below tolerance. Cost scales
as O(d · log(range/tol)) — independent of the number of grid points.

History — DEPRECATED brute-force comparison
    Earlier versions of this script also included a brute-force grid
    sweep for comparison. At d = 2 it was competitive (121 PFs vs ~75);
    by d = 5 the cliff is decisive: 11^5 = 161 051 PFs (~5 minutes) vs
    binary search's ~400 PFs (~1 second), at ~150x worse precision.
    Brute force was retired in this version. The functions
    sweep_grid_brute_force, find_minimum_brute_force,
    print_grid_2d_brute_force, run_brute_force_method, and
    print_comparison were removed; the git history preserves them.
"""

import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import newton_raphson_pf as pf


INF = 9999.0
FLEX_BUS_NUMS = (3, 4, 6, 8, 9)
PHI = (np.sqrt(5) - 1) / 2  # golden ratio reciprocal, ~0.618


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


def find_bus_idx(bus, bus_num):
    return int(np.where(bus[:, pf.BUS_NUM] == bus_num)[0][0])


def set_flex_qs(bus, flex_idx_list, q_vec):
    trial = bus.copy()
    for idx, q in zip(flex_idx_list, q_vec):
        trial[idx, pf.Q_GEN] = q
    return trial


def total_p_loss(V, theta, lines):
    return sum(l["P_loss"] for l in pf.line_losses(V, theta, lines))


def run_loss(bus, lines, flex_idx_list, q_vec):
    """Set Q_GEN at each flex bus, solve PF, return total real-power loss."""
    trial = set_flex_qs(bus, flex_idx_list, q_vec)
    try:
        V, theta, _, _, _ = pf.newton_raphson(trial, lines, verbose=False)
    except RuntimeError:
        return float("inf")
    return total_p_loss(V, theta, lines)


# ---------- binary search (golden-section coordinate descent) ----------

def _golden_section_1d(f, lo, hi, tol):
    """Find x in [lo, hi] that minimises a unimodal scalar f.

    Returns (x_star, n_evals). Each iteration shrinks the bracket by
    factor PHI (~0.618) and costs one new f evaluation.
    """
    a, b = lo, hi
    x1 = a + (1 - PHI) * (b - a)
    x2 = a + PHI * (b - a)
    f1, f2 = f(x1), f(x2)
    n_evals = 2
    while (b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + (1 - PHI) * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + PHI * (b - a)
            f2 = f(x2)
        n_evals += 1
    return 0.5 * (a + b), n_evals


def find_minimum_binary_search(bus, lines, flex_idx_list, bounds,
                               tol=1e-3, max_outer=20):
    """Coordinate descent + golden-section line search.

    Runs golden-section minimisation along each axis in turn, repeats
    outer sweeps until improvement falls below tol or max_outer hit.
    Returns (q_star, loss_star, n_evals).
    """
    d = len(flex_idx_list)
    q = np.array([0.5 * (lo + hi) for lo, hi in bounds])
    n_evals = 0
    prev_loss = run_loss(bus, lines, flex_idx_list, q.tolist())
    n_evals += 1
    new_loss = prev_loss

    for _ in range(max_outer):
        for axis in range(d):
            def f_axis(qa, _axis=axis):
                trial = q.copy()
                trial[_axis] = qa
                return run_loss(bus, lines, flex_idx_list, trial.tolist())

            lo, hi = bounds[axis]
            x_star, n_calls = _golden_section_1d(f_axis, lo, hi, tol)
            q[axis] = x_star
            n_evals += n_calls

        new_loss = run_loss(bus, lines, flex_idx_list, q.tolist())
        n_evals += 1
        if abs(prev_loss - new_loss) < tol:
            break
        prev_loss = new_loss

    return tuple(float(qi) for qi in q), float(new_loss), n_evals


# ---------- timing wrapper ----------

def run_binary_search_method(bus, lines, flex_idx_list, bounds,
                             tol=1e-3, max_outer=20):
    t0 = time.perf_counter()
    q_star, loss_star, n_evals = find_minimum_binary_search(
        bus, lines, flex_idx_list, bounds, tol=tol, max_outer=max_outer)
    elapsed = time.perf_counter() - t0
    return q_star, loss_star, n_evals, elapsed


# ---------- output ----------

def print_header(bus, flex_bus_nums):
    n_pq = int(np.sum(bus[:, pf.BUS_TYPE].astype(int) == pf.PQ))
    n_flex = len(flex_bus_nums)
    pct = 100.0 * n_flex / n_pq
    print("=" * 78)
    print("OPF on 10-bus system: binary search (golden-section coordinate descent)")
    print("=" * 78)
    print(f"Total load:  P = {np.sum(bus[:, pf.P_LOAD]):.2f} pu, "
          f"Q = {np.sum(bus[:, pf.Q_LOAD]):.2f} pu")
    print(f"Flex buses:  {flex_bus_nums} — "
          f"{n_flex} of {n_pq} PQ buses ({pct:.0f}%)")


def print_optimization_result(q_star, flex_bus_nums, n_evals, wall_s,
                              tol, max_outer):
    print(f"\nSolver:      tol = {tol:.0e}, max_outer = {max_outer}")
    print(f"Result:      converged in {n_evals} PF calls, "
          f"{wall_s:.2f} s wall time")
    print("\nOptimum (Q at flex buses):")
    for b, q in zip(flex_bus_nums, q_star):
        print(f"  Q_{b} = {q:+.3f} pu")


def print_summary(baseline, loss_star):
    reduction = (baseline - loss_star) / baseline * 100
    print("\n" + "=" * 78)
    print(f"Baseline (all flex Q = 0):   {baseline:>9.5f} pu")
    print(f"Minimum loss:                 {loss_star:>9.5f} pu")
    print(f"Loss reduction:               {reduction:>9.2f} %")
    print("=" * 78)


def main():
    bus = bus_data()
    lines = line_data()
    flex_idx_list = [find_bus_idx(bus, n) for n in FLEX_BUS_NUMS]

    print_header(bus, FLEX_BUS_NUMS)

    bounds = [(-0.5, 1.0)] * len(flex_idx_list)
    tol = 1e-3
    max_outer = 20

    q_star, loss_star, n_evals, elapsed = run_binary_search_method(
        bus, lines, flex_idx_list, bounds, tol=tol, max_outer=max_outer)

    print_optimization_result(q_star, FLEX_BUS_NUMS, n_evals,
                              elapsed, tol, max_outer)

    baseline = run_loss(bus, lines, flex_idx_list, [0.0] * len(flex_idx_list))
    print_summary(baseline, loss_star)


if __name__ == "__main__":
    main()
