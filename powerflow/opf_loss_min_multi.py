"""OPF phase 1b: brute-force sweep with multiple flexible Q injectors.

Same 6-bus system as opf_loss_min.py, but now with two compensators —
one at bus 4 (a load bus) and one at bus 6 (the radial tail). The PF
is run over an N-D grid of (Q_4, Q_6) values to find the combination
that minimises total real-power losses.

The implementation is N-D in principle (any number of flex buses);
the demo uses two for visualisation, and the loss table is printed as
a 2-D grid. With more than two, only the minimum is reported.
"""

import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import newton_raphson_pf as pf


INF = 9999.0
FLEX_BUS_NUMS = (4, 2)
PHI = (np.sqrt(5) - 1) / 2  # golden ratio reciprocal, ~0.618


def bus_data():
    # bus, type, V, theta_deg, Pgen, Qgen, Pload, Qload, Q_min, Q_max
    return np.array([
        [1, pf.SLACK, 1.05, 0.0, 0.0,  0.0, 0.0,  0.0,  -INF, INF],
        [2, pf.PV,    1.03, 0.0, 0.80, 0.0, 0.0,  0.0,  -INF, INF],
        [3, pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.60, 0.25, -INF, INF],
        [4, pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.70, 0.30, -INF, INF],
        [5, pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.40, 0.15, -INF, INF],
        [6, pf.PQ,    1.00, 0.0, 0.0,  0.0, 0.0,  0.0,  -INF, INF],
    ])


def line_data():
    return np.array([
        [1, 2, 0.02, 0.06, 0.06],
        [1, 3, 0.08, 0.24, 0.04],
        [2, 4, 0.04, 0.12, 0.04],
        [3, 4, 0.03, 0.09, 0.03],
        [3, 5, 0.05, 0.15, 0.02],
        [4, 5, 0.02, 0.06, 0.02],
        [5, 6, 0.01, 0.03, 0.01],
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


def sweep_grid_brute_force(bus, lines, flex_idx_list, q_axes):
    """N-D sweep. Returns a loss array shaped like the grid axes."""
    shape = tuple(len(ax) for ax in q_axes)
    losses = np.empty(shape)
    for idx in np.ndindex(*shape):
        q_vec = [q_axes[d][idx[d]] for d in range(len(q_axes))]
        losses[idx] = run_loss(bus, lines, flex_idx_list, q_vec)
    return losses


def find_minimum_brute_force(q_axes, losses):
    flat = int(np.argmin(losses))
    multi = np.unravel_index(flat, losses.shape)
    q_star = tuple(q_axes[d][multi[d]] for d in range(len(q_axes)))
    return q_star, float(losses[multi]), multi


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


# ---------- method runners (timing wrappers) ----------

def run_brute_force_method(bus, lines, flex_idx_list, n_per_axis, bounds):
    q_axes = [np.linspace(lo, hi, n_per_axis) for lo, hi in bounds]
    t0 = time.perf_counter()
    losses = sweep_grid_brute_force(bus, lines, flex_idx_list, q_axes)
    q_star, loss_star, idx_star = find_minimum_brute_force(q_axes, losses)
    elapsed = time.perf_counter() - t0
    n_evals = int(np.prod([len(ax) for ax in q_axes]))
    return q_star, loss_star, n_evals, elapsed, q_axes, losses, idx_star


def run_binary_search_method(bus, lines, flex_idx_list, bounds, tol=1e-3):
    t0 = time.perf_counter()
    q_star, loss_star, n_evals = find_minimum_binary_search(
        bus, lines, flex_idx_list, bounds, tol=tol)
    elapsed = time.perf_counter() - t0
    return q_star, loss_star, n_evals, elapsed


def print_header(bus, flex_bus_nums):
    print("=" * 70)
    print(f"OPF phase 1b: brute-force sweep at flex buses {flex_bus_nums}")
    print("=" * 70)
    print(f"Total load:  P = {np.sum(bus[:, pf.P_LOAD]):.2f} pu, "
          f"Q = {np.sum(bus[:, pf.Q_LOAD]):.2f} pu")


def print_grid_2d_brute_force(q_axes, losses, idx_star, flex_bus_nums):
    qa, qb = q_axes
    bus_a, bus_b = flex_bus_nums
    print(f"\nLoss grid (P_loss in pu).  cols = Q_{bus_a},  rows = Q_{bus_b}\n")
    header = "         " + "".join(f"{q:>8.2f}" for q in qa)
    print(header)
    print("        +" + "-" * (8 * len(qa)))
    for j, q_b in enumerate(qb):
        line = f"{q_b:>+7.2f} |"
        for i in range(len(qa)):
            val = losses[i, j]
            mark = "*" if (i, j) == idx_star else " "
            line += f"{val:>7.4f}{mark}"
        print(line)
    print("\n(* marks the minimum)")


def print_summary(baseline, q_star, loss_star, flex_bus_nums):
    reduction = (baseline - loss_star) / baseline * 100
    q_str = ", ".join(f"Q_{b}={q:+.3f}" for b, q in zip(flex_bus_nums, q_star))
    print("\n" + "=" * 70)
    print(f"Baseline (all flex Q = 0):  {baseline:>9.5f} pu")
    print(f"Minimum loss:                {loss_star:>9.5f} pu  ({q_str})")
    print(f"Loss reduction:              {reduction:>9.2f} %")
    print("=" * 70)


def print_comparison(results):
    """Print a comparison table for multiple OPF method results.

    `results` items: (method_name, q_star, loss_star, n_evals, wall_ms).
    """
    print("\n" + "=" * 80)
    print("METHOD COMPARISON")
    print("=" * 80)
    print(f"{'Method':<32} {'P_loss (pu)':>13} {'PF calls':>10} {'wall (ms)':>12}")
    print("-" * 80)
    for name, _, loss_star, n_evals, wall_ms in results:
        print(f"{name:<32} {loss_star:>13.6f} {n_evals:>10d} {wall_ms:>12.2f}")
    print("-" * 80)
    print("\nMinimisers found (Q values per flex bus):")
    for name, q_star, *_ in results:
        q_str = ", ".join(f"{q:+.4f}" for q in q_star)
        print(f"  {name:<32} ({q_str})")
    print("=" * 80)


def main():
    bus = bus_data()
    lines = line_data()
    flex_idx_list = [find_bus_idx(bus, n) for n in FLEX_BUS_NUMS]

    print_header(bus, FLEX_BUS_NUMS)

    bounds = [(-0.5, 1.0)] * len(flex_idx_list)

    bf_q, bf_loss, bf_calls, bf_t, q_axes, losses, idx_star = \
        run_brute_force_method(bus, lines, flex_idx_list, 11, bounds)
    if len(flex_idx_list) == 2:
        print_grid_2d_brute_force(q_axes, losses, idx_star, FLEX_BUS_NUMS)

    bs_q, bs_loss, bs_calls, bs_t = run_binary_search_method(
        bus, lines, flex_idx_list, bounds, tol=1e-3)

    bf21_q, bf21_loss, bf21_calls, bf21_t, *_ = run_brute_force_method(
        bus, lines, flex_idx_list, 21, bounds)
    bf51_q, bf51_loss, bf51_calls, bf51_t, *_ = run_brute_force_method(
        bus, lines, flex_idx_list, 51, bounds)

    results = [
        ("brute force 11x11",        bf_q,   bf_loss,   bf_calls,   bf_t * 1000),
        ("brute force 21x21",        bf21_q, bf21_loss, bf21_calls, bf21_t * 1000),
        ("brute force 51x51",        bf51_q, bf51_loss, bf51_calls, bf51_t * 1000),
        ("binary search (gs c-desc)",bs_q,   bs_loss,   bs_calls,   bs_t * 1000),
    ]
    print_comparison(results)

    baseline = run_loss(bus, lines, flex_idx_list, [0.0] * len(flex_idx_list))
    print_summary(baseline, bs_q, bs_loss, FLEX_BUS_NUMS)


if __name__ == "__main__":
    main()
