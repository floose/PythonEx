"""OPF on the 10-bus system using the adjoint method (A2).

Total real-power losses are minimised by adjusting reactive power
injections at the same five flex buses as opf_loss_min_gscd.py.

Algorithm: scipy L-BFGS-B with **analytical** gradients computed via
the adjoint method. Each (F, ∇F) evaluation costs:
    1 PF call  +  1 transpose back-solve of the Jacobian
regardless of the number of flex buses — the back-solve gives the full
gradient via reverse-mode chain rule, not d separate forward sensitivities.

Sign convention (matching the existing PF residual):
  g(x, u) = (P_calc - P_spec, Q_calc - Q_spec)
  J = ∂g/∂x  (the Jacobian newton_raphson factors)
  For u_k = Q_gen at flex bus i:  ∂Q_spec_i/∂u_k = +1, so
  ∂g_Q_at_i/∂u_k = -1.
  Adjoint:  J^T λ = -(∂f/∂x)^T
  Gradient: dF/du_k = -λ[Q-row index of bus i in the active state vector]

System data (bus and line tables, flex bus selection) is defined inline
in this script — system inputs are kept explicit per project convention.
"""

import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).parent))
import newton_raphson_pf as pf


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


def find_bus_idx(bus, bus_num):
    return int(np.where(bus[:, pf.BUS_NUM] == bus_num)[0][0])


def set_flex_qs(bus, flex_idx_list, q_vec):
    trial = bus.copy()
    for idx, q in zip(flex_idx_list, q_vec):
        trial[idx, pf.Q_GEN] = q
    return trial


def total_p_loss(V, theta, lines):
    return sum(l["P_loss"] for l in pf.line_losses(V, theta, lines))


def make_adjoint_F_and_grad(bus, lines, flex_idx_list, non_slack_idx, pq_idx):
    """Return (callable, counter): F_and_grad runs PF + adjoint solve.

    counter[0] tracks the number of PF calls (= calls to F_and_grad).
    """
    counter = [0]
    n_ns = len(non_slack_idx)
    flex_q_pos = np.array(
        [int(np.where(pq_idx == idx)[0][0]) for idx in flex_idx_list])

    def F_and_grad(u_vec):
        counter[0] += 1
        bus_trial = set_flex_qs(bus, flex_idx_list, u_vec)
        try:
            V, theta, Y, _, _ = pf.newton_raphson(bus_trial, lines, verbose=False)
        except RuntimeError:
            return float("inf"), np.zeros(len(u_vec))

        F_val = total_p_loss(V, theta, lines)

        df_dtheta, df_dV = pf.loss_state_gradient(V, theta, lines)
        df_dx_active = np.concatenate([df_dtheta[non_slack_idx],
                                       df_dV[pq_idx]])

        P, Q = pf.power_injections(V, theta, Y)
        J = pf.jacobian(V, theta, Y, P, Q, pq_idx, non_slack_idx)
        Jlu = splu(J.tocsc())
        lam = Jlu.solve(-df_dx_active, trans="T")

        grad = -lam[n_ns + flex_q_pos]
        return F_val, grad

    return F_and_grad, counter


def find_minimum_adjoint(bus, lines, flex_idx_list, bounds, tol=1e-6):
    """L-BFGS-B with analytical gradients (adjoint method)."""
    non_slack_idx, pq_idx = pf.state_indices(bus)
    F_and_grad, counter = make_adjoint_F_and_grad(
        bus, lines, flex_idx_list, non_slack_idx, pq_idx)
    x0 = np.array([0.5 * (lo + hi) for lo, hi in bounds])
    res = minimize(F_and_grad, x0, method="L-BFGS-B", jac=True,
                   bounds=bounds, options={"gtol": tol})
    q_star = tuple(float(qi) for qi in res.x)
    return q_star, float(res.fun), counter[0]


def run_adjoint_method(bus, lines, flex_idx_list, bounds, tol=1e-6):
    t0 = time.perf_counter()
    q_star, loss_star, n_evals = find_minimum_adjoint(
        bus, lines, flex_idx_list, bounds, tol=tol)
    elapsed = time.perf_counter() - t0
    return q_star, loss_star, n_evals, elapsed


def run_loss(bus, lines, flex_idx_list, q_vec):
    """Direct PF + loss for the baseline (no gradient)."""
    bus_trial = set_flex_qs(bus, flex_idx_list, q_vec)
    try:
        V, theta, _, _, _ = pf.newton_raphson(bus_trial, lines, verbose=False)
    except RuntimeError:
        return float("inf")
    return total_p_loss(V, theta, lines)


# ---------- output ----------

def print_header(bus, flex_bus_nums):
    n_pq = int(np.sum(bus[:, pf.BUS_TYPE].astype(int) == pf.PQ))
    n_flex = len(flex_bus_nums)
    pct = 100.0 * n_flex / n_pq
    print("=" * 78)
    print("OPF on 10-bus system: adjoint method (L-BFGS-B + analytical ∇F)")
    print("=" * 78)
    print(f"Total load:  P = {np.sum(bus[:, pf.P_LOAD]):.2f} pu, "
          f"Q = {np.sum(bus[:, pf.Q_LOAD]):.2f} pu")
    print(f"Flex buses:  {flex_bus_nums} — "
          f"{n_flex} of {n_pq} PQ buses ({pct:.0f}%)")


def print_optimization_result(q_star, flex_bus_nums, n_evals, wall_s, tol):
    print(f"\nSolver:      L-BFGS-B with adjoint gradients, gtol = {tol:.0e}")
    print(f"Result:      converged in {n_evals} (F, ∇F) calls, "
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
    bounds = [(-0.5, 1.0)] * len(flex_idx_list)
    tol = 1e-6

    print_header(bus, FLEX_BUS_NUMS)

    q_star, loss_star, n_evals, elapsed = run_adjoint_method(
        bus, lines, flex_idx_list, bounds, tol=tol)

    print_optimization_result(q_star, FLEX_BUS_NUMS, n_evals, elapsed, tol)

    baseline = run_loss(bus, lines, flex_idx_list, [0.0] * len(flex_idx_list))
    print_summary(baseline, loss_star)


if __name__ == "__main__":
    main()
