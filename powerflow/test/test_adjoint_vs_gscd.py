"""Validate the adjoint method against GSCD on the 10-bus OPF.

Two checks:
  [1] Gradient unit test — at three probe points, the analytical adjoint
      gradient must match a central-difference reference within abs_tol.
      Catches sign convention / index mapping errors.
  [2] End-to-end comparison — both methods solve the same problem and
      should reach the same local optimum within tolerance, with PF call
      counts and wall times reported.

System data (bus and line tables, flex bus selection) is defined inline
in this file by deliberate choice — system inputs are kept explicit in
test scripts so the test's assumptions are self-contained.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import newton_raphson_pf as pf
import opf_loss_min_gscd as gscd
import opf_loss_min_adjoint as adj


# Verdict tolerances
LOSS_TOL = 5e-4
Q_TOL = 3e-1
GRAD_TOL = 1e-4

# Solver tolerances
GSCD_TOL = 1e-3
ADJ_TOL = 1e-6


# ---------- system data (inline by design) ----------

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


# ---------- gradient unit test ----------

def fd_central_gradient(bus, lines, flex_idx_list, u_vec, h=1e-5):
    """Central-difference gradient: O(h^2) truncation, two PFs per axis."""
    n = len(u_vec)
    grad = np.zeros(n)
    u = np.asarray(u_vec, dtype=float)
    for i in range(n):
        up = u.copy(); up[i] += h
        um = u.copy(); um[i] -= h
        F_p = gscd.run_loss(bus, lines, flex_idx_list, up.tolist())
        F_m = gscd.run_loss(bus, lines, flex_idx_list, um.tolist())
        grad[i] = (F_p - F_m) / (2 * h)
    return grad


def gradient_unit_test(bus, lines, flex_idx_list, probes, abs_tol):
    """Compare adjoint gradient to central-FD reference at each probe.

    Returns (all_passed, [(name, max_abs_err, grad_an, grad_fd, passed), ...]).
    """
    non_slack_idx, pq_idx = pf.state_indices(bus)
    F_and_grad, _ = adj.make_adjoint_F_and_grad(
        bus, lines, flex_idx_list, non_slack_idx, pq_idx)

    results = []
    all_passed = True
    for name, u_vec in probes:
        u = np.asarray(u_vec, dtype=float)
        _, grad_an = F_and_grad(u)
        grad_fd = fd_central_gradient(bus, lines, flex_idx_list, u)
        max_err = float(np.max(np.abs(grad_an - grad_fd)))
        passed = max_err < abs_tol
        all_passed = all_passed and passed
        results.append((name, max_err, grad_an, grad_fd, passed))
    return all_passed, results


# ---------- method runners ----------

def run_gscd(bus, lines, flex_idx_list, bounds):
    return gscd.run_gscd_method(bus, lines, flex_idx_list, bounds, tol=GSCD_TOL)


def run_adjoint(bus, lines, flex_idx_list, bounds):
    return adj.run_adjoint_method(bus, lines, flex_idx_list, bounds, tol=ADJ_TOL)


def compare_results(gscd_res, adj_res, loss_tol, q_tol):
    """Return (passed, gap_loss, gap_q_max) between two method results."""
    gscd_q, gscd_loss, *_ = gscd_res
    adj_q, adj_loss, *_ = adj_res
    gap_loss = abs(gscd_loss - adj_loss)
    gap_q_max = float(np.max(np.abs(np.array(gscd_q) - np.array(adj_q))))
    passed = gap_loss < loss_tol and gap_q_max < q_tol
    return passed, gap_loss, gap_q_max


# ---------- output ----------

def print_header(bus, flex_bus_nums):
    n_pq = int(np.sum(bus[:, pf.BUS_TYPE].astype(int) == pf.PQ))
    n_flex = len(flex_bus_nums)
    pct = 100.0 * n_flex / n_pq
    print("=" * 78)
    print("Adjoint method vs GSCD — 10-bus system")
    print("=" * 78)
    print(f"Total load:  P = {np.sum(bus[:, pf.P_LOAD]):.2f} pu, "
          f"Q = {np.sum(bus[:, pf.Q_LOAD]):.2f} pu")
    print(f"Flex buses:  {flex_bus_nums} — "
          f"{n_flex} of {n_pq} PQ buses ({pct:.0f}%)")
    print("Bounds:      Q_i ∈ [-0.5, 1.0] pu per axis")
    print(f"Tolerances:  GSCD tol = {GSCD_TOL:.0e},  "
          f"adjoint gtol = {ADJ_TOL:.0e}")


def print_gradient_test(results, abs_tol, all_passed):
    print("\n" + "-" * 78)
    print("[1] Gradient unit test  (adjoint vs central finite differences, h=1e-5)")
    print(f"    abs_tol = {abs_tol:.0e}")
    print()
    for name, max_err, _, _, passed in results:
        verdict = "PASS" if passed else "FAIL"
        print(f"  {name:<28}  max |∇F_adj − ∇F_fd| = {max_err:.2e}  [{verdict}]")
    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")


def print_method_block(name, q_star, loss_star, n_evals, wall_s, flex_bus_nums):
    print(f"\n{name}")
    print(f"  P_loss:    {loss_star:.5f} pu")
    print(f"  PF calls:  {n_evals}")
    print(f"  wall:      {wall_s:.2f} s")
    q_str = "  ".join(f"Q_{b}={q:+.3f}" for b, q in zip(flex_bus_nums, q_star))
    print(f"  Q*:        {q_str}")


def print_comparison_block(gap_loss, gap_q_max, loss_tol, q_tol, passed):
    print("\n" + "-" * 78)
    print("[2] End-to-end comparison")
    print(f"  ΔP_loss:    {gap_loss:.2e} pu     (tol {loss_tol:.0e})")
    print(f"  max |Δq|:   {gap_q_max:.2e} pu     (tol {q_tol:.0e})")
    print(f"  Verdict:    {'PASS' if passed else 'FAIL'}")
    print("=" * 78)


def main():
    bus = bus_data()
    lines = line_data()
    flex_idx_list = [gscd.find_bus_idx(bus, n) for n in FLEX_BUS_NUMS]
    bounds = [(-0.5, 1.0)] * len(flex_idx_list)

    print_header(bus, FLEX_BUS_NUMS)

    gscd_res = run_gscd(bus, lines, flex_idx_list, bounds)
    adj_res = run_adjoint(bus, lines, flex_idx_list, bounds)

    probes = [
        ("baseline (Q_i = 0)", [0.0] * len(flex_idx_list)),
        ("midpoint (Q_i = +0.25)", [0.25] * len(flex_idx_list)),
        ("optimum (adjoint u*)", list(adj_res[0])),
    ]
    grad_passed, grad_results = gradient_unit_test(
        bus, lines, flex_idx_list, probes, GRAD_TOL)
    print_gradient_test(grad_results, GRAD_TOL, grad_passed)

    print_method_block("GSCD (golden-section coord. descent)",
                       *gscd_res, FLEX_BUS_NUMS)
    print_method_block("Adjoint (L-BFGS-B + analytical ∇F)",
                       *adj_res, FLEX_BUS_NUMS)

    e2e_passed, gap_loss, gap_q_max = compare_results(
        gscd_res, adj_res, LOSS_TOL, Q_TOL)
    print_comparison_block(gap_loss, gap_q_max, LOSS_TOL, Q_TOL, e2e_passed)


if __name__ == "__main__":
    main()
