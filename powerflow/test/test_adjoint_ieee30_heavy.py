"""Adjoint OPF on the IEEE 30-bus benchmark, heavy-peak loading.

The IEEE 30-bus is a canonical 1961-vintage AEP system snapshot.
Native voltage levels: 132 kV transmission + 33 kV sub-transmission.
Default total load: 283.4 MW; we apply a 1.05× heavy-peak scaling factor
to all loads, producing a stressed baseline ~6.7 % losses (in the
literature-supported 5–7 % heavy-peak range). Note: our PF model omits
bus-shunt entries and transformer tap ratios that the canonical IEEE 30
includes — those simplifications already raise losses at default loading
to ~5 %, so a small extra peak factor (1.05×) is sufficient to land in
the target window. With full MATPOWER fidelity, hitting 5–7 % requires
a much larger scaling (~1.5–1.6×).

Option B (revised) flex selection (d = 10):
  - 10 shunt-compensator controls at buses 10, 12, 15, 17, 19, 20, 21, 23,
    24, 29 (continuous-Q proxy for SVCs/STATCOMs at the heaviest load buses).
  - The 5 PV generators (2, 5, 8, 11, 13) keep their published V setpoints
    and dispatch Q automatically through PV-mode operation. Demoting them
    to PQ-at-Q=0 (originally d=15) removes all generator voltage support
    simultaneously, which makes the baseline PF non-convergent at 1.6×.
    Real ORPD studies treat generator-V as the control variable, not
    generator-Q; we approximate that here by leaving gens as PV.

Baseline: Q_i = 0 ∀ flex variable. Standard academic-benchmark convention
(Granville 1994; Wood, Wollenberg & Sheblé 2014, ch. 8). The 5-7 % loss
range under heavy peak is documented in stressed-IEEE-30 studies
(Alsac & Stott 1974; Mantovani & Garcia 1996).

System data is defined inline (see ieee30_system.md for the data sheet).
Transformer branches are treated as plain π-model — tap ratios skipped.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import newton_raphson_pf as pf
import opf_loss_min_adjoint as adj


# ---------- bases and tolerances ----------

BASE_MVA = 100.0
INF = 9999.0

HEAVY_PEAK_FACTOR = 1.05  # multiplies all P_LOAD and Q_LOAD entries

GRAD_TOL = 1e-3   # looser at d=15 with bigger gradient magnitudes
ADJ_TOL = 1e-6


# ---------- flex selection (Option B revised: shunts only) ----------

SHUNT_BUSES = (10, 12, 15, 17, 19, 20, 21, 23, 24, 29)
FLEX_BUS_NUMS = SHUNT_BUSES   # d = 10

SHUNT_Q_BOUNDS = (-0.10, 0.30)


# ---------- system construction ----------

def bus_data():
    """IEEE 30-bus, gens kept as PV (V-setpoint controlled).

    P_LOAD and Q_LOAD scaled by HEAVY_PEAK_FACTOR. P_GEN values match
    the canonical case (only bus 2 has non-zero scheduled P; the slack
    absorbs the rest). PV buses 2, 5, 8, 11, 13 carry their published
    V setpoints; their Q is solved by the PF (PV mode).
    """
    F = HEAVY_PEAK_FACTOR
    return np.array([
        [1,  pf.SLACK, 1.060, 0.0, 0.00, 0.0, 0.0,        0.0,        -INF, INF],
        [2,  pf.PV,    1.043, 0.0, 0.40, 0.0, 0.217 * F,  0.127 * F,  -0.40, 0.50],
        [3,  pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.024 * F,  0.012 * F,  -INF, INF],
        [4,  pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.076 * F,  0.016 * F,  -INF, INF],
        [5,  pf.PV,    1.010, 0.0, 0.00, 0.0, 0.942 * F,  0.190 * F,  -0.40, 0.40],
        [6,  pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.0,        0.0,        -INF, INF],
        [7,  pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.228 * F,  0.109 * F,  -INF, INF],
        [8,  pf.PV,    1.010, 0.0, 0.00, 0.0, 0.300 * F,  0.300 * F,  -0.10, 0.40],
        [9,  pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.0,        0.0,        -INF, INF],
        [10, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.058 * F,  0.020 * F,  -INF, INF],
        [11, pf.PV,    1.082, 0.0, 0.00, 0.0, 0.0,        0.0,        -0.06, 0.24],
        [12, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.112 * F,  0.075 * F,  -INF, INF],
        [13, pf.PV,    1.071, 0.0, 0.00, 0.0, 0.0,        0.0,        -0.06, 0.24],
        [14, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.062 * F,  0.016 * F,  -INF, INF],
        [15, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.082 * F,  0.025 * F,  -INF, INF],
        [16, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.035 * F,  0.018 * F,  -INF, INF],
        [17, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.090 * F,  0.058 * F,  -INF, INF],
        [18, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.032 * F,  0.009 * F,  -INF, INF],
        [19, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.095 * F,  0.034 * F,  -INF, INF],
        [20, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.022 * F,  0.007 * F,  -INF, INF],
        [21, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.175 * F,  0.112 * F,  -INF, INF],
        [22, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.0,        0.0,        -INF, INF],
        [23, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.032 * F,  0.016 * F,  -INF, INF],
        [24, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.087 * F,  0.067 * F,  -INF, INF],
        [25, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.0,        0.0,        -INF, INF],
        [26, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.035 * F,  0.023 * F,  -INF, INF],
        [27, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.0,        0.0,        -INF, INF],
        [28, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.0,        0.0,        -INF, INF],
        [29, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.024 * F,  0.009 * F,  -INF, INF],
        [30, pf.PQ,    1.000, 0.0, 0.00, 0.0, 0.106 * F,  0.019 * F,  -INF, INF],
    ])


def line_data():
    """41 branches: 37 lines + 4 transformers (treated as plain π)."""
    return np.array([
        [1,  2,  0.0192, 0.0575, 0.0528],
        [1,  3,  0.0452, 0.1652, 0.0408],
        [2,  4,  0.0570, 0.1737, 0.0368],
        [3,  4,  0.0132, 0.0379, 0.0084],
        [2,  5,  0.0472, 0.1983, 0.0418],
        [2,  6,  0.0581, 0.1763, 0.0374],
        [4,  6,  0.0119, 0.0414, 0.0090],
        [5,  7,  0.0460, 0.1160, 0.0204],
        [6,  7,  0.0267, 0.0820, 0.0170],
        [6,  8,  0.0120, 0.0420, 0.0090],
        [6,  9,  0.0,    0.2080, 0.0],     # transformer
        [6,  10, 0.0,    0.5560, 0.0],     # transformer
        [9,  11, 0.0,    0.2080, 0.0],
        [9,  10, 0.0,    0.1100, 0.0],
        [4,  12, 0.0,    0.2560, 0.0],     # transformer
        [12, 13, 0.0,    0.1400, 0.0],
        [12, 14, 0.1231, 0.2559, 0.0],
        [12, 15, 0.0662, 0.1304, 0.0],
        [12, 16, 0.0945, 0.1987, 0.0],
        [14, 15, 0.2210, 0.1997, 0.0],
        [16, 17, 0.0824, 0.1923, 0.0],
        [15, 18, 0.1073, 0.2185, 0.0],
        [18, 19, 0.0639, 0.1292, 0.0],
        [19, 20, 0.0340, 0.0680, 0.0],
        [10, 20, 0.0936, 0.2090, 0.0],
        [10, 17, 0.0324, 0.0845, 0.0],
        [10, 21, 0.0348, 0.0749, 0.0],
        [10, 22, 0.0727, 0.1499, 0.0],
        [21, 22, 0.0116, 0.0236, 0.0],
        [15, 23, 0.1000, 0.2020, 0.0],
        [22, 24, 0.1150, 0.1790, 0.0],
        [23, 24, 0.1320, 0.2700, 0.0],
        [24, 25, 0.1885, 0.3292, 0.0],
        [25, 26, 0.2544, 0.3800, 0.0],
        [25, 27, 0.1093, 0.2087, 0.0],
        [28, 27, 0.0,    0.3960, 0.0],     # transformer
        [27, 29, 0.2198, 0.4153, 0.0],
        [27, 30, 0.3202, 0.6027, 0.0],
        [29, 30, 0.2399, 0.4533, 0.0],
        [8,  28, 0.0636, 0.2000, 0.0428],
        [6,  28, 0.0169, 0.0599, 0.0130],
    ])


def make_bounds():
    """Uniform shunt bounds, one per flex bus."""
    return [SHUNT_Q_BOUNDS] * len(FLEX_BUS_NUMS)


# ---------- gradient unit test ----------

def fd_central_gradient(bus, lines, flex_idx_list, u_vec, h=1e-5):
    n = len(u_vec)
    grad = np.zeros(n)
    u = np.asarray(u_vec, dtype=float)
    for i in range(n):
        up = u.copy(); up[i] += h
        um = u.copy(); um[i] -= h
        F_p = adj.run_loss(bus, lines, flex_idx_list, up.tolist())
        F_m = adj.run_loss(bus, lines, flex_idx_list, um.tolist())
        grad[i] = (F_p - F_m) / (2 * h)
    return grad


def gradient_unit_test(bus, lines, flex_idx_list, probes, abs_tol):
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
        results.append((name, max_err, passed))
    return all_passed, results


# ---------- baseline ----------

def baseline_summary(bus, lines):
    """PF at the uncompensated baseline (Q_GEN = 0 at every flex bus)."""
    V, theta, _, _, _ = pf.newton_raphson(bus, lines, verbose=False,
                                          max_iter=50)
    loss = sum(l["P_loss"] for l in pf.line_losses(V, theta, lines))
    return float(loss), float(V.min()), float(V.max())


# ---------- output ----------

def print_header(bus, flex_bus_nums):
    p_total_load = float(np.sum(bus[:, pf.P_LOAD]))
    print("=" * 78)
    print(f"Adjoint OPF on IEEE 30-bus, heavy-peak ({HEAVY_PEAK_FACTOR}×) loading")
    print("=" * 78)
    print(f"System:      30 buses, 41 branches (4 transformers, treated as π)")
    print(f"Voltages:    132 kV transmission + 33 kV sub-transmission")
    print(f"Loading:     {HEAVY_PEAK_FACTOR}× default (heavy peak)")
    print(f"Total load:  {p_total_load:.3f} pu = "
          f"{p_total_load * BASE_MVA:.1f} MW")
    print(f"Flex vars:   d = {len(flex_bus_nums)} "
          f"(10 shunt-Q at heavy-load buses; gens at PV)")
    print(f"Tolerance:   adjoint gtol = {ADJ_TOL:.0e}, "
          f"grad abs tol = {GRAD_TOL:.0e}")


def print_baseline(loss, v_min, v_max, total_p):
    print("\n" + "-" * 78)
    print("[0] Baseline (uncompensated reference: Q_i = 0 ∀ flex)")
    print(f"  losses:              {loss:.5f} pu  =  {loss * BASE_MVA:.4f} MW")
    print(f"  losses / total load: {100 * loss / total_p:.2f} %")
    print(f"  voltage range:       [{v_min:.4f}, {v_max:.4f}] pu")


def print_method_block(name, q_star, loss_star, n_evals, wall_s):
    print("\n" + "-" * 78)
    print(f"[1] {name}")
    print(f"  P_loss:    {loss_star:.5f} pu  ({loss_star * BASE_MVA:.4f} MW)")
    print(f"  PF calls:  {n_evals}")
    print(f"  wall:      {wall_s:.2f} s")
    q_arr = np.asarray(q_star)
    print(f"  Q* range:  [{q_arr.min():+.4f}, {q_arr.max():+.4f}] pu  "
          f"(mean = {q_arr.mean():+.4f})")


def print_gradient_test(results, abs_tol, all_passed):
    print("\n" + "-" * 78)
    print("[2] Gradient unit test  (adjoint vs central FD, h = 1e-5)")
    print(f"    abs_tol = {abs_tol:.0e}")
    for name, max_err, passed in results:
        verdict = "PASS" if passed else "FAIL"
        print(f"  {name:<28}  max |∇F_adj − ∇F_fd| = {max_err:.2e}  [{verdict}]")
    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")


def print_improvement(loss_baseline, loss_optimum, total_p):
    delta = loss_baseline - loss_optimum
    rel_loss = 100 * delta / loss_baseline if loss_baseline > 0 else 0.0
    rel_load = 100 * delta / total_p if total_p > 0 else 0.0
    print("\n" + "-" * 78)
    print("[3] Improvement vs baseline")
    print(f"  baseline P_loss:  {loss_baseline:.5f} pu  "
          f"({loss_baseline * BASE_MVA:.4f} MW)")
    print(f"  optimum P_loss:   {loss_optimum:.5f} pu  "
          f"({loss_optimum * BASE_MVA:.4f} MW)")
    print(f"  reduction:        {delta:.5f} pu  ({delta * BASE_MVA:.4f} MW)")
    print(f"  vs baseline loss: {rel_loss:.2f} %")
    print(f"  vs total load:    {rel_load:.4f} %")
    print("=" * 78)


def main():
    bus = bus_data()
    lines = line_data()
    flex_idx_list = [adj.find_bus_idx(bus, n) for n in FLEX_BUS_NUMS]
    bounds = make_bounds()
    p_total = float(np.sum(bus[:, pf.P_LOAD]))

    print_header(bus, FLEX_BUS_NUMS)

    base_loss, v_min, v_max = baseline_summary(bus, lines)
    print_baseline(base_loss, v_min, v_max, p_total)

    adj_res = adj.run_adjoint_method(bus, lines, flex_idx_list, bounds,
                                     tol=ADJ_TOL)
    print_method_block("Adjoint OPF (L-BFGS-B + analytical ∇F)", *adj_res)

    probes = [
        ("baseline (Q_i = 0)", np.zeros(len(flex_idx_list))),
        ("midpoint of bounds", np.array([0.5 * (lo + hi) for lo, hi in bounds])),
        ("optimum (adjoint u*)", np.array(adj_res[0])),
    ]
    grad_passed, grad_results = gradient_unit_test(
        bus, lines, flex_idx_list, probes, GRAD_TOL)
    print_gradient_test(grad_results, GRAD_TOL, grad_passed)

    print_improvement(base_loss, adj_res[1], p_total)


if __name__ == "__main__":
    main()
