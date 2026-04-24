"""Test Newton-Raphson on a synthetic 500-bus system.

Generates a large test case to evaluate convergence and performance.
Reports: iteration count, convergence time, Jacobian time, solve time.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
import newton_raphson_pf as pf


def generate_500bus_system():
    """Create synthetic 500-bus system with realistic structure."""
    n = 500
    INF = 9999.0

    # Bus data: mostly PQ loads with some generators
    bus = np.zeros((n, 10))
    bus[:, 0] = np.arange(1, n + 1)  # Bus numbers
    bus[:, 1] = pf.PQ  # Default: PQ
    bus[:, 2] = 1.0  # V = 1.0 pu

    # Bus 1: slack
    bus[0, 1] = pf.SLACK
    bus[0, 2] = 1.05

    # Buses 2-26: PV generators (distributed throughout)
    pv_buses = np.arange(1, 26)
    bus[pv_buses, 1] = pf.PV
    bus[pv_buses, 2] = 1.04

    # Buses 27-60: SVC buses with Q limits
    svc_buses = np.arange(26, 60)
    bus[svc_buses, 1] = pf.SVC
    bus[svc_buses, 8] = -0.3  # Q_min
    bus[svc_buses, 9] = 0.3   # Q_max

    # Remaining buses: PQ loads
    np.random.seed(42)
    load_factor = 0.5
    p_loads = np.random.uniform(0.05, 0.35, n) * load_factor
    q_loads = p_loads * 0.15

    # Slack has no load
    p_loads[0] = 0.0
    q_loads[0] = 0.0

    # Generators have no load
    p_loads[pv_buses] = 0.0
    q_loads[pv_buses] = 0.0
    p_loads[svc_buses] = 0.0
    q_loads[svc_buses] = 0.0

    bus[:, 6] = p_loads   # Pload
    bus[:, 7] = q_loads   # Qload

    # Set generator powers to match load
    total_load_p = np.sum(p_loads)
    total_load_q = np.sum(q_loads)
    gen_p_per_unit = total_load_p * 1.10 / len(pv_buses)  # 10% margin
    gen_q_per_unit = total_load_q * 0.5 / len(pv_buses)

    bus[pv_buses, 4] = gen_p_per_unit   # Pgen
    bus[pv_buses, 5] = gen_q_per_unit   # Qgen

    # Q limits for regular PQ buses (unconstrained)
    bus[:, 8] = -INF
    bus[:, 9] = INF

    return bus.astype(float)


def generate_500bus_lines(n=500):
    """Create synthetic network lines (radial tree structure)."""
    lines = []

    # Simple radial tree: each bus connects to parent
    # Bus 1 is root (slack)
    # Buses 2-5 connect to bus 1
    # Buses 6-25 connect to buses 2-5, etc.
    parent_map = [0] * (n + 1)
    parent_map[1] = 1  # Bus 1 is root

    bus_idx = 2
    for parent in range(1, n):
        if bus_idx > n:
            break
        # Each bus gets ~4 children
        for _ in range(4):
            if bus_idx > n:
                break
            parent_map[bus_idx] = parent
            bus_idx += 1

    # Create lines from each bus to its parent
    for bus in range(2, n + 1):
        parent = parent_map[bus]
        r = 0.01 + 0.005 * np.random.rand()
        x = 0.03 + 0.01 * np.random.rand()
        b = 0.01 + 0.005 * np.random.rand()
        lines.append([parent, bus, r, x, b])

    # Add ~5% cross-ties for weak meshing
    np.random.seed(42)
    for _ in range(int(0.02 * n)):
        bus_f = np.random.randint(1, n + 1)
        bus_t = np.random.randint(bus_f + 1, n + 1)
        if bus_f != bus_t:
            r = 0.02 + 0.01 * np.random.rand()
            x = 0.06 + 0.02 * np.random.rand()
            b = 0.02 + 0.01 * np.random.rand()
            lines.append([bus_f, bus_t, r, x, b])

    return np.array(lines)


def newton_raphson_timed(bus, lines, tol=1e-5, max_iter=100):
    """Wrapper to time the NR solver."""
    t_start = time.time()
    V, theta, Y, Q, iters = pf.newton_raphson(bus, lines, tol=tol, max_iter=max_iter, verbose=False)
    t_total = time.time() - t_start
    return V, theta, Y, Q, iters, t_total


def main():
    print("=" * 70)
    print("500-Bus System - Newton-Raphson Power Flow Test")
    print("=" * 70)

    print("\n[1] Generating 500-bus synthetic system...")
    t0 = time.time()
    bus = generate_500bus_system()
    lines = generate_500bus_lines()
    gen_time = time.time() - t0

    n_slack = np.sum(bus[:, 1] == pf.SLACK)
    n_pv = np.sum(bus[:, 1] == pf.PV)
    n_pq = np.sum(bus[:, 1] == pf.PQ)
    n_svc = np.sum(bus[:, 1] == pf.SVC)
    n_lines = len(lines)

    print(f"    Buses: {len(bus)} total")
    print(f"      Slack:  {n_slack}")
    print(f"      PV:     {n_pv}")
    print(f"      PQ:     {n_pq}")
    print(f"      SVC:    {n_svc}")
    print(f"    Lines: {n_lines}")
    print(f"    Generation time: {gen_time*1000:.2f} ms")

    print("\n[2] Running Newton-Raphson solver...")
    try:
        V, theta, _, _, iters, t_total = newton_raphson_timed(bus, lines, tol=1e-5, max_iter=100)

        print(f"\n    CONVERGED in {iters} iterations")
        print(f"\n[3] Timing:")
        print(f"    Total time:    {t_total*1000:8.2f} ms")
        print(f"    Per iteration: {(t_total)/iters*1000:8.2f} ms")

        # Voltage statistics
        V_min, V_max = np.min(V), np.max(V)
        V_mean = np.mean(V)
        theta_min, theta_max = np.min(np.rad2deg(theta)), np.max(np.rad2deg(theta))

        print(f"\n[4] Solution Statistics:")
        print(f"    Voltage (pu):    min={V_min:.4f}, mean={V_mean:.4f}, max={V_max:.4f}")
        print(f"    Angle (deg):     min={theta_min:7.2f}, max={theta_max:7.2f}")

        # Transmission losses
        losses = pf.line_losses(V, theta, lines)
        total_P_loss = sum(l['P_loss'] for l in losses)
        total_Q_loss = sum(l['Q_loss'] for l in losses)

        print(f"\n[5] Transmission Losses:")
        print(f"    Total Real Power Loss:  {total_P_loss:8.4f} pu")
        print(f"    Total Reactive Loss:    {total_Q_loss:8.4f} pu")
        print(f"    Loss as % of generation: {total_P_loss/np.sum(bus[:, 4])*100:6.2f}%")

        print("\n" + "=" * 70)
        print("TEST PASSED")
        print("=" * 70)

    except RuntimeError as e:
        print(f"\n    ERROR: {e}")
        print("\n" + "=" * 70)
        print("TEST FAILED")
        print("=" * 70)


if __name__ == "__main__":
    main()
