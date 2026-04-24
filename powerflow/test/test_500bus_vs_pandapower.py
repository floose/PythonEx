"""Validate the in-house Newton-Raphson solver against pandapower on the 500-bus system.

Builds the synthetic 500-bus network from test_500bus, solves it with both
engines, compares every bus |V| and θ, and reports timing.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandapower as pp

sys.path.insert(0, str(Path(__file__).parent.parent))
import newton_raphson_pf as pf

sys.path.insert(0, str(Path(__file__).parent))
from test_500bus import generate_500bus_system, generate_500bus_lines
from test_vs_pandapower import to_pandapower


def main():
    print("=" * 78)
    print("500-Bus System - In-house NR vs pandapower")
    print("=" * 78)

    bus = generate_500bus_system()
    lines = generate_500bus_lines()
    print(f"\n[1] System: {len(bus)} buses, {len(lines)} lines")

    print("\n[2] In-house Newton-Raphson...")
    t0 = time.perf_counter()
    V, theta, _, _, iters_mine = pf.newton_raphson(
        bus, lines, tol=1e-5, max_iter=50, verbose=False
    )
    t_mine = time.perf_counter() - t0
    theta_deg = np.rad2deg(theta)
    print(f"    converged in {iters_mine} iterations, {t_mine*1000:.1f} ms")

    print("\n[3] pandapower...")
    t0 = time.perf_counter()
    net, pp_bus = to_pandapower(bus, lines)
    t_conv = time.perf_counter() - t0

    t0 = time.perf_counter()
    pp.runpp(net, algorithm="nr", calculate_voltage_angles=True,
             init="flat", enforce_q_lims=True, tolerance_mva=1e-5,
             max_iteration=50)
    t_pp = time.perf_counter() - t0
    iters_pp = net._ppc["iterations"]
    print(f"    converted in {t_conv*1000:.1f} ms")
    print(f"    converged in {iters_pp} iterations, {t_pp*1000:.1f} ms")

    # Align results
    V_pp = np.zeros(len(bus))
    A_pp = np.zeros(len(bus))
    for i in range(len(bus)):
        num = int(bus[i, pf.BUS_NUM])
        pp_i = pp_bus[num]
        V_pp[i] = net.res_bus.at[pp_i, "vm_pu"]
        A_pp[i] = net.res_bus.at[pp_i, "va_degree"]

    dV = V - V_pp
    dA = theta_deg - A_pp
    absV = np.abs(dV)
    absA = np.abs(dA)

    print("\n[4] Voltage & angle comparison (all 500 buses)")
    print(f"    ΔV (pu):   max={absV.max():.3e}  mean={absV.mean():.3e}  rms={np.sqrt(np.mean(dV**2)):.3e}")
    print(f"    Δθ (deg):  max={absA.max():.3e}  mean={absA.mean():.3e}  rms={np.sqrt(np.mean(dA**2)):.3e}")

    worst_V = np.argsort(-absV)[:5]
    worst_A = np.argsort(-absA)[:5]

    type_name = {pf.SLACK: "SLACK", pf.PV: "PV", pf.PQ: "PQ", pf.SVC: "SVC"}

    print(f"\n    Top 5 |ΔV|:")
    print(f"      {'bus':>4} {'type':>5} {'|V| mine':>10} {'|V| pp':>10} {'ΔV':>11}")
    for idx in worst_V:
        num = int(bus[idx, pf.BUS_NUM])
        tname = type_name[int(bus[idx, pf.BUS_TYPE])]
        print(f"      {num:>4} {tname:>5} {V[idx]:>10.6f} {V_pp[idx]:>10.6f} {dV[idx]:>+11.3e}")

    print(f"\n    Top 5 |Δθ|:")
    print(f"      {'bus':>4} {'type':>5} {'θ mine':>10} {'θ pp':>10} {'Δθ':>11}")
    for idx in worst_A:
        num = int(bus[idx, pf.BUS_NUM])
        tname = type_name[int(bus[idx, pf.BUS_TYPE])]
        print(f"      {num:>4} {tname:>5} {theta_deg[idx]:>10.4f} {A_pp[idx]:>10.4f} {dA[idx]:>+11.3e}")

    print("\n[5] Timing")
    print(f"    in-house solver:       {t_mine*1000:>8.1f} ms "
          f"({iters_mine} iter, {t_mine/iters_mine*1000:.2f} ms/iter)")
    print(f"    pandapower solver:     {t_pp*1000:>8.1f} ms "
          f"({iters_pp} iter, {t_pp/iters_pp*1000:.2f} ms/iter)")
    print(f"    pandapower conversion: {t_conv*1000:>8.1f} ms (one-time)")
    ratio = t_pp / t_mine
    if ratio >= 1:
        print(f"    → in-house is {ratio:.2f}× faster than pandapower")
    else:
        print(f"    → pandapower is {1/ratio:.2f}× faster than in-house")

    tol_V = 1e-4
    tol_A = 1e-2
    ok = absV.max() < tol_V and absA.max() < tol_A
    print("\n" + "=" * 78)
    print(f"{'MATCH' if ok else 'MISMATCH'}: tol |ΔV| < {tol_V:.0e} pu, |Δθ| < {tol_A:.0e} deg")
    print("=" * 78)


if __name__ == "__main__":
    main()
