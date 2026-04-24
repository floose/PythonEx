"""Validate the in-house Newton-Raphson solver against pandapower.

Builds a 6-bus meshed system, solves it with both engines, and compares
bus voltage magnitudes and angles.
"""

import sys
from pathlib import Path
import numpy as np
import pandapower as pp

sys.path.insert(0, str(Path(__file__).parent.parent))
import newton_raphson_pf as pf


SN_MVA = 100.0
VN_KV = 110.0
F_HZ = 50.0
Z_BASE = VN_KV ** 2 / SN_MVA


def bus_data():
    # bus, type, V, theta_deg, Pgen, Qgen, Pload, Qload, Q_min, Q_max
    INF = 9999.0
    return np.array([
        [1, pf.SLACK, 1.06,  0.0, 0.0, 0.0, 0.0,  0.0,  -INF, INF],
        [2, pf.PV,    1.045, 0.0, 0.5, 0.0, 0.0,  0.0,  -INF, INF],
        [3, pf.PV,    1.03,  0.0, 0.6, 0.0, 0.0,  0.0,  -INF, INF],
        [4, pf.PQ,    1.0,   0.0, 0.0, 0.0, 0.7,  0.30, -INF, INF],
        [5, pf.PQ,    1.0,   0.0, 0.0, 0.0, 0.7,  0.20, -INF, INF],
        [6, pf.PQ,    1.0,   0.0, 0.0, 0.0, 0.4,  0.15, -INF, INF],
    ])


def line_data():
    # from, to, R, X, B_total
    return np.array([
        [1, 2, 0.02, 0.06, 0.06],
        [1, 4, 0.08, 0.24, 0.05],
        [2, 3, 0.06, 0.18, 0.04],
        [2, 4, 0.06, 0.18, 0.04],
        [2, 5, 0.04, 0.12, 0.03],
        [3, 5, 0.01, 0.03, 0.02],
        [3, 6, 0.08, 0.24, 0.05],
        [4, 5, 0.08, 0.24, 0.05],
        [5, 6, 0.04, 0.12, 0.03],
    ])


def to_pandapower(bus, lines):
    net = pp.create_empty_network(sn_mva=SN_MVA, f_hz=F_HZ)

    pp_bus = {}
    for i in range(len(bus)):
        num = int(bus[i, pf.BUS_NUM])
        pp_bus[num] = pp.create_bus(net, vn_kv=VN_KV, name=f"Bus {num}")

    for i in range(len(bus)):
        num = int(bus[i, pf.BUS_NUM])
        btype = int(bus[i, pf.BUS_TYPE])
        vm = bus[i, pf.V_INIT]
        pgen = bus[i, pf.P_GEN]
        pload = bus[i, pf.P_LOAD]
        qload = bus[i, pf.Q_LOAD]
        b = pp_bus[num]

        if btype == pf.SLACK:
            pp.create_ext_grid(net, bus=b, vm_pu=vm, va_degree=0.0)
        elif btype == pf.PV:
            pp.create_gen(net, bus=b, p_mw=pgen * SN_MVA, vm_pu=vm)
        elif btype == pf.SVC:
            qmin = bus[i, pf.Q_MIN] * SN_MVA
            qmax = bus[i, pf.Q_MAX] * SN_MVA
            pp.create_gen(net, bus=b, p_mw=0.0, vm_pu=vm,
                          min_q_mvar=qmin, max_q_mvar=qmax)

        if pload != 0.0 or qload != 0.0:
            pp.create_load(net, bus=b, p_mw=pload * SN_MVA, q_mvar=qload * SN_MVA)

    for line in lines:
        f = int(line[pf.FROM_BUS])
        t = int(line[pf.TO_BUS])
        r_ohm = line[pf.R] * Z_BASE
        x_ohm = line[pf.X] * Z_BASE
        b_siemens = line[pf.B] / Z_BASE
        c_nf = b_siemens / (2 * np.pi * F_HZ) * 1e9

        pp.create_line_from_parameters(
            net,
            from_bus=pp_bus[f], to_bus=pp_bus[t],
            length_km=1.0,
            r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm,
            c_nf_per_km=c_nf,
            max_i_ka=10.0,
        )

    return net, pp_bus


def main():
    print("=" * 78)
    print("6-Bus System - In-house NR vs pandapower")
    print("=" * 78)

    bus = bus_data()
    lines = line_data()

    print("\n[1] In-house Newton-Raphson...")
    V, theta, _, _, iters = pf.newton_raphson(bus, lines, verbose=False)
    theta_deg = np.rad2deg(theta)
    print(f"    converged in {iters} iterations")

    print("\n[2] pandapower...")
    net, pp_bus = to_pandapower(bus, lines)
    pp.runpp(net, algorithm="nr", calculate_voltage_angles=True, init="flat")
    print(f"    converged in {net._ppc['iterations']} iterations")

    print("\n[3] Bus-by-bus comparison")
    print(f"  {'bus':>4} {'|V| mine':>10} {'|V| pp':>10} {'ΔV':>11}"
          f"  {'θ mine':>10} {'θ pp':>10} {'Δθ':>11}")
    print("  " + "-" * 76)

    dV_max = 0.0
    dA_max = 0.0
    for i in range(len(bus)):
        num = int(bus[i, pf.BUS_NUM])
        pp_i = pp_bus[num]
        vm_pp = net.res_bus.at[pp_i, "vm_pu"]
        va_pp = net.res_bus.at[pp_i, "va_degree"]
        dV = V[i] - vm_pp
        dA = theta_deg[i] - va_pp
        dV_max = max(dV_max, abs(dV))
        dA_max = max(dA_max, abs(dA))
        print(f"  {num:>4} {V[i]:>10.6f} {vm_pp:>10.6f} {dV:>+11.2e}"
              f"  {theta_deg[i]:>10.4f} {va_pp:>10.4f} {dA:>+11.2e}")

    print("  " + "-" * 76)
    print(f"  max |ΔV| = {dV_max:.2e} pu")
    print(f"  max |Δθ| = {dA_max:.2e} deg")

    tol_V = 1e-4
    tol_A = 1e-3
    ok = dV_max < tol_V and dA_max < tol_A

    print("\n" + "=" * 78)
    print(f"{'MATCH' if ok else 'MISMATCH'}: tol |ΔV| < {tol_V:.0e}, |Δθ| < {tol_A:.0e} deg")
    print("=" * 78)


if __name__ == "__main__":
    main()
