"""Newton-Raphson power flow on a 5-bus system (Stagg & El-Abiad).

All quantities are in per-unit on a 100 MVA base.
Bus types: 3 = slack, 2 = PV, 1 = PQ.
"""

import numpy as np

SLACK, PV, PQ, SVC = 3, 2, 1, 4


# ---------- system data ----------

def bus_data():
    # bus, type, V, theta_deg, Pgen, Qgen, Pload, Qload, Q_min, Q_max
    # Q limits: ±9999 for unconstrained, realistic range for SVC
    INF = 9999.0
    return np.array([
        [1, SLACK, 1.06,  0.0, 0.0, 0.0, 0.0,  0.0,  -INF,  INF],
        [2, PV,    1.045, 0.0, 0.4, 0.0, 0.2,  0.1,  -INF,  INF],
        [3, PQ,    1.0,   0.0, 0.0, 0.0, 0.45, 0.15, -INF,  INF],
        [4, SVC,   1.03,  0.0, 0.0, 0.0, 0.0,  0.0,  -0.3,  0.3],
        [5, PQ,    1.0,   0.0, 0.0, 0.0, 0.6,  0.10, -INF,  INF],
        [6, PQ,    1.0,   0.0, 0.0, 0.0, 0.6,  0.10, -INF,  INF],
    ])


def line_data():
    # from, to, R, X, B_total (line charging)
    return np.array([
        [1, 2, 0.02, 0.06, 0.06],
        [1, 3, 0.08, 0.24, 0.05],
        [2, 3, 0.06, 0.18, 0.04],
        [2, 4, 0.06, 0.18, 0.04],
        [2, 5, 0.04, 0.12, 0.03],
        [3, 4, 0.01, 0.03, 0.02],
        [4, 5, 0.08, 0.24, 0.05],
        [6, 5, 0.08, 0.24, 0.05],
    ])


# ---------- admittance matrix ----------

def build_ybus(n_bus, lines):
    Y = np.zeros((n_bus, n_bus), dtype=complex)
    for f, t, r, x, b in lines:
        i, j = int(f) - 1, int(t) - 1
        y = 1.0 / complex(r, x)
        Y[i, i] += y + 1j * b / 2
        Y[j, j] += y + 1j * b / 2
        Y[i, j] -= y
        Y[j, i] -= y
    return Y


# ---------- power injections ----------

def power_injections(V, theta, Y):
    G, B = Y.real, Y.imag
    n = len(V)
    P = np.zeros(n)
    Q = np.zeros(n)
    for i in range(n):
        for k in range(n):
            dth = theta[i] - theta[k]
            P[i] += V[i] * V[k] * (G[i, k] * np.cos(dth) + B[i, k] * np.sin(dth))
            Q[i] += V[i] * V[k] * (G[i, k] * np.sin(dth) - B[i, k] * np.cos(dth))
    return P, Q


def mismatches(P_calc, Q_calc, P_spec, Q_spec, pq_idx, non_slack_idx, bus, Q_limited):
    dP = (P_spec - P_calc)[non_slack_idx]
    dQ = (Q_spec - Q_calc)[pq_idx]
    # For SVC buses in PQ set that are limited, zero out their mismatch
    for i, pq_bus_idx in enumerate(pq_idx):
        if pq_bus_idx in Q_limited:
            dQ[i] = 0.0
    return np.concatenate([dP, dQ])


# ---------- Jacobian ----------

def jacobian(V, theta, Y, P, Q, pq_idx, non_slack_idx):
    G, B = Y.real, Y.imag
    n = len(V)

    H = np.zeros((n, n))  # dP/dtheta
    N = np.zeros((n, n))  # dP/dV
    J = np.zeros((n, n))  # dQ/dtheta
    L = np.zeros((n, n))  # dQ/dV

    for i in range(n):
        for k in range(n):
            if i == k:
                H[i, i] = -Q[i] - B[i, i] * V[i] ** 2
                N[i, i] = P[i] / V[i] + G[i, i] * V[i]
                J[i, i] = P[i] - G[i, i] * V[i] ** 2
                L[i, i] = Q[i] / V[i] - B[i, i] * V[i]
            else:
                dth = theta[i] - theta[k]
                s, c = np.sin(dth), np.cos(dth)
                H[i, k] =  V[i] * V[k] * (G[i, k] * s - B[i, k] * c)
                N[i, k] =  V[i] * (G[i, k] * c + B[i, k] * s)
                J[i, k] = -V[i] * V[k] * (G[i, k] * c + B[i, k] * s)
                L[i, k] =  V[i] * (G[i, k] * s - B[i, k] * c)

    Hns = H[np.ix_(non_slack_idx, non_slack_idx)]
    Nns = N[np.ix_(non_slack_idx, pq_idx)]
    Jns = J[np.ix_(pq_idx,        non_slack_idx)]
    Lns = L[np.ix_(pq_idx,        pq_idx)]

    top    = np.hstack([Hns, Nns])
    bottom = np.hstack([Jns, Lns])
    return np.vstack([top, bottom])


# ---------- Q limit checking ----------

def check_q_limits(Q, bus):
    """Clamp Q for SVC buses and return set of limited bus indices."""
    Q_limited = set()
    types = bus[:, 1].astype(int)
    Q_min = bus[:, 8]
    Q_max = bus[:, 9]
    for i in range(len(Q)):
        if types[i] == SVC:
            if Q[i] > Q_max[i]:
                Q[i] = Q_max[i]
                Q_limited.add(i)
            elif Q[i] < Q_min[i]:
                Q[i] = Q_min[i]
                Q_limited.add(i)
    return Q_limited




def update_state(V, theta, dx, pq_idx, non_slack_idx):
    n_ns = len(non_slack_idx)
    dtheta = dx[:n_ns]
    dV = dx[n_ns:]
    theta[non_slack_idx] += dtheta
    V[pq_idx] += dV
    return V, theta


# ---------- solver ----------

def newton_raphson(bus, lines, tol=1e-6, max_iter=20):
    n = bus.shape[0]
    Y = build_ybus(n, lines)

    types = bus[:, 1].astype(int)
    V     = bus[:, 2].copy()
    theta = np.deg2rad(bus[:, 3].copy())
    P_spec = (bus[:, 4] - bus[:, 6])  # Pgen - Pload
    Q_spec = (bus[:, 5] - bus[:, 7])  # Qgen - Qload

    pq_idx        = np.where((types == PQ) | (types == SVC))[0]
    non_slack_idx = np.where(types != SLACK)[0]
    Q_limited     = set()

    for it in range(1, max_iter + 1):
        P, Q = power_injections(V, theta, Y)

        # Check and clamp Q limits for SVC buses
        Q_limited = check_q_limits(Q, bus)

        f = mismatches(P, Q, P_spec, Q_spec, pq_idx, non_slack_idx, bus, Q_limited)
        err = np.max(np.abs(f))
        print(f"iter {it:2d}   max mismatch = {err:.3e}", end="")
        if Q_limited:
            print(f"   Q limited at buses: {sorted(Q_limited)}", end="")
        print()

        if err < tol:
            return V, theta, Y, Q, it
        Jac = jacobian(V, theta, Y, P, Q, pq_idx, non_slack_idx)
        dx = np.linalg.solve(Jac, f)
        V, theta = update_state(V, theta, dx, pq_idx, non_slack_idx)

    raise RuntimeError(f"did not converge in {max_iter} iterations")


# ---------- reporting ----------

def slack_and_pv_injections(V, theta, Y, bus):
    P, Q = power_injections(V, theta, Y)
    types = bus[:, 1].astype(int)
    Pload = bus[:, 6]
    Qload = bus[:, 7]
    Pgen = P + Pload
    Qgen = Q + Qload
    # keep specified gens where not solved
    Pgen[types != SLACK] = bus[types != SLACK, 4]
    return Pgen, Qgen


def print_results(V, theta, Y, Q, bus):
    P, _ = power_injections(V, theta, Y)
    types = bus[:, 1].astype(int)
    Pload = bus[:, 6]
    Qload = bus[:, 7]
    Pgen = P + Pload
    Qgen = Q + Qload
    # keep specified gens where not solved
    Pgen[types != SLACK] = bus[types != SLACK, 4]
    print("\nBus results")
    print(f"{'bus':>4} {'type':>6} {'|V| pu':>9} {'angle deg':>11} {'Pgen':>8} {'Qgen':>8} {'Q range':>12}")
    for i in range(len(V)):
        type_str = {SLACK: "slack", PV: "PV", PQ: "PQ", SVC: "SVC"}[types[i]]
        qrange = f"[{bus[i,8]:5.2f},{bus[i,9]:5.2f}]"
        print(f"{int(bus[i,0]):>4} {type_str:>6} {V[i]:>9.4f} {np.rad2deg(theta[i]):>11.4f} "
              f"{Pgen[i]:>8.4f} {Qgen[i]:>8.4f} {qrange:>12}")


if __name__ == "__main__":
    bus = bus_data()
    lines = line_data()
    V, theta, Y, Q, iters = newton_raphson(bus, lines)
    print(f"\nconverged in {iters} iterations")
    print_results(V, theta, Y, Q, bus)
