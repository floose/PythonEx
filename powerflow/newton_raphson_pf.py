"""Newton-Raphson power flow on a 5-bus system (Stagg & El-Abiad).

All quantities are in per-unit on a 100 MVA base.
Bus types: 3 = slack, 2 = PV, 1 = PQ.
"""

import numpy as np
from scipy.sparse import coo_matrix, bmat
from scipy.sparse.linalg import spsolve

#MACROS for bus types
SLACK, PV, PQ, SVC = 3, 2, 1, 4
#MACROS for bus data columns
BUS_NUM, BUS_TYPE, V_INIT, THETA_INIT, P_GEN, Q_GEN, P_LOAD, Q_LOAD, Q_MIN, Q_MAX = range(10)   
#Macros for line data columns
FROM_BUS, TO_BUS, R, X, B = range(5)    

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
        [7, PQ,    1.0,   0.0, 0.0, 0.0, 0.0,  0.20, INF,  INF],
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
        [7, 5, 0.08, 0.24, 0.05],
    ])


# ---------- admittance matrix ----------

def _build_ybus(n_bus, lines):
    rows, cols, data = [], [], []
    for f, t, r, x, b in lines:
        i, j = int(f) - 1, int(t) - 1
        y = 1.0 / complex(r, x)
        half_b = 1j * b / 2
        rows += [i, j, i, j]
        cols += [i, j, j, i]
        data += [y + half_b, y + half_b, -y, -y]
    Y = coo_matrix((data, (rows, cols)),
                   shape=(n_bus, n_bus), dtype=complex).tocsr()
    return Y


# ---------- bus impedance matrix ----------

def build_zbus(Y):
    """Compute Z-bus = Y^-1 as a dense complex matrix.

    Y can be sparse (csr/csc) or a dense ndarray.
    """
    Y_dense = Y.toarray() if hasattr(Y, "toarray") else Y
    return np.linalg.inv(Y_dense)


def _print_zbus_full(Z, bus_nums):
    """Print R-bus and X-bus (real and imaginary parts of Z) side by side."""
    n = len(Z)
    cell = 9
    header = " " * 7 + "".join(f"Bus {b:>2d}".rjust(cell) for b in bus_nums)
    for title, M in (("R-bus = Re(Z)  [pu]", Z.real),
                     ("X-bus = Im(Z)  [pu]", Z.imag)):
        print(f"\n{title}")
        print(header)
        print(" " * 7 + "-" * (cell * n))
        for i in range(n):
            row = f"Bus {bus_nums[i]:>2d}:"
            for j in range(n):
                row += f"{M[i, j]:>+9.4f}"
            print(row)


def _print_zbus_diagonal(Z, bus_nums):
    """Print only the Thévenin impedance per bus."""
    print(f"\nDiagonal only (matrix too large to display in full)")
    print(f"{'Bus':>4} {'R (pu)':>10} {'X (pu)':>10} {'|Z| (pu)':>10}")
    print("-" * 38)
    for i in range(len(Z)):
        z = Z[i, i]
        print(f"{bus_nums[i]:>4} {z.real:>10.5f} {z.imag:>10.5f} {abs(z):>10.5f}")


def print_zbus(Z, bus, max_full=15):
    """Print Z-bus. Full R/X matrices for small systems, diagonal otherwise."""
    bus_nums = bus[:, BUS_NUM].astype(int)
    print("\n" + "=" * 70)
    print("Z-bus matrix (Z = Y^-1, in per unit)")
    print("=" * 70)
    if len(Z) <= max_full:
        _print_zbus_full(Z, bus_nums)
    else:
        _print_zbus_diagonal(Z, bus_nums)
    print("=" * 70)


# ---------- power injections ----------

def _power_injections(V, theta, Y):
    n = len(V)
    Ycoo = Y.tocoo()
    rows, cols = Ycoo.row, Ycoo.col
    G_nz = Ycoo.data.real
    B_nz = Ycoo.data.imag

    dth = theta[rows] - theta[cols]
    c = np.cos(dth)
    s = np.sin(dth)
    Vc = V[cols]

    P_terms = Vc * (G_nz * c + B_nz * s)
    Q_terms = Vc * (G_nz * s - B_nz * c)

    P = V * np.bincount(rows, weights=P_terms, minlength=n)
    Q = V * np.bincount(rows, weights=Q_terms, minlength=n)
    return P, Q


def _mismatches(P_calc, Q_calc, P_spec, Q_spec, pq_idx, non_slack_idx, Q_limited):
    dP = (P_spec - P_calc)[non_slack_idx]
    dQ = (Q_spec - Q_calc)[pq_idx]
    # For SVC buses in PQ set that are limited, zero out their mismatch
    for i, pq_bus_idx in enumerate(pq_idx):
        if pq_bus_idx in Q_limited:
            dQ[i] = 0.0
    return np.concatenate([dP, dQ])


# ---------- Jacobian ----------

def _jacobian(V, theta, Y, P, Q, pq_idx, non_slack_idx):
    n = len(V)
    Ycoo = Y.tocoo()
    rows, cols = Ycoo.row, Ycoo.col
    G_nz = Ycoo.data.real
    B_nz = Ycoo.data.imag

    dth = theta[rows] - theta[cols]
    c = np.cos(dth)
    s = np.sin(dth)
    Vr = V[rows]
    VrVc = Vr * V[cols]

    H_data = VrVc * (G_nz * s - B_nz * c)
    N_data = Vr * (G_nz * c + B_nz * s)
    J_data = -VrVc * (G_nz * c + B_nz * s)
    L_data = Vr * (G_nz * s - B_nz * c)

    # Overwrite diagonal entries (Y always has a non-zero diagonal in this model)
    is_diag = rows == cols
    d_bus = rows[is_diag]
    V_d = V[d_bus]
    G_d = G_nz[is_diag]
    B_d = B_nz[is_diag]
    H_data[is_diag] = -Q[d_bus] - B_d * V_d ** 2
    N_data[is_diag] = P[d_bus] / V_d + G_d * V_d
    J_data[is_diag] = P[d_bus] - G_d * V_d ** 2
    L_data[is_diag] = Q[d_bus] / V_d - B_d * V_d

    shape = (n, n)
    H = coo_matrix((H_data, (rows, cols)), shape=shape).tocsr()
    N = coo_matrix((N_data, (rows, cols)), shape=shape).tocsr()
    J = coo_matrix((J_data, (rows, cols)), shape=shape).tocsr()
    L = coo_matrix((L_data, (rows, cols)), shape=shape).tocsr()

    Hns = H[non_slack_idx, :][:, non_slack_idx]
    Nns = N[non_slack_idx, :][:, pq_idx]
    Jns = J[pq_idx, :][:, non_slack_idx]
    Lns = L[pq_idx, :][:, pq_idx]

    return bmat([[Hns, Nns], [Jns, Lns]], format="csc")


# ---------- Q limit checking ----------

def _check_q_limits(Q, bus):
    """Clamp Q for SVC buses and return set of limited bus indices."""
    Q_limited = set()
    types = bus[:, BUS_TYPE].astype(int)
    Q_min = bus[:, Q_MIN]
    Q_max = bus[:, Q_MAX]
    for i in range(len(Q)):
        if types[i] == SVC:
            if Q[i] > Q_max[i]:
                Q[i] = Q_max[i]
                Q_limited.add(i)
            elif Q[i] < Q_min[i]:
                Q[i] = Q_min[i]
                Q_limited.add(i)
    return Q_limited




def _update_state(V, theta, dx, pq_idx, non_slack_idx):
    n_ns = len(non_slack_idx)
    dtheta = dx[:n_ns]
    dV = dx[n_ns:]
    theta[non_slack_idx] += dtheta
    V[pq_idx] += dV
    return V, theta


# ---------- solver ----------

def newton_raphson(bus, lines, tol=1e-6, max_iter=20, verbose=True):
    n = bus.shape[0]
    Y = _build_ybus(n, lines)

    types = bus[:, BUS_TYPE].astype(int)
    V     = bus[:, V_INIT].copy()
    theta = np.deg2rad(bus[:, THETA_INIT].copy())
    P_spec = (bus[:, P_GEN] - bus[:, P_LOAD])  # Pgen - Pload
    Q_spec = (bus[:, Q_GEN] - bus[:, Q_LOAD])  # Qgen - Qload

    pq_idx        = np.where((types == PQ) | (types == SVC))[0]
    non_slack_idx = np.where(types != SLACK)[0]
    Q_limited     = set()

    if verbose:
        print("PQ/SVC bus indices:", pq_idx)
        print("SVC bus Q limits:")
        for i in pq_idx:
            if types[i] == SVC:
                print(f"  Bus {int(bus[i,BUS_NUM])} (idx {i}): Q_min={bus[i,Q_MIN]:.2f}, Q_max={bus[i,Q_MAX]:.2f}")

    for it in range(1, max_iter + 1):
        P, Q = _power_injections(V, theta, Y)

        # Check and clamp Q limits for SVC buses
        Q_limited = _check_q_limits(Q, bus)

        # Debug: show SVC Q values
        if it == 1 and verbose and len(pq_idx) < 50:  # Only on first iteration and for small systems
            print("\nFirst iteration Q values for SVCs:")
            for i in pq_idx:
                if types[i] == SVC:
                    print(f"  Bus {int(bus[i,BUS_NUM])} (idx {i}): Q_calc={Q[i]:7.4f}, limits=[{bus[i,Q_MIN]:6.2f}, {bus[i,Q_MAX]:6.2f}]")

        f = _mismatches(P, Q, P_spec, Q_spec, pq_idx, non_slack_idx, Q_limited)
        err = np.max(np.abs(f))
        if verbose:
            print(f"iter {it:2d}   max mismatch = {err:.3e}", end="")
            if Q_limited:
                print(f"   Q limited at buses: {sorted(Q_limited)}", end="")
            print()

        if err < tol:
            return V, theta, Y, Q, it
        Jac = _jacobian(V, theta, Y, P, Q, pq_idx, non_slack_idx)
        dx = spsolve(Jac, f)
        V, theta = _update_state(V, theta, dx, pq_idx, non_slack_idx)

    raise RuntimeError(f"did not converge in {max_iter} iterations")


# ---------- transmission losses ----------

def line_losses(V, theta, lines):
    """Calculate real and reactive power losses on each transmission line."""
    losses = []
    for f, t, r, x, b in lines:
        i, j = int(f) - 1, int(t) - 1
        y = 1.0 / complex(r, x)

        # Complex voltages
        Vi = V[i] * np.exp(1j * theta[i])
        Vj = V[j] * np.exp(1j * theta[j])

        # Current from i to j
        I_ij = y * (Vi - Vj)
        # Power from i to j
        S_ij = Vi * np.conj(I_ij)

        # Current from j to i
        I_ji = y * (Vj - Vi)
        # Power from j to i
        S_ji = Vj * np.conj(I_ji)

        # Loss = power out of one end + power out of other end
        P_loss = S_ij.real + S_ji.real
        Q_loss = S_ij.imag + S_ji.imag
        S_loss = np.sqrt(P_loss**2 + Q_loss**2)

        losses.append({
            'from': int(f),
            'to': int(t),
            'P_loss': P_loss,
            'Q_loss': Q_loss,
            'S_loss': S_loss,
            'P_ij': S_ij.real,
            'P_ji': S_ji.real,
        })

    return losses


def print_losses(V, theta, lines, bus=None):
    """Print total and per-line losses sorted by real power loss (highest first).

    If bus data is provided, also shows loss as percentage of generation.
    """
    losses = line_losses(V, theta, lines)

    # Sort by real power loss (descending)
    losses_sorted = sorted(losses, key=lambda x: x['P_loss'], reverse=True)

    total_P_loss = sum(l['P_loss'] for l in losses)
    total_Q_loss = sum(l['Q_loss'] for l in losses)
    total_S_loss = np.sqrt(total_P_loss**2 + total_Q_loss**2)

    print("\n" + "=" * 90)
    print("Transmission Losses (sorted by real power loss)")
    print("=" * 90)
    print(f"\nTotal Real Power Loss:     {total_P_loss:8.4f} pu")
    print(f"Total Reactive Loss:       {total_Q_loss:8.4f} pu")
    print(f"Total Apparent Loss:       {total_S_loss:8.4f} pu")

    if bus is not None:
        # Total generation recovered from conservation: gen = load + losses
        total_gen = np.sum(bus[:, P_LOAD]) + total_P_loss
        if total_gen > 0:
            loss_pct = total_P_loss / total_gen * 100
            print(f"Loss as % of Generation:   {loss_pct:8.2f}%")

    print(f"\n{'Line':>12} {'P Loss':>10} {'Q Loss':>10} {'S Loss':>10} {'P i→j':>10} {'P j→i':>10}")
    print("-" * 90)

    for loss in losses_sorted:
        line_label = f"{loss['from']:3d}→{loss['to']:3d}"
        print(f"{line_label:>12} {loss['P_loss']:10.4f} {loss['Q_loss']:10.4f} {loss['S_loss']:10.4f} "
              f"{loss['P_ij']:10.4f} {loss['P_ji']:10.4f}")

    print("=" * 90)




def print_results(V, theta, Y, Q, bus):
    P, _ = _power_injections(V, theta, Y)
    types = bus[:, BUS_TYPE].astype(int)
    Pload = bus[:, P_LOAD]
    Qload = bus[:, Q_LOAD]
    Pgen = P + Pload
    Qgen = Q + Qload
    # keep specified gens where not solved
    Pgen[types != SLACK] = bus[types != SLACK, P_GEN]
    print("\nBus results")
    print(f"{'bus':>4} {'type':>6} {'|V| pu':>9} {'angle deg':>11} {'Pgen':>8} {'Qgen':>8} {'Pcon':>8} {'Qcon':>8} {'Q range':>12}")
    for i in range(len(V)):
        type_str = {SLACK: "slack", PV: "PV", PQ: "PQ", SVC: "SVC"}[types[i]]
        qrange = f"[{bus[i,Q_MIN]:5.2f},{bus[i,Q_MAX]:5.2f}]"
        print(f"{int(bus[i,BUS_NUM]):>4} {type_str:>6} {V[i]:>9.4f} {np.rad2deg(theta[i]):>11.4f} "
              f"{Pgen[i]:>8.4f} {Qgen[i]:>8.4f} {Pload[i]:>8.4f} {Qload[i]:>8.4f} {qrange:>12}")


if __name__ == "__main__":
    bus = bus_data()
    lines = line_data()
    V, theta, Y, Q, iters = newton_raphson(bus, lines)
    print(f"\nconverged in {iters} iterations")
    print_results(V, theta, Y, Q, bus)
    print_losses(V, theta, lines, bus)
    Z = build_zbus(Y)
    print_zbus(Z, bus)
