# Adjoint method for OPF gradients

Computes `∇F(u)` — the gradient of total losses with respect to the
control vector `u` — in **one linear back-solve per evaluation,
regardless of how many controls `d` there are**. This note documents the
math behind `opf_loss_min_adjoint.py`.

## Setup

- `u ∈ ℝᵈ`: controls (Q injections at flex buses).
- `x ∈ ℝⁿˣ`: state (non-slack angles + PQ voltages — the unknowns NR
  solves).
- `g(x, u) = 0`: power-flow residual, driven to zero by NR.
- `f(x, u)`: objective (total real-power losses).
- `J = ∂g/∂x`: the NR Jacobian, factored at the converged point.

By the implicit function theorem, `x(u)` is smoothly defined wherever
`J` is non-singular. The reduced objective is `F(u) = f(x(u), u)`.

## Forward sensitivity (the obvious path)

Differentiate `g(x(u), u) ≡ 0`:

```
dx/du = -J⁻¹ · ∂g/∂u
```

Then by chain rule

```
dF/du = ∂f/∂x · dx/du + ∂f/∂u
```

Building `dx/du` directly requires **`d` linear solves**, one per control.

## Adjoint (reverse-mode)

Rearrange the chain:

```
dF/du = -(∂f/∂x · J⁻¹) · ∂g/∂u + ∂f/∂u
         └─────────────┘
              λᵀ
```

Define the adjoint state `λ` by

```
Jᵀ λ = -(∂f/∂x)ᵀ        (one transpose back-solve)
```

Then

```
∇F(u) = ∂f/∂u + λᵀ · ∂g/∂u
```

**One back-solve, regardless of `d`.** Associativity of matrix multiplication
picks the cheaper contraction order — small dimension on the outside.

## Lagrangian derivation (equivalent, cleaner)

Form `L(x, u, λ) = f + λᵀ g`. Since `g ≡ 0` along `x(u)`,
`F(u) = L(x(u), u, λ)` for any `λ`. Differentiate:

```
dF/du = (∂f/∂x + λᵀ J) · dx/du + ∂f/∂u + λᵀ · ∂g/∂u
```

Choose `λ` to kill the parenthesised term: `Jᵀ λ = -(∂f/∂x)ᵀ`. The adjoint
absorbs all dependence on `dx/du`. Same equation as above.

## Application in the code

The PF residual convention used by `newton_raphson` is

```
g = (P_calc - P_spec,  Q_calc - Q_spec)
```

Therefore:

- `∂g/∂x = J` — the same Jacobian NR factors.
- For `u_k` = Q injection at flex bus `i`: `∂Q_spec_i/∂u_k = +1`, hence
  `∂g_Q_at_i/∂u_k = -1`.
- `dF/du_k = -λ[Q-mismatch row index of bus i]`.

The Q-row of bus `i` lives at `len(non_slack_idx) + position_in(pq_idx)`
in the active state vector. `Jᵀ` is solved with
`splu(J).solve(b, trans='T')`, reusing the LU factor with no extra
factorisation cost.

## Cost summary

| approach            | linear solves per gradient |
|---------------------|---------------------------:|
| finite differences  | `d + 1` PF re-solves       |
| forward sensitivity | `d`                        |
| **adjoint**         | **1**                      |

For our 10-bus problem with `d = 5`, `opf_loss_min_adjoint.py` converged
in **13 PF calls vs 72 for FD-based L-BFGS-B** — a 5.5× reduction. At
larger `d` the gap grows linearly: 50 controls give ~50× fewer solves.

The same structure is **backpropagation** in neural networks and the
**costate equation** in optimal control: forward pass builds the state,
the adjoint runs backward through state dependence to produce the gradient.
