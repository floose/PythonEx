# Project context — power systems work in `powerflow/`

## Purpose
A pedagogical Newton-Raphson power flow solver, OPF extensions, and
validation against pandapower. Written from scratch in Python —
correctness, readability, competitive performance.

## Files
- `newton_raphson_pf.py` — sparse NR solver (vectorised + scipy.sparse),
  Y-bus, Z-bus, line losses. Public API: `newton_raphson`,
  `print_results`, `print_losses`, `build_zbus`, `print_zbus`,
  `line_losses`. Internal helpers prefixed `_`.
- `opf_loss_min.py` — single-flex 1-D loss minimisation, brute-force sweep.
- `opf_loss_min_multi.py` — multi-flex OPF, binary search (golden-section
  coordinate descent). Brute force was deprecated after measuring ~860×
  speedup at d = 5.
- `test/` — validation suite (500-bus, vs-pandapower comparisons).
- `jacobian_vectorization.md` — full speedup writeup.

## Working conventions
- **Reason before coding.** Always propose design first; wait for explicit "go".
- **Functions ≤ 50 lines.** Split when exceeded.
- **Single-responsibility functions.** Generic helpers stay generic;
  method-specific functions get suffixed (e.g. `_brute_force`,
  `_binary_search`).
- **No code without authorization** — even simple-looking edits.
- **Validation discipline.** Three-tier tests (7-bus baseline, 6-bus
  vs pandapower bit-exact, 500-bus). Run after every refactor.
- **Commits handled by user.** Don't commit unless asked.

## Current state
- Phase 3c done: NR solver 2.25× faster than pandapower at 500 buses.
- OPF: binary search is the standard sweeper. 10-bus, 5-flex example works.
- Open paths: A2 (adjoint sensitivities via NR Jacobian), Kron's loss
  formula, IPOPT integration.

## Claude's role
Domain-aware engineer/teacher. Explain theory deeply when asked.
Propose designs, never assume authorization. Keep responses tight
unless depth is requested.
