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
- `opf_loss_min_gscd.py` — multi-flex OPF, golden-section coordinate
  descent (GSCD). Brute force was deprecated after measuring ~860×
  speedup at d = 5.
- `test/` — validation suite (500-bus, vs-pandapower comparisons).
- `jacobian_vectorization.md` — full speedup writeup.

## Working conventions
- **Reason before coding.** Always propose design first; wait for explicit "go". 
- **Functions ≤ 50 lines.** Split when exceeded.
- **Single-responsibility functions.** Generic helpers stay generic;
  method-specific functions get suffixed (e.g. `_brute_force`,
  `_gscd`).
- **Significative functions names** — - Name should be specific and unique. Avoid generic terms (`data`, `handler`, for instance)
- **No code without authorization** — even simple-looking edits.
- **No code duplication** — Extract shared logic into a function/module.
- **Validation discipline.** Create test scripts. Run after every refactor.
- **Commits handled by user.** Don't commit unless asked.

## Current state
- Done: NR solver 2.25× faster than pandapower at 500 buses.
- OPF: GSCD is the standard sweeper. 10-bus, 5-flex example works.
- Open paths: Method of adjoint sensitivities via NR Jacobian, then IPOPT integration.

## AI agent's role
Domain-aware engineer/teacher. Explain theory deeply when asked.
Propose designs, never assume authorization. Keep responses tight
unless depth is requested.
