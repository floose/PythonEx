# IEEE 30-bus benchmark — system data

Canonical OPF benchmark derived from a 1961-vintage American Electric
Power System snapshot. Used in countless OPF / ORPD studies (Alsac &
Stott 1974; Granville 1994; Mantovani & Garcia 1996; MATPOWER `case30`).

Used here as a **stressed-transmission test bed for the adjoint OPF
method** — see `test/test_adjoint_ieee30_heavy.py`.

## Per-unit base and voltage levels

| quantity | value |
|---|---:|
| `BASE_MVA` | 100.0 MVA |
| Transmission voltage | 132 kV (buses 1–8, 28) |
| Sub-transmission voltage | 33 kV (buses 9–27, 29–30) |

The 132 kV / 33 kV split is preserved from the original 1961 case and
is *informational* — per-unit math is unchanged. The R, X, B values
in the branch table below are already on the 100 MVA base.

## Heavy-peak loading factor

A single multiplicative factor on every `P_LOAD` and `Q_LOAD`:

```
HEAVY_PEAK_FACTOR = 1.05
```

This places the baseline at ~6.65 % losses — within the 5–7 % range
documented for stressed IEEE 30 operation (Alsac & Stott 1974;
Mantovani & Garcia 1996). The very modest 1.05× factor reflects two
PF-model simplifications in our solver compared to canonical MATPOWER:

1. **No bus-shunt support** — the 4.3 MVAr fixed shunt at bus 24 is
   omitted.
2. **No transformer tap ratios** — the 4 transformers in the case
   (which have published tap ratios slightly off-nominal) are treated
   as plain π-model branches with R, X, B = 0.

Both omissions raise losses at default loading to ~5 % already, so a
small extra peak factor lands in the target. With a full-fidelity PF,
the equivalent stress requires ~1.5–1.6× scaling.

## Bus data — 30 rows

`P_LOAD` and `Q_LOAD` here are **default values**; multiply by 1.05 in
the inline `bus_data()` to reach the heavy-peak operating point.

| bus | type | V_set | P_GEN (MW) | Q_min (MVAr) | Q_max (MVAr) | P_LOAD (MW) | Q_LOAD (MVAr) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1  | SLACK | 1.060 | (open) | -10  | 150 | 0    | 0    |
| 2  | PV    | 1.043 | 40     | -40  | 50  | 21.7 | 12.7 |
| 3  | PQ    | —     | —      | —    | —   | 2.4  | 1.2  |
| 4  | PQ    | —     | —      | —    | —   | 7.6  | 1.6  |
| 5  | PV    | 1.010 | 0      | -40  | 40  | 94.2 | 19.0 |
| 6  | PQ    | —     | —      | —    | —   | 0    | 0    |
| 7  | PQ    | —     | —      | —    | —   | 22.8 | 10.9 |
| 8  | PV    | 1.010 | 0      | -10  | 40  | 30.0 | 30.0 |
| 9  | PQ    | —     | —      | —    | —   | 0    | 0    |
| 10 | PQ    | —     | —      | —    | —   | 5.8  | 2.0  |
| 11 | PV    | 1.082 | 0      | -6   | 24  | 0    | 0    |
| 12 | PQ    | —     | —      | —    | —   | 11.2 | 7.5  |
| 13 | PV    | 1.071 | 0      | -6   | 24  | 0    | 0    |
| 14 | PQ    | —     | —      | —    | —   | 6.2  | 1.6  |
| 15 | PQ    | —     | —      | —    | —   | 8.2  | 2.5  |
| 16 | PQ    | —     | —      | —    | —   | 3.5  | 1.8  |
| 17 | PQ    | —     | —      | —    | —   | 9.0  | 5.8  |
| 18 | PQ    | —     | —      | —    | —   | 3.2  | 0.9  |
| 19 | PQ    | —     | —      | —    | —   | 9.5  | 3.4  |
| 20 | PQ    | —     | —      | —    | —   | 2.2  | 0.7  |
| 21 | PQ    | —     | —      | —    | —   | 17.5 | 11.2 |
| 22 | PQ    | —     | —      | —    | —   | 0    | 0    |
| 23 | PQ    | —     | —      | —    | —   | 3.2  | 1.6  |
| 24 | PQ    | —     | —      | —    | —   | 8.7  | 6.7  |
| 25 | PQ    | —     | —      | —    | —   | 0    | 0    |
| 26 | PQ    | —     | —      | —    | —   | 3.5  | 2.3  |
| 27 | PQ    | —     | —      | —    | —   | 0    | 0    |
| 28 | PQ    | —     | —      | —    | —   | 0    | 0    |
| 29 | PQ    | —     | —      | —    | —   | 2.4  | 0.9  |
| 30 | PQ    | —     | —      | —    | —   | 10.6 | 1.9  |

**Default totals:** P_LOAD = 283.4 MW, Q_LOAD = 126.2 MVAr. At 1.05× → 297.6 MW, 132.5 MVAr.

## Branch data — 41 rows (R, X, B in pu on 100 MVA base)

37 transmission lines + 4 transformers. Transformers marked **T**.

| # | from | to | R (pu) | X (pu) | B (pu) | kind |
|---:|---:|---:|---:|---:|---:|---|
| 1  | 1 | 2 | 0.0192 | 0.0575 | 0.0528 | line |
| 2  | 1 | 3 | 0.0452 | 0.1652 | 0.0408 | line |
| 3  | 2 | 4 | 0.0570 | 0.1737 | 0.0368 | line |
| 4  | 3 | 4 | 0.0132 | 0.0379 | 0.0084 | line |
| 5  | 2 | 5 | 0.0472 | 0.1983 | 0.0418 | line |
| 6  | 2 | 6 | 0.0581 | 0.1763 | 0.0374 | line |
| 7  | 4 | 6 | 0.0119 | 0.0414 | 0.0090 | line |
| 8  | 5 | 7 | 0.0460 | 0.1160 | 0.0204 | line |
| 9  | 6 | 7 | 0.0267 | 0.0820 | 0.0170 | line |
| 10 | 6 | 8 | 0.0120 | 0.0420 | 0.0090 | line |
| 11 | 6 | 9  | 0.0    | 0.2080 | 0      | **T** |
| 12 | 6 | 10 | 0.0    | 0.5560 | 0      | **T** |
| 13 | 9 | 11 | 0.0    | 0.2080 | 0      | line |
| 14 | 9 | 10 | 0.0    | 0.1100 | 0      | line |
| 15 | 4 | 12 | 0.0    | 0.2560 | 0      | **T** |
| 16 | 12 | 13 | 0.0   | 0.1400 | 0      | line |
| 17 | 12 | 14 | 0.1231 | 0.2559 | 0      | line |
| 18 | 12 | 15 | 0.0662 | 0.1304 | 0      | line |
| 19 | 12 | 16 | 0.0945 | 0.1987 | 0      | line |
| 20 | 14 | 15 | 0.2210 | 0.1997 | 0      | line |
| 21 | 16 | 17 | 0.0824 | 0.1923 | 0      | line |
| 22 | 15 | 18 | 0.1073 | 0.2185 | 0      | line |
| 23 | 18 | 19 | 0.0639 | 0.1292 | 0      | line |
| 24 | 19 | 20 | 0.0340 | 0.0680 | 0      | line |
| 25 | 10 | 20 | 0.0936 | 0.2090 | 0      | line |
| 26 | 10 | 17 | 0.0324 | 0.0845 | 0      | line |
| 27 | 10 | 21 | 0.0348 | 0.0749 | 0      | line |
| 28 | 10 | 22 | 0.0727 | 0.1499 | 0      | line |
| 29 | 21 | 22 | 0.0116 | 0.0236 | 0      | line |
| 30 | 15 | 23 | 0.1000 | 0.2020 | 0      | line |
| 31 | 22 | 24 | 0.1150 | 0.1790 | 0      | line |
| 32 | 23 | 24 | 0.1320 | 0.2700 | 0      | line |
| 33 | 24 | 25 | 0.1885 | 0.3292 | 0      | line |
| 34 | 25 | 26 | 0.2544 | 0.3800 | 0      | line |
| 35 | 25 | 27 | 0.1093 | 0.2087 | 0      | line |
| 36 | 28 | 27 | 0.0    | 0.3960 | 0      | **T** |
| 37 | 27 | 29 | 0.2198 | 0.4153 | 0      | line |
| 38 | 27 | 30 | 0.3202 | 0.6027 | 0      | line |
| 39 | 29 | 30 | 0.2399 | 0.4533 | 0      | line |
| 40 | 8 | 28 | 0.0636 | 0.2000 | 0.0428 | line |
| 41 | 6 | 28 | 0.0169 | 0.0599 | 0.0130 | line |

## OPF flex selection (Option B revised)

`d = 10` shunt compensators at the heaviest-load 33 kV buses:

```
FLEX_BUS_NUMS = (10, 12, 15, 17, 19, 20, 21, 23, 24, 29)
SHUNT_Q_BOUNDS = (-0.10, +0.30) pu     (-10 to +30 MVAr per bus)
```

Generators 2, 5, 8, 11, 13 stay as **PV** (V-setpoint mode) — their Q
is dispatched automatically by the PF and bounded by `Q_min`/`Q_max`
from the bus table. Originally we proposed gen-Q dispatch as additional
flex variables (d = 15), but demoting all 5 PV gens to PQ-at-Q=0
simultaneously removes the system's reactive support and the baseline
PF fails to converge. Real ORPD studies treat generator-V (not -Q) as
the control variable; we honour that by leaving gens as PV.

## Operating snapshot summary (heavy-peak 1.05× loading)

| quantity | value |
|---|---:|
| total P load | 297.6 MW |
| total Q load | 132.5 MVAr |
| baseline cable losses (no shunt dispatch) | 19.80 MW (6.65 % of load) |
| voltage range at baseline | [0.946, 1.082] pu |
| flex variables | 10 |

The baseline Q = 0 at every shunt is the **uncompensated reference**
standard in OPF benchmarks (Granville 1994; Wood, Wollenberg & Sheblé
2014, ch. 8).
