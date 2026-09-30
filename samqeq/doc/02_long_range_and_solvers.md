# 02 — Long-range Coulomb and the charge solvers

Condensed-phase electrostatics needs long-range Coulomb in the **charge solve**, not just the forces. samQEq does
this with a reciprocal-space term inside the matrix-vector product (`pppm/samqeq`), and offers several solvers and
regularizers for the cases where the charge operator goes near-singular.

## Turning on long-range

```
pair_style hybrid/overlay lj/cut 10.0 coul/long 10.0 coul/shield/intra 5.0 all
pair_coeff * * coul/long
pair_coeff * * coul/shield/intra 1.11
kspace_style pppm/samqeq 1.0e-6
fix chg FQ qeq/sam 1 0.0 10.0 1.0e-6 water.param
```

`special_bonds` weights do not change the net coul/long + coul/shield/intra Hamiltonian; they only change how it is
split between the two styles (exact with `pair_modify table 0`, 1e-7 relative with the default table). Under a kspace
style even `lj 0 coul 0` is legal: LAMMPS keeps those bonded pairs in the list with factor 0. The pair refuses
only the no-kspace 0/0 case, where the pairs really are dropped, and `neigh_modify exclude molecule/intra` (the
excluded intramolecular pairs lose the shielded correction and their solve coupling while kspace still sums them).
Other exclusions are allowed; fix gcmc/widom/charge-regulation rely on an internal group exclusion.

**You do not need `kspace_modify gewald`.** The fix takes the Ewald split from the kspace style at setup
(`lr_alpha = force->kspace->g_ewald`), so the solve and the forces use the same split by construction. The param
header's **first** field is `gamma_align`, which no solve path uses (it must be positive); `gewald` does not appear in the param
file at all. Setting `kspace_modify gewald` explicitly is legal and pins the split, but nothing has to match.

**`lr_ewald` is the SIXTH field of the param header, and the deck has to be made consistent with it.** It is not
inferred from the deck, and it is not inferred from the kspace setup. This is the single most expensive mistake
you can make here, because it is silent: with `lr_ewald=2` and a `coul/shield/intra` cutoff that omits `all`,
every intermolecular pair is shielded in the charge matrix and bare in the forces. The charges are bit-identical
either way — the pair scope never enters the solve — so only the forces move, and nothing in the output says so.
The initialization check compares scope and aborts, but the check is what saves you, not the deck reading
sensibly. Read the sixth field of your param file before you write the pair style:

- **`lr_ewald=2`** (fully shielded; the production condensed-phase path): set the `coul/shield/intra` cutoff
  keyword to `all` so *every* pair is shielded in both the solve and the forces (force/solve consistency — this
  is what makes NVE conserve energy and prevents the bare-1/r close-contact collapse near a charged solute/ion).
- **`lr_ewald=1`** (intra-shield / inter-bare): a near-critical variant, NVT-only.

**Which one to pick.** `lr_ewald=2`'s all-pair shielding is **samQEq's own stabilizing
choice** — correct for a water/electrolyte parameterized directly under it (the production path above). But
published bare-Coulomb FQ models (e.g. Rick's TIP4P-FQ) shield **only** their fixed intramolecular geometry and
leave every intermolecular pair bare 1/r — that is exactly `lr_ewald=1`'s structure. Reproducing such a model
**faithfully** (its own published dipole, its own raw parameters, no refit) needs `lr_ewald=1`, not `lr_ewald=2`:
validated end-to-end on the TIP4P-FQ port (`examples/tip4pfq_liquid/`) — with the exact Slater kernel (below)
and Rick's raw parameters, `lr_ewald=1` gives liquid μ=2.635 ± 0.002 D (eight 100 ps runs) against his published 2.637 D, while
`lr_ewald=2` under the *same* exact kernel under-polarizes by ~11% (μ=2.340 D) — a structural difference in
*which pairs get shielded*, not a kernel-fit residual. Rule of thumb:
**native samQEq waters → `lr_ewald=2`; faithful ports of a bare-inter-Coulomb published FQ model → `lr_ewald=1`**
(and budget for `lr_ewald=1` being intrinsically closer to criticality — use XL or a short BO timestep).

## The shielding kernel (`fix_modify <id> shield ...`)

The off-diagonal `J(r)` used by `lr_ewald>=2` (and the short-range gas path) has three interchangeable forms:

```
fix_modify chg shield cbrt # default: J(r) = 1/cbrt(r^3 + 1/gamma^3) -> 1/r; param col 4 = gamma
fix_modify chg shield gaussian [<lambda>] # PQEq Gaussian-Gaussian overlap erf(a_ij r)/r; param col 4 = Rc
fix_modify chg shield slater [2s <type>...] # EXACT Slater-orbital Coulomb integral (Rick JCP101,6141); param col 4 = zeta (1/A)
```

Mirror the same choice on the pair style so force and solve stay consistent:
`pair_style coul/shield/intra <cut> [intra|all] [cbrt|gaussian|slater] [lambda <val>] [2s <type>...]`.

### The force↔solve deck contract (checked at startup)

The pair style owns the **forces/energy**; the fix owns the **charge solve**. They select the shielding kernel
independently, so a deck can silently integrate forces from one kernel while equilibrating charges against
another, with no error, no warning and plausible-looking output (a thermostat-off molten-salt NVE with
mismatched kernels reaches 12,400 K in 4.5 ps). `FixQEqSam::init()` asserts four things
and errors at startup if they disagree:

| channel | pair side | fix side |
|---|---|---|
| kernel mode | `cbrt`/`gaussian`/`slater` | `fix_modify <id> shield <mode>` |
| PQEq λ | `lambda <val>` | `fix_modify <id> shield gaussian <lambda>` |
| Slater 2s flags | `2s <type>...` | `fix_modify <id> shield slater 2s <type>...` |
| range | `<cut>` | the fix's `swb` |

**The range channel is a magnitude test, not `cut == swb`.** A cutoff difference only matters to the extent
the correction `J(r) − 1/r` is still nonzero where one side stops applying it, and that depends entirely on the
kernel and the col-4 parameters. The check computes the actual residual `qqrd2e·|J(r_trunc) − 1/r_trunc|` at
`r_trunc = min(cut, swb)` and errors above a tolerance (default **0.02 eV per unit charge pair**). The residual
spans five orders of magnitude between systems: a compact kernel leaves ~0.0001 eV at `cut 5 / swb 8`, while wide
Gaussians (`Rc` 1.856 Å) leave **0.56 eV** at the same 5/8, because they are far from converged at 5 Å. So
`cut < swb` is *tolerated and common* (most
decks in `examples/` use it and are perfectly consistent in practice) and is only rejected when it actually
costs you. `cut == swb` always passes.

If a deck does trip it, the fixes in order of preference are: set the pair cutoff equal to `swb`; or, if you have
established the residual is acceptable for your system, raise the bar with
`fix_modify <id> shieldcheck <tol>` (`shieldcheck off` disables the whole assertion and warns).

`slater` is the kernel behind published Slater-overlap FQ water models — with it, a model's own raw ζ
parameters (e.g. Rick's TIP4P-FQ: H 1s ζ=1.7007, O/M 2s ζ=3.0803 Å⁻¹) port with **no refit at all** (`2s <type>
...` marks which types use the O/M 2s form factor; unlisted types default to 1s, e.g. H). All three kernels
tabulate to `1/r` at large `r` and are dimensionally unit-correct in both `metal` and `real` (ζ/γ/Rc are
lengths⁻¹, never scaled by `ev_scale`). Validated: `examples/tip4pfq_liquid/in.gas_slater` reproduces Rick's gas
monomer μ=1.8611 D (paper 1.85 D) with zero refit; `check_slater_table.py` cross-checks the tabulated kernel
against an independent reference quadrature (max rel. deviation 5.2e-15). `lr_ewald=1` also honors `shield
slater`/`gaussian` (this is exactly what makes the faithful TIP4P-FQ port above exact). `shield slater`
combined with plain DSF (`lr_alpha>0`) is refused. The Kokkos fix, including its device solve, uses the Slater
kernel (`tests/cases/slater_gas_kernel` is bit-identical under `-sf kk`).

## The solvers (`fix_modify <id> solver ...`)

| Directive | Effect |
|-----------|--------|
| *(default)* | Projected **CG** (per-fragment-neutral subspace) + adaptive ridge, with an automatic **CG→MINRES fallback** if CG diverges. Fast on well-conditioned systems, robust on hard ones. |
| `solver minres [<qcap>]` | Force **projected MINRES** (Paige–Saunders, indefinite-safe). Use for genuinely indefinite/near-singular operators (the close-pair FQ catastrophe, near-critical transients). Optional `<qcap>` truncates a stalled, blown-up solve. |
| `solver cg nofallback` | Disable the auto-fallback (CG errors instead of retrying with MINRES). |
| `solver warmstart on` | **Charge predictor**: seed each solve from the previous converged charges. Cuts the iteration count ~30% in dynamics (consecutive configs differ little). Default off ⇒ first/single-point solves cold-start. |

Unpreconditioned MINRES costs essentially the same as preconditioned CG on well-conditioned water (≈18 vs 18
iterations at tol 1e-7), so the auto-fallback is "free" robustness — you rarely need to set the solver by hand.

## The ridge (`fix_modify <id> ridge ...`) — for near-critical configs

The fluctuating-charge operator can go soft (a collective mode crosses zero) and produce runaway charges. The
ridge regularizes it:

| Directive | Effect |
|-----------|--------|
| `ridge off` | Discrete ×4 ridge escalation. |
| `ridge eig <floor> [<m>] [<every>] [<delta>]` | **Recommended for hard systems.** An `m`-step Lanczos estimate of the smallest eigenvalue λ_min preconditions the diagonal so `λ_min(A+ridge) ≥ floor` — continuous through criticality (smooth charges/forces). E.g. `ridge eig 1.0 16`. The optional `<every>` (default 1) **recomputes λ_min only every N steps** — see the performance note below. The optional `<delta>` (default 0) is the estimate-error budget — see *The estimate is an upper bound* below. |
| `ridge <q_onset> <gain>` | Smooth max\|q\| ramp. |
| `ridge local <q_onset> <factor>` | Per-atom ratchet ridge (bounds a single close-pair catastrophe without touching the bulk). |

Rule of thumb: a +2 ion in water or a dense metal ⇒ add `ridge eig 1.0 16` (or
`solver minres`).

> **The estimate is an upper bound.** A Lanczos (Ritz) value is a Rayleigh quotient on a small Krylov space, so it
> can only err *upward*: an unconverged `m` can miss a negative mode, never invent one. On a collapsed interface cell
> `m = 15` gives λ_min = +2.17 eV where the converged value is −0.31, so the ridge never engages on an
> indefinite operator. The log reports how far the estimate moved over its last 10 steps (and warns once if a later
> estimate is still moving). Two defences: raise `m` (**`m ≥ 60` for interface cells**; `m` is capped at 200, with a
> warning), and set `<delta>` to the estimate error you measured in an `m` scan. The ridge then targets
> `floor + delta`, so `λ_min(A+ridge) = floor + (delta − error) ≥ floor` whenever the error is within `delta`.
> `delta = 0` targets `floor` itself.

> **Performance — cache the Lanczos λ_min.** Each `ridge eig` step costs `m` extra matvecs (each a PPPM FFT) just
> to estimate λ_min — on top of the actual solve. But λ_min varies *slowly* with configuration, so it need not be
> recomputed every step. `ridge eig 1.0 16 10` recomputes it **every 10 steps** instead: a 512-water NVE runs
> **~1.5× faster (3:00 → 2:00) with byte-for-byte the same energy drift**. The default `every=1` recomputes it
> every step (and is exactly bit-reproducible). A stale λ_min that under-bounds a sudden near-critical
> config is caught by the adaptive re-solve + the CG→MINRES auto-fallback, so larger cadences stay safe.
> *(ASPC, below, skips the Lanczos entirely — it sets the ridge from the base value.)*

## Faster dynamics: ASPC predictor–corrector (`fix_modify <id> aspc ...`)

Warm-start seeds each solve from the *previous* charges. **ASPC** goes further: it predicts the charges from a
**multi-step time-reversible history** (Kolafa always-stable coefficients) and then runs only a **fixed few
corrector iterations** instead of converging the CG to tolerance — the **Born–Oppenheimer ↔ XL middle ground**
(more accurate than XL, much cheaper than a full solve, and needs *no* charge thermostat).

```
fix_modify chg aspc 2 # n_corr = 2 corrector iterations, predictor order k = 2 (default)
fix_modify chg aspc 2 3 # ... with order k = 3
fix_modify chg aspc rtol 0.01 # accept the corrector only if its rel-residual < 0.01 (default), else exact BO
fix_modify chg aspc off # back to full Born–Oppenheimer
```

It works because the lr_ewald=2 solve (`qeq_solve`) is already a projected CG on the per-fragment-neutral
subspace (q-direct): predict `q`, run `n_corr` projected-CG steps from the prediction, ω-damp toward the
predictor (`ω=(k+2)/(2k+3)`), and store the corrected `q`. The history migrates with the atoms (so it is *not*
reset on reneighbor). A corrected step is **accepted only if it is both non-runaway (`|q|<4`) and actually
converged** — its relative residual must beat `rtol` (default `0.01`); otherwise it falls back to the full
adaptive-ridge solve (which also re-seeds the predictor history with the exact charges). This makes ASPC an
**adaptive BO↔ASPC switch**: cheap where the predictor is good, exact BO where it is not.

> **What ASPC accuracy means.** An accepted step is *not* converged to `tolerance`. The corrector runs
> `n_corr` iterations, and the committed charge is the ω-mix (ω = 0.571 at k = 2) of the corrector output and the
> predictor, so it keeps (1−ω) of the predictor error whatever `n_corr` is. Raising `n_corr` improves the corrector's
> residual but not the committed charge.
>
> Every 200 steps the log prints `samqeq ASPC-DIAG step N: resid/b corrector=… committed=…` — quote the committed
> number. A once-per-run warning fires if the corrector sits above `tolerance` for 100 consecutive accepted steps.
>
> The capped corrector does not raise the "projected CG did not converge" warning. Ridge, Picard and
> over-polarization warnings are rate-limited, and SOLVER-DIAG reports the ridge engagement count.

| | matvecs/solve | energy drift | notes |
|---|---|---|---|
| Born–Oppenheimer | 16.8 | baseline | full CG to tol each step |
| **`aspc 2`** | **2.9** | **= BO (within noise)** | **~6× cheaper**, no thermostat |
| `aspc 1` | — | ~4× worse | under-converges this stiff operator — use `n_corr=2` |

*(512-water SPC-FQ NVE, lr_ewald=2; ~3.4× wall-clock vs full BO since it also skips the per-step `ridge eig`
Lanczos.)* Watch `SOLVER-DIAG` drop from ~17 iters to ~2 once the history fills (after `k+2` steps).
Test case: `tests/cases/aspc_corrector`.

## Energy-conserving charge dynamics (XL)

Instead of solving the charges every step (Born–Oppenheimer), propagate them by an extended Lagrangian:

```
fix_modify chg xl <q_mass> <q_tdamp> # e.g. 1.0e-3 0.1 ; requires lr_ewald>0
fix_modify chg spikeguard 2.0 10.0 # optional: soft cap on |q| spikes
```

XL gives a single pass per step (no inner solve), good for long NVE and clean velocity autocorrelations (2PT
entropy). Combine with a tight base tolerance for the initial solve.

## OpenMP threading (`package omp N`)

The solve is matvec-bound; standard LAMMPS OpenMP threading (`package omp N`, `comm->nthreads`) threads the
hot loops **inside the existing fix** — no new `fix_modify` keyword needed:

```
package omp 4
fix chg all qeq/sam 1 0.0 10.0 1.0e-6 sys.param # unchanged
```

It is **grain-gated**: below ~4096 atoms/rank the serial loops run unchanged (small cells measured
36% *slower* under naive threading, because fork/join overhead swamps tiny loop bodies), so small/
medium decks should expect **no change**. Results at `nthreads>1` are deterministic run-to-run at a *fixed*
thread count (`schedule(static)` + thread-ordered reductions) but **not** bit-identical to the serial path or to
a different thread count (FP summation reorder — same policy as `-sf opt`; never use it as a bit-test
baseline). Best win: relieving MPI-collective pressure on large, rank-heavy runs (packed concurrent jobs),
not raw single-job speedup on small cells. See the top-level README's *OpenMP
threading* section for the measured numbers.

## Run boundaries, restarts and `run 0`

At a run *boundary* — a second `run` in the same deck, or the first `run` after `read_restart` (any setup at
`ntimestep > 0`) — the Ewald routes do **not** solve the charges during setup: they reuse the converged charges
already in `atom->q` (a cold re-solve can over-polarise a state the previous run held), and the first in-loop step
re-solves. For MD this is invisible. For a **`run 0`** it means no solve happens at all: the forces and energy are
those of the *stored* charges, and any model change since they were stored (`gself`, parameters, …) is **not**
reflected. The log says so:

```
samqeq: setup solve SKIPPED at run-boundary step 500000 (2nd+ run / restart): the setup forces use the STORED atom->q; this `run 0` performs NO charge solve, ...
```

To solve at the boundary, reset the step counter before the run:

```lammps
read_restart equil.restart
reset_timestep 0
fix_modify chg gself on width 0.50 types 12
run 0 # the setup now solves the charges
```

The gas path (`lr_ewald 0`) always solves at a boundary.

## Diagnostics

The solve prints `samqeq SOLVER-DIAG step N: CG <k> iters ...` every 200 steps (tagged `[warm]` when warm-start is
on) — watch the iteration count to see the solver cost and confirm warm-start is helping.

## Next

Total energy and free-energy methods → [10](10_free_energy.md).
