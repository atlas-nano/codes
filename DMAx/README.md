# DMAx

DMAx measures high-frequency mechanical response by applying sinusoidal axial
strain in atomistic molecular dynamics. This standalone publication package
contains structure builders, explicit LAMMPS preparation/measurement inputs,
an execution entrypoint, measured-strain Fourier harmonics, finite-window runtime
diagnostics, independent-packing summaries and native-frequency/TTS diagnostics.
It preserves the earlier `lockin` and `legacy` estimators for comparison.

**Scientific scope:** fixed transverse dimensions give the constrained
longitudinal response **M₁***. They do not directly give Young's modulus.
The 2% default is a finite-amplitude starting candidate requiring qualification,
not a polymer-independent linear-response standard. Preparation is finite and
diagnosed; stable density does not establish chain equilibration. Experimental
WLF markers are explicitly anchored sensitivity calculations, not independent
glass-transition predictions.

## Installation

Use Python 3.10 or later:

```bash
python -m pip install '.[structures,test]'
python -m pytest -q
dmax --help
```

Install **LAMMPS** separately with MOLECULE, KSPACE and EXTRA-PAIR support
(`pair_style soft`, `lj/cut/coul/long`, harmonic bonds/angles and OPLS dihedrals).
PDMS packing also requires **Packmol** on `PATH`, or `dmax build --packmol /path/to/packmol`.
RDKit supplies PDMS starting conformers and general repeat-graph parsing;
PVDF can be built from the built-in model without RDKit. No credentials,
scheduler, site account, or network download is hidden in the commands below.

## Minimal polymer input

```bash
dmax init --polymer PVDF --output pvdf
dmax init --smiles '[*:1]O[Si](C)(C)[*:2]' --output pdms
```

Alternatively use `--monomer-file repeat.smi` or a MOL/SDF file with two
attachment dummy atoms. Supported parameterized chemistries are PVDF and PDMS;
recognizing a chemical graph does not create force-field parameters.

| Preset | Repeat/end-group convention | Starting architecture |
|---|---|---|
| PVDF | CH₃–[CF₂–CH₂]ₙ–CF₃; nominal VDF count = n+1 | 20 chains × 100 nominal VDF |
| PDMS | M–Dₙ–M; terminal trimethylsiloxy M; n+1 siloxane O links | 40 chains × 30 internal D |

These intermediate architectures are benchmarked starting points, **not** a
size-convergence or entanglement certification. Override with `--repeats`,
`--chains` and `--packings`. The default is three independently built packings.
The builder verifies chain connectivity, caps, typed composition and neutrality.
It records all geometry seeds, force-field asset hashes, RDKit/Packmol details
and the initial-data checksum. Starting coordinates can overlap and require
the complete preparation protocol before measurement.

For other chemistries, provide a two-attachment repeat and
`--parameterized-data polymer.data --packings 1 --chains ACTUAL_COUNT`.
The supported import contract is an orthogonal periodic, full-style LAMMPS file
with harmonic bond/angle, OPLS dihedral and diagonal LJ coefficients, no impropers,
and neutral connected chains. You remain responsible for correct chemistry and
validated parameters. A supplied coordinate file is one packing, regardless of
how many velocity seeds are drawn. General automatic OPLS typing is not implemented.

## Build, prepare and measure

Build substantial structures and execute MD **inside a compute allocation**:

```bash
dmax build pvdf/study.json
dmax plan pvdf/study.json
```

This writes a dependency-aware plan and full input files without running MD.
`study.json` is editable before planning. Defaults are:

- Charge-preserving soft initialization (75 ps) and limited integration (25 ps).
- Hot conditioning: 1000 ps NPT then 200 ps NVT (450 K PVDF; 400 K PDMS).
- Each target: 150 ps ramp, two 750 ps NPT density stages, then 500 ps NVT hold.
- 0.5 fs integration; 2 fs synchronous stress/strain acquisition; total and
  virial stress tensors stored separately; 1 atm preparation pressure.
- Fixed lateral cell; NVT during axial oscillation; thermostat chain length 3.
- Initial record: at least 15 complete cycles **and** 400 ps, rounded upward
  to whole cycles. Further runtime is decided from explicit diagnostics.

The quoted 3350 ps preparation age excludes the 100 ps soft/limited initialization.
Target preparations differ with temperature and packing, while frequencies and
amplitudes branch from the same prepared endpoint. NEMD has no barostat: its
reference density is inherited from NPT and its instantaneous volume oscillates
with axial strain. The source-specific LJ cutoff is 10 Å for PVDF and 11 Å for
PDMS; change such settings only as a new, recorded protocol.

Execute, for example, with an explicit MPI command in your allocation:

```bash
dmax run pvdf/runs/plan.json --command-json '["srun","-n","64","-c","2","--exact","--exclusive","lmp"]'
```

For two independent workers on a sufficiently large allocation add
`--workers 2 --worker-logical-cpus 128`. Dependencies ensure that a measurement
waits for its own preparation. Worker counts must fit the allocated CPUs; tune
MPI ranks/threads to the actual node and system size. `dmax run` never submits
Slurm jobs. It preserves failed inputs, launcher logs and execution receipts.
`--resume` skips only previously successful, checksum-matching endpoints;
it does not overwrite or blindly retry failures. Successful execution is a
numerical completion, not a physical equilibrium or linearity result.

## Preparation and amplitude qualification

```bash
dmax assess-preparation --trace pvdf/runs/p01_T300_preparation/thermal.dat \
  --data pvdf/runs/p01_T300_preparation/prepared.data \
  --stages pvdf/runs/p01_T300_preparation/stages.json --polymer PVDF \
  --geometry pvdf/runs/p01_T300_preparation/geometry.lammpstrj --output prep_check
```

This compares consecutive, fixed-temperature windows for density, potential
energy per mass, pressure, backbone radius of gyration Rg and end-to-end distance
Re. Density is imposed in NVT and is not treated as its convergence test.
`--targets targets.json` supplies **explicit, observable-specific** tolerances;
without targets the result remains descriptive. Neither immobile chains nor
passing finite windows prove global polymer equilibrium. See
[scientific conventions](docs/SCIENTIFIC_CONVENTIONS.md).

Generate an amplitude screen in a separate fresh directory:

```bash
dmax plan pvdf/study.json --workdir pvdf/amplitude_runs --amplitude-screen
```

This tests 0.25, 0.5, 1, 1.5, 2, 4 and 6% at 25 GHz, using matched endpoint/velocity
branches for each packing and target temperature. It writes its own complete
preparation plan; to reuse completed endpoints, generate a new measurement-only
plan with explicitly pinned source endpoints rather than silently duplicating
or relabeling histories. Analyze all amplitudes and compare independent packings,
H2/H1, H3/H1, first-harmonic uncertainty and amplitude dependence. Higher strain
may improve H1 precision while increasing nonlinear distortion. Point harmonic
magnitudes have positive noise bias; this package does not certify linearity
from a small-looking ratio or an R² threshold.

## Analyze harmonics, runtime and uncertainty

```bash
dmax analyze pvdf/runs/p01_T300_f25_a0.02/trace.dat --frequency 25 \
  --case-json pvdf/runs/p01_T300_f25_a0.02/case.json --operator total --output response_total
dmax analyze pvdf/runs/p01_T300_f25_a0.02/trace.dat --frequency 25 \
  --case-json pvdf/runs/p01_T300_f25_a0.02/case.json --operator virial --output response_virial
dmax plot response_total
```

Analysis projects measured strain and stress onto H1–H5 over complete,
uniformly sampled cycles. Equivalent constant-plus-harmonic least squares is
checked numerically. A drift-adjusted fit is a different model and is exposed
as a diagnostic. Irregular sampling, missing points and incomplete cycles are
rejected rather than silently interpolated. A duplicated right endpoint is
explicitly removed. Outputs retain signed storage/loss, actual strain amplitude
and phase, harmonic coefficients, cumulative estimates and source checksums.

Runtime diagnostics bootstrap **joint stress/strain coefficients** in physical
time blocks (default 200 ps, rounded up to whole cycles), and compare disjoint
early/late halves. These are conditional temporal intervals for one history,
not independent-packing confidence intervals. Specify precision and consistency
targets using `--precision-targets` and `--consistency-targets`, each a JSON file
or dictionary with `storage_MPa`, `loss_MPa` and `tan_delta` **relative** tolerances
(optional `phase_deg` is absolute). No universal tolerance is imposed. At least
six blocks are needed for a positive finite-window decision. The initial short
record will often require extension; reaching a cap is reported as unresolved.

```bash
dmax write-continuation --case-json pvdf/runs/p01_T300_f25_a0.02/case.json \
  --checkpoint pvdf/runs/p01_T300_f25_a0.02/final.restart \
  --references pvdf/runs/p01_T300_f25_a0.02/reference_state.dat \
  --study pvdf/study.json --additional-ps 400 --output continuation_1
```

This writes a continuation with the original reference cell, deformation phase,
thermostat ID and checkpoint clock, without reseeding. Verify boundary continuity
before stitching records. The command plans an extension; it does not execute
MD or claim bitwise trajectory equivalence. The proposed 200 ps extension and
2400 ps cap in `study.json` are finite workflow settings, not automatic claims
of runtime convergence. Do not remove initial cycles merely to force a pass.
The extension includes `plan.json` for the same `dmax run` execution command and
an updated `case.json` for a further, separately reviewed continuation.

Aggregate genuinely independent packings after analysis:

```bash
dmax aggregate response_p1/response.json response_p2/response.json response_p3/response.json --output packings
dmax plot-summary packings/packing_summary.csv --output packing_figures
dmax sampling-check trace.dat --frequency 25 --strides 2 5 10 --output sampling_check
```

The aggregation rejects cloned initial coordinates with new labels or velocities.
It reports per-packing values, sample SD and SEM; ratios of mean moduli are kept
distinct from means of individual loss tangents. Sampling checks compare one
dense record with every phase offset of integer-stride decimation, separately
for total and virial stress. They do not substitute for a timestep-convergence test.

## Native-frequency comparison, master curves and WLF

```bash
dmax tts-diagnostics packings/packing_summary.csv --reference-temperature 400 --output tts_check
dmax wlf-marker --reference-temperature 238.15 --reference-frequency 0.01 \
  --target-frequency 25000000000 --c1 17.44 --c2 51.6
```

TTS diagnostics retain chemistry, architecture, amplitude and stress operator,
compare independent storage/loss shifts with one common shift, use only measured
frequency overlap and flag support-bound optima. A numerical optimum is not
validated TTS. Every frequency-edge loss maximum is marked as censored; a
temperature maximum at one frequency does not identify an absolute relaxation
time. No low-frequency extrapolation, WLF/VFT fit or Tg assignment is automatic.
The WLF example is an explicitly assumed archival-anchor sensitivity only.

## Paper reproduction and compatibility

```bash
dmax reproduce-data --output paper_source_tables
dmax reproduce-figures --data paper_source_tables --output paper_figures
```

This verifies and exports the bundled reduced CSVs and their provenance manifest.
See [paper reproduction](docs/PAPER_REPRODUCTION.md) for table/figure regeneration
and the distinction between rebuilding reduced results and new MD realizations.
The measured paper grid uses **1%**, 2.4 ns records with 400–2400 ps retained;
it must not be relabeled as the new 2% candidate default. Historical checkpoint
bit patterns cannot be reconstructed from a new random packing seed alone.

To generate all six **new statistical realizations** of the 432-history grid:

```bash
for polymer in PVDF PDMS; do
  for architecture in short intermediate long; do
    dmax init --polymer "$polymer" --paper-grid --architecture "$architecture" --output "${polymer}_${architecture}"
  done
done
```

Then use `build`, `plan` and allocated `run` for each study. The presets preserve
the material-specific four temperatures, six 10–500 GHz frequencies, three
packings, 1% strain and 2.4 ns duration; use `analyze --start-ps 400`. This can be
expensive. No MD is triggered by `init`. Exact archived checkpoint/seed and
exceptional neighbor-control ancestry remain external-archive requirements;
fresh generated samples should not be presented as the original measurements.

The `dmax2` CLI and import API preserve earlier estimators/interfaces for
comparison. Its original synthetic demonstrations and automatic legacy Tg/TTS
paths are compatibility examples, **not** the recommended publication workflow.
Use the explicit `dmax` entrypoint above for the current workflow.

Parameter origins, source hashes and validation limits are recorded in
`provenance/` and [parameter provenance](docs/PARAMETERS.md). No simulation
trajectories, manuscript documents, private credentials or scheduler controllers
are required by the installed package.
