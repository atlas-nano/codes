# Parameter and builder provenance

The package vendors the actual PVDF parameter/topology tables and PDMS typed
writer used for the validation campaign, rather than assigning a new force
field from a chemical name. `provenance/chemistry_sources.json` records exact
source SHA256 values; generated build manifests also pin the shipped asset.
Archive paths identify provenance only; they are not runtime dependencies.

PVDF coefficients and cap-aware topology are extracted from
`generate_pvdf_1600x2_psp_oplsaa.py` (source SHA
`f8c91c767eb9721a00c97e437567b325d2dca351bee72518b2da7fcdf84141f4`).
The modern builder extends unconfined periodic coils with carbon–carbon bond
length 1.529 Å and angle 112.7°. Initial trans/gauche probabilities 0.6/0.2/0.2
are geometry initialization heuristics, not force-field-derived equilibrium
populations or phase assignments. Nominal VDF count includes the cap convention.

PDMS uses the actual cap-aware OPLS-AA writer
`adv1_pdms_builder_v1_pdms_oplsaa.py` (SHA
`adc137fd59ecdac4d682f7dd0cc2e2854905dc30feedd8c8be7a06aef91c8bd2`),
whose parameter provenance names Xiang et al., *J. Phys. Chem. B* **129**,
1864–1873 (2025), Supporting Information Table S4. End/interior partial charges
differ even for one LAMMPS atom type. The writer retains the explicit small
uniform neutralization correction of the rounded charge table; it is recorded
per build. Do not replace the actual atom charges by an atom-type lookup.

PDMS follows the modern heavy-first route: one independent ETKDGv3 conformer
per chain (`useRandomCoords=True`), explicit hydrogens, UFF starting geometry,
then Packmol at 0.1 g cm⁻³. UFF success/status is recorded without relabeling
an iteration-limited optimizer as converged. Packmol's outcome and periodic
interchain geometry are checked. Charge-preserving soft initialization compresses
the box to 0.65 g cm⁻³ before physical hot conditioning. UFF and Packmol do not
define the production force field or establish equilibration.

Production conventions are `units real`, full atoms, harmonic bonds/angles,
OPLS dihedrals, `special_bonds lj/coul 0 0 0.5`, geometric mixing and PPPM
accuracy 10⁻⁵. Presets retain the archived 10 Å PVDF and 11 Å PDMS cutoffs.
The pressure temperature compute, total/virial definitions, timestep, stress
acquisition interval, neighbor policy, thermostat chain, actual velocities and
all finite preparation stages appear in generated inputs. Inputs are evidence;
scientific suitability still requires the diagnostics and comparisons described
in the paper.

The tiny typed exports in `tests/fixtures/` test chemistry/connectivity/export
contracts. They are **not** equilibrated samples. The 13 November 2026 allocated
LAMMPS run-0 smoke integrated zero MD steps and checked syntax plus finite
compute values only. The tiny PVDF box has very large initial forces and is
unsuitable for dynamics before full initialization. The smoke does not validate
physical equilibrium, force-field accuracy, modulus convergence, or the complete
preparation trajectory. See `provenance/lammps_run0_validation.json`.
