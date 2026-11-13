"""Force-field parameterization planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ForceFieldSpec, PolymerSpec


@dataclass
class ForceFieldPlan:
    provider: str
    data_file: Path
    script: Path | None
    notes: str


def write_forcefield_notes(polymer: PolymerSpec, spec: ForceFieldSpec, workdir: str | Path) -> Path:
    workdir = Path(workdir)
    path = workdir / "forcefield_plan.md"
    text = f"""# Force-Field Plan

Polymer: `{polymer.name}`
Provider: `{spec.provider}`
Force field: `{spec.forcefield}`

LAMMPS styles used by generated inputs:

- `pair_style {spec.pair_style}`
- `bond_style {spec.bond_style}`
- `angle_style {spec.angle_style}`
- `dihedral_style {spec.dihedral_style}`
- `improper_style {spec.improper_style}`
- `kspace_style {spec.kspace_style}`
- `pair_modify {spec.pair_modify}`
- `special_bonds {spec.special_bonds}`

DMAx2 keeps OPLS 1-4 scaling explicit because missing `special_bonds
lj/coul 0.0 0.0 0.5` changes polymer density, stress, and the DMA phase.

For publication-quality DMA, validate density, static pressure, static elastic
response, and a short strain-amplitude sweep after any force-field change.
"""
    path.write_text(text)
    return path


def plan_forcefield(polymer: PolymerSpec, spec: ForceFieldSpec, workdir: str | Path) -> ForceFieldPlan:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    notes = write_forcefield_notes(polymer, spec, workdir)
    data_file = workdir / f"data.{polymer.name}"

    if spec.provider in {"psp_oplsaa", "existing_lammps_data"}:
        return ForceFieldPlan(spec.provider, data_file, notes, "LAMMPS data is produced upstream or provided.")
    if spec.provider in {"foyer", "mosdef_foyer"}:
        script = workdir / "parameterize_with_foyer.py"
        script.write_text(
            f'''"""Foyer parameterization stub for {polymer.name}."""
from pathlib import Path
import foyer

xml = {spec.foyer_xml!r}
if xml is None:
    raise SystemExit("Set forcefield.foyer_xml in the DMAx2 config.")

# Load your mBuild/ParmEd structure here, apply the XML, and save LAMMPS data.
ff = foyer.Forcefield(forcefield_files=xml)
print(ff)
print("Write data.{polymer.name} after applying atom types.")
'''
        )
        return ForceFieldPlan(spec.provider, data_file, script, "Foyer stub written.")
    if spec.provider == "ligpargen":
        script = workdir / "import_ligpargen_lammps.py"
        script.write_text(
            '"""LigParGen import stub.\n\nConvert LigParGen OPLS-AA output into a full LAMMPS data file, then set forcefield.provider to existing_lammps_data.\n"""\n'
        )
        return ForceFieldPlan(spec.provider, data_file, script, "LigParGen import stub written.")
    raise ValueError(f"Unknown force-field provider: {spec.provider}")
