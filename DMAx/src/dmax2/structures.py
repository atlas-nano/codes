"""Structure generation helpers for PSP, MoSDeF/Foyer, and existing data files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PolymerSpec


@dataclass
class StructureGenerationPlan:
    method: str
    script: Path | None
    data_file: Path
    notes: str


def write_psp_structure_script(spec: PolymerSpec, workdir: str | Path) -> Path:
    """Write a PSP script for the polymer structure.

    The script mirrors the local PDMS 11-mer/100-chain convention while keeping
    parameters configurable.
    """

    workdir = Path(workdir)
    script = workdir / "generate_structure_psp.py"
    smiles = spec.smiles.replace("\\", "\\\\").replace('"', '\\"')
    text = f'''from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PSP_DIR = ROOT / "PSP"
if not PSP_DIR.exists():
    PSP_DIR = Path.cwd().parents[0] / "PSP"
sys.path.insert(0, str(PSP_DIR))

import psp.AmorphousBuilder as ab

data = pd.DataFrame(
    [["{spec.name}", "{smiles}", {spec.monomers_per_chain}, {spec.chains}, {spec.conformers}, {spec.cyclic}]],
    columns=["ID", "smiles", "Len", "Num", "NumConf", "Loop"],
)

builder = ab.Builder(
    data,
    density={spec.initial_density_g_cm3},
    input_dir=Path.cwd(),
    output_dir=Path.cwd(),
    length=None,
    box_type="c",
)
builder.Build()
builder.get_opls(output_fname="data.{spec.name}", forcefield="oplsaa", neutralize=True)
'''
    script.write_text(text)
    return script


def write_mosdef_foyer_stub(spec: PolymerSpec, workdir: str | Path) -> Path:
    """Write a MoSDeF/Foyer starter script for users with a local builder."""

    workdir = Path(workdir)
    script = workdir / "generate_structure_mosdef_foyer.py"
    text = f'''"""MoSDeF/Foyer structure generation stub for {spec.name}.

DMAx2 keeps this explicit because the exact chain builder and XML force field
are project-specific. Fill in the mBuild compound construction, then call Foyer
to apply atom types and write LAMMPS data.
"""

from pathlib import Path

import foyer
import mbuild as mb

out = Path("data.{spec.name}")

# TODO: construct an mBuild Compound containing:
#   smiles={spec.smiles!r}
#   monomers_per_chain={spec.monomers_per_chain}
#   chains={spec.chains}
#   initial_density_g_cm3={spec.initial_density_g_cm3}
compound = mb.Compound()

# TODO: replace with a validated XML path.
forcefield = foyer.Forcefield(forcefield_files="oplsaa.xml")
typed = forcefield.apply(compound)
typed.save(str(out), overwrite=True)
print(f"Wrote {{out}}")
'''
    script.write_text(text)
    return script


def plan_structure_generation(spec: PolymerSpec, workdir: str | Path) -> StructureGenerationPlan:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    data_file = workdir / f"data.{spec.name}"

    if spec.structure_method == "existing_data":
        if spec.existing_data_file is None:
            raise ValueError("existing_data_file is required for structure_method='existing_data'.")
        return StructureGenerationPlan(
            method="existing_data",
            script=None,
            data_file=Path(spec.existing_data_file),
            notes="Using user-provided LAMMPS data file.",
        )
    if spec.structure_method == "psp":
        return StructureGenerationPlan(
            method="psp",
            script=write_psp_structure_script(spec, workdir),
            data_file=data_file,
            notes="PSP amorphous builder followed by PSP OPLS-AA LAMMPS data export.",
        )
    if spec.structure_method in {"mosdef", "foyer", "mosdef_foyer"}:
        return StructureGenerationPlan(
            method="mosdef_foyer",
            script=write_mosdef_foyer_stub(spec, workdir),
            data_file=data_file,
            notes="MoSDeF/Foyer stub written; fill project-specific compound builder.",
        )
    raise ValueError(f"Unknown structure method: {spec.structure_method}")
