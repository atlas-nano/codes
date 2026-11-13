"""PDMS OPLS-AA LAMMPS writer.

The parameters here are from Table S4 of the Supporting Information for
Xiang et al., J. Phys. Chem. B 2025, 129, 1864-1873
("Force-Field Benchmark for Polydimethylsiloxane...").

The writer intentionally emits LAMMPS ``real`` units directly:
pair_style lj/cut/coul/long, bond_style harmonic, angle_style harmonic,
and dihedral_style opls.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import re
import subprocess


@dataclass(frozen=True)
class AtomRecord:
    atom_id: int
    name: str
    element: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Topology:
    atoms: dict[int, AtomRecord]
    bonds: list[tuple[int, int]]
    box: tuple[float, float, float, float, float, float] | None = None


AVOGADRO = 6.02214076e23


ATOM_TYPES = {
    "C": {
        "id": 1,
        "label": "opls_pdms_C",
        "mass": 12.011,
        "epsilon": 0.066,
        "sigma": 3.5,
        "charge_inside": -0.342,
        "charge_end": -0.386,
    },
    "H": {
        "id": 2,
        "label": "opls_pdms_H",
        "mass": 1.008,
        "epsilon": 0.030,
        "sigma": 2.5,
        "charge_inside": 0.064,
        "charge_end": 0.083,
    },
    "O": {
        "id": 3,
        "label": "opls_pdms_O",
        "mass": 15.999,
        "epsilon": 0.140,
        "sigma": 2.9,
        "charge_inside": -0.600,
        "charge_end": -0.578,
    },
    "Si": {
        "id": 4,
        "label": "opls_pdms_Si",
        "mass": 28.086,
        "epsilon": 0.100,
        "sigma": 4.0,
        "charge_inside": 0.900,
        "charge_end": 0.700,
    },
}

BOND_TYPES = {
    ("O", "Si"): {"id": 1, "label": "O-Si", "k": 374.0, "r0": 1.64},
    ("C", "Si"): {"id": 2, "label": "Si-C", "k": 187.0, "r0": 1.86},
    ("C", "H"): {"id": 3, "label": "C-H", "k": 340.0, "r0": 1.09},
}

ANGLE_TYPES = {
    ("H", "C", "H"): {"id": 1, "label": "H-C-H", "k": 33.0, "theta0": 107.8},
    ("Si", "C", "H"): {"id": 2, "label": "Si-C-H", "k": 35.0, "theta0": 109.5},
    ("Si", "O", "Si"): {"id": 3, "label": "Si-O-Si", "k": 20.0, "theta0": 145.0},
    ("C", "Si", "C"): {"id": 4, "label": "C-Si-C", "k": 60.0, "theta0": 110.0},
    ("C", "Si", "O"): {"id": 5, "label": "C-Si-O", "k": 60.0, "theta0": 100.0},
    ("O", "Si", "O"): {"id": 6, "label": "O-Si-O", "k": 60.0, "theta0": 110.0},
}

DIHEDRAL_TYPES = {
    ("C", "Si", "C", "H"): {
        "id": 1,
        "label": "C-Si-C-H",
        "coeffs": (0.0, 0.0, 0.18, 0.0),
    },
    ("O", "Si", "C", "H"): {
        "id": 2,
        "label": "O-Si-C-H",
        "coeffs": (0.0, 0.0, 0.0, 0.0),
    },
    ("Si", "O", "Si", "O"): {
        "id": 3,
        "label": "Si-O-Si-O",
        "coeffs": (0.0, 0.0, 0.0, 0.0),
    },
    ("C", "Si", "O", "Si"): {
        "id": 4,
        "label": "C-Si-O-Si",
        "coeffs": (0.0, 0.0, 0.0, 0.0),
    },
}


def paper_pdms_smiles(monomers: int) -> str:
    """Return trimethylsiloxy-capped PDMS with ``monomers`` siloxane links."""
    if monomers < 1:
        raise ValueError("monomers must be positive")
    return "C[Si](C)(C)" + "O[Si](C)(C)" * monomers + "C"


def build_chain_pdb(
    monomers: int,
    output_pdb: str | Path,
    random_seed: int = 20260503,
):
    """Build one capped PDMS chain and write it as PDB with CONECT records."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = paper_pdms_smiles(monomers)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("RDKit failed to embed the PDMS chain")
    AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToPDBFile(mol, str(output_pdb))
    return mol


def cubic_box_length_angstrom(mol, chain_count: int, density_g_cm3: float) -> float:
    """Return the cubic box length for ``chain_count`` molecules at density."""
    molecular_weight = sum(atom.GetMass() for atom in mol.GetAtoms())
    mass_g = molecular_weight * chain_count / AVOGADRO
    volume_cm3 = mass_g / density_g_cm3
    volume_angstrom3 = volume_cm3 * 1.0e24
    return volume_angstrom3 ** (1.0 / 3.0)


def find_packmol_executable(packmol_exec: str | Path | None = None) -> Path:
    """Locate Packmol for PDMS amorphous packing."""
    candidates = [
        packmol_exec,
        os.environ.get("PACKMOL_EXEC"),
        Path(__file__).resolve().parents[1] / ".conda" / "psp" / "bin" / "packmol",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Packmol executable not found. Set PACKMOL_EXEC or pass packmol_exec."
    )


def run_packmol(
    molecule_pdb: str | Path,
    output_pdb: str | Path,
    chain_count: int,
    box_length: float,
    packmol_dir: str | Path,
    packmol_exec: str | Path | None = None,
    tolerance: float = 2.0,
) -> None:
    """Pack capped PDMS chains into a cubic periodic box."""
    packmol = find_packmol_executable(packmol_exec)
    molecule_pdb = Path(molecule_pdb)
    output_pdb = Path(output_pdb)
    packmol_dir = Path(packmol_dir)
    packmol_dir.mkdir(parents=True, exist_ok=True)

    input_file = packmol_dir / "packmol.inp"
    output_log = packmol_dir / "packmol.out"
    lower = 1.0
    upper = box_length - 1.0
    input_file.write_text(
        "\n".join(
            [
                f"tolerance {tolerance}",
                "nloop 20",
                f"output {output_pdb}",
                "filetype pdb",
                "",
                f"structure {molecule_pdb}",
                f"  number {chain_count}",
                f"  inside box {lower:.6f} {lower:.6f} {lower:.6f} {upper:.6f} {upper:.6f} {upper:.6f}",
                "end structure",
                "",
            ]
        )
    )
    with input_file.open("rb") as stdin, output_log.open("wb") as stdout:
        result = subprocess.run(
            [str(packmol)],
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not output_pdb.exists():
        raise RuntimeError(
            f"Packmol did not write {output_pdb}; exit code {result.returncode}. "
            f"See {output_log}."
        )


def write_poscar(topology: Topology, output_vasp: str | Path, box_length: float) -> None:
    """Write a VASP POSCAR-style coordinate file for the packed PDMS system."""
    output_vasp = Path(output_vasp)
    elements = ["C", "H", "O", "Si"]
    atoms_by_element = {
        element: [atom for atom in topology.atoms.values() if atom.element == element]
        for element in elements
    }
    with output_vasp.open("w") as handle:
        handle.write("### paper-style PDMS amorphous model ###\n")
        handle.write("1.0\n")
        handle.write(f"{box_length:.12f} 0.0 0.0\n")
        handle.write(f"0.0 {box_length:.12f} 0.0\n")
        handle.write(f"0.0 0.0 {box_length:.12f}\n")
        handle.write(" ".join(elements) + "\n")
        handle.write(" ".join(str(len(atoms_by_element[element])) for element in elements) + "\n")
        handle.write("Cartesian\n")
        for element in elements:
            for atom in atoms_by_element[element]:
                handle.write(f"{atom.x:14.8f} {atom.y:14.8f} {atom.z:14.8f}\n")


def write_unparametrized_data(
    topology: Topology,
    output_data: str | Path,
    box_length: float,
) -> None:
    """Write the unparameterized LAMMPS topology used as box metadata."""
    output_data = Path(output_data)
    type_ids = {"C": 1, "H": 2, "O": 3, "Si": 4}
    masses = {"C": 12.011, "H": 1.008, "O": 15.999, "Si": 28.086}
    with output_data.open("w") as handle:
        handle.write("### paper-style PDMS topology from PSP Packmol ###\n")
        handle.write(f"{len(topology.atoms)} atoms\n")
        handle.write(f"{len(topology.bonds)} bonds\n")
        handle.write(f"{len(type_ids)} atom types\n")
        handle.write(f"0.0 {box_length:.12f} xlo xhi\n")
        handle.write(f"0.0 {box_length:.12f} ylo yhi\n")
        handle.write(f"0.0 {box_length:.12f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        for element, type_id in sorted(type_ids.items(), key=lambda item: item[1]):
            handle.write(f"{type_id} {masses[element]:.6f} # {element}\n")
        handle.write("\nAtoms\n\n")
        for atom_id in sorted(topology.atoms):
            atom = topology.atoms[atom_id]
            handle.write(
                f"{atom_id:8d} {type_ids[atom.element]:4d} 0.0 "
                f"{atom.x:14.6f} {atom.y:14.6f} {atom.z:14.6f}\n"
            )
        handle.write("\nBonds\n\n")
        for bond_id, (atom1, atom2) in enumerate(topology.bonds, start=1):
            handle.write(f"{bond_id:8d} 1 {atom1:8d} {atom2:8d}\n")


def build_amorphous_pdms_oplsaa(
    output_dir: str | Path,
    monomers: int = 10,
    chain_count: int = 10,
    density_g_cm3: float = 0.97,
    output_lammps: str | None = "pdms_oplsaa.lammps",
    out_file: str = "amor_model",
    title: str = "PSP AmorphousBuilder paper-style PDMS OPLS-AA LAMMPS data",
    packmol_exec: str | Path | None = None,
    random_seed: int = 20260503,
    molecule_dir: str | Path | None = None,
    packmol_dir: str | Path | None = None,
    neutralize: bool = False,
) -> dict:
    """Build paper-style amorphous PDMS and optionally parameterize it."""
    output_dir = Path(output_dir)
    molecule_dir = (
        Path(molecule_dir) if molecule_dir is not None else output_dir / "molecules"
    )
    packmol_dir = (
        Path(packmol_dir) if packmol_dir is not None else output_dir / "packmol"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    molecule_dir.mkdir(parents=True, exist_ok=True)
    packmol_dir.mkdir(parents=True, exist_ok=True)

    molecule_pdb = molecule_dir / f"PDMS_paper_N{monomers}_C1.pdb"
    packmol_pdb = packmol_dir / "packmol.pdb"
    output_vasp = output_dir / f"{out_file}.vasp"
    output_data = output_dir / f"{out_file}.data"
    output_lammps_path = output_dir / output_lammps if output_lammps is not None else None

    mol = build_chain_pdb(monomers, molecule_pdb, random_seed=random_seed)
    box_length = cubic_box_length_angstrom(mol, chain_count, density_g_cm3)
    run_packmol(
        molecule_pdb,
        packmol_pdb,
        chain_count,
        box_length,
        packmol_dir,
        packmol_exec=packmol_exec,
    )

    topology = parse_pdb(packmol_pdb)
    write_poscar(topology, output_vasp, box_length)
    write_unparametrized_data(topology, output_data, box_length)
    if output_lammps_path is None:
        adjacency = adjacency_from_bonds(topology.bonds)
        roles = assign_roles(topology, adjacency)
        angles, dihedrals, unsupported = validate_terms(topology, adjacency)
        if any(unsupported.values()):
            unsupported_text = ", ".join(
                f"{kind}: {count}" for kind, count in sorted(unsupported.items())
            )
            raise ValueError(
                "Unsupported terms for the SI Table S4 PDMS OPLS-AA model. "
                f"Counts by kind: {unsupported_text}"
            )
        total_charge = 0.0
        for atom_id, atom in topology.atoms.items():
            atom_type = ATOM_TYPES[atom.element]
            total_charge += atom_type[f"charge_{roles[atom_id]}"]
        stats = {
            "output": None,
            "atoms": len(topology.atoms),
            "bonds": len(topology.bonds),
            "angles": len(angles),
            "dihedrals": len(dihedrals),
            "atom_types": len(ATOM_TYPES),
            "bond_types": len(BOND_TYPES),
            "angle_types": len(ANGLE_TYPES),
            "dihedral_types": len(DIHEDRAL_TYPES),
            "total_charge": total_charge,
        }
    else:
        stats = write_lammps_data(
            packmol_pdb,
            output_lammps_path,
            box_data=output_data,
            title=title,
            neutralize=neutralize,
        )
    stats.update(
        {
            "smiles": paper_pdms_smiles(monomers),
            "molecule_pdb": str(molecule_pdb),
            "packed_pdb": str(packmol_pdb),
            "vasp": str(output_vasp),
            "unparameterized_data": str(output_data),
            "box_length": box_length,
            "density_g_cm3": density_g_cm3,
            "monomers": monomers,
            "chain_count": chain_count,
        }
    )
    return stats


def normalize_element(raw: str) -> str:
    value = raw.strip()
    if not value:
        return value
    value = re.sub(r"[^A-Za-z]", "", value)
    if value.upper() == "SI":
        return "Si"
    if value:
        return value[0].upper() + value[1:].lower()
    return value


def parse_pdb(pdb_file: str | Path) -> Topology:
    atoms: dict[int, AtomRecord] = {}
    bonds: set[tuple[int, int]] = set()
    box = None
    pdb_file = Path(pdb_file)

    with pdb_file.open() as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "CRYST1":
                try:
                    xhi = float(line[6:15])
                    yhi = float(line[15:24])
                    zhi = float(line[24:33])
                    box = (0.0, xhi, 0.0, yhi, 0.0, zhi)
                except ValueError:
                    box = None
            elif record in {"ATOM", "HETATM"}:
                atom_id = int(line[6:11])
                name = line[12:16].strip()
                element = normalize_element(line[76:78] or name)
                atoms[atom_id] = AtomRecord(
                    atom_id=atom_id,
                    name=name,
                    element=element,
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                )
            elif record == "CONECT":
                fields = [line[6:11]]
                fields.extend(line[i : i + 5] for i in range(11, len(line), 5))
                ids = [int(field) for field in fields if field.strip().isdigit()]
                if len(ids) > 1:
                    root = ids[0]
                    for neighbor in ids[1:]:
                        if root in atoms and neighbor in atoms:
                            bonds.add(tuple(sorted((root, neighbor))))

    if not atoms:
        raise ValueError(f"No atoms found in {pdb_file}")
    if not bonds:
        raise ValueError(f"No CONECT bonds found in {pdb_file}")

    unsupported = sorted({atom.element for atom in atoms.values()} - set(ATOM_TYPES))
    if unsupported:
        raise ValueError(f"Unsupported elements in {pdb_file}: {unsupported}")

    return Topology(atoms=atoms, bonds=sorted(bonds), box=box)


def read_lammps_box(data_file: str | Path | None):
    if data_file is None:
        return None
    data_file = Path(data_file)
    if not data_file.exists():
        return None

    bounds = {}
    with data_file.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 4 and parts[2:4] == ["xlo", "xhi"]:
                bounds["x"] = (float(parts[0]), float(parts[1]))
            elif len(parts) >= 4 and parts[2:4] == ["ylo", "yhi"]:
                bounds["y"] = (float(parts[0]), float(parts[1]))
            elif len(parts) >= 4 and parts[2:4] == ["zlo", "zhi"]:
                bounds["z"] = (float(parts[0]), float(parts[1]))
    if {"x", "y", "z"} <= set(bounds):
        return (*bounds["x"], *bounds["y"], *bounds["z"])
    return None


def infer_box(topology: Topology, box_data: str | Path | None = None, padding: float = 2.0):
    explicit_box = read_lammps_box(box_data)
    if explicit_box is not None:
        return explicit_box
    if topology.box is not None:
        return topology.box

    xs = [atom.x for atom in topology.atoms.values()]
    ys = [atom.y for atom in topology.atoms.values()]
    zs = [atom.z for atom in topology.atoms.values()]
    return (
        min(xs) - padding,
        max(xs) + padding,
        min(ys) - padding,
        max(ys) + padding,
        min(zs) - padding,
        max(zs) + padding,
    )


def adjacency_from_bonds(bonds: list[tuple[int, int]]):
    adjacency = defaultdict(set)
    for atom1, atom2 in bonds:
        adjacency[atom1].add(atom2)
        adjacency[atom2].add(atom1)
    return adjacency


def connected_components(atom_ids: list[int], adjacency):
    seen = set()
    components = []
    for atom_id in sorted(atom_ids):
        if atom_id in seen:
            continue
        queue = deque([atom_id])
        seen.add(atom_id)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def assign_roles(topology: Topology, adjacency):
    atoms = topology.atoms

    end_silicon = set()
    for atom_id, atom in atoms.items():
        if atom.element != "Si":
            continue
        oxygen_neighbors = [
            nbr for nbr in adjacency[atom_id] if atoms[nbr].element == "O"
        ]
        has_hydroxyl_oxygen = any(
            any(atoms[nbr2].element == "H" for nbr2 in adjacency[oxygen])
            for oxygen in oxygen_neighbors
        )
        if len(oxygen_neighbors) <= 1 or has_hydroxyl_oxygen:
            end_silicon.add(atom_id)

    roles = {}
    for atom_id, atom in atoms.items():
        if atom.element == "Si":
            roles[atom_id] = "end" if atom_id in end_silicon else "inside"
        elif atom.element == "C":
            roles[atom_id] = (
                "end"
                if any(nbr in end_silicon for nbr in adjacency[atom_id])
                else "inside"
            )
        elif atom.element == "O":
            silicon_neighbors = [
                nbr for nbr in adjacency[atom_id] if atoms[nbr].element == "Si"
            ]
            roles[atom_id] = (
                "end"
                if len(silicon_neighbors) < 2
                or any(nbr in end_silicon for nbr in silicon_neighbors)
                else "inside"
            )
        elif atom.element == "H":
            heavy_neighbors = [
                nbr for nbr in adjacency[atom_id] if atoms[nbr].element != "H"
            ]
            roles[atom_id] = (
                "end"
                if heavy_neighbors and roles.get(heavy_neighbors[0]) == "end"
                else "inside"
            )
    return roles


def bond_type_for(elements: tuple[str, str]):
    key = tuple(sorted(elements))
    for candidate, params in BOND_TYPES.items():
        if tuple(sorted(candidate)) == key:
            return params
    return None


def angle_type_for(elements: tuple[str, str, str]):
    left, center, right = elements
    normalized = (left, center, right)
    reverse = (right, center, left)
    return ANGLE_TYPES.get(normalized) or ANGLE_TYPES.get(reverse)


def dihedral_type_for(elements: tuple[str, str, str, str]):
    return DIHEDRAL_TYPES.get(elements) or DIHEDRAL_TYPES.get(tuple(reversed(elements)))


def enumerate_angles(topology: Topology, adjacency):
    angles = []
    unsupported = defaultdict(int)
    for center in sorted(topology.atoms):
        neighbors = sorted(adjacency[center])
        for i, atom1 in enumerate(neighbors):
            for atom3 in neighbors[i + 1 :]:
                elements = (
                    topology.atoms[atom1].element,
                    topology.atoms[center].element,
                    topology.atoms[atom3].element,
                )
                params = angle_type_for(elements)
                if params is None:
                    unsupported["-".join(elements)] += 1
                else:
                    angles.append((params["id"], atom1, center, atom3))
    return angles, unsupported


def enumerate_dihedrals(topology: Topology, adjacency):
    candidates = set()
    for atom2, neighbors2 in adjacency.items():
        for atom1 in neighbors2:
            for atom3 in neighbors2:
                if atom1 == atom3:
                    continue
                for atom4 in adjacency[atom3]:
                    if atom4 == atom2 or atom4 == atom1:
                        continue
                    forward = (atom1, atom2, atom3, atom4)
                    reverse = tuple(reversed(forward))
                    candidates.add(min(forward, reverse))

    dihedrals = []
    unsupported = defaultdict(int)
    for atom1, atom2, atom3, atom4 in sorted(candidates):
        elements = (
            topology.atoms[atom1].element,
            topology.atoms[atom2].element,
            topology.atoms[atom3].element,
            topology.atoms[atom4].element,
        )
        params = dihedral_type_for(elements)
        if params is None:
            unsupported["-".join(elements)] += 1
        else:
            dihedrals.append((params["id"], atom1, atom2, atom3, atom4))
    return dihedrals, unsupported


def validate_terms(topology: Topology, adjacency):
    unsupported_bonds = defaultdict(int)
    for atom1, atom2 in topology.bonds:
        elements = (topology.atoms[atom1].element, topology.atoms[atom2].element)
        if bond_type_for(elements) is None:
            unsupported_bonds["-".join(elements)] += 1

    angles, unsupported_angles = enumerate_angles(topology, adjacency)
    dihedrals, unsupported_dihedrals = enumerate_dihedrals(topology, adjacency)
    unsupported = {
        "bonds": dict(unsupported_bonds),
        "angles": dict(unsupported_angles),
        "dihedrals": dict(unsupported_dihedrals),
    }
    return angles, dihedrals, unsupported


def write_lammps_data(
    pdb_file: str | Path,
    output_file: str | Path,
    box_data: str | Path | None = None,
    title: str = "PDMS OPLS-AA LAMMPS data",
    strict: bool = True,
    neutralize: bool = False,
):
    topology = parse_pdb(pdb_file)
    adjacency = adjacency_from_bonds(topology.bonds)
    roles = assign_roles(topology, adjacency)
    angles, dihedrals, unsupported = validate_terms(topology, adjacency)

    if strict and any(unsupported.values()):
        details = "; ".join(
            f"{section}: {values}" for section, values in unsupported.items() if values
        )
        raise ValueError(
            "Unsupported terms for the SI Table S4 PDMS OPLS-AA model. "
            f"{details}"
        )

    components = connected_components(list(topology.atoms), adjacency)
    molecule_ids = {}
    for molecule_id, component in enumerate(components, start=1):
        for atom_id in component:
            molecule_ids[atom_id] = molecule_id

    xlo, xhi, ylo, yhi, zlo, zhi = infer_box(topology, box_data)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    atom_charges = {}
    role_counts = defaultdict(int)
    for atom_id, atom in topology.atoms.items():
        role = roles[atom_id]
        role_counts[(atom.element, role)] += 1
        atom_type = ATOM_TYPES[atom.element]
        atom_charges[atom_id] = atom_type[f"charge_{role}"]

    raw_total_charge = sum(atom_charges.values())
    charge_correction = 0.0
    if neutralize and atom_charges:
        charge_correction = -raw_total_charge / len(atom_charges)
        atom_charges = {
            atom_id: charge + charge_correction
            for atom_id, charge in atom_charges.items()
        }
    total_charge = sum(atom_charges.values())

    with output_file.open("w") as handle:
        handle.write(f"{title}\n")
        handle.write("# units real\n")
        handle.write("# atom_style full\n")
        handle.write("# pair_style lj/cut/coul/long 11.0 11.0\n")
        handle.write("# pair_modify mix geometric\n")
        handle.write("# bond_style harmonic\n")
        handle.write("# angle_style harmonic\n")
        handle.write("# dihedral_style opls\n")
        handle.write("# parameters: JPCB 2025 PDMS benchmark SI Table S4\n\n")
        if neutralize:
            handle.write(
                "# charges neutralized by adding "
                f"{charge_correction:.12f} e to every atom; "
                f"raw rounded total charge {raw_total_charge:.12f} e\n\n"
            )

        handle.write(f"{len(topology.atoms)} atoms\n")
        handle.write(f"{len(topology.bonds)} bonds\n")
        handle.write(f"{len(angles)} angles\n")
        handle.write(f"{len(dihedrals)} dihedrals\n")
        handle.write("0 impropers\n\n")

        handle.write(f"{len(ATOM_TYPES)} atom types\n")
        handle.write(f"{len(BOND_TYPES)} bond types\n")
        handle.write(f"{len(ANGLE_TYPES)} angle types\n")
        handle.write(f"{len(DIHEDRAL_TYPES)} dihedral types\n")
        handle.write("0 improper types\n\n")

        handle.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        handle.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        handle.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")

        handle.write("Masses\n\n")
        for element, params in sorted(ATOM_TYPES.items(), key=lambda item: item[1]["id"]):
            handle.write(
                f"{params['id']:3d} {params['mass']:10.5f} # {params['label']} ({element})\n"
            )

        handle.write("\nPair Coeffs\n\n")
        for _element, params in sorted(ATOM_TYPES.items(), key=lambda item: item[1]["id"]):
            handle.write(
                f"{params['id']:3d} {params['epsilon']:10.6f} {params['sigma']:10.6f} # {params['label']}\n"
            )

        handle.write("\nBond Coeffs\n\n")
        for _key, params in sorted(BOND_TYPES.items(), key=lambda item: item[1]["id"]):
            handle.write(
                f"{params['id']:3d} {params['k']:10.6f} {params['r0']:10.6f} # {params['label']}\n"
            )

        handle.write("\nAngle Coeffs\n\n")
        for _key, params in sorted(ANGLE_TYPES.items(), key=lambda item: item[1]["id"]):
            handle.write(
                f"{params['id']:3d} {params['k']:10.6f} {params['theta0']:10.6f} # {params['label']}\n"
            )

        handle.write("\nDihedral Coeffs\n\n")
        for _key, params in sorted(DIHEDRAL_TYPES.items(), key=lambda item: item[1]["id"]):
            k1, k2, k3, k4 = params["coeffs"]
            handle.write(
                f"{params['id']:3d} {k1:10.6f} {k2:10.6f} {k3:10.6f} {k4:10.6f} # {params['label']}\n"
            )

        handle.write("\nAtoms\n\n")
        for atom_id in sorted(topology.atoms):
            atom = topology.atoms[atom_id]
            atom_type = ATOM_TYPES[atom.element]
            handle.write(
                f"{atom_id:8d} {molecule_ids[atom_id]:6d} {atom_type['id']:4d} "
                f"{atom_charges[atom_id]:18.12f} "
                f"{atom.x:14.6f} {atom.y:14.6f} {atom.z:14.6f}\n"
            )

        handle.write("\nBonds\n\n")
        for bond_id, (atom1, atom2) in enumerate(topology.bonds, start=1):
            elements = (topology.atoms[atom1].element, topology.atoms[atom2].element)
            params = bond_type_for(elements)
            if params is not None:
                handle.write(f"{bond_id:8d} {params['id']:4d} {atom1:8d} {atom2:8d}\n")

        handle.write("\nAngles\n\n")
        for angle_id, (angle_type, atom1, atom2, atom3) in enumerate(angles, start=1):
            handle.write(
                f"{angle_id:8d} {angle_type:4d} {atom1:8d} {atom2:8d} {atom3:8d}\n"
            )

        handle.write("\nDihedrals\n\n")
        for dihedral_id, (dihedral_type, atom1, atom2, atom3, atom4) in enumerate(
            dihedrals, start=1
        ):
            handle.write(
                f"{dihedral_id:8d} {dihedral_type:4d} {atom1:8d} {atom2:8d} {atom3:8d} {atom4:8d}\n"
            )
        handle.write("\n\n")

    return {
        "output": str(output_file),
        "atoms": len(topology.atoms),
        "bonds": len(topology.bonds),
        "angles": len(angles),
        "dihedrals": len(dihedrals),
        "molecules": len(components),
        "total_charge": total_charge,
        "raw_total_charge": raw_total_charge,
        "neutralized": neutralize,
        "neutralization_correction_per_atom": charge_correction,
        "role_counts": dict(role_counts),
        "unsupported": unsupported,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_file")
    parser.add_argument("output_file")
    parser.add_argument("--box-data", default=None)
    parser.add_argument("--title", default="PDMS OPLS-AA LAMMPS data")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument(
        "--neutralize",
        action="store_true",
        help="Redistribute the rounded total charge uniformly over all atoms.",
    )
    args = parser.parse_args(argv)

    stats = write_lammps_data(
        args.pdb_file,
        args.output_file,
        box_data=args.box_data,
        title=args.title,
        strict=not args.non_strict,
        neutralize=args.neutralize,
    )
    print(f"Wrote {stats['output']}")
    print(
        "Atoms: {atoms}, bonds: {bonds}, angles: {angles}, dihedrals: {dihedrals}, "
        "molecules: {molecules}, total charge: {total_charge:.6f}".format(**stats)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
