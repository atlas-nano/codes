"""Campaign PVDF typed chemistry; unconfined geometry is supplied by the public builder.
Origin SHA256: f8c91c767eb9721a00c97e437567b325d2dca351bee72518b2da7fcdf84141f4
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import random
import math
N_REPEAT_PAIRS=9
MASS = {
    1: (12.011, "opls_135 terminal CH3 carbon"),
    2: (12.011, "opls_136 internal CH2 carbon"),
    3: (1.008, "opls_140 aliphatic hydrogen"),
    4: (12.011, "opls_961 terminal CF3 carbon"),
    5: (12.011, "opls_962 internal CF2 carbon"),
    6: (18.998, "opls_965 fluorine"),
}

CHARGE = {1: -0.18, 2: -0.12, 3: 0.06, 4: 0.36, 5: 0.24, 6: -0.12}

PAIR = {
    1: (0.06600, 3.50000),
    2: (0.06600, 3.50000),
    3: (0.03000, 2.50000),
    4: (0.06600, 3.50000),
    5: (0.06600, 3.50000),
    6: (0.05300, 2.95000),
}

BOND_COEFF = {
    1: (340.00000, 1.09000, "opls_135 opls_140"),
    2: (268.00000, 1.52900, "opls_135 opls_962"),
    3: (340.00000, 1.09000, "opls_136 opls_140"),
    4: (268.00000, 1.52900, "opls_136 opls_961"),
    5: (268.00000, 1.52900, "opls_136 opls_962"),
    6: (367.00000, 1.33200, "opls_961 opls_965"),
    7: (367.00000, 1.33200, "opls_962 opls_965"),
}

BOND_TYPE = {
    (1, 3): 1,
    (1, 5): 2,
    (2, 3): 3,
    (2, 4): 4,
    (2, 5): 5,
    (4, 6): 6,
    (5, 6): 7,
}

ANGLE_COEFF = {
    1: (33.00000, 107.80000, "opls_140 opls_135 opls_140"),
    2: (37.50000, 110.70000, "opls_962 opls_135 opls_140"),
    3: (33.00000, 107.80000, "opls_140 opls_136 opls_140"),
    4: (37.50000, 110.70000, "opls_961 opls_136 opls_140"),
    5: (37.50000, 110.70000, "opls_962 opls_136 opls_140"),
    6: (58.35000, 112.70000, "opls_962 opls_136 opls_961"),
    7: (58.35000, 112.70000, "opls_962 opls_136 opls_962"),
    8: (50.00000, 109.50000, "opls_136 opls_961 opls_965"),
    9: (77.00000, 109.10000, "opls_965 opls_961 opls_965"),
    10: (58.35000, 112.70000, "opls_136 opls_962 opls_135"),
    11: (50.00000, 109.50000, "opls_135 opls_962 opls_965"),
    12: (58.35000, 112.70000, "opls_136 opls_962 opls_136"),
    13: (50.00000, 109.50000, "opls_136 opls_962 opls_965"),
    14: (77.00000, 109.10000, "opls_965 opls_962 opls_965"),
}

ANGLE_TYPE = {
    (1, (3, 3)): 1,
    (1, (3, 5)): 2,
    (2, (3, 3)): 3,
    (2, (3, 4)): 4,
    (2, (3, 5)): 5,
    (2, (4, 5)): 6,
    (2, (5, 5)): 7,
    (4, (2, 6)): 8,
    (4, (6, 6)): 9,
    (5, (1, 2)): 10,
    (5, (1, 6)): 11,
    (5, (2, 2)): 12,
    (5, (2, 6)): 13,
    (5, (6, 6)): 14,
}

DIHEDRAL_COEFF = {
    1: (0.00000, -0.00000, 0.30000, -0.00000, "opls_140 opls_135 opls_962 opls_136"),
    2: (0.00000, -0.00000, 0.40000, -0.00000, "opls_140 opls_135 opls_962 opls_965"),
    3: (0.00000, -0.00000, 0.40000, -0.00000, "opls_140 opls_136 opls_961 opls_965"),
    4: (0.30000, -0.00000, 0.40000, -0.00000, "opls_962 opls_136 opls_961 opls_965"),
    5: (0.00000, -0.00000, 0.30000, -0.00000, "opls_140 opls_136 opls_962 opls_135"),
    6: (0.00000, -0.00000, 0.30000, -0.00000, "opls_140 opls_136 opls_962 opls_136"),
    7: (0.00000, -0.00000, 0.40000, -0.00000, "opls_140 opls_136 opls_962 opls_965"),
    8: (1.30000, -0.05000, 0.20000, -0.00000, "opls_961 opls_136 opls_962 opls_136"),
    9: (0.30000, -0.00000, 0.40000, -0.00000, "opls_961 opls_136 opls_962 opls_965"),
    10: (1.30000, -0.05000, 0.20000, -0.00000, "opls_962 opls_136 opls_962 opls_135"),
    11: (1.30000, -0.05000, 0.20000, -0.00000, "opls_962 opls_136 opls_962 opls_136"),
    12: (0.30000, -0.00000, 0.40000, -0.00000, "opls_962 opls_136 opls_962 opls_965"),
}

DIHEDRAL_TYPE = {
    (3, 1, 5, 2): 1,
    (3, 1, 5, 6): 2,
    (3, 2, 4, 6): 3,
    (5, 2, 4, 6): 4,
    (3, 2, 5, 1): 5,
    (3, 2, 5, 2): 6,
    (3, 2, 5, 6): 7,
    (4, 2, 5, 2): 8,
    (4, 2, 5, 6): 9,
    (5, 2, 5, 1): 10,
    (5, 2, 5, 2): 11,
    (5, 2, 5, 6): 12,
}

DIHEDRAL_TYPE.update({tuple(reversed(k)): v for k, v in list(DIHEDRAL_TYPE.items())})

@dataclass
class Atom:
    atom_id: int
    mol: int
    typ: int
    x: float
    y: float
    z: float
    name: str
    element: str

def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)

def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def vcross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

def vnorm(a):
    mag = math.sqrt(vdot(a, a))
    if mag < 1.0e-12:
        return (1.0, 0.0, 0.0)
    return (a[0] / mag, a[1] / mag, a[2] / mag)

def random_unit(rng: random.Random):
    z = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r * math.cos(theta), r * math.sin(theta), z)

def orthonormal_basis(axis):
    axis = vnorm(axis)
    ref = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.85 else (0.0, 1.0, 0.0)
    e1 = vnorm(vcross(axis, ref))
    e2 = vnorm(vcross(axis, e1))
    return e1, e2

def carbon_type_sequence(n_repeat_pairs):
    seq = [1]
    for _ in range(n_repeat_pairs):
        seq.extend([5, 2])
    seq.append(4)
    return seq

def build_atoms(chain_centers, box_length, seed):
    atoms = []
    chain_atom_ids = []
    carbon_ids_by_chain = []
    atom_id = 1
    seq = carbon_type_sequence(N_REPEAT_PAIRS)
    for mol, center in enumerate(chain_centers, start=1):
        rng = random.Random(seed + mol * 9973)
        backbone = generate_backbone(len(seq), center, rng, box_length)
        carbon_ids = []
        this_chain = []
        for idx, typ in enumerate(seq):
            carbon_ids.append(atom_id)
            if typ in {1, 2}:
                name, element = "C", "C"
            else:
                name, element = "CF", "C"
            x, y, z = backbone[idx]
            atoms.append(Atom(atom_id, mol, typ, x, y, z, name, element))
            this_chain.append(atom_id)
            atom_id += 1

            if typ in {1, 2}:
                sub_typ = 3
                sub_count = 3 if typ == 1 else 2
                sub_len = 1.090
                sub_name = "H"
                sub_element = "H"
            else:
                sub_typ = 6
                sub_count = 3 if typ == 4 else 2
                sub_len = 1.332
                sub_name = "F"
                sub_element = "F"
            for direction in substituent_dirs(backbone, idx, sub_count, rng):
                sx, sy, sz = vadd((x, y, z), vscale(direction, sub_len))
                atoms.append(Atom(atom_id, mol, sub_typ, sx, sy, sz, sub_name, sub_element))
                this_chain.append(atom_id)
                atom_id += 1
        chain_atom_ids.append(this_chain)
        carbon_ids_by_chain.append(carbon_ids)
    return atoms, chain_atom_ids, carbon_ids_by_chain

def build_topology(atoms, carbon_ids_by_chain):
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    neighbors = {atom.atom_id: set() for atom in atoms}
    bonds = []
    bond_id = 1
    for carbon_ids in carbon_ids_by_chain:
        for a, b in zip(carbon_ids, carbon_ids[1:]):
            typ = BOND_TYPE[tuple(sorted((atom_by_id[a].typ, atom_by_id[b].typ)))]
            bonds.append((bond_id, typ, a, b))
            neighbors[a].add(b)
            neighbors[b].add(a)
            bond_id += 1
    carbon_set = {cid for chain in carbon_ids_by_chain for cid in chain}
    for atom in atoms:
        if atom.atom_id in carbon_set:
            continue
        carbon = max(i for i in range(atom.atom_id - 1, 0, -1) if i in carbon_set)
        typ = BOND_TYPE[tuple(sorted((atom_by_id[carbon].typ, atom.typ)))]
        bonds.append((bond_id, typ, carbon, atom.atom_id))
        neighbors[carbon].add(atom.atom_id)
        neighbors[atom.atom_id].add(carbon)
        bond_id += 1

    angles = []
    angle_id = 1
    for center in sorted(neighbors):
        neigh = sorted(neighbors[center])
        for i, a in enumerate(neigh):
            for c in neigh[i + 1 :]:
                key = (atom_by_id[center].typ, tuple(sorted((atom_by_id[a].typ, atom_by_id[c].typ))))
                angles.append((angle_id, ANGLE_TYPE[key], a, center, c))
                angle_id += 1

    dihedrals = []
    dihedral_id = 1
    seen_bonds = set()
    for _, _, b, c in bonds:
        key_bond = tuple(sorted((b, c)))
        if key_bond in seen_bonds:
            continue
        seen_bonds.add(key_bond)
        for a in sorted(neighbors[b] - {c}):
            for d in sorted(neighbors[c] - {b}):
                key = tuple(atom_by_id[i].typ for i in (a, b, c, d))
                typ = DIHEDRAL_TYPE.get(key)
                if typ is None:
                    raise ValueError(f"Unsupported dihedral type {key} for {(a, b, c, d)}")
                dihedrals.append((dihedral_id, typ, a, b, c, d))
                dihedral_id += 1
    return bonds, angles, dihedrals

def write_pdb(path, atoms, atom_ids=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = atoms if atom_ids is None else [atoms[i - 1] for i in atom_ids]
    with path.open("w") as handle:
        handle.write("REMARK PVDF 1600x2 PSP-style OPLS-AA starting coordinates\n")
        for serial, atom in enumerate(selected, start=1):
            handle.write(
                f"ATOM  {serial:5d} {atom.name:<4s} PVF A{atom.mol:4d}    "
                f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}  1.00  0.00          {atom.element:>2s}\n"
            )
        handle.write("END\n")

def write_data(path, atoms, bonds, angles, dihedrals, box_length):
    with path.open("w") as handle:
        handle.write("PVDF_1600x2 PSP-style amorphous OPLS-AA data generated by DMAx2\n\n")
        handle.write(f"{len(atoms):12d} atoms\n")
        handle.write(f"{len(bonds):12d} bonds\n")
        handle.write(f"{len(angles):12d} angles\n")
        handle.write(f"{len(dihedrals):12d} dihedrals\n")
        handle.write("           0 impropers\n\n")
        handle.write("           6 atom types\n")
        handle.write("           7 bond types\n")
        handle.write("          14 angle types\n")
        handle.write("          12 dihedral types\n")
        handle.write("           0 improper types\n\n")
        handle.write(f"{0.0:12.6f} {box_length:12.6f} xlo xhi\n")
        handle.write(f"{0.0:12.6f} {box_length:12.6f} ylo yhi\n")
        handle.write(f"{0.0:12.6f} {box_length:12.6f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        for typ in sorted(MASS):
            mass, label = MASS[typ]
            handle.write(f"{typ:5d} {mass:12.6f} # {label}\n")
        handle.write("\nPair Coeffs # 4*epsilon*(-sigma**6/r**6 + sigma**12/r**12)\n\n")
        for typ in sorted(PAIR):
            eps, sigma = PAIR[typ]
            handle.write(f"{typ:5d} {eps:12.5f} {sigma:12.5f} # {MASS[typ][1]}\n")
        handle.write("\nBond Coeffs # LAMMPSHarmonicBondPotential\n\n")
        for typ in sorted(BOND_COEFF):
            k, r0, label = BOND_COEFF[typ]
            handle.write(f"{typ:5d} {k:12.5f} {r0:12.5f} # {label}\n")
        handle.write("\nAngle Coeffs # LAMMPSHarmonicAnglePotential\n\n")
        for typ in sorted(ANGLE_COEFF):
            k, theta, label = ANGLE_COEFF[typ]
            handle.write(f"{typ:5d} {k:12.5f} {theta:12.5f} # {label}\n")
        handle.write("\nDihedral Coeffs # OPLSTorsionPotential\n\n")
        for typ in sorted(DIHEDRAL_COEFF):
            k1, k2, k3, k4, label = DIHEDRAL_COEFF[typ]
            handle.write(f"{typ:5d} {k1:12.5f} {k2:12.5f} {k3:12.5f} {k4:12.5f} # {label}\n")
        handle.write("\nAtoms # full\n\n")
        for atom in atoms:
            handle.write(
                f"{atom.atom_id:8d} {atom.mol:8d} {atom.typ:5d} {CHARGE[atom.typ]:12.8f} "
                f"{atom.x:14.6f} {atom.y:14.6f} {atom.z:14.6f}\n"
            )
        handle.write("\nBonds\n\n")
        for term in bonds:
            handle.write(f"{term[0]:8d} {term[1]:5d} {term[2]:8d} {term[3]:8d}\n")
        handle.write("\nAngles\n\n")
        for term in angles:
            handle.write(f"{term[0]:8d} {term[1]:5d} {term[2]:8d} {term[3]:8d} {term[4]:8d}\n")
        handle.write("\nDihedrals\n\n")
        for term in dihedrals:
            handle.write(
                f"{term[0]:8d} {term[1]:5d} {term[2]:8d} {term[3]:8d} {term[4]:8d} {term[5]:8d}\n"
            )
