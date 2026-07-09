# -*- coding: utf-8 -*-
"""
System builder and trajectory I/O layer for py-xPT

MDAnalysis ``Universe`` objects handle every supported file format —
LAMMPS, AMBER, CHARMM/DCD, GROMACS, XYZ, PDB, and many more — through a
single, consistent API.  This module wraps the Universe to:

  1. Build the atomic topology (atoms, masses, bonds, molecules, groups).
  2. Provide a clean iterator that yields per-frame velocity and position
     arrays as plain NumPy arrays, so the compute layer never touches
     MDAnalysis internals directly.
  3. Read group definitions from the optional group file.

Fallback mode
-------------
If MDAnalysis is not installed, a lightweight pure-NumPy LAMMPS-dump
reader is used instead.  This covers the most common production use-case
and allows the package to run without MDAnalysis for basic workflows.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from pyxpt.config import Config
from pyxpt.constants import ELEMENTS, R as GAS_R

log = logging.getLogger(__name__)

# ── Optional MDAnalysis import ────────────────────────────────────────────────

try:
    import MDAnalysis as mda                        # type: ignore[import]
    from MDAnalysis.analysis import distances        # type: ignore[import]
    _HAS_MDA = True
    # Teach MDAnalysis about AMOEBA-specific LAMMPS DATA sections (e.g.
    # "Tinker Types") so they act as section boundaries and don't bleed
    # into the preceding Impropers section causing a parse error.
    try:
        from MDAnalysis.topology.LAMMPSParser import SECTIONS as _MDA_LAMMPS_SECTS
        for _s in ("Tinker Types",):
            _MDA_LAMMPS_SECTS.add(_s)
            _MDA_LAMMPS_SECTS.add(_s.split()[0])   # first-word token
        del _s, _MDA_LAMMPS_SECTS
    except Exception:
        pass
except ImportError:                                  # pragma: no cover
    _HAS_MDA = False
    log.warning(
        "MDAnalysis is not installed.  Install it with:\n"
        "    pip install MDAnalysis\n"
        "Falling back to the built-in LAMMPS-dump reader."
    )

# Native AMBER/NetCDF reader — used for trajectory_format = NCDF / AMBER.
# Bypasses MDAnalysis (which itself uses scipy.io._netcdf and therefore
# cannot read CDF-5 files produced by LAMMPS' `dump netcdf` style).
# Optional non-LAMMPS readers are not shipped in this LAMMPS-focused build.
try:
    from pyxpt.io.amber_ncdf import CDFReader, iter_frames_amber_ncdf  # type: ignore
except ModuleNotFoundError:
    CDFReader = None
    def iter_frames_amber_ncdf(*a, **k):
        raise RuntimeError("Amber NetCDF trajectory support is not available in this build "
                           "(LAMMPS-only). Convert to a LAMMPS dump.")


# AMBER convention says time should be picoseconds, but LAMMPS' `dump netcdf`
# writes time in the simulation's internal time unit (fs for units real,
# ps for units metal, etc.) and sets `units` on the time variable accordingly.
# Convert whatever the file actually carries into pyxpt-internal ps.
_AMBER_TIME_TO_PS: dict[str, float] = {
    "picosecond": 1.0,  "picoseconds": 1.0,  "ps": 1.0,
    "femtosecond": 1e-3, "femtoseconds": 1e-3, "fs": 1e-3,
    "nanosecond":  1e3,  "nanoseconds":  1e3,  "ns": 1e3,
    "second":      1e12, "seconds":      1e12, "s":  1e12,
}


def _decode_units(attrs) -> str:
    """Return a lowercased units string from an AMBER attr dict (bytes or str)."""
    u = attrs.get("units") if attrs else None
    if isinstance(u, (bytes, bytearray)):
        u = u.decode("utf-8", errors="replace")
    return (u or "").strip().lower()


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AtomInfo:
    """
    Lightweight per-atom topology record.

    Velocity and position arrays are *not* stored here — they live in the
    per-frame NumPy arrays produced by :meth:`System.iter_frames`.
    """
    id: int              # 0-based global index
    mass: float          # g/mol
    mol_id: int          # 0-based molecule index
    grp_id: int = 0      # 0-based group index
    charge: float = 0.0  # partial charge (elementary charge units, e)
    type_label: str = "" # atom type label (from topology; used for BPM bond matching)


@dataclass
class MolInfo:
    """Per-molecule topology record."""
    id: int
    atom_ids: list[int] = field(default_factory=list)   # 0-based atom indices
    mass: float = 0.0
    linear: bool = False
    bonds: list = field(default_factory=list)           # global atom-index pairs (i, j)


@dataclass
class GroupInfo:
    """Atom group for independent thermodynamic reporting."""
    id: int
    atom_ids: list[int] = field(default_factory=list)
    mol_ids: list[int] = field(default_factory=list)
    mass: float = 0.0
    rotsym: int = 1
    linear: int = 0
    constraint: int = 0
    volume: float = 0.0        # override volume (Å³), 0 = use system volume
    eng_avg: float = 0.0       # override energy (kcal/mol)
    eng_std: float = 0.0

    @property
    def natom(self) -> int:
        """Number of atoms in this group."""
        return len(self.atom_ids)

    @property
    def nmol(self) -> int:
        """Number of molecules in this group."""
        return len(self.mol_ids)


@dataclass
class FrameData:
    """
    All data extracted from a single trajectory frame.

    Positions and velocities are shaped (natom, 3), dtype float64, units Å
    and Å/ps respectively.  ``box`` is the 3×3 cell matrix H (row = lattice
    vector).
    """
    timestep: float              # ps
    positions: np.ndarray        # (natom, 3) Å
    velocities: np.ndarray       # (natom, 3) Å/ps
    forces: np.ndarray           # (natom, 3) kJ/mol/Å
    stresses: np.ndarray      
    box: np.ndarray              # (3, 3)
    temperature: float = 0.0     # K  (from LAMMPS thermo if available)
    pressure: float = 0.0        # GPa
    volume: float = 0.0          # Å³
    total_energy: float = 0.0    # kcal/mol
    atom_energies: Optional[np.ndarray] = None     # (natom,) kcal/mol
    charges: Optional[np.ndarray] = None            # (natom,) per-frame partial charges (e)
    induced_dipoles: Optional[np.ndarray] = None    # (natom, 3) e·Å  per-atom induced dipole (AMOEBA)


# ── System ────────────────────────────────────────────────────────────────────

class System:
    """
    Unified interface to atomic topology + trajectory.

    Build with :meth:`from_config`; iterate frames with :meth:`iter_frames`.

    Attributes
    ----------
    atoms : list[AtomInfo]
    mols  : list[MolInfo]
    groups: list[GroupInfo]   – always at least one (the whole system)
    natom, nmol, ngrp : int
    periodic : bool
    total_mass : float  g/mol
    """

    def __init__(self) -> None:
        self.atoms: list[AtomInfo] = []
        self.mols: list[MolInfo] = []
        self.groups: list[GroupInfo] = []
        self.natom = self.nmol = self.ngrp = 0
        self.periodic = False
        self.total_mass = 0.0
        self.has_frame_charges: bool = False      # True when trajectory dump contains "q" column
        self.has_induced_dipoles: bool = False    # True when dump contains "f_dipole[1..3]" (AMOEBA)
        self._cfg: Optional[Config] = None
        self._universe: Optional[object] = None   # mda.Universe if available
        self._pe_ke_warned: bool = False          # one-shot guard for duplicate-column warning

    @property
    def has_charges(self) -> bool:
        """True if any atom carries a non-zero partial charge."""
        return any(a.charge != 0.0 for a in self.atoms)

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: Config) -> "System":
        """
        Build a System from a :class:`Config`.

        Uses MDAnalysis when available; falls back to the built-in reader.
        """
        sys_obj = cls()
        sys_obj._cfg = cfg

        # For LAMMPSDUMP trajectories, the built-in data-file + first-frame
        # reader is much faster than MDAnalysis (which scans the entire
        # trajectory file during Universe construction to count frames).
        # Profiling on a 5.9 GB / 10001-frame TIP4P/2005 trajectory showed
        # MDA Universe construction dominated total runtime at 42 %.
        #
        # NCDF/AMBER trajectories go through a native CDF-1/2/5 reader
        # (pyxpt.io.amber_ncdf).  MDA's NCDF path uses scipy.io._netcdf
        # which only supports CDF-1/2, so LAMMPS-produced CDF-5 files
        # would otherwise fail with "ValueError: Unexpected header".
        trj_fmt = (cfg.trajectory_format or "").upper()
        _is_lammps_dump = trj_fmt in ("LAMMPSDUMP", "LAMMPS")
        _is_amber_ncdf  = trj_fmt in ("NCDF", "AMBER", "AMBERNCDF", "NETCDF")

        _is_dlpoly = trj_fmt in ("DLPOLY", "DL_POLY")
        _is_md4d   = trj_fmt in ("MD4D", "RAWVEL")

        if _is_md4d:
            sys_obj._build_from_md4d(cfg)
        elif _is_amber_ncdf:
            sys_obj._build_from_amber_ncdf(cfg)
        elif _is_dlpoly:
            # DL_POLY split trajectory: topology comes from the LAMMPS data file
            # (cfg.topology); the position/velocity files are read by the native
            # DL_POLY iterator, not MDAnalysis.
            sys_obj._build_from_lammps_dump(cfg)
        elif _HAS_MDA and not _is_lammps_dump:
            sys_obj._build_from_mda(cfg)
        else:
            sys_obj._build_from_lammps_dump(cfg)

        sys_obj._build_groups(cfg)
        sys_obj.total_mass = sum(a.mass for a in sys_obj.atoms)
        log.info(
            "%d atoms  %d molecules  %d groups  total mass %.4f g/mol",
            sys_obj.natom, sys_obj.nmol, sys_obj.ngrp, sys_obj.total_mass,
        )
        return sys_obj

    # ── MDAnalysis path ───────────────────────────────────────────────────────

    def _build_from_mda(self, cfg: Config) -> None:
        """Build topology from MDAnalysis Universe."""
        trj_files = cfg.trajectory
        kwargs: dict = {}

        dt = (cfg.dump_freq if cfg.dump_freq > 0 else 1) * cfg.timestep
        kwargs["dt"] = dt
        trj_fmt = cfg.trajectory_format or None
        if cfg.topology:
            top_fmt = cfg.topology_format or None
            if top_fmt:
                kwargs["topology_format"] = top_fmt
            if trj_fmt:
                kwargs["format"] = trj_fmt
            u = mda.Universe(cfg.topology, *trj_files, **kwargs)
        else:
            # Topology-free: let MDAnalysis infer from the trajectory
            if trj_fmt:
                kwargs["format"] = trj_fmt
            u = mda.Universe(*trj_files, **kwargs)

        # For LAMMPS dump trajectories, re-open with additional_columns=True so
        # custom per-atom columns (e.g. v_atomEng, c_pe) are stored in ts.data.
        _lammps_fmt = trj_fmt and trj_fmt.upper() in ("LAMMPSDUMP", "LAMMPS")
        if _lammps_fmt:
            kwargs["additional_columns"] = True
            if cfg.topology:
                u = mda.Universe(cfg.topology, *trj_files, **kwargs)
            else:
                u = mda.Universe(*trj_files, **kwargs)

        self._universe = u
        ag = u.atoms

        # ── Masses ──────────────────────────────────────────────────────────
        try:
            masses = ag.masses.astype(float)
        except (AttributeError, mda.exceptions.NoDataError):
            masses = _infer_masses_from_names(ag.names)
            log.warning("Mass data not in topology; inferred from element symbols.")

        # ── Charges ───────────────────────────────────────────────────────────
        # Prefer per-frame "q" column from LAMMPS dump (fluctuating charge models)
        # over static topology charges.  Peek at the first frame to detect it.
        charges = np.zeros(len(ag))
        _topo_charges_loaded = False
        try:
            charges = ag.charges.astype(float)
            _topo_charges_loaded = True
        except (AttributeError, mda.exceptions.NoDataError):
            pass

        if _lammps_fmt:
            try:
                ts0 = u.trajectory[0]
                ts0_data = getattr(ts0, "data", {}) or {}
                q_arr = ts0_data.get("q", None)
                if q_arr is not None and len(q_arr) == len(ag):
                    charges = np.asarray(q_arr, dtype=float)
                    self.has_frame_charges = True
                    log.info("LAMMPS dump contains 'q' column — using per-frame charges.")
                # Detect AMOEBA induced dipoles.  Preferred column names are
                # ``mux``, ``muy``, ``muz`` (set via LAMMPS ``dump_modify
                # colname``); the legacy ``f_dipole[1..3]`` form is kept as
                # a backward-compatible fallback.
                if all(c in ts0_data for c in ("mux", "muy", "muz")):
                    self.has_induced_dipoles = True
                    log.info("LAMMPS dump contains 'mux/muy/muz' columns — "
                             "AMOEBA induced dipoles detected.")
                elif all(f"f_dipole[{i}]" in ts0_data for i in (1, 2, 3)):
                    self.has_induced_dipoles = True
                    log.info("LAMMPS dump contains legacy 'f_dipole' columns "
                             "— AMOEBA induced dipoles detected (consider "
                             "renaming to mux/muy/muz via dump_modify colname).")
            except Exception:
                pass

        # ── Molecule IDs via bond-connectivity graph ──────────────────────────
        # Build connected components directly from the bond list using union-find,
        # so molecule detection is independent of MDAnalysis internals (resids,
        # fragments).  Falls back gracefully when no bond data exists (monatomic
        # systems), where each atom becomes its own molecule.
        natoms = len(ag)
        parent = list(range(natoms))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        try:
            for bond in u.bonds:
                _union(int(bond.indices[0]), int(bond.indices[1]))
        except (AttributeError, mda.exceptions.NoDataError):
            pass  # no bond data → each atom is its own component

        roots = [_find(i) for i in range(natoms)]
        root_to_mol: dict[int, int] = {}
        mol_ids = np.zeros(natoms, dtype=int)
        for i, r in enumerate(roots):
            if r not in root_to_mol:
                root_to_mol[r] = len(root_to_mol)
            mol_ids[i] = root_to_mol[r]

        # ── Atom type labels ─────────────────────────────────────────────────
        try:
            type_labels = [str(t).strip() for t in ag.types]
        except (AttributeError, mda.exceptions.NoDataError):
            type_labels = [""] * len(ag)

        # ── AtomInfo list ────────────────────────────────────────────────────
        self.atoms = [
            AtomInfo(id=i, mass=float(masses[i]), mol_id=int(mol_ids[i]),
                     charge=float(charges[i]), type_label=type_labels[i])
            for i in range(len(ag))
        ]
        self.natom = len(self.atoms)

        # ── MolInfo list ─────────────────────────────────────────────────────
        nmol = int(mol_ids.max()) + 1
        self.mols = [MolInfo(id=m) for m in range(nmol)]
        for a in self.atoms:
            m = self.mols[a.mol_id]
            m.atom_ids.append(a.id)
            m.mass += a.mass
        self.nmol = nmol

        # ── Bond list per molecule ────────────────────────────────────────────
        # Populate MolInfo.bonds with (global_i, global_j) pairs so that the
        # bond polarizability model (BPM) can look up bond vectors at runtime.
        try:
            for bond in u.bonds:
                ai, aj = int(bond.indices[0]), int(bond.indices[1])
                mid = self.atoms[ai].mol_id
                self.mols[mid].bonds.append((ai, aj))
        except (AttributeError, mda.exceptions.NoDataError):
            pass  # no bond data — BPM will be a no-op
        self.periodic = u.dimensions is not None

    # ── Fallback LAMMPS-dump path ─────────────────────────────────────────────

    def _build_from_lammps_dump(self, cfg: Config) -> None:
        """
        Topology builder from the LAMMPS data file plus first frame of dump.

        Replaces the MDAnalysis path for LAMMPSDUMP runs.  Reads everything
        from the data file (masses, charges, molecule IDs, bond list) — does
        not scan the entire trajectory the way MDAnalysis Universe does.
        Falls back to the legacy "each atom is its own molecule" form when
        no data file is given (rare; mostly LJ test setups).
        """
        # Read first frame of the dump to detect available per-frame columns
        # (q for fluctuating charges, mux/muy/muz for AMOEBA dipoles, etc.).
        dump_path = cfg.trajectory[0]
        if (cfg.trajectory_format or "").upper() in ("DLPOLY", "DL_POLY"):
            # DL_POLY split format: the position file is not a LAMMPS dump, so
            # there is nothing to column-detect here — topology is taken wholly
            # from the LAMMPS data file (Path 1 below).  No per-frame q/dipoles.
            first_frame: dict = {}
        else:
            first_frame = _read_lammps_dump_frame(dump_path, frame_idx=0)
            if first_frame is None:
                raise RuntimeError(f"Cannot read first frame from {dump_path}")

        # Per-frame "q" detection (fluctuating-charge force fields)
        if first_frame.get("q", None) is not None:
            self.has_frame_charges = True
            log.info("LAMMPS dump contains 'q' column — using per-frame charges.")

        # Induced-dipole detection (AMOEBA)
        if all(c in first_frame for c in ("mux", "muy", "muz")):
            self.has_induced_dipoles = True
            log.info("LAMMPS dump contains 'mux/muy/muz' columns — "
                     "AMOEBA induced dipoles detected.")
        elif "f_dipole" in first_frame:
            self.has_induced_dipoles = True
            log.info("LAMMPS dump contains legacy 'f_dipole' columns — "
                     "AMOEBA induced dipoles detected.")

        # ── Path 1: data file present → read full topology ──────────────────
        if cfg.topology and Path(cfg.topology).exists():
            data = _read_lammps_data_full(cfg.topology)
            natom         = data["n_atoms"] if data["n_atoms"] > 0 else len(data["atom_id"])
            mass_map      = data["masses"]
            type_labels   = data["type_labels"]
            atom_id_arr   = data["atom_id"]
            atom_type_arr = data["atom_type"]
            atom_chg_arr  = data["atom_charge"]
            atom_mol_arr  = data["atom_mol_id"]   # 1-based LAMMPS molecule IDs

            # If per-frame q in dump, overwrite static charges with the first
            # frame's value (matches the MDA path's behavior so AtomInfo.charge
            # receives the same initial value).
            initial_charges = atom_chg_arr.copy()
            if self.has_frame_charges:
                q_first = first_frame.get("q", None)
                ids_first = first_frame.get(
                    "id", np.arange(1, natom + 1, dtype=int))
                if q_first is not None and len(q_first) == natom:
                    sort_order = np.argsort(ids_first)
                    initial_charges = np.asarray(q_first, dtype=float)[sort_order]

            # Build atoms (in id-sorted order, internally 0-indexed)
            self.atoms = []
            for i in range(natom):
                atype = int(atom_type_arr[i])
                mass = mass_map.get(atype, 1.0)
                tlabel = type_labels.get(atype, str(atype))
                self.atoms.append(AtomInfo(
                    id=i, mass=mass, mol_id=int(atom_mol_arr[i]),
                    charge=float(initial_charges[i]), type_label=tlabel,
                ))
            self.natom = natom

            # Build molecules via union-find on bonds (matches MDA path).
            # If atom_style was "full", atom_mol_arr already gives molecule IDs;
            # we still run union-find on bonds to mirror the MDA-built path
            # exactly (handles atom_style "atomic" with bonds, and gives
            # internally-renumbered molecule IDs starting at 0).
            #
            # Fallback: when the data file has atom_style=full with molecule
            # IDs but NO Bonds section (typical for ML potentials like DeePMD —
            # bonds are unused at run time), union-find collapses to a per-atom
            # identity, which would mis-label every atom as its own molecule.
            # Use atom_mol_arr directly in that case.
            parent = list(range(natom))

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def _union(a: int, b: int) -> None:
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent[ra] = rb

            has_bonds = len(data["bonds"]) > 0
            if has_bonds:
                for a, b in data["bonds"]:
                    _union(int(a), int(b))
                roots = [_find(i) for i in range(natom)]
            else:
                # atom_mol_arr-driven grouping (DeePMD/ML pair-style data files).
                roots = [int(atom_mol_arr[i]) for i in range(natom)]
            root_to_mol: dict[int, int] = {}
            mol_ids = np.zeros(natom, dtype=int)
            for i, r in enumerate(roots):
                if r not in root_to_mol:
                    root_to_mol[r] = len(root_to_mol)
                mol_ids[i] = root_to_mol[r]

            # Re-tag atoms with internal molecule IDs (matches MDA convention)
            for i, mid in enumerate(mol_ids):
                self.atoms[i].mol_id = int(mid)

            nmol = int(mol_ids.max()) + 1 if natom > 0 else 0
            self.mols = [MolInfo(id=m) for m in range(nmol)]
            for a in self.atoms:
                m = self.mols[a.mol_id]
                m.atom_ids.append(a.id)
                m.mass += a.mass
            self.nmol = nmol

            # Per-molecule bond list (for BPM Raman)
            for a, b in data["bonds"]:
                ai, bi = int(a), int(b)
                mid = self.atoms[ai].mol_id
                self.mols[mid].bonds.append((ai, bi))

        # ── Path 2: no data file → minimal monatomic topology ──────────────
        else:
            natom = first_frame["natom"]
            atom_types = first_frame.get("type", np.ones(natom, dtype=int))
            atom_ids_raw = first_frame.get(
                "id", np.arange(1, natom + 1, dtype=int))
            order = np.argsort(atom_ids_raw)
            atom_types = atom_types[order]
            q_raw = first_frame.get("q", None)
            q_sorted = q_raw[order] if q_raw is not None else None
            # When out_units = lj, honour [output] lj_mass for the per-atom
            # mass so the velocity → T mapping and downstream D, μ, energies
            # carry the correct unit dimension.  In any other case the user
            # has not declared an atomic mass and we fall back to 1 g/mol —
            # the validate() routine has already warned about the resulting
            # mass-dependent corruption.  Single-type assumption is fine for
            # monatomic dumps (LJ-Ar, KA binary with type as group key, etc.);
            # multi-type real systems need a data file regardless.
            mass_default = (cfg.lj_mass if cfg.out_units == "lj"
                            and cfg.lj_mass != 1.0
                            else 1.0)
            self.atoms = []
            for i in range(natom):
                atype = int(atom_types[i])
                charge = float(q_sorted[i]) if q_sorted is not None else 0.0
                self.atoms.append(AtomInfo(
                    id=i, mass=mass_default, mol_id=i, charge=charge,
                ))
            self.natom = natom
            self.mols = [MolInfo(id=i, atom_ids=[i], mass=mass_default)
                         for i in range(natom)]
            self.nmol = natom

        self.periodic = True   # LAMMPS dumps are typically periodic

    # ── Native AMBER/NetCDF path ──────────────────────────────────────────────

    def _build_from_amber_ncdf(self, cfg: Config) -> None:
        """Topology from LAMMPS data file; trajectory metadata via CDFReader.

        Mirrors _build_from_lammps_dump's "Path 1 — data file present" branch
        but skips the dump-first-frame inspection (NCDF has no variable-column
        scheme like LAMMPSDUMP's `q`/`mux`/etc.).  Stores the parsed CDFReader
        on ``self._ncdf_reader`` for reuse by ``_iter_frames_amber_ncdf``.
        """
        if not (cfg.topology and Path(cfg.topology).exists()):
            raise RuntimeError(
                "trajectory_format=NCDF requires a topology file (cfg.topology) "
                "pointing at a LAMMPS data file — NCDF files alone don't carry "
                "molecule/charge info."
            )
        if not cfg.trajectory:
            raise RuntimeError("trajectory_format=NCDF requires cfg.trajectory")

        # Open the CDF file once and inspect.
        nc_path = cfg.trajectory[0]
        rd = CDFReader(nc_path)
        self._ncdf_reader = rd
        log.info(
            "AMBER NetCDF (CDF-%d) trajectory: %d frames, vars=%s",
            rd.fmt, rd.numrecs, sorted(rd.vars.keys())
        )

        # Topology from data file — verbatim from _build_from_lammps_dump's
        # data-file branch.  No per-frame charge override (NCDF doesn't ship
        # a `q` column).
        data = _read_lammps_data_full(cfg.topology)
        natom         = data["n_atoms"] if data["n_atoms"] > 0 else len(data["atom_id"])
        mass_map      = data["masses"]
        type_labels   = data["type_labels"]
        atom_type_arr = data["atom_type"]
        atom_chg_arr  = data["atom_charge"]
        atom_mol_arr  = data["atom_mol_id"]

        self.atoms = []
        for i in range(natom):
            atype = int(atom_type_arr[i])
            mass = mass_map.get(atype, 1.0)
            tlabel = type_labels.get(atype, str(atype))
            self.atoms.append(AtomInfo(
                id=i, mass=mass, mol_id=int(atom_mol_arr[i]),
                charge=float(atom_chg_arr[i]), type_label=tlabel,
            ))
        self.natom = natom

        # Cross-check atom count against the trajectory.  AMBER coords dims are
        # (frame, atom, spatial=3); select the `atom` dim by name (not the trailing
        # spatial=3, which produced a spurious "atom dimension (3) ≠ natom" warning).
        coords_var = rd.vars.get("unwrapped_coordinates") or rd.vars.get("coordinates")
        traj_natom = None
        if coords_var is not None:
            for d in coords_var.dim_ids:
                if rd.dims[d].name == "atom":
                    traj_natom = rd.dims[d].length
                    break
            if traj_natom is None:        # fallback: non-record, non-spatial(3) dim
                cand = [rd.dims[d].length for d in coords_var.dim_ids
                        if not rd.dims[d].is_record and rd.dims[d].length not in (0, 3)]
                traj_natom = cand[-1] if cand else None
        if traj_natom is not None and traj_natom != natom:
            log.warning(
                "NCDF atom dimension (%d) ≠ data file natom (%d); "
                "results will be inconsistent.", traj_natom, natom)

        # Molecule grouping via union-find on bonds (identical to LAMMPSDUMP path).
        # Fallback to atom_mol_arr when no Bonds section (DeePMD/ML data files).
        parent = list(range(natom))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        has_bonds = len(data["bonds"]) > 0
        if has_bonds:
            for a, b in data["bonds"]:
                _union(int(a), int(b))
            roots = [_find(i) for i in range(natom)]
        else:
            roots = [int(atom_mol_arr[i]) for i in range(natom)]
        root_to_mol: dict[int, int] = {}
        mol_ids = np.zeros(natom, dtype=int)
        for i, r in enumerate(roots):
            if r not in root_to_mol:
                root_to_mol[r] = len(root_to_mol)
            mol_ids[i] = root_to_mol[r]
        for i, mid in enumerate(mol_ids):
            self.atoms[i].mol_id = int(mid)

        nmol = int(mol_ids.max()) + 1 if natom > 0 else 0
        self.mols = [MolInfo(id=m) for m in range(nmol)]
        for a in self.atoms:
            m = self.mols[a.mol_id]
            m.atom_ids.append(a.id)
            m.mass += a.mass
        self.nmol = nmol

        for a, b in data["bonds"]:
            ai, bi = int(a), int(b)
            mid = self.atoms[ai].mol_id
            self.mols[mid].bonds.append((ai, bi))

        self.periodic = True

    # ── Group builder ─────────────────────────────────────────────────────────

    def _build_groups(self, cfg: Config) -> None:
        """
        Read optional group file; fall back to one all-atom group.

        Always appends a final group containing all atoms (the whole system).
        """
        grp_path = cfg.group_file
        if grp_path.lower() == "none" or not Path(grp_path).exists():
            # Single all-atom group + whole-system group (same when ngrp=1)
            g = GroupInfo(id=0,
                          atom_ids=list(range(self.natom)),
                          mass=sum(a.mass for a in self.atoms),
                          rotsym=cfg.mol_rotsym,
                          linear=cfg.mol_linear,
                          constraint=cfg.constraints)
            self.groups = [g]
        else:
            self.groups = _read_group_file(grp_path, self.atoms, cfg)

        # Attach molecule ids to each group
        atom_to_mol = {a.id: a.mol_id for a in self.atoms}
        for g in self.groups:
            seen: set[int] = set()
            for aid in g.atom_ids:
                mid = atom_to_mol[aid]
                if mid not in seen:
                    g.mol_ids.append(mid)
                    seen.add(mid)

        # Auto-set linear = 1 for any group whose molecules all have exactly 2 atoms.
        # Diatomic molecules are always linear regardless of user setting.
        for g in self.groups:
            if g.mol_ids and all(len(self.mols[mid].atom_ids) == 2 for mid in g.mol_ids):
                if not g.linear:
                    g.linear = 1
                    log.info("Group %d: all molecules are diatomic — auto-setting linear = 1",
                             g.id + 1)

        self.ngrp = len(self.groups)

    # ── Frame iteration ───────────────────────────────────────────────────────

    def iter_frames(self, cfg: Config) -> Iterator[FrameData]:
        """
        Yield :class:`FrameData` for each selected trajectory frame.

        Frame selection follows ``cfg.start``, ``cfg.stop``, ``cfg.step``
        (all 1-based, consistent with the original control file convention).

        The velocity scaling factor ``cfg.vel_scale`` is applied here so
        that the compute layer always receives velocities in Å/ps.

        For LAMMPSDUMP trajectories a fast bulk reader is used regardless of
        whether MDAnalysis is installed: it reads all lines in one shot and
        parses the entire atom block with :func:`numpy.fromstring`, typically
        2-3× faster than the MDAnalysis per-frame path.
        """
        start = max(0, cfg.start - 1)   # convert to 0-based
        stop = cfg.stop if cfg.stop > 0 else None
        step = cfg.step

        trj_fmt = (cfg.trajectory_format or "").upper()
        _is_lammps_dump = trj_fmt in ("LAMMPSDUMP", "LAMMPS")
        _is_amber_ncdf  = trj_fmt in ("NCDF", "AMBER", "AMBERNCDF", "NETCDF")
        _is_dlpoly      = trj_fmt in ("DLPOLY", "DL_POLY")
        _is_md4d        = trj_fmt in ("MD4D", "RAWVEL")

        if _is_md4d:
            yield from self._iter_frames_md4d(start, stop, step, cfg)
        elif _is_amber_ncdf:
            yield from self._iter_frames_amber_ncdf(start, stop, step, cfg)
        elif _is_dlpoly:
            yield from self._iter_frames_dlpoly(start, stop, step, cfg)
        elif _is_lammps_dump or not (_HAS_MDA and self._universe is not None):
            yield from self._iter_frames_lammps_bulk(start, stop, step, cfg)
        else:
            yield from self._iter_frames_mda(start, stop, step, cfg)

    # ── DL_POLY split-trajectory frame iterator ───────────────────────────────

    def _iter_frames_dlpoly(self, start: int, stop, step: int,
                            cfg: Config) -> Iterator[FrameData]:
        """Iterate a DL_POLY split trajectory: positions + cell from
        ``cfg.trajectory[0]`` and velocities from ``cfg.velocity_file``, read
        frame-locked.  Topology (masses, bonds/angles, atom order) comes from
        ``cfg.topology`` as usual; the per-atom element labels in the dump are
        ignored.  Positions are Å and velocities Å/ps (no unit conversion)."""
        from pyxpt.io.dlpoly import iter_frames_dlpoly
        if not cfg.trajectory:
            raise RuntimeError(
                "trajectory_format=DLPOLY requires [files] trajectory = <POSITION file>")
        if not cfg.velocity_file:
            raise RuntimeError(
                "trajectory_format=DLPOLY requires [files] velocity_file = <VELOCITY file>")
        # DL_POLY ASCII velocities are natively Å/ps (the reference
        # trj_reader.cpp applies no conversion on this path).  The
        # lammps_units auto-scale (×1000 for 'real') targets LAMMPS Å/fs dumps
        # and must NOT apply here — honor only an explicitly user-set vel_scale.
        scale = cfg.vel_scale if getattr(cfg, "_vel_scale_provided", False) else 1.0
        for time_ps, pos, vel, box in iter_frames_dlpoly(
                cfg.trajectory[0], cfg.velocity_file, start, stop, step):
            if scale != 1.0:
                vel = vel * scale
            z = np.zeros_like(pos)
            yield FrameData(
                timestep=time_ps,
                positions=pos,
                velocities=vel,
                forces=z,
                stresses=z,
                box=box,
                volume=float(abs(np.linalg.det(box))),
            )

    # ── md4d (d-dimensional reduced-LJ) reader ────────────────────────────────
    @staticmethod
    def _read_md4d_meta(meta_path: str) -> dict:
        """Parse a md4d ``.meta`` sidecar (key=value lines; '#' comments)."""
        m: dict = {}
        with open(meta_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                m[k.strip()] = v.strip()
        return m

    def _md4d_paths(self, cfg: Config) -> tuple[str, str]:
        """Resolve (.vel, .meta) paths from cfg.trajectory[0] (a .vel file or prefix)."""
        if not cfg.trajectory:
            raise RuntimeError(
                "trajectory_format=MD4D requires [files] trajectory = <prefix or .vel file>")
        base = cfg.trajectory[0]
        if base.endswith(".vel"):
            vel_path, meta_path = base, base[:-4] + ".meta"
        else:
            vel_path, meta_path = base + ".vel", base + ".meta"
        return vel_path, meta_path

    def _build_from_md4d(self, cfg: Config) -> None:
        """Build topology for a monatomic d-dimensional reduced-LJ md4d run.

        Reads the ``.meta`` sidecar (dim, N, L_reduced, Tset, dt_reduced,
        velstride, sigma_A, epsilon_K, mass_amu).  Every atom is its own
        single-atom "molecule" (no rotational/vibrational decomposition), so
        the engine takes the atomic translational path.  The velocity-space
        dimension d is taken from the meta and must equal ``cfg.dimension``.
        """
        from pyxpt.constants import KB, NA
        vel_path, meta_path = self._md4d_paths(cfg)
        meta = self._read_md4d_meta(meta_path)
        D = int(meta["dim"]); N = int(meta["N"])
        if int(getattr(cfg, "dimension", 3)) != D:
            raise RuntimeError(
                f"trajectory_format=MD4D: meta dim={D} but cfg.dimension="
                f"{cfg.dimension}; set [thermodynamics] dimension = {D}.")
        L_red = float(meta["L_reduced"]); Tstar = float(meta["Tset"])
        dt_red = float(meta["dt_reduced"]); velstride = int(meta.get("velstride", 1))
        sigma_A = float(meta.get("sigma_A", getattr(cfg, "lj_sigma", 3.405)))
        eps_K = float(meta.get("epsilon_K", 119.78))
        mass_amu = float(meta.get("mass_amu", getattr(cfg, "lj_mass", 39.948)))

        # reduced→physical maps (argon by default): tau = sigma*sqrt(m/eps)
        m_kg = mass_amu * 1e-3 / NA
        eps_J = eps_K * KB
        tau_ps = sigma_A * 1e-10 * (m_kg / eps_J) ** 0.5 * 1e12
        self._md4d = dict(
            vel_path=vel_path, D=D, N=N,
            box_A=L_red * sigma_A,                       # box edge length [Å]
            volume=(L_red * sigma_A) ** D,               # d-dim measure [Å^d]
            dt_ps=dt_red * velstride * tau_ps,           # frame spacing [ps]
            vfac=sigma_A / tau_ps,                       # reduced v → Å/ps
            T_K=Tstar * eps_K, mass_amu=mass_amu,
        )

        # one single-atom "molecule" per atom (monatomic; no internal DoF)
        self.atoms = [AtomInfo(id=i, mass=mass_amu, mol_id=i) for i in range(N)]
        self.mols = [MolInfo(id=i, atom_ids=[i], mass=mass_amu) for i in range(N)]
        self.natom = N
        self.nmol = N
        self.periodic = True
        log.info("md4d: d=%d  N=%d  box=%.4f Å  V=%.3f Å^%d  dt=%.5g ps  T=%.2f K",
                 D, N, self._md4d["box_A"], self._md4d["volume"], D,
                 self._md4d["dt_ps"], self._md4d["T_K"])

    def _iter_frames_md4d(self, start: int, stop, step: int,
                          cfg: Config) -> Iterator[FrameData]:
        """Iterate a md4d binary ``.vel`` trajectory (float32, (nframes,N,D),
        row-major, reduced LJ units).  Velocities are converted to Å/ps so the
        engine's mass-weighted VACF recovers T = Tset·ε/k_B by equipartition."""
        md = self._md4d
        D, N = md["D"], md["N"]
        arr = np.fromfile(md["vel_path"], dtype=np.float32)
        nfr = arr.size // (N * D)
        arr = arr[:nfr * N * D].reshape(nfr, N, D)
        vfac = md["vfac"]; dt_ps = md["dt_ps"]; vol = md["volume"]; T_K = md["T_K"]
        box = np.eye(D) * md["box_A"]
        stop = nfr if stop is None else min(stop, nfr)
        zeros = np.zeros((N, D), dtype=np.float64)
        for fi in range(start, stop, step):
            vel = arr[fi].astype(np.float64) * vfac        # (N, D) Å/ps
            yield FrameData(
                timestep=fi * dt_ps,                        # cumulative ps → Δt auto-derived
                positions=zeros,
                velocities=vel,
                forces=zeros,
                stresses=zeros,
                box=box,
                temperature=T_K,
                volume=vol,
            )

    # ── MDAnalysis frame iterator ─────────────────────────────────────────────

    def _iter_frames_mda(self, start: int, stop, step: int,
                          cfg: Config) -> Iterator[FrameData]:
        u = self._universe
        trj = u.trajectory

        # Clamp stop
        n_total = len(trj)
        end = n_total if stop is None else min(stop, n_total)

        scale = cfg.vel_scale

        for ts in trj[start:end:step]:
            positions = u.atoms.positions.copy().astype(np.float64)    # Å

            # Velocities — not all formats provide them
            try:
                velocities = u.atoms.velocities.copy().astype(np.float64) * scale
            except (AttributeError, mda.exceptions.NoDataError):
                velocities = np.zeros_like(positions)
                log.debug("Frame %d: no velocities in trajectory.", fi)

            # Box matrix H (3×3, row = lattice vector)
            if ts.dimensions is not None:
                box = _triclinic_box(ts.dimensions)
            else:
                box = np.eye(3) * 1e6   # effectively infinite box

            # Scalar thermo (LAMMPS aux data, if available)
            temp = getattr(ts, "_data", {}).get("temperature", 0.0) or 0.0
            pres = getattr(ts, "_data", {}).get("pressure", 0.0) or 0.0
            vol  = abs(np.linalg.det(box))
            etot = getattr(ts, "_data", {}).get("total_energy", 0.0) or 0.0

            # Per-atom energies and charges (if available in trajectory).
            # MDAnalysis stores additional LAMMPS columns in ts.data (requires
            # additional_columns=True on the Universe, set above for LAMMPSDUMP).
            #
            # Preferred energy convention: separate ``pe`` (potential) and
            # ``ke`` (kinetic) columns; the per-atom total is pe + ke.  Legacy
            # single-column names (c_pe, v_atomEng, v_pe, c_eng) are kept as a
            # fallback when the new pair is absent.
            atom_eng = None
            frame_charges = None
            ts_data = getattr(ts, "data", None) or getattr(ts, "_data", {})
            if "pe" in ts_data and "ke" in ts_data:
                _pe = np.asarray(ts_data["pe"], dtype=float)
                _ke = np.asarray(ts_data["ke"], dtype=float)
                if len(_pe) == self.natom and len(_ke) == self.natom:
                    if not self._pe_ke_warned and np.array_equal(_pe, _ke):
                        log.warning(
                            "LAMMPS dump 'pe' and 'ke' columns are bit-identical — "
                            "this almost certainly indicates a typo in the LAMMPS "
                            "input (e.g., `compute atomKE all pe/atom`).  "
                            "The convective term in the heat current will be 2× "
                            "the true Σ pe·v and κ from thermal_conductivity will "
                            "be biased.  Fix with `compute atomKE all ke/atom`."
                        )
                        self._pe_ke_warned = True
                    atom_eng = _pe + _ke
            elif "pe" in ts_data:
                _pe = np.asarray(ts_data["pe"], dtype=float)
                if len(_pe) == self.natom:
                    atom_eng = _pe
            else:
                for field_name in ("c_pe", "v_atomEng", "v_pe", "c_eng"):
                    if field_name in ts_data:
                        atom_eng = np.asarray(ts_data[field_name], dtype=float)
                        break
            if self.has_frame_charges:
                q_arr = ts_data.get("q", None)
                if q_arr is not None and len(q_arr) == self.natom:
                    frame_charges = np.asarray(q_arr, dtype=float)

            # AMOEBA induced dipoles via MDAnalysis auxiliary data.
            # Prefer mux/muy/muz; fall back to legacy f_dipole[1..3].
            ind_dip = None
            if self.has_induced_dipoles:
                try:
                    if all(c in ts_data for c in ("mux", "muy", "muz")):
                        dx = np.asarray(ts_data["mux"], dtype=float)
                        dy = np.asarray(ts_data["muy"], dtype=float)
                        dz = np.asarray(ts_data["muz"], dtype=float)
                    else:
                        dx = np.asarray(ts_data["f_dipole[1]"], dtype=float)
                        dy = np.asarray(ts_data["f_dipole[2]"], dtype=float)
                        dz = np.asarray(ts_data["f_dipole[3]"], dtype=float)
                    if len(dx) == self.natom:
                        ind_dip = np.column_stack([dx, dy, dz])
                except (KeyError, TypeError):
                    pass

            yield FrameData(
                timestep=float(ts.time),
                positions=positions,
                velocities=velocities,
                box=box,
                temperature=float(temp),
                pressure=float(pres),
                volume=float(vol),
                total_energy=float(etot),
                atom_energies=atom_eng,
                charges=frame_charges,
                induced_dipoles=ind_dip,
            )

    # ── Fast bulk LAMMPS dump reader ─────────────────────────────────────────

    def _iter_frames_lammps_bulk(self, start: int, stop, step: int,
                                  cfg: Config) -> Iterator[FrameData]:
        """
        Streaming LAMMPS dump frame iterator (constant memory per frame).

        Reads the dump one frame at a time.  Memory: O(natoms) per yield.
        Pre-2026-05 versions slurped the entire trajectory into a Python
        list of strings before parsing, which used ~2× the file size in
        RAM and OOM'd on ≥10 GB trajectories.  This streaming form
        scales to any trajectory length.

        Assumes atom ordering is consistent across frames (standard for
        NVT/NPT without atom insertion/deletion).  Atoms are sorted per
        frame by LAMMPS ID.
        """
        if not cfg.trajectory:
            return

        # ── Inspect the first frame to detect natoms, columns, layout ────────
        with open(cfg.trajectory[0]) as fh:
            first_header = [fh.readline() for _ in range(9)]
        if any(not h for h in first_header):
            return
        natoms = int(first_header[3].strip())
        col_header = first_header[8].strip().split()
        cols = col_header[2:]
        col_idx = {c: i for i, c in enumerate(cols)}
        ncols = len(cols)

        # Position column names (unwrapped > scaled > wrapped)
        for px, py, pz, scaled in (
            ("xu", "yu", "zu", False),
            ("x",  "y",  "z",  False),
            ("xs", "ys", "zs", True),
        ):
            if px in col_idx:
                break
        else:
            px = py = pz = None   # type: ignore[assignment]
            scaled = False

        has_vel = all(c in col_idx for c in ("vx", "vy", "vz"))
        # Fallback: LAMMPS `fix ave/atom <N> <M> <N> vx vy vz` writes block-
        # averaged velocities under `f_avg<id>[1..3]` rather than vx/vy/vz —
        # used for multi-res dumps at coarse stride (Nyquist-band-limited).
        # Detect a contiguous triple of `f_avgX[1..3]` columns and treat them
        # as the velocity source when raw vx/vy/vz are absent.
        _vfb = None
        if not has_vel:
            import re as _re
            _avg_groups: dict[str, dict[int, int]] = {}
            for _cname, _ci in col_idx.items():
                _m = _re.match(r"f_(avg\w*)\[([123])\]$", _cname)
                if _m:
                    _avg_groups.setdefault(_m.group(1), {})[int(_m.group(2))] = _ci
            for _gname, _cmap in _avg_groups.items():
                if set(_cmap.keys()) == {1, 2, 3}:
                    _vfb = [_cmap[1], _cmap[2], _cmap[3]]
                    log.info("LAMMPS dump uses block-averaged velocities "
                             "f_%s[1..3] (no vx/vy/vz); treating as velocity.",
                             _gname)
                    has_vel = True
                    break
        has_force = all(c in col_idx for c in ("fx", "fy", "fz"))
        _stress_new = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
        _stress_old = ("c_stress[1]", "c_stress[2]", "c_stress[3]",
                       "c_stress[4]", "c_stress[5]", "c_stress[6]")
        if all(c in col_idx for c in _stress_new):
            stress_cols = _stress_new
            has_stress = True
        elif all(c in col_idx for c in _stress_old):
            stress_cols = _stress_old
            has_stress = True
        else:
            stress_cols = None
            has_stress = False
        has_id = "id" in col_idx

        pe_cidx = col_idx.get("pe", None)
        ke_cidx = col_idx.get("ke", None)
        eng_cidx = None
        if pe_cidx is None and ke_cidx is None:
            for _eng_col in ("c_pe", "v_atomEng", "v_pe", "c_eng"):
                if _eng_col in col_idx:
                    eng_cidx = col_idx[_eng_col]
                    break

        q_cidx = col_idx.get("q", None)

        _dip_new = ("mux", "muy", "muz")
        _dip_old = ("f_dipole[1]", "f_dipole[2]", "f_dipole[3]")
        if all(c in col_idx for c in _dip_new):
            dip_cidx = [col_idx[c] for c in _dip_new]
        elif all(c in col_idx for c in _dip_old):
            dip_cidx = [col_idx[c] for c in _dip_old]
        else:
            dip_cidx = None

        scale = cfg.vel_scale
        pos_cidx = ([col_idx[px], col_idx[py], col_idx[pz]]
                    if px is not None else None)
        if _vfb is not None:
            vel_cidx = _vfb
        else:
            vel_cidx = ([col_idx["vx"], col_idx["vy"], col_idx["vz"]]
                        if has_vel else None)
        force_cidx = ([col_idx["fx"], col_idx["fy"], col_idx["fz"]]
                    if has_force else None)
        stress_cidx = ([col_idx[c] for c in stress_cols]
                       if has_stress else None)

        # Triclinic detection from first frame's box-bounds line (line 5)
        _box_line = first_header[5].split()
        _triclinic = len(_box_line) == 3

        # ── Stream frames across all trajectory files ────────────────────────
        # Estimate bytes per atom line from the first frame's actual data
        # (used to size the per-frame chunked read; 20 % over-estimate is
        # safe and avoids a second read in the steady state).
        with open(cfg.trajectory[0], "rb") as fh:
            for _ in range(9):
                fh.readline()                              # skip header
            atom_block_start = fh.tell()
            for _ in range(natoms):
                fh.readline()
            atom_block_bytes = fh.tell() - atom_block_start
        bytes_per_atom_line = max(80, atom_block_bytes // max(natoms, 1))
        frame_chunk_bytes = int(bytes_per_atom_line * natoms * 1.2)

        frame_count = 0   # global frame index across files
        for fpath in cfg.trajectory:
            with open(fpath, "rb") as fh:               # binary for vectorized parse
                while True:
                    # Read 9-line frame header (decoded for downstream str ops)
                    header = [fh.readline().decode("ascii", errors="ignore")
                              for _ in range(9)]
                    if not header[0]:
                        break  # EOF on this file
                    if any(not h for h in header):
                        # Truncated frame — silently stop
                        return

                    # Skip frames before start, or that step doesn't select
                    if frame_count < start or (frame_count - start) % step != 0:
                        # Skip natoms atom lines without parsing
                        for _ in range(natoms):
                            fh.readline()
                        frame_count += 1
                        if stop is not None and frame_count >= stop:
                            return
                        continue

                    # Read this frame's atom block as one bytes chunk
                    chunk = fh.read(frame_chunk_bytes)
                    if not chunk:
                        return  # truncated mid-frame
                    # Find newline positions via vectorized C scan
                    buf = np.frombuffer(chunk, dtype=np.uint8)
                    nl = np.where(buf == 0x0A)[0]
                    if len(nl) < natoms:
                        # Read extra bytes until we have enough newlines
                        extra = fh.read(frame_chunk_bytes)
                        if not extra:
                            return  # truncated mid-frame
                        chunk = chunk + extra
                        buf = np.frombuffer(chunk, dtype=np.uint8)
                        nl = np.where(buf == 0x0A)[0]
                        if len(nl) < natoms:
                            return  # really truncated
                    end = int(nl[natoms - 1]) + 1
                    data = np.fromstring(chunk[:end], sep=" ",
                                          dtype=np.float32)
                    if data.size != natoms * ncols:
                        raise ValueError(
                            f"LAMMPS dump frame at index {frame_count}: "
                            f"expected {natoms * ncols} values, got {data.size}"
                        )
                    data = data.reshape(natoms, ncols)
                    # Rewind any over-read bytes so the next frame's header
                    # reads from the correct position
                    if end < len(chunk):
                        fh.seek(end - len(chunk), 1)
                    del chunk, buf, nl

                    # Sort atoms by ID
                    if has_id:
                        ids = data[:, col_idx["id"]].astype(np.int32)
                        order = np.argsort(ids)
                        data = data[order]

                    # Timestep + box
                    ts_val = float(header[1].strip()) * cfg.timestep
                    b5 = header[5].split()
                    b6 = header[6].split()
                    b7 = header[7].split()
                    if _triclinic:
                        xlo, xhi, xy = float(b5[0]), float(b5[1]), float(b5[2])
                        ylo, yhi, xz = float(b6[0]), float(b6[1]), float(b6[2])
                        zlo, zhi, yz = float(b7[0]), float(b7[1]), float(b7[2])
                        box = np.array([[xhi-xlo, 0.0, 0.0],
                                        [xy,      yhi-ylo, 0.0],
                                        [xz,      yz,      zhi-zlo]])
                    else:
                        xlo, xhi = float(b5[0]), float(b5[1])
                        ylo, yhi = float(b6[0]), float(b6[1])
                        zlo, zhi = float(b7[0]), float(b7[1])
                        box = np.diag([xhi-xlo, yhi-ylo, zhi-zlo])

                    # Positions
                    if pos_cidx is not None:
                        pos = data[:, pos_cidx].astype(np.float64)
                        if scaled:
                            pos = pos @ box
                    else:
                        pos = np.zeros((natoms, 3))

                    vel = (data[:, vel_cidx].astype(np.float64) * scale
                           if vel_cidx is not None else np.zeros((natoms, 3)))

                    force = (data[:, force_cidx].astype(np.float64)
                             if force_cidx is not None else np.zeros((natoms, 3)))

                    stress = (data[:, stress_cidx].astype(np.float64)
                              if stress_cidx is not None else np.zeros((natoms, 3)))

                    if pe_cidx is not None and ke_cidx is not None:
                        _pe_arr = data[:, pe_cidx].astype(np.float64)
                        _ke_arr = data[:, ke_cidx].astype(np.float64)
                        if not self._pe_ke_warned and np.array_equal(_pe_arr, _ke_arr):
                            log.warning(
                                "LAMMPS dump 'pe' and 'ke' columns are bit-"
                                "identical — this almost certainly indicates a "
                                "typo in the LAMMPS input (e.g., "
                                "`compute atomKE all pe/atom`).  The convective "
                                "term in the heat current will be 2× the true "
                                "Σ pe·v and κ from thermal_conductivity will "
                                "be biased.  Fix with `compute atomKE all ke/atom`."
                            )
                            self._pe_ke_warned = True
                        atom_eng = _pe_arr + _ke_arr
                    elif pe_cidx is not None:
                        atom_eng = data[:, pe_cidx].astype(np.float64)
                    elif eng_cidx is not None:
                        atom_eng = data[:, eng_cidx].astype(np.float64)
                    else:
                        atom_eng = None

                    frame_charges = (data[:, q_cidx].astype(np.float64)
                                     if q_cidx is not None else None)

                    ind_dip = (data[:, dip_cidx].astype(np.float64)
                               if dip_cidx is not None else None)

                    yield FrameData(
                        timestep=ts_val,
                        positions=pos,
                        velocities=vel,
                        forces=force,
                        stresses=stress,
                        box=box,
                        temperature=0.0,
                        pressure=0.0,
                        volume=float(abs(np.linalg.det(box))),
                        total_energy=0.0,
                        atom_energies=atom_eng,
                        charges=frame_charges,
                        induced_dipoles=ind_dip,
                    )

                    frame_count += 1
                    if stop is not None and frame_count >= stop:
                        return

    # ── Native AMBER/NetCDF frame iterator ──────────────────────────────────

    def _iter_frames_amber_ncdf(self, start: int, stop, step: int,
                                cfg: Config) -> Iterator[FrameData]:
        """Yield FrameData from an AMBER-convention CDF-1/2/5 trajectory.

        Reads coordinates, velocities (if present), and per-frame box vectors
        (from cell_lengths + cell_angles).  Builds the 3×3 H matrix the same
        way the MDA path does, so downstream box-derived quantities are
        bit-comparable across paths.
        """
        rd: CDFReader = getattr(self, "_ncdf_reader", None) or CDFReader(cfg.trajectory[0])
        # LAMMPS' `dump netcdf id type xu yu zu ...` writes unwrapped positions
        # under the variable name ``unwrapped_coordinates``; ``x y z`` dumps
        # write ``coordinates``.  Either is acceptable here.
        coords = (rd.vars.get("unwrapped_coordinates")
                  or rd.vars.get("coordinates"))
        vels   = rd.vars.get("velocities")
        cell_l = rd.vars.get("cell_lengths")
        cell_a = rd.vars.get("cell_angles")
        time_v = rd.vars.get("time")
        if coords is None:
            raise ValueError(
                f"{cfg.trajectory[0]}: no 'coordinates' or "
                f"'unwrapped_coordinates' variable; "
                f"available = {sorted(rd.vars.keys())}"
            )

        end = stop if stop is not None and stop > 0 else rd.numrecs
        end = min(end, rd.numrecs)
        scale = cfg.vel_scale
        dt_step = (cfg.dump_freq if cfg.dump_freq > 0 else 1) * cfg.timestep

        # Per-atom energies: production LAMMPS decks dump c_atomPE (+ c_atomKE)
        # as named compute variables.  LAMMPS' AMBER NCDF writer keeps the
        # compute name as the variable name (text dumps use `dump_modify
        # colname` to alias, but NCDF doesn't honor that).  Engine semantics
        # match the LAMMPSDUMP path: sum PE + KE for the per-atom MD total
        # energy, fall back to PE-only if KE missing.  Without this lookup
        # the engine fell back to cfg.energy_avg × DOF-fraction, which
        # mis-attributes ion-water binding to water in mixtures.
        atom_pe_v = (rd.vars.get("c_atomPE") or rd.vars.get("pe")
                     or rd.vars.get("c_pe"))
        atom_ke_v = (rd.vars.get("c_atomKE") or rd.vars.get("ke")
                     or rd.vars.get("c_ke"))

        # Resolve the time-axis conversion to ps.  LAMMPS writes `time` in
        # the simulation's internal time unit (fs for `units real`, ps for
        # `metal`, …) and annotates the variable with the matching `units`
        # attribute.  Without this conversion the engine's VACF Δt is off
        # by 10³ for real units → D, g(0), fluidicity all wrong by 10³,
        # f saturates at 1, S blows up to Sackur–Tetrode levels.
        if time_v is not None:
            t_units = _decode_units(getattr(time_v, "attrs", {}))
            t_scale_factor = float(getattr(time_v, "attrs", {}).get(
                "scale_factor", 1.0))
            t_to_ps = _AMBER_TIME_TO_PS.get(t_units)
            if t_to_ps is None:
                if t_units:
                    log.warning(
                        "NCDF `time` carries units=%r which pyxpt does not "
                        "recognise; leaving values unscaled (assuming ps).",
                        t_units)
                t_to_ps = 1.0
            time_conv = t_scale_factor * t_to_ps
        else:
            time_conv = 1.0

        for i in range(start, end, step):
            pos = rd.read_frame(coords, i).astype(np.float64)
            if vels is not None:
                vel = rd.read_frame(vels, i).astype(np.float64) * scale
            else:
                vel = np.zeros_like(pos)

            # Box: cell_lengths (a,b,c in Å) + cell_angles (α,β,γ in deg).
            # Fall back to an effectively-infinite cubic box if absent.
            if cell_l is not None and cell_a is not None:
                cl = rd.read_frame(cell_l, i).astype(np.float64)
                ca = rd.read_frame(cell_a, i).astype(np.float64)
                box = _triclinic_box(np.concatenate([cl, ca]))
            else:
                box = np.eye(3) * 1e6

            ts_val = (float(rd.read_frame(time_v, i)) * time_conv
                       if time_v is not None
                       else float(i) * dt_step)

            atom_eng = None
            if atom_pe_v is not None:
                pe_arr = rd.read_frame(atom_pe_v, i).astype(np.float64)
                if atom_ke_v is not None:
                    ke_arr = rd.read_frame(atom_ke_v, i).astype(np.float64)
                    atom_eng = pe_arr + ke_arr
                else:
                    atom_eng = pe_arr

            yield FrameData(
                timestep=ts_val,
                positions=pos,
                velocities=vel,
                forces=np.zeros_like(pos),
                # Empty (size 0) — engine's heat-current path checks
                # `fd.stresses.size > 0 and shape[1] == 6` and skips when
                # absent.  Avoid `np.zeros((6,))` (1-D length 6) here: that
                # has size 6 and trips the `shape[1]` access since the
                # engine assumes (natom, 6) per-atom stress when size > 0.
                stresses=np.zeros((0,)),
                box=box,
                temperature=0.0,
                pressure=0.0,
                volume=float(abs(np.linalg.det(box))),
                total_energy=0.0,
                atom_energies=atom_eng,
                charges=None,
                induced_dipoles=None,
            )

    # ── Convenience ──────────────────────────────────────────────────────────

    def total_frames(self, cfg: Config) -> int:
        """Return total number of frames that will be iterated."""
        trj_fmt = (cfg.trajectory_format or "").upper()
        _is_lammps_dump = trj_fmt in ("LAMMPSDUMP", "LAMMPS")
        _is_amber_ncdf  = trj_fmt in ("NCDF", "AMBER", "AMBERNCDF", "NETCDF")
        _is_md4d        = trj_fmt in ("MD4D", "RAWVEL")

        if _is_md4d:
            md = self._md4d
            n = int(Path(md["vel_path"]).stat().st_size // (md["N"] * md["D"] * 4))
        elif _is_amber_ncdf:
            rd = getattr(self, "_ncdf_reader", None) or CDFReader(cfg.trajectory[0])
            n = rd.numrecs
        elif _is_lammps_dump or not (_HAS_MDA and self._universe is not None):
            n = 0
            for fpath in cfg.trajectory:
                # O(few MB) fast path: extrapolate from first / second / last
                # timestep markers, assuming uniform dt per frame.  If the
                # extrapolation is non-integer (variable dt, mixed dumps,
                # or truncated trailing frame), fall back to vectorized
                # byte-count of all "ITEM: TIMESTEP" markers in the file.
                try:
                    n_file = _count_frames_extrapolated(fpath)
                except (ValueError, IndexError, OSError):
                    n_file = _count_frames_marker_count(fpath)
                n += n_file
        else:
            n = len(self._universe.trajectory)

        start = max(0, cfg.start - 1)
        stop = cfg.stop if cfg.stop > 0 else n
        return max(0, math.ceil((min(stop, n) - start) / cfg.step))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _triclinic_box(dimensions: np.ndarray) -> np.ndarray:
    """
    Convert MDAnalysis ``ts.dimensions`` [a, b, c, α, β, γ]
    to a 3×3 H matrix (row = lattice vector), matching the original C++.
    """
    import math
    a, b, c = dimensions[:3]
    alpha, beta, gamma = np.radians(dimensions[3:6])
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)
    cx = cb
    cy = (ca - cb * cg) / sg
    cz = math.sqrt(max(0.0, 1.0 - cx**2 - cy**2))
    H = np.array([
        [a,      0.0,    0.0],
        [b * cg, b * sg, 0.0],
        [c * cx, c * cy, c * cz],
    ])
    return H

def _infer_masses_from_names(names) -> np.ndarray:
    """Guess atom masses from element symbol strings."""
    masses = np.ones(len(names))
    strip = re.compile(r"[-0-9._]")
    
    for i, name in enumerate(names):
        # Strip numbers and symbols, convert to lowercase for the dict lookup
        clean_name = strip.sub("", str(name)).lower()
        
        sym_2 = clean_name[:2]
        sym_1 = clean_name[:1]
        
        if sym_2 in ELEMENTS:
            masses[i] = ELEMENTS[sym_2]
        elif sym_1 in ELEMENTS:
            masses[i] = ELEMENTS[sym_1]
            
    return masses

def _read_lammps_masses(data_path: str) -> dict[int, float]:
    """Extract type→mass mapping from a LAMMPS data file."""
    masses: dict[int, float] = {}
    in_masses = False
    for line in Path(data_path).read_text().splitlines():
        s = line.strip()
        if s == "Masses":
            in_masses = True
            continue
        if in_masses:
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[0].isdigit():
                masses[int(parts[0])] = float(parts[1])
            elif s and not parts[0].isdigit():
                break   # end of Masses section
    return masses


def _read_lammps_data_full(data_path: str) -> dict:
    """Single-pass LAMMPS data file parser.

    Returns:
        n_atoms, n_bonds          — header counts
        masses                    — dict[type_id → mass]
        type_labels               — dict[type_id → label] from masses-section comments
        atoms_id, atoms_mol_id,
        atoms_type, atoms_charge  — np.ndarrays sorted by atom id
        bonds                     — (n_bonds, 2) np.ndarray of 0-indexed atom pairs

    Replaces the MDAnalysis Universe construction for LAMMPSDUMP runs.
    Supports atom_style ``full`` (id mol_id type q x y z [nx ny nz]) and
    ``charge`` (id type q x y z [nx ny nz]).  For ``atomic`` (id type x y z),
    every atom becomes its own molecule (mol_id = atom_id) and charges are 0.
    """
    n_atoms = 0
    n_bonds = 0
    masses: dict[int, float] = {}
    type_labels: dict[int, str] = {}
    section: str | None = None
    section_buf: list[str] = []
    sections: dict[str, list[str]] = {}
    section_keywords = {
        "Masses", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers",
        "Pair Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs",
        "Improper Coeffs", "Velocities", "Ellipsoids",
    }
    text = Path(data_path).read_text()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()   # strip trailing comment
        s = line.strip()
        if s in section_keywords:
            if section is not None:
                sections[section] = section_buf
            section = s
            section_buf = []
            continue
        if section is None:
            # Header lines — parse counts
            parts = s.split()
            if len(parts) >= 2 and parts[0].isdigit():
                if parts[1] == "atoms":
                    n_atoms = int(parts[0])
                elif parts[1] == "bonds":
                    n_bonds = int(parts[0])
        else:
            section_buf.append(s)
    if section is not None:
        sections[section] = section_buf

    # ── Masses (with optional comment label) ────────────────────────────────
    raw_masses = sections.get("Masses", [])
    raw_text   = text.splitlines()
    # Re-scan with original lines to recover masses-section comments
    in_masses = False
    for raw in raw_text:
        s_strip = raw.strip()
        if s_strip == "Masses":
            in_masses = True
            continue
        if in_masses:
            if not s_strip:
                if masses:                           # blank line after entries
                    in_masses = False
                continue
            if s_strip in section_keywords:
                in_masses = False
                continue
            stripped, _, comment = raw.partition("#")
            parts = stripped.split()
            if len(parts) >= 2 and parts[0].isdigit():
                tid = int(parts[0])
                masses[tid] = float(parts[1])
                if comment.strip():
                    type_labels[tid] = comment.strip().split()[0]

    # ── Atoms section: detect atom_style by column count ───────────────────
    atom_lines = [l for l in sections.get("Atoms", []) if l]
    if not atom_lines:
        raise ValueError(
            f"LAMMPS data file {data_path}: no Atoms section found"
        )
    first_cols = atom_lines[0].split()
    ncol = len(first_cols)
    # Guess atom_style:
    #   full:   id mol_id type charge x y z [nx ny nz]              → 7 or 10
    #   charge: id type charge x y z [nx ny nz]                     → 6 or 9
    #   atomic: id type x y z [nx ny nz]                            → 5 or 8
    if ncol in (7, 10):
        atom_style = "full"
    elif ncol in (6, 9):
        atom_style = "charge"
    elif ncol in (5, 8):
        atom_style = "atomic"
    else:
        # Fallback: assume full if there are bonds, else atomic
        atom_style = "full" if n_bonds > 0 else "atomic"
    arr = np.array([line.split() for line in atom_lines], dtype=object)
    if atom_style == "full":
        atom_id     = arr[:, 0].astype(int)
        atom_mol_id = arr[:, 1].astype(int)
        atom_type   = arr[:, 2].astype(int)
        atom_charge = arr[:, 3].astype(float)
    elif atom_style == "charge":
        atom_id     = arr[:, 0].astype(int)
        atom_type   = arr[:, 1].astype(int)
        atom_charge = arr[:, 2].astype(float)
        atom_mol_id = atom_id.copy()                 # one mol per atom
    else:                                              # atomic
        atom_id     = arr[:, 0].astype(int)
        atom_type   = arr[:, 1].astype(int)
        atom_charge = np.zeros(len(arr), dtype=float)
        atom_mol_id = atom_id.copy()
    # Sort by atom id
    order = np.argsort(atom_id)
    atom_id     = atom_id[order]
    atom_mol_id = atom_mol_id[order]
    atom_type   = atom_type[order]
    atom_charge = atom_charge[order]

    # ── Bonds section ──────────────────────────────────────────────────────
    bonds_arr = np.zeros((0, 2), dtype=int)
    if n_bonds > 0:
        bond_lines = [l for l in sections.get("Bonds", []) if l]
        if bond_lines:
            bonds_data = np.array([line.split() for line in bond_lines],
                                  dtype=object)
            # cols: bond_id type atom1 atom2  (LAMMPS atom IDs are 1-based)
            a1 = bonds_data[:, 2].astype(int) - 1
            a2 = bonds_data[:, 3].astype(int) - 1
            bonds_arr = np.column_stack([a1, a2])

    return {
        "n_atoms":      n_atoms,
        "n_bonds":      n_bonds,
        "atom_style":   atom_style,
        "masses":       masses,
        "type_labels":  type_labels,
        "atom_id":      atom_id,
        "atom_mol_id":  atom_mol_id,
        "atom_type":    atom_type,
        "atom_charge":  atom_charge,
        "bonds":        bonds_arr,
    }


def _count_frames_marker_count(fpath: str) -> int:
    """Count LAMMPS dump frames by tallying ``b"ITEM: TIMESTEP"`` occurrences.

    Vectorized C-level byte scan on 64 MB chunks; ~1–2 GB/s on local SSD.
    Robust to variable-width timestep numbers (where file_size / first_frame
    bytes would mis-divide).  Used as the safe fallback when timestep
    extrapolation cannot be applied.
    """
    CHUNK = 64 * 1024 * 1024
    MARKER = b"ITEM: TIMESTEP"
    OVERLAP = len(MARKER) - 1
    n = 0
    with open(fpath, "rb") as fh:
        leftover = b""
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            combined = leftover + chunk
            n += combined.count(MARKER) - leftover.count(MARKER)
            leftover = combined[-OVERLAP:]
    return n


def _last_timestep_in_dump(fpath: str,
                            search_window: int = 4 * 1024 * 1024) -> int | None:
    """Return the last ``ITEM: TIMESTEP`` value in fpath by scanning only the
    final few MB.  Returns None if the marker is not found in the window."""
    import os
    file_size = os.path.getsize(fpath)
    seek_pos = max(0, file_size - search_window)
    MARKER = b"ITEM: TIMESTEP"
    with open(fpath, "rb") as fh:
        fh.seek(seek_pos)
        chunk = fh.read()
    idx = chunk.rfind(MARKER)
    if idx < 0:
        return None
    nl = chunk.find(b"\n", idx)
    if nl < 0:
        return None
    nl_end = chunk.find(b"\n", nl + 1)
    if nl_end < 0:
        return None
    try:
        return int(chunk[nl + 1:nl_end].strip())
    except ValueError:
        return None


def _count_frames_extrapolated(fpath: str) -> int:
    """O(few MB) frame-count via timestep extrapolation.

    Reads frames 0 and 1 to determine ``dt_per_frame = ts1 - ts0``, then
    finds the last frame's timestep in the trailing few MB of the file.
    Returns ``(ts_last - ts0) / dt + 1`` rounded.  Raises ``ValueError``
    when the file has fewer than 2 frames, dt is non-positive, the
    extrapolation is non-integer (suggesting variable dt or truncated
    file), or any other inconsistency.

    Caller falls back to :func:`_count_frames_marker_count` on failure.
    """
    f0 = _read_lammps_dump_frame(fpath, frame_idx=0)
    if f0 is None or "timestep" not in f0 or "natom" not in f0:
        raise ValueError(f"{fpath}: cannot read first frame")
    ts0 = float(f0["timestep"])
    natoms = int(f0["natom"])
    # Find offset of the second frame: just after the first frame's atom
    # block (9 header lines + natoms data lines).
    with open(fpath, "rb") as fh:
        for _ in range(9 + natoms):
            line = fh.readline()
            if not line:
                raise ValueError(f"{fpath}: truncated first frame")
        offset_f1 = fh.tell()
    f1 = _read_lammps_dump_frame(fpath, offset=offset_f1)
    if f1 is None or "timestep" not in f1:
        # Only one frame in the file
        return 1
    ts1 = float(f1["timestep"])
    dt_per_frame = ts1 - ts0
    if dt_per_frame <= 0:
        raise ValueError(f"{fpath}: non-positive dt {ts0}→{ts1}")
    ts_last = _last_timestep_in_dump(fpath)
    if ts_last is None:
        raise ValueError(f"{fpath}: cannot find last timestep")
    raw = (ts_last - ts0) / dt_per_frame + 1.0
    n = int(round(raw))
    if abs(raw - n) > 0.5:
        raise ValueError(
            f"{fpath}: non-integer extrapolated frame count {raw:.3f} "
            f"(ts0={ts0}, ts_last={ts_last}, dt={dt_per_frame})"
        )
    if n <= 0:
        raise ValueError(f"{fpath}: extrapolated n={n}")
    return n


def _index_lammps_dump(fpath: str) -> list[tuple[str, int]]:
    """Return list of (filepath, byte_offset) for each frame in a LAMMPS dump."""
    offsets: list[tuple[str, int]] = []
    with open(fpath, "rb") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if line.strip() == b"ITEM: TIMESTEP":
                offsets.append((fpath, pos))
    return offsets


def _read_lammps_dump_frame(fpath: str,
                             frame_idx: int = 0,
                             offset: int | None = None) -> dict | None:
    """
    Read one frame from a LAMMPS dump file.

    Either ``frame_idx`` (sequential) or ``offset`` (byte offset) must be given.
    Returns a dict with keys: natom, timestep, box, id, type, x, y, z,
    vx, vy, vz, c_pe (all as numpy arrays where applicable).
    """
    if offset is None:
        if frame_idx == 0:
            # Frame 0 is at byte 0 — skip the full-file index scan
            offset = 0
        else:
            # Index to find the right frame
            index = _index_lammps_dump(fpath)
            if frame_idx >= len(index):
                return None
            _, offset = index[frame_idx]

    result: dict = {}
    col_map: dict[str, int] = {}

    with open(fpath, "rb") as fh:
        fh.seek(offset)
        while True:
            line = fh.readline().decode(errors="replace").strip()
            if not line:
                return None
            if line == "ITEM: TIMESTEP":
                result["timestep"] = float(fh.readline())
            elif line == "ITEM: NUMBER OF ATOMS":
                result["natom"] = int(fh.readline())
            elif line.startswith("ITEM: BOX BOUNDS"):
                bounds = []
                for _ in range(3):
                    b = fh.readline().decode(errors="replace").split()
                    bounds.append((float(b[0]), float(b[1])))
                H = np.diag([bounds[0][1]-bounds[0][0],
                              bounds[1][1]-bounds[1][0],
                              bounds[2][1]-bounds[2][0]])
                result["box"] = H
            elif line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]
                col_map = {c: i for i, c in enumerate(cols)}
                break

        n = result.get("natom", 0)
        if n == 0:
            return result

        # Read atom data block into a 2-D array
        rows = []
        for _ in range(n):
            rows.append(fh.readline().decode(errors="replace").split())
        data = np.array(rows, dtype=float)

        int_cols = {"id", "type"}
        # Per-atom columns we may consume downstream.  ``pe`` and ``ke``
        # are the new convention (per-atom total = pe + ke); the legacy
        # single-column names below are kept for back-compat and are
        # consumed exclusively by callers that didn't see ``pe``.
        for col_name in ("id", "type", "x", "y", "z", "xs", "ys", "zs",
                          "vx", "vy", "vz", "q",
                          "pe", "ke",
                          "c_pe", "c_eng", "v_atomEng", "v_pe"):
            if col_name in col_map:
                col_data = data[:, col_map[col_name]]
                result[col_name] = col_data.astype(int) if col_name in int_cols else col_data

        # AMOEBA induced dipole vector.  Preferred columns are mux/muy/muz
        # (set via ``dump_modify colname``); legacy f_dipole[1..3] is the
        # backward-compatible fallback.
        _dip_new = ("mux", "muy", "muz")
        _dip_old = ("f_dipole[1]", "f_dipole[2]", "f_dipole[3]")
        if all(c in col_map for c in _dip_new):
            result["f_dipole"] = np.column_stack(
                [data[:, col_map[c]] for c in _dip_new]
            )  # (natom, 3)
            for c in _dip_new:
                result[c] = data[:, col_map[c]]
        elif all(c in col_map for c in _dip_old):
            result["f_dipole"] = np.column_stack(
                [data[:, col_map[c]] for c in _dip_old]
            )  # (natom, 3)

        # Per-atom stress tensor.  Preferred names sxx/syy/szz/sxy/sxz/syz;
        # legacy c_stress[1..6] is the backward-compatible fallback.
        _stress_new = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
        _stress_old = ("c_stress[1]", "c_stress[2]", "c_stress[3]",
                       "c_stress[4]", "c_stress[5]", "c_stress[6]")
        if all(c in col_map for c in _stress_new):
            for c in _stress_new:
                result[c] = data[:, col_map[c]]
        elif all(c in col_map for c in _stress_old):
            for old, new in zip(_stress_old, _stress_new):
                result[new] = data[:, col_map[old]]

        # Convert scaled coords if needed
        H = result.get("box", np.eye(3))
        for sc, c in (("xs", "x"), ("ys", "y"), ("zs", "z")):
            if sc in result and c not in result:
                idx = ("xs", "ys", "zs").index(sc)
                result[c] = result[sc] * H[idx, idx]

    return result


def _expand_atom_ids(token_str: str, atoms: list[AtomInfo]) -> tuple[list[int], float]:
    """Expand a space-separated atom-id string (supports 'start - end' ranges).

    Returns (atom_ids, total_mass) where atom_ids are 0-based.
    """
    atom_ids: list[int] = []
    mass = 0.0
    toks = token_str.split()
    i = 0
    while i < len(toks):
        if i + 2 < len(toks) and toks[i + 1] == "-":
            for a in range(int(toks[i]), int(toks[i + 2]) + 1):
                aid = a - 1
                atom_ids.append(aid)
                mass += atoms[aid].mass if aid < len(atoms) else 0.0
            i += 3
        else:
            aid = int(toks[i]) - 1
            atom_ids.append(aid)
            mass += atoms[aid].mass if aid < len(atoms) else 0.0
            i += 1
    return atom_ids, mass


def _read_group_file(path: str, atoms: list[AtomInfo],
                     cfg: Config) -> list[GroupInfo]:
    """Parse an INI-format 2PT group file and return GroupInfo objects."""
    import configparser
    text = Path(path).read_text()

    if "Total Groups:" in text:
        raise ValueError(
            f"Group file '{path}' appears to be in legacy flat-keyword format.  "
            "Convert it first with:  python helper_scripts/convert-grp.py "
            f"{path} {path}.ini"
        )

    cp = configparser.ConfigParser()
    cp.read_string(text)

    groups: list[GroupInfo] = []
    for gi, sec in enumerate(cp.sections()):
        raw_atoms = cp.get(sec, "atoms", fallback="")
        atom_ids, mass = _expand_atom_ids(raw_atoms, atoms)

        g = GroupInfo(
            id=gi,
            atom_ids=atom_ids,
            mass=mass,
            rotsym=cp.getint(sec, "rotsym", fallback=cfg.mol_rotsym),
            linear=cp.getint(sec, "linear", fallback=cfg.mol_linear),
            constraint=cp.getint(sec, "constraints", fallback=cfg.constraints),
            volume=cp.getfloat(sec, "volume", fallback=0.0),
            eng_avg=cp.getfloat(sec, "energy_avg", fallback=0.0),
            eng_std=cp.getfloat(sec, "energy_std", fallback=0.0),
        )
        groups.append(g)

    if not groups:
        log.warning("Group file %s: no group sections found; using single group", path)

    return groups
