"""
Physical constants, unit conversions and enumerations for py-xPT.
"""

from __future__ import annotations
from enum import Enum, IntEnum
import math
import json
import importlib.resources

# ── Physical constants ────────────────────────────────────────────────────────

NA     = 6.0221367e23       # Avogadro (mol⁻¹)
KB     = 1.380658e-23       # Boltzmann (J/K)
H      = 6.62606896e-34     # Planck (J·s)
R      = KB * NA            # Gas constant (J/mol/K)
VLIGHT = 2.99792458e8       # Speed of light (m/s)
PI     = math.pi
CALTOJ = 4.184              # 1 cal = 4.184 J

# Yeh-Hummer self-diffusion finite-size correction constant for cubic boxes
# (J. Phys. Chem. B 108, 15873–15879, 2004).  D_inf = D_PBC + ξ·k_B·T/(6π η L).
# Approximate constant from Madelung-sum on a cubic lattice; same value works
# to ~1% on triclinic boxes when L = V^(1/3) (Kikugawa et al. JCP 2015).
YEH_HUMMER_XI = 2.837297

# Rotational analogue of the Yeh-Hummer translational FSC.  For a freely-
# rotating spherical molecule in a cubic PBC box (Fushiki, JCP 119, 6553, 2003;
# Hess, JCP 116, 209, 2002), the rotational image-sum decays as 1/r³ (rotlet)
# rather than 1/r (Stokeslet), giving:
#     D_R^∞ - D_R^PBC ≈ ξ_R · k_B T / (8π η L³)
# Combined with Stokes-Einstein-Debye D_R ≈ k_B T / (8π η a³) the fractional
# correction reduces to scale_R = 1 + ξ_R · (a/L)³.  ξ_R is the same lattice
# sum as the translational case to leading order; the resulting magnitude is
# typically two orders of magnitude smaller than the translational FSC for
# small molecules in 5-nm boxes.
YEH_HUMMER_XI_ROT = 2.837297

# ── Velocity-component indices ────────────────────────────────────────────────

class VelType(IntEnum):
    TRANS = 0   # centre-of-mass translation
    ANGUL = 1   # angular (weighted by √I)
    IMVIB = 2   # internal vibration
    ROTAT = 3   # rotation (alias used in report)
    TOTAL = 4   # all degrees of freedom

VELTYPE = int(VelType.TOTAL)

# ── 2PT analysis mode ─────────────────────────────────────────────────────────

class ModexPT(str, Enum):
    """
    Entropy-functional taxonomy — three publishable modes.

        1PT : all-solid harmonic baseline (no gas phase); diagnostic reference.
        2PT : gas + solid hard-sphere partition.  The gas/solid treatment is
              selected by `refinement`:
                rigorous   — rigorous-HS Sackur-Tetrode + CS excess (no ln z)
                lin2003    — empirical Lin-Goddard +ln z convention (comparison)
                desjarlais — Desjarlais moment-matched memory-function gas DoS
                r2pt       — Sun 2017 revised-2PT (delta-tuned f, F_a-inclusive)
        3PT : 2PT-rigorous + the parameter-free cage (gas | cage | solid) —
              the high-fidelity, first-principles, transferable functional.

    `refinement` (Config) carries the 2PT sub-method; 3PT implies rigorous + cage.

    ModexPT inherits from str so cfg.mode behaves as a string for serialization
    and comparison; INI files write `mode = 3PT` and code does
    `cfg.mode == ModexPT.THREE_PT`.
    """
    # ── publishable taxonomy (the only modes selectable from the INI) ────────
    ONE_PT   = "1PT"
    TWO_PT   = "2PT"
    THREE_PT = "3PT"

# ── Element table ─────────────────────────────────────────────────────────────

with importlib.resources.open_text(__package__, "elements.json") as _f:
    ELEMENTS: dict[str, float] = json.load(_f)

# ── Copyright banner ──────────────────────────────────────────────────────────

COPYRIGHT = """\
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│           py-xPT — Two- and Three-Phase Thermodynamics (2PT / 3PT)           │
│          absolute entropy & free energy from a single MD trajectory          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Authors:                                                                    │
│    Tod A. Pascal          (tpascal@ucsd.edu) - UCSD                          │
│    Shiang-Tai Lin         (stlin@ntu.edu.tw) - National Taiwan University    │
│    Prabal K. Maiti        (maiti@physics.iisc.ernet.in) - IISc Bangalore     │
│    William A. Goddard III (wag@wag.caltech.edu) - Caltech                    │
│                                                                              │
│  Copyright (c) 2026 Pascal, Lin, Maiti, Goddard.                             │
│                                                                              │
│  Please Cite:                                                                │
│    1. Lin, Blanco, Goddard. J. Chem. Phys., 2003, 119, 11792-11805           │
│    2. Lin, Maiti, Goddard. J. Phys. Chem. B., 2010, 114, 8191-8198           │
│    3. Pascal, Lin, Goddard. PCCP, 2011, 13 (1), 169-181                      │
└──────────────────────────────────────────────────────────────────────────────┘"""
