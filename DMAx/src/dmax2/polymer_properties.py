"""Small polymer-property lookup layer for workflow defaults.

The local registry is deliberately tiny and dependency-free. It gives DMAx2 a
stable boundary that a future atomate2/database adapter can replace with a
Materials Project, PoLyInfo, or in-house polymer-property query.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GlassTransitionSpec, PolymerSpec


@dataclass(frozen=True)
class PolymerTgPrior:
    key: str
    name: str
    tg_k: float
    source: str
    notes: str
    name_aliases: tuple[str, ...] = ()
    smiles_aliases: tuple[str, ...] = ()


LOCAL_TG_PRIORS: tuple[PolymerTgPrior, ...] = (
    PolymerTgPrior(
        key="PVDF",
        name="poly(vinylidene fluoride)",
        tg_k=238.15,
        source="local literature prior",
        notes="Representative amorphous PVDF Tg near -35 C; reports vary with tacticity, crystallinity, and protocol.",
        name_aliases=("PVDF", "polyvinylidene fluoride", "poly(vinylidene fluoride)"),
        smiles_aliases=("[*]CC(F)(F)[*]", "[*]C([H])([H])C(F)(F)[*]"),
    ),
    PolymerTgPrior(
        key="PDMS",
        name="poly(dimethylsiloxane)",
        tg_k=150.15,
        source="local literature prior",
        notes="Representative PDMS Tg near -123 C for amorphous linear PDMS.",
        name_aliases=("PDMS", "poly(dimethylsiloxane)", "polydimethylsiloxane"),
        smiles_aliases=("[*]O[Si](C)(C)O[*]", "[*][Si](C)(C)O[*]"),
    ),
)


def _norm(value: str | None) -> str:
    return "".join((value or "").lower().split())


def lookup_tg_prior(polymer: PolymerSpec, fallback_key: str = "PVDF") -> tuple[PolymerTgPrior, str]:
    """Resolve a Tg prior from polymer name/SMILES, falling back to PVDF."""

    name = _norm(polymer.name)
    smiles = _norm(polymer.smiles)
    for prior in LOCAL_TG_PRIORS:
        if name in {_norm(alias) for alias in prior.name_aliases}:
            return prior, "polymer_name"
        if smiles in {_norm(alias) for alias in prior.smiles_aliases}:
            return prior, "polymer_smiles"
    fallback = next((p for p in LOCAL_TG_PRIORS if p.key.lower() == fallback_key.lower()), LOCAL_TG_PRIORS[0])
    return fallback, "fallback"


def glass_temperature_grid(
    polymer: PolymerSpec,
    spec: GlassTransitionSpec,
) -> tuple[np.ndarray, PolymerTgPrior, str]:
    """Return the Tg-centered temperature grid requested by the glass sweep."""

    prior, match_reason = lookup_tg_prior(polymer, spec.default_reference_polymer)
    center = float(spec.expected_tg_k if spec.expected_tg_k is not None else prior.tg_k)
    half = 0.5 * float(spec.temperature_window_k)
    low = max(float(spec.min_temperature_k), center - half)
    high = center + half
    if spec.max_temperature_k is not None:
        high = min(float(spec.max_temperature_k), high)
    points = max(2, int(spec.temperature_points))
    return np.linspace(low, high, points), prior, match_reason
