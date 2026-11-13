"""Future atomate2/jobflow compatibility layer.

This module intentionally has no atomate2 dependency. It exports step metadata
that can later be translated into Makers or jobs.
"""

from __future__ import annotations

from .schemas import WorkflowStep
from .polymer_properties import lookup_tg_prior


def to_jobflow_like_plan(steps: list[WorkflowStep]) -> list[dict]:
    """Return serializable job records with dependency information."""

    return [step.to_dict() for step in steps]


def make_jobflow(*_args, **_kwargs):
    raise NotImplementedError(
        "DMAx2 currently runs standalone. Use to_jobflow_like_plan() as the stable adapter boundary."
    )


def resolve_polymer_tg_prior_for_job(polymer_spec, database_client=None) -> dict:
    """Return Tg-prior metadata for a future atomate2 Maker.

    A future implementation can pass a database client with a
    ``get_glass_transition_by_smiles(smiles)`` method. Until then, this function
    falls back to the dependency-free local registry used by standalone DMAx2.
    """

    if database_client is not None and getattr(polymer_spec, "smiles", None):
        record = database_client.get_glass_transition_by_smiles(polymer_spec.smiles)
        if record:
            return {
                "tg_k": float(record["tg_k"]),
                "source": record.get("source", "polymer_database"),
                "matched_by": "database_smiles",
            }
    prior, matched_by = lookup_tg_prior(polymer_spec)
    return {
        "tg_k": prior.tg_k,
        "source": prior.source,
        "matched_by": matched_by,
        "key": prior.key,
        "name": prior.name,
        "notes": prior.notes,
    }
