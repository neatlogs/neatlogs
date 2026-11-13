"""Deterministic identities shared by the live plugin and the importer.

Every record the integration writes is keyed by a uuid5 derived from Harbor's
own identities (job id, task name, trial name). The plugin and
``neatlogs harbor import`` derive the same values from the same Harbor job, so
a second write reconciles with the first instead of creating a duplicate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from uuid import NAMESPACE_URL, UUID, uuid5

NAMESPACE: UUID = uuid5(NAMESPACE_URL, "https://neatlogs.com/integrations/harbor")


def _stable_uuid(kind: str, *parts: object) -> str:
    normalized = ":".join(str(part) for part in parts)
    return str(uuid5(NAMESPACE, f"{kind}:{normalized}"))


def trial_uuid(job_id: object, task_name: str, trial_name: str) -> str:
    """Identity of one Harbor trial (one task attempt) inside one job."""

    return _stable_uuid("trial", job_id, task_name, trial_name)


def run_uuid(job_id: object, agent: str, model: str | None, dataset: str) -> str:
    """Identity of the Neatlogs run for one (agent, model, dataset) in one job."""

    return _stable_uuid("run", job_id, agent, model or "", dataset)


def case_key(dataset: str, task_name: str) -> str:
    """Stable case key: one case per task name, shared by every dataset version."""

    return _stable_uuid("case", dataset, task_name)


def dataset_version_key(task_digests: Iterable[tuple[str, str]]) -> str:
    """Content key of a dataset version: the set of ``(task name, task digest)``.

    Order and repeats (several attempts of one task) do not matter, so two jobs
    over the same task contents land on the same version, and an edited task
    produces a new one.
    """

    canonical = "\n".join(sorted({f"{name}\t{digest}" for name, digest in task_digests}))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_publication_uuid(dataset: str, version_key: str) -> str:
    """Body ``idempotencyKey`` for publishing one dataset version: same tasks, same key."""

    return _stable_uuid("dataset", dataset, version_key)


def trace_seed(trial_id: str) -> str:
    """Seed handed to ``harbor_atif2otel.convert_trajectory`` for the trial's trace."""

    return f"neatlogs-harbor:{trial_id}"


def trace_id_hex(trial_id: str) -> str:
    """The OTLP trace id ``convert_trajectory`` derives from :func:`trace_seed`."""

    return hashlib.sha256(trace_seed(trial_id).encode()).digest()[:16].hex()
