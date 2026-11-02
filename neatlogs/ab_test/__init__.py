from .client import AbTestClient
from .protocol import AbAssignment, AbDecision, ArmKey, Params

_shared: AbTestClient | None = None


def configure_ab_tests(
    *, keys: list[str], api_key: str | None = None, base_url: str | None = None
) -> AbTestClient:
    """Fetch settings once, then refresh every 60 seconds outside assignment calls."""
    global _shared
    from ..init import get_session_config

    config = get_session_config()
    client = AbTestClient(
        keys=keys,
        api_key=api_key or config.get("_api_key") or "",
        base_url=base_url or config.get("_base_url") or "",
    )
    close_ab_tests()
    _shared = client
    client.start()
    return client


def ab_test(key: str, *, unit: str, fallback: Params | None = None) -> AbAssignment:
    if _shared:
        return _shared.assign(key, unit=unit, fallback=fallback)
    return AbAssignment(key, "a", dict(fallback or {}), "unavailable", None)


def choose_ab_test(key: str, *, unit: str, fallback: Params | None = None) -> AbDecision:
    if _shared:
        return _shared.choose(key, unit=unit, fallback=fallback)
    return AbDecision(key, unit, "a", dict(fallback or {}), "unavailable", None)


def record_ab_test_run(decision: AbDecision) -> bool:
    return bool(_shared and _shared.record_run(decision))


def complete_ab_test_decision(decision: AbDecision, trace_id: str) -> bool:
    return bool(_shared and _shared.complete_decision(decision, trace_id))


def log_exposure(
    key: str,
    *,
    unit_id: str,
    arm: ArmKey,
    trace_id: str | None = None,
    unit_complete: bool = False,
    event_id: str | None = None,
    allocation_version: int | None = None,
) -> bool:
    return bool(
        _shared
        and _shared.log_exposure(
            key,
            unit_id=unit_id,
            arm=arm,
            trace_id=trace_id,
            unit_complete=unit_complete,
            event_id=event_id,
            allocation_version=allocation_version,
        )
    )


def complete_ab_test_unit(
    key: str,
    *,
    unit_id: str,
    arm: ArmKey,
    trace_id: str,
    source: str = "sdk",
    allocation_version: int | None = None,
) -> bool:
    return bool(
        _shared
        and _shared.complete_unit(
            key,
            unit_id=unit_id,
            arm=arm,
            trace_id=trace_id,
            source=source,
            allocation_version=allocation_version,
        )
    )


def flush_ab_tests(timeout_seconds: float = 5) -> bool:
    return _shared.flush(timeout_seconds) if _shared else True


def close_ab_tests() -> None:
    global _shared
    if _shared:
        _shared.close()
    _shared = None
