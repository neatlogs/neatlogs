"""Reconcile ambiguous requests using private idempotency keys and owned status reads."""

import time

from ..replay.api import LocalRunApiError


def terminal(status):
    return status.get("state") in {
        "completed",
        "partial",
        "failed",
        "cancelled",
        "timed_out",
    }


def is_uncertain(error):
    return isinstance(error, LocalRunApiError) and (
        error.status == 504
        or error.code
        in {
            "RUN_API_TRANSPORT_FAILURE",
            "IDEMPOTENCY_REQUEST_IN_PROGRESS",
            "IDEMPOTENCY_STORE_UNAVAILABLE",
        }
    )


def recover(operation, deadline, *, run_api=None, run_id=None):
    """Return the request result or the terminal run observed after an uncertain write."""

    def timeout():
        return max(0.001, min(30, deadline - time.monotonic()))

    while True:
        try:
            return operation(timeout()), None
        except LocalRunApiError as error:
            closed = error.code == "RUN_CLOSED"
            if not (is_uncertain(error) or closed):
                raise
            if run_id:
                try:
                    # A mutation can consume the deadline before the final status read.
                    status = run_api.status(run_id, timeout=max(1, timeout()))
                    if terminal(status):
                        return None, status
                except LocalRunApiError as probe_error:
                    if not is_uncertain(probe_error):
                        raise
            if closed and run_id:
                raise LocalRunApiError(
                    "The closed run status could not be confirmed.",
                    code="RUN_API_TRANSPORT_FAILURE",
                ) from error
            remaining = deadline - time.monotonic()
            if closed or remaining <= 0:
                raise
            time.sleep(min(1, remaining))


def uncertainty_message(run_id, state_path):
    run = f"Run {run_id}" if run_id else "Run registration"
    hint = f' Use --resume --state "{state_path}" to reconcile the saved run.' if state_path else ""
    return LocalRunApiError(
        f"{run} could not be confirmed before its deadline.{hint}",
        code="RUN_API_TRANSPORT_FAILURE",
    )
