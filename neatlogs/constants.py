"""Stable SDK-wide defaults shared by every NeatLogs entry point."""

DEFAULT_INGEST_ENDPOINT = "https://ingest.neatlogs.com"
DEFAULT_MAX_QUEUE_ITEMS = 2048


def export_queue_capacity(batch_size: int) -> int:
    """Keep queues bounded while ensuring a configured batch always fits."""

    return max(DEFAULT_MAX_QUEUE_ITEMS, batch_size * 4)
