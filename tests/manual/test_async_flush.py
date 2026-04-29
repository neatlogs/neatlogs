"""
TEST 1: Async flush/shutdown pattern
Covers troubleshooting.md lines 145-161 (Async Gotcha section).

What to verify:
  - `asyncio.to_thread(neatlogs.flush)` and `asyncio.to_thread(neatlogs.shutdown)`
    complete without raising or blocking the event loop.
  - The span created inside the async function appears in the NeatLogs dashboard
    under workflow "test-async-flush".

Run:
    NEATLOGS_API_KEY=<your-key> python tests/manual/test_async_flush.py

Expected output (no errors):
    [async_flush] span created
    [async_flush] flush done
    [async_flush] shutdown done
    PASS
"""

import asyncio
import os

import neatlogs


async def main():
    neatlogs.init(
        api_key=None,  # reads NEATLOGS_API_KEY from env
        endpoint=os.environ.get("NEATLOGS_ENDPOINT", "https://staging-cloud.neatlogs.com"),
        workflow_name="test-async-flush",
    )

    @neatlogs.span(kind="CHAIN")
    def do_work():
        return "hello"

    result = do_work()
    print(f"[async_flush] span created, result={result!r}")

    # Pattern from troubleshooting.md — must NOT block event loop
    await asyncio.to_thread(neatlogs.flush)
    print("[async_flush] flush done")

    await asyncio.to_thread(neatlogs.shutdown)
    print("[async_flush] shutdown done")

    print("PASS")


asyncio.run(main())
