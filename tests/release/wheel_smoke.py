"""Installed-wheel quick start covering sync, async, generator and re-init."""

import asyncio

import neatlogs


@neatlogs.span(kind="WORKFLOW")
def sync_workflow(value: str) -> str:
    return value.upper()


@neatlogs.span(kind="CHAIN")
async def async_workflow(value: str) -> str:
    await asyncio.sleep(0)
    return value[::-1]


@neatlogs.span(kind="CHAIN")
def streaming_workflow():
    yield "one"
    yield "two"


def main() -> None:
    neatlogs.verify_telemetry_schema()
    neatlogs.init(
        api_key="wheel-smoke",
        disable_export=True,
        instrumentations=[],
        register_shutdown_handlers=False,
    )
    assert sync_workflow("ready") == "READY"
    assert asyncio.run(async_workflow("ready")) == "ydaer"
    assert list(streaming_workflow()) == ["one", "two"]
    assert neatlogs.flush(timeout_millis=1000)
    assert neatlogs.shutdown(timeout_millis=1000)

    # A completed generation must be safely re-initializable in the same process.
    neatlogs.init(
        api_key="wheel-smoke",
        disable_export=True,
        instrumentations=[],
        register_shutdown_handlers=False,
    )
    assert neatlogs.shutdown(timeout_millis=1000)


if __name__ == "__main__":
    main()
