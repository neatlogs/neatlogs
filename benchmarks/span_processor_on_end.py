"""Repeatable debug-off span-processor microbenchmark.

Run from the repository root:
    python benchmarks/span_processor_on_end.py
"""

from __future__ import annotations

import json
import statistics
import time

from opentelemetry.sdk.trace import TracerProvider

from neatlogs.core.span_processor import NeatlogsSpanProcessor


def run_once(span_count: int = 5000) -> float:
    provider = TracerProvider()
    processor = NeatlogsSpanProcessor(debug=False, emit_completion_markers=False)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("neatlogs.benchmark")
    started = time.perf_counter()
    for index in range(span_count):
        span = tracer.start_span("benchmark")
        span.set_attribute("neatlogs.span.kind", "chain")
        span.set_attribute("input.value", f"input-{index}")
        span.add_event("chunk", {"index": index, "payload": "x" * 128})
        span.set_attribute("output.value", f"output-{index}")
        span.end()
    elapsed = time.perf_counter() - started
    provider.shutdown()
    return elapsed * 1_000_000 / span_count


def main() -> None:
    samples = [run_once() for _ in range(5)]
    print(
        json.dumps(
            {
                "span_count_per_sample": 5000,
                "samples_us_per_span": [round(value, 3) for value in samples],
                "median_us_per_span": round(statistics.median(samples), 3),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
