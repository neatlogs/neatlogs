"""Explicit W3C propagation for the private Neatlogs trace context.

Port of the TypeScript SDK's ``core/propagation.ts``. The global OpenTelemetry
propagator is deliberately not used or changed: callers opt in at the exact
cross-process boundary where Neatlogs context should travel.

Why an explicit helper is needed at all: every Neatlogs SDK runs on a PRIVATE
provider/context and never populates the process-global OTel context. So a bare
``opentelemetry.propagate.inject(headers)`` — which reads the global context —
writes nothing while a Neatlogs span is active. This helper resolves the active
Neatlogs span from its private context key and injects THAT.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional

from opentelemetry import context as context_api
from opentelemetry import trace as otel_trace
from opentelemetry.propagate import extract, inject
from opentelemetry.propagators.textmap import default_getter


def _has_traceparent(carrier: Any) -> bool:
    try:
        for k in carrier.keys():
            if str(k).lower() == "traceparent":
                return True
    except Exception:
        pass
    return False


def inject_trace_context(carrier: MutableMapping[str, str]) -> bool:
    """Inject the active private Neatlogs span into ``carrier`` as W3C
    ``traceparent``/``tracestate`` headers.

    ``carrier`` is any mutable header mapping (a plain ``dict``, a ``requests``
    ``CaseInsensitiveDict``, etc.). Call it right before an outbound request so
    the remote service can continue the same trace:

        headers = {"content-type": "application/json"}
        neatlogs.inject_trace_context(headers)
        requests.post(url, headers=headers, json=payload)

    Returns ``True`` when a Neatlogs span was active and headers were written,
    ``False`` when there is no active Neatlogs span (nothing to propagate) — the
    carrier is left untouched. Mirrors the TypeScript SDK's ``injectTraceContext``.
    """
    if not isinstance(carrier, MutableMapping):
        return False
    if _has_traceparent(carrier):
        # An upstream traceparent is already present — don't clobber it.
        return True
    try:
        from .._wrap_utils import active_neatlogs_context

        ctx = active_neatlogs_context()
        if ctx is None:
            return False
        before = len(carrier)
        inject(carrier, context=ctx)
        # inject() writes traceparent (and optionally tracestate) when the context
        # carries a valid, sampled span context.
        return _has_traceparent(carrier) or len(carrier) > before
    except Exception:
        return False


@contextmanager
def extract_trace_context(
    carrier: Mapping[str, Any],
    *,
    session_id: Optional[str] = None,
    end_user_id: Optional[str] = None,
    end_user_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    """Continue a remote Neatlogs trace from an inbound ``carrier`` (W3C
    ``traceparent``/``tracestate`` headers) for the enclosed block.

    Call it on the RECEIVING side, right after a request arrives, so the next
    ``trace()`` / ``@span`` / ``wrap()`` root nests under the caller's span and
    shares its ``trace_id``:

        @app.post("/tool")
        def handle(req):
            with neatlogs.extract_trace_context(req.headers,
                                                session_id=req.session_id,
                                                end_user_id=req.user_id):
                with neatlogs.trace("do_tool"):   # joins the caller's trace
                    ...

    Identity does NOT ride the wire — the ``traceparent`` carries trace/span
    linkage only — so re-bind ``session_id`` / ``end_user_id`` here from the
    request payload (mirrors the Go SDK's ``ExtractTraceContext`` + ``Identify``).

    Isolation-safe: the remote parent is threaded on Neatlogs' PRIVATE context key
    (never the OTel global current-span), so a co-tenant instrumentor (Datadog /
    langfuse / openlit) neither sees it nor parents from it. When the carrier has
    no valid ``traceparent`` this is a no-op passthrough (identity re-binding, if
    any, still applies)."""
    from .._wrap_utils import (
        _is_neatlogs_span,
        _isolation_active,
        set_neatlogs_span_in_context,
    )
    from .identity import identify

    tokens = []
    try:
        # Extract the remote span into a bare context (never the active one).
        remote_ctx = extract(carrier, context=context_api.Context(), getter=default_getter)
        remote_span = otel_trace.get_current_span(remote_ctx)
        sc = remote_span.get_span_context()

        if sc.is_valid:
            if _isolation_active():
                # Thread the remote parent on the PRIVATE key so neatlogs children
                # resolve it, while the global current-span stays the host's.
                # force_owned: the remote span's scope isn't "neatlogs.*", but it
                # IS our cross-process parent, so treat it as owned.
                ctx = set_neatlogs_span_in_context(remote_span, force_owned=True)
            else:
                # DEFAULT mode: a single provider, so making the remote span the
                # OTel current-span is correct and also flags a neatlogs ancestor.
                ctx = set_neatlogs_span_in_context(
                    remote_span,
                    context=otel_trace.set_span_in_context(remote_span),
                    force_owned=not _is_neatlogs_span(remote_span),
                )
            tokens.append(context_api.attach(ctx))

        # Re-bind identity for the enclosed block (no-op if all None).
        with identify(
            session_id=session_id,
            end_user_id=end_user_id,
            end_user_metadata=end_user_metadata,
        ):
            yield
    finally:
        for token in reversed(tokens):
            try:
                context_api.detach(token)
            except Exception:
                pass
