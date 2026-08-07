"""
Replay processed or raw OpenTelemetry span JSONL logs as a human-readable tree.

When ``NEATLOGS_LOG_SPANS=true`` is set in the environment, the SDK writes one
JSON object per span to ``spans_optimized.log`` in the working directory (or
whatever path ``NEATLOGS_LOG_SPANS_FILE`` points to). When
``NEATLOGS_LOG_RAW_SPANS=true`` is set, the raw ``ReadableSpan.to_json()``
output is written to ``spans_raw_optimized.log``. Both formats are line-
delimited at the brace level. Newlines inside string values are possible, so
we brace-balance across the file when reading.

This module exposes a programmatic API and a CLI that read one or more paths,
reconstruct the parent/child tree per trace, and print a colored indented summary.
Programmatic::

    from neatlogs.replay import replay, format_tree
    trees = replay("spans_optimized.log")
    print(format_tree(trees))

CLI::

    neatlogs-replay path/to/spans_optimized.log
    neatlogs-replay --format json --max-depth 3 spans_optimized.log
    neatlogs-replay spans_a.log spans_b.log
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Union

PathLike = Union[str, os.PathLike]


@dataclasses.dataclass
class SpanRecord:
    """Normalized representation of a single span, in either the processed
    ``span_data`` shape (NEATLOGS_LOG_SPANS) or the raw
    ``ReadableSpan.to_json()`` shape (NEATLOGS_LOG_RAW_SPANS)."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    start_time: Optional[int]
    end_time: Optional[int]
    duration_ns: Optional[int]
    status_code: str
    status_description: str
    attributes: Dict[str, Any]
    source: str  # "processed" | "raw" | "unknown"

    def duration_ms(self) -> Optional[float]:
        if self.duration_ns is None:
            return None
        return round(self.duration_ns / 1_000_000, 3)


@dataclasses.dataclass
class TraceTree:
    """A reconstructed tree of spans for one trace_id."""

    trace_id: str
    roots: List["TreeNode"]

    def count(self) -> int:
        n = 0
        for r in self.roots:
            n += r.count()
        return n


@dataclasses.dataclass
class TreeNode:
    span: SpanRecord
    children: List["TreeNode"] = dataclasses.field(default_factory=list)

    def count(self) -> int:
        n = 1
        for c in self.children:
            n += c.count()
        return n


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def _strip_hex(value: Any) -> Optional[str]:
    """Normalize a hex trace/span id to a lowercase hex string with no 0x prefix.

    Returns None for None, empty strings, or non-hex values. OTel ids that arrive
    as ints are also accepted (some encoders coerce to int)."""
    if value is None:
        return None
    if isinstance(value, int):
        if value == 0:
            return None
        return format(value, "x")
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    if v.startswith("0x") or v.startswith("0X"):
        v = v[2:]
    v = v.lower()
    if not _HEX_RE.match(v):
        return None
    return v


def _coerce_ns(value: Any) -> Optional[int]:
    """OTel epoch-nanoseconds can arrive as int, float, ISO string, or None.

    Strings are only accepted when they parse as ISO 8601 with a timezone;
    floats and ints are passed through. The 0 sentinel (the OTel "no parent"
    convention) maps to None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # ISO 8601 with timezone
        try:
            from datetime import datetime

            ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return int(ts.timestamp() * 1_000_000_000)
        except Exception:
            return None
    return None


def _iter_json_objects(text: str) -> Iterator[Dict[str, Any]]:
    """Yield top-level JSON objects from a brace-balanced span-log file.

    The dojo's dump.py writes one object per line, but the SDK's
    ``spans_raw_optimized.log`` and ``spans_processed.log`` are written via
    ``json.dump`` without indent (raw) or with ``json.dumps(span_data) + "\\n"``
    (processed). Both end up with one object per line in practice, but
    newlines inside string values would break a strict line-splitter. The
    brace-balancer below handles both.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                chunk = text[start : i + 1]
                try:
                    yield json.loads(chunk)
                except json.JSONDecodeError:
                    pass
                start = -1
            elif depth < 0:
                depth = 0


def _detect_source(obj: Dict[str, Any]) -> str:
    """Return 'raw' for ReadableSpan.to_json(), 'processed' for span_data, 'unknown' otherwise."""
    if isinstance(obj.get("context"), dict) and "trace_id" in obj.get("context", {}):
        return "raw"
    if (
        "trace_id" in obj
        and "span_id" in obj
        and isinstance(obj.get("parent_span_id"), (str, type(None)))
    ):
        return "processed"
    return "unknown"


def _parse_processed(obj: Dict[str, Any]) -> Optional[SpanRecord]:
    trace_id = _strip_hex(obj.get("trace_id"))
    span_id = _strip_hex(obj.get("span_id"))
    if not trace_id or not span_id:
        return None
    parent = _strip_hex(obj.get("parent_span_id"))
    name = obj.get("name") or ""
    kind = obj.get("kind") or "UNKNOWN"
    start_time = _coerce_ns(obj.get("start_time"))
    end_time = _coerce_ns(obj.get("end_time"))
    if start_time is not None and end_time is not None and end_time >= start_time:
        duration_ns = end_time - start_time
    else:
        duration_ns = obj.get("duration_ns")
        if duration_ns is not None and duration_ns < 0:
            duration_ns = None
    status = obj.get("status") or {}
    status_code = ""
    status_description = ""
    if isinstance(status, dict):
        status_code = str(status.get("code") or "")
        status_description = str(status.get("description") or "")
    attributes = obj.get("attributes") or {}
    if not isinstance(attributes, dict):
        attributes = {}
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent if parent else None,
        name=name,
        kind=kind,
        start_time=start_time,
        end_time=end_time,
        duration_ns=duration_ns,
        status_code=status_code,
        status_description=status_description,
        attributes=attributes,
        source="processed",
    )


def _parse_raw(obj: Dict[str, Any]) -> Optional[SpanRecord]:
    ctx = obj.get("context")
    if not isinstance(ctx, dict):
        return None
    trace_id = _strip_hex(ctx.get("trace_id"))
    span_id = _strip_hex(ctx.get("span_id"))
    if not trace_id or not span_id:
        return None
    parent = _strip_hex(obj.get("parent_id"))
    name = obj.get("name") or ""
    kind = str(obj.get("kind") or "UNKNOWN")
    # ReadableSpan.to_json writes start_time / end_time as ISO 8601 strings.
    start_time = _coerce_ns(obj.get("start_time"))
    end_time = _coerce_ns(obj.get("end_time"))
    if start_time is not None and end_time is not None and end_time >= start_time:
        duration_ns = end_time - start_time
    else:
        duration_ns = None
    status = obj.get("status") or {}
    status_code = ""
    status_description = ""
    if isinstance(status, dict):
        status_code = str(status.get("status_code") or "")
        status_description = str(status.get("description") or "")
    attributes = obj.get("attributes") or {}
    if not isinstance(attributes, dict):
        attributes = {}
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent if parent else None,
        name=name,
        kind=kind,
        start_time=start_time,
        end_time=end_time,
        duration_ns=duration_ns,
        status_code=status_code,
        status_description=status_description,
        attributes=attributes,
        source="raw",
    )


def parse_line(line: str) -> Optional[SpanRecord]:
    """Parse a single JSON line and return a SpanRecord, or None if unparseable."""
    if not line or not line.strip():
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    source = _detect_source(obj)
    if source == "processed":
        return _parse_processed(obj)
    if source == "raw":
        return _parse_raw(obj)
    return None


def _read_file(path: PathLike) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


@dataclasses.dataclass
class ParseResult:
    spans: List[SpanRecord]
    malformed_lines: int
    files_read: int

    def total(self) -> int:
        return len(self.spans) + self.malformed_lines


def parse_paths(paths: Sequence[PathLike]) -> ParseResult:
    """Read all paths, brace-balance the JSON, and return SpanRecords.

    Counts malformed objects (JSON parse failures and unparseable shapes) but
    does not raise on them. Empty / missing paths contribute 0 spans and 0 errors.

    ``malformed_lines`` is the count of non-blank lines that did not produce a
    span. This includes JSON parse errors and lines whose shape we don't
    recognize. Blank lines are not counted.
    """
    spans: List[SpanRecord] = []
    malformed = 0
    files_read = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        files_read += 1
        text = _read_file(path)
        if not text.strip():
            continue
        # Non-blank lines are the malformed denominator: brace-balanced parsing
        # may yield fewer objects when a line is invalid JSON or has a
        # non-span shape.
        non_blank = sum(1 for line in text.splitlines() if line.strip())
        file_spans: List[SpanRecord] = []
        for obj in _iter_json_objects(text):
            source = _detect_source(obj)
            record: Optional[SpanRecord] = None
            if source == "processed":
                record = _parse_processed(obj)
            elif source == "raw":
                record = _parse_raw(obj)
            if record is not None:
                file_spans.append(record)
        spans.extend(file_spans)
        malformed += max(0, non_blank - len(file_spans))
    return ParseResult(spans=spans, malformed_lines=malformed, files_read=files_read)


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------


def build_trees(
    spans: Iterable[SpanRecord],
    *,
    warn_stream: Optional[TextIO] = None,
) -> List[TraceTree]:
    """Reconstruct TraceTrees from a flat span list.

    Each trace_id groups its own tree. Spans whose parent_span_id is None or
    that have a parent in a different trace become roots. Spans whose
    parent_span_id points to a missing span are wrapped under a synthetic
    ``_orphan`` root (named after the missing parent) so the trace renders and
    the warning surfaces in the output.
    """
    spans_by_id: Dict[str, SpanRecord] = {}
    children: Dict[str, List[SpanRecord]] = {}
    by_trace: Dict[str, List[SpanRecord]] = {}

    for s in spans:
        spans_by_id.setdefault(s.span_id, s)
        by_trace.setdefault(s.trace_id, []).append(s)
        if s.parent_span_id:
            children.setdefault(s.parent_span_id, []).append(s)

    def make_node(span: SpanRecord) -> TreeNode:
        node = TreeNode(span=span)
        for child in sorted(children.get(span.span_id, []), key=lambda c: c.start_time or 0):
            node.children.append(make_node(child))
        return node

    trees: List[TraceTree] = []
    for trace_id, members in sorted(by_trace.items(), key=lambda kv: kv[0]):
        root_nodes: List[TreeNode] = []
        seen_orphans: Dict[str, TreeNode] = {}
        for s in members:
            parent_id = s.parent_span_id
            if not parent_id or parent_id == "0" * len(parent_id):
                root_nodes.append(make_node(s))
                continue
            if parent_id not in spans_by_id:
                # Orphan: parent is missing in this log file. Park under a
                # synthetic root named after the missing parent.
                bucket = seen_orphans.get(parent_id)
                if bucket is None:
                    orphan = SpanRecord(
                        trace_id=s.trace_id,
                        span_id=parent_id,
                        parent_span_id=None,
                        name="_orphan",
                        kind="INTERNAL",
                        start_time=None,
                        end_time=None,
                        duration_ns=None,
                        status_code="",
                        status_description="",
                        attributes={
                            "_orphan_reason": "parent span not present in log file",
                            "_missing_parent_id": parent_id,
                        },
                        source="synthetic",
                    )
                    bucket = TreeNode(span=orphan)
                    seen_orphans[parent_id] = bucket
                    root_nodes.append(bucket)
                    if warn_stream is not None:
                        warn_stream.write(
                            f"[neatlogs-replay] warning: orphan span {s.span_id} "
                            f"(name={s.name!r}) references missing parent {parent_id}; "
                            f"wrapping under synthetic '_orphan' root.\n"
                        )
                bucket.children.append(make_node(s))
                continue
            # Parent in a different trace (very rare; usually a corrupted file):
            # treat as a root so it isn't silently dropped.
            if spans_by_id[parent_id].trace_id != s.trace_id:
                root_nodes.append(make_node(s))
        trees.append(TraceTree(trace_id=trace_id, roots=root_nodes))
    return trees


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


_USE_COLOR = True


def _supports_color(stream: TextIO) -> bool:
    if not _USE_COLOR:
        return False
    if not hasattr(stream, "isatty"):
        return False
    return bool(stream.isatty())


def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


_KIND_COLORS = {
    "LLM": "36",  # cyan
    "AGENT": "35",  # magenta
    "WORKFLOW": "33",  # yellow
    "CHAIN": "33",
    "TOOL": "32",  # green
    "RETRIEVER": "34",  # blue
    "EMBEDDING": "34",
    "VECTOR_STORE": "34",
    "RERANKER": "34",
    "GUARDRAIL": "31",  # red
    "MCP_TOOL": "33",
    "HTTP": "90",  # bright black / gray
    "INTERNAL": "37",
    "CLIENT": "90",
    "PRODUCER": "90",
    "CONSUMER": "90",
    "SERVER": "90",
}


def _normalize_kind(kind: Optional[str]) -> str:
    """Strip a leading ``SpanKind.`` / ``SPANKIND.`` prefix and uppercase.

    The raw ``ReadableSpan.to_json()`` exporter writes the full enum name
    (``"SpanKind.INTERNAL"``); the processed ``span_data`` shape uses a short
    lowercase string (``"llm"``, ``"tool"``). We display both consistently.
    """
    if not kind:
        return "?"
    k = kind.strip()
    if not k:
        return "?"
    for prefix in ("SpanKind.", "SPANKIND."):
        if k.startswith(prefix):
            k = k[len(prefix) :]
            break
    return k.upper() or "?"


def _kind_color(kind: str) -> Optional[str]:
    return _KIND_COLORS.get(_normalize_kind(kind))


def _format_node_text(node: TreeNode) -> str:
    """Render a single TreeNode as a single line."""
    span = node.span
    kind = _normalize_kind(span.kind)
    name = span.name or "?"
    ms = span.duration_ms()
    ms_str = f"  ({ms} ms)" if ms is not None else ""
    extra: List[str] = []
    if span.status_code and span.status_code not in ("OK", "UNSET"):
        extra.append(f"status={span.status_code}")
    if span.attributes.get("neatlogs.llm.model_name"):
        model = span.attributes["neatlogs.llm.model_name"]
        if isinstance(model, str) and model:
            extra.append(f"model={model}")
    if span.attributes.get("neatlogs.llm.token_count.total"):
        toks = span.attributes["neatlogs.llm.token_count.total"]
        extra.append(f"tokens={toks}")
    extra_str = f"  [{', '.join(extra)}]" if extra else ""
    line = f"[{kind:7s}] {name}{ms_str}{extra_str}"
    return line


def format_tree(
    trees: Sequence[TraceTree],
    *,
    style: str = "text",
    max_depth: Optional[int] = None,
    filter_kind: Optional[str] = None,
    use_color: Optional[bool] = None,
    stream: Optional[TextIO] = None,
) -> str:
    """Render one or more TraceTrees as a string.

    style:
        - "text": colored indented tree (the default).
        - "json": one JSON object per trace with the full tree.

    max_depth truncates the tree at the given depth; spans below are still
    counted in totals but not rendered.

    filter_kind keeps only the spans whose kind matches (case-insensitive).
    """
    if style not in ("text", "json"):
        raise ValueError(f"unknown style: {style!r}")
    sink = stream or sys.stdout
    if use_color is None:
        use_color = _supports_color(sink)
    if style == "json":
        return _format_json(trees, max_depth=max_depth, filter_kind=filter_kind)
    return _format_text(trees, max_depth=max_depth, filter_kind=filter_kind, use_color=use_color)


def _format_json(
    trees: Sequence[TraceTree],
    *,
    max_depth: Optional[int],
    filter_kind: Optional[str],
) -> str:
    out: List[str] = []
    for tree in trees:
        out.append(
            json.dumps(
                {
                    "trace_id": tree.trace_id,
                    "span_count": tree.count(),
                    "roots": [
                        _node_to_dict(r, max_depth=max_depth, filter_kind=filter_kind)
                        for r in tree.roots
                    ],
                },
                default=str,
            )
        )
    return "\n".join(out) + ("\n" if out else "")


def _node_to_dict(
    node: TreeNode,
    *,
    max_depth: Optional[int],
    filter_kind: Optional[str],
    depth: int = 0,
) -> Dict[str, Any]:
    keep = filter_kind is None or _normalize_kind(node.span.kind) == filter_kind.upper()
    children: List[Dict[str, Any]] = []
    if max_depth is None or depth < max_depth:
        for child in node.children:
            children.append(
                _node_to_dict(
                    child,
                    max_depth=max_depth,
                    filter_kind=filter_kind,
                    depth=depth + 1 if keep else depth + 1,
                )
            )
    if not keep and not children:
        return {"_pruned": True}
    if not keep:
        return {"_pruned": True, "children": children}
    return {
        "trace_id": node.span.trace_id,
        "span_id": node.span.span_id,
        "parent_span_id": node.span.parent_span_id,
        "name": node.span.name,
        "kind": node.span.kind,
        "start_time": node.span.start_time,
        "end_time": node.span.end_time,
        "duration_ms": node.span.duration_ms(),
        "status": {
            "code": node.span.status_code,
            "description": node.span.status_description,
        },
        "attributes": node.span.attributes,
        "source": node.span.source,
        "children": children,
    }


def _format_text(
    trees: Sequence[TraceTree],
    *,
    max_depth: Optional[int],
    filter_kind: Optional[str],
    use_color: bool,
) -> str:
    buf = io.StringIO()
    if not trees:
        buf.write("(no spans found)\n")
        return buf.getvalue()
    for tree in trees:
        header = f"=== trace {tree.trace_id}  ({tree.count()} span{'s' if tree.count() != 1 else ''}) ==="
        if use_color:
            header = _ansi("1;33", header)
        buf.write(header + "\n")
        if not tree.roots:
            buf.write("  (no root span: every span points to a missing parent)\n\n")
            continue
        for root in tree.roots:
            _render_text_node(
                buf,
                root,
                depth=0,
                max_depth=max_depth,
                filter_kind=filter_kind,
                use_color=use_color,
            )
        buf.write("\n")
    return buf.getvalue()


def _render_text_node(
    buf: io.StringIO,
    node: TreeNode,
    *,
    depth: int,
    max_depth: Optional[int],
    filter_kind: Optional[str],
    use_color: bool,
) -> None:
    if max_depth is not None and depth > max_depth:
        return
    span = node.span
    keep = filter_kind is None or _normalize_kind(span.kind) == filter_kind.upper()
    indent = "  " * depth
    if keep:
        line = _format_node_text(node)
        if use_color:
            color = _kind_color(span.kind)
            if color is not None:
                line = _ansi(color, line)
        buf.write(indent + line + "\n")
    child_indent_depth = depth + 1
    if max_depth is not None and child_indent_depth > max_depth:
        return
    for child in node.children:
        _render_text_node(
            buf,
            child,
            depth=child_indent_depth,
            max_depth=max_depth,
            filter_kind=filter_kind,
            use_color=use_color,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def replay(
    paths: Union[PathLike, Sequence[PathLike]],
    *,
    max_depth: Optional[int] = None,
    filter_kind: Optional[str] = None,
    warn_stream: Optional[TextIO] = None,
) -> List[TraceTree]:
    """Read span log files and return a list of reconstructed TraceTrees.

    Args:
        paths: One path, a list of paths, or a single globbed path. Missing
            files are skipped silently.
        max_depth: If set, the printed tree is truncated at this depth; counts
            still include deeper spans.
        filter_kind: If set, only render spans whose ``kind`` matches
            (case-insensitive). Useful to focus on a single layer.
        warn_stream: Where to write orphan-span warnings. Defaults to stderr.
            Pass ``io.StringIO()`` to silence.
    """
    if isinstance(paths, (str, os.PathLike)):
        seq: List[PathLike] = [paths]
    else:
        seq = list(paths)
    parsed = parse_paths(seq)
    return build_trees(parsed.spans, warn_stream=warn_stream or sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expand_paths(argv: Sequence[str]) -> List[PathLike]:
    out: List[PathLike] = []
    for a in argv:
        out.append(a)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neatlogs-replay",
        description=(
            "Replay processed or raw OpenTelemetry span JSONL logs as a tree. "
            "Read span log files written by NEATLOGS_LOG_SPANS or "
            "NEATLOGS_LOG_RAW_SPANS and print the parent/child hierarchy."
        ),
    )
    p.add_argument(
        "paths",
        nargs="+",
        help="One or more span log file paths.",
    )
    p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. 'text' is a colored indented tree; 'json' is one JSON object per trace.",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Truncate the rendered tree at this depth (counts still include deeper spans).",
    )
    p.add_argument(
        "--filter-kind",
        default=None,
        help="Render only spans whose kind matches this value (case-insensitive).",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output even when stdout is a TTY.",
    )
    p.add_argument(
        "--color",
        action="store_true",
        help="Force ANSI color output even when stdout is not a TTY.",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="After the tree, print a one-line summary per trace: span count, kind breakdown.",
    )
    return p


def _summary_line(tree: TraceTree) -> str:
    kinds: Dict[str, int] = {}

    def walk(node: TreeNode) -> None:
        kind = _normalize_kind(node.span.kind)
        kinds[kind] = kinds.get(kind, 0) + 1
        for c in node.children:
            walk(c)

    for r in tree.roots:
        walk(r)
    parts = [f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))]
    return f"  trace {tree.trace_id}: {tree.count()} spans; " + ", ".join(parts)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False
    elif args.color:
        _USE_COLOR = True
    paths = _expand_paths(args.paths)
    try:
        trees = replay(
            paths,
            max_depth=args.max_depth,
            filter_kind=args.filter_kind,
        )
    except FileNotFoundError as exc:
        print(f"[neatlogs-replay] error: {exc}", file=sys.stderr)
        return 2
    if not trees:
        print("[neatlogs-replay] no spans found", file=sys.stderr)
    out = format_tree(
        trees,
        style=args.format,
        max_depth=args.max_depth,
        filter_kind=args.filter_kind,
    )
    sys.stdout.write(out)
    if args.summary and args.format == "text":
        for tree in trees:
            sys.stdout.write(_summary_line(tree) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
