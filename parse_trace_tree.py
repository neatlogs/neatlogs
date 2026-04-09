"""
Parse raw_spans.log and spans.log to build and print trace trees.
Usage: python parse_trace_tree.py
"""
import json

SDK_LOG = "langgraph_multiagent_spans.log"
TRACE_ID_FILTER = "1bf024a6b5fd1125a197f2d8cb63da86"


def load_spans(path):
    spans = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                spans.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return spans


def ga(span, key, default=None):
    """Get attribute."""
    return span.get("attributes", {}).get(key, default)


def sid(span):
    return span.get("span_id", "")


def pid(span):
    return span.get("parent_span_id") or span.get("parent_id") or None


def flags(span):
    f = []
    kind = ga(span, "neatlogs.span.kind", "")
    if kind:
        f.append(kind.upper())

    if ga(span, "neatlogs.internal"):
        f.append("INTERNAL")

    model = ga(span, "neatlogs.llm.model_name") or ga(span, "llm.model_name", "")
    if model:
        f.append(f"model={model}")

    attrs = span.get("attributes", {})
    has_in = any(("input" in k.lower() and "token" not in k.lower() and "mime" not in k.lower()
                  and attrs[k]) for k in attrs)
    has_out = any(("output" in k.lower() and "token" not in k.lower() and "mime" not in k.lower()
                   and attrs[k]) for k in attrs)
    f.append("IN" if has_in else "no-in")
    f.append("OUT" if has_out else "no-out")

    if any("prompt_template" in k and "user" not in k and "variable" not in k and attrs[k] for k in attrs):
        f.append("SYS_TPL")
    if any("user_prompt_template" in k and "variable" not in k and attrs[k] for k in attrs):
        f.append("USR_TPL")
    if any("template_variables" in k and attrs[k] for k in attrs):
        f.append("VARS")

    instr = ga(span, "neatlogs.instrumentation.name", "")
    if instr:
        short = instr.split(".")[-1] if "." in instr else instr
        f.append(f"@{short}")

    return f


def print_tree(by_id, children, nid, indent=0):
    s = by_id[nid]
    pre = "  " * indent + ("|- " if indent > 0 else "")
    f = flags(s)
    print(f"{pre}{s['name']} [{sid(s)[-16:]}] ({', '.join(f)})")
    for cid in sorted(children.get(nid, []), key=lambda x: by_id[x].get("start_time", 0)):
        print_tree(by_id, children, cid, indent + 1)


def main():
    spans = load_spans(SDK_LOG)
    filtered = [s for s in spans if s.get("trace_id") == TRACE_ID_FILTER]
    print(f"Total spans: {len(spans)}, for trace: {len(filtered)}")

    by_id = {sid(s): s for s in filtered}
    children = {}
    roots = []
    all_sids = set(by_id.keys())

    for s in filtered:
        p = pid(s)
        if p and p in all_sids:
            children.setdefault(p, []).append(sid(s))
        else:
            roots.append(sid(s))

    # Sort roots by start time
    roots.sort(key=lambda x: by_id[x].get("start_time", 0))

    print(f"Roots: {len(roots)}")
    print()

    # ================================================================
    # TREE
    # ================================================================
    print("=" * 120)
    print("  TRACE TREE")
    print("=" * 120)
    for r in roots:
        print_tree(by_id, children, r)
    print()

    # ================================================================
    # SPECIFIC SPANS
    # ================================================================
    check = {
        "b018934857448b2e": "User issue: missing prompt template",
        "58bb3684e5054079": "User issue: wrong parent in simplified",
        "505ada23638892fe": "Expected parent of 58bb3684e5054079",
    }
    print("=" * 120)
    print("  SPECIFIC SPAN DETAILS")
    print("=" * 120)
    for target, desc in check.items():
        matches = [s for s in filtered if sid(s).endswith(target)]
        for s in matches:
            p = pid(s)
            parent_name = by_id[p]["name"] if p and p in by_id else "(ORPHAN/ROOT)"
            print(f"\n  [{desc}]")
            print(f"    name:       {s['name']}")
            print(f"    span_id:    {sid(s)}")
            print(f"    parent:     {p} -> {parent_name}")
            print(f"    kind:       {s.get('kind', '')}")
            print(f"    flags:      {', '.join(flags(s))}")
            print(f"    attr keys:")
            for k in sorted(s.get("attributes", {}).keys()):
                v = str(s["attributes"][k])[:100]
                print(f"      {k}: {v}")

    # ================================================================
    # INTERNAL vs NON-INTERNAL PAIRS
    # ================================================================
    print()
    print("=" * 120)
    print("  INTERNAL SPAN PAIRS (neatlogs.trace + OpenInference)")
    print("=" * 120)
    internal_spans = [s for s in filtered if ga(s, "neatlogs.internal")]
    for s in sorted(internal_spans, key=lambda x: x.get("start_time", 0)):
        my_children = [by_id[c] for c in children.get(sid(s), []) if c in by_id]
        print(f"\n  INTERNAL: {s['name']} [{sid(s)[-16:]}]")
        print(f"    parent: {pid(s)} -> {by_id[pid(s)]['name'] if pid(s) and pid(s) in by_id else 'ROOT'}")
        print(f"    has_tpl: SYS={bool(ga(s,'neatlogs.llm.prompt_template'))} USR={bool(ga(s,'neatlogs.llm.user_prompt_template'))} VARS={bool(ga(s,'neatlogs.llm.user_prompt_template_variables'))}")
        for c in my_children:
            print(f"    CHILD: {c['name']} [{sid(c)[-16:]}] instrumented_by={ga(c,'neatlogs.instrumentation.name','?')}")

    # ================================================================
    # ORPHAN ANALYSIS
    # ================================================================
    print()
    print("=" * 120)
    print("  ORPHAN / ROOT SPAN ANALYSIS")
    print("=" * 120)
    for r in roots:
        s = by_id[r]
        p = pid(s)
        print(f"\n  ROOT: {s['name']} [{sid(s)[-16:]}]")
        print(f"    declared parent: {p}")
        print(f"    parent exists in trace: {p in all_sids if p else 'N/A (true root)'}")
        print(f"    instrumentation: {ga(s, 'neatlogs.instrumentation.name', 'unknown')}")
        print(f"    internal: {ga(s, 'neatlogs.internal', False)}")
        child_count = len(children.get(sid(s), []))
        print(f"    children: {child_count}")


if __name__ == "__main__":
    main()
