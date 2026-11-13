import asyncio
import importlib
from uuid import uuid4

import pytest

FakeMessagesListChatModel = pytest.importorskip(
    "langchain_core.language_models.fake_chat_models"
).FakeMessagesListChatModel
AIMessage = pytest.importorskip("langchain_core.messages").AIMessage
RunnableLambda = pytest.importorskip("langchain_core.runnables").RunnableLambda


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_right", [False, True])
async def test_explicit_model_callbacks_keep_task_local_node_parent(
    monkeypatch, tracer_provider, in_memory_span_exporter, fail_right
):
    integration = importlib.import_module("neatlogs.langchain")
    monkeypatch.setattr(
        integration, "get_tracer", lambda: tracer_provider.get_tracer("neatlogs.test")
    )
    monkeypatch.setattr(integration, "_auto_root_enabled", lambda: False)
    handler = integration.NeatlogsCallbackHandler()

    def branch(name):
        async def invoke(_):
            await asyncio.sleep(0)
            if fail_right and name == "right":
                raise ValueError("node failure")
            model = FakeMessagesListChatModel(responses=[AIMessage(content=name)])
            return await model.ainvoke(name, config={"callbacks": [handler]})

        return RunnableLambda(invoke).with_config(
            callbacks=[handler], metadata={"langgraph_node": name}
        )

    await asyncio.gather(
        branch("left").ainvoke({}), branch("right").ainvoke({}), return_exceptions=True
    )
    standalone = FakeMessagesListChatModel(responses=[AIMessage(content="standalone")])
    await standalone.ainvoke("standalone", config={"callbacks": [handler]})

    spans = in_memory_span_exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans}
    models = [span for span in spans if span.attributes.get("neatlogs.span.kind") == "llm"]
    assert len(models) == (2 if fail_right else 3)
    for span in models:
        prompt = span.attributes["neatlogs.llm.input_messages.0.content"]
        if prompt == "standalone":
            assert span.parent is None
        else:
            assert span.parent is not None
            assert by_id[span.parent.span_id].name == prompt
    assert handler._active_chains.get() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("manual_parent", [False, True])
async def test_node_parent_preserves_newer_manual_scope(
    monkeypatch, tracer_provider, in_memory_span_exporter, isolated, manual_parent
):
    integration = importlib.import_module("neatlogs.langchain")
    utils = importlib.import_module("neatlogs._wrap_utils")
    tracer = tracer_provider.get_tracer("neatlogs.test")
    monkeypatch.setattr(integration, "get_tracer", lambda: tracer)
    monkeypatch.setattr(integration, "_auto_root_enabled", lambda: False)
    monkeypatch.setattr(utils, "_isolation_active", lambda: isolated)
    handler = integration.NeatlogsCallbackHandler()
    node_id = uuid4()
    workflow = tracer.start_span("workflow")
    workflow_token = utils.attach_as_current(workflow)
    try:
        await handler.on_chain_start({}, {}, run_id=node_id, name="node")
        manual = tracer.start_span("manual") if manual_parent else workflow
        manual_token = utils.attach_as_current(manual)
        try:
            model = FakeMessagesListChatModel(responses=[AIMessage(content="reply")])
            await model.ainvoke("question", config={"callbacks": [handler]})
        finally:
            utils.detach(manual_token)
            if manual_parent:
                manual.end()
        node = handler._spans[node_id]
        await handler.on_chain_end({}, run_id=node_id)
    finally:
        utils.detach(workflow_token)
        workflow.end()

    model_span = next(
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.attributes.get("neatlogs.span.kind") == "llm"
    )
    expected_parent = manual if manual_parent else node
    assert model_span.parent.span_id == expected_parent.get_span_context().span_id


@pytest.mark.asyncio
async def test_explicit_callback_parent_overrides_task_local_node(
    monkeypatch, tracer_provider, in_memory_span_exporter
):
    integration = importlib.import_module("neatlogs.langchain")
    monkeypatch.setattr(
        integration, "get_tracer", lambda: tracer_provider.get_tracer("neatlogs.test")
    )
    handler = integration.NeatlogsCallbackHandler()
    outer, inner, model = uuid4(), uuid4(), uuid4()
    await handler.on_chain_start({}, {}, run_id=outer, metadata={"langgraph_node": "outer"})
    await handler.on_chain_start(
        {}, {}, run_id=inner, parent_run_id=outer, metadata={"langgraph_node": "inner"}
    )
    await handler.on_chat_model_start({}, [], run_id=model, parent_run_id=outer)
    await handler.on_llm_error(ValueError("model failure"), run_id=model)
    await handler.on_chain_end({}, run_id=inner)
    await handler.on_chain_end({}, run_id=outer)

    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    assert spans["langchain.chat_model"].parent.span_id == spans["outer"].context.span_id


@pytest.mark.asyncio
@pytest.mark.parametrize("isolated", [False, True])
async def test_trace_context_parents_custom_spans_and_restores_after_error(
    monkeypatch, tracer_provider, in_memory_span_exporter, isolated
):
    import neatlogs

    integration = importlib.import_module("neatlogs.langchain")
    utils = importlib.import_module("neatlogs._wrap_utils")
    tracer = tracer_provider.get_tracer("neatlogs.test")
    monkeypatch.setattr(integration, "get_tracer", lambda: tracer)
    monkeypatch.setattr(
        utils, "get_internal_tracer", lambda *args: utils._ForeignParentGuardTracer(tracer)
    )
    monkeypatch.setattr(utils, "_isolation_active", lambda: isolated)
    handler = integration.NeatlogsCallbackHandler()

    @neatlogs.span(kind="EMBEDDING", name="embed")
    def embed():
        return [0.1]

    @neatlogs.span(kind="RETRIEVER", name="search")
    def search():
        with handler.trace_context():
            return embed()

    workflow = tracer.start_span("workflow")
    token = utils.attach_as_current(workflow)
    node_id = uuid4()
    try:
        await handler.on_chain_start({}, {}, run_id=node_id, metadata={"langgraph_node": "node"})
        with pytest.raises(ValueError, match="operation failed"):
            with handler.trace_context():
                search()
                await asyncio.sleep(0)
                raise ValueError("operation failed")
        parent = (
            utils._current_neatlogs_parent()
            if isolated
            else integration.otel_trace.get_current_span()
        )
        assert parent.get_span_context() == workflow.get_span_context()
        await handler.on_chain_end({}, run_id=node_id)
    finally:
        utils.detach(token)
        workflow.end()

    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    assert spans["search"].parent.span_id == spans["node"].context.span_id
    assert spans["embed"].parent.span_id == spans["search"].context.span_id
