"""
Anthropic Event Handler
======================

Handles Anthropic specific API patterns and response formats.
Enhanced with tracking including streaming support.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from .base import BaseEventHandler
from ..semconv import (
    LLMAttributes, MessageAttributes, LLMRequestTypeValues, LLMEvents,
    format_tools_for_attribute, extract_tool_calls_data
)
from ..token_counting import estimate_cost


class AnthropicHandler(BaseEventHandler):
    """Event handler specifically for Anthropic Claude, with streaming support"""

    def extract_request_params(self, *args, **kwargs) -> Dict[str, Any]:
        params = super().extract_request_params(*args, **kwargs)
        params.update({
            'system_prompt': kwargs.get('system'),
            'tools': kwargs.get('tools'),
            'tool_choice': kwargs.get('tool_choice')
        })
        return params

    def extract_messages(self, *args, **kwargs) -> List[Dict[str, str]]:
        """Extract messages from Anthropic function arguments"""
        messages = kwargs.get('messages', [])
        if not isinstance(messages, list):
            return []

        extracted = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')

                if isinstance(content, str):
                    extracted.append({'role': role, 'content': content})
                elif isinstance(content, list):
                    # Handle multi-modal content
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_content += item.get('text', '') + " "
                        elif isinstance(item, dict) and item.get('type') == 'tool_result':
                            text_content += f"[Tool Result: {item.get('content', '')}] "
                    extracted.append(
                        {'role': role, 'content': text_content.strip()})
        return extracted

    def extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extract data from Anthropic response object"""
        data = {
            'completion': '',
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'model': None,
            'finish_reason': None,
            'tool_calls': []
        }

        try:
            # Extract completion text
            if hasattr(response, 'content') and response.content:
                text_parts = [
                    block.text for block in response.content if hasattr(block, 'text')]
                data['completion'] = "\n".join(text_parts)

                # Extract tool calls
                data['tool_calls'] = extract_tool_calls_data(response.content)

            # Extract usage information
            if hasattr(response, 'usage'):
                usage = response.usage
                data['prompt_tokens'] = getattr(usage, 'input_tokens', 0)
                data['completion_tokens'] = getattr(usage, 'output_tokens', 0)
                data['total_tokens'] = data['prompt_tokens'] + \
                    data['completion_tokens']

            # Extract model information
            if hasattr(response, 'model'):
                data['model'] = response.model

            # Extract stop reason
            if hasattr(response, 'stop_reason'):
                data['finish_reason'] = response.stop_reason

        except Exception as e:
            logging.warning(f"Error extracting Anthropic response data: {e}")

        return data

    def handle_call_start(self, span: 'LLMSpan', *args, **kwargs):
        super().handle_call_start(span, *args, **kwargs)

    # --- Streaming Support ---

    def wrap_stream_method(self, original_method, provider: str):
        from functools import wraps
        from ..core import current_span_id_context

        @wraps(original_method)
        def wrapped(*args, **kwargs):
            model = kwargs.get('model', 'unknown')
            span = self.create_span(
                model=model, provider=provider, operation="llm_stream", node_type="llm_call", node_name=model)

            self.handle_call_start(span, *args, **kwargs)

            token = current_span_id_context.set(span.span_id)
            try:
                stream = original_method(*args, **kwargs)
                return self.handle_stream_response(span, stream, token)
            except Exception as e:
                current_span_id_context.reset(token)
                self.handle_call_end(span, None, success=False, error=e)
                raise

        return wrapped

    def handle_stream_response(self, span: 'LLMSpan', stream: Any, token: Any):
        from ..core import current_span_id_context
        # Anthropic streams are context managers
        class TracedStreamManager:
            def __enter__(self):
                self.original_stream = stream.__enter__()
                return self.traced_generator(self.original_stream)

            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    final_message = None
                    if hasattr(self.original_stream, 'get_final_message'):
                        final_message = self.original_stream.get_final_message()

                    self.finalize_stream_span(span, final_message, error=exc_val)
                finally:
                    current_span_id_context.reset(token)
                return stream.__exit__(exc_type, exc_val, exc_tb)

            def traced_generator(self, original_generator):
                for chunk in original_generator:
                    self.process_stream_chunk(span, chunk)
                    yield chunk

        return TracedStreamManager()

    def wrap_async_stream_method(self, original_method, provider: str):
        from functools import wraps
        from ..core import current_span_id_context

        @wraps(original_method)
        async def wrapped(*args, **kwargs):
            model = kwargs.get('model', 'unknown')
            span = self.create_span(
                model=model, provider=provider, operation="llm_stream_async", node_type="llm_call", node_name=model)

            self.handle_call_start(span, *args, **kwargs)

            token = current_span_id_context.set(span.span_id)
            try:
                stream = await original_method(*args, **kwargs)
                return self.handle_async_stream_response(span, stream, token)
            except Exception as e:
                current_span_id_context.reset(token)
                self.handle_call_end(span, None, success=False, error=e)
                raise

        return wrapped

    def handle_async_stream_response(self, span: 'LLMSpan', stream: Any, token: Any):
        from ..core import current_span_id_context
        class TracedAsyncStreamManager:
            async def __aenter__(self):
                self.original_stream = await stream.__aenter__()
                return self.traced_async_generator(self.original_stream)

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                try:
                    final_message = None
                    if hasattr(self.original_stream, 'get_final_message'):
                        final_message = await self.original_stream.get_final_message()

                    self.finalize_stream_span(span, final_message, error=exc_val)
                finally:
                    current_span_id_context.reset(token)
                return await stream.__aexit__(exc_type, exc_val, exc_tb)

            async def traced_async_generator(self, original_generator):
                async for chunk in original_generator:
                    self.process_stream_chunk(span, chunk)
                    yield chunk

        return TracedAsyncStreamManager()

    def process_stream_chunk(self, span: 'LLMSpan', chunk: Any):

        if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
            if hasattr(chunk.delta, 'text'):
                span.completion += chunk.delta.text

    def finalize_stream_span(self, span: 'LLMSpan', final_message: Any, error: Optional[Exception] = None):
        if error:
            self.handle_call_end(span, None, success=False, error=error)
            return

        if final_message:
            response_data = self.extract_response_data(final_message)
            span.completion = response_data.get('completion', span.completion)
            span.prompt_tokens = response_data.get('prompt_tokens', 0)
            span.completion_tokens = response_data.get('completion_tokens', 0)
            span.total_tokens = response_data.get('total_tokens', 0)
            span.cost = estimate_cost(
                span.model, span.prompt_tokens, span.completion_tokens)

        self.handle_call_end(span, final_message, success=True)
