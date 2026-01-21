"""
Google GenAI Event Handler
==========================

Handles Google GenAI specific API patterns and response formats.
Enhanced with comprehensive tracking including streaming support.
"""

import logging
from typing import Dict, List, Any, Optional
from .base import BaseEventHandler


class GoogleGenAIHandler(BaseEventHandler):
    """Event handler for Google GenAI, with streaming and comprehensive tracking"""

    def extract_request_params(self, *args, **kwargs) -> Dict[str, Any]:
        params = super().extract_request_params(*args, **kwargs)

        # Google GenAI specific parameters - can be in different places
        config = kwargs.get("generation_config", kwargs.get("config", {}))
        if isinstance(config, dict):
            params.update(
                {
                    "max_output_tokens": config.get(
                        "max_output_tokens", config.get("maxOutputTokens")
                    ),
                    "top_k": config.get("top_k", config.get("topK")),
                    "candidate_count": config.get("candidate_count"),
                    "stop_sequences": config.get("stop_sequences"),
                }
            )
        elif hasattr(config, "model_dump"):
            config_dict = config.model_dump()
            params.update(
                {
                    "max_output_tokens": config_dict.get("max_output_tokens"),
                    "top_k": config_dict.get("top_k"),
                    "candidate_count": config_dict.get("candidate_count"),
                    "stop_sequences": config_dict.get("stop_sequences"),
                }
            )

        params.update(
            {
                "safety_settings": kwargs.get("safety_settings"),
                "tools": kwargs.get("tools"),
            }
        )
        return params

    def extract_messages(self, *args, **kwargs) -> List[Dict[str, str]]:
        """Extract messages from Google GenAI function arguments"""
        messages = []

        # Handle system instruction first - check both direct kwargs and config object
        system_instruction = kwargs.get("system_instruction")
        config = kwargs.get("generation_config", kwargs.get("config", {}))

        if not system_instruction and hasattr(config, "system_instruction"):
            system_instruction = config.system_instruction

        if system_instruction:
            system_text = str(system_instruction)
            if hasattr(system_instruction, "parts"):
                system_text = " ".join(
                    p.text for p in system_instruction.parts if hasattr(p, "text")
                )
            messages.append({"role": "system", "content": system_text})

        # Handle contents
        contents = kwargs.get("contents", [])
        if isinstance(contents, list):
            for content in contents:
                if isinstance(content, dict):
                    role = content.get("role", "user")
                    parts = content.get("parts", [])
                    text_content = " ".join(
                        p.get("text", "")
                        for p in parts
                        if isinstance(p, dict) and "text" in p
                    )
                    if text_content.strip():
                        messages.append({"role": role, "content": text_content.strip()})
                elif isinstance(content, str):
                    messages.append({"role": "user", "content": content})
        elif isinstance(contents, str):
            messages.append({"role": "user", "content": contents})

        return messages

    def extract_response_data(self, response: Any) -> Dict[str, Any]:
        """Extract data from Google GenAI response object"""
        data = {
            "completion": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": None,
            "tool_calls": [],
            "safety_ratings": [],
        }

        try:
            # Handle direct text response
            if hasattr(response, "text"):
                data["completion"] = response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]

                if hasattr(candidate, "content") and hasattr(
                    candidate.content, "parts"
                ):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)
                        elif hasattr(part, "function_call"):
                            fc = part.function_call
                            data["tool_calls"].append(
                                {"name": fc.name, "arguments": getattr(fc, "args", {})}
                            )
                    data["completion"] = "\n".join(text_parts)

                if hasattr(candidate, "finish_reason"):
                    data["finish_reason"] = str(candidate.finish_reason)

                if hasattr(candidate, "safety_ratings"):
                    data["safety_ratings"] = [
                        str(rating) for rating in candidate.safety_ratings
                    ]

            # Extract usage information
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                data["prompt_tokens"] = getattr(usage, "prompt_token_count", 0)
                data["completion_tokens"] = getattr(usage, "candidates_token_count", 0)
                data["total_tokens"] = getattr(usage, "total_token_count", 0)

            # Estimate tokens if not available
            if data["total_tokens"] == 0 and data["completion"]:
                data["completion_tokens"] = len(data["completion"].split())
                data["total_tokens"] = data["completion_tokens"]

        except Exception as e:
            logging.warning(f"Error extracting Google GenAI response data: {e}")

        return data

    def handle_call_start(self, span: "LLMSpan", *args, **kwargs):
        super().handle_call_start(span, *args, **kwargs)

    # --- Streaming Support ---

    def wrap_stream_method(self, original_method, provider: str):
        from functools import wraps
        from ..core import current_span_id_context

        @wraps(original_method)
        def wrapped(*args, **kwargs):
            model = kwargs.get("model", "unknown")
            span = self.create_span(
                model=model,
                provider=provider,
                operation="llm_stream",
                node_type="llm_call",
                node_name=model,
            )

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

    def handle_stream_response(self, span: "LLMSpan", stream: Any, token: Any):
        from ..core import current_span_id_context

        full_completion = ""
        final_response = None

        try:
            for chunk in stream:
                self.process_stream_chunk(span, chunk)
                if hasattr(chunk, "text"):
                    full_completion += chunk.text
                final_response = chunk
                yield chunk
        finally:
            current_span_id_context.reset(token)
            span.completion = full_completion
            self.finalize_stream_span(span, final_response)

    def process_stream_chunk(self, span: "LLMSpan", chunk: Any):
        pass

    def finalize_stream_span(
        self, span: "LLMSpan", final_response: Any, error: Optional[Exception] = None
    ):
        if error:
            self.handle_call_end(span, None, success=False, error=error)
            return

        if final_response:
            response_data = self.extract_response_data(final_response)
            span.prompt_tokens = response_data.get("prompt_tokens", 0)
            span.completion_tokens = response_data.get(
                "completion_tokens", len(span.completion.split())
            )
            span.total_tokens = response_data.get("total_tokens", 0)
            span.cost = self.estimate_cost(
                span.model, span.prompt_tokens, span.completion_tokens
            )

        self.handle_call_end(span, final_response, success=True)
