"""
Universal Prompt Template - Eliminates Variable Duplication

This module implements a PromptTemplate class that solves the duplication problem
where users had to specify variables twice:
1. Once in with trace(prompt_variables={...})
2. Again when calling the LLM

With PromptTemplate:
- Variables specified ONCE in template.compile(**variables)
- Works universally across ALL LLM frameworks (19+ tested)
- Auto-captured via ContextVars for tracing
- Supports streaming, async, structured outputs

Example:
    ```python
    from neatlogs import trace, PromptTemplate

    template = PromptTemplate("Answer {{question}} using {{context}}")

    with trace("query", prompt_template=template):
        # Variables specified ONCE - no duplication!
        prompt_text = template.compile(
            question="What is AI?",
            context="AI is..."
        )

        # Use with ANY framework
        response = openai.create(messages=[{"role": "user", "content": prompt_text}])
        # response = anthropic.create(messages=[{"role": "user", "content": prompt_text}])
        # response = llm.invoke(prompt_text)
        # etc...
    ```
"""

import re
import json
from typing import Dict, Any, Union, List, Optional
from contextvars import ContextVar

# Context variables for automatic capture
_prompt_template: ContextVar[Optional[str]] = ContextVar("prompt.template", default=None)
_prompt_variables: ContextVar[Optional[Dict[str, Any]]] = ContextVar("prompt.variables", default=None)


class PromptContext:
    """Manages prompt metadata in context for automatic tracing"""

    @staticmethod
    def set(template: str, variables: Dict[str, Any]):
        """Store template and variables in context"""
        _prompt_template.set(template)
        _prompt_variables.set(variables)

    @staticmethod
    def get_template() -> Optional[str]:
        """Retrieve template from context"""
        return _prompt_template.get(None)

    @staticmethod
    def get_variables() -> Optional[Dict[str, Any]]:
        """Retrieve variables from context"""
        return _prompt_variables.get(None)

    @staticmethod
    def clear():
        """Clear context"""
        _prompt_template.set(None)
        _prompt_variables.set(None)


class PromptTemplate:
    """
    Universal prompt template that works with ALL LLM frameworks.

    Eliminates variable duplication by allowing users to specify variables
    ONCE in the compile() method. The compiled output (strings or message arrays)
    works universally across all frameworks.

    Attributes:
        template: String template with {{variable}} syntax, or list of message dicts
        variables: List of variable names extracted from template

    Example with string template:
        ```python
        template = PromptTemplate("Explain {{topic}} in {{style}} style")

        with trace("query", prompt_template=template):
            # Variables specified ONCE
            prompt_text = template.compile(topic="AI", style="simple")

            # Works with any framework
            response = openai.create(messages=[{"role": "user", "content": prompt_text}])
        ```

    Example with chat messages:
        ```python
        template = PromptTemplate([
            {"role": "system", "content": "You are a {{role}}"},
            {"role": "user", "content": "Help with {{task}}"}
        ])

        with trace("query", prompt_template=template):
            # Variables specified ONCE
            messages = template.compile(role="expert", task="coding")

            # Works with any framework
            response = openai.create(messages=messages)
        ```

    Supports:
        - Simple string templates
        - Chat message templates (OpenAI/Anthropic/etc format)
        - Streaming (compile once, stream result)
        - Async (compile once, await result)
        - Structured outputs (compile prompt, get structured result)
    """

    def __init__(self, template: Union[str, List[Dict[str, str]]]):
        """
        Initialize prompt template.

        Args:
            template: Either:
                - String with {{variable}} placeholders (e.g., "Hello {{name}}")
                - List of message dicts with {{variable}} in content:
                  [{"role": "user", "content": "Hello {{name}}"}]
        """
        self.template = template
        self._variables = self._extract_variables()

    @property
    def variables(self) -> List[str]:
        """List of variable names in this template"""
        return self._variables

    def _extract_variables(self) -> List[str]:
        """
        Extract {{variable}} names from template.

        Returns:
            List of unique variable names
        """
        if isinstance(self.template, str):
            return list(set(re.findall(r'\{\{(\w+)\}\}', self.template)))

        # Extract from message list
        vars_found = []
        for msg in self.template:
            if isinstance(msg, dict) and "content" in msg:
                vars_found.extend(re.findall(r'\{\{(\w+)\}\}', msg["content"]))
        return list(set(vars_found))

    def compile(self, **variables) -> Union[str, List[Dict[str, str]]]:
        """
        Compile template with variables - NO DUPLICATION!

        This is where you specify variables ONCE. The method:
        1. Renders the template with your variables
        2. Auto-captures variables via ContextVars for tracing
        3. Returns standard strings/messages that work everywhere

        Args:
            **variables: Variable names and values matching {{placeholders}}

        Returns:
            - If template is a string: returns rendered string
            - If template is message list: returns list of message dicts

        Raises:
            ValueError: If required variables are missing

        Example:
            ```python
            template = PromptTemplate("Hello {{name}}, you are {{age}} years old")

            # Compile with variables (specified ONCE!)
            prompt_text = template.compile(name="Alice", age=30)
            # Returns: "Hello Alice, you are 30 years old"

            # Use the compiled output with any framework
            response = llm.invoke(prompt_text)
            ```
        """
        # Validate all required variables are provided
        missing = set(self._variables) - set(variables.keys())
        if missing:
            raise ValueError(
                f"Missing required variables: {missing}. "
                f"Template requires: {self._variables}"
            )

        # Auto-capture for tracing via ContextVars
        # This allows the trace context manager to read variables without explicit passing
        PromptContext.set(str(self.template), variables)

        # Render and return
        if isinstance(self.template, str):
            return self._render_string(self.template, variables)

        # Render message list
        return [
            {
                "role": msg["role"],
                "content": self._render_string(msg["content"], variables)
            }
            for msg in self.template
        ]

    def _render_string(self, text: str, variables: Dict[str, Any]) -> str:
        """
        Replace {{variable}} placeholders with values.

        Args:
            text: Template string with {{placeholders}}
            variables: Variable values

        Returns:
            Rendered string with all placeholders replaced
        """
        result = text
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def __str__(self) -> str:
        """String representation showing template structure"""
        if isinstance(self.template, str):
            return f"PromptTemplate('{self.template[:50]}...')" if len(str(self.template)) > 50 else f"PromptTemplate('{self.template}')"
        return f"PromptTemplate({len(self.template)} messages, variables={self.variables})"

    def __repr__(self) -> str:
        """Detailed representation"""
        return f"PromptTemplate(template={self.template!r}, variables={self.variables})"
