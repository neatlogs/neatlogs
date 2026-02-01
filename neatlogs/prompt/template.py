import re
import json
from typing import Dict, Any, Union, List, Optional
from contextvars import ContextVar

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
        missing = set(self._variables) - set(variables.keys())
        if missing:
            raise ValueError(
                f"Missing required variables: {missing}. "
                f"Template requires: {self._variables}"
            )

        PromptContext.set(str(self.template), variables)

        if isinstance(self.template, str):
            return self._render_string(self.template, variables)

        return [
            {
                "role": msg["role"],
                "content": self._render_string(msg["content"], variables)
            }
            for msg in self.template
        ]

    def _render_string(self, text: str, variables: Dict[str, Any]) -> str:
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
