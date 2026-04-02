import json
import re
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Union

_prompt_template: ContextVar[Optional[str]] = ContextVar("prompt.template", default=None)
_prompt_variables: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "prompt.variables", default=None
)

_user_prompt_template: ContextVar[Optional[str]] = ContextVar("user_prompt.template", default=None)
_user_prompt_variables: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "user_prompt.variables", default=None
)


class PromptContext:
    """Manages system prompt metadata in context for automatic tracing"""

    @staticmethod
    def set(template: str, variables: Dict[str, Any]):
        """Store system prompt template and variables in context"""
        _prompt_template.set(template)
        _prompt_variables.set(variables)

    @staticmethod
    def get_template() -> Optional[str]:
        """Retrieve system prompt template from context"""
        return _prompt_template.get(None)

    @staticmethod
    def get_variables() -> Optional[Dict[str, Any]]:
        """Retrieve system prompt variables from context"""
        return _prompt_variables.get(None)

    @staticmethod
    def clear():
        """Clear system prompt context"""
        _prompt_template.set(None)
        _prompt_variables.set(None)


class UserPromptContext:
    """Manages user/human prompt metadata in context for automatic tracing"""

    @staticmethod
    def set(template: str, variables: Dict[str, Any]):
        """Store user prompt template and variables in context"""
        _user_prompt_template.set(template)
        _user_prompt_variables.set(variables)

    @staticmethod
    def get_template() -> Optional[str]:
        """Retrieve user prompt template from context"""
        return _user_prompt_template.get(None)

    @staticmethod
    def get_variables() -> Optional[Dict[str, Any]]:
        """Retrieve user prompt variables from context"""
        return _user_prompt_variables.get(None)

    @staticmethod
    def clear():
        """Clear user prompt context"""
        _user_prompt_template.set(None)
        _user_prompt_variables.set(None)


class PromptTemplate:
    """Template for the system/AI instruction prompt with {{variable}} placeholders."""

    def __init__(self, template: Union[str, List[Dict[str, str]]]):
        """
        Initialize system prompt template.

        Args:
            template: Either:
                - String with {{variable}} placeholders (e.g., "You are a {{role}} assistant")
                - List of message dicts with {{variable}} in content:
                  [{"role": "system", "content": "You are a {{role}} assistant"}]
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
            return list(set(re.findall(r"\{\{(\w+)\}\}", self.template)))

        # Extract from message list
        vars_found = []
        for msg in self.template:
            if isinstance(msg, dict) and "content" in msg:
                vars_found.extend(re.findall(r"\{\{(\w+)\}\}", msg["content"]))
        return list(set(vars_found))

    def compile(self, **variables) -> Union[str, List[Dict[str, str]]]:
        """Compile the system prompt template with the given variables."""
        missing = set(self._variables) - set(variables.keys())
        if missing:
            raise ValueError(
                f"Missing required variables: {missing}. " f"Template requires: {self._variables}"
            )

        PromptContext.set(str(self.template), variables)

        if isinstance(self.template, str):
            return self._render_string(self.template, variables)

        return [
            {"role": msg["role"], "content": self._render_string(msg["content"], variables)}
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
            return (
                f"PromptTemplate('{self.template[:50]}...')"
                if len(str(self.template)) > 50
                else f"PromptTemplate('{self.template}')"
            )
        return f"PromptTemplate({len(self.template)} messages, variables={self.variables})"

    def __repr__(self) -> str:
        """Detailed representation"""
        return f"PromptTemplate(template={self.template!r}, variables={self.variables})"


class UserPromptTemplate:
    """Template for the user/human turn prompt with {{variable}} placeholders."""

    def __init__(self, template: Union[str, List[Dict[str, str]]]):
        """
        Initialize user prompt template.

        Args:
            template: Either:
                - String with {{variable}} placeholders (e.g., "Tell me about {{topic}}")
                - List of message dicts with {{variable}} in content:
                  [{"role": "user", "content": "Tell me about {{topic}}"}]
        """
        self.template = template
        self._variables = self._extract_variables()

    @property
    def variables(self) -> List[str]:
        """List of variable names in this template"""
        return self._variables

    def _extract_variables(self) -> List[str]:
        """Extract {{variable}} names from template."""
        if isinstance(self.template, str):
            return list(set(re.findall(r"\{\{(\w+)\}\}", self.template)))

        vars_found = []
        for msg in self.template:
            if isinstance(msg, dict) and "content" in msg:
                vars_found.extend(re.findall(r"\{\{(\w+)\}\}", msg["content"]))
        return list(set(vars_found))

    def compile(self, **variables) -> Union[str, List[Dict[str, str]]]:
        """Compile the user prompt template with the given variables."""
        missing = set(self._variables) - set(variables.keys())
        if missing:
            raise ValueError(
                f"Missing required variables: {missing}. " f"Template requires: {self._variables}"
            )

        UserPromptContext.set(str(self.template), variables)

        if isinstance(self.template, str):
            return self._render_string(self.template, variables)

        return [
            {"role": msg["role"], "content": self._render_string(msg["content"], variables)}
            for msg in self.template
        ]

    def _render_string(self, text: str, variables: Dict[str, Any]) -> str:
        result = text
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def __str__(self) -> str:
        if isinstance(self.template, str):
            return (
                f"UserPromptTemplate('{self.template[:50]}...')"
                if len(str(self.template)) > 50
                else f"UserPromptTemplate('{self.template}')"
            )
        return f"UserPromptTemplate({len(self.template)} messages, variables={self.variables})"

    def __repr__(self) -> str:
        return f"UserPromptTemplate(template={self.template!r}, variables={self.variables})"
