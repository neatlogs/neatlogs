# Neatlogs Tests

This directory contains tests for the Neatlogs library.

## Running Tests

### Install test dependencies (Poetry)

```bash
poetry install --with dev
```

### Run unit tests (default)

```bash
poetry run pytest
```

### Run unit tests explicitly

```bash
poetry run pytest -q tests/unit
```

### Run a specific unit test file

```bash
poetry run pytest tests/unit/test_unified_attribute_processor_pipeline.py
```

## Test Structure

- `conftest.py` - Shared pytest fixtures and configuration
- `tests/unit/` - Unit tests (run by default)

## Writing Tests

Tests use mocking to avoid making real API calls:
- OpenAI client is mocked using `unittest.mock`
- Spans are captured using OpenTelemetry's `InMemorySpanExporter`
- No real API keys or network calls are required
