"""Shared path formatting for application source locations."""

from pathlib import Path


def code_file_path(filename: str) -> str:
    """Omit the home-directory prefix from user source paths."""
    try:
        return Path(filename).relative_to(Path.home()).as_posix()
    except (RuntimeError, ValueError):
        return filename
