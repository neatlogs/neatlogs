"""Neatlogs integration for Harbor.

``harbor run ... --plugin neatlogs`` streams a job to Neatlogs;
``neatlogs harbor import jobs/`` sends jobs that already ran.
"""

from .version import __version__


def __getattr__(name: str):
    # Harbor resolves the plugin lazily; importing the package for the CLI should
    # not import Harbor's job machinery.
    if name == "NeatlogsPlugin":
        from .plugin import NeatlogsPlugin

        return NeatlogsPlugin
    raise AttributeError(name)


__all__ = ["NeatlogsPlugin", "__version__"]
