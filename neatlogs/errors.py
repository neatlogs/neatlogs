"""Public, catchable Neatlogs SDK errors."""


class NeatlogsError(Exception):
    """Base class for Neatlogs-specific failures."""


class NeatlogsConfigurationError(NeatlogsError, ValueError):
    """Raised before startup when SDK configuration is invalid or contradictory."""
