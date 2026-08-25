class NeatlogsError(Exception):
    """Base class for typed Neatlogs SDK failures."""


class NeatlogsConfigurationError(NeatlogsError, ValueError):
    """Raised when SDK configuration cannot be applied safely."""
