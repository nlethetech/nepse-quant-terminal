"""
Custom exception hierarchy for NEPSE Quant Pro.

Distinguishes data staleness vs vendor failures vs position limit breaches
vs DB errors vs configuration mistakes so callers can handle each precisely.
"""


class NepseQuantError(Exception):
    """Base exception for all NEPSE Quant Pro errors."""


class DataStalenessError(NepseQuantError):
    """Market data is too stale for safe signal generation."""


class VendorAPIError(NepseQuantError):
    """Upstream vendor (Merolagani) request failed after retries."""


class PositionLimitError(NepseQuantError):
    """Position sizing or portfolio limit breached."""


class DatabaseError(NepseQuantError):
    """SQLite connection, lock, or integrity error."""


class ConfigurationError(NepseQuantError):
    """Invalid or missing configuration value."""
