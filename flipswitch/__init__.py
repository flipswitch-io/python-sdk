"""
Flipswitch SDK with real-time SSE support for OpenFeature.

This SDK wraps OFREP-compatible flag evaluation with automatic
cache invalidation via Server-Sent Events (SSE). When flags change
in your Flipswitch dashboard, connected clients receive updates
in real-time.

Example:
    >>> from flipswitch import FlipswitchProvider
    >>> from openfeature import api
    >>>
    >>> # Only API key is required
    >>> provider = FlipswitchProvider(api_key="your-api-key")
    >>>
    >>> api.set_provider(provider)
    >>> client = api.get_client()
    >>>
    >>> # Flags automatically update when changed in dashboard
    >>> dark_mode = client.get_boolean_value("dark-mode", False)
"""

from importlib.metadata import version, PackageNotFoundError

from flipswitch.provider import FlipswitchProvider
from flipswitch.sse_client import SseClient, ConnectionStatus
from flipswitch.types import (
    FlipswitchOptions,
    FlagChangeEvent,
    FlagUpdatedEvent,
    ConfigUpdatedEvent,
    FlagEvaluation,
)

__all__ = [
    "FlipswitchProvider",
    "SseClient",
    "ConnectionStatus",
    "FlipswitchOptions",
    "FlagChangeEvent",
    "FlagUpdatedEvent",
    "ConfigUpdatedEvent",
    "FlagEvaluation",
]


def _get_version() -> str:
    """Get version from installed package metadata."""
    try:
        return version("flipswitch-sdk")
    except PackageNotFoundError:
        return "dev"


__version__ = _get_version()
