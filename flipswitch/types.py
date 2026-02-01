"""Type definitions for the Flipswitch SDK."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


@dataclass
class FlipswitchOptions:
    """Configuration options for the Flipswitch provider.

    Attributes:
        api_key: The API key for your environment (required).
        base_url: The base URL of your Flipswitch server.
        enable_realtime: Enable real-time flag updates via SSE.
    """

    api_key: str
    base_url: str = "https://api.flipswitch.io"
    enable_realtime: bool = True


@dataclass
class FlagUpdatedEvent:
    """Event emitted when a single flag is updated.

    Attributes:
        flag_key: The key of the flag that changed.
        timestamp: ISO timestamp of when the change occurred.
    """

    flag_key: str
    timestamp: str

    def get_timestamp_as_datetime(self) -> Optional[datetime]:
        """Get the timestamp as a datetime object."""
        if self.timestamp:
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        return None


@dataclass
class ConfigUpdatedEvent:
    """Event emitted when configuration changes that may affect multiple flags.

    Attributes:
        timestamp: ISO timestamp of when the change occurred.
    """

    timestamp: str

    def get_timestamp_as_datetime(self) -> Optional[datetime]:
        """Get the timestamp as a datetime object."""
        if self.timestamp:
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        return None


@dataclass
class ApiKeyRotatedEvent:
    """Event emitted when an API key has been rotated.

    Attributes:
        valid_until: ISO timestamp when the current key expires.
        timestamp: ISO timestamp of when the rotation occurred.
    """

    valid_until: str
    timestamp: str

    def get_valid_until_as_datetime(self) -> Optional[datetime]:
        """Get the valid_until as a datetime object."""
        if self.valid_until:
            return datetime.fromisoformat(self.valid_until.replace("Z", "+00:00"))
        return None

    def get_timestamp_as_datetime(self) -> Optional[datetime]:
        """Get the timestamp as a datetime object."""
        if self.timestamp:
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        return None


@dataclass
class FlagChangeEvent:
    """Event emitted when a flag changes (legacy format, used internally).

    Attributes:
        flag_key: The key of the flag that changed, or None for bulk invalidation.
        timestamp: ISO timestamp of when the change occurred.
    """

    flag_key: Optional[str]
    timestamp: str

    def get_timestamp_as_datetime(self) -> Optional[datetime]:
        """Get the timestamp as a datetime object."""
        if self.timestamp:
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        return None


@dataclass
class FlagEvaluation:
    """Represents the result of evaluating a single flag.

    Attributes:
        key: The flag key.
        value: The evaluated value.
        value_type: The type of the value (boolean, string, number, etc.).
        reason: The reason for this evaluation result.
        variant: The variant that matched, if applicable.
    """

    key: str
    value: Any
    value_type: str
    reason: Optional[str] = None
    variant: Optional[str] = None

    def as_boolean(self) -> bool:
        """Get the value as a boolean."""
        return bool(self.value) if self.value is not None else False

    def as_int(self) -> int:
        """Get the value as an integer."""
        return int(self.value) if self.value is not None else 0

    def as_float(self) -> float:
        """Get the value as a float."""
        return float(self.value) if self.value is not None else 0.0

    def as_string(self) -> Optional[str]:
        """Get the value as a string."""
        return str(self.value) if self.value is not None else None

    def get_value_as_string(self) -> str:
        """Get the value formatted for display."""
        if self.value is None:
            return "null"
        if isinstance(self.value, str):
            return f'"{self.value}"'
        if isinstance(self.value, bool):
            return str(self.value).lower()
        return str(self.value)

    def __str__(self) -> str:
        variant_str = f", variant={self.variant}" if self.variant else ""
        return (
            f"{self.key} ({self.value_type}): {self.get_value_as_string()} "
            f"[reason={self.reason}{variant_str}]"
        )


# Type aliases for callbacks
FlagChangeCallback = Callable[[FlagChangeEvent], None]
ConnectionStatusCallback = Callable[["ConnectionStatus"], None]


class ConnectionStatus(Enum):
    """SSE connection status."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
