"""SSE client for real-time flag change notifications."""

import json
import logging
import threading
import time
from typing import Callable, Optional

import httpx

from flipswitch.types import ConnectionStatus, FlagChangeEvent

logger = logging.getLogger(__name__)

MIN_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 30.0  # seconds


class SseClient:
    """SSE client for real-time flag change notifications.

    Handles automatic reconnection with exponential backoff.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        on_flag_change: Callable[[FlagChangeEvent], None],
        on_status_change: Optional[Callable[[ConnectionStatus], None]] = None,
        telemetry_headers: Optional[dict[str, str]] = None,
    ):
        """Create a new SSE client.

        Args:
            base_url: The Flipswitch server base URL.
            api_key: The environment API key.
            on_flag_change: Callback for flag change events.
            on_status_change: Callback for connection status changes.
            telemetry_headers: Optional telemetry headers to send with SSE requests.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.on_flag_change = on_flag_change
        self.on_status_change = on_status_change
        self.telemetry_headers = telemetry_headers or {}

        self._status = ConnectionStatus.DISCONNECTED
        self._retry_delay = MIN_RETRY_DELAY
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status

    def get_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status

    def connect(self) -> None:
        """Start the SSE connection in a background thread."""
        with self._lock:
            if self._closed:
                return
            if self._thread is not None and self._thread.is_alive():
                return

            self._thread = threading.Thread(target=self._connect_loop, daemon=True)
            self._thread.start()

    def _connect_loop(self) -> None:
        """Connection loop with automatic reconnection."""
        while not self._closed:
            try:
                self._connect()
            except Exception as e:
                if not self._closed:
                    logger.error(f"SSE connection error: {e}")
                    self._update_status(ConnectionStatus.ERROR)
                    self._schedule_reconnect()

    def _connect(self) -> None:
        """Establish SSE connection and process events."""
        self._update_status(ConnectionStatus.CONNECTING)

        url = f"{self.base_url}/api/v1/flags/events"
        headers = {
            "X-API-Key": self.api_key,
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }
        # Add telemetry headers
        headers.update(self.telemetry_headers)

        with httpx.Client(timeout=None) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code != 200:
                    raise Exception(f"SSE connection failed: {response.status_code}")

                logger.info("SSE connection established")
                self._update_status(ConnectionStatus.CONNECTED)
                self._retry_delay = MIN_RETRY_DELAY

                event_type = ""
                event_data = ""

                for line in response.iter_lines():
                    if self._closed:
                        break

                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        event_data = line[5:].strip()
                    elif line == "" and event_data:
                        self._handle_event(event_type, event_data)
                        event_type = ""
                        event_data = ""

        # Stream ended
        if not self._closed:
            logger.info("SSE connection closed")
            self._update_status(ConnectionStatus.DISCONNECTED)
            self._schedule_reconnect()

    def _handle_event(self, event_type: str, data: str) -> None:
        """Handle incoming SSE events."""
        if event_type == "heartbeat":
            logger.debug("Heartbeat received")
            return

        try:
            if event_type == "flag-updated":
                # Single flag was modified
                parsed = json.loads(data)
                event = FlagChangeEvent(
                    flag_key=parsed["flagKey"],
                    timestamp=parsed["timestamp"],
                )
                logger.debug(f"Flag updated event: {event}")
                self.on_flag_change(event)
            elif event_type == "config-updated":
                # Configuration changed, need to refresh all flags
                parsed = json.loads(data)
                reason = parsed["reason"]

                # Log warning for api-key-rotated
                if reason == "api-key-rotated":
                    logger.warning(
                        "API key has been rotated. You may need to update your API key configuration."
                    )

                event = FlagChangeEvent(
                    flag_key=None,  # None indicates all flags should be refreshed
                    timestamp=parsed["timestamp"],
                )
                logger.debug(f"Config updated event (reason={reason}): {event}")
                self.on_flag_change(event)
            elif event_type == "flag-change":
                # Legacy event format for backward compatibility
                parsed = json.loads(data)
                event = FlagChangeEvent(
                    flag_key=parsed.get("flagKey"),
                    timestamp=parsed["timestamp"],
                )
                logger.debug(f"Flag change event: {event}")
                self.on_flag_change(event)
        except Exception as e:
            logger.error(f"Failed to parse {event_type} event: {e}")

    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt with exponential backoff."""
        if self._closed:
            return

        logger.info(f"Scheduling SSE reconnect in {self._retry_delay}s")
        time.sleep(self._retry_delay)

        # Increase backoff for next attempt
        self._retry_delay = min(self._retry_delay * 2, MAX_RETRY_DELAY)

    def _update_status(self, new_status: ConnectionStatus) -> None:
        """Update and broadcast connection status."""
        self._status = new_status
        if self.on_status_change:
            try:
                self.on_status_change(new_status)
            except Exception as e:
                logger.error(f"Error in status change callback: {e}")

    def close(self) -> None:
        """Close the SSE connection and stop reconnection attempts."""
        self._closed = True
        self._update_status(ConnectionStatus.DISCONNECTED)
