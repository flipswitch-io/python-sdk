"""Flipswitch OpenFeature provider with real-time SSE support."""

import json
import logging
import platform
import sys
import threading
from importlib.metadata import version, PackageNotFoundError
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
from openfeature.evaluation_context import EvaluationContext
from openfeature.exception import (
    ErrorCode,
    OpenFeatureError,
)
from openfeature.flag_evaluation import FlagResolutionDetails, Reason
from openfeature.event import ProviderEventDetails
from openfeature.provider import AbstractProvider, Metadata
from openfeature.contrib.provider.ofrep import OFREPProvider

from flipswitch.sse_client import SseClient, ConnectionStatus
from flipswitch.types import FlagChangeEvent, FlagEvaluation

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.flipswitch.io"


def _get_sdk_version() -> str:
    """Get SDK version from installed package metadata."""
    try:
        return version("flipswitch-sdk")
    except PackageNotFoundError:
        return "dev"


SDK_VERSION = _get_sdk_version()
DEFAULT_POLLING_INTERVAL = 30.0  # seconds
DEFAULT_MAX_SSE_RETRIES = 5


class FlipswitchProvider(AbstractProvider):
    """Flipswitch OpenFeature provider with real-time SSE support.

    This provider wraps the OFREP provider for flag evaluation and adds
    real-time updates via Server-Sent Events (SSE).

    Example:
        >>> from flipswitch import FlipswitchProvider
        >>> from openfeature import api
        >>>
        >>> # API key is required, all other options have sensible defaults
        >>> provider = FlipswitchProvider(api_key="your-api-key")
        >>>
        >>> api.set_provider(provider)
        >>> client = api.get_client()
        >>>
        >>> dark_mode = client.get_boolean_value("dark-mode", False)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        enable_realtime: bool = True,
        http_client: Optional[httpx.Client] = None,
        enable_polling_fallback: bool = True,
        polling_interval: float = DEFAULT_POLLING_INTERVAL,
        max_sse_retries: int = DEFAULT_MAX_SSE_RETRIES,
    ):
        """Create a new FlipswitchProvider.

        Args:
            api_key: The environment API key (required).
            base_url: The Flipswitch server base URL.
            enable_realtime: Enable SSE for real-time flag updates.
            http_client: Custom HTTP client (optional).
            enable_polling_fallback: Enable polling fallback when SSE fails.
            polling_interval: Polling interval in seconds (default: 30).
            max_sse_retries: Max SSE retries before falling back to polling.

        Raises:
            ValueError: If api_key is not provided.
        """
        if not api_key:
            raise ValueError("api_key is required")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._enable_realtime = enable_realtime
        self._http_client = http_client or httpx.Client()
        self._owns_http_client = http_client is None

        # Polling fallback configuration
        self._enable_polling_fallback = enable_polling_fallback
        self._polling_interval = polling_interval
        self._max_sse_retries = max_sse_retries
        self._sse_retry_count = 0
        self._polling_active = False
        self._polling_timer: Optional[threading.Timer] = None
        self._polling_lock = threading.Lock()

        self._flag_change_listeners: List[Callable[[FlagChangeEvent], None]] = []
        self._sse_client: Optional[SseClient] = None
        self._initialized = False

        # Create underlying OFREP provider for flag evaluation
        # Note: OFREPProvider automatically appends /ofrep/v1 to the base_url
        self._ofrep_provider = OFREPProvider(
            base_url=self._base_url,
            headers_factory=self._get_headers,
        )

    def _get_telemetry_sdk_header(self) -> str:
        """Get SDK telemetry header value."""
        return f"python/{SDK_VERSION}"

    def _get_telemetry_runtime_header(self) -> str:
        """Get runtime telemetry header value."""
        return f"python/{platform.python_version()}"

    def _get_telemetry_os_header(self) -> str:
        """Get OS telemetry header value."""
        os_name = platform.system().lower()
        arch = platform.machine().lower()

        # Normalize OS name
        if os_name == "darwin":
            os_name = "darwin"
        elif os_name == "windows":
            os_name = "windows"
        elif os_name == "linux":
            os_name = "linux"

        # Normalize architecture
        if arch in ("x86_64", "amd64"):
            arch = "amd64"
        elif arch in ("aarch64", "arm64"):
            arch = "arm64"

        return f"{os_name}/{arch}"

    def _get_telemetry_features_header(self) -> str:
        """Get features telemetry header value."""
        return f"sse={str(self._enable_realtime).lower()}"

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for HTTP requests."""
        return {
            "X-API-Key": self._api_key,
            "X-Flipswitch-SDK": self._get_telemetry_sdk_header(),
            "X-Flipswitch-Runtime": self._get_telemetry_runtime_header(),
            "X-Flipswitch-OS": self._get_telemetry_os_header(),
            "X-Flipswitch-Features": self._get_telemetry_features_header(),
        }

    def get_metadata(self) -> Metadata:
        """Get provider metadata."""
        return Metadata(name="flipswitch")

    def initialize(self, evaluation_context: EvaluationContext) -> None:
        """Initialize the provider.

        Validates the API key and starts SSE connection if real-time is enabled.

        Args:
            evaluation_context: The evaluation context.

        Raises:
            OpenFeatureError: If initialization fails.
        """
        # Validate API key first
        self._validate_api_key()

        # Initialize the underlying OFREP provider
        self._ofrep_provider.initialize(evaluation_context)

        # Start SSE connection for real-time updates
        if self._enable_realtime:
            self._start_sse_connection()

        self._initialized = True
        logger.info(f"Flipswitch provider initialized (realtime={self._enable_realtime})")

    def _validate_api_key(self) -> None:
        """Validate the API key by making a bulk evaluation request.

        Raises:
            OpenFeatureError: If the API key is invalid or server error.
        """
        url = f"{self._base_url}/ofrep/v1/evaluate/flags"
        headers = {
            "Content-Type": "application/json",
            **self._get_headers(),
        }
        body = {"context": {"targetingKey": "_init_"}}

        try:
            response = self._http_client.post(url, headers=headers, json=body)

            if response.status_code in (401, 403):
                raise OpenFeatureError(
                    ErrorCode.PROVIDER_NOT_READY,
                    "Invalid API key",
                )

            if not response.is_success and response.status_code != 404:
                raise OpenFeatureError(
                    ErrorCode.GENERAL,
                    f"Failed to connect to Flipswitch: {response.status_code}",
                )
        except httpx.HTTPError as e:
            raise OpenFeatureError(
                ErrorCode.GENERAL,
                f"Failed to connect to Flipswitch: {e}",
            )

    def shutdown(self) -> None:
        """Shutdown the provider."""
        # Stop polling if active
        self._stop_polling()

        if self._sse_client:
            self._sse_client.close()
            self._sse_client = None

        self._ofrep_provider.shutdown()

        if self._owns_http_client:
            self._http_client.close()

        self._initialized = False
        logger.info("Flipswitch provider shut down")

    def _start_polling_fallback(self) -> None:
        """Start polling fallback when SSE fails."""
        with self._polling_lock:
            if self._polling_active or not self._enable_polling_fallback:
                return

            logger.info(f"Starting polling fallback (interval: {self._polling_interval}s)")
            self._polling_active = True
            self._schedule_poll()

    def _schedule_poll(self) -> None:
        """Schedule the next poll."""
        if not self._polling_active:
            return

        self._polling_timer = threading.Timer(self._polling_interval, self._poll_flags)
        self._polling_timer.daemon = True
        self._polling_timer.start()

    def _poll_flags(self) -> None:
        """Poll for flag updates."""
        if not self._polling_active:
            return

        try:
            # Invalidate OFREP cache to force refresh on next evaluation
            if hasattr(self._ofrep_provider, "_cache"):
                self._ofrep_provider._cache.clear()
            logger.debug("Polling: refreshed flag cache")
        except Exception as e:
            logger.warning(f"Polling refresh failed: {e}")

        # Schedule next poll
        self._schedule_poll()

    def _stop_polling(self) -> None:
        """Stop polling fallback."""
        with self._polling_lock:
            self._polling_active = False
            if self._polling_timer:
                self._polling_timer.cancel()
                self._polling_timer = None

    def is_polling_active(self) -> bool:
        """Check if polling fallback is active."""
        return self._polling_active

    def _start_sse_connection(self) -> None:
        """Start the SSE connection for real-time updates."""
        telemetry_headers = self._get_telemetry_headers_dict()
        self._sse_client = SseClient(
            base_url=self._base_url,
            api_key=self._api_key,
            on_flag_change=self._handle_flag_change,
            on_status_change=self._handle_status_change,
            telemetry_headers=telemetry_headers,
        )
        self._sse_client.connect()

    def _get_telemetry_headers_dict(self) -> dict[str, str]:
        """Get telemetry headers as a dictionary."""
        return {
            "X-Flipswitch-SDK": self._get_telemetry_sdk_header(),
            "X-Flipswitch-Runtime": self._get_telemetry_runtime_header(),
            "X-Flipswitch-OS": self._get_telemetry_os_header(),
            "X-Flipswitch-Features": self._get_telemetry_features_header(),
        }

    def _handle_flag_change(self, event: FlagChangeEvent) -> None:
        """Handle a flag change event from SSE.

        Triggers OFREP cache refresh and notifies user listeners.
        """
        # Trigger OFREP provider to refresh its cache
        # The OFREP provider's cache is invalidated by re-initializing
        try:
            # Call the provider's internal cache invalidation
            if hasattr(self._ofrep_provider, "_cache"):
                self._ofrep_provider._cache.clear()
        except Exception as e:
            # Log but don't fail - stale data is still usable
            logger.warning(f"Failed to refresh flags after SSE event: {e}")

        # Emit OpenFeature configuration changed event
        if event.flag_key is not None:
            self.emit_provider_configuration_changed(
                ProviderEventDetails(flags_changed=[event.flag_key])
            )
        else:
            self.emit_provider_configuration_changed(ProviderEventDetails())

        # Notify user-registered listeners
        for listener in self._flag_change_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error in flag change listener: {e}")

    def _handle_status_change(self, status: ConnectionStatus) -> None:
        """Handle SSE connection status change."""
        if status == ConnectionStatus.ERROR:
            self._sse_retry_count += 1
            logger.warning(f"SSE connection error (retry {self._sse_retry_count}), provider is stale")

            # Check if we should fall back to polling
            if self._sse_retry_count >= self._max_sse_retries and self._enable_polling_fallback:
                logger.warning(
                    f"SSE failed after {self._sse_retry_count} retries - falling back to polling"
                )
                self._start_polling_fallback()

        elif status == ConnectionStatus.CONNECTED:
            # SSE connected - reset retry count and stop polling
            self._sse_retry_count = 0

            if self._polling_active:
                logger.info("SSE reconnected - stopping polling fallback")
                self._stop_polling()

            logger.info("SSE connection restored")

    def add_flag_change_listener(
        self, listener: Callable[[FlagChangeEvent], None]
    ) -> None:
        """Add a listener for flag change events."""
        self._flag_change_listeners.append(listener)

    def remove_flag_change_listener(
        self, listener: Callable[[FlagChangeEvent], None]
    ) -> None:
        """Remove a flag change listener."""
        if listener in self._flag_change_listeners:
            self._flag_change_listeners.remove(listener)

    def get_sse_status(self) -> ConnectionStatus:
        """Get SSE connection status."""
        if self._sse_client:
            return self._sse_client.get_status()
        return ConnectionStatus.DISCONNECTED

    def reconnect_sse(self) -> None:
        """Force reconnect SSE connection."""
        if self._enable_realtime and self._sse_client:
            self._sse_client.close()
            self._start_sse_connection()

    # ===============================
    # Flag Resolution Methods - Delegated to OFREP Provider
    # ===============================

    def resolve_boolean_details(
        self,
        flag_key: str,
        default_value: bool,
        evaluation_context: Optional[EvaluationContext] = None,
    ) -> FlagResolutionDetails[bool]:
        """Resolve a boolean flag."""
        return self._ofrep_provider.resolve_boolean_details(
            flag_key, default_value, evaluation_context
        )

    def resolve_string_details(
        self,
        flag_key: str,
        default_value: str,
        evaluation_context: Optional[EvaluationContext] = None,
    ) -> FlagResolutionDetails[str]:
        """Resolve a string flag."""
        return self._ofrep_provider.resolve_string_details(
            flag_key, default_value, evaluation_context
        )

    def resolve_integer_details(
        self,
        flag_key: str,
        default_value: int,
        evaluation_context: Optional[EvaluationContext] = None,
    ) -> FlagResolutionDetails[int]:
        """Resolve an integer flag."""
        return self._ofrep_provider.resolve_integer_details(
            flag_key, default_value, evaluation_context
        )

    def resolve_float_details(
        self,
        flag_key: str,
        default_value: float,
        evaluation_context: Optional[EvaluationContext] = None,
    ) -> FlagResolutionDetails[float]:
        """Resolve a float flag."""
        return self._ofrep_provider.resolve_float_details(
            flag_key, default_value, evaluation_context
        )

    def resolve_object_details(
        self,
        flag_key: str,
        default_value: Union[Dict, List],
        evaluation_context: Optional[EvaluationContext] = None,
    ) -> FlagResolutionDetails[Union[Dict, List]]:
        """Resolve an object flag."""
        return self._ofrep_provider.resolve_object_details(
            flag_key, default_value, evaluation_context
        )

    # ===============================
    # Bulk Flag Evaluation (Direct HTTP - OFREP providers don't expose bulk API)
    # ===============================

    def _transform_context(
        self, context: Optional[EvaluationContext]
    ) -> Dict[str, Any]:
        """Transform OpenFeature context to OFREP context format."""
        if not context:
            return {}

        result: Dict[str, Any] = {}

        if context.targeting_key:
            result["targetingKey"] = context.targeting_key

        # Copy all context attributes
        if context.attributes:
            for key, value in context.attributes.items():
                if key != "targetingKey":
                    result[key] = value

        return result

    def _infer_type(self, value: Any) -> str:
        """Infer the type of a value."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "unknown"

    def _get_flag_type(self, flag: Dict[str, Any]) -> str:
        """Get flag type from metadata or infer from value."""
        # Prefer metadata.flagType if available
        metadata = flag.get("metadata", {})
        if metadata and "flagType" in metadata:
            meta_type = metadata["flagType"]
            if meta_type == "boolean":
                return "boolean"
            if meta_type == "string":
                return "string"
            if meta_type == "integer":
                return "integer"
            if meta_type == "decimal":
                return "number"

        # Fall back to inferring from value
        return self._infer_type(flag.get("value"))

    def evaluate_all_flags(
        self, context: Optional[EvaluationContext] = None
    ) -> List[FlagEvaluation]:
        """Evaluate all flags for the given context.

        Returns a list of all flag evaluations with their keys, values, types, and reasons.

        Note: This method makes direct HTTP calls since OFREP providers don't expose
        the bulk evaluation API.

        Args:
            context: The evaluation context.

        Returns:
            List of flag evaluations.
        """
        results: List[FlagEvaluation] = []

        try:
            url = f"{self._base_url}/ofrep/v1/evaluate/flags"
            headers = {
                "Content-Type": "application/json",
                **self._get_headers(),
            }
            body = {"context": self._transform_context(context)}

            response = self._http_client.post(url, headers=headers, json=body)

            if not response.is_success:
                logger.error(f"Failed to evaluate all flags: {response.status_code}")
                return results

            data = response.json()
            flags = data.get("flags", [])

            for flag in flags:
                key = flag.get("key")
                if key:
                    results.append(
                        FlagEvaluation(
                            key=key,
                            value=flag.get("value"),
                            value_type=self._get_flag_type(flag),
                            reason=flag.get("reason"),
                            variant=flag.get("variant"),
                        )
                    )

        except Exception as e:
            logger.error(f"Error evaluating all flags: {e}")

        return results

    def evaluate_flag(
        self, flag_key: str, context: Optional[EvaluationContext] = None
    ) -> Optional[FlagEvaluation]:
        """Evaluate a single flag and return its evaluation result.

        Note: This method makes direct HTTP calls for demo purposes.
        For standard flag evaluation, use the OpenFeature client methods.

        Args:
            flag_key: The flag key to evaluate.
            context: The evaluation context.

        Returns:
            The flag evaluation, or None if the flag doesn't exist.
        """
        try:
            url = f"{self._base_url}/ofrep/v1/evaluate/flags/{flag_key}"
            headers = {
                "Content-Type": "application/json",
                **self._get_headers(),
            }
            body = {"context": self._transform_context(context)}

            response = self._http_client.post(url, headers=headers, json=body)

            if not response.is_success:
                return None

            data = response.json()

            return FlagEvaluation(
                key=data.get("key", flag_key),
                value=data.get("value"),
                value_type=self._get_flag_type(data),
                reason=data.get("reason"),
                variant=data.get("variant"),
            )

        except Exception as e:
            logger.error(f"Error evaluating flag '{flag_key}': {e}")
            return None
