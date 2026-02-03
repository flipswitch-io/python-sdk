"""Tests for FlipswitchProvider.

Note: These tests focus on FlipswitchProvider's specific functionality:
- Initialization and API key validation
- SSE connection management
- Bulk flag evaluation methods (evaluate_all_flags, evaluate_flag)

The OpenFeature SDK evaluation methods are delegated to the underlying OFREP provider,
which has its own test suite.
"""

import json
import platform
import threading
import time
import pytest
import httpx
from pytest_httpserver import HTTPServer

from openfeature.evaluation_context import EvaluationContext

from flipswitch import FlipswitchProvider, ConnectionStatus
from flipswitch.types import FlagChangeEvent


@pytest.fixture
def mock_server(httpserver: HTTPServer):
    """Fixture providing a mock HTTP server."""
    return httpserver


def create_provider(server: HTTPServer, enable_realtime: bool = False) -> FlipswitchProvider:
    """Create a provider configured to use the mock server."""
    return FlipswitchProvider(
        api_key="test-api-key",
        base_url=server.url_for(""),
        enable_realtime=enable_realtime,
    )


def setup_bulk_response(server: HTTPServer, response_body: dict, status: int = 200):
    """Setup the bulk evaluation endpoint."""
    server.expect_request(
        "/ofrep/v1/evaluate/flags",
        method="POST",
    ).respond_with_json(response_body, status=status)


def setup_flag_response(server: HTTPServer, flag_key: str, response_body: dict, status: int = 200):
    """Setup a single flag evaluation endpoint."""
    server.expect_request(
        f"/ofrep/v1/evaluate/flags/{flag_key}",
        method="POST",
    ).respond_with_json(response_body, status=status)


# ========================================
# Initialization Tests
# ========================================


class TestInitialization:
    """Test provider initialization."""

    def test_initialization_should_succeed(self, mock_server: HTTPServer):
        """Provider initializes with valid API key."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        assert provider.get_metadata().name == "flipswitch"
        provider.shutdown()

    def test_initialization_should_fail_on_invalid_api_key(self, mock_server: HTTPServer):
        """Returns error on 401."""
        setup_bulk_response(mock_server, {}, status=401)

        provider = create_provider(mock_server)

        with pytest.raises(Exception) as exc_info:
            provider.initialize(EvaluationContext())

        assert "Invalid API key" in str(exc_info.value)
        provider.shutdown()

    def test_initialization_should_fail_on_forbidden(self, mock_server: HTTPServer):
        """Returns error on 403."""
        setup_bulk_response(mock_server, {}, status=403)

        provider = create_provider(mock_server)

        with pytest.raises(Exception) as exc_info:
            provider.initialize(EvaluationContext())

        assert "Invalid API key" in str(exc_info.value)
        provider.shutdown()

    def test_initialization_should_fail_on_server_error(self, mock_server: HTTPServer):
        """Returns error on 500."""
        setup_bulk_response(mock_server, {}, status=500)

        provider = create_provider(mock_server)

        with pytest.raises(Exception) as exc_info:
            provider.initialize(EvaluationContext())

        assert "Failed to connect" in str(exc_info.value)
        provider.shutdown()


# ========================================
# Metadata Tests
# ========================================


class TestMetadata:
    """Test provider metadata."""

    def test_metadata_should_return_flipswitch(self, mock_server: HTTPServer):
        """Provider metadata name is 'flipswitch'."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        assert provider.get_metadata().name == "flipswitch"
        provider.shutdown()


# ========================================
# Bulk Evaluation Tests
# ========================================


class TestBulkEvaluation:
    """Test bulk flag evaluation."""

    def test_evaluate_all_flags_should_return_all_flags(self, mock_server: HTTPServer):
        """Bulk evaluation returns all flags."""
        setup_bulk_response(mock_server, {
            "flags": [
                {"key": "flag-1", "value": True, "reason": "DEFAULT"},
                {"key": "flag-2", "value": "test", "reason": "TARGETING_MATCH"},
            ]
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        # Setup for the actual evaluation call
        mock_server.clear()
        setup_bulk_response(mock_server, {
            "flags": [
                {"key": "flag-1", "value": True, "reason": "DEFAULT"},
                {"key": "flag-2", "value": "test", "reason": "TARGETING_MATCH"},
            ]
        })

        context = EvaluationContext(targeting_key="user-1")
        flags = provider.evaluate_all_flags(context)

        assert len(flags) == 2
        assert flags[0].key == "flag-1"
        assert flags[0].as_boolean() is True
        assert flags[1].key == "flag-2"
        assert flags[1].as_string() == "test"
        provider.shutdown()

    def test_evaluate_all_flags_should_return_empty_list_on_error(self, mock_server: HTTPServer):
        """Returns empty list on error."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        # Setup error response for the actual evaluation call
        mock_server.clear()
        setup_bulk_response(mock_server, {}, status=500)

        context = EvaluationContext(targeting_key="user-1")
        flags = provider.evaluate_all_flags(context)

        assert len(flags) == 0
        provider.shutdown()


# ========================================
# Single Flag Evaluation Tests
# ========================================


class TestSingleFlagEvaluation:
    """Test single flag evaluation."""

    def test_evaluate_flag_should_return_single_flag(self, mock_server: HTTPServer):
        """Single flag evaluation works."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "my-flag", {
            "key": "my-flag",
            "value": "hello",
            "reason": "DEFAULT",
            "variant": "v1",
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.evaluate_flag("my-flag", EvaluationContext())

        assert result is not None
        assert result.key == "my-flag"
        assert result.as_string() == "hello"
        assert result.reason == "DEFAULT"
        assert result.variant == "v1"
        provider.shutdown()

    def test_evaluate_flag_should_return_none_for_nonexistent(self, mock_server: HTTPServer):
        """Returns None for 404."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "nonexistent", {
            "key": "nonexistent",
            "errorCode": "FLAG_NOT_FOUND",
        }, status=404)

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.evaluate_flag("nonexistent", EvaluationContext())

        assert result is None
        provider.shutdown()

    def test_evaluate_flag_should_handle_boolean_values(self, mock_server: HTTPServer):
        """Boolean type handling."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "bool-flag", {
            "key": "bool-flag",
            "value": True,
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.evaluate_flag("bool-flag", EvaluationContext())

        assert result is not None
        assert result.as_boolean() is True
        provider.shutdown()

    def test_evaluate_flag_should_handle_string_values(self, mock_server: HTTPServer):
        """String type handling."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "string-flag", {
            "key": "string-flag",
            "value": "test-value",
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.evaluate_flag("string-flag", EvaluationContext())

        assert result is not None
        assert result.as_string() == "test-value"
        provider.shutdown()

    def test_evaluate_flag_should_handle_numeric_values(self, mock_server: HTTPServer):
        """Numeric type handling."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "num-flag", {
            "key": "num-flag",
            "value": 42,
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.evaluate_flag("num-flag", EvaluationContext())

        assert result is not None
        assert result.as_int() == 42
        provider.shutdown()


# ========================================
# SSE Status Tests
# ========================================


class TestSseStatus:
    """Test SSE connection status."""

    def test_sse_status_should_be_disconnected_when_realtime_disabled(self, mock_server: HTTPServer):
        """SSE status is DISCONNECTED when realtime is disabled."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server, enable_realtime=False)
        provider.initialize(EvaluationContext())

        assert provider.get_sse_status() == ConnectionStatus.DISCONNECTED
        provider.shutdown()


# ========================================
# Flag Change Listener Tests
# ========================================


class TestFlagChangeListener:
    """Test flag change listener management."""

    def test_flag_change_listener_can_be_added_and_removed(self, mock_server: HTTPServer):
        """Listener management works without exceptions."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []

        def listener(event):
            events.append(event)

        provider.add_flag_change_listener(listener)
        provider.remove_flag_change_listener(listener)

        # Verify no exceptions thrown - listener management works
        assert len(events) == 0
        provider.shutdown()


# ========================================
# Builder Tests
# ========================================


class TestBuilder:
    """Test provider builder/constructor."""

    def test_builder_should_require_api_key(self):
        """API key validation."""
        with pytest.raises(ValueError):
            FlipswitchProvider(api_key=None)

        with pytest.raises(ValueError):
            FlipswitchProvider(api_key="")

    def test_builder_should_use_defaults(self, mock_server: HTTPServer):
        """Default values work."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = FlipswitchProvider(
            api_key="test-key",
            base_url=mock_server.url_for(""),
            enable_realtime=False,
        )

        assert provider.get_metadata().name == "flipswitch"
        provider.shutdown()

    def test_builder_should_allow_custom_base_url(self, mock_server: HTTPServer):
        """Custom base URL works."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        # If we get here without exception, the custom baseUrl was used
        assert provider.get_metadata().name == "flipswitch"
        provider.shutdown()


# ========================================
# URL Path Tests
# ========================================


class TestUrlPath:
    """Test that OFREP requests use correct paths."""

    def test_ofrep_requests_should_use_correct_path(self, mock_server: HTTPServer):
        """Verify requests don't have duplicated /ofrep/v1 path segments."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "test-flag", {
            "key": "test-flag",
            "value": True,
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        # Trigger a single flag evaluation
        provider.evaluate_flag("test-flag", EvaluationContext())

        # Check the request log to verify correct path
        requests = mock_server.log
        flag_request = [r for r in requests if "test-flag" in r[0].path]

        assert len(flag_request) > 0, "Expected flag evaluation request"
        assert flag_request[0][0].path == "/ofrep/v1/evaluate/flags/test-flag"
        provider.shutdown()


# ========================================
# Polling Fallback Tests
# ========================================


class TestPollingFallback:
    """Test polling fallback behavior."""

    def test_polling_activates_after_max_sse_retries(self, mock_server: HTTPServer):
        """Polling activates after maxSseRetries error status changes."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = FlipswitchProvider(
            api_key="test-api-key",
            base_url=mock_server.url_for(""),
            enable_realtime=False,
            enable_polling_fallback=True,
            max_sse_retries=3,
        )
        provider.initialize(EvaluationContext())

        # Simulate SSE error status changes
        for _ in range(3):
            provider._handle_status_change(ConnectionStatus.ERROR)

        assert provider.is_polling_active() is True
        provider.shutdown()

    def test_polling_deactivates_on_sse_reconnect(self, mock_server: HTTPServer):
        """Polling stops when SSE reconnects."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = FlipswitchProvider(
            api_key="test-api-key",
            base_url=mock_server.url_for(""),
            enable_realtime=False,
            enable_polling_fallback=True,
            max_sse_retries=2,
        )
        provider.initialize(EvaluationContext())

        # Trigger polling
        for _ in range(2):
            provider._handle_status_change(ConnectionStatus.ERROR)
        assert provider.is_polling_active() is True

        # Simulate SSE reconnect
        provider._handle_status_change(ConnectionStatus.CONNECTED)
        assert provider.is_polling_active() is False
        provider.shutdown()

    def test_polling_disabled_when_enable_polling_fallback_false(self, mock_server: HTTPServer):
        """Polling never activates when enablePollingFallback=False."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = FlipswitchProvider(
            api_key="test-api-key",
            base_url=mock_server.url_for(""),
            enable_realtime=False,
            enable_polling_fallback=False,
            max_sse_retries=2,
        )
        provider.initialize(EvaluationContext())

        # Simulate many SSE errors
        for _ in range(10):
            provider._handle_status_change(ConnectionStatus.ERROR)

        assert provider.is_polling_active() is False
        provider.shutdown()


# ========================================
# Flag Change Handling Tests
# ========================================


class TestFlagChangeHandling:
    """Test flag change listener notification."""

    def test_listener_notification_on_flag_change(self, mock_server: HTTPServer):
        """Registered listeners receive flag change events."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []
        provider.add_flag_change_listener(lambda e: events.append(e))

        event = FlagChangeEvent(flag_key="test-flag", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(events) == 1
        assert events[0].flag_key == "test-flag"
        provider.shutdown()

    def test_listener_error_isolation(self, mock_server: HTTPServer):
        """One listener throwing doesn't prevent other listeners from being called."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []

        def bad_listener(event):
            raise RuntimeError("listener error")

        def good_listener(event):
            events.append(event)

        provider.add_flag_change_listener(bad_listener)
        provider.add_flag_change_listener(good_listener)

        event = FlagChangeEvent(flag_key="test", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(events) == 1
        provider.shutdown()

    def test_multiple_listeners_all_called(self, mock_server: HTTPServer):
        """All registered listeners are called on flag change."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events1 = []
        events2 = []
        events3 = []

        provider.add_flag_change_listener(lambda e: events1.append(e))
        provider.add_flag_change_listener(lambda e: events2.append(e))
        provider.add_flag_change_listener(lambda e: events3.append(e))

        event = FlagChangeEvent(flag_key="test", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(events1) == 1
        assert len(events2) == 1
        assert len(events3) == 1
        provider.shutdown()


# ========================================
# Shutdown / Cleanup Tests
# ========================================


class TestShutdown:
    """Test provider shutdown behavior."""

    def test_shutdown_clears_state(self, mock_server: HTTPServer):
        """After shutdown, provider is no longer initialized."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        assert provider._initialized is True

        provider.shutdown()

        assert provider._initialized is False

    def test_shutdown_is_idempotent(self, mock_server: HTTPServer):
        """Calling shutdown twice doesn't throw."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        provider.shutdown()
        provider.shutdown()  # Should not raise


# ========================================
# Context Transformation Tests
# ========================================


class TestContextTransformation:
    """Test context transformation to OFREP format."""

    def test_targeting_key_only(self):
        """Context with just targetingKey transforms correctly."""
        provider = FlipswitchProvider(
            api_key="test-key",
            enable_realtime=False,
        )

        context = EvaluationContext(targeting_key="user-123")
        result = provider._transform_context(context)

        assert result["targetingKey"] == "user-123"
        provider.shutdown()

    def test_with_attributes(self):
        """Context with additional attributes are included."""
        provider = FlipswitchProvider(
            api_key="test-key",
            enable_realtime=False,
        )

        context = EvaluationContext(
            targeting_key="user-123",
            attributes={"email": "test@example.com", "plan": "premium"},
        )
        result = provider._transform_context(context)

        assert result["targetingKey"] == "user-123"
        assert result["email"] == "test@example.com"
        assert result["plan"] == "premium"
        provider.shutdown()

    def test_empty_context(self):
        """Empty context produces empty dict."""
        provider = FlipswitchProvider(
            api_key="test-key",
            enable_realtime=False,
        )

        result = provider._transform_context(None)
        assert result == {}

        result = provider._transform_context(EvaluationContext())
        assert isinstance(result, dict)
        provider.shutdown()


# ========================================
# Type Inference Tests
# ========================================


class TestTypeInference:
    """Test type inference logic."""

    def setup_method(self):
        self.provider = FlipswitchProvider(
            api_key="test-key",
            enable_realtime=False,
        )

    def teardown_method(self):
        self.provider.shutdown()

    def test_infer_boolean(self):
        assert self.provider._infer_type(True) == "boolean"
        assert self.provider._infer_type(False) == "boolean"

    def test_infer_string(self):
        assert self.provider._infer_type("hello") == "string"

    def test_infer_integer(self):
        assert self.provider._infer_type(42) == "integer"

    def test_infer_float(self):
        assert self.provider._infer_type(3.14) == "number"

    def test_infer_null(self):
        assert self.provider._infer_type(None) == "null"

    def test_infer_object(self):
        assert self.provider._infer_type({"key": "value"}) == "object"

    def test_infer_array(self):
        assert self.provider._infer_type([1, 2, 3]) == "array"

    def test_metadata_override(self):
        """getFlagType with metadata.flagType takes precedence."""
        flag = {"value": None, "metadata": {"flagType": "boolean"}}
        assert self.provider._get_flag_type(flag) == "boolean"

    def test_metadata_decimal_maps_to_number(self):
        """metadata.flagType 'decimal' maps to 'number'."""
        flag = {"value": 3.14, "metadata": {"flagType": "decimal"}}
        assert self.provider._get_flag_type(flag) == "number"


# ========================================
# Telemetry Headers Tests
# ========================================


class TestTelemetryHeaders:
    """Test telemetry header generation."""

    @staticmethod
    def _get_header(headers: dict, name: str):
        """Case-insensitive header lookup (werkzeug normalizes casing)."""
        lower = name.lower()
        for k, v in headers.items():
            if k.lower() == lower:
                return v
        return None

    def test_sdk_header(self, mock_server: HTTPServer):
        """Request includes X-Flipswitch-SDK with correct format."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        # Make a request
        mock_server.clear()
        setup_bulk_response(mock_server, {"flags": []})
        provider.evaluate_all_flags(EvaluationContext(targeting_key="user-1"))

        requests = mock_server.log
        assert len(requests) > 0
        headers = dict(requests[-1][0].headers)
        sdk_header = self._get_header(headers, "X-Flipswitch-SDK")
        assert sdk_header is not None, f"X-Flipswitch-SDK not found in {list(headers.keys())}"
        assert sdk_header.startswith("python/")
        provider.shutdown()

    def test_runtime_header(self, mock_server: HTTPServer):
        """Request includes X-Flipswitch-Runtime header."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        mock_server.clear()
        setup_bulk_response(mock_server, {"flags": []})
        provider.evaluate_all_flags(EvaluationContext(targeting_key="user-1"))

        requests = mock_server.log
        headers = dict(requests[-1][0].headers)
        runtime_header = self._get_header(headers, "X-Flipswitch-Runtime")
        assert runtime_header is not None
        assert runtime_header.startswith("python/")
        provider.shutdown()

    def test_os_header(self, mock_server: HTTPServer):
        """Request includes X-Flipswitch-OS header."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        mock_server.clear()
        setup_bulk_response(mock_server, {"flags": []})
        provider.evaluate_all_flags(EvaluationContext(targeting_key="user-1"))

        requests = mock_server.log
        headers = dict(requests[-1][0].headers)
        os_header = self._get_header(headers, "X-Flipswitch-OS")
        assert os_header is not None, f"X-Flipswitch-OS not found in {list(headers.keys())}"
        assert "/" in os_header
        provider.shutdown()

    def test_features_header(self, mock_server: HTTPServer):
        """Request includes X-Flipswitch-Features with sse value."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server, enable_realtime=False)
        provider.initialize(EvaluationContext())

        mock_server.clear()
        setup_bulk_response(mock_server, {"flags": []})
        provider.evaluate_all_flags(EvaluationContext(targeting_key="user-1"))

        requests = mock_server.log
        headers = dict(requests[-1][0].headers)
        features_header = self._get_header(headers, "X-Flipswitch-Features")
        assert features_header == "sse=false"
        provider.shutdown()


# ========================================
# Custom HTTP Client Tests
# ========================================


class TestCustomHttpClient:
    """Test custom httpx.Client injection."""

    def test_custom_client_is_used(self, mock_server: HTTPServer):
        """Verify custom HTTP client is used for requests."""
        setup_bulk_response(mock_server, {"flags": []})

        custom_client = httpx.Client(timeout=5.0)

        provider = FlipswitchProvider(
            api_key="test-api-key",
            base_url=mock_server.url_for(""),
            enable_realtime=False,
            http_client=custom_client,
        )
        provider.initialize(EvaluationContext())

        # Provider should not own the client (won't close it)
        assert provider._owns_http_client is False

        provider.shutdown()

        # Custom client should still be usable after provider shutdown
        assert not custom_client.is_closed
        custom_client.close()
