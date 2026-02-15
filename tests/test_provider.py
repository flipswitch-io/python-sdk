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
from openfeature.event import ProviderEvent, ProviderEventDetails

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
# Flag Change Event Details Tests
# ========================================


class TestFlagChangeEventDetails:
    """Tests for OpenFeature PROVIDER_CONFIGURATION_CHANGED event emission."""

    def test_flag_updated_emits_configuration_changed_with_flag_key(self, mock_server: HTTPServer):
        """flag-updated event should emit PROVIDER_CONFIGURATION_CHANGED with flags_changed."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        emitted_events = []

        def on_emit(provider_ref, event, details):
            emitted_events.append((event, details))

        provider.attach(on_emit)

        event = FlagChangeEvent(flag_key="my-feature", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        config_events = [
            (e, d) for e, d in emitted_events
            if e == ProviderEvent.PROVIDER_CONFIGURATION_CHANGED
        ]
        assert len(config_events) == 1
        _, details = config_events[0]
        assert details.flags_changed == ["my-feature"]
        provider.shutdown()

    def test_config_updated_emits_configuration_changed_without_flag_key(self, mock_server: HTTPServer):
        """config-updated event should emit PROVIDER_CONFIGURATION_CHANGED with empty flags_changed."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        emitted_events = []

        def on_emit(provider_ref, event, details):
            emitted_events.append((event, details))

        provider.attach(on_emit)

        event = FlagChangeEvent(flag_key=None, timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        config_events = [
            (e, d) for e, d in emitted_events
            if e == ProviderEvent.PROVIDER_CONFIGURATION_CHANGED
        ]
        assert len(config_events) == 1
        _, details = config_events[0]
        assert details.flags_changed is None
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


# ========================================
# Resolve Delegation Tests
# ========================================


class TestResolveDelegation:
    """Test that resolve_*_details methods delegate to the OFREP provider."""

    def test_resolve_boolean_details(self, mock_server: HTTPServer):
        """resolve_boolean_details delegates to OFREP provider."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "my-flag", {
            "key": "my-flag", "value": True, "reason": "DEFAULT", "variant": "on",
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.resolve_boolean_details("my-flag", False)
        assert result is not None
        assert result.value is True
        provider.shutdown()

    def test_resolve_string_details(self, mock_server: HTTPServer):
        """resolve_string_details delegates to OFREP provider."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "my-flag", {
            "key": "my-flag", "value": "hello", "reason": "DEFAULT", "variant": "v1",
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.resolve_string_details("my-flag", "default")
        assert result is not None
        assert result.value == "hello"
        provider.shutdown()

    def test_resolve_integer_details(self, mock_server: HTTPServer):
        """resolve_integer_details delegates to OFREP provider."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "my-flag", {
            "key": "my-flag", "value": 42, "reason": "DEFAULT", "variant": "v1",
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.resolve_integer_details("my-flag", 0)
        assert result is not None
        assert result.value == 42
        provider.shutdown()

    def test_resolve_float_details(self, mock_server: HTTPServer):
        """resolve_float_details delegates to OFREP provider."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "my-flag", {
            "key": "my-flag", "value": 3.14, "reason": "DEFAULT", "variant": "v1",
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.resolve_float_details("my-flag", 0.0)
        assert result is not None
        assert result.value == 3.14
        provider.shutdown()

    def test_resolve_object_details(self, mock_server: HTTPServer):
        """resolve_object_details delegates to OFREP provider."""
        setup_bulk_response(mock_server, {"flags": []})
        setup_flag_response(mock_server, "my-flag", {
            "key": "my-flag", "value": {"nested": "data"}, "reason": "DEFAULT", "variant": "v1",
        })

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        result = provider.resolve_object_details("my-flag", {})
        assert result is not None
        assert result.value == {"nested": "data"}
        provider.shutdown()


# ========================================
# Error Path Tests
# ========================================


class TestErrorPaths:
    """Test error handling in evaluate methods."""

    def test_evaluate_all_flags_returns_empty_on_network_exception(self):
        """evaluate_all_flags returns [] when httpx raises a network error."""
        provider = FlipswitchProvider(
            api_key="test-key",
            base_url="http://127.0.0.1:1",  # Port 1 - connection refused
            enable_realtime=False,
        )

        result = provider.evaluate_all_flags(EvaluationContext())
        assert result == []
        provider.shutdown()

    def test_evaluate_flag_returns_none_on_network_exception(self):
        """evaluate_flag returns None when httpx raises a network error."""
        provider = FlipswitchProvider(
            api_key="test-key",
            base_url="http://127.0.0.1:1",  # Port 1 - connection refused
            enable_realtime=False,
        )

        result = provider.evaluate_flag("some-flag", EvaluationContext())
        assert result is None
        provider.shutdown()


# ========================================
# Type Inference Edge Cases
# ========================================


class TestTypeInferenceEdgeCases:
    """Test edge cases in type inference."""

    def setup_method(self):
        self.provider = FlipswitchProvider(
            api_key="test-key",
            enable_realtime=False,
        )

    def teardown_method(self):
        self.provider.shutdown()

    def test_infer_type_unknown_for_unsupported(self):
        """_infer_type returns 'unknown' for unsupported types."""
        assert self.provider._infer_type(set()) == "unknown"
        assert self.provider._infer_type(object()) == "unknown"

    def test_get_flag_type_integer_from_metadata(self):
        """_get_flag_type returns 'integer' from metadata."""
        flag = {"value": 42, "metadata": {"flagType": "integer"}}
        assert self.provider._get_flag_type(flag) == "integer"

    def test_get_flag_type_string_from_metadata(self):
        """_get_flag_type returns 'string' from metadata."""
        flag = {"value": "hello", "metadata": {"flagType": "string"}}
        assert self.provider._get_flag_type(flag) == "string"


# ========================================
# SSE Lifecycle Tests
# ========================================


class TestSseLifecycle:
    """Test SSE connection lifecycle."""

    def test_initialize_starts_sse_when_realtime_enabled(self, mock_server: HTTPServer):
        """SSE client is created when enable_realtime=True."""
        setup_bulk_response(mock_server, {"flags": []})

        # Setup SSE endpoint so it doesn't error out
        mock_server.expect_request(
            "/api/v1/flags/events",
            method="GET",
        ).respond_with_data(
            "event: heartbeat\ndata: {}\n\n",
            content_type="text/event-stream",
        )

        provider = FlipswitchProvider(
            api_key="test-api-key",
            base_url=mock_server.url_for(""),
            enable_realtime=True,
        )
        provider.initialize(EvaluationContext())

        assert provider._sse_client is not None
        provider.shutdown()

    def test_shutdown_closes_sse_client(self, mock_server: HTTPServer):
        """shutdown() sets _sse_client to None."""
        setup_bulk_response(mock_server, {"flags": []})

        mock_server.expect_request(
            "/api/v1/flags/events",
            method="GET",
        ).respond_with_data(
            "event: heartbeat\ndata: {}\n\n",
            content_type="text/event-stream",
        )

        provider = FlipswitchProvider(
            api_key="test-api-key",
            base_url=mock_server.url_for(""),
            enable_realtime=True,
        )
        provider.initialize(EvaluationContext())

        assert provider._sse_client is not None

        provider.shutdown()

        assert provider._sse_client is None

    def test_reconnect_sse_is_noop_when_disabled(self):
        """reconnect_sse() does nothing when realtime is disabled."""
        provider = FlipswitchProvider(
            api_key="test-key",
            enable_realtime=False,
        )

        # Should not raise
        provider.reconnect_sse()

        assert provider._sse_client is None
        provider.shutdown()

    def test_get_sse_status_returns_client_status(self, mock_server: HTTPServer):
        """get_sse_status returns the SSE client's status when available."""
        setup_bulk_response(mock_server, {"flags": []})

        mock_server.expect_request(
            "/api/v1/flags/events",
            method="GET",
        ).respond_with_data(
            "event: heartbeat\ndata: {}\n\n",
            content_type="text/event-stream",
        )

        provider = FlipswitchProvider(
            api_key="test-api-key",
            base_url=mock_server.url_for(""),
            enable_realtime=True,
        )
        provider.initialize(EvaluationContext())

        # SSE client exists, so status should come from it
        status = provider.get_sse_status()
        assert isinstance(status, ConnectionStatus)
        provider.shutdown()


# ========================================
# Polling Internal Tests
# ========================================


class TestPollingInternals:
    """Test internal polling methods."""

    def test_poll_flags_does_not_crash(self, mock_server: HTTPServer):
        """_poll_flags runs without error and schedules next poll."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = FlipswitchProvider(
            api_key="test-api-key",
            base_url=mock_server.url_for(""),
            enable_realtime=False,
            enable_polling_fallback=True,
            polling_interval=100.0,  # Long interval so timer doesn't fire
        )
        provider.initialize(EvaluationContext())

        # Activate polling
        provider._polling_active = True
        provider._poll_flags()

        # Should have scheduled the next poll
        assert provider._polling_timer is not None
        provider.shutdown()

    def test_poll_flags_noop_when_not_active(self, mock_server: HTTPServer):
        """_poll_flags returns immediately when polling is not active."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = FlipswitchProvider(
            api_key="test-api-key",
            base_url=mock_server.url_for(""),
            enable_realtime=False,
        )
        provider.initialize(EvaluationContext())

        # Polling is not active
        provider._poll_flags()

        # No timer should have been scheduled
        assert provider._polling_timer is None
        provider.shutdown()


# ========================================
# Flag-Key-Specific Listener Tests
# ========================================


class TestFlagKeySpecificListeners:
    """Test flag-key-specific listener management and dispatch."""

    def test_key_listener_fires_on_matching_key(self, mock_server: HTTPServer):
        """Key-specific listener fires when flag key matches."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []
        provider.add_flag_change_listener(lambda e: events.append(e), flag_key="dark-mode")

        event = FlagChangeEvent(flag_key="dark-mode", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(events) == 1
        assert events[0].flag_key == "dark-mode"
        provider.shutdown()

    def test_key_listener_does_not_fire_on_non_matching_key(self, mock_server: HTTPServer):
        """Key-specific listener does NOT fire for a different flag key."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []
        provider.add_flag_change_listener(lambda e: events.append(e), flag_key="dark-mode")

        event = FlagChangeEvent(flag_key="other-flag", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(events) == 0
        provider.shutdown()

    def test_key_listener_fires_on_bulk_invalidation(self, mock_server: HTTPServer):
        """Key-specific listener fires on bulk invalidation (null flag_key)."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []
        provider.add_flag_change_listener(lambda e: events.append(e), flag_key="dark-mode")

        event = FlagChangeEvent(flag_key=None, timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(events) == 1
        assert events[0].flag_key is None
        provider.shutdown()

    def test_unsubscribe_removes_key_listener(self, mock_server: HTTPServer):
        """Calling the returned unsubscribe function removes the listener."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []
        unsub = provider.add_flag_change_listener(lambda e: events.append(e), flag_key="dark-mode")

        event = FlagChangeEvent(flag_key="dark-mode", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)
        assert len(events) == 1

        unsub()

        provider._handle_flag_change(event)
        assert len(events) == 1  # still 1 - unsubscribed
        provider.shutdown()

    def test_unsubscribe_removes_global_listener(self, mock_server: HTTPServer):
        """Global listener unsubscribe works."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []
        unsub = provider.add_flag_change_listener(lambda e: events.append(e))

        event = FlagChangeEvent(flag_key="test", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)
        assert len(events) == 1

        unsub()

        provider._handle_flag_change(event)
        assert len(events) == 1
        provider.shutdown()

    def test_multiple_listeners_for_same_key(self, mock_server: HTTPServer):
        """Multiple listeners for the same key all fire."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events1 = []
        events2 = []
        provider.add_flag_change_listener(lambda e: events1.append(e), flag_key="dark-mode")
        provider.add_flag_change_listener(lambda e: events2.append(e), flag_key="dark-mode")

        event = FlagChangeEvent(flag_key="dark-mode", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(events1) == 1
        assert len(events2) == 1
        provider.shutdown()

    def test_key_listener_exception_isolated(self, mock_server: HTTPServer):
        """One key listener throwing doesn't prevent others from firing."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []

        def bad_listener(event):
            raise RuntimeError("boom")

        provider.add_flag_change_listener(bad_listener, flag_key="dark-mode")
        provider.add_flag_change_listener(lambda e: events.append(e), flag_key="dark-mode")

        event = FlagChangeEvent(flag_key="dark-mode", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(events) == 1
        provider.shutdown()

    def test_remove_flag_change_listener_by_key(self, mock_server: HTTPServer):
        """remove_flag_change_listener with flag_key removes the correct listener."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        events = []

        def listener(event):
            events.append(event)

        provider.add_flag_change_listener(listener, flag_key="dark-mode")
        provider.remove_flag_change_listener(listener, flag_key="dark-mode")

        event = FlagChangeEvent(flag_key="dark-mode", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(events) == 0
        provider.shutdown()

    def test_global_and_key_listeners_both_fire(self, mock_server: HTTPServer):
        """Global and key-specific listeners both fire on matching event."""
        setup_bulk_response(mock_server, {"flags": []})

        provider = create_provider(mock_server)
        provider.initialize(EvaluationContext())

        global_events = []
        key_events = []

        provider.add_flag_change_listener(lambda e: global_events.append(e))
        provider.add_flag_change_listener(lambda e: key_events.append(e), flag_key="dark-mode")

        event = FlagChangeEvent(flag_key="dark-mode", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event)

        assert len(global_events) == 1
        assert len(key_events) == 1

        # Non-matching key: only global fires
        event2 = FlagChangeEvent(flag_key="other", timestamp="2024-01-01T00:00:00Z")
        provider._handle_flag_change(event2)

        assert len(global_events) == 2
        assert len(key_events) == 1
        provider.shutdown()
