"""Tests for FlipswitchProvider.

Note: These tests focus on FlipswitchProvider's specific functionality:
- Initialization and API key validation
- SSE connection management
- Bulk flag evaluation methods (evaluate_all_flags, evaluate_flag)

The OpenFeature SDK evaluation methods are delegated to the underlying OFREP provider,
which has its own test suite.
"""

import json
import pytest
from pytest_httpserver import HTTPServer

from openfeature.evaluation_context import EvaluationContext

from flipswitch import FlipswitchProvider, ConnectionStatus


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
