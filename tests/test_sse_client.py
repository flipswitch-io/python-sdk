"""Tests for SseClient.

Unit tests exercise internal methods directly (callbacks, status, backoff).
Integration tests use a threaded HTTP server that writes SSE frames to verify
end-to-end connection behaviour including reconnection and error handling.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import MagicMock, patch

import pytest

from flipswitch.sse_client import SseClient, MIN_RETRY_DELAY, MAX_RETRY_DELAY
from flipswitch.types import ConnectionStatus, FlagChangeEvent


# ========================================
# Helpers
# ========================================


def _make_client(
    base_url="http://localhost:9999",
    api_key="test-key",
    on_flag_change=None,
    on_status_change=None,
):
    """Create an SseClient with sensible defaults for testing."""
    return SseClient(
        base_url=base_url,
        api_key=api_key,
        on_flag_change=on_flag_change or MagicMock(),
        on_status_change=on_status_change,
    )


# ========================================
# Unit Tests
# ========================================


class TestInitialStatus:
    """Test initial client state."""

    def test_initial_status_is_disconnected(self):
        """A freshly created client should report DISCONNECTED."""
        client = _make_client()
        assert client.get_status() == ConnectionStatus.DISCONNECTED
        assert client.status == ConnectionStatus.DISCONNECTED


class TestClose:
    """Test close behaviour."""

    def test_close_sets_disconnected(self):
        """Closing the client transitions status to DISCONNECTED."""
        client = _make_client()
        # Move to a non-DISCONNECTED state first
        client._update_status(ConnectionStatus.CONNECTED)
        assert client.get_status() == ConnectionStatus.CONNECTED

        client.close()

        assert client.get_status() == ConnectionStatus.DISCONNECTED
        assert client._closed is True

    def test_close_prevents_reconnection(self):
        """_schedule_reconnect returns immediately when the client is closed."""
        client = _make_client()
        client._closed = True

        with patch("flipswitch.sse_client.time.sleep") as mock_sleep:
            client._schedule_reconnect()

        # sleep should never be called when _closed is True
        mock_sleep.assert_not_called()
        # _retry_delay should not have changed
        assert client._retry_delay == MIN_RETRY_DELAY


class TestHandleEvent:
    """Test _handle_event dispatching."""

    def test_handle_event_flag_updated(self):
        """flag-updated event invokes on_flag_change with the correct flag_key."""
        callback = MagicMock()
        client = _make_client(on_flag_change=callback)

        data = json.dumps({"flagKey": "my-flag", "timestamp": "2025-01-01T00:00:00Z"})
        client._handle_event("flag-updated", data)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, FlagChangeEvent)
        assert event.flag_key == "my-flag"
        assert event.timestamp == "2025-01-01T00:00:00Z"

    def test_handle_event_config_updated(self):
        """config-updated event invokes on_flag_change with flag_key=None."""
        callback = MagicMock()
        client = _make_client(on_flag_change=callback)

        data = json.dumps({"timestamp": "2025-01-01T00:00:00Z"})
        client._handle_event("config-updated", data)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, FlagChangeEvent)
        assert event.flag_key is None
        assert event.timestamp == "2025-01-01T00:00:00Z"

    def test_handle_event_api_key_rotated(self):
        """api-key-rotated event does NOT invoke on_flag_change."""
        callback = MagicMock()
        client = _make_client(on_flag_change=callback)

        data = json.dumps({
            "validUntil": "2025-06-01T00:00:00Z",
            "timestamp": "2025-01-01T00:00:00Z",
        })
        client._handle_event("api-key-rotated", data)

        callback.assert_not_called()

    def test_handle_event_heartbeat(self):
        """heartbeat event does NOT invoke on_flag_change."""
        callback = MagicMock()
        client = _make_client(on_flag_change=callback)

        client._handle_event("heartbeat", "")

        callback.assert_not_called()

    def test_handle_event_malformed_json(self):
        """Malformed JSON in an event that requires parsing should not raise."""
        callback = MagicMock()
        client = _make_client(on_flag_change=callback)

        # Should log an error but not crash
        client._handle_event("flag-updated", "not-valid-json{{{")

        callback.assert_not_called()


class TestStatusChangeCallback:
    """Test _update_status callback invocation."""

    def test_status_change_callback_invoked(self):
        """_update_status calls on_status_change with the new status."""
        status_callback = MagicMock()
        client = _make_client(on_status_change=status_callback)

        client._update_status(ConnectionStatus.CONNECTING)
        client._update_status(ConnectionStatus.CONNECTED)
        client._update_status(ConnectionStatus.ERROR)

        assert status_callback.call_count == 3
        assert status_callback.call_args_list[0][0][0] == ConnectionStatus.CONNECTING
        assert status_callback.call_args_list[1][0][0] == ConnectionStatus.CONNECTED
        assert status_callback.call_args_list[2][0][0] == ConnectionStatus.ERROR


class TestExponentialBackoff:
    """Test _schedule_reconnect backoff logic."""

    def test_exponential_backoff(self):
        """_retry_delay doubles after each call, capped at MAX_RETRY_DELAY."""
        client = _make_client()
        assert client._retry_delay == MIN_RETRY_DELAY  # 1.0

        delays_observed = []

        with patch("flipswitch.sse_client.time.sleep") as mock_sleep:
            # Simulate several reconnect cycles
            for _ in range(10):
                client._schedule_reconnect()
                delays_observed.append(mock_sleep.call_args[0][0])

        # First call sleeps for the initial delay (1.0)
        assert delays_observed[0] == MIN_RETRY_DELAY
        # Subsequent delays double: 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, ...
        assert delays_observed[1] == 2.0
        assert delays_observed[2] == 4.0
        assert delays_observed[3] == 8.0
        assert delays_observed[4] == 16.0
        # Should cap at MAX_RETRY_DELAY
        assert delays_observed[5] == MAX_RETRY_DELAY
        assert delays_observed[6] == MAX_RETRY_DELAY

        # The internal delay is now at the cap
        assert client._retry_delay == MAX_RETRY_DELAY


# ========================================
# Integration Tests - Mock SSE Server
# ========================================


class MockSseHandler(BaseHTTPRequestHandler):
    """HTTP request handler that serves SSE frames.

    The handler looks up behaviour controls on the server instance
    (self.server) which is set up by MockSseServer.
    """

    def do_GET(self):
        if self.path != "/api/v1/flags/events":
            self.send_error(404)
            return

        status_code = getattr(self.server, "response_status", 200)
        if status_code != 200:
            self.send_response(status_code)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Signal that a connection has been accepted
        connected_event = getattr(self.server, "connected_event", None)
        if connected_event:
            connected_event.set()

        # Write any queued events, then wait for more
        try:
            while True:
                # Check for shutdown
                shutdown_event = getattr(self.server, "shutdown_connection_event", None)
                if shutdown_event and shutdown_event.is_set():
                    break

                # Try to get a frame to write
                event_queue = getattr(self.server, "event_queue", None)
                if event_queue:
                    try:
                        frame = event_queue.pop(0)
                        self.wfile.write(frame.encode("utf-8"))
                        self.wfile.flush()
                    except IndexError:
                        pass

                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

        # Forcefully close the socket so httpx.iter_lines() gets EOF
        try:
            self.request.shutdown(2)  # SHUT_RDWR
        except Exception:
            pass
        try:
            self.request.close()
        except Exception:
            pass

    def log_message(self, format, *args):
        """Suppress request logging during tests."""
        pass


class MockSseServer:
    """A lightweight mock SSE server for integration tests.

    Runs an HTTPServer in a daemon thread.  Provides helpers to send
    SSE frames and control the connection lifecycle.
    """

    def __init__(self, response_status=200):
        self.server = HTTPServer(("127.0.0.1", 0), MockSseHandler)
        self.server.response_status = response_status
        self.server.connected_event = threading.Event()
        self.server.shutdown_connection_event = threading.Event()
        self.server.event_queue = []
        self._thread = None

    @property
    def url(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    @property
    def connected_event(self):
        return self.server.connected_event

    def start(self):
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def send_event(self, event_type, data):
        """Queue an SSE frame to be written to the next connected client."""
        frame = f"event: {event_type}\ndata: {data}\n\n"
        self.server.event_queue.append(frame)

    def close_connection(self):
        """Tell the handler to close the current connection (triggers reconnect)."""
        self.server.shutdown_connection_event.set()

    def reset_connection_gate(self):
        """Reset events so a new connection can be detected."""
        self.server.connected_event.clear()
        self.server.shutdown_connection_event.clear()
        self.server.event_queue = []

    def shutdown(self):
        # Signal the handler to close the connection
        self.server.shutdown_connection_event.set()
        # Give handler time to notice the event and break out of its loop
        time.sleep(0.15)
        # Forcefully close the server socket to unblock any pending httpx reads.
        # This is necessary because httpx.iter_lines() blocks on network I/O
        # and won't check the client's _closed flag until data arrives.
        try:
            self.server.socket.close()
        except Exception:
            pass
        # Now shutdown is non-blocking since the socket is gone
        try:
            self.server.shutdown()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=5)


@pytest.fixture
def mock_sse_server():
    """Fixture that starts and tears down a MockSseServer."""
    server = MockSseServer()
    server.start()
    yield server
    server.shutdown()


class TestConnectionReceivesConnectedStatus:
    """Integration: verify that a real connection transitions to CONNECTED."""

    def test_connection_receives_connected_status(self, mock_sse_server):
        statuses = []
        status_event = threading.Event()

        def on_status(s):
            statuses.append(s)
            if s == ConnectionStatus.CONNECTED:
                status_event.set()

        client = _make_client(
            base_url=mock_sse_server.url,
            on_status_change=on_status,
        )
        try:
            client.connect()
            assert status_event.wait(timeout=5), "Timed out waiting for CONNECTED"
            assert ConnectionStatus.CONNECTING in statuses
            assert ConnectionStatus.CONNECTED in statuses
            assert client.get_status() == ConnectionStatus.CONNECTED
        finally:
            client.close()


class TestFlagUpdatedEventDelivered:
    """Integration: flag-updated SSE event reaches the on_flag_change callback."""

    def test_flag_updated_event_delivered(self, mock_sse_server):
        received = []
        event_received = threading.Event()

        def on_flag_change(evt):
            received.append(evt)
            event_received.set()

        client = _make_client(
            base_url=mock_sse_server.url,
            on_flag_change=on_flag_change,
        )
        try:
            client.connect()
            assert mock_sse_server.connected_event.wait(timeout=5)

            mock_sse_server.send_event(
                "flag-updated",
                json.dumps({"flagKey": "feat-x", "timestamp": "2025-01-01T00:00:00Z"}),
            )

            assert event_received.wait(timeout=5), "Timed out waiting for flag event"
            assert len(received) == 1
            assert received[0].flag_key == "feat-x"
        finally:
            client.close()


class TestConfigUpdatedEventDeliveredWithNoneFlagKey:
    """Integration: config-updated SSE event sets flag_key to None."""

    def test_config_updated_event_delivered_with_none_flag_key(self, mock_sse_server):
        received = []
        event_received = threading.Event()

        def on_flag_change(evt):
            received.append(evt)
            event_received.set()

        client = _make_client(
            base_url=mock_sse_server.url,
            on_flag_change=on_flag_change,
        )
        try:
            client.connect()
            assert mock_sse_server.connected_event.wait(timeout=5)

            mock_sse_server.send_event(
                "config-updated",
                json.dumps({"timestamp": "2025-01-01T00:00:00Z"}),
            )

            assert event_received.wait(timeout=5), "Timed out waiting for config event"
            assert len(received) == 1
            assert received[0].flag_key is None
        finally:
            client.close()


class TestReconnectionOnServerClose:
    """Integration: client reconnects after the server drops the connection."""

    def test_reconnection_on_server_close(self, mock_sse_server):
        connection_count = {"value": 0}
        connected_again = threading.Event()

        def on_status(s):
            if s == ConnectionStatus.CONNECTED:
                connection_count["value"] += 1
                if connection_count["value"] >= 2:
                    connected_again.set()

        client = _make_client(
            base_url=mock_sse_server.url,
            on_status_change=on_status,
        )
        # Use a very short retry so the test does not take long
        client._retry_delay = 0.1

        try:
            client.connect()
            assert mock_sse_server.connected_event.wait(timeout=5), (
                "First connection not established"
            )

            # Drop the connection
            mock_sse_server.close_connection()
            # Prepare for the next connection
            time.sleep(0.2)
            mock_sse_server.reset_connection_gate()

            assert connected_again.wait(timeout=10), (
                "Client did not reconnect after server closed connection"
            )
            assert connection_count["value"] >= 2
        finally:
            client.close()


class TestErrorStatusOnNon200:
    """Integration: non-200 response sets status to ERROR."""

    def test_error_status_on_non_200(self):
        server = MockSseServer(response_status=503)
        server.start()

        statuses = []
        error_seen = threading.Event()

        def on_status(s):
            statuses.append(s)
            if s == ConnectionStatus.ERROR:
                error_seen.set()

        client = _make_client(
            base_url=server.url,
            on_status_change=on_status,
        )
        # Short retry to keep test fast
        client._retry_delay = 0.1

        try:
            client.connect()
            assert error_seen.wait(timeout=5), "Timed out waiting for ERROR status"
            assert ConnectionStatus.ERROR in statuses
        finally:
            client.close()
            server.shutdown()
