"""Tests for flipswitch.types module."""

from datetime import datetime, timezone

from flipswitch.types import (
    ApiKeyRotatedEvent,
    ConfigUpdatedEvent,
    FlagChangeEvent,
    FlagEvaluation,
    FlagUpdatedEvent,
)


# ========================================
# Timestamp Helpers
# ========================================


class TestFlagUpdatedEventTimestamp:
    def test_get_timestamp_as_datetime(self):
        event = FlagUpdatedEvent(flag_key="f", timestamp="2025-01-15T10:30:00Z")
        dt = event.get_timestamp_as_datetime()
        assert dt == datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_get_timestamp_as_datetime_empty(self):
        event = FlagUpdatedEvent(flag_key="f", timestamp="")
        assert event.get_timestamp_as_datetime() is None


class TestConfigUpdatedEventTimestamp:
    def test_get_timestamp_as_datetime(self):
        event = ConfigUpdatedEvent(timestamp="2025-06-01T12:00:00Z")
        dt = event.get_timestamp_as_datetime()
        assert dt == datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_get_timestamp_as_datetime_empty(self):
        event = ConfigUpdatedEvent(timestamp="")
        assert event.get_timestamp_as_datetime() is None


class TestApiKeyRotatedEventTimestamp:
    def test_get_valid_until_as_datetime(self):
        event = ApiKeyRotatedEvent(valid_until="2025-12-31T23:59:59Z", timestamp="2025-01-01T00:00:00Z")
        dt = event.get_valid_until_as_datetime()
        assert dt == datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_get_valid_until_as_datetime_none(self):
        event = ApiKeyRotatedEvent(valid_until=None, timestamp="2025-01-01T00:00:00Z")
        assert event.get_valid_until_as_datetime() is None

    def test_get_timestamp_as_datetime(self):
        event = ApiKeyRotatedEvent(valid_until=None, timestamp="2025-03-10T08:00:00Z")
        dt = event.get_timestamp_as_datetime()
        assert dt == datetime(2025, 3, 10, 8, 0, 0, tzinfo=timezone.utc)


class TestFlagChangeEventTimestamp:
    def test_get_timestamp_as_datetime(self):
        event = FlagChangeEvent(flag_key="my-flag", timestamp="2025-07-04T16:45:00Z")
        dt = event.get_timestamp_as_datetime()
        assert dt == datetime(2025, 7, 4, 16, 45, 0, tzinfo=timezone.utc)

    def test_get_timestamp_as_datetime_empty(self):
        event = FlagChangeEvent(flag_key=None, timestamp="")
        assert event.get_timestamp_as_datetime() is None


# ========================================
# FlagEvaluation Helpers
# ========================================


class TestFlagEvaluationAsFloat:
    def test_as_float_returns_float_value(self):
        fe = FlagEvaluation(key="f", value=3.14, value_type="number")
        assert fe.as_float() == 3.14

    def test_as_float_returns_zero_for_none(self):
        fe = FlagEvaluation(key="f", value=None, value_type="number")
        assert fe.as_float() == 0.0


class TestFlagEvaluationGetValueAsString:
    def test_none_value(self):
        fe = FlagEvaluation(key="f", value=None, value_type="null")
        assert fe.get_value_as_string() == "null"

    def test_string_value(self):
        fe = FlagEvaluation(key="f", value="hello", value_type="string")
        assert fe.get_value_as_string() == '"hello"'

    def test_bool_value(self):
        fe = FlagEvaluation(key="f", value=True, value_type="boolean")
        assert fe.get_value_as_string() == "true"

    def test_bool_false_value(self):
        fe = FlagEvaluation(key="f", value=False, value_type="boolean")
        assert fe.get_value_as_string() == "false"

    def test_number_value(self):
        fe = FlagEvaluation(key="f", value=42, value_type="integer")
        assert fe.get_value_as_string() == "42"


class TestFlagEvaluationStr:
    def test_str_without_variant(self):
        fe = FlagEvaluation(key="dark-mode", value=True, value_type="boolean", reason="DEFAULT")
        result = str(fe)
        assert result == "dark-mode (boolean): true [reason=DEFAULT]"

    def test_str_with_variant(self):
        fe = FlagEvaluation(key="color", value="blue", value_type="string", reason="TARGETING_MATCH", variant="v2")
        result = str(fe)
        assert result == 'color (string): "blue" [reason=TARGETING_MATCH, variant=v2]'


class TestFlagEvaluationAsBoolean:
    def test_as_boolean_for_none(self):
        fe = FlagEvaluation(key="f", value=None, value_type="boolean")
        assert fe.as_boolean() is False
