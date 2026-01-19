# Flipswitch Python SDK

Flipswitch SDK for Python with real-time SSE support for OpenFeature.

## Installation

```bash
pip install flipswitch-sdk
```

## Quick Start

```python
from flipswitch import FlipswitchProvider
from openfeature import api
from openfeature.evaluation_context import EvaluationContext

# Only API key is required
provider = FlipswitchProvider(api_key="YOUR_API_KEY")

# Register with OpenFeature
api.set_provider(provider)

# Get a client and evaluate flags
client = api.get_client()
dark_mode = client.get_boolean_value("dark-mode", False)
welcome_message = client.get_string_value("welcome-message", "Hello!")
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_key` | `str` | *required* | Environment API key from dashboard |
| `base_url` | `str` | `https://api.flipswitch.dev` | Your Flipswitch server URL |
| `enable_realtime` | `bool` | `True` | Enable SSE for real-time flag updates |

```python
provider = FlipswitchProvider(
    api_key="YOUR_API_KEY",
    base_url="https://api.flipswitch.dev",
    enable_realtime=True,
)
```

## Evaluation Context

Pass user attributes for targeting:

```python
context = EvaluationContext(
    targeting_key="user-123",
    attributes={
        "email": "user@example.com",
        "plan": "premium",
        "country": "SE",
    },
)

show_feature = client.get_boolean_value("new-feature", False, context)
```

## Real-Time Updates

When `enable_realtime=True` (default), the SDK maintains an SSE connection to receive instant flag changes:

### Event Listeners

```python
from flipswitch import FlagChangeEvent

def on_flag_change(event: FlagChangeEvent):
    print(f"Flag changed: {event.flag_key}")

provider.add_flag_change_listener(on_flag_change)
```

### Connection Status

```python
# Check current SSE status
status = provider.get_sse_status()
# ConnectionStatus.CONNECTING, CONNECTED, DISCONNECTED, ERROR

# Force reconnect
provider.reconnect_sse()
```

## Bulk Flag Evaluation

Evaluate all flags at once:

```python
# Evaluate all flags
flags = provider.evaluate_all_flags(context)
for flag in flags:
    print(f"{flag.key} ({flag.value_type}): {flag.get_value_as_string()}")

# Evaluate a single flag with full details
flag = provider.evaluate_flag("dark-mode", context)
if flag:
    print(f"Value: {flag.value}, Reason: {flag.reason}, Variant: {flag.variant}")
```

## Shutdown

Always shutdown the provider when done:

```python
provider.shutdown()
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run demo
python examples/demo.py <api-key>
```

## License

MIT
