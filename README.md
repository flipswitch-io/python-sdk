# Flipswitch Python SDK 

[![CI](https://github.com/flipswitch-io/python-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/flipswitch-io/python-sdk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/flipswitch-sdk.svg)](https://pypi.org/project/flipswitch-sdk/)
[![codecov](https://codecov.io/gh/flipswitch-io/python-sdk/branch/main/graph/badge.svg)](https://codecov.io/gh/flipswitch-io/python-sdk)

Flipswitch SDK for Python with real-time SSE support for OpenFeature.

This SDK provides an OpenFeature-compatible provider that wraps OFREP flag evaluation with automatic cache invalidation via Server-Sent Events (SSE). When flags change in your Flipswitch dashboard, connected clients receive updates in real-time.

## Overview

- **OpenFeature Compatible**: Works with the OpenFeature standard for feature flags
- **Real-Time Updates**: SSE connection delivers instant flag changes
- **Polling Fallback**: Automatic fallback when SSE connection fails
- **Thread-Safe**: Safe for multi-threaded applications

## Requirements

- Python 3.9+
- `openfeature-sdk`
- `openfeature-provider-ofrep`
- `httpx`

## Installation

```bash
pip install flipswitch-sdk
```

## Quick Start

```python
from flipswitch import FlipswitchProvider
from openfeature import api

# Create and register the provider
provider = FlipswitchProvider(api_key="your-environment-api-key")
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
| `base_url` | `str` | `https://api.flipswitch.io` | Your Flipswitch server URL |
| `enable_realtime` | `bool` | `True` | Enable SSE for real-time flag updates |
| `http_client` | `httpx.Client` | `None` | Custom HTTP client |
| `enable_polling_fallback` | `bool` | `True` | Fall back to polling when SSE fails |
| `polling_interval` | `float` | `30.0` | Polling interval in seconds |
| `max_sse_retries` | `int` | `5` | Max SSE retries before polling fallback |

## Usage Examples

### Basic Flag Evaluation

```python
client = api.get_client()

# Boolean flag
dark_mode = client.get_boolean_value("dark-mode", False)

# String flag
welcome_message = client.get_string_value("welcome-message", "Hello!")

# Integer flag
max_items = client.get_integer_value("max-items", 10)

# Float flag
discount = client.get_float_value("discount-rate", 0.0)

# Object flag
config = client.get_object_value("feature-config", {"enabled": False})
```

### Evaluation Context

Target specific users or segments:

```python
from openfeature.evaluation_context import EvaluationContext

context = EvaluationContext(
    targeting_key="user-123",
    attributes={
        "email": "user@example.com",
        "plan": "premium",
        "country": "US",
        "beta_user": True,
    },
)

show_feature = client.get_boolean_value("new-feature", False, context)
```

### Real-Time Updates (SSE)

Listen for flag changes:

```python
provider = FlipswitchProvider(api_key="your-api-key")

# Listen for all flag changes (flag_key is None for bulk invalidation)
provider.add_flag_change_listener(lambda e: print(f"Flag changed: {e.flag_key}"))

# Listen for a specific flag (also fires on bulk invalidation)
unsub = provider.add_flag_change_listener(
    lambda e: print("dark-mode changed, re-evaluating..."),
    flag_key="dark-mode",
)
unsub()  # stop listening

provider.get_sse_status()  # current status
provider.reconnect_sse()   # force reconnect
```

### Bulk Flag Evaluation

Evaluate all flags at once:

```python
flags = provider.evaluate_all_flags(context)
for flag in flags:
    print(f"{flag.key} ({flag.value_type}): {flag.get_value_as_string()}")
    print(f"  Reason: {flag.reason}, Variant: {flag.variant}")

# Single flag with full details
flag = provider.evaluate_flag("dark-mode", context)
if flag:
    print(f"Value: {flag.value}")
    print(f"Reason: {flag.reason}")
    print(f"Variant: {flag.variant}")
```

## Advanced Features

### Polling Fallback

When SSE connection fails repeatedly, the SDK falls back to polling:

```python
provider = FlipswitchProvider(
    api_key="your-api-key",
    enable_polling_fallback=True,  # default: True
    polling_interval=30.0,         # Poll every 30 seconds
    max_sse_retries=5,             # Fall back after 5 failed SSE attempts
)

# Check if polling is active
if provider.is_polling_active():
    print("Polling fallback is active")
```

### Custom HTTP Client

Provide a custom httpx client for special requirements:

```python
import httpx

custom_client = httpx.Client(
    timeout=60.0,
    limits=httpx.Limits(max_connections=10),
)

provider = FlipswitchProvider(
    api_key="your-api-key",
    http_client=custom_client,
)
```

## Framework Integration

### Django

```python
# settings.py or apps.py
from flipswitch import FlipswitchProvider
from openfeature import api

FLIPSWITCH_API_KEY = "your-api-key"

def configure_feature_flags():
    provider = FlipswitchProvider(api_key=FLIPSWITCH_API_KEY)
    api.set_provider(provider)

# Call in AppConfig.ready()
```

```python
# views.py
from openfeature import api
from openfeature.evaluation_context import EvaluationContext

def my_view(request):
    client = api.get_client()

    context = EvaluationContext(
        targeting_key=str(request.user.id),
        attributes={"email": request.user.email},
    )

    if client.get_boolean_value("new-feature", False, context):
        return render(request, "new_feature.html")
    return render(request, "old_feature.html")
```

### Flask

```python
from flask import Flask
from flipswitch import FlipswitchProvider
from openfeature import api

app = Flask(__name__)

@app.before_first_request
def setup_feature_flags():
    provider = FlipswitchProvider(api_key="your-api-key")
    api.set_provider(provider)

@app.route("/")
def index():
    client = api.get_client()
    dark_mode = client.get_boolean_value("dark-mode", False)
    return f"Dark mode: {dark_mode}"
```

### FastAPI

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from flipswitch import FlipswitchProvider
from openfeature import api

provider = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider
    provider = FlipswitchProvider(api_key="your-api-key")
    api.set_provider(provider)
    yield
    provider.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    client = api.get_client()
    dark_mode = client.get_boolean_value("dark-mode", False)
    return {"dark_mode": dark_mode}
```

## Error Handling

The SDK handles errors gracefully:

```python
from openfeature.exception import OpenFeatureError

try:
    provider = FlipswitchProvider(api_key="your-api-key")
    api.set_provider(provider)
except OpenFeatureError as e:
    print(f"Failed to initialize: {e}")
    # Provider will use default values

# Flag evaluation never throws - returns default value on error
value = client.get_boolean_value("my-flag", False)
```

## Logging

Configure logging to debug issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("flipswitch")
logger.setLevel(logging.DEBUG)

# You'll see logs like:
# INFO:flipswitch.provider:Flipswitch provider initialized (realtime=True)
# DEBUG:flipswitch.sse_client:SSE connection established
# DEBUG:flipswitch.sse_client:Flag updated event: FlagChangeEvent(...)
```

## Testing

Mock the provider in your tests:

```python
from unittest.mock import Mock
from openfeature import api
from openfeature.provider.in_memory_provider import InMemoryProvider

def test_with_mock_flags():
    # Use InMemoryProvider for testing
    test_provider = InMemoryProvider({
        "dark-mode": True,
        "max-items": 10,
    })
    api.set_provider(test_provider)

    client = api.get_client()
    assert client.get_boolean_value("dark-mode", False) == True
```

## API Reference

### FlipswitchProvider

```python
class FlipswitchProvider(AbstractProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.flipswitch.io",
        enable_realtime: bool = True,
        http_client: Optional[httpx.Client] = None,
        enable_polling_fallback: bool = True,
        polling_interval: float = 30.0,
        max_sse_retries: int = 5,
    ): ...

    # OpenFeature Provider interface
    def initialize(self, evaluation_context: EvaluationContext) -> None: ...
    def shutdown(self) -> None: ...
    def resolve_boolean_details(...) -> FlagResolutionDetails[bool]: ...
    def resolve_string_details(...) -> FlagResolutionDetails[str]: ...
    def resolve_integer_details(...) -> FlagResolutionDetails[int]: ...
    def resolve_float_details(...) -> FlagResolutionDetails[float]: ...
    def resolve_object_details(...) -> FlagResolutionDetails[Union[Dict, List]]: ...

    # Flipswitch-specific methods
    def get_sse_status(self) -> ConnectionStatus: ...
    def reconnect_sse(self) -> None: ...
    def is_polling_active(self) -> bool: ...
    def add_flag_change_listener(listener: Callable[[FlagChangeEvent], None]) -> None: ...
    def remove_flag_change_listener(listener: Callable[[FlagChangeEvent], None]) -> None: ...
    def evaluate_all_flags(context: Optional[EvaluationContext]) -> List[FlagEvaluation]: ...
    def evaluate_flag(flag_key: str, context: Optional[EvaluationContext]) -> Optional[FlagEvaluation]: ...
```

### Types

```python
@dataclass
class FlagChangeEvent:
    flag_key: Optional[str]  # None for bulk invalidation
    timestamp: str

class ConnectionStatus(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class FlagEvaluation:
    key: str
    value: Any
    value_type: str
    reason: Optional[str]
    variant: Optional[str]
```

## Troubleshooting

### SSE Connection Fails

- Check that your API key is valid
- Verify your server URL is correct
- Check for network/firewall issues blocking SSE
- The SDK will automatically fall back to polling

### Flags Not Updating in Real-Time

- Ensure `enable_realtime` is not set to `False`
- Check SSE status with `provider.get_sse_status()`
- Check logs for error messages

### Provider Initialization Fails

- Verify your API key is correct
- Check network connectivity to the Flipswitch server
- Review logs for detailed error messages

## Demo

Run the included demo:

```bash
pip install -e ".[dev]"
python examples/demo.py <your-api-key>
```

The demo will connect, display all flags, and listen for real-time updates.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE) for details.
