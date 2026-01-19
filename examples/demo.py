#!/usr/bin/env python3
"""
Sample application demonstrating Flipswitch integration with real-time SSE support.

Run this demo with:
    python examples/demo.py <api-key>
"""

import sys
import time
from typing import List

from openfeature import api
from openfeature.evaluation_context import EvaluationContext

from flipswitch import FlipswitchProvider, FlagEvaluation, FlagChangeEvent


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py <api-key>", file=sys.stderr)
        sys.exit(1)

    api_key = sys.argv[1]

    print("Flipswitch Python SDK Demo")
    print("=" * 26)
    print()

    # API key is required, all other options have sensible defaults
    provider = FlipswitchProvider(api_key=api_key)

    # Register the provider with OpenFeature
    try:
        api.set_provider(provider)
    except Exception as e:
        print(f"Failed to connect to Flipswitch: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Connected! SSE Status: {provider.get_sse_status().value}")

    # Create evaluation context with user information
    context = EvaluationContext(
        targeting_key="user-123",
        attributes={
            "email": "user@example.com",
            "plan": "premium",
            "country": "SE",
        },
    )

    # Add a listener for flag changes - re-evaluate and show new value
    def on_flag_change(event: FlagChangeEvent):
        flag_key = event.flag_key
        print(f"\n*** Flag changed: {flag_key or 'all flags'} ***")

        if flag_key:
            # Re-evaluate the specific flag that changed
            eval_result = provider.evaluate_flag(flag_key, context)
            if eval_result:
                print_flag(eval_result)
        else:
            # Bulk invalidation - re-evaluate all flags
            print_all_flags(provider, context)
        print()

    provider.add_flag_change_listener(on_flag_change)

    print("\nEvaluating flags for user: user-123")
    print("Context: email=user@example.com, plan=premium, country=SE\n")

    print_all_flags(provider, context)

    # Keep the application running to demonstrate real-time updates
    print("\n--- Listening for real-time flag updates (Ctrl+C to exit) ---")
    print("Change a flag in the Flipswitch dashboard to see it here!\n")

    # Keep running for 5 minutes to demonstrate real-time updates
    try:
        time.sleep(300)
    except KeyboardInterrupt:
        pass

    # Cleanup
    provider.shutdown()
    print("\nDemo complete!")


def print_all_flags(provider: FlipswitchProvider, context: EvaluationContext):
    """Print all flags."""
    flags: List[FlagEvaluation] = provider.evaluate_all_flags(context)

    if not flags:
        print("No flags found.")
        return

    print(f"Flags ({len(flags)}):")
    print("-" * 60)

    for flag in flags:
        print_flag(flag)


def print_flag(flag: FlagEvaluation):
    """Print a single flag evaluation."""
    variant_str = f", variant={flag.variant}" if flag.variant else ""
    print(f"  {flag.key:<30} ({flag.value_type}) = {flag.get_value_as_string()}")
    print(f"    └─ reason={flag.reason}{variant_str}")


if __name__ == "__main__":
    main()
