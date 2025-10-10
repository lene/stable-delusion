#!/usr/bin/env python3
"""
CLI tool to display token usage statistics.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import argparse
from pathlib import Path

from stable_delusion.services.token_usage_tracker import TokenUsageTracker


def format_number(num: int) -> str:
    return f"{num:,}"


def display_statistics(tracker: TokenUsageTracker) -> None:
    stats = tracker.get_statistics()

    print("\n" + "=" * 60)
    print("TOKEN USAGE STATISTICS")
    print("=" * 60)

    print(f"\nTotal Requests: {format_number(stats.total_requests)}")
    print(f"Total Tokens:   {format_number(stats.total_tokens)}")

    if stats.tokens_by_model:
        print("\n" + "-" * 60)
        print("By Model:")
        print("-" * 60)
        for model, tokens in sorted(
            stats.tokens_by_model.items(), key=lambda x: x[1], reverse=True
        ):
            requests = stats.requests_by_model.get(model, 0)
            avg_tokens = tokens / requests if requests > 0 else 0
            print(
                f"  {model:30} {format_number(tokens):>12} tokens "
                f"({format_number(requests)} requests, avg: {avg_tokens:,.0f})"
            )

    if stats.tokens_by_operation:
        print("\n" + "-" * 60)
        print("By Operation:")
        print("-" * 60)
        for operation, tokens in sorted(
            stats.tokens_by_operation.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {operation:30} {format_number(tokens):>12} tokens")

    print("\n" + "=" * 60 + "\n")


def display_history(tracker: TokenUsageTracker, limit: int) -> None:
    history = tracker.get_usage_history(limit=limit)

    if not history:
        print("\nNo token usage history found.\n")
        return

    print("\n" + "=" * 80)
    print(f"RECENT TOKEN USAGE (Last {len(history)} entries)")
    print("=" * 80)
    print(f"{'Timestamp':22} {'Model':25} {'Operation':12} {'Tokens':>10} {'Prompt Hash':>15}")
    print("-" * 80)

    for entry in history:
        prompt_hash = entry.prompt_hash or "N/A"
        print(
            f"{entry.timestamp:22} {entry.model:25} {entry.operation:12} "
            f"{entry.tokens:>10} {prompt_hash:>15}"
        )

    print("=" * 80 + "\n")


def clear_history(tracker: TokenUsageTracker) -> None:
    tracker.clear_history()
    print("\n✓ Token usage history cleared.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Display token usage statistics for image generation APIs."
    )
    parser.add_argument(
        "--history",
        "-H",
        type=int,
        metavar="N",
        help="Show the last N token usage entries",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all token usage history",
    )
    parser.add_argument(
        "--storage-file",
        type=Path,
        help="Path to token usage storage file (default: ~/.stable-delusion/token_usage.json)",
    )

    args = parser.parse_args()

    tracker = TokenUsageTracker(storage_file=args.storage_file)

    if args.clear:
        clear_history(tracker)
        return

    if args.history is not None:
        display_history(tracker, args.history)
    else:
        display_statistics(tracker)


if __name__ == "__main__":
    main()
