"""CLI entry point for the Supervisor-Workers RAG assignment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from supervisor_workers import SupervisorAgent, resolve_data_dir


DEFAULT_QUESTION = "Hình phạt tàng trữ trái phép chất ma túy là gì?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Day08 RAG agent improved with Supervisor-Workers pattern."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="User question to answer.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DAY08_DATA_DIR"),
        help="Path to Day08 data/standardized directory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of evidence chunks to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(Path(args.data_dir) if args.data_dir else None)

    print("=" * 80)
    print("Day08 RAG Agent - Supervisor/Workers Assignment")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Question: {args.question}")
    print("-" * 80)

    supervisor = SupervisorAgent(data_dir=data_dir)
    result = supervisor.run(args.question, top_k=args.top_k)

    print("\nANSWER")
    print("-" * 80)
    print(result["answer"])

    print("\nWORKER TRACE")
    print("-" * 80)
    for item in result["trace"]:
        print(f"- {item}")

    print("\nSOURCES")
    print("-" * 80)
    for index, source in enumerate(result["sources"], start=1):
        metadata = source.get("metadata", {})
        print(
            f"{index}. [{source.get('score', 0.0):.3f}] "
            f"{metadata.get('source', 'unknown')} "
            f"({metadata.get('type', 'unknown')})"
        )


if __name__ == "__main__":
    main()

