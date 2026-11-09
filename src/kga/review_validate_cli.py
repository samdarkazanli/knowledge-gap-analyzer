"""Validate a human review workspace against its consolidation snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kga.review import ReviewValidationError, load_consolidation_snapshot, load_review_workspace, validate_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kga-review-validate", description="Validate human review decisions and snapshot binding.")
    parser.add_argument("--consolidation", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = validate_review(load_consolidation_snapshot(args.consolidation), load_review_workspace(args.review))
        print(summary.model_dump_json(indent=2))
        if summary.validation_passed:
            print(f"Review workspace is valid with {len(summary.validation_warnings)} warning(s).", file=sys.stderr)
            return 0
        print(f"Review workspace has {len(summary.validation_errors)} validation error(s).", file=sys.stderr)
        return 1
    except (ReviewValidationError, OSError, ValueError) as exc:
        print(f"kga-review-validate: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
