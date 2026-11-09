"""Initialize a snapshot-bound human review workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kga.review import ReviewValidationError, initialize_review_workspace, load_consolidation_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kga-review-init", description="Initialize pending human reviews for a consolidation snapshot.")
    parser.add_argument("--consolidation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = initialize_review_workspace(load_consolidation_snapshot(args.consolidation), args.output)
        print(manifest.model_dump_json(indent=2))
        return 0
    except (ReviewValidationError, OSError, ValueError) as exc:
        print(f"kga-review-init: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
