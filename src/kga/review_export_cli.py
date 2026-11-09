"""Export human-approved remediation proposals without mutating source artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kga.review import ReviewValidationError, export_review, load_consolidation_snapshot, load_review_workspace, validate_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kga-review-export", description="Export validated human-approved remediation proposals.")
    parser.add_argument("--consolidation", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshot = load_consolidation_snapshot(args.consolidation)
        workspace = load_review_workspace(args.review)
        summary = validate_review(snapshot, workspace)
        if not summary.validation_passed:
            raise ReviewValidationError(summary.validation_errors)
        manifest = export_review(snapshot, workspace, summary, args.output)
        print(manifest.model_dump_json(indent=2))
        return 0
    except (ReviewValidationError, OSError, ValueError) as exc:
        print(f"kga-review-export: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
