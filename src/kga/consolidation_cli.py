"""CLI for offline knowledge-gap consolidation and remediation planning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kga.consolidation import (
    ConsolidationValidationError,
    consolidate_gaps,
    export_consolidation,
    load_consolidation_config,
    load_consolidation_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kga-consolidate",
        description="Consolidate report findings and produce a remediation plan.",
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = load_consolidation_inputs(args.report)
        config = load_consolidation_config(args.config)
        inventory, plan, summary = consolidate_gaps(inputs, config)
        export_consolidation(args.output, inventory, plan, summary)
        print(summary.model_dump_json(indent=2))
        return 0
    except (ConsolidationValidationError, OSError, ValueError) as exc:
        print(f"kga-consolidate: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
