"""CLI for deterministic knowledge-gap reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kga.gap_reporting import (
    ReportValidationError,
    build_report,
    export_report,
    load_reporting_inputs,
    make_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kga-report",
        description="Join a formal KGA model with answerability evaluations.",
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict", action="store_true", help="Require evaluation-run provenance metadata.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model, evaluations, run, warnings, fingerprints = load_reporting_inputs(
            args.model, args.evaluation, strict=args.strict
        )
        coverage, gaps, summary = build_report(model, evaluations, evaluation_run=run)
        manifest = make_manifest(
            model, coverage, gaps, summary, fingerprints, warnings,
            strict=args.strict, evaluation_run=run,
        )
        export_report(args.output, coverage, gaps, summary, manifest)
        print(manifest.model_dump_json(indent=2))
        return 0
    except (ReportValidationError, OSError, ValueError) as exc:
        print(f"kga-report: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
