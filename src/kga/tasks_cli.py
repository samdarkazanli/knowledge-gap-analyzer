"""CLI for generating documentation tasks from approved remediations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kga.documentation_tasks import TaskValidationError, export_documentation_tasks, generate_documentation_tasks, load_approved_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kga-tasks", description="Generate documentation tasks from human-approved remediations.")
    parser.add_argument("--approved-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = load_approved_plan(args.approved_plan)
        inventory = generate_documentation_tasks(plan)
        print(export_documentation_tasks(args.output, inventory, plan).model_dump_json(indent=2))
        return 0
    except (TaskValidationError, OSError, ValueError) as exc:
        print(f"kga-tasks: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
