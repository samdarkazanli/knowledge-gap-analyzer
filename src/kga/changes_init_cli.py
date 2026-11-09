from __future__ import annotations
import argparse, sys
from pathlib import Path
from kga.documentation_changes import ChangeValidationError, initialize_changes
from kga.documentation_tasks import TaskValidationError, load_task_set

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kga-changes-init", description="Initialize documentation change declarations for tasks.")
    parser.add_argument("--tasks", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(initialize_changes(load_task_set(args.tasks), args.output).model_dump_json(indent=2)); return 0
    except (TaskValidationError, ChangeValidationError, OSError, ValueError) as exc:
        print(f"kga-changes-init: error: {exc}", file=sys.stderr); return 1
if __name__ == "__main__": raise SystemExit(main())
