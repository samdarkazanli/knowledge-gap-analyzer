from __future__ import annotations
import argparse, sys
from pathlib import Path
from kga.documentation_changes import ChangeValidationError, load_and_validate

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kga-changes-validate", description="Validate declared documentation changes against an updated corpus.")
    parser.add_argument("--tasks", required=True, type=Path); parser.add_argument("--changes", required=True, type=Path); parser.add_argument("--corpus", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _, _, result = load_and_validate(args.tasks, args.changes, args.corpus); print(result.model_dump_json(indent=2)); return 0 if result.validation_passed else 1
    except (ChangeValidationError, OSError, ValueError) as exc:
        print(f"kga-changes-validate: error: {exc}", file=sys.stderr); return 1
if __name__ == "__main__": raise SystemExit(main())
