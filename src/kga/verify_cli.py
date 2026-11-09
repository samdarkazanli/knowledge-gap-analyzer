from __future__ import annotations
import argparse, sys
from pathlib import Path
from kga.documentation_changes import load_and_validate, ChangeValidationError
from kga.remediation_verification import VerificationValidationError, export_verification, load_evaluation, verify_remediations

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kga-verify", description="Compare baseline and updated answerability for documentation tasks.")
    parser.add_argument("--tasks", required=True, type=Path); parser.add_argument("--changes", required=True, type=Path)
    parser.add_argument("--baseline-evaluation", required=True, type=Path); parser.add_argument("--updated-evaluation", required=True, type=Path)
    parser.add_argument("--baseline-corpus", required=True, type=Path); parser.add_argument("--updated-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); args = parser.parse_args(argv)
    try:
        tasks, changes, validation = load_and_validate(args.tasks, args.changes, args.updated_corpus)
        inventory, summary, manifest = verify_remediations(
            tasks, changes, validation, load_evaluation(args.baseline_evaluation),
            load_evaluation(args.updated_evaluation), args.baseline_corpus, args.updated_corpus,
        )
        export_verification(args.output, inventory, summary, manifest); print(summary.model_dump_json(indent=2)); return 0
    except (ChangeValidationError, VerificationValidationError, OSError, ValueError) as exc:
        print(f"kga-verify: error: {exc}", file=sys.stderr); return 1
if __name__ == "__main__": raise SystemExit(main())
