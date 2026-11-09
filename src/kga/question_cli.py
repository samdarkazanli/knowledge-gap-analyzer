"""Command-line interface for deterministic knowledge-question generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kga.models import ConceptCategory
from kga.question_generators import TemplateQuestionGenerator, generate_questions
from kga.question_reporting import (
    ConceptInventoryError,
    export_question_reports,
    load_concept_inventory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kga-questions",
        description="Generate company-specific knowledge questions from concepts.json.",
    )
    parser.add_argument("--concepts", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        concepts = load_concept_inventory(args.concepts)
        questions = generate_questions(concepts, TemplateQuestionGenerator())
        export_question_reports(args.output, questions)
        summary = {
            "concepts_loaded": len(concepts),
            "company_specific_concepts": sum(
                concept.category is ConceptCategory.COMPANY_SPECIFIC
                for concept in concepts
            ),
            "questions_generated": len(questions),
        }
        print(json.dumps(summary, indent=2))
        return 0
    except (ConceptInventoryError, OSError) as exc:
        print(f"kga-questions: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
