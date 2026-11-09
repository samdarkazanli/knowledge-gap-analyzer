"""CLI for deterministic formal knowledge-model construction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kga.evaluation_reporting import QuestionInventoryError, load_question_inventory
from kga.knowledge_model import (
    ModelIntegrityError,
    build_formal_knowledge_model,
    export_formal_knowledge_model,
)
from kga.question_reporting import ConceptInventoryError, load_concept_inventory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kga-model",
        description="Build formal knowledge requirements from KGA inventories.",
    )
    parser.add_argument("--concepts", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        concepts = load_concept_inventory(args.concepts)
        questions = load_question_inventory(args.questions)
        model = build_formal_knowledge_model(concepts, questions)
        export_formal_knowledge_model(args.output, model)
        print(model.manifest.model_dump_json(indent=2))
        return 0
    except (
        ConceptInventoryError,
        QuestionInventoryError,
        ModelIntegrityError,
        OSError,
        ValueError,
    ) as exc:
        print(f"kga-model: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
