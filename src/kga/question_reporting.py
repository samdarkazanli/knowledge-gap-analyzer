"""Load concept inventories and export knowledge-question reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from kga.models import Concept, KnowledgeQuestion


class ConceptInventoryError(ValueError):
    """Raised when a concepts inventory cannot be loaded or validated."""


def load_concept_inventory(path: str | Path) -> list[Concept]:
    inventory_path = Path(path)
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        return TypeAdapter(list[Concept]).validate_python(payload)
    except OSError as exc:
        raise ConceptInventoryError(f"Could not read concept inventory: {inventory_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConceptInventoryError(
            f"Concept inventory is not valid JSON: {inventory_path}"
        ) from exc
    except ValidationError as exc:
        raise ConceptInventoryError(
            f"Concept inventory does not match the expected schema: {inventory_path}"
        ) from exc


def export_question_reports(
    output_directory: str | Path, questions: list[KnowledgeQuestion]
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "questions.json").write_text(
        json.dumps(
            [question.model_dump(mode="json") for question in questions],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_questions_csv(output_path / "questions.csv", questions)


def _write_questions_csv(path: Path, questions: list[KnowledgeQuestion]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "question",
                "dimension",
                "concept_name",
                "concept_normalized_name",
                "concept_category",
                "concept_type",
                "source_path",
                "start_line",
                "end_line",
                "evidence",
            ],
        )
        writer.writeheader()
        for question in questions:
            for evidence in question.evidence:
                writer.writerow(
                    {
                        "question_id": question.question_id,
                        "question": question.question,
                        "dimension": question.dimension.value,
                        "concept_name": question.concept_name,
                        "concept_normalized_name": question.concept_normalized_name,
                        "concept_category": question.concept_category.value,
                        "concept_type": question.concept_type.value,
                        "source_path": evidence.source_path,
                        "start_line": evidence.start_line,
                        "end_line": evidence.end_line,
                        "evidence": evidence.text,
                    }
                )
