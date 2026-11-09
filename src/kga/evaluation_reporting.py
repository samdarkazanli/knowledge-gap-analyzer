"""Load question inventories and export answerability evaluation reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from kga.evaluation_provenance import formalize_evaluations
from kga.formal_models import EvaluationRun
from kga.models import EvaluationSummary, KnowledgeQuestion, QuestionEvaluation


class QuestionInventoryError(ValueError):
    """Raised when a generated-question inventory cannot be loaded."""


def load_question_inventory(path: str | Path) -> list[KnowledgeQuestion]:
    inventory_path = Path(path)
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        return TypeAdapter(list[KnowledgeQuestion]).validate_python(payload)
    except OSError as exc:
        raise QuestionInventoryError(
            f"Could not read question inventory: {inventory_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise QuestionInventoryError(
            f"Question inventory is not valid JSON: {inventory_path}"
        ) from exc
    except ValidationError as exc:
        raise QuestionInventoryError(
            f"Question inventory does not match the expected schema: {inventory_path}"
        ) from exc


def export_evaluation_reports(
    output_directory: str | Path,
    results: list[QuestionEvaluation],
    summary: EvaluationSummary,
    evaluation_run: EvaluationRun | None = None,
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "answerability.json").write_text(
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_path / "evaluation-summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_path / "answerability.csv", results)
    if evaluation_run is not None:
        (output_path / "evaluation-run.json").write_text(
            evaluation_run.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (output_path / "formal-evaluations.json").write_text(
            json.dumps(
                [
                    item.model_dump(mode="json")
                    for item in formalize_evaluations(results, evaluation_run)
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _write_csv(path: Path, results: list[QuestionEvaluation]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "question",
                "concept_name",
                "concept_normalized_name",
                "concept_category",
                "concept_type",
                "dimension",
                "status",
                "answer",
                "explanation",
                "supporting_evidence",
                "conflicting_evidence",
                "confidence",
                "evaluation_attempts",
                "retry_attempted",
                "retry_succeeded",
                "retry_skipped_reason",
                "evidence_validation_diagnostics",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "question_id": result.question_id,
                    "question": result.question,
                    "concept_name": result.concept_name,
                    "concept_normalized_name": result.concept_normalized_name,
                    "concept_category": result.concept_category.value,
                    "concept_type": result.concept_type.value,
                    "dimension": result.dimension.value,
                    "status": result.status.value,
                    "answer": result.answer or "",
                    "explanation": result.explanation,
                    "supporting_evidence": json.dumps(
                        [item.model_dump(mode="json") for item in result.supporting_evidence],
                        ensure_ascii=False,
                    ),
                    "conflicting_evidence": json.dumps(
                        [item.model_dump(mode="json") for item in result.conflicting_evidence],
                        ensure_ascii=False,
                    ),
                    "confidence": result.confidence,
                    "evaluation_attempts": result.evaluation_attempts,
                    "retry_attempted": result.retry_attempted,
                    "retry_succeeded": result.retry_succeeded,
                    "retry_skipped_reason": result.retry_skipped_reason or "",
                    "evidence_validation_diagnostics": json.dumps(
                        [
                            item.model_dump(mode="json")
                            for item in result.evidence_validation_diagnostics
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
