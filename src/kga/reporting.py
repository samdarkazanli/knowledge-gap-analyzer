"""Build and export KGA V0.1 reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kga.models import (
    CategoryStatistics,
    Concept,
    ConceptCategory,
    ExecutionSummary,
)


def build_summary(
    concepts: list[Concept],
    *,
    documents_processed: int,
    passages_processed: int,
    api_calls: int,
) -> ExecutionSummary:
    counts = {category: 0 for category in ConceptCategory}
    for concept in concepts:
        counts[concept.category] += 1
    return ExecutionSummary(
        documents_processed=documents_processed,
        passages_processed=passages_processed,
        api_calls=api_calls,
        concepts_discovered=len(concepts),
        concepts_by_category=CategoryStatistics(
            general_technical=counts[ConceptCategory.GENERAL_TECHNICAL],
            company_specific=counts[ConceptCategory.COMPANY_SPECIFIC],
            uncertain=counts[ConceptCategory.UNCERTAIN],
        ),
    )


def export_reports(
    output_directory: str | Path,
    concepts: list[Concept],
    summary: ExecutionSummary,
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "concepts.json").write_text(
        json.dumps(
            [concept.model_dump(mode="json") for concept in concepts],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_path / "execution-summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_path / "concepts.csv", concepts)


def _write_csv(path: Path, concepts: list[Concept]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "normalized_name",
                "category",
                "concept_type",
                "description",
                "source_path",
                "start_line",
                "end_line",
                "evidence",
            ],
        )
        writer.writeheader()
        for concept in concepts:
            for evidence in concept.evidence:
                writer.writerow(
                    {
                        "name": concept.name,
                        "normalized_name": concept.normalized_name,
                        "category": concept.category.value,
                        "concept_type": concept.concept_type.value,
                        "description": concept.description or "",
                        "source_path": evidence.source_path,
                        "start_line": evidence.start_line,
                        "end_line": evidence.end_line,
                        "evidence": evidence.text,
                    }
                )
