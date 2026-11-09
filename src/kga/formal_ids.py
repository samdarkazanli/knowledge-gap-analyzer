"""Canonical, output-directory-independent identifiers for formal entities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from pydantic import BaseModel

from kga.models import Concept, CorpusDocument, KnowledgeQuestion, SourcePassage


def stable_id(prefix: str, value: Any) -> str:
    digest = sha256(canonical_json(value).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def document_id(document: CorpusDocument) -> str:
    return stable_id(
        "doc", {"source_path": document.source_path, "sha256": document.sha256}
    )


def evidence_id(evidence: SourcePassage) -> str:
    return stable_id("ev", evidence.model_dump(mode="json"))


def concept_id(concept: Concept) -> str:
    return stable_id(
        "concept",
        {
            "normalized_name": concept.normalized_name,
            "category": concept.category.value,
            "concept_type": concept.concept_type.value,
            "evidence_ids": sorted(evidence_id(item) for item in concept.evidence),
        },
    )


def corpus_snapshot_id(documents: Sequence[CorpusDocument]) -> str:
    snapshot = sorted(
        (
            {"source_path": item.source_path, "sha256": item.sha256}
            for item in documents
        ),
        key=lambda item: item["source_path"],
    )
    return stable_id("corpus", snapshot)


def question_set_id(questions: Sequence[KnowledgeQuestion]) -> str:
    payload = sorted(
        (_question_identity_payload(item) for item in questions),
        key=lambda item: item["question_id"],
    )
    return stable_id("qset", payload)


def concept_inventory_id(concepts: Sequence[Concept]) -> str:
    return stable_id("cset", sorted(concept_id(item) for item in concepts))


def evaluation_run_id(
    *,
    corpus_id: str,
    question_set: str,
    model: str | None,
    evaluator_version: str | None,
    prompt_version: str | None,
    started_at: datetime | None,
    configuration: Mapping[str, str | int | float | bool | None],
) -> str:
    return stable_id(
        "run",
        {
            "corpus_snapshot_id": corpus_id,
            "question_set_id": question_set,
            "model": model,
            "evaluator_version": evaluator_version,
            "prompt_version": prompt_version,
            "started_at": started_at.isoformat() if started_at else None,
            "configuration": dict(configuration),
        },
    )


def evaluation_id(run_id: str, question_id: str) -> str:
    return stable_id(
        "eval", {"evaluation_run_id": run_id, "question_id": question_id}
    )


def _question_identity_payload(question: KnowledgeQuestion) -> dict[str, Any]:
    """Exclude additive formal references from a legacy question-set snapshot."""

    return {
        "question_id": question.question_id,
        "question": question.question,
        "dimension": question.dimension.value,
        "concept_name": question.concept_name,
        "concept_normalized_name": question.concept_normalized_name,
        "concept_category": question.concept_category.value,
        "concept_type": question.concept_type.value,
        "evidence": [item.model_dump(mode="json") for item in question.evidence],
    }


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value
