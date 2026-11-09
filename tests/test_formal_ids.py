from hashlib import sha256

import pytest
from pydantic import ValidationError

from kga.formal_ids import (
    corpus_snapshot_id,
    document_id,
    evidence_id,
    question_set_id,
)
from kga.formal_models import FormalDocument, FormalEvidence, FormalQuestion
from kga.models import CorpusDocument, DocumentFormat, SourcePassage
from test_evaluation import make_question


def document(path: str, content: str) -> CorpusDocument:
    return CorpusDocument(
        source_path=path,
        format=DocumentFormat.TEXT,
        content=content,
        sha256=sha256(content.encode("utf-8")).hexdigest(),
    )


def test_document_identity_uses_relative_path_and_content_fingerprint():
    original = document("one.txt", "content")
    same = document("one.txt", "content")
    moved = document("nested/one.txt", "content")
    changed = document("one.txt", "changed")

    assert document_id(original) == document_id(same)
    assert document_id(original) != document_id(moved)
    assert document_id(original) != document_id(changed)

    formal = FormalDocument(
        document_id=document_id(original),
        source_path=original.source_path,
        format=original.format,
        sha256=original.sha256,
    )
    assert formal.model_dump(mode="json")["document_id"].startswith("doc_")


def test_evidence_identity_is_deterministic_and_provenance_sensitive():
    first = SourcePassage(
        source_path="one.txt", start_line=2, end_line=2, text="exact quote"
    )
    changed_line = first.model_copy(update={"start_line": 3, "end_line": 3})

    assert evidence_id(first) == evidence_id(first.model_copy(deep=True))
    assert evidence_id(first) != evidence_id(changed_line)

    with pytest.raises(ValidationError, match="end_line"):
        FormalEvidence(
            evidence_id=evidence_id(first),
            source_path="one.txt",
            start_line=3,
            end_line=2,
            text="invalid",
        )


def test_corpus_snapshot_is_order_independent_and_changes_with_content():
    first = document("one.txt", "one")
    second = document("two.txt", "two")

    assert corpus_snapshot_id([first, second]) == corpus_snapshot_id([second, first])
    assert corpus_snapshot_id([first, second]) != corpus_snapshot_id(
        [first, document("two.txt", "changed")]
    )


def test_question_set_fingerprint_is_order_independent_and_ignores_enrichment():
    first = make_question()
    second = make_question(
        "q_0000000000000002", "Which team owns Atlas?"
    )
    enriched = FormalQuestion(
        **first.model_dump(),
        concept_id="concept_000000000000000000000000",
        requirement_id="kr_000000000000000000000000",
    )

    assert question_set_id([first, second]) == question_set_id([second, first])
    assert question_set_id([first]) == question_set_id([enriched])
    assert question_set_id([first]) != question_set_id([second])
