from hashlib import sha256

import pytest

from kga.models import CorpusDocument, DocumentFormat
from kga.passages import segment_document


def make_document(content: str) -> CorpusDocument:
    return CorpusDocument(
        source_path="nested/runbook.md",
        format=DocumentFormat.MARKDOWN,
        content=content,
        sha256=sha256(content.encode("utf-8")).hexdigest(),
    )


def test_segments_document_with_deterministic_line_ranges():
    document = make_document("one\ntwo\n\nfour\nfive\n")

    passages = segment_document(document, max_lines=2)

    assert [
        (item.start_line, item.end_line, item.text) for item in passages
    ] == [
        (1, 2, "one\ntwo"),
        (3, 4, "\nfour"),
        (5, 5, "five"),
    ]
    assert all(item.source_path == document.source_path for item in passages)
    assert all(item.document_sha256 == document.sha256 for item in passages)


def test_empty_document_has_no_passages():
    assert segment_document(make_document("")) == []


def test_rejects_invalid_passage_size():
    with pytest.raises(ValueError, match="greater than zero"):
        segment_document(make_document("content"), max_lines=0)
