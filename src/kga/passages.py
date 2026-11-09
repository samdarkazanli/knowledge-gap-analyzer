"""Deterministic segmentation of corpus documents."""

from __future__ import annotations

from collections.abc import Iterable

from kga.models import CorpusDocument, DocumentPassage


def segment_document(
    document: CorpusDocument, *, max_lines: int = 100
) -> list[DocumentPassage]:
    """Split a document into consecutive fixed-size line ranges.

    Line endings are omitted from passage text, while blank lines within a
    passage are retained. Empty documents produce no passages.
    """

    if max_lines < 1:
        raise ValueError("max_lines must be greater than zero")

    lines = document.content.splitlines()
    passages: list[DocumentPassage] = []
    for offset in range(0, len(lines), max_lines):
        passage_lines = lines[offset : offset + max_lines]
        passages.append(
            DocumentPassage(
                source_path=document.source_path,
                document_sha256=document.sha256,
                start_line=offset + 1,
                end_line=offset + len(passage_lines),
                text="\n".join(passage_lines),
            )
        )
    return passages


def segment_corpus(
    documents: Iterable[CorpusDocument], *, max_lines: int = 100
) -> list[DocumentPassage]:
    """Segment documents in input order and passages in line order."""

    return [
        passage
        for document in documents
        for passage in segment_document(document, max_lines=max_lines)
    ]
