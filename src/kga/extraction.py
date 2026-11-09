"""Concept extraction orchestration, evidence validation, and deduplication."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
import unicodedata

from kga.extractors import ConceptExtractor
from kga.models import Concept, CorpusDocument, SourcePassage
from kga.passages import segment_corpus


class EvidenceValidationError(ValueError):
    """Raised when extracted evidence is not traceable to the corpus."""


def verify_evidence(
    concepts: Iterable[Concept], documents: Iterable[CorpusDocument]
) -> None:
    """Verify every evidence quotation against its declared source and lines."""

    documents_by_path: dict[str, CorpusDocument] = {}
    for document in documents:
        if document.source_path in documents_by_path:
            raise EvidenceValidationError(
                f"Duplicate source path in corpus: {document.source_path}"
            )
        actual_hash = sha256(document.content.encode("utf-8")).hexdigest()
        if actual_hash != document.sha256:
            raise EvidenceValidationError(
                f"Document hash does not match content: {document.source_path}"
            )
        documents_by_path[document.source_path] = document

    for concept in concepts:
        for evidence in concept.evidence:
            document = documents_by_path.get(evidence.source_path)
            if document is None:
                raise EvidenceValidationError(
                    f"Unknown evidence source path: {evidence.source_path}"
                )
            lines = document.content.splitlines()
            if evidence.start_line > len(lines) or evidence.end_line > len(lines):
                raise EvidenceValidationError(
                    "Evidence line range is outside document "
                    f"{evidence.source_path}: {evidence.start_line}-{evidence.end_line}"
                )
            selected_text = "\n".join(
                lines[evidence.start_line - 1 : evidence.end_line]
            )
            if evidence.text not in selected_text:
                raise EvidenceValidationError(
                    "Evidence quotation is not present at declared location "
                    f"{evidence.source_path}:{evidence.start_line}-{evidence.end_line}"
                )


def deduplicate_concepts(concepts: Iterable[Concept]) -> list[Concept]:
    """Merge concepts sharing a normalized name, retaining unique evidence."""

    deduplicated: dict[str, Concept] = {}
    for concept in concepts:
        key = normalize_concept_name(concept.name)
        concept = concept.model_copy(update={"normalized_name": key}, deep=True)
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = concept.model_copy(deep=True)
            continue

        evidence_keys = {_evidence_key(item) for item in existing.evidence}
        for evidence in concept.evidence:
            if _evidence_key(evidence) not in evidence_keys:
                existing.evidence.append(evidence.model_copy(deep=True))
                evidence_keys.add(_evidence_key(evidence))
    return list(deduplicated.values())


def normalize_concept_name(name: str) -> str:
    """Return the stable normalization used as the deduplication key."""

    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def extract_concepts(
    documents: Iterable[CorpusDocument],
    extractor: ConceptExtractor,
    *,
    max_lines: int = 100,
) -> list[Concept]:
    """Extract, validate, and deduplicate concepts from corpus passages."""

    corpus = list(documents)
    extracted: list[Concept] = []
    for passage in segment_corpus(corpus, max_lines=max_lines):
        passage_concepts = extractor.extract(passage)
        # Validate immediately so unsupported output stops further API calls.
        verify_evidence(passage_concepts, corpus)
        extracted.extend(passage_concepts)
    return deduplicate_concepts(extracted)


def _evidence_key(evidence: SourcePassage) -> tuple[str, int, int, str]:
    return (
        evidence.source_path,
        evidence.start_line,
        evidence.end_line,
        evidence.text,
    )
