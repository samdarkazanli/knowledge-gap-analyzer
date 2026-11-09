from hashlib import sha256

import pytest

from kga.extraction import EvidenceValidationError, extract_concepts, verify_evidence
from kga.extractors import FakeConceptExtractor
from kga.models import (
    Concept,
    ConceptCategory,
    ConceptType,
    CorpusDocument,
    DocumentFormat,
    SourcePassage,
)


def make_document(content: str = "Alpha service\nUses PGW\nFinal line") -> CorpusDocument:
    return CorpusDocument(
        source_path="systems/payments.txt",
        format=DocumentFormat.TEXT,
        content=content,
        sha256=sha256(content.encode("utf-8")).hexdigest(),
    )


def make_concept(
    *,
    name: str = "PGW",
    normalized_name: str = "pgw",
    line: int = 2,
    text: str = "PGW",
    source_path: str = "systems/payments.txt",
) -> Concept:
    return Concept(
        name=name,
        normalized_name=normalized_name,
        category=ConceptCategory.COMPANY_SPECIFIC,
        concept_type=ConceptType.ACRONYM,
        evidence=[
            SourcePassage(
                source_path=source_path,
                start_line=line,
                end_line=line,
                text=text,
            )
        ],
    )


def test_fake_extractor_drives_passage_extraction_deterministically():
    document = make_document()
    expected = make_concept()
    extractor = FakeConceptExtractor(
        {("systems/payments.txt", 1, 2): [expected]}
    )

    first = extract_concepts([document], extractor, max_lines=2)
    second = extract_concepts([document], extractor, max_lines=2)

    assert first == [expected]
    assert second == first
    assert first[0] is not expected


@pytest.mark.parametrize(
    ("concept", "message"),
    [
        (make_concept(source_path="fabricated.txt"), "Unknown evidence source path"),
        (make_concept(line=99), "outside document"),
        (make_concept(text="P-G-W"), "quotation is not present"),
    ],
)
def test_rejects_untraceable_evidence(concept, message):
    with pytest.raises(EvidenceValidationError, match=message):
        verify_evidence([concept], [make_document()])


def test_rejects_document_whose_content_no_longer_matches_hash():
    document = make_document()
    changed = document.model_copy(update={"content": "altered"})

    with pytest.raises(EvidenceValidationError, match="hash does not match"):
        verify_evidence([make_concept()], [changed])


def test_duplicate_normalized_concepts_are_merged_with_unique_evidence():
    document = make_document("PGW is deployed\nPGW handles payments")
    first = make_concept(line=1, text="PGW")
    duplicate = make_concept(
        name="Pgw",
        normalized_name=" PGW ",
        line=2,
        text="PGW handles payments",
    )
    repeated = first.model_copy(deep=True)
    extractor = FakeConceptExtractor(
        {
            ("systems/payments.txt", 1, 1): [first, repeated],
            ("systems/payments.txt", 2, 2): [duplicate],
        }
    )

    concepts = extract_concepts([document], extractor, max_lines=1)

    assert len(concepts) == 1
    assert concepts[0].name == "PGW"
    assert [(item.start_line, item.text) for item in concepts[0].evidence] == [
        (1, "PGW"),
        (2, "PGW handles payments"),
    ]
