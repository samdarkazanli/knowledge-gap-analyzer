import pytest
from pydantic import ValidationError

from kga.models import (
    Concept,
    ConceptCategory,
    ConceptType,
    SourcePassage,
)


def test_concept_requires_traceable_evidence():
    passage = SourcePassage(
        source_path="incidents/INC-001.md",
        start_line=10,
        end_line=11,
        text="The Payments Gateway uses the PGW identifier.",
    )

    concept = Concept(
        name="Payments Gateway",
        normalized_name="payments gateway",
        category=ConceptCategory.COMPANY_SPECIFIC,
        concept_type=ConceptType.APPLICATION,
        evidence=[passage],
    )

    assert concept.evidence[0].source_path == "incidents/INC-001.md"


def test_concept_rejects_missing_evidence():
    with pytest.raises(ValidationError):
        Concept(
            name="Payments Gateway",
            normalized_name="payments gateway",
            category=ConceptCategory.COMPANY_SPECIFIC,
            concept_type=ConceptType.APPLICATION,
            evidence=[],
        )


def test_source_passage_rejects_reversed_line_range():
    with pytest.raises(ValidationError, match="end_line"):
        SourcePassage(
            source_path="notes.txt",
            start_line=5,
            end_line=4,
            text="Evidence",
        )


def test_models_reject_unknown_fields():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SourcePassage(
            source_path="notes.txt",
            start_line=1,
            end_line=1,
            text="Evidence",
            invented_reference="not allowed",
        )
