from hashlib import sha256
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from kga.models import (
    Concept,
    ConceptCategory,
    ConceptExtraction,
    ConceptType,
    DocumentPassage,
    SourcePassage,
)
from kga.openai_extractor import (
    ApiCallLimitError,
    ExtractionProviderError,
    ExtractorConfigurationError,
    OpenAIConceptExtractor,
)


class MockCompletions:
    def __init__(self, parsed=None, *, refusal=None):
        self.calls = []
        self.parsed = parsed
        self.refusal = refusal

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self.parsed, refusal=self.refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FailingCompletions:
    def parse(self, **kwargs):
        raise OpenAIError("mock provider unavailable")


def mock_client(completions):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )


def passage() -> DocumentPassage:
    content = "Uses PGW for payment routing."
    return DocumentPassage(
        source_path="runbooks/payments.md",
        document_sha256=sha256(content.encode()).hexdigest(),
        start_line=8,
        end_line=8,
        text=content,
    )


def concept() -> Concept:
    return Concept(
        name="PGW",
        normalized_name="pgw",
        category=ConceptCategory.COMPANY_SPECIFIC,
        concept_type=ConceptType.ACRONYM,
        description="Internal payment routing component.",
        evidence=[
            SourcePassage(
                source_path="runbooks/payments.md",
                start_line=8,
                end_line=8,
                text="PGW",
            )
        ],
    )


def classification_concept(
    name: str,
    category: ConceptCategory,
    concept_type: ConceptType,
) -> Concept:
    return Concept(
        name=name,
        normalized_name=name.casefold(),
        category=category,
        concept_type=concept_type,
        evidence=[
            SourcePassage(
                source_path="incidents/INC-001.txt",
                start_line=1,
                end_line=1,
                text=name,
            )
        ],
    )


def test_uses_structured_pydantic_response_without_network():
    completions = MockCompletions(ConceptExtraction(concepts=[concept()]))
    extractor = OpenAIConceptExtractor(
        model="test-model", client=mock_client(completions), max_api_calls=2
    )

    result = extractor.extract(passage())

    assert result == [concept()]
    assert extractor.api_calls == 1
    request = completions.calls[0]
    assert request["model"] == "test-model"
    assert request["response_format"] is ConceptExtraction
    assert "temperature" not in request
    user_prompt = request["messages"][1]["content"]
    assert "runbooks/payments.md" in user_prompt
    assert "8: Uses PGW for payment routing." in user_prompt
    assert passage().document_sha256 not in user_prompt


def test_classification_keeps_standard_mechanisms_separate_from_internal_names():
    expected = [
        classification_concept(
            "Elasticsearch", ConceptCategory.GENERAL_TECHNICAL, ConceptType.APPLICATION
        ),
        classification_concept(
            "ILM", ConceptCategory.GENERAL_TECHNICAL, ConceptType.ACRONYM
        ),
        classification_concept(
            "check-rollover-ready",
            ConceptCategory.GENERAL_TECHNICAL,
            ConceptType.OPERATIONAL_CONCEPT,
        ),
        classification_concept(
            "is_write_index=true",
            ConceptCategory.GENERAL_TECHNICAL,
            ConceptType.CONFIGURATION,
        ),
        classification_concept(
            "payments-write",
            ConceptCategory.COMPANY_SPECIFIC,
            ConceptType.IDENTIFIER,
        ),
        classification_concept(
            "INC-001", ConceptCategory.COMPANY_SPECIFIC, ConceptType.INCIDENT
        ),
    ]
    completions = MockCompletions(ConceptExtraction(concepts=expected))
    extractor = OpenAIConceptExtractor(
        model="test-model", client=mock_client(completions)
    )

    result = extractor.extract(passage())

    assert [(item.name, item.category, item.concept_type) for item in result] == [
        (item.name, item.category, item.concept_type) for item in expected
    ]
    system_prompt = completions.calls[0]["messages"][0]["content"]
    assert "standard mechanism remains general" in system_prompt
    assert "Category and concept_type are independent dimensions" in system_prompt
    assert "merely because its type is" in system_prompt


def test_enforces_api_call_limit_before_sending_request():
    completions = MockCompletions(ConceptExtraction(concepts=[]))
    extractor = OpenAIConceptExtractor(
        model="test-model", client=mock_client(completions), max_api_calls=1
    )
    extractor.extract(passage())

    with pytest.raises(ApiCallLimitError, match="no request was sent"):
        extractor.extract(passage())

    assert len(completions.calls) == 1


def test_reports_refusal_explicitly():
    extractor = OpenAIConceptExtractor(
        model="test-model",
        client=mock_client(MockCompletions(refusal="Cannot process")),
    )

    with pytest.raises(ExtractionProviderError, match="Provider refused"):
        extractor.extract(passage())


def test_reports_missing_structured_output():
    extractor = OpenAIConceptExtractor(
        model="test-model", client=mock_client(MockCompletions())
    )

    with pytest.raises(ExtractionProviderError, match="no validated structured"):
        extractor.extract(passage())


def test_wraps_provider_errors_with_passage_context():
    extractor = OpenAIConceptExtractor(
        model="test-model", client=mock_client(FailingCompletions())
    )

    with pytest.raises(
        ExtractionProviderError,
        match=r"runbooks/payments.md lines 8-8: mock provider unavailable",
    ):
        extractor.extract(passage())


def test_reads_environment_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-value")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("KGA_MODEL", "environment-model")
    monkeypatch.setenv("KGA_API_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("KGA_API_MAX_RETRIES", "1")
    monkeypatch.setenv("KGA_MAX_API_CALLS", "7")

    extractor = OpenAIConceptExtractor.from_env(
        client=mock_client(MockCompletions(ConceptExtraction(concepts=[])))
    )

    assert extractor.model == "environment-model"
    assert extractor.max_api_calls == 7


def test_requires_model_configuration(monkeypatch):
    monkeypatch.delenv("KGA_MODEL", raising=False)

    with pytest.raises(ExtractorConfigurationError, match="model is required"):
        OpenAIConceptExtractor.from_env(client=mock_client(MockCompletions()))
