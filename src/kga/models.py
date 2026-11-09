"""Core data contracts for KGA concept discovery."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KgaModel(BaseModel):
    """Base model that rejects accidental or misspelled fields."""

    model_config = ConfigDict(extra="forbid")


class DocumentFormat(str, Enum):
    MARKDOWN = "markdown"
    TEXT = "text"
    ASCIIDOC = "asciidoc"


class CorpusDocument(KgaModel):
    """A UTF-8 document loaded from the corpus."""

    source_path: str = Field(
        min_length=1,
        description="POSIX-style path relative to the requested corpus root.",
    )
    format: DocumentFormat
    content: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DocumentPassage(KgaModel):
    """A deterministic, line-addressable portion of a corpus document."""

    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "DocumentPassage":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ConceptCategory(str, Enum):
    GENERAL_TECHNICAL = "general_technical"
    COMPANY_SPECIFIC = "company_specific"
    UNCERTAIN = "uncertain"


class ConceptType(str, Enum):
    APPLICATION = "application"
    IDENTIFIER = "identifier"
    ACRONYM = "acronym"
    INCIDENT = "incident"
    CONFIGURATION = "configuration"
    PROCEDURE = "procedure"
    OPERATIONAL_CONCEPT = "operational_concept"
    OTHER = "other"


class SourcePassage(KgaModel):
    """Verbatim evidence tying a concept to one corpus document."""

    source_path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "SourcePassage":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class Concept(KgaModel):
    """A normalized concept with corpus-backed supporting evidence."""

    name: str = Field(min_length=1)
    normalized_name: str = Field(min_length=1)
    category: ConceptCategory = Field(
        description=(
            "Organizational specificity, assessed independently of concept_type. "
            "Standard technologies and mechanisms are general_technical; internal "
            "implementations, names, identifiers, and practices are company_specific."
        )
    )
    concept_type: ConceptType = Field(
        description="The concept's functional type, independent of its category."
    )
    description: str | None = None
    evidence: list[SourcePassage] = Field(min_length=1)


class ConceptExtraction(KgaModel):
    """Structured response returned by a concept extraction provider."""

    concepts: list[Concept]


class CategoryStatistics(KgaModel):
    general_technical: int = Field(default=0, ge=0)
    company_specific: int = Field(default=0, ge=0)
    uncertain: int = Field(default=0, ge=0)


class ExecutionSummary(KgaModel):
    documents_processed: int = Field(ge=0)
    passages_processed: int = Field(default=0, ge=0)
    api_calls: int = Field(default=0, ge=0)
    concepts_discovered: int = Field(ge=0)
    concepts_by_category: CategoryStatistics


class ConceptReport(KgaModel):
    concepts: list[Concept]
    summary: ExecutionSummary


class QuestionDimension(str, Enum):
    IDENTIFICATION = "identification"
    PURPOSE = "purpose"
    RELATIONSHIPS = "relationships"
    OPERATIONS = "operations"
    TROUBLESHOOTING = "troubleshooting"
    IMPACT = "impact"
    CAUSE = "cause"
    DIAGNOSIS = "diagnosis"
    RESOLUTION = "resolution"


class KnowledgeQuestion(KgaModel):
    """A knowledge question traceable to an extracted concept and its evidence."""

    question_id: str = Field(pattern=r"^q_[0-9a-f]{16}$")
    question: str = Field(min_length=1)
    dimension: QuestionDimension
    concept_name: str = Field(min_length=1)
    concept_normalized_name: str = Field(min_length=1)
    concept_category: ConceptCategory
    concept_type: ConceptType
    evidence: list[SourcePassage] = Field(min_length=1)


class AnswerabilityStatus(str, Enum):
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    UNANSWERED = "unanswered"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNCERTAIN = "uncertain"
    EVALUATION_ERROR = "evaluation_error"


class EvaluationDecision(KgaModel):
    """Provider response before immutable question metadata is attached."""

    status: AnswerabilityStatus
    answer: str | None = None
    explanation: str = Field(min_length=1)
    supporting_evidence: list[SourcePassage] = Field(default_factory=list)
    conflicting_evidence: list[SourcePassage] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def evidence_must_match_status(self) -> "EvaluationDecision":
        if self.status in {
            AnswerabilityStatus.ANSWERED,
            AnswerabilityStatus.PARTIALLY_ANSWERED,
        } and not self.supporting_evidence:
            raise ValueError("answered decisions require supporting_evidence")
        if (
            self.status is AnswerabilityStatus.CONFLICTING_EVIDENCE
            and not self.conflicting_evidence
        ):
            raise ValueError("conflicting_evidence status requires conflicting_evidence")
        if self.status in {
            AnswerabilityStatus.UNANSWERED,
            AnswerabilityStatus.CONFLICTING_EVIDENCE,
            AnswerabilityStatus.UNCERTAIN,
            AnswerabilityStatus.EVALUATION_ERROR,
        } and self.answer is not None:
            raise ValueError(f"{self.status.value} decisions must not include an answer")
        return self


class EvidenceField(str, Enum):
    SUPPORTING_EVIDENCE = "supporting_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class EvidenceValidationFailureReason(str, Enum):
    UNKNOWN_SOURCE_PATH = "unknown_source_path"
    INVALID_LINE_RANGE = "invalid_line_range"
    QUOTATION_NOT_PRESENT = "quotation_not_present"


class EvidenceValidationDiagnostic(KgaModel):
    """Non-sensitive details about one rejected evidence citation."""

    question_id: str = Field(pattern=r"^q_[0-9a-f]{16}$")
    source_path: str = Field(min_length=1)
    declared_start_line: int = Field(ge=1)
    declared_end_line: int = Field(ge=1)
    failure_reason: EvidenceValidationFailureReason
    evidence_field: EvidenceField
    attempt_number: int = Field(ge=1)


class QuestionEvaluation(KgaModel):
    """Evidence-grounded answerability result for one generated question."""

    question_id: str = Field(pattern=r"^q_[0-9a-f]{16}$")
    question: str = Field(min_length=1)
    concept_name: str = Field(min_length=1)
    concept_normalized_name: str = Field(min_length=1)
    concept_category: ConceptCategory
    concept_type: ConceptType
    dimension: QuestionDimension
    status: AnswerabilityStatus
    answer: str | None = None
    explanation: str = Field(min_length=1)
    supporting_evidence: list[SourcePassage] = Field(default_factory=list)
    conflicting_evidence: list[SourcePassage] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    evaluation_attempts: int = Field(default=1, ge=1)
    retry_attempted: bool = False
    retry_succeeded: bool = False
    retry_skipped_reason: str | None = None
    evidence_validation_diagnostics: list[EvidenceValidationDiagnostic] = Field(
        default_factory=list
    )


class EvaluationSummary(KgaModel):
    questions_evaluated: int = Field(ge=0)
    counts_by_status: dict[AnswerabilityStatus, int]
    counts_by_knowledge_dimension: dict[QuestionDimension, int]
    counts_by_concept_category: dict[ConceptCategory, int]
    api_calls: int = Field(ge=0)
    evaluation_errors: int = Field(ge=0)
    total_evaluation_attempts: int = Field(default=0, ge=0)
    questions_requiring_retry: int = Field(default=0, ge=0)
    successful_retries: int = Field(default=0, ge=0)
    failed_retries: int = Field(default=0, ge=0)
    retries_skipped_due_to_api_budget: int = Field(default=0, ge=0)
    evidence_validation_failures: int = Field(default=0, ge=0)
