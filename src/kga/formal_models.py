"""Formal domain entities layered over KGA's operational models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from kga.models import (
    ConceptCategory,
    ConceptType,
    DocumentFormat,
    KgaModel,
    KnowledgeQuestion,
    QuestionDimension,
    QuestionEvaluation,
    SourcePassage,
)


class RequirementType(str, Enum):
    IDENTIFICATION = "identification"
    PURPOSE = "purpose"
    OWNERSHIP = "ownership"
    DEPENDENCIES = "dependencies"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    TROUBLESHOOTING = "troubleshooting"
    INCIDENT_IMPACT = "incident_impact"
    INCIDENT_CAUSE = "incident_cause"
    INCIDENT_DIAGNOSIS = "incident_diagnosis"
    INCIDENT_RESOLUTION = "incident_resolution"
    CONFIGURATION_VALUES = "configuration_values"
    CONFIGURATION_CHANGE = "configuration_change"
    PROCEDURE_STEPS = "procedure_steps"
    PROCEDURE_PREREQUISITES = "procedure_prerequisites"
    RESOLUTION = "resolution"
    SAFETY_RATIONALE = "safety_rationale"
    PROCEDURE_RATIONALE = "procedure_rationale"
    OTHER = "other"


class FormalDocument(KgaModel):
    """Document identity derived from corpus-relative path and content hash."""

    document_id: str = Field(pattern=r"^doc_[0-9a-f]{24}$")
    source_path: str = Field(min_length=1)
    format: DocumentFormat
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FormalEvidence(KgaModel):
    """Stable evidence identity retaining exact source provenance."""

    evidence_id: str = Field(pattern=r"^ev_[0-9a-f]{24}$")
    document_id: str | None = Field(default=None, pattern=r"^doc_[0-9a-f]{24}$")
    source_path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def line_range_must_be_ordered(self) -> "FormalEvidence":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class FormalConcept(KgaModel):
    """A concept with a stable identity and explicit evidence references."""

    concept_id: str = Field(pattern=r"^concept_[0-9a-f]{24}$")
    name: str = Field(min_length=1)
    normalized_name: str = Field(min_length=1)
    category: ConceptCategory
    concept_type: ConceptType
    description: str | None = None
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_ids_must_be_unique(self) -> "FormalConcept":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must be unique")
        return self


class KnowledgeRequirement(KgaModel):
    """One independently assessable statement of what should be known."""

    requirement_id: str = Field(pattern=r"^kr_[0-9a-f]{24}$")
    concept_id: str = Field(pattern=r"^concept_[0-9a-f]{24}$")
    dimension: QuestionDimension
    requirement_type: RequirementType
    description: str = Field(min_length=1)
    question_ids: list[str] = Field(min_length=1)
    source_evidence: list[SourcePassage] = Field(min_length=1)

    @model_validator(mode="after")
    def references_must_be_unique(self) -> "KnowledgeRequirement":
        if len(set(self.question_ids)) != len(self.question_ids):
            raise ValueError("question_ids must be unique")
        evidence_keys = {
            (item.source_path, item.start_line, item.end_line, item.text)
            for item in self.source_evidence
        }
        if len(evidence_keys) != len(self.source_evidence):
            raise ValueError("source_evidence must not contain duplicates")
        return self


class FormalQuestion(KnowledgeQuestion):
    """A legacy-compatible question enriched with formal references."""

    concept_id: str = Field(pattern=r"^concept_[0-9a-f]{24}$")
    requirement_id: str = Field(pattern=r"^kr_[0-9a-f]{24}$")


class FormalEvaluation(KgaModel):
    """An answerability evaluation linked to an explicit execution run."""

    evaluation_id: str = Field(pattern=r"^eval_[0-9a-f]{24}$")
    evaluation_run_id: str = Field(pattern=r"^run_[0-9a-f]{24}$")
    result: QuestionEvaluation


class EvaluationRun(KgaModel):
    """Provenance for one evaluation execution over immutable snapshots."""

    evaluation_run_id: str = Field(pattern=r"^run_[0-9a-f]{24}$")
    corpus_snapshot_id: str = Field(pattern=r"^corpus_[0-9a-f]{24}$")
    question_set_id: str = Field(pattern=r"^qset_[0-9a-f]{24}$")
    model: str | None = None
    evaluator_version: str | None = None
    prompt_version: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    questions_evaluated: int = Field(ge=0)
    api_calls: int = Field(ge=0)
    evaluation_errors: int = Field(ge=0)
    configuration: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def timestamps_must_be_consistent(self) -> "EvaluationRun":
        for value in (self.started_at, self.completed_at):
            if value is not None and value.tzinfo is None:
                raise ValueError("evaluation run timestamps must include a timezone")
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("completed_at must not precede started_at")
        return self


class FormalModelManifest(KgaModel):
    """Deterministic manifest for a formalized concept/question inventory."""

    model_schema_version: str
    concept_inventory_id: str = Field(pattern=r"^cset_[0-9a-f]{24}$")
    question_set_id: str = Field(pattern=r"^qset_[0-9a-f]{24}$")
    concept_count: int = Field(ge=0)
    question_count: int = Field(ge=0)
    requirement_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    validation_passed: bool
    validation_errors: list[str] = Field(default_factory=list)


class FormalKnowledgeModel(KgaModel):
    """In-memory formal projection used for integrity validation and export."""

    concepts: list[FormalConcept]
    evidence: list[FormalEvidence]
    questions: list[FormalQuestion]
    requirements: list[KnowledgeRequirement]
    manifest: FormalModelManifest
