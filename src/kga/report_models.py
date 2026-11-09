"""Pydantic contracts for deterministic knowledge-gap reporting."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from kga.formal_models import RequirementType
from kga.models import (
    AnswerabilityStatus,
    ConceptCategory,
    ConceptType,
    KgaModel,
    QuestionDimension,
    QuestionEvaluation,
    SourcePassage,
)


class RequirementCoverageStatus(str, Enum):
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    UNANSWERED = "unanswered"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNCERTAIN = "uncertain"
    EVALUATION_ERROR = "evaluation_error"
    NOT_EVALUATED = "not_evaluated"


class GapType(str, Enum):
    EXPLICIT_GAP = "explicit_gap"
    PARTIAL_GAP = "partial_gap"
    CONFLICTING_KNOWLEDGE = "conflicting_knowledge"
    UNCERTAIN_COVERAGE = "uncertain_coverage"


class RequirementCoverage(KgaModel):
    requirement_id: str
    concept_id: str
    question_ids: list[str]
    dimension: QuestionDimension
    requirement_type: RequirementType
    description: str
    evaluation_status: RequirementCoverageStatus
    answer: str | None = None
    explanation: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    supporting_evidence: list[SourcePassage] = Field(default_factory=list)
    conflicting_evidence: list[SourcePassage] = Field(default_factory=list)
    evaluation_ids: list[str] = Field(default_factory=list)
    evaluation_run_id: str | None = None
    question_evaluations: list[QuestionEvaluation] = Field(default_factory=list)


class KnowledgeGap(KgaModel):
    gap_id: str = Field(pattern=r"^gap_[0-9a-f]{24}$")
    requirement_id: str
    concept_id: str
    concept_name: str
    concept_category: ConceptCategory
    concept_type: ConceptType
    dimension: QuestionDimension
    requirement_type: RequirementType
    question_ids: list[str]
    questions: list[str]
    gap_type: GapType
    description: str
    evaluation_status: AnswerabilityStatus
    evaluation_explanation: str
    supporting_evidence: list[SourcePassage] = Field(default_factory=list)
    conflicting_evidence: list[SourcePassage] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    recommended_documentation_action: str


class CoverageCounts(KgaModel):
    total_requirements: int = Field(ge=0)
    evaluated_requirements: int = Field(ge=0)
    not_evaluated_requirements: int = Field(ge=0)
    answered_requirements: int = Field(ge=0)
    partially_answered_requirements: int = Field(ge=0)
    unanswered_requirements: int = Field(ge=0)
    conflicting_evidence_requirements: int = Field(ge=0)
    uncertain_requirements: int = Field(ge=0)
    evaluation_error_requirements: int = Field(ge=0)
    evaluation_coverage_percent: float = Field(ge=0, le=100)
    answerability_coverage_percent: float = Field(ge=0, le=100)


class CoverageGroup(KgaModel):
    key: str
    label: str
    counts: CoverageCounts


class EvaluationAnomaly(KgaModel):
    requirement_id: str
    question_ids: list[str]
    anomaly_type: str
    explanation: str


class GapSummary(KgaModel):
    counts: CoverageCounts
    gap_counts_by_type: dict[GapType, int]
    by_concept: list[CoverageGroup]
    by_concept_category: list[CoverageGroup]
    by_concept_type: list[CoverageGroup]
    by_dimension: list[CoverageGroup]
    by_requirement_type: list[CoverageGroup]
    evaluation_anomalies: list[EvaluationAnomaly] = Field(default_factory=list)


class ReportManifest(KgaModel):
    report_schema_version: str
    formal_model_identity: str
    formal_model_schema_version: str
    evaluation_run_identity: str | None = None
    input_artifact_fingerprints: dict[str, str]
    input_question_set_identity: str
    requirement_count: int = Field(ge=0)
    evaluated_requirement_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    evaluation_anomaly_count: int = Field(ge=0)
    reporting_configuration: dict[str, str | bool | int | float | None]
    validation_warnings: list[str] = Field(default_factory=list)
    generated_at: datetime
