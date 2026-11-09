"""Versioned contracts for deterministic gap consolidation and remediation."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from kga.formal_models import RequirementType
from kga.models import AnswerabilityStatus, KgaModel, QuestionDimension, SourcePassage
from kga.report_models import GapType


class ConsolidationMethod(str, Enum):
    SINGLETON = "singleton"
    SAME_FORMAL_REQUIREMENT = "same_formal_requirement"


class ProvisionalPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConsolidatedGap(KgaModel):
    consolidated_gap_id: str = Field(pattern=r"^cgap_[0-9a-f]{24}$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    knowledge_dimension: QuestionDimension
    requirement_type: RequirementType
    gap_status: GapType
    concept_ids: list[str] = Field(min_length=1)
    concept_names: list[str] = Field(min_length=1)
    requirement_ids: list[str] = Field(min_length=1)
    question_ids: list[str] = Field(min_length=1)
    questions: list[str] = Field(min_length=1)
    source_gap_ids: list[str] = Field(min_length=1)
    evaluation_statuses: list[AnswerabilityStatus] = Field(min_length=1)
    evaluation_explanations: list[str] = Field(min_length=1)
    evaluation_ids: list[str] = Field(default_factory=list)
    evaluation_run_ids: list[str] = Field(default_factory=list)
    supporting_evidence: list[SourcePassage] = Field(default_factory=list)
    conflicting_evidence: list[SourcePassage] = Field(default_factory=list)
    missing_information: list[str] = Field(min_length=1)
    consolidation_method: ConsolidationMethod
    requires_human_validation: bool

    @model_validator(mode="after")
    def references_are_unique(self) -> "ConsolidatedGap":
        for name in (
            "concept_ids", "concept_names", "requirement_ids", "question_ids",
            "questions", "source_gap_ids", "evaluation_statuses",
            "evaluation_explanations", "evaluation_ids", "evaluation_run_ids",
            "missing_information",
        ):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        return self


class ConsolidatedGapInventory(KgaModel):
    schema_version: str
    consolidated_gaps: list[ConsolidatedGap]


class RemediationProposal(KgaModel):
    remediation_id: str = Field(pattern=r"^rem_[0-9a-f]{24}$")
    consolidated_gap_id: str = Field(pattern=r"^cgap_[0-9a-f]{24}$")
    recommended_action: str = Field(min_length=1)
    suggested_document_type: str = Field(min_length=1)
    information_to_collect: list[str] = Field(min_length=1)
    provisional_priority: ProvisionalPriority
    priority_rationale: str = Field(min_length=1)
    requires_human_validation: bool


class RemediationPlan(KgaModel):
    schema_version: str
    remediations: list[RemediationProposal]


class ConsolidationSummary(KgaModel):
    schema_version: str
    input_report_identity: str
    input_report_schema_version: str
    input_artifact_fingerprints: dict[str, str]
    original_gap_count: int = Field(ge=0)
    consolidated_gap_count: int = Field(ge=0)
    merged_gap_count: int = Field(ge=0)
    singleton_gap_count: int = Field(ge=0)
    remediation_count: int = Field(ge=0)
    priority_distribution: dict[ProvisionalPriority, int]
    consolidation_method_distribution: dict[ConsolidationMethod, int]
    validation_warnings: list[str] = Field(default_factory=list)


def _default_priorities() -> dict[GapType, ProvisionalPriority]:
    return {
        GapType.EXPLICIT_GAP: ProvisionalPriority.MEDIUM,
        GapType.PARTIAL_GAP: ProvisionalPriority.LOW,
        GapType.CONFLICTING_KNOWLEDGE: ProvisionalPriority.HIGH,
        GapType.UNCERTAIN_COVERAGE: ProvisionalPriority.MEDIUM,
    }


class ConsolidationConfig(KgaModel):
    """Transparent policy inputs; never inferred from concept names."""

    priority_by_gap_type: dict[GapType, ProvisionalPriority] = Field(
        default_factory=_default_priorities
    )

    @model_validator(mode="after")
    def all_gap_types_have_priorities(self) -> "ConsolidationConfig":
        missing = set(GapType) - set(self.priority_by_gap_type)
        if missing:
            raise ValueError(
                "priority_by_gap_type is missing: "
                + ", ".join(sorted(item.value for item in missing))
            )
        return self
