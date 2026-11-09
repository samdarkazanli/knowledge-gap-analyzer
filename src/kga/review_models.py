"""Versioned human-review and approved-export contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, field_validator, model_validator

from kga.consolidation_models import ProvisionalPriority
from kga.models import KgaModel, SourcePassage


class GapReviewDecision(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class RemediationReviewDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class ConsolidationReviewDecision(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED_EQUIVALENT = "confirmed_equivalent"
    REJECTED_EQUIVALENCE = "rejected_equivalence"


class HumanDecision(KgaModel):
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    comment: str | None = None

    @field_validator("reviewer", "comment")
    @classmethod
    def nonempty_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be nonempty when provided")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def timestamp_is_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone")
        return value

    def require_human_metadata(self) -> None:
        if not self.reviewer or not self.reviewed_at or not self.comment:
            raise ValueError(
                "non-pending decisions require reviewer, timezone-aware reviewed_at, "
                "and a nonempty comment"
            )


class GapReview(HumanDecision):
    gap_id: str = Field(pattern=r"^cgap_[0-9a-f]{24}$")
    decision: GapReviewDecision = GapReviewDecision.PENDING

    @model_validator(mode="after")
    def validate_decision_metadata(self) -> "GapReview":
        if self.decision is not GapReviewDecision.PENDING:
            self.require_human_metadata()
        return self


class RemediationReview(HumanDecision):
    remediation_id: str = Field(pattern=r"^rem_[0-9a-f]{24}$")
    decision: RemediationReviewDecision = RemediationReviewDecision.PENDING
    priority_override: ProvisionalPriority | None = None

    @model_validator(mode="after")
    def validate_decision_metadata(self) -> "RemediationReview":
        if self.decision is not RemediationReviewDecision.PENDING:
            self.require_human_metadata()
        if self.priority_override is not None and self.decision is not RemediationReviewDecision.APPROVED:
            raise ValueError("priority_override is allowed only for approved remediation")
        return self


class ConsolidationReview(HumanDecision):
    review_id: str = Field(pattern=r"^creview_[0-9a-f]{24}$")
    gap_ids: list[str] = Field(min_length=2)
    decision: ConsolidationReviewDecision = ConsolidationReviewDecision.PROPOSED

    @model_validator(mode="after")
    def validate_equivalence(self) -> "ConsolidationReview":
        if len(set(self.gap_ids)) != len(self.gap_ids):
            raise ValueError("gap_ids must contain at least two distinct gap IDs")
        if self.decision is not ConsolidationReviewDecision.PROPOSED:
            self.require_human_metadata()
        return self


class GapReviewInventory(KgaModel):
    schema_version: str
    reviews: list[GapReview]


class RemediationReviewInventory(KgaModel):
    schema_version: str
    reviews: list[RemediationReview]


class ConsolidationReviewInventory(KgaModel):
    schema_version: str
    reviews: list[ConsolidationReview]


class ReviewManifest(KgaModel):
    review_schema_version: str
    workspace_identity: str = Field(pattern=r"^review_[0-9a-f]{24}$")
    input_consolidation_identity: str = Field(pattern=r"^csnapshot_[0-9a-f]{24}$")
    input_consolidation_schema_version: str
    input_artifact_fingerprints: dict[str, str]
    gap_review_count: int = Field(ge=0)
    remediation_review_count: int = Field(ge=0)
    consolidation_review_count: int = Field(ge=0)
    initialized_at: datetime


class ReviewSummary(KgaModel):
    review_schema_version: str
    validation_passed: bool
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    gap_decision_counts: dict[GapReviewDecision, int]
    remediation_decision_counts: dict[RemediationReviewDecision, int]
    consolidation_decision_counts: dict[ConsolidationReviewDecision, int]
    approved_remediation_count: int = Field(ge=0)


class ApprovedRemediation(KgaModel):
    remediation_id: str
    consolidated_gap_id: str
    original_recommendation: str
    suggested_document_type: str
    information_to_collect: list[str]
    original_provisional_priority: ProvisionalPriority
    approved_effective_priority: ProvisionalPriority
    reviewer: str
    approval_timestamp: datetime
    review_comment: str
    source_gap_ids: list[str]
    question_ids: list[str]
    requirement_ids: list[str]
    supporting_evidence: list[SourcePassage]
    conflicting_evidence: list[SourcePassage]


class ApprovedRemediationInventory(KgaModel):
    export_schema_version: str
    approved_remediations: list[ApprovedRemediation]


class ReviewedEquivalence(KgaModel):
    review_id: str
    gap_ids: list[str]
    reviewer: str
    confirmed_at: datetime
    rationale: str


class ReviewedEquivalenceInventory(KgaModel):
    export_schema_version: str
    confirmed_equivalences: list[ReviewedEquivalence]


class ReviewExportManifest(KgaModel):
    input_consolidation_identity: str
    input_artifact_fingerprints: dict[str, str]
    review_workspace_identity: str
    review_file_fingerprints: dict[str, str]
    review_schema_version: str
    export_schema_version: str
    approved_remediation_count: int = Field(ge=0)
    confirmed_gap_count: int = Field(ge=0)
    rejected_gap_count: int = Field(ge=0)
    deferred_gap_count: int = Field(ge=0)
    pending_gap_count: int = Field(ge=0)
    confirmed_equivalence_count: int = Field(ge=0)
    validation_warnings: list[str] = Field(default_factory=list)
    generated_at: datetime
