"""Documentation execution and verification contracts for KGA V0.2.5."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath

from pydantic import Field, field_validator, model_validator

from kga.consolidation_models import ProvisionalPriority
from kga.models import AnswerabilityStatus, KgaModel, SourcePassage


class DocumentationTaskStatus(str, Enum):
    OPEN = "open"


class DocumentationTask(KgaModel):
    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    remediation_id: str = Field(pattern=r"^rem_[0-9a-f]{24}$")
    consolidated_gap_id: str = Field(pattern=r"^cgap_[0-9a-f]{24}$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    priority: ProvisionalPriority
    target_requirement_ids: list[str] = Field(min_length=1)
    target_question_ids: list[str] = Field(min_length=1)
    information_to_collect: list[str] = Field(min_length=1)
    suggested_document_type: str = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1)
    status: DocumentationTaskStatus = DocumentationTaskStatus.OPEN
    original_recommendation: str = Field(min_length=1)
    human_reviewer: str = Field(min_length=1)
    approval_timestamp: datetime
    human_review_rationale: str = Field(min_length=1)
    source_gap_ids: list[str] = Field(min_length=1)
    supporting_evidence: list[SourcePassage] = Field(default_factory=list)
    conflicting_evidence: list[SourcePassage] = Field(default_factory=list)

    @field_validator("approval_timestamp")
    @classmethod
    def approval_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("approval_timestamp must include a timezone")
        return value


class DocumentationTaskInventory(KgaModel):
    schema_version: str
    tasks: list[DocumentationTask]


class TaskManifest(KgaModel):
    task_schema_version: str
    task_set_identity: str = Field(pattern=r"^taskset_[0-9a-f]{24}$")
    approved_plan_identity: str = Field(pattern=r"^aplan_[0-9a-f]{24}$")
    input_artifact_fingerprints: dict[str, str]
    task_count: int = Field(ge=0)
    generated_at: datetime


class DocumentationChangeStatus(str, Enum):
    PENDING = "pending"
    DECLARED = "declared"


class ChangedFileDeclaration(KgaModel):
    source_path: str = Field(min_length=1)

    @field_validator("source_path")
    @classmethod
    def path_is_corpus_relative(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("source_path must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("source_path must be a safe corpus-relative path")
        return value


class DocumentationChange(KgaModel):
    change_id: str = Field(pattern=r"^change_[0-9a-f]{24}$")
    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    changed_files: list[ChangedFileDeclaration] = Field(default_factory=list)
    contributor: str | None = None
    recorded_at: datetime | None = None
    description: str | None = None
    status: DocumentationChangeStatus = DocumentationChangeStatus.PENDING

    @field_validator("contributor", "description")
    @classmethod
    def optional_text_is_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be nonempty when provided")
        return value

    @field_validator("recorded_at")
    @classmethod
    def recorded_at_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("recorded_at must include a timezone")
        return value

    @model_validator(mode="after")
    def declared_change_has_metadata(self) -> "DocumentationChange":
        if self.status is DocumentationChangeStatus.DECLARED:
            if not self.changed_files or not self.contributor or not self.recorded_at or not self.description:
                raise ValueError("declared changes require files, contributor, recorded_at, and description")
        if len({item.source_path for item in self.changed_files}) != len(self.changed_files):
            raise ValueError("changed_files must be unique")
        return self


class DocumentationChangeInventory(KgaModel):
    schema_version: str
    changes: list[DocumentationChange]


class ChangeManifest(KgaModel):
    change_schema_version: str
    change_set_identity: str = Field(pattern=r"^changeset_[0-9a-f]{24}$")
    task_set_identity: str = Field(pattern=r"^taskset_[0-9a-f]{24}$")
    task_artifact_fingerprints: dict[str, str]
    change_count: int = Field(ge=0)
    initialized_at: datetime


class VerifiedChangedFile(KgaModel):
    source_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VerifiedDocumentationChange(KgaModel):
    change_id: str
    task_id: str
    declared_status: DocumentationChangeStatus
    files_verified: bool
    changed_files: list[VerifiedChangedFile]


class ChangeValidationResult(KgaModel):
    schema_version: str
    validation_passed: bool
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    corpus_snapshot_id: str | None = None
    verified_changes: list[VerifiedDocumentationChange] = Field(default_factory=list)


class RemediationOutcomeStatus(str, Enum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"


class QuestionStatusComparison(KgaModel):
    question_id: str
    baseline_status: AnswerabilityStatus | None
    updated_status: AnswerabilityStatus | None
    baseline_answer: str | None = None
    updated_answer: str | None = None
    baseline_explanation: str | None = None
    updated_explanation: str | None = None
    baseline_evidence: list[SourcePassage] = Field(default_factory=list)
    updated_evidence: list[SourcePassage] = Field(default_factory=list)


class RemediationOutcome(KgaModel):
    outcome_id: str = Field(pattern=r"^outcome_[0-9a-f]{24}$")
    verification_run_id: str = Field(pattern=r"^verify_[0-9a-f]{24}$")
    task_id: str
    remediation_id: str
    consolidated_gap_id: str
    target_requirement_ids: list[str]
    target_question_ids: list[str]
    comparisons: list[QuestionStatusComparison]
    outcome: RemediationOutcomeStatus
    rationale: str
    human_reviewer: str
    approval_timestamp: datetime
    human_review_rationale: str
    baseline_corpus_snapshot_id: str
    updated_corpus_snapshot_id: str


class RemediationOutcomeInventory(KgaModel):
    schema_version: str
    outcomes: list[RemediationOutcome]


class VerificationRun(KgaModel):
    verification_run_id: str = Field(pattern=r"^verify_[0-9a-f]{24}$")
    task_set_identity: str
    change_set_identity: str
    baseline_corpus_snapshot_id: str
    updated_corpus_snapshot_id: str
    baseline_evaluation_run_id: str | None = None
    updated_evaluation_run_id: str | None = None
    question_set_id: str | None = None
    input_artifact_fingerprints: dict[str, str]
    provenance_warnings: list[str] = Field(default_factory=list)
    generated_at: datetime


class VerificationSummary(KgaModel):
    schema_version: str
    task_count: int = Field(ge=0)
    outcome_counts: dict[RemediationOutcomeStatus, int]
    validation_warnings: list[str] = Field(default_factory=list)


class VerificationManifest(KgaModel):
    verification_schema_version: str
    verification_run: VerificationRun
    outcome_count: int = Field(ge=0)
