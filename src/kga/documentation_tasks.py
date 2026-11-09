"""Generate actionable documentation tasks from human-approved remediations."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from kga.formal_ids import stable_id
from kga.remediation_models import DocumentationTask, DocumentationTaskInventory, TaskManifest
from kga.review_models import ApprovedRemediationInventory, ReviewExportManifest


TASK_SCHEMA_VERSION = "0.2.5"
SUPPORTED_APPROVED_PLAN_SCHEMAS = {"0.2.4"}


class TaskValidationError(ValueError):
    pass


class ApprovedPlan:
    def __init__(self, inventory: ApprovedRemediationInventory,
                 manifest: ReviewExportManifest, fingerprints: dict[str, str], identity: str) -> None:
        self.inventory = inventory
        self.manifest = manifest
        self.fingerprints = fingerprints
        self.identity = identity


class TaskSet:
    def __init__(self, inventory: DocumentationTaskInventory, manifest: TaskManifest,
                 fingerprints: dict[str, str]) -> None:
        self.inventory = inventory
        self.manifest = manifest
        self.fingerprints = fingerprints


def load_task_set(directory: str | Path) -> TaskSet:
    root = Path(directory)
    required = ["documentation-tasks.json", "documentation-tasks.md", "task-manifest.json"]
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise TaskValidationError("; ".join(f"Missing task artifact: {item}" for item in missing))
    try:
        inventory = DocumentationTaskInventory.model_validate(json.loads((root / "documentation-tasks.json").read_text(encoding="utf-8")))
        manifest = TaskManifest.model_validate(json.loads((root / "task-manifest.json").read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise TaskValidationError(f"Invalid task artifact: {exc}") from exc
    if inventory.schema_version != TASK_SCHEMA_VERSION or manifest.task_schema_version != TASK_SCHEMA_VERSION:
        raise TaskValidationError("Unsupported or inconsistent task schema version")
    ids = [item.task_id for item in inventory.tasks]
    if len(ids) != len(set(ids)):
        raise TaskValidationError("Duplicate documentation task ID")
    if manifest.task_count != len(inventory.tasks):
        raise TaskValidationError("Task manifest count does not match task inventory")
    if manifest.task_set_identity != stable_id("taskset", sorted(ids)):
        raise TaskValidationError("Task set identity does not match task inventory")
    return TaskSet(inventory, manifest, {name: _fingerprint(root / name) for name in sorted(required)})


def load_approved_plan(directory: str | Path) -> ApprovedPlan:
    root = Path(directory)
    definitions: dict[str, Any] = {
        "approved-remediations.json": ApprovedRemediationInventory,
        "review-export-manifest.json": ReviewExportManifest,
    }
    required = list(definitions) + ["approved-remediations.md", "review-summary.json", "reviewed-equivalences.json"]
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise TaskValidationError("; ".join(f"Missing approved-plan artifact: {item}" for item in missing))
    try:
        values = {name: TypeAdapter(kind).validate_python(json.loads((root / name).read_text(encoding="utf-8"))) for name, kind in definitions.items()}
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise TaskValidationError(f"Invalid approved-plan artifact: {exc}") from exc
    inventory = values["approved-remediations.json"]
    manifest = values["review-export-manifest.json"]
    if inventory.export_schema_version not in SUPPORTED_APPROVED_PLAN_SCHEMAS or manifest.export_schema_version != inventory.export_schema_version:
        raise TaskValidationError("Unsupported or inconsistent approved-plan schema version")
    if manifest.approved_remediation_count != len(inventory.approved_remediations):
        raise TaskValidationError("Approved remediation count does not match export manifest")
    ids = [item.remediation_id for item in inventory.approved_remediations]
    if len(ids) != len(set(ids)):
        raise TaskValidationError("Duplicate approved remediation ID")
    fingerprints = {name: _fingerprint(root / name) for name in sorted(required)}
    identity = stable_id("aplan", {"schema_version": inventory.export_schema_version, "artifacts": fingerprints})
    return ApprovedPlan(inventory, manifest, fingerprints, identity)


def generate_documentation_tasks(plan: ApprovedPlan) -> DocumentationTaskInventory:
    tasks: list[DocumentationTask] = []
    for remediation in plan.inventory.approved_remediations:
        if not remediation.requirement_ids or not remediation.question_ids:
            raise TaskValidationError(f"Approved remediation {remediation.remediation_id} lacks target references")
        task_id = stable_id("task", {
            "remediation_id": remediation.remediation_id,
            "consolidated_gap_id": remediation.consolidated_gap_id,
            "requirement_ids": sorted(remediation.requirement_ids),
            "question_ids": sorted(remediation.question_ids),
        })
        criteria = [
            f"The documentation explicitly addresses: {question}"
            for question in remediation.information_to_collect
        ]
        criteria.extend(
            f"Re-evaluation preserves target question ID {question_id}."
            for question_id in sorted(remediation.question_ids)
        )
        criteria.append("Any factual statements are supported by the updated corpus and cite corpus-relative sources.")
        criteria.append("The existing target question IDs can be re-evaluated without regenerating questions.")
        tasks.append(DocumentationTask(
            task_id=task_id,
            remediation_id=remediation.remediation_id,
            consolidated_gap_id=remediation.consolidated_gap_id,
            title=f"Document remediation for {remediation.consolidated_gap_id}",
            description=remediation.original_recommendation,
            priority=remediation.approved_effective_priority,
            target_requirement_ids=sorted(remediation.requirement_ids),
            target_question_ids=sorted(remediation.question_ids),
            information_to_collect=list(remediation.information_to_collect),
            suggested_document_type=remediation.suggested_document_type,
            acceptance_criteria=criteria,
            original_recommendation=remediation.original_recommendation,
            human_reviewer=remediation.reviewer,
            approval_timestamp=remediation.approval_timestamp,
            human_review_rationale=remediation.review_comment,
            source_gap_ids=sorted(remediation.source_gap_ids),
            supporting_evidence=remediation.supporting_evidence,
            conflicting_evidence=remediation.conflicting_evidence,
        ))
    tasks.sort(key=lambda item: item.task_id)
    if len({item.task_id for item in tasks}) != len(tasks):
        raise TaskValidationError("Duplicate documentation task ID")
    return DocumentationTaskInventory(schema_version=TASK_SCHEMA_VERSION, tasks=tasks)


def export_documentation_tasks(output_directory: str | Path, inventory: DocumentationTaskInventory,
                               plan: ApprovedPlan) -> TaskManifest:
    identity = stable_id("taskset", [item.task_id for item in inventory.tasks])
    manifest = TaskManifest(
        task_schema_version=TASK_SCHEMA_VERSION,
        task_set_identity=identity,
        approved_plan_identity=plan.identity,
        input_artifact_fingerprints=plan.fingerprints,
        task_count=len(inventory.tasks),
        generated_at=datetime.now(timezone.utc),
    )
    lines = ["# Documentation Tasks", "", f"Tasks: {len(inventory.tasks)}", ""]
    if not inventory.tasks:
        lines.extend(["No human-approved remediations are available for execution.", ""])
    for task in inventory.tasks:
        lines.extend([
            f"## {task.task_id}", "", f"- Priority: `{task.priority.value}`",
            f"- Remediation: `{task.remediation_id}`", f"- Gap: `{task.consolidated_gap_id}`",
            f"- Recommendation: {task.original_recommendation}",
            f"- Reviewer: {task.human_reviewer} at {task.approval_timestamp.isoformat()}",
            f"- Review rationale: {task.human_review_rationale}",
            "- Acceptance criteria:",
            *[f"  - {item}" for item in task.acceptance_criteria], "",
        ])
    _atomic_files(Path(output_directory), {
        "documentation-tasks.json": _json_bytes(inventory),
        "documentation-tasks.md": "\n".join(lines).encode("utf-8"),
        "task-manifest.json": _json_bytes(manifest),
    })
    return manifest


def _atomic_files(output: Path, payloads: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        for name, payload in payloads.items():
            temp = output / f".{name}.tmp"
            temp.write_bytes(payload)
            temporary.append((temp, output / name))
        for temp, target in temporary:
            os.replace(temp, target)
    finally:
        for temp, _ in temporary:
            if temp.exists():
                temp.unlink()


def _fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
