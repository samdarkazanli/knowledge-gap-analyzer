"""Offline initialization, validation, and export of human review decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from pydantic import TypeAdapter, ValidationError

from kga.consolidation_models import (
    ConsolidatedGapInventory,
    ConsolidationSummary,
    RemediationPlan,
)
from kga.formal_ids import stable_id
from kga.review_models import (
    ApprovedRemediation,
    ApprovedRemediationInventory,
    ConsolidationReviewDecision,
    ConsolidationReviewInventory,
    GapReview,
    GapReviewDecision,
    GapReviewInventory,
    RemediationReview,
    RemediationReviewDecision,
    RemediationReviewInventory,
    ReviewExportManifest,
    ReviewManifest,
    ReviewedEquivalence,
    ReviewedEquivalenceInventory,
    ReviewSummary,
)

REVIEW_SCHEMA_VERSION = "0.2.4"
EXPORT_SCHEMA_VERSION = "0.2.4"
SUPPORTED_CONSOLIDATION_SCHEMAS = {"0.2.3"}


class ReviewValidationError(ValueError):
    def __init__(self, diagnostics: Sequence[str]) -> None:
        self.diagnostics = list(diagnostics)
        super().__init__("; ".join(self.diagnostics))


class ConsolidationSnapshot:
    def __init__(self, inventory: ConsolidatedGapInventory, plan: RemediationPlan,
                 summary: ConsolidationSummary, fingerprints: dict[str, str], identity: str) -> None:
        self.inventory = inventory
        self.plan = plan
        self.summary = summary
        self.fingerprints = fingerprints
        self.identity = identity


class ReviewWorkspace:
    def __init__(self, manifest: ReviewManifest, gaps: GapReviewInventory,
                 remediations: RemediationReviewInventory,
                 consolidations: ConsolidationReviewInventory,
                 fingerprints: dict[str, str]) -> None:
        self.manifest = manifest
        self.gaps = gaps
        self.remediations = remediations
        self.consolidations = consolidations
        self.fingerprints = fingerprints


def load_consolidation_snapshot(directory: str | Path) -> ConsolidationSnapshot:
    root = Path(directory)
    definitions: dict[str, Any] = {
        "consolidated-gaps.json": ConsolidatedGapInventory,
        "remediation-plan.json": RemediationPlan,
        "consolidation-summary.json": ConsolidationSummary,
    }
    required = list(definitions) + ["remediation-plan.md"]
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise ReviewValidationError([f"Missing consolidation artifact: {item}" for item in missing])
    try:
        values = {name: TypeAdapter(kind).validate_python(_read_json(root / name)) for name, kind in definitions.items()}
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ReviewValidationError([f"Invalid consolidation artifact: {exc}"]) from exc
    inventory = values["consolidated-gaps.json"]
    plan = values["remediation-plan.json"]
    summary = values["consolidation-summary.json"]
    errors: list[str] = []
    schemas = {inventory.schema_version, plan.schema_version, summary.schema_version}
    if len(schemas) != 1 or next(iter(schemas)) not in SUPPORTED_CONSOLIDATION_SCHEMAS:
        errors.append(f"Unsupported or inconsistent consolidation schemas: {sorted(schemas)}")
    gap_ids = _unique_ids("consolidated gap", [item.consolidated_gap_id for item in inventory.consolidated_gaps], errors)
    _unique_ids("remediation", [item.remediation_id for item in plan.remediations], errors)
    for item in plan.remediations:
        if item.consolidated_gap_id not in gap_ids:
            errors.append(f"Remediation {item.remediation_id} references missing gap {item.consolidated_gap_id}")
    if summary.consolidated_gap_count != len(inventory.consolidated_gaps):
        errors.append("Consolidation summary gap count does not match inventory")
    if summary.remediation_count != len(plan.remediations):
        errors.append("Consolidation summary remediation count does not match plan")
    if errors:
        raise ReviewValidationError(errors)
    fingerprints = {name: _fingerprint(root / name) for name in sorted(required)}
    identity = stable_id("csnapshot", {"schema_version": inventory.schema_version, "artifacts": fingerprints})
    return ConsolidationSnapshot(inventory, plan, summary, fingerprints, identity)


def initialize_review_workspace(snapshot: ConsolidationSnapshot, output_directory: str | Path) -> ReviewManifest:
    output = Path(output_directory)
    if output.exists():
        raise ReviewValidationError([f"Review workspace already exists; refusing to overwrite: {output}"])
    gaps = GapReviewInventory(schema_version=REVIEW_SCHEMA_VERSION, reviews=[
        GapReview(gap_id=item.consolidated_gap_id)
        for item in sorted(snapshot.inventory.consolidated_gaps, key=lambda value: value.consolidated_gap_id)
    ])
    remediations = RemediationReviewInventory(schema_version=REVIEW_SCHEMA_VERSION, reviews=[
        RemediationReview(remediation_id=item.remediation_id)
        for item in sorted(snapshot.plan.remediations, key=lambda value: value.remediation_id)
    ])
    consolidations = ConsolidationReviewInventory(schema_version=REVIEW_SCHEMA_VERSION, reviews=[])
    workspace_identity = stable_id("review", {
        "input_consolidation_identity": snapshot.identity,
        "gap_ids": [item.gap_id for item in gaps.reviews],
        "remediation_ids": [item.remediation_id for item in remediations.reviews],
    })
    manifest = ReviewManifest(
        review_schema_version=REVIEW_SCHEMA_VERSION,
        workspace_identity=workspace_identity,
        input_consolidation_identity=snapshot.identity,
        input_consolidation_schema_version=snapshot.inventory.schema_version,
        input_artifact_fingerprints=snapshot.fingerprints,
        gap_review_count=len(gaps.reviews),
        remediation_review_count=len(remediations.reviews),
        consolidation_review_count=0,
        initialized_at=datetime.now(timezone.utc),
    )
    _create_directory_atomically(output, {
        "review-manifest.json": _json_bytes(manifest),
        "gap-reviews.json": _json_bytes(gaps),
        "remediation-reviews.json": _json_bytes(remediations),
        "consolidation-reviews.json": _json_bytes(consolidations),
        "review-guide.md": _review_guide().encode("utf-8"),
    })
    return manifest


def load_review_workspace(directory: str | Path) -> ReviewWorkspace:
    root = Path(directory)
    definitions: dict[str, Any] = {
        "review-manifest.json": ReviewManifest,
        "gap-reviews.json": GapReviewInventory,
        "remediation-reviews.json": RemediationReviewInventory,
        "consolidation-reviews.json": ConsolidationReviewInventory,
    }
    missing = [str(root / name) for name in definitions if not (root / name).is_file()]
    if missing:
        raise ReviewValidationError([f"Missing review artifact: {item}" for item in missing])
    try:
        values = {name: TypeAdapter(kind).validate_python(_read_json(root / name)) for name, kind in definitions.items()}
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ReviewValidationError([f"Invalid review artifact: {exc}"]) from exc
    fingerprints = {name: _fingerprint(root / name) for name in sorted(definitions)}
    return ReviewWorkspace(values["review-manifest.json"], values["gap-reviews.json"],
                           values["remediation-reviews.json"], values["consolidation-reviews.json"], fingerprints)


def validate_review(snapshot: ConsolidationSnapshot, workspace: ReviewWorkspace) -> ReviewSummary:
    errors: list[str] = []
    warnings: list[str] = list(snapshot.summary.validation_warnings)
    manifest = workspace.manifest
    versions = {manifest.review_schema_version, workspace.gaps.schema_version,
                workspace.remediations.schema_version, workspace.consolidations.schema_version}
    if versions != {REVIEW_SCHEMA_VERSION}:
        errors.append(f"Unsupported or inconsistent review schema versions: {sorted(versions)}")
    if manifest.input_consolidation_identity != snapshot.identity:
        errors.append("Review workspace consolidation snapshot identity does not match input")
    if manifest.input_artifact_fingerprints != snapshot.fingerprints:
        errors.append("Review workspace consolidation artifact fingerprints do not match input")
    if manifest.input_consolidation_schema_version != snapshot.inventory.schema_version:
        errors.append("Review workspace consolidation schema version does not match input")
    expected_workspace_id = stable_id("review", {
        "input_consolidation_identity": manifest.input_consolidation_identity,
        "gap_ids": sorted(item.gap_id for item in workspace.gaps.reviews),
        "remediation_ids": sorted(item.remediation_id for item in workspace.remediations.reviews),
    })
    if manifest.workspace_identity != expected_workspace_id:
        errors.append("Review workspace identity does not match its bound record identifiers")

    gap_ids = {item.consolidated_gap_id for item in snapshot.inventory.consolidated_gaps}
    remediation_map = {item.remediation_id: item for item in snapshot.plan.remediations}
    gap_reviews = _review_map("gap review", workspace.gaps.reviews, "gap_id", errors)
    remediation_reviews = _review_map("remediation review", workspace.remediations.reviews, "remediation_id", errors)
    _exact_coverage("gap reviews", gap_ids, set(gap_reviews), errors)
    _exact_coverage("remediation reviews", set(remediation_map), set(remediation_reviews), errors)
    if manifest.gap_review_count != len(workspace.gaps.reviews):
        errors.append("Review manifest gap count does not match gap-reviews.json")
    if manifest.remediation_review_count != len(workspace.remediations.reviews):
        errors.append("Review manifest remediation count does not match remediation-reviews.json")
    if manifest.consolidation_review_count != len(workspace.consolidations.reviews):
        errors.append("Review manifest consolidation count does not match consolidation-reviews.json")
    for remediation_id, review in remediation_reviews.items():
        proposal = remediation_map.get(remediation_id)
        if proposal is None:
            continue
        gap_review = gap_reviews.get(proposal.consolidated_gap_id)
        if review.decision is RemediationReviewDecision.APPROVED and (
            gap_review is None or gap_review.decision is not GapReviewDecision.CONFIRMED
        ):
            errors.append(f"Approved remediation {remediation_id} requires confirmed gap {proposal.consolidated_gap_id}")

    consolidation_reviews = _review_map("consolidation review", workspace.consolidations.reviews, "review_id", errors)
    active: dict[str, list[str]] = defaultdict(list)
    for review in consolidation_reviews.values():
        unknown = sorted(set(review.gap_ids) - gap_ids)
        if unknown:
            errors.append(f"Consolidation review {review.review_id} references unknown gaps: {', '.join(unknown)}")
        if review.decision is not ConsolidationReviewDecision.REJECTED_EQUIVALENCE:
            for gap_id in review.gap_ids:
                active[gap_id].append(review.review_id)
        if review.decision is ConsolidationReviewDecision.CONFIRMED_EQUIVALENT:
            invalid = [gap_id for gap_id in review.gap_ids if gap_reviews.get(gap_id) is None or gap_reviews[gap_id].decision is not GapReviewDecision.CONFIRMED]
            if invalid:
                errors.append(f"Confirmed equivalence {review.review_id} requires confirmed member gaps: {', '.join(sorted(invalid))}")
    for gap_id, review_ids in sorted(active.items()):
        if len(review_ids) > 1:
            errors.append(f"Gap {gap_id} appears in overlapping active equivalence reviews: {', '.join(sorted(review_ids))}")

    gap_counts = _enum_counts(GapReviewDecision, [item.decision for item in workspace.gaps.reviews])
    remediation_counts = _enum_counts(RemediationReviewDecision, [item.decision for item in workspace.remediations.reviews])
    consolidation_counts = _enum_counts(ConsolidationReviewDecision, [item.decision for item in workspace.consolidations.reviews])
    if gap_counts[GapReviewDecision.PENDING]:
        warnings.append(f"{gap_counts[GapReviewDecision.PENDING]} gap reviews are pending")
    if gap_counts[GapReviewDecision.DEFERRED]:
        warnings.append(f"{gap_counts[GapReviewDecision.DEFERRED]} gap reviews are deferred")
    if remediation_counts[RemediationReviewDecision.PENDING]:
        warnings.append(f"{remediation_counts[RemediationReviewDecision.PENDING]} remediation reviews are pending")
    if remediation_counts[RemediationReviewDecision.DEFERRED]:
        warnings.append(f"{remediation_counts[RemediationReviewDecision.DEFERRED]} remediation reviews are deferred")
    unapproved = len(workspace.remediations.reviews) - remediation_counts[RemediationReviewDecision.APPROVED]
    if unapproved:
        warnings.append(f"{unapproved} remediation proposals are not approved")
    if consolidation_counts[ConsolidationReviewDecision.PROPOSED]:
        warnings.append(f"{consolidation_counts[ConsolidationReviewDecision.PROPOSED]} equivalence reviews are proposed")
    return ReviewSummary(
        review_schema_version=REVIEW_SCHEMA_VERSION,
        validation_passed=not errors,
        validation_errors=errors,
        validation_warnings=warnings,
        gap_decision_counts=gap_counts,
        remediation_decision_counts=remediation_counts,
        consolidation_decision_counts=consolidation_counts,
        approved_remediation_count=remediation_counts[RemediationReviewDecision.APPROVED],
    )


def export_review(snapshot: ConsolidationSnapshot, workspace: ReviewWorkspace,
                  summary: ReviewSummary, output_directory: str | Path) -> ReviewExportManifest:
    if not summary.validation_passed:
        raise ReviewValidationError(summary.validation_errors)
    gaps = {item.consolidated_gap_id: item for item in snapshot.inventory.consolidated_gaps}
    proposals = {item.remediation_id: item for item in snapshot.plan.remediations}
    gap_reviews = {item.gap_id: item for item in workspace.gaps.reviews}
    remediation_reviews = {item.remediation_id: item for item in workspace.remediations.reviews}
    approved: list[ApprovedRemediation] = []
    for remediation_id in sorted(remediation_reviews):
        review = remediation_reviews[remediation_id]
        proposal = proposals[remediation_id]
        gap = gaps[proposal.consolidated_gap_id]
        if review.decision is not RemediationReviewDecision.APPROVED:
            continue
        if gap_reviews[gap.consolidated_gap_id].decision is not GapReviewDecision.CONFIRMED:
            continue
        approved.append(ApprovedRemediation(
            remediation_id=proposal.remediation_id,
            consolidated_gap_id=gap.consolidated_gap_id,
            original_recommendation=proposal.recommended_action,
            suggested_document_type=proposal.suggested_document_type,
            information_to_collect=proposal.information_to_collect,
            original_provisional_priority=proposal.provisional_priority,
            approved_effective_priority=review.priority_override or proposal.provisional_priority,
            reviewer=review.reviewer or "",
            approval_timestamp=review.reviewed_at or datetime.min.replace(tzinfo=timezone.utc),
            review_comment=review.comment or "",
            source_gap_ids=gap.source_gap_ids,
            question_ids=gap.question_ids,
            requirement_ids=gap.requirement_ids,
            supporting_evidence=gap.supporting_evidence,
            conflicting_evidence=gap.conflicting_evidence,
        ))
    equivalences = [ReviewedEquivalence(
        review_id=item.review_id,
        gap_ids=sorted(item.gap_ids),
        reviewer=item.reviewer or "",
        confirmed_at=item.reviewed_at or datetime.min.replace(tzinfo=timezone.utc),
        rationale=item.comment or "",
    ) for item in workspace.consolidations.reviews if item.decision is ConsolidationReviewDecision.CONFIRMED_EQUIVALENT]
    equivalences.sort(key=lambda item: item.review_id)
    approved_inventory = ApprovedRemediationInventory(export_schema_version=EXPORT_SCHEMA_VERSION, approved_remediations=approved)
    equivalence_inventory = ReviewedEquivalenceInventory(export_schema_version=EXPORT_SCHEMA_VERSION, confirmed_equivalences=equivalences)
    gap_counts = summary.gap_decision_counts
    manifest = ReviewExportManifest(
        input_consolidation_identity=snapshot.identity,
        input_artifact_fingerprints=snapshot.fingerprints,
        review_workspace_identity=workspace.manifest.workspace_identity,
        review_file_fingerprints=workspace.fingerprints,
        review_schema_version=REVIEW_SCHEMA_VERSION,
        export_schema_version=EXPORT_SCHEMA_VERSION,
        approved_remediation_count=len(approved),
        confirmed_gap_count=gap_counts[GapReviewDecision.CONFIRMED],
        rejected_gap_count=gap_counts[GapReviewDecision.REJECTED],
        deferred_gap_count=gap_counts[GapReviewDecision.DEFERRED],
        pending_gap_count=gap_counts[GapReviewDecision.PENDING],
        confirmed_equivalence_count=len(equivalences),
        validation_warnings=summary.validation_warnings,
        generated_at=datetime.now(timezone.utc),
    )
    _replace_files_atomically(Path(output_directory), {
        "approved-remediations.json": _json_bytes(approved_inventory),
        "approved-remediations.md": _approved_markdown(approved_inventory).encode("utf-8"),
        "review-summary.json": _json_bytes(summary),
        "reviewed-equivalences.json": _json_bytes(equivalence_inventory),
        "review-export-manifest.json": _json_bytes(manifest),
    })
    return manifest


def _review_guide() -> str:
    return """# KGA Human Review Guide

This workspace records human decisions separately from machine-generated consolidation artifacts. Edit only the three review JSON files; do not alter their stable IDs.

## Gap reviews

Decisions are `pending`, `confirmed`, `rejected`, or `deferred`. Any decision other than `pending` requires a reviewer, timezone-aware ISO 8601 `reviewed_at`, and nonempty `comment`.

## Remediation reviews

Decisions are `pending`, `approved`, `rejected`, or `deferred`. Non-pending decisions require the same human metadata. `priority_override` may be `high`, `medium`, or `low`, and is valid only with `approved`. Approval also requires the linked gap to be confirmed.

## Equivalence reviews

Add records only when a human wants to assess two or more consolidated gaps. Use a stable `creview_` ID, distinct `gap_ids`, and `proposed`, `confirmed_equivalent`, or `rejected_equivalence`. Confirmed or rejected decisions require reviewer, timestamp, and rationale in `comment`. Every confirmed-equivalent member gap must itself be confirmed. Overlapping active groups are invalid and are never merged transitively.

Run `kga-review-validate` after edits. Pending work is valid and reported as warnings. No command fabricates approvals or changes the source consolidation artifacts.
"""


def _approved_markdown(inventory: ApprovedRemediationInventory) -> str:
    lines = ["# Approved Remediation Plan", "", f"Approved remediations: {len(inventory.approved_remediations)}", ""]
    if not inventory.approved_remediations:
        lines.extend(["No remediation proposals have been approved.", ""])
    for item in inventory.approved_remediations:
        lines.extend([
            f"## {item.remediation_id}", "",
            f"- Consolidated gap: `{item.consolidated_gap_id}`",
            f"- Machine recommendation: {item.original_recommendation}",
            f"- Original provisional priority: `{item.original_provisional_priority.value}`",
            f"- Human-approved effective priority: `{item.approved_effective_priority.value}`",
            f"- Reviewer: {item.reviewer}",
            f"- Approved at: {item.approval_timestamp.isoformat()}",
            f"- Review comment: {item.review_comment}",
            f"- Requirements: {', '.join(f'`{value}`' for value in item.requirement_ids)}",
            f"- Questions: {', '.join(f'`{value}`' for value in item.question_ids)}",
            f"- Source gaps: {', '.join(f'`{value}`' for value in item.source_gap_ids)}",
            f"- Evidence: {_evidence_refs(item.supporting_evidence + item.conflicting_evidence)}", "",
        ])
    return "\n".join(lines)


def _evidence_refs(items: Sequence[Any]) -> str:
    if not items:
        return "None cited by the source findings"
    return ", ".join(sorted({f"`{item.source_path}:{item.start_line}-{item.end_line}`" for item in items}))


def _review_map(label: str, values: Sequence[Any], field: str, errors: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        identifier = getattr(item, field)
        if identifier in result:
            errors.append(f"Duplicate {label}: {identifier}")
        result[identifier] = item
    return result


def _unique_ids(label: str, values: Sequence[str], errors: list[str]) -> set[str]:
    counts = Counter(values)
    for identifier, count in sorted(counts.items()):
        if count > 1:
            errors.append(f"Duplicate {label} ID: {identifier}")
    return set(values)


def _exact_coverage(label: str, expected: set[str], actual: set[str], errors: list[str]) -> None:
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        errors.append(f"Missing {label}: {', '.join(missing)}")
    if unknown:
        errors.append(f"Unknown {label}: {', '.join(unknown)}")


def _enum_counts(enum_type: Any, values: Sequence[Any]) -> dict[Any, int]:
    return {member: sum(value is member for value in values) for member in enum_type}


def _create_directory_atomically(output: Path, payloads: dict[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, payload in payloads.items():
            (temporary / name).write_bytes(payload)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()


def _replace_files_atomically(output: Path, payloads: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary: list[tuple[Path, Path]] = []
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
