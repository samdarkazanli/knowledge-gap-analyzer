import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from kga.consolidation import ConsolidationInputs, consolidate_gaps, export_consolidation
from kga.gap_reporting import build_report, make_manifest
from kga.models import AnswerabilityStatus
from kga.review import (
    ConsolidationReviewDecision,
    GapReviewDecision,
    RemediationReviewDecision,
    ReviewValidationError,
    export_review,
    initialize_review_workspace,
    load_consolidation_snapshot,
    load_review_workspace,
    validate_review,
)
from kga.review_models import (
    ConsolidationReview,
    GapReview,
    ProvisionalPriority,
    RemediationReview,
)
from test_gap_reporting import _evaluation, _model


NOW = "2026-10-17T10:00:00+00:00"


def _consolidation(tmp_path: Path):
    model = _model()
    evaluations = [
        _evaluation(question, AnswerabilityStatus.UNANSWERED)
        for question in model.questions[:3]
    ]
    coverage, gaps, gap_summary = build_report(model, evaluations)
    report_manifest = make_manifest(
        model, coverage, gaps, gap_summary, {},
        ["Evaluation run provenance is unavailable; legacy evaluation input accepted"],
        strict=False, evaluation_run=None,
    )
    inputs = ConsolidationInputs(gaps, coverage, gap_summary, report_manifest, {})
    inventory, plan, summary = consolidate_gaps(inputs)
    directory = tmp_path / "consolidation"
    export_consolidation(directory, inventory, plan, summary)
    return directory


def _workspace(tmp_path: Path):
    consolidation = _consolidation(tmp_path)
    review = tmp_path / "review"
    snapshot = load_consolidation_snapshot(consolidation)
    initialize_review_workspace(snapshot, review)
    return consolidation, review, snapshot


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _decide(review: Path, snapshot, gap_decision="confirmed", remediation_decision="approved", priority=None):
    gap_id = snapshot.inventory.consolidated_gaps[0].consolidated_gap_id
    remediation = next(item for item in snapshot.plan.remediations if item.consolidated_gap_id == gap_id)
    gaps = _json(review / "gap-reviews.json")
    gap_record = next(item for item in gaps["reviews"] if item["gap_id"] == gap_id)
    gap_record.update(decision=gap_decision, reviewer="reviewer@example", reviewed_at=NOW, comment="Reviewed against source evidence.")
    _write(review / "gap-reviews.json", gaps)
    remediations = _json(review / "remediation-reviews.json")
    remediation_record = next(item for item in remediations["reviews"] if item["remediation_id"] == remediation.remediation_id)
    remediation_record.update(decision=remediation_decision, reviewer="reviewer@example", reviewed_at=NOW, comment="Action is appropriate.")
    if priority:
        remediation_record["priority_override"] = priority
    _write(review / "remediation-reviews.json", remediations)
    return gap_id, remediation.remediation_id


def test_initialization_creates_pending_complete_workspace(tmp_path):
    _, review, snapshot = _workspace(tmp_path)
    workspace = load_review_workspace(review)
    assert len(workspace.gaps.reviews) == len(snapshot.inventory.consolidated_gaps) == 3
    assert len(workspace.remediations.reviews) == len(snapshot.plan.remediations) == 3
    assert all(item.decision is GapReviewDecision.PENDING for item in workspace.gaps.reviews)
    assert all(item.decision is RemediationReviewDecision.PENDING for item in workspace.remediations.reviews)
    assert workspace.consolidations.reviews == []
    assert (review / "review-guide.md").is_file()


def test_pending_workspace_is_valid_with_warnings(tmp_path):
    _, review, snapshot = _workspace(tmp_path)
    summary = validate_review(snapshot, load_review_workspace(review))
    assert summary.validation_passed is True
    assert summary.validation_errors == []
    assert summary.gap_decision_counts[GapReviewDecision.PENDING] == 3
    assert summary.validation_warnings
    assert any("legacy evaluation input" in item for item in summary.validation_warnings)


@pytest.mark.parametrize("decision", ["confirmed", "rejected", "deferred"])
def test_non_pending_gap_decisions_require_human_metadata(decision):
    with pytest.raises(ValidationError, match="non-pending decisions require"):
        GapReview(gap_id="cgap_" + "1" * 24, decision=decision)
    value = GapReview(
        gap_id="cgap_" + "1" * 24, decision=decision,
        reviewer="alex", reviewed_at=NOW, comment="Rationale",
    )
    assert value.decision.value == decision


def test_invalid_decision_and_naive_timestamp_are_rejected():
    with pytest.raises(ValidationError):
        GapReview(gap_id="cgap_" + "1" * 24, decision="invented")
    with pytest.raises(ValidationError, match="timezone"):
        GapReview(
            gap_id="cgap_" + "1" * 24, decision="confirmed",
            reviewer="alex", reviewed_at="2026-10-17T10:00:00", comment="Rationale",
        )


def test_duplicate_missing_and_broken_gap_reviews_are_errors(tmp_path):
    _, review, snapshot = _workspace(tmp_path)
    path = review / "gap-reviews.json"
    values = _json(path)
    values["reviews"].append(values["reviews"][0])
    values["reviews"].pop(1)
    values["reviews"][1]["gap_id"] = "cgap_" + "f" * 24
    _write(path, values)
    summary = validate_review(snapshot, load_review_workspace(review))
    assert not summary.validation_passed
    assert any("Duplicate gap review" in item for item in summary.validation_errors)
    assert any("Missing gap reviews" in item for item in summary.validation_errors)
    assert any("Unknown gap reviews" in item for item in summary.validation_errors)


def test_broken_remediation_reference_is_error(tmp_path):
    _, review, snapshot = _workspace(tmp_path)
    path = review / "remediation-reviews.json"
    values = _json(path)
    values["reviews"][0]["remediation_id"] = "rem_" + "f" * 24
    _write(path, values)
    summary = validate_review(snapshot, load_review_workspace(review))
    assert not summary.validation_passed
    assert any("Unknown remediation reviews" in item for item in summary.validation_errors)


def test_snapshot_identity_and_fingerprint_mismatches_are_errors(tmp_path):
    consolidation, review, snapshot = _workspace(tmp_path)
    manifest_path = review / "review-manifest.json"
    manifest = _json(manifest_path)
    manifest["input_consolidation_identity"] = "csnapshot_" + "f" * 24
    _write(manifest_path, manifest)
    summary = validate_review(snapshot, load_review_workspace(review))
    assert any("snapshot identity" in item for item in summary.validation_errors)
    manifest = _json(manifest_path)
    manifest["input_consolidation_identity"] = snapshot.identity
    manifest["input_artifact_fingerprints"]["remediation-plan.md"] = "0" * 64
    _write(manifest_path, manifest)
    summary = validate_review(snapshot, load_review_workspace(review))
    assert any("fingerprints" in item for item in summary.validation_errors)


@pytest.mark.parametrize("gap_decision", ["pending", "rejected", "deferred"])
def test_approval_is_blocked_unless_gap_is_confirmed(tmp_path, gap_decision):
    _, review, snapshot = _workspace(tmp_path)
    _decide(review, snapshot, gap_decision=gap_decision)
    summary = validate_review(snapshot, load_review_workspace(review))
    assert not summary.validation_passed
    assert any("requires confirmed gap" in item for item in summary.validation_errors)


def test_approved_remediation_priority_override_and_provenance_export(tmp_path):
    _, review, snapshot = _workspace(tmp_path)
    gap_id, remediation_id = _decide(review, snapshot, priority="high")
    workspace = load_review_workspace(review)
    summary = validate_review(snapshot, workspace)
    assert summary.validation_passed
    output = tmp_path / "approved"
    export_review(snapshot, workspace, summary, output)
    approved = _json(output / "approved-remediations.json")["approved_remediations"]
    assert len(approved) == 1
    item = approved[0]
    source_gap = next(value for value in snapshot.inventory.consolidated_gaps if value.consolidated_gap_id == gap_id)
    assert item["remediation_id"] == remediation_id
    assert item["approved_effective_priority"] == "high"
    assert item["source_gap_ids"] == source_gap.source_gap_ids
    assert item["question_ids"] == source_gap.question_ids
    assert item["requirement_ids"] == source_gap.requirement_ids
    assert item["supporting_evidence"] == [value.model_dump(mode="json") for value in source_gap.supporting_evidence]


def test_zero_approved_export_is_valid_and_deterministic(tmp_path):
    _, review, snapshot = _workspace(tmp_path)
    workspace = load_review_workspace(review)
    summary = validate_review(snapshot, workspace)
    one, two = tmp_path / "one", tmp_path / "two"
    export_review(snapshot, workspace, summary, one)
    export_review(snapshot, workspace, summary, two)
    for name in ["approved-remediations.json", "approved-remediations.md", "review-summary.json", "reviewed-equivalences.json"]:
        assert (one / name).read_bytes() == (two / name).read_bytes()
    assert _json(one / "approved-remediations.json")["approved_remediations"] == []


def test_confirmed_equivalence_is_exported_but_does_not_change_task_count(tmp_path):
    _, review, snapshot = _workspace(tmp_path)
    gap_ids = [item.consolidated_gap_id for item in snapshot.inventory.consolidated_gaps[:2]]
    gaps = _json(review / "gap-reviews.json")
    for item in gaps["reviews"]:
        if item["gap_id"] in gap_ids:
            item.update(decision="confirmed", reviewer="alex", reviewed_at=NOW, comment="Gap confirmed.")
    _write(review / "gap-reviews.json", gaps)
    equivalences = _json(review / "consolidation-reviews.json")
    equivalences["reviews"] = [{
        "review_id": "creview_" + "1" * 24,
        "gap_ids": list(reversed(gap_ids)),
        "decision": "confirmed_equivalent",
        "reviewer": "alex", "reviewed_at": NOW,
        "comment": "Both records describe the same human-verified requirement.",
    }]
    _write(review / "consolidation-reviews.json", equivalences)
    manifest = _json(review / "review-manifest.json")
    manifest["consolidation_review_count"] = 1
    _write(review / "review-manifest.json", manifest)
    workspace = load_review_workspace(review)
    summary = validate_review(snapshot, workspace)
    assert summary.validation_passed
    output = tmp_path / "approved"
    export_review(snapshot, workspace, summary, output)
    assert len(_json(output / "reviewed-equivalences.json")["confirmed_equivalences"]) == 1
    assert _json(output / "approved-remediations.json")["approved_remediations"] == []


def test_invalid_and_overlapping_equivalence_groups_are_rejected(tmp_path):
    _, review, snapshot = _workspace(tmp_path)
    ids = [item.consolidated_gap_id for item in snapshot.inventory.consolidated_gaps]
    values = _json(review / "consolidation-reviews.json")
    values["reviews"] = [
        {"review_id": "creview_" + "1" * 24, "gap_ids": ids[:2], "decision": "proposed"},
        {"review_id": "creview_" + "2" * 24, "gap_ids": ids[1:3], "decision": "proposed"},
    ]
    _write(review / "consolidation-reviews.json", values)
    manifest = _json(review / "review-manifest.json")
    manifest["consolidation_review_count"] = 2
    _write(review / "review-manifest.json", manifest)
    summary = validate_review(snapshot, load_review_workspace(review))
    assert any("overlapping active equivalence" in item for item in summary.validation_errors)
    with pytest.raises(ValidationError, match="distinct"):
        ConsolidationReview(review_id="creview_" + "3" * 24, gap_ids=[ids[0], ids[0]])


def test_initialization_refuses_to_overwrite_human_edits(tmp_path):
    _, review, snapshot = _workspace(tmp_path)
    path = review / "gap-reviews.json"
    values = _json(path)
    values["reviews"][0]["comment"] = "Human draft"
    _write(path, values)
    before = path.read_bytes()
    with pytest.raises(ReviewValidationError, match="refusing to overwrite"):
        initialize_review_workspace(snapshot, review)
    assert path.read_bytes() == before
