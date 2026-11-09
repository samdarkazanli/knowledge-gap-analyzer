"""Deterministic offline comparison of baseline and updated evaluations."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from kga.corpus import load_corpus
from kga.documentation_changes import ChangeSet, ChangeValidationResult, ChangeValidationError, load_and_validate
from kga.documentation_tasks import TaskSet
from kga.formal_ids import corpus_snapshot_id, stable_id
from kga.formal_models import EvaluationRun
from kga.models import AnswerabilityStatus, QuestionEvaluation, SourcePassage
from kga.remediation_models import (
    DocumentationChangeStatus, QuestionStatusComparison, RemediationOutcome,
    RemediationOutcomeInventory, RemediationOutcomeStatus, VerificationManifest,
    VerificationRun, VerificationSummary,
)

VERIFICATION_SCHEMA_VERSION = "0.2.5"


class VerificationValidationError(ValueError):
    pass


class EvaluationArtifact:
    def __init__(self, results: list[QuestionEvaluation], run: EvaluationRun | None,
                 fingerprints: dict[str, str]) -> None:
        self.results = results
        self.run = run
        self.fingerprints = fingerprints


def load_evaluation(directory: str | Path) -> EvaluationArtifact:
    root = Path(directory)
    answerability = root / "answerability.json"
    if not answerability.is_file():
        raise VerificationValidationError(f"Missing evaluation artifact: {answerability}")
    try:
        results = TypeAdapter(list[QuestionEvaluation]).validate_python(json.loads(answerability.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise VerificationValidationError(f"Invalid evaluation artifact: {exc}") from exc
    ids = [item.question_id for item in results]
    if len(ids) != len(set(ids)):
        raise VerificationValidationError("Evaluation contains duplicate question IDs")
    run_path = root / "evaluation-run.json"
    run = None
    files = [answerability]
    if run_path.is_file():
        try:
            run = EvaluationRun.model_validate(json.loads(run_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise VerificationValidationError(f"Invalid evaluation run provenance: {exc}") from exc
        files.append(run_path)
    summary = root / "evaluation-summary.json"
    if summary.is_file(): files.append(summary)
    return EvaluationArtifact(results, run, {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(files)})


def verify_remediations(
    tasks: TaskSet, changes: ChangeSet, change_validation: ChangeValidationResult,
    baseline: EvaluationArtifact, updated: EvaluationArtifact,
    baseline_corpus: str | Path, updated_corpus: str | Path,
) -> tuple[RemediationOutcomeInventory, VerificationSummary, VerificationManifest]:
    errors: list[str] = []
    warnings = list(change_validation.validation_warnings)
    if not change_validation.validation_passed:
        errors.extend(change_validation.validation_errors)
    baseline_docs = load_corpus(baseline_corpus)
    updated_docs = load_corpus(updated_corpus)
    baseline_corpus_id = corpus_snapshot_id(baseline_docs)
    updated_corpus_id = corpus_snapshot_id(updated_docs)
    _validate_evidence(baseline.results, baseline_docs, "baseline", errors)
    _validate_evidence(updated.results, updated_docs, "updated", errors)
    baseline_by_id = {item.question_id: item for item in baseline.results}
    updated_by_id = {item.question_id: item for item in updated.results}
    if set(baseline_by_id) != set(updated_by_id):
        errors.append("Baseline and updated evaluations have mismatched question sets")
    for question_id in sorted(set(baseline_by_id) & set(updated_by_id)):
        before, after = baseline_by_id[question_id], updated_by_id[question_id]
        if (before.question, before.dimension, before.concept_normalized_name) != (after.question, after.dimension, after.concept_normalized_name):
            errors.append(f"Question metadata changed for {question_id}")
    if baseline.run and baseline.run.corpus_snapshot_id != baseline_corpus_id:
        errors.append("Baseline evaluation corpus snapshot does not match baseline corpus")
    if updated.run and updated.run.corpus_snapshot_id != updated_corpus_id:
        errors.append("Updated evaluation corpus snapshot does not match updated corpus")
    if baseline.run and updated.run and baseline.run.question_set_id != updated.run.question_set_id:
        errors.append("Baseline and updated evaluation question-set identities differ")
    missing_provenance = baseline.run is None or updated.run is None
    if baseline.run is None: warnings.append("Baseline evaluation has legacy or missing run provenance")
    if updated.run is None: warnings.append("Updated evaluation has legacy or missing run provenance")
    all_target_questions = {qid for task in tasks.inventory.tasks for qid in task.target_question_ids}
    unknown_targets = sorted(all_target_questions - set(baseline_by_id) - set(updated_by_id))
    if unknown_targets:
        errors.append("Tasks reference questions absent from both evaluations: " + ", ".join(unknown_targets))
    if errors:
        raise VerificationValidationError("; ".join(errors))

    fingerprints = {
        **{f"tasks/{key}": value for key, value in tasks.fingerprints.items()},
        **{f"changes/{key}": value for key, value in changes.fingerprints.items()},
        **{f"baseline-evaluation/{key}": value for key, value in baseline.fingerprints.items()},
        **{f"updated-evaluation/{key}": value for key, value in updated.fingerprints.items()},
    }
    run_id = stable_id("verify", {
        "task_set_identity": tasks.manifest.task_set_identity,
        "change_set_identity": changes.manifest.change_set_identity,
        "baseline_corpus": baseline_corpus_id, "updated_corpus": updated_corpus_id,
        "input_fingerprints": fingerprints,
    })
    verified_changes = {item.task_id: item for item in change_validation.verified_changes}
    outcomes: list[RemediationOutcome] = []
    for task in tasks.inventory.tasks:
        comparisons = [_comparison(qid, baseline_by_id.get(qid), updated_by_id.get(qid)) for qid in task.target_question_ids]
        change = verified_changes.get(task.task_id)
        if missing_provenance:
            outcome, rationale = RemediationOutcomeStatus.INCONCLUSIVE, "Evaluation run provenance is missing, so corpus and question-set compatibility cannot be established fully."
        elif change is None or change.declared_status is not DocumentationChangeStatus.DECLARED or not change.files_verified:
            outcome, rationale = RemediationOutcomeStatus.INCONCLUSIVE, "No verified documentation change is declared for this task."
        else:
            outcome, rationale = _classify(comparisons)
        outcomes.append(RemediationOutcome(
            outcome_id=stable_id("outcome", {"verification_run_id": run_id, "task_id": task.task_id}),
            verification_run_id=run_id, task_id=task.task_id,
            remediation_id=task.remediation_id, consolidated_gap_id=task.consolidated_gap_id,
            target_requirement_ids=task.target_requirement_ids,
            target_question_ids=task.target_question_ids, comparisons=comparisons,
            outcome=outcome, rationale=rationale,
            human_reviewer=task.human_reviewer, approval_timestamp=task.approval_timestamp,
            human_review_rationale=task.human_review_rationale,
            baseline_corpus_snapshot_id=baseline_corpus_id,
            updated_corpus_snapshot_id=updated_corpus_id,
        ))
    outcomes.sort(key=lambda item: item.task_id)
    inventory = RemediationOutcomeInventory(schema_version=VERIFICATION_SCHEMA_VERSION, outcomes=outcomes)
    counts = Counter(item.outcome for item in outcomes)
    summary = VerificationSummary(
        schema_version=VERIFICATION_SCHEMA_VERSION, task_count=len(tasks.inventory.tasks),
        outcome_counts={status: counts[status] for status in RemediationOutcomeStatus},
        validation_warnings=warnings,
    )
    verification_run = VerificationRun(
        verification_run_id=run_id, task_set_identity=tasks.manifest.task_set_identity,
        change_set_identity=changes.manifest.change_set_identity,
        baseline_corpus_snapshot_id=baseline_corpus_id, updated_corpus_snapshot_id=updated_corpus_id,
        baseline_evaluation_run_id=baseline.run.evaluation_run_id if baseline.run else None,
        updated_evaluation_run_id=updated.run.evaluation_run_id if updated.run else None,
        question_set_id=baseline.run.question_set_id if baseline.run and updated.run else None,
        input_artifact_fingerprints=fingerprints, provenance_warnings=warnings,
        generated_at=datetime.now(timezone.utc),
    )
    return inventory, summary, VerificationManifest(
        verification_schema_version=VERIFICATION_SCHEMA_VERSION,
        verification_run=verification_run, outcome_count=len(outcomes),
    )


def export_verification(output_directory: str | Path, inventory: RemediationOutcomeInventory,
                        summary: VerificationSummary, manifest: VerificationManifest) -> None:
    lines = ["# Remediation Verification Outcomes", "", f"Outcomes: {len(inventory.outcomes)}", ""]
    for item in inventory.outcomes:
        lines.extend([
            f"## {item.task_id}", "", f"- Outcome: `{item.outcome.value}`",
            f"- Remediation: `{item.remediation_id}`", f"- Rationale: {item.rationale}",
            *[f"- `{value.question_id}`: `{value.baseline_status.value if value.baseline_status else 'missing'}` → `{value.updated_status.value if value.updated_status else 'missing'}`" for value in item.comparisons], "",
        ])
    _atomic_files(Path(output_directory), {
        "remediation-outcomes.json": _json_bytes(inventory),
        "remediation-outcomes.md": "\n".join(lines).encode("utf-8"),
        "verification-summary.json": _json_bytes(summary),
        "verification-manifest.json": _json_bytes(manifest),
    })


def _comparison(question_id: str, before: QuestionEvaluation | None,
                after: QuestionEvaluation | None) -> QuestionStatusComparison:
    return QuestionStatusComparison(
        question_id=question_id,
        baseline_status=before.status if before else None,
        updated_status=after.status if after else None,
        baseline_answer=before.answer if before else None,
        updated_answer=after.answer if after else None,
        baseline_explanation=before.explanation if before else None,
        updated_explanation=after.explanation if after else None,
        baseline_evidence=(before.supporting_evidence + before.conflicting_evidence) if before else [],
        updated_evidence=(after.supporting_evidence + after.conflicting_evidence) if after else [],
    )


def _classify(comparisons: list[QuestionStatusComparison]) -> tuple[RemediationOutcomeStatus, str]:
    special = {AnswerabilityStatus.EVALUATION_ERROR, AnswerabilityStatus.UNCERTAIN, AnswerabilityStatus.CONFLICTING_EVIDENCE}
    if any(item.baseline_status is None or item.updated_status is None or item.baseline_status in special or item.updated_status in special for item in comparisons):
        return RemediationOutcomeStatus.INCONCLUSIVE, "A missing, technical-error, uncertain, or conflicting evaluation prevents reliable comparison."
    rank = {AnswerabilityStatus.UNANSWERED: 0, AnswerabilityStatus.PARTIALLY_ANSWERED: 1, AnswerabilityStatus.ANSWERED: 2}
    if any(rank[item.updated_status] < rank[item.baseline_status] for item in comparisons):
        return RemediationOutcomeStatus.REGRESSED, "At least one targeted question became less answerable; regression takes precedence."
    if all(item.updated_status is AnswerabilityStatus.ANSWERED for item in comparisons):
        return RemediationOutcomeStatus.RESOLVED, "All targeted questions are fully answered in the updated evaluation."
    if any(rank[item.updated_status] > rank[item.baseline_status] for item in comparisons):
        return RemediationOutcomeStatus.PARTIALLY_RESOLVED, "Answerability improved, but at least one targeted question is not fully answered."
    return RemediationOutcomeStatus.UNRESOLVED, "Comparable evaluations show no sufficient improvement and no regression."


def _validate_evidence(results: list[QuestionEvaluation], documents, label: str, errors: list[str]) -> None:
    by_path = {item.source_path: item for item in documents}
    for result in results:
        for evidence in result.supporting_evidence + result.conflicting_evidence:
            document = by_path.get(evidence.source_path)
            if document is None:
                errors.append(f"{label} evaluation {result.question_id} cites unknown source {evidence.source_path}"); continue
            lines = document.content.splitlines()
            if evidence.end_line > len(lines):
                errors.append(f"{label} evaluation {result.question_id} cites invalid lines in {evidence.source_path}"); continue
            selected = "\n".join(lines[evidence.start_line - 1:evidence.end_line])
            if evidence.text not in selected:
                errors.append(f"{label} evaluation {result.question_id} quotation does not match {evidence.source_path}")


def _atomic_files(output: Path, payloads: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True); temporary = []
    try:
        for name, payload in payloads.items():
            temp = output / f".{name}.tmp"; temp.write_bytes(payload); temporary.append((temp, output / name))
        for temp, target in temporary: os.replace(temp, target)
    finally:
        for temp, _ in temporary:
            if temp.exists(): temp.unlink()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
