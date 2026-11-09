import json
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from kga.documentation_changes import (
    initialize_changes, load_change_set, validate_changes,
)
from kga.documentation_tasks import (
    TaskValidationError, export_documentation_tasks, generate_documentation_tasks,
    load_approved_plan, load_task_set,
)
from kga.models import AnswerabilityStatus
from kga.models import SourcePassage
from kga.formal_models import EvaluationRun
from kga.formal_ids import corpus_snapshot_id
from kga.corpus import load_corpus
from kga.remediation_models import (
    ChangedFileDeclaration, QuestionStatusComparison, RemediationOutcomeStatus,
)
from kga.remediation_verification import (
    EvaluationArtifact, VerificationValidationError, _classify, load_evaluation,
    export_verification, verify_remediations,
)
from kga.tasks_cli import main as tasks_main


ROOT = Path(__file__).parents[1]
APPROVED = ROOT / "output/approved-plan-v024-test"
EMPTY_APPROVED = ROOT / "output/approved-plan-v024"
CORPUS = ROOT / "corpus/incident-base"
EVALUATION = ROOT / "output/evaluation-v021b-full"


def _tasks(tmp_path: Path):
    plan = load_approved_plan(APPROVED)
    inventory = generate_documentation_tasks(plan)
    directory = tmp_path / "tasks"
    export_documentation_tasks(directory, inventory, plan)
    return directory, load_task_set(directory)


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_generates_only_two_human_approved_tasks_with_provenance():
    plan = load_approved_plan(APPROVED)
    inventory = generate_documentation_tasks(plan)
    assert len(inventory.tasks) == 2
    ownership = next(item for item in inventory.tasks if "Which team owns payments?" in item.information_to_collect)
    assert ownership.priority.value == "high"
    assert ownership.human_reviewer == "sam"
    assert ownership.human_review_rationale
    assert ownership.supporting_evidence
    assert ownership.target_requirement_ids == ["kr_4341f84436f064132e950e16"]
    assert ownership.target_question_ids == ["q_aaabdc7038346fca"]


def test_empty_approved_plan_generates_no_tasks():
    assert generate_documentation_tasks(load_approved_plan(EMPTY_APPROVED)).tasks == []


def test_task_ids_are_stable_and_input_order_independent():
    plan = load_approved_plan(APPROVED)
    first = generate_documentation_tasks(plan)
    plan.inventory.approved_remediations.reverse()
    second = generate_documentation_tasks(plan)
    assert first == second


def test_invalid_target_references_are_rejected():
    plan = load_approved_plan(APPROVED)
    plan.inventory.approved_remediations[0].requirement_ids = []
    with pytest.raises(TaskValidationError, match="lacks target references"):
        generate_documentation_tasks(plan)


def test_duplicate_task_ids_are_rejected(tmp_path):
    directory, _ = _tasks(tmp_path)
    path = directory / "documentation-tasks.json"
    value = json.loads(path.read_text())
    value["tasks"].append(value["tasks"][0])
    _write(path, value)
    with pytest.raises(TaskValidationError, match="Duplicate documentation task ID"):
        load_task_set(directory)


def test_invalid_approved_plan_does_not_create_partial_output(tmp_path):
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    output = tmp_path / "output"
    assert tasks_main(["--approved-plan", str(invalid), "--output", str(output)]) == 1
    assert not output.exists()


def test_change_initialization_and_pending_validation(tmp_path):
    _, tasks = _tasks(tmp_path)
    changes_dir = tmp_path / "changes"
    initialize_changes(tasks, changes_dir)
    changes = load_change_set(changes_dir)
    result = validate_changes(tasks, changes, CORPUS)
    assert len(changes.inventory.changes) == 2
    assert result.validation_passed
    assert len(result.validation_warnings) == 2
    assert all(not item.files_verified for item in result.verified_changes)


def test_path_traversal_and_absolute_paths_are_rejected():
    for value in ("../secret.txt", "/etc/passwd", "nested/../../secret.txt"):
        with pytest.raises(ValidationError, match="safe corpus-relative"):
            ChangedFileDeclaration(source_path=value)


def test_declared_change_requires_existing_file_and_records_fingerprint(tmp_path):
    _, tasks = _tasks(tmp_path)
    changes_dir = tmp_path / "changes"
    initialize_changes(tasks, changes_dir)
    path = changes_dir / "documentation-changes.json"
    value = json.loads(path.read_text())
    record = value["changes"][0]
    record.update(
        status="declared", contributor="alex", recorded_at="2026-10-18T10:00:00+00:00",
        description="Updated incident documentation.",
        changed_files=[{"source_path": "INC-001-ilm-rollover.txt"}],
    )
    _write(path, value)
    result = validate_changes(tasks, load_change_set(changes_dir), CORPUS)
    verified = next(item for item in result.verified_changes if item.task_id == record["task_id"])
    assert result.validation_passed
    assert verified.files_verified
    assert len(verified.changed_files[0].sha256) == 64
    record["changed_files"] = [{"source_path": "missing.txt"}]
    _write(path, value)
    result = validate_changes(tasks, load_change_set(changes_dir), CORPUS)
    assert not result.validation_passed
    assert any("does not exist" in item for item in result.validation_errors)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("unanswered", "answered", RemediationOutcomeStatus.RESOLVED),
        ("unanswered", "partially_answered", RemediationOutcomeStatus.PARTIALLY_RESOLVED),
        ("unanswered", "unanswered", RemediationOutcomeStatus.UNRESOLVED),
        ("answered", "partially_answered", RemediationOutcomeStatus.REGRESSED),
        ("evaluation_error", "answered", RemediationOutcomeStatus.INCONCLUSIVE),
        ("unanswered", "uncertain", RemediationOutcomeStatus.INCONCLUSIVE),
    ],
)
def test_deterministic_outcome_classification(before, after, expected):
    comparison = QuestionStatusComparison(
        question_id="q_0000000000000001",
        baseline_status=AnswerabilityStatus(before),
        updated_status=AnswerabilityStatus(after),
    )
    assert _classify([comparison])[0] is expected


def test_regression_precedes_improvement_for_multiple_questions():
    comparisons = [
        QuestionStatusComparison(question_id="q_0000000000000001", baseline_status="unanswered", updated_status="answered"),
        QuestionStatusComparison(question_id="q_0000000000000002", baseline_status="answered", updated_status="partially_answered"),
    ]
    assert _classify(comparisons)[0] is RemediationOutcomeStatus.REGRESSED


def test_legacy_evaluation_provenance_produces_inconclusive_outcomes(tmp_path):
    _, tasks = _tasks(tmp_path)
    changes_dir = tmp_path / "changes"
    initialize_changes(tasks, changes_dir)
    changes = load_change_set(changes_dir)
    validation = validate_changes(tasks, changes, CORPUS)
    evaluation = load_evaluation(EVALUATION)
    inventory, summary, _ = verify_remediations(
        tasks, changes, validation, evaluation, evaluation, CORPUS, CORPUS
    )
    assert all(item.outcome is RemediationOutcomeStatus.INCONCLUSIVE for item in inventory.outcomes)
    assert any("legacy or missing" in item for item in summary.validation_warnings)


def test_mismatched_question_sets_are_rejected(tmp_path):
    _, tasks = _tasks(tmp_path)
    changes_dir = tmp_path / "changes"
    initialize_changes(tasks, changes_dir)
    changes = load_change_set(changes_dir)
    validation = validate_changes(tasks, changes, CORPUS)
    baseline = load_evaluation(EVALUATION)
    updated = load_evaluation(EVALUATION)
    updated.results.pop()
    with pytest.raises(VerificationValidationError, match="mismatched question sets"):
        verify_remediations(tasks, changes, validation, baseline, updated, CORPUS, CORPUS)


def test_offline_fixture_verifies_full_resolution_with_updated_evidence(tmp_path):
    _, tasks = _tasks(tmp_path)
    updated_corpus = tmp_path / "updated-corpus"
    shutil.copytree(CORPUS, updated_corpus)
    synthetic = updated_corpus / "SYNTHETIC-payments-ownership.txt"
    synthetic.write_text(
        "SYNTHETIC VALIDATION DOCUMENT\n"
        "Responsible team: Payments Platform Team\n"
        "Escalation: Payments Platform handles application incidents.\n"
        "Boundary: Elasticsearch Platform owns only the shared service.\n",
        encoding="utf-8",
    )
    changes_dir = tmp_path / "changes"
    initialize_changes(tasks, changes_dir)
    change_path = changes_dir / "documentation-changes.json"
    values = json.loads(change_path.read_text())
    ownership_task = next(
        item for item in tasks.inventory.tasks
        if "Which team owns payments?" in item.information_to_collect
    )
    record = next(item for item in values["changes"] if item["task_id"] == ownership_task.task_id)
    record.update(
        status="declared", contributor="fixture", recorded_at="2026-10-18T10:00:00+00:00",
        description="Synthetic ownership documentation fixture.",
        changed_files=[{"source_path": synthetic.name}],
    )
    _write(change_path, values)
    changes = load_change_set(changes_dir)
    validation = validate_changes(tasks, changes, updated_corpus)
    source = load_evaluation(EVALUATION)
    target_ids = {qid for task in tasks.inventory.tasks for qid in task.target_question_ids}
    baseline_results = [item for item in source.results if item.question_id in target_ids]
    updated_results = [item.model_copy(deep=True) for item in baseline_results]
    ownership = next(item for item in updated_results if item.question_id in ownership_task.target_question_ids)
    ownership.status = AnswerabilityStatus.ANSWERED
    ownership.answer = "The synthetic document explicitly assigns ownership and escalation responsibilities."
    ownership.explanation = "The updated corpus explicitly documents the requested ownership information."
    ownership.supporting_evidence = [SourcePassage(
        source_path=synthetic.name, start_line=2, end_line=4,
        text=("Responsible team: Payments Platform Team\n"
              "Escalation: Payments Platform handles application incidents.\n"
              "Boundary: Elasticsearch Platform owns only the shared service."),
    )]
    baseline_id = corpus_snapshot_id(load_corpus(CORPUS))
    updated_id = corpus_snapshot_id(load_corpus(updated_corpus))
    common = dict(
        question_set_id="qset_" + "1" * 24, model="offline-fixture",
        evaluator_version="fixture", prompt_version="fixture",
        questions_evaluated=len(baseline_results), api_calls=0, evaluation_errors=0,
    )
    baseline_run = EvaluationRun(
        evaluation_run_id="run_" + "2" * 24,
        corpus_snapshot_id=baseline_id, **common,
    )
    updated_run = EvaluationRun(
        evaluation_run_id="run_" + "3" * 24,
        corpus_snapshot_id=updated_id, **common,
    )
    inventory, summary, manifest = verify_remediations(
        tasks, changes, validation,
        EvaluationArtifact(baseline_results, baseline_run, {"fixture": "a" * 64}),
        EvaluationArtifact(updated_results, updated_run, {"fixture": "b" * 64}),
        CORPUS, updated_corpus,
    )
    outcome = next(item for item in inventory.outcomes if item.task_id == ownership_task.task_id)
    assert outcome.outcome is RemediationOutcomeStatus.RESOLVED
    assert outcome.comparisons[0].updated_evidence[0].source_path == synthetic.name
    first, second = tmp_path / "verification-one", tmp_path / "verification-two"
    export_verification(first, inventory, summary, manifest)
    export_verification(second, inventory, summary, manifest)
    for name in ["remediation-outcomes.json", "remediation-outcomes.md", "verification-summary.json"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
