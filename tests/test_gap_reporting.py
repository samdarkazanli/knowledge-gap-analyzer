import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kga.gap_reporting import (
    ReportValidationError,
    build_report,
    export_report,
    load_reporting_inputs,
    make_manifest,
)
from kga.knowledge_model import build_formal_knowledge_model, export_formal_knowledge_model
from kga.models import (
    AnswerabilityStatus,
    Concept,
    ConceptCategory,
    ConceptType,
    QuestionEvaluation,
    SourcePassage,
)
from kga.question_generators import TemplateQuestionGenerator, generate_questions
from kga.report_cli import main
from kga.report_models import GapType, RequirementCoverageStatus


def _model():
    evidence = SourcePassage(
        source_path="systems.txt", start_line=1, end_line=2,
        text="Application: Atlas\nOwner: Platform",
    )
    concept = Concept(
        name="Atlas", normalized_name="atlas",
        category=ConceptCategory.COMPANY_SPECIFIC,
        concept_type=ConceptType.APPLICATION,
        evidence=[evidence],
    )
    questions = generate_questions([concept], TemplateQuestionGenerator())
    return build_formal_knowledge_model([concept], questions)


def _evaluation(question, status=AnswerabilityStatus.ANSWERED):
    supporting = question.evidence if status in {
        AnswerabilityStatus.ANSWERED, AnswerabilityStatus.PARTIALLY_ANSWERED
    } else []
    conflicting = question.evidence if status is AnswerabilityStatus.CONFLICTING_EVIDENCE else []
    answer = "Platform" if status in {
        AnswerabilityStatus.ANSWERED, AnswerabilityStatus.PARTIALLY_ANSWERED
    } else None
    return QuestionEvaluation(
        question_id=question.question_id, question=question.question,
        concept_name=question.concept_name,
        concept_normalized_name=question.concept_normalized_name,
        concept_category=question.concept_category,
        concept_type=question.concept_type,
        dimension=question.dimension, status=status, answer=answer,
        explanation=f"Fixture result: {status.value}",
        supporting_evidence=supporting, conflicting_evidence=conflicting,
        confidence=0.8,
    )


def _write_inputs(tmp_path: Path, evaluations, *, run=None):
    model = _model()
    model_dir = tmp_path / "model"
    evaluation_dir = tmp_path / "evaluation"
    export_formal_knowledge_model(model_dir, model)
    evaluation_dir.mkdir()
    (evaluation_dir / "answerability.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in evaluations], indent=2),
        encoding="utf-8",
    )
    if run is not None:
        (evaluation_dir / "evaluation-run.json").write_text(
            json.dumps(run, indent=2), encoding="utf-8"
        )
    return model, model_dir, evaluation_dir


@pytest.mark.parametrize(
    ("status", "gap_type"),
    [
        (AnswerabilityStatus.UNANSWERED, GapType.EXPLICIT_GAP),
        (AnswerabilityStatus.PARTIALLY_ANSWERED, GapType.PARTIAL_GAP),
        (AnswerabilityStatus.CONFLICTING_EVIDENCE, GapType.CONFLICTING_KNOWLEDGE),
        (AnswerabilityStatus.UNCERTAIN, GapType.UNCERTAIN_COVERAGE),
    ],
)
def test_statuses_create_conservative_gap_types(status, gap_type):
    model = _model()
    question = model.questions[0]
    coverage, gaps, _ = build_report(model, [_evaluation(question, status)])
    item = next(value for value in coverage if question.question_id in value.question_ids)
    gap = next(value for value in gaps if value.requirement_id == item.requirement_id)
    assert item.question_ids == [question.question_id]
    assert gap.gap_type is gap_type
    assert gap.supporting_evidence == _evaluation(question, status).supporting_evidence
    assert gap.description.startswith(("The evaluated corpus", "The evaluation"))


def test_answered_produces_no_gap_and_evaluation_error_is_anomaly():
    model = _model()
    answered = _evaluation(model.questions[0])
    failed = _evaluation(model.questions[1], AnswerabilityStatus.EVALUATION_ERROR)
    coverage, gaps, summary = build_report(model, [answered, failed])
    assert all(item.requirement_id not in {model.questions[0].requirement_id, model.questions[1].requirement_id} for item in gaps)
    assert summary.counts.evaluation_error_requirements == 1
    assert summary.evaluation_anomalies[0].anomaly_type == "evaluation_error"
    assert summary.counts.not_evaluated_requirements == len(model.requirements) - 2


def test_missing_evaluation_is_not_unanswered():
    model = _model()
    coverage, gaps, summary = build_report(model, [])
    assert all(item.evaluation_status is RequirementCoverageStatus.NOT_EVALUATED for item in coverage)
    assert not gaps
    assert summary.counts.evaluated_requirements == 0
    assert summary.counts.evaluation_coverage_percent == 0


def test_multiple_questions_are_aggregated_once_and_disagreement_requires_review():
    model = _model()
    requirement = model.requirements[0]
    original = next(item for item in model.questions if item.requirement_id == requirement.requirement_id)
    duplicate = original.model_copy(update={"question_id": "q_1111111111111111"})
    requirement.question_ids.append(duplicate.question_id)
    model.questions.append(duplicate)
    first = _evaluation(original, AnswerabilityStatus.ANSWERED)
    second = _evaluation(duplicate, AnswerabilityStatus.UNANSWERED)
    coverage, gaps, summary = build_report(model, [first, second])
    item = next(value for value in coverage if value.requirement_id == requirement.requirement_id)
    assert item.evaluation_status is RequirementCoverageStatus.UNCERTAIN
    assert {value.question_id for value in item.question_evaluations} == {
        first.question_id, second.question_id
    }
    assert summary.counts.total_requirements == len(model.requirements)
    assert next(gap for gap in gaps if gap.requirement_id == requirement.requirement_id).gap_type is GapType.UNCERTAIN_COVERAGE


def test_legacy_evaluations_join_by_question_id_and_duplicate_is_rejected(tmp_path):
    model = _model()
    evaluation = _evaluation(model.questions[0])
    _, model_dir, evaluation_dir = _write_inputs(tmp_path, [evaluation])
    loaded, values, run, warnings, _ = load_reporting_inputs(model_dir, evaluation_dir)
    assert loaded.questions[0].requirement_id
    assert values[0].question_id == evaluation.question_id
    assert run is None and warnings
    (evaluation_dir / "answerability.json").write_text(
        json.dumps([evaluation.model_dump(mode="json")] * 2), encoding="utf-8"
    )
    with pytest.raises(ReportValidationError, match="Duplicate evaluation"):
        load_reporting_inputs(model_dir, evaluation_dir)


def test_wrong_question_set_identity_is_rejected(tmp_path):
    model = _model()
    run = {
        "evaluation_run_id": "run_" + "1" * 24,
        "corpus_snapshot_id": "corpus_" + "2" * 24,
        "question_set_id": "qset_" + "3" * 24,
        "questions_evaluated": 1, "api_calls": 1, "evaluation_errors": 0,
    }
    _, model_dir, evaluation_dir = _write_inputs(tmp_path, [_evaluation(model.questions[0])], run=run)
    with pytest.raises(ReportValidationError, match="question-set identity"):
        load_reporting_inputs(model_dir, evaluation_dir)


def test_gap_ids_and_analytical_outputs_are_deterministic(tmp_path):
    model = _model()
    evaluations = [_evaluation(model.questions[0], AnswerabilityStatus.UNANSWERED)]
    coverage1, gaps1, summary1 = build_report(model, evaluations)
    coverage2, gaps2, summary2 = build_report(model, evaluations)
    assert gaps1[0].gap_id == gaps2[0].gap_id
    fingerprints = {"fixture": "a" * 64}
    manifest1 = make_manifest(model, coverage1, gaps1, summary1, fingerprints, [], strict=False, evaluation_run=None)
    manifest2 = manifest1.model_copy(update={"generated_at": datetime(2030, 1, 1, tzinfo=timezone.utc)})
    first, second = tmp_path / "one", tmp_path / "two"
    export_report(first, coverage1, gaps1, summary1, manifest1)
    export_report(second, coverage2, gaps2, summary2, manifest2)
    for name in ["requirement-coverage.json", "knowledge-gaps.json", "gap-summary.json", "knowledge-gap-report.md"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / "report-manifest.json").read_bytes() != (second / "report-manifest.json").read_bytes()


def test_invalid_input_leaves_no_partial_report(tmp_path):
    model = _model()
    _, model_dir, evaluation_dir = _write_inputs(tmp_path, [_evaluation(model.questions[0])] * 2)
    output = tmp_path / "report"
    assert main(["--model", str(model_dir), "--evaluation", str(evaluation_dir), "--output", str(output)]) == 1
    assert not output.exists()


def test_coverage_denominators_exclude_errors_and_missing():
    model = _model()
    evaluations = [
        _evaluation(model.questions[0], AnswerabilityStatus.ANSWERED),
        _evaluation(model.questions[1], AnswerabilityStatus.UNANSWERED),
        _evaluation(model.questions[2], AnswerabilityStatus.EVALUATION_ERROR),
    ]
    _, _, summary = build_report(model, evaluations)
    assert summary.counts.evaluated_requirements == 2
    assert summary.counts.answerability_coverage_percent == 50
    assert summary.counts.evaluation_coverage_percent == round(200 / len(model.requirements), 2)
