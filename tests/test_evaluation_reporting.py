import csv
import json

from kga.evaluation import FakeQuestionEvaluator, build_evaluation_summary, evaluate_questions
from kga.evaluation_reporting import export_evaluation_reports, load_question_inventory
from kga.models import AnswerabilityStatus, QuestionEvaluation
from test_evaluation import make_document, make_question


def test_loads_v020_questions_and_exports_all_reports(tmp_path):
    question = make_question()
    inventory = tmp_path / "questions-v020.json"
    inventory.write_text(
        json.dumps([question.model_dump(mode="json")]), encoding="utf-8"
    )

    questions = load_question_inventory(inventory)
    results = evaluate_questions(questions, [make_document()], FakeQuestionEvaluator())
    summary = build_evaluation_summary(results, api_calls=0)
    export_evaluation_reports(tmp_path / "evaluation", results, summary)

    answerability = json.loads(
        (tmp_path / "evaluation" / "answerability.json").read_text(encoding="utf-8")
    )
    summary_json = json.loads(
        (tmp_path / "evaluation" / "evaluation-summary.json").read_text(
            encoding="utf-8"
        )
    )
    with (tmp_path / "evaluation" / "answerability.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert questions == [question]
    assert answerability[0]["question_id"] == question.question_id
    assert answerability[0]["concept_category"] == "company_specific"
    assert summary_json["questions_evaluated"] == 1
    assert summary_json["counts_by_status"]["unanswered"] == 1
    assert summary_json["api_calls"] == 0
    assert summary_json["evaluation_errors"] == 0
    assert summary_json["total_evaluation_attempts"] == 1
    assert summary_json["questions_requiring_retry"] == 0
    assert summary_json["successful_retries"] == 0
    assert summary_json["failed_retries"] == 0
    assert summary_json["retries_skipped_due_to_api_budget"] == 0
    assert summary_json["evidence_validation_failures"] == 0
    assert rows[0]["status"] == "unanswered"
    assert json.loads(rows[0]["supporting_evidence"]) == []
    assert rows[0]["evaluation_attempts"] == "1"
    assert rows[0]["retry_attempted"] == "False"
    assert json.loads(rows[0]["evidence_validation_diagnostics"]) == []


def test_existing_status_values_and_pre_diagnostic_results_remain_compatible():
    assert {item.value for item in AnswerabilityStatus} == {
        "answered",
        "partially_answered",
        "unanswered",
        "conflicting_evidence",
        "uncertain",
        "evaluation_error",
    }
    legacy_result = {
        "question_id": "q_0000000000000001",
        "question": "What purpose does Atlas serve?",
        "concept_name": "Atlas",
        "concept_normalized_name": "atlas",
        "concept_category": "company_specific",
        "concept_type": "application",
        "dimension": "purpose",
        "status": "unanswered",
        "answer": None,
        "explanation": "No supporting information was found.",
        "supporting_evidence": [],
        "conflicting_evidence": [],
        "confidence": 0.9,
    }

    parsed = QuestionEvaluation.model_validate(legacy_result)

    assert parsed.status is AnswerabilityStatus.UNANSWERED
    assert parsed.evaluation_attempts == 1
    assert parsed.evidence_validation_diagnostics == []
