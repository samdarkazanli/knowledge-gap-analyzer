from datetime import datetime, timezone

from kga.evaluation import FakeQuestionEvaluator, build_evaluation_summary, evaluate_questions
from kga.evaluation_provenance import build_evaluation_run, formalize_evaluations
from kga.evaluation_reporting import export_evaluation_reports
from test_evaluation import make_document, make_question


def test_evaluation_run_captures_snapshots_configuration_and_statistics(tmp_path):
    question = make_question()
    documents = [make_document()]
    results = evaluate_questions([question], documents, FakeQuestionEvaluator())
    summary = build_evaluation_summary(results, api_calls=0)
    started = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 10, 17, 10, 1, tzinfo=timezone.utc)

    run = build_evaluation_run(
        documents=documents,
        questions=[question],
        summary=summary,
        model=None,
        evaluator_version="0.2.3",
        prompt_version="answerability-v0.2.2",
        started_at=started,
        completed_at=completed,
        configuration={"full_corpus": True},
    )
    repeated = build_evaluation_run(
        documents=documents,
        questions=[question],
        summary=summary,
        model=None,
        evaluator_version="0.2.3",
        prompt_version="answerability-v0.2.2",
        started_at=started,
        completed_at=completed,
        configuration={"full_corpus": True},
    )
    evaluations = formalize_evaluations(results, run)
    export_evaluation_reports(tmp_path, results, summary, evaluation_run=run)

    assert run == repeated
    assert run.model is None
    assert run.questions_evaluated == 1
    assert run.api_calls == 0
    assert run.evaluation_errors == 0
    assert run.configuration == {"full_corpus": True}
    assert evaluations[0].evaluation_run_id == run.evaluation_run_id
    assert evaluations[0].result.question_id == question.question_id
    assert "secret" not in run.model_dump_json().casefold()
    assert (tmp_path / "evaluation-run.json").is_file()
    assert (tmp_path / "formal-evaluations.json").is_file()
