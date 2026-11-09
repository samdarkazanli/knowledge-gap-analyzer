"""Evaluation-run provenance and formal evaluation identities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from kga.formal_ids import (
    corpus_snapshot_id,
    evaluation_id,
    evaluation_run_id,
    question_set_id,
)
from kga.formal_models import EvaluationRun, FormalEvaluation
from kga.models import (
    CorpusDocument,
    EvaluationSummary,
    KnowledgeQuestion,
    QuestionEvaluation,
)


def build_evaluation_run(
    *,
    documents: Sequence[CorpusDocument],
    questions: Sequence[KnowledgeQuestion],
    summary: EvaluationSummary,
    model: str | None,
    evaluator_version: str | None,
    prompt_version: str | None,
    started_at: datetime | None,
    completed_at: datetime | None,
    configuration: Mapping[str, str | int | float | bool | None] | None = None,
) -> EvaluationRun:
    corpus_id = corpus_snapshot_id(documents)
    question_set = question_set_id(questions)
    safe_configuration = dict(configuration or {})
    run_id = evaluation_run_id(
        corpus_id=corpus_id,
        question_set=question_set,
        model=model,
        evaluator_version=evaluator_version,
        prompt_version=prompt_version,
        started_at=started_at,
        configuration=safe_configuration,
    )
    return EvaluationRun(
        evaluation_run_id=run_id,
        corpus_snapshot_id=corpus_id,
        question_set_id=question_set,
        model=model,
        evaluator_version=evaluator_version,
        prompt_version=prompt_version,
        started_at=started_at,
        completed_at=completed_at,
        questions_evaluated=summary.questions_evaluated,
        api_calls=summary.api_calls,
        evaluation_errors=summary.evaluation_errors,
        configuration=safe_configuration,
    )


def formalize_evaluations(
    results: Sequence[QuestionEvaluation], run: EvaluationRun
) -> list[FormalEvaluation]:
    return [
        FormalEvaluation(
            evaluation_id=evaluation_id(run.evaluation_run_id, result.question_id),
            evaluation_run_id=run.evaluation_run_id,
            result=result.model_copy(deep=True),
        )
        for result in results
    ]
