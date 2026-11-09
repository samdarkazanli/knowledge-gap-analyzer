"""Provider-independent corpus-wide answerability evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from kga.models import (
    AnswerabilityStatus,
    ConceptCategory,
    CorpusDocument,
    EvidenceField,
    EvidenceValidationDiagnostic,
    EvidenceValidationFailureReason,
    EvaluationDecision,
    EvaluationSummary,
    KnowledgeQuestion,
    QuestionDimension,
    QuestionEvaluation,
    SourcePassage,
)


class EvaluationError(RuntimeError):
    """Base class for technical failures evaluating one question."""


class EvaluationEvidenceError(EvaluationError):
    """Raised when an evaluator returns evidence not present in the corpus."""

    def __init__(self, diagnostics: Sequence[EvidenceValidationDiagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        first = self.diagnostics[0]
        super().__init__(
            "Evaluation evidence validation failed: "
            f"{first.failure_reason.value} for {first.evidence_field.value} at "
            f"{first.source_path}:{first.declared_start_line}-"
            f"{first.declared_end_line} (attempt {first.attempt_number})"
        )


class EvaluationContextError(EvaluationError):
    """Raised when the complete corpus cannot fit the configured context window."""


class EvaluationApiBudgetError(EvaluationError):
    """Raised before a request would exceed the strict evaluation API budget."""


class QuestionEvaluator(Protocol):
    """Evaluate one question against the complete supplied corpus."""

    @property
    def api_calls(self) -> int:
        """Return external provider calls made, or zero for offline evaluators."""
        ...

    def evaluate(
        self,
        question: KnowledgeQuestion,
        documents: Sequence[CorpusDocument],
        *,
        validation_diagnostics: Sequence[EvidenceValidationDiagnostic] = (),
    ) -> EvaluationDecision:
        """Return a decision, optionally correcting prior citation failures."""
        ...


EvaluationOutcome = EvaluationDecision | Exception


class FakeQuestionEvaluator:
    """Deterministic evaluator supporting bounded-retry tests offline."""

    def __init__(
        self,
        decisions: dict[
            str, EvaluationOutcome | Sequence[EvaluationOutcome]
        ] | None = None,
    ) -> None:
        self._decisions = decisions or {}
        self._calls = 0
        self._question_calls: dict[str, int] = {}
        self.received_diagnostics: list[list[EvidenceValidationDiagnostic]] = []

    @property
    def api_calls(self) -> int:
        return 0

    @property
    def evaluation_calls(self) -> int:
        return self._calls

    def evaluate(
        self,
        question: KnowledgeQuestion,
        documents: Sequence[CorpusDocument],
        *,
        validation_diagnostics: Sequence[EvidenceValidationDiagnostic] = (),
    ) -> EvaluationDecision:
        self._calls += 1
        self.received_diagnostics.append(
            [item.model_copy(deep=True) for item in validation_diagnostics]
        )
        question_call = self._question_calls.get(question.question_id, 0)
        self._question_calls[question.question_id] = question_call + 1
        configured = self._decisions.get(question.question_id)
        if isinstance(configured, (list, tuple)):
            outcome = configured[min(question_call, len(configured) - 1)]
        else:
            outcome = configured
        if outcome is None:
            return EvaluationDecision(
                status=AnswerabilityStatus.UNANSWERED,
                answer=None,
                explanation="No configured evidence answers this question.",
                confidence=1,
            )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome.model_copy(deep=True)


def evaluate_questions(
    questions: Sequence[KnowledgeQuestion],
    documents: Sequence[CorpusDocument],
    evaluator: QuestionEvaluator,
) -> list[QuestionEvaluation]:
    """Evaluate in order, retrying citation-validation failures at most once."""

    results: list[QuestionEvaluation] = []
    corpus = list(documents)
    for question in questions:
        results.append(_evaluate_question(question, corpus, evaluator))
    return results


def _evaluate_question(
    question: KnowledgeQuestion,
    corpus: Sequence[CorpusDocument],
    evaluator: QuestionEvaluator,
) -> QuestionEvaluation:
    try:
        decision = evaluator.evaluate(question, corpus)
    except (EvaluationError, ValueError, TypeError) as exc:
        return _evaluation_error(question, _error_text(exc), evaluation_attempts=1)

    try:
        validate_evaluation_evidence(
            decision, corpus, question_id=question.question_id, attempt_number=1
        )
    except EvaluationEvidenceError as first_error:
        return _retry_after_evidence_failure(
            question, corpus, evaluator, first_error.diagnostics
        )
    return _attach_question(question, decision, evaluation_attempts=1)


def _retry_after_evidence_failure(
    question: KnowledgeQuestion,
    corpus: Sequence[CorpusDocument],
    evaluator: QuestionEvaluator,
    diagnostics: list[EvidenceValidationDiagnostic],
) -> QuestionEvaluation:
    try:
        decision = evaluator.evaluate(
            question, corpus, validation_diagnostics=diagnostics
        )
    except EvaluationApiBudgetError as exc:
        return _evaluation_error(
            question,
            _error_text(exc),
            evaluation_attempts=1,
            retry_skipped_reason="api_call_budget_exhausted",
            diagnostics=diagnostics,
        )
    except (EvaluationError, ValueError, TypeError) as exc:
        return _evaluation_error(
            question,
            _error_text(exc),
            evaluation_attempts=2,
            retry_attempted=True,
            diagnostics=diagnostics,
        )

    try:
        validate_evaluation_evidence(
            decision, corpus, question_id=question.question_id, attempt_number=2
        )
    except EvaluationEvidenceError as retry_error:
        all_diagnostics = [*diagnostics, *retry_error.diagnostics]
        return _evaluation_error(
            question,
            _error_text(retry_error),
            evaluation_attempts=2,
            retry_attempted=True,
            diagnostics=all_diagnostics,
        )
    return _attach_question(
        question,
        decision,
        evaluation_attempts=2,
        retry_attempted=True,
        retry_succeeded=True,
        diagnostics=diagnostics,
    )


def validate_evaluation_evidence(
    decision: EvaluationDecision,
    documents: Sequence[CorpusDocument],
    *,
    question_id: str = "q_0000000000000000",
    attempt_number: int = 1,
) -> None:
    """Strictly validate every quotation and raise structured diagnostics."""

    documents_by_path = {document.source_path: document for document in documents}
    if len(documents_by_path) != len(documents):
        raise EvaluationError("Corpus contains duplicate source paths")
    diagnostics: list[EvidenceValidationDiagnostic] = []
    evidence_groups = (
        (EvidenceField.SUPPORTING_EVIDENCE, decision.supporting_evidence),
        (EvidenceField.CONFLICTING_EVIDENCE, decision.conflicting_evidence),
    )
    for evidence_field, passages in evidence_groups:
        for evidence in passages:
            reason = _validation_failure_reason(evidence, documents_by_path)
            if reason is not None:
                diagnostics.append(
                    EvidenceValidationDiagnostic(
                        question_id=question_id,
                        source_path=evidence.source_path,
                        declared_start_line=evidence.start_line,
                        declared_end_line=evidence.end_line,
                        failure_reason=reason,
                        evidence_field=evidence_field,
                        attempt_number=attempt_number,
                    )
                )
    if diagnostics:
        raise EvaluationEvidenceError(diagnostics)


def build_evaluation_summary(
    results: Sequence[QuestionEvaluation], *, api_calls: int
) -> EvaluationSummary:
    status_counts = {status: 0 for status in AnswerabilityStatus}
    dimension_counts = {dimension: 0 for dimension in QuestionDimension}
    category_counts = {category: 0 for category in ConceptCategory}
    for result in results:
        status_counts[result.status] += 1
        dimension_counts[result.dimension] += 1
        category_counts[result.concept_category] += 1
    return EvaluationSummary(
        questions_evaluated=len(results),
        counts_by_status=status_counts,
        counts_by_knowledge_dimension=dimension_counts,
        counts_by_concept_category=category_counts,
        api_calls=api_calls,
        evaluation_errors=status_counts[AnswerabilityStatus.EVALUATION_ERROR],
        total_evaluation_attempts=sum(item.evaluation_attempts for item in results),
        questions_requiring_retry=sum(
            item.retry_attempted or item.retry_skipped_reason is not None
            for item in results
        ),
        successful_retries=sum(item.retry_succeeded for item in results),
        failed_retries=sum(
            item.retry_attempted and not item.retry_succeeded for item in results
        ),
        retries_skipped_due_to_api_budget=sum(
            item.retry_skipped_reason == "api_call_budget_exhausted"
            for item in results
        ),
        evidence_validation_failures=sum(
            len(item.evidence_validation_diagnostics) for item in results
        ),
    )


def _validation_failure_reason(
    evidence: SourcePassage, documents_by_path: dict[str, CorpusDocument]
) -> EvidenceValidationFailureReason | None:
    document = documents_by_path.get(evidence.source_path)
    if document is None:
        return EvidenceValidationFailureReason.UNKNOWN_SOURCE_PATH
    lines = document.content.splitlines()
    if evidence.start_line > len(lines) or evidence.end_line > len(lines):
        return EvidenceValidationFailureReason.INVALID_LINE_RANGE
    selected = "\n".join(lines[evidence.start_line - 1 : evidence.end_line])
    if evidence.text not in selected:
        return EvidenceValidationFailureReason.QUOTATION_NOT_PRESENT
    return None


def _attach_question(
    question: KnowledgeQuestion,
    decision: EvaluationDecision,
    *,
    evaluation_attempts: int,
    retry_attempted: bool = False,
    retry_succeeded: bool = False,
    diagnostics: Sequence[EvidenceValidationDiagnostic] = (),
) -> QuestionEvaluation:
    return QuestionEvaluation(
        question_id=question.question_id,
        question=question.question,
        concept_name=question.concept_name,
        concept_normalized_name=question.concept_normalized_name,
        concept_category=question.concept_category,
        concept_type=question.concept_type,
        dimension=question.dimension,
        **decision.model_dump(),
        evaluation_attempts=evaluation_attempts,
        retry_attempted=retry_attempted,
        retry_succeeded=retry_succeeded,
        evidence_validation_diagnostics=[
            item.model_copy(deep=True) for item in diagnostics
        ],
    )


def _evaluation_error(
    question: KnowledgeQuestion,
    explanation: str,
    *,
    evaluation_attempts: int,
    retry_attempted: bool = False,
    retry_skipped_reason: str | None = None,
    diagnostics: Sequence[EvidenceValidationDiagnostic] = (),
) -> QuestionEvaluation:
    return QuestionEvaluation(
        question_id=question.question_id,
        question=question.question,
        concept_name=question.concept_name,
        concept_normalized_name=question.concept_normalized_name,
        concept_category=question.concept_category,
        concept_type=question.concept_type,
        dimension=question.dimension,
        status=AnswerabilityStatus.EVALUATION_ERROR,
        answer=None,
        explanation=explanation,
        confidence=0,
        evaluation_attempts=evaluation_attempts,
        retry_attempted=retry_attempted,
        retry_succeeded=False,
        retry_skipped_reason=retry_skipped_reason,
        evidence_validation_diagnostics=[
            item.model_copy(deep=True) for item in diagnostics
        ],
    )


def _error_text(exc: Exception) -> str:
    return str(exc) or type(exc).__name__
