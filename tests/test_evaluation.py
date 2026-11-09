from hashlib import sha256

from kga.evaluation import (
    EvaluationApiBudgetError,
    EvaluationContextError,
    FakeQuestionEvaluator,
    build_evaluation_summary,
    evaluate_questions,
)
from kga.models import (
    AnswerabilityStatus,
    ConceptCategory,
    ConceptType,
    CorpusDocument,
    DocumentFormat,
    EvaluationDecision,
    EvidenceField,
    EvidenceValidationFailureReason,
    KnowledgeQuestion,
    QuestionDimension,
    SourcePassage,
)


def make_document(
    content: str = "Atlas routes internal orders.\nOwner: Commerce Platform"
) -> CorpusDocument:
    return CorpusDocument(
        source_path="systems/atlas.txt",
        format=DocumentFormat.TEXT,
        content=content,
        sha256=sha256(content.encode("utf-8")).hexdigest(),
    )


def make_question(
    question_id: str = "q_0000000000000001",
    question: str = "What purpose does Atlas serve?",
    *,
    dimension: QuestionDimension = QuestionDimension.PURPOSE,
) -> KnowledgeQuestion:
    return KnowledgeQuestion(
        question_id=question_id,
        question=question,
        dimension=dimension,
        concept_name="Atlas",
        concept_normalized_name="atlas",
        concept_category=ConceptCategory.COMPANY_SPECIFIC,
        concept_type=ConceptType.APPLICATION,
        evidence=[
            SourcePassage(
                source_path="systems/atlas.txt",
                start_line=1,
                end_line=1,
                text="Atlas",
            )
        ],
    )


def evidence(line: int, text: str) -> SourcePassage:
    return SourcePassage(
        source_path="systems/atlas.txt",
        start_line=line,
        end_line=line,
        text=text,
    )


def test_evaluates_answered_partial_and_unanswered_questions_in_order():
    answered = make_question()
    partial = make_question(
        "q_0000000000000002",
        "Which systems interact with Atlas?",
        dimension=QuestionDimension.RELATIONSHIPS,
    )
    missing = make_question(
        "q_0000000000000003",
        "How is Atlas deployed?",
        dimension=QuestionDimension.OPERATIONS,
    )
    evaluator = FakeQuestionEvaluator(
        {
            answered.question_id: EvaluationDecision(
                status=AnswerabilityStatus.ANSWERED,
                answer="Atlas routes internal orders.",
                explanation="The purpose is explicit.",
                supporting_evidence=[evidence(1, "Atlas routes internal orders.")],
                confidence=0.98,
            ),
            partial.question_id: EvaluationDecision(
                status=AnswerabilityStatus.PARTIALLY_ANSWERED,
                answer="The documentation associates Atlas with internal orders.",
                explanation="Other interacting systems are not listed.",
                supporting_evidence=[evidence(1, "internal orders")],
                confidence=0.7,
            ),
        }
    )

    results = evaluate_questions([answered, partial, missing], [make_document()], evaluator)

    assert [item.question_id for item in results] == [
        answered.question_id,
        partial.question_id,
        missing.question_id,
    ]
    assert [item.status for item in results] == [
        AnswerabilityStatus.ANSWERED,
        AnswerabilityStatus.PARTIALLY_ANSWERED,
        AnswerabilityStatus.UNANSWERED,
    ]
    assert results[2].answer is None


def test_missing_definition_and_ownership_remain_unanswered():
    definition = make_question(question="What is Atlas?")
    ownership = make_question(
        "q_0000000000000002",
        "Which team owns Atlas?",
        dimension=QuestionDimension.RELATIONSHIPS,
    )
    document = make_document("Atlas appears in this deployment note.")

    results = evaluate_questions(
        [definition, ownership], [document], FakeQuestionEvaluator()
    )

    assert all(item.status is AnswerabilityStatus.UNANSWERED for item in results)
    assert all(item.answer is None for item in results)


def test_speculative_incident_cause_is_not_promoted_to_confirmed():
    question = make_question(
        question="What was the confirmed cause of INC-001?",
        dimension=QuestionDimension.CAUSE,
    )
    corpus = make_document(
        "Preliminary hypothesis: disk pressure may have caused INC-001.\n"
        "The root cause is not yet confirmed."
    )
    evaluator = FakeQuestionEvaluator(
        {
            question.question_id: EvaluationDecision(
                status=AnswerabilityStatus.UNANSWERED,
                answer=None,
                explanation="The only cause is explicitly preliminary and unconfirmed.",
                confidence=0.99,
            )
        }
    )

    result = evaluate_questions([question], [corpus], evaluator)[0]

    assert result.status is AnswerabilityStatus.UNANSWERED
    assert result.answer is None
    assert result.supporting_evidence == []


def test_preserves_conflicting_evidence():
    question = make_question()
    document = make_document("Owner: Team Red\nOwner: Team Blue")
    decision = EvaluationDecision(
        status=AnswerabilityStatus.CONFLICTING_EVIDENCE,
        answer=None,
        explanation="Two current ownership statements conflict.",
        conflicting_evidence=[
            evidence(1, "Owner: Team Red"),
            evidence(2, "Owner: Team Blue"),
        ],
        confidence=0.95,
    )

    result = evaluate_questions(
        [question], [document], FakeQuestionEvaluator({question.question_id: decision})
    )[0]

    assert result.status is AnswerabilityStatus.CONFLICTING_EVIDENCE
    assert len(result.conflicting_evidence) == 2


def test_fabricated_citation_becomes_evaluation_error():
    question = make_question()
    decision = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Fabricated answer.",
        explanation="Invalid citation.",
        supporting_evidence=[
            SourcePassage(
                source_path="fabricated.txt",
                start_line=1,
                end_line=1,
                text="Fabricated answer.",
            )
        ],
        confidence=0.9,
    )

    result = evaluate_questions(
        [question],
        [make_document()],
        FakeQuestionEvaluator({question.question_id: decision}),
    )[0]

    assert result.status is AnswerabilityStatus.EVALUATION_ERROR
    assert "unknown_source_path" in result.explanation
    assert result.supporting_evidence == []
    assert result.evaluation_attempts == 2
    assert result.retry_attempted is True
    assert [item.attempt_number for item in result.evidence_validation_diagnostics] == [
        1,
        2,
    ]


def test_invalid_citation_line_becomes_evaluation_error():
    question = make_question()
    decision = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Invalid.",
        explanation="Invalid line.",
        supporting_evidence=[evidence(99, "Invalid")],
        confidence=0.9,
    )

    result = evaluate_questions(
        [question],
        [make_document()],
        FakeQuestionEvaluator({question.question_id: decision}),
    )[0]

    assert result.status is AnswerabilityStatus.EVALUATION_ERROR
    assert "invalid_line_range" in result.explanation


def test_unsupported_quotation_becomes_evaluation_error():
    question = make_question()
    decision = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Unsupported.",
        explanation="The quotation is not present on the cited line.",
        supporting_evidence=[evidence(1, "Atlas has a fabricated capability")],
        confidence=0.9,
    )

    result = evaluate_questions(
        [question],
        [make_document()],
        FakeQuestionEvaluator({question.question_id: decision}),
    )[0]

    assert result.status is AnswerabilityStatus.EVALUATION_ERROR
    assert "quotation_not_present" in result.explanation


def test_corrected_citation_is_accepted_after_one_retry():
    question = make_question()
    invalid = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Atlas routes internal orders.",
        explanation="The first citation has an incorrect quotation.",
        supporting_evidence=[evidence(1, "Atlas routes orders")],
        confidence=0.8,
    )
    corrected = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Atlas routes internal orders.",
        explanation="The corrected citation is exact.",
        supporting_evidence=[evidence(1, "Atlas routes internal orders.")],
        confidence=0.95,
    )
    evaluator = FakeQuestionEvaluator(
        {question.question_id: [invalid, corrected]}
    )

    result = evaluate_questions([question], [make_document()], evaluator)[0]
    summary = build_evaluation_summary([result], api_calls=2)

    assert result.status is AnswerabilityStatus.ANSWERED
    assert result.evaluation_attempts == 2
    assert result.retry_attempted is True
    assert result.retry_succeeded is True
    assert evaluator.evaluation_calls == 2
    assert len(result.evidence_validation_diagnostics) == 1
    diagnostic = result.evidence_validation_diagnostics[0]
    assert diagnostic.question_id == question.question_id
    assert diagnostic.source_path == "systems/atlas.txt"
    assert diagnostic.declared_start_line == 1
    assert diagnostic.declared_end_line == 1
    assert diagnostic.failure_reason is EvidenceValidationFailureReason.QUOTATION_NOT_PRESENT
    assert diagnostic.evidence_field is EvidenceField.SUPPORTING_EVIDENCE
    assert diagnostic.attempt_number == 1
    assert evaluator.received_diagnostics[0] == []
    assert evaluator.received_diagnostics[1] == [diagnostic]
    assert summary.total_evaluation_attempts == 2
    assert summary.questions_requiring_retry == 1
    assert summary.successful_retries == 1
    assert summary.failed_retries == 0
    assert summary.evidence_validation_failures == 1


def test_retry_is_skipped_when_api_budget_is_exhausted():
    question = make_question()
    invalid = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Invalid.",
        explanation="Invalid citation.",
        supporting_evidence=[evidence(99, "Invalid")],
        confidence=0.8,
    )
    evaluator = FakeQuestionEvaluator(
        {
            question.question_id: [
                invalid,
                EvaluationApiBudgetError("API call limit reached; no request was sent"),
            ]
        }
    )

    result = evaluate_questions([question], [make_document()], evaluator)[0]
    summary = build_evaluation_summary([result], api_calls=1)

    assert result.status is AnswerabilityStatus.EVALUATION_ERROR
    assert result.evaluation_attempts == 1
    assert result.retry_attempted is False
    assert result.retry_skipped_reason == "api_call_budget_exhausted"
    assert len(result.evidence_validation_diagnostics) == 1
    assert summary.questions_requiring_retry == 1
    assert summary.retries_skipped_due_to_api_budget == 1
    assert summary.total_evaluation_attempts == 1


def test_invalid_retry_stops_after_two_attempts_and_preserves_diagnostics():
    question = make_question()
    invalid = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Invalid.",
        explanation="Invalid citation.",
        supporting_evidence=[evidence(99, "Invalid")],
        confidence=0.8,
    )
    evaluator = FakeQuestionEvaluator(
        {question.question_id: [invalid, invalid, invalid]}
    )

    result = evaluate_questions([question], [make_document()], evaluator)[0]
    summary = build_evaluation_summary([result], api_calls=2)

    assert result.status is AnswerabilityStatus.EVALUATION_ERROR
    assert evaluator.evaluation_calls == 2
    assert result.evaluation_attempts == 2
    assert result.retry_attempted is True
    assert result.retry_succeeded is False
    assert [item.attempt_number for item in result.evidence_validation_diagnostics] == [
        1,
        2,
    ]
    assert summary.failed_retries == 1
    assert summary.successful_retries == 0
    assert summary.evidence_validation_failures == 2


def test_context_failure_becomes_evaluation_error_and_summary_counts_it():
    question = make_question()
    evaluator = FakeQuestionEvaluator(
        {question.question_id: EvaluationContextError("complete corpus is too large")}
    )

    results = evaluate_questions([question], [make_document()], evaluator)
    summary = build_evaluation_summary(results, api_calls=0)

    assert results[0].status is AnswerabilityStatus.EVALUATION_ERROR
    assert summary.questions_evaluated == 1
    assert summary.evaluation_errors == 1
    assert summary.counts_by_status[AnswerabilityStatus.EVALUATION_ERROR] == 1
    assert summary.counts_by_knowledge_dimension[QuestionDimension.PURPOSE] == 1
    assert summary.counts_by_concept_category[ConceptCategory.COMPANY_SPECIFIC] == 1
