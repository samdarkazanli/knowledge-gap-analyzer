from types import SimpleNamespace

import pytest

from kga.evaluation import evaluate_questions
from kga.models import AnswerabilityStatus, EvaluationDecision
from kga.openai_evaluator import (
    EvaluationApiCallLimitError,
    EvaluationContextError,
    OpenAIQuestionEvaluator,
)
from test_evaluation import evidence, make_document, make_question


class MockCompletions:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.decision, list):
            parsed = self.decision[min(len(self.calls) - 1, len(self.decision) - 1)]
        else:
            parsed = self.decision
        message = SimpleNamespace(parsed=parsed, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def mock_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def unanswered_decision():
    return EvaluationDecision(
        status=AnswerabilityStatus.UNANSWERED,
        answer=None,
        explanation="The complete corpus does not contain the requested information.",
        confidence=0.9,
    )


def test_sends_every_document_with_paths_and_line_numbers_without_live_api():
    completions = MockCompletions(unanswered_decision())
    evaluator = OpenAIQuestionEvaluator(
        model="test-model",
        client=mock_client(completions),
        context_window_tokens=100_000,
        output_token_reserve=0,
    )
    documents = [
        make_document("First fact\nSecond fact"),
        make_document("Another fact").model_copy(
            update={"source_path": "notes/other.txt"}
        ),
    ]

    result = evaluator.evaluate(make_question(), documents)

    assert result.status is AnswerabilityStatus.UNANSWERED
    request = completions.calls[0]
    assert request["model"] == "test-model"
    assert request["response_format"] is EvaluationDecision
    assert "temperature" not in request
    prompt = request["messages"][1]["content"]
    assert "BEGIN DOCUMENT systems/atlas.txt" in prompt
    assert "1: First fact" in prompt
    assert "2: Second fact" in prompt
    assert "BEGIN DOCUMENT notes/other.txt" in prompt
    assert "1: Another fact" in prompt
    system_prompt = request["messages"][0]["content"]
    assert "Preliminary hypotheses and unverified statements are not confirmed" in system_prompt
    assert "outdated information" in system_prompt.casefold()
    assert "documented association does not by" in system_prompt
    assert "does not establish purpose or function" in system_prompt
    assert "incident-specific action may partially describe maintenance" in system_prompt


def test_context_overflow_is_explicit_and_sends_no_request():
    completions = MockCompletions(unanswered_decision())
    evaluator = OpenAIQuestionEvaluator(
        model="test-model",
        client=mock_client(completions),
        context_window_tokens=100,
        output_token_reserve=0,
    )

    with pytest.raises(EvaluationContextError, match="Complete corpus does not fit"):
        evaluator.evaluate(make_question(), [make_document("x" * 500)])

    assert completions.calls == []
    assert evaluator.api_calls == 0


def test_enforces_api_call_limit_before_sending_an_extra_request():
    completions = MockCompletions(unanswered_decision())
    evaluator = OpenAIQuestionEvaluator(
        model="test-model",
        client=mock_client(completions),
        max_api_calls=1,
        context_window_tokens=100_000,
        output_token_reserve=0,
    )
    evaluator.evaluate(make_question(), [make_document()])

    with pytest.raises(EvaluationApiCallLimitError, match="no request was sent"):
        evaluator.evaluate(make_question(), [make_document()])

    assert len(completions.calls) == 1
    assert evaluator.api_calls == 1


def test_retry_uses_a_second_api_call_and_supplies_structured_feedback():
    invalid = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Atlas routes internal orders.",
        explanation="Citation needs correction.",
        supporting_evidence=[evidence(1, "Atlas routes orders")],
        confidence=0.8,
    )
    corrected = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Atlas routes internal orders.",
        explanation="Citation is exact.",
        supporting_evidence=[evidence(1, "Atlas routes internal orders.")],
        confidence=0.9,
    )
    completions = MockCompletions([invalid, corrected])
    evaluator = OpenAIQuestionEvaluator(
        model="test-model",
        client=mock_client(completions),
        max_api_calls=2,
        context_window_tokens=100_000,
        output_token_reserve=0,
    )

    result = evaluate_questions(
        [make_question()], [make_document()], evaluator
    )[0]

    assert result.status is AnswerabilityStatus.ANSWERED
    assert result.retry_succeeded is True
    assert evaluator.api_calls == 2
    assert len(completions.calls) == 2
    retry_prompt = completions.calls[1]["messages"][1]["content"]
    assert "RETRY REQUIRED" in retry_prompt
    assert "field=supporting_evidence" in retry_prompt
    assert "failure_reason=quotation_not_present" in retry_prompt
    assert "source_path=systems/atlas.txt" in retry_prompt
    assert "Atlas routes orders" not in retry_prompt


def test_retry_does_not_exceed_openai_evaluator_api_budget():
    invalid = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="Invalid.",
        explanation="Citation needs correction.",
        supporting_evidence=[evidence(99, "Invalid")],
        confidence=0.8,
    )
    completions = MockCompletions(invalid)
    evaluator = OpenAIQuestionEvaluator(
        model="test-model",
        client=mock_client(completions),
        max_api_calls=1,
        context_window_tokens=100_000,
        output_token_reserve=0,
    )

    result = evaluate_questions(
        [make_question()], [make_document()], evaluator
    )[0]

    assert result.status is AnswerabilityStatus.EVALUATION_ERROR
    assert result.retry_skipped_reason == "api_call_budget_exhausted"
    assert evaluator.api_calls == 1
    assert len(completions.calls) == 1
