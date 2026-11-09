"""OpenAI-compatible full-corpus answerability evaluator."""

from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Any

from openai import OpenAI, OpenAIError

from kga.evaluation import (
    EvaluationApiBudgetError,
    EvaluationContextError,
    EvaluationError,
)
from kga.models import (
    CorpusDocument,
    EvidenceValidationDiagnostic,
    EvaluationDecision,
    KnowledgeQuestion,
)


EVALUATOR_VERSION = "0.2.3"
PROMPT_VERSION = "answerability-v0.2.2"


SYSTEM_PROMPT = """You evaluate whether one knowledge question can be answered
from the complete documentation corpus supplied by the user. The audience is an
experienced engineer unfamiliar with the organization's internal environment.

Use only explicit corpus evidence. Do not answer from general knowledge. Every
factual claim in an answer must be supported by one or more exact verbatim
quotations with the supplied source path and absolute line numbers. Never cite
the originating question evidence merely because it is attached to a question;
judge answerability from the full corpus shown in this request.

Judge whether the evidence satisfies the specific knowledge dimension requested,
not merely whether it mentions the concept. Apply these conservative rules:
- A label, application name, or identifier does not establish purpose or function.
- Co-location, paired metadata fields, or a documented association does not by
  itself establish that systems interact. Relevant association evidence may make
  an interaction question partially_answered only when the answer explicitly
  states that interaction is not established.
- An incident-specific action may partially describe maintenance or operations,
  but it does not establish a complete ongoing process.
- Preliminary hypotheses and unverified statements are not confirmed facts.
  Later confirmed findings may supersede earlier hypotheses; that chronology is
  not conflicting evidence merely because the statements differ.
- Outdated information must not override current documented facts. Use
  conflicting_evidence only for materially incompatible claims that cannot be
  resolved using documented context, authority, or chronology.

Choose exactly one status:
- answered: the corpus explicitly supports a sufficiently complete answer to
  the question.
- partially_answered: relevant information is explicit, but it does not fully
  establish the requested facts; the answer must state the limitation.
- unanswered: the complete corpus does not establish the requested information.
- conflicting_evidence: materially incompatible evidence prevents a reliable
  answer and cannot be resolved by documented context, authority, or chronology.
- uncertain: the available evidence does not permit reliable classification.

The answer must be null for unanswered, conflicting_evidence, or uncertain.
Confidence is self-assessed from 0 to 1, not a calibrated probability. Do not
return evaluation_error; technical errors are assigned by the application. Do
not infer documentation gaps, prioritize questions, generate documentation, or
perform retrieval.
"""


class EvaluatorConfigurationError(ValueError):
    """Raised when evaluator configuration is missing or invalid."""


class EvaluationProviderError(EvaluationError):
    """Raised when the configured provider cannot produce a usable decision."""


class EvaluationApiCallLimitError(EvaluationApiBudgetError):
    """Raised before a request would exceed the configured API-call limit."""


class OpenAIQuestionEvaluator:
    """Evaluate each question against a complete corpus through Chat Completions."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_api_calls: int = 100,
        context_window_tokens: int = 128_000,
        output_token_reserve: int = 4_096,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise EvaluatorConfigurationError("Model identifier must not be empty")
        if timeout <= 0:
            raise EvaluatorConfigurationError("API timeout must be greater than zero")
        if max_retries < 0:
            raise EvaluatorConfigurationError("API retries must not be negative")
        if max_api_calls < 1:
            raise EvaluatorConfigurationError("API call limit must be greater than zero")
        if context_window_tokens < 1:
            raise EvaluatorConfigurationError("Context window must be greater than zero")
        if output_token_reserve < 0:
            raise EvaluatorConfigurationError("Output token reserve must not be negative")
        if output_token_reserve >= context_window_tokens:
            raise EvaluatorConfigurationError(
                "Output token reserve must be smaller than the context window"
            )
        if client is None and not api_key:
            raise EvaluatorConfigurationError("OPENAI_API_KEY is required")

        self.model = model
        self.max_api_calls = max_api_calls
        self.context_window_tokens = context_window_tokens
        self.output_token_reserve = output_token_reserve
        self._api_calls = 0
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def api_calls(self) -> int:
        return self._api_calls

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        max_api_calls: int | None = None,
        context_window_tokens: int | None = None,
        client: Any | None = None,
    ) -> "OpenAIQuestionEvaluator":
        resolved_model = model or os.getenv("KGA_MODEL")
        if not resolved_model:
            raise EvaluatorConfigurationError(
                "A model is required via --model or KGA_MODEL"
            )
        try:
            timeout = float(os.getenv("KGA_API_TIMEOUT_SECONDS", "30"))
            max_retries = int(os.getenv("KGA_API_MAX_RETRIES", "2"))
            call_limit = (
                max_api_calls
                if max_api_calls is not None
                else int(os.getenv("KGA_MAX_API_CALLS", "100"))
            )
            context_window = (
                context_window_tokens
                if context_window_tokens is not None
                else int(os.getenv("KGA_MODEL_CONTEXT_TOKENS", "128000"))
            )
            output_reserve = int(os.getenv("KGA_EVALUATION_OUTPUT_TOKENS", "4096"))
        except ValueError as exc:
            raise EvaluatorConfigurationError(
                "Timeout, retry, API-call, and context settings must be numeric"
            ) from exc
        return cls(
            model=resolved_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=timeout,
            max_retries=max_retries,
            max_api_calls=call_limit,
            context_window_tokens=context_window,
            output_token_reserve=output_reserve,
            client=client,
        )

    def evaluate(
        self,
        question: KnowledgeQuestion,
        documents: Sequence[CorpusDocument],
        *,
        validation_diagnostics: Sequence[EvidenceValidationDiagnostic] = (),
    ) -> EvaluationDecision:
        user_prompt = build_evaluation_prompt(
            question, documents, validation_diagnostics=validation_diagnostics
        )
        estimated_tokens = estimate_prompt_tokens(SYSTEM_PROMPT, user_prompt)
        required_tokens = estimated_tokens + self.output_token_reserve
        if required_tokens > self.context_window_tokens:
            raise EvaluationContextError(
                "Complete corpus does not fit the configured model context window: "
                f"estimated {estimated_tokens} input tokens plus "
                f"{self.output_token_reserve} reserved output tokens exceeds "
                f"{self.context_window_tokens} tokens; no request was sent"
            )
        if self.api_calls >= self.max_api_calls:
            raise EvaluationApiCallLimitError(
                f"API call limit reached ({self.max_api_calls}); no request was sent"
            )

        self._api_calls += 1
        try:
            completion = self._client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=EvaluationDecision,
            )
        except (OpenAIError, ValueError, TypeError) as exc:
            raise EvaluationProviderError(
                f"Answerability request failed for {question.question_id}: {exc}"
            ) from exc

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise EvaluationProviderError(
                f"Provider refused answerability evaluation: {message.refusal}"
            )
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise EvaluationProviderError(
                "Provider returned no validated structured evaluation response"
            )
        if not isinstance(parsed, EvaluationDecision):
            try:
                parsed = EvaluationDecision.model_validate(parsed)
            except ValueError as exc:
                raise EvaluationProviderError(
                    "Provider response failed evaluation schema validation"
                ) from exc
        return parsed.model_copy(deep=True)


def build_corpus_context(documents: Sequence[CorpusDocument]) -> str:
    """Render every corpus line with its source path and absolute line number."""

    sections: list[str] = []
    for document in documents:
        numbered = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(document.content.splitlines(), start=1)
        )
        sections.append(
            f"BEGIN DOCUMENT {document.source_path}\n"
            f"{numbered}\n"
            f"END DOCUMENT {document.source_path}"
        )
    return "\n\n".join(sections)


def build_evaluation_prompt(
    question: KnowledgeQuestion,
    documents: Sequence[CorpusDocument],
    *,
    validation_diagnostics: Sequence[EvidenceValidationDiagnostic] = (),
) -> str:
    prompt = (
        f"Question ID: {question.question_id}\n"
        f"Question: {question.question}\n"
        f"Concept: {question.concept_name}\n"
        f"Knowledge dimension: {question.dimension.value}\n\n"
        "The following is the complete corpus. Numeric prefixes are line metadata; "
        "exclude them from quoted evidence text.\n\n"
        f"{build_corpus_context(documents)}"
    )
    if validation_diagnostics:
        prompt += "\n\n" + _validation_retry_feedback(validation_diagnostics)
    return prompt


def _validation_retry_feedback(
    diagnostics: Sequence[EvidenceValidationDiagnostic],
) -> str:
    lines = [
        "RETRY REQUIRED: The previous structured response failed strict citation "
        "validation. Return a complete corrected evaluation for the same question "
        "and corpus. Do not repeat these invalid citations and do not invent or "
        "silently repair evidence:"
    ]
    for item in diagnostics:
        lines.append(
            "- "
            f"field={item.evidence_field.value}; "
            f"source_path={item.source_path}; "
            f"declared_lines={item.declared_start_line}-{item.declared_end_line}; "
            f"failure_reason={item.failure_reason.value}; "
            f"attempt={item.attempt_number}"
        )
    return "\n".join(lines)


def estimate_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    """Return a conservative tokenizer-independent upper bound using UTF-8 bytes."""

    message_overhead = 512
    return (
        len(system_prompt.encode("utf-8"))
        + len(user_prompt.encode("utf-8"))
        + message_overhead
    )
