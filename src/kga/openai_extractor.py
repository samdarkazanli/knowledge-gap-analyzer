"""OpenAI-compatible structured concept extraction."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI, OpenAIError

from kga.models import Concept, ConceptExtraction, DocumentPassage


SYSTEM_PROMPT = """You discover technical concepts in internal documentation.
The audience is an experienced engineer who knows common technologies but does
not know this organization's systems.

Extract concepts explicitly supported by the supplied passage, including
technical concepts, internal applications, identifiers, configurations,
incidents, procedures, operational concepts, and acronyms.

Classify organizational specificity from the concept's meaning and context, not
from how specialized, unfamiliar, abbreviated, or syntactically distinctive its
name looks:
- general_technical: a standard technology, vendor/product term, protocol, API,
  mechanism, lifecycle step, command, configuration key or ordinary setting
  defined outside this organization. A standard mechanism remains general even
  when this organization uses it or an incident mentions it.
- company_specific: an internally created or owned application, service, alias,
  resource name, incident/ticket identifier, environment, configuration choice,
  implementation, convention, runbook practice, or organization-specific use
  whose identity or meaning depends on this organization.
- uncertain: available passage context cannot safely establish whether the
  concept is standard or organization-specific.

Category and concept_type are independent dimensions. First choose category
using the rules above, then choose the functional concept_type. For example, a
configuration, procedure, identifier, acronym, or operational concept may be
general_technical or company_specific. Do not label a concept company_specific
merely because its type is configuration, procedure, identifier, or acronym.

For every concept, return at least one evidence item. Evidence text must be an
exact, verbatim quotation from the supplied passage. Use the supplied source
path and absolute line numbers. Never invent, paraphrase, or correct evidence.
Do not infer missing definitions, assess documentation gaps, create knowledge
graphs, or perform RAG evaluation. Return no concept when evidence is absent.
"""


class ExtractorConfigurationError(ValueError):
    """Raised when required extractor configuration is missing or invalid."""


class ExtractionProviderError(RuntimeError):
    """Raised when the configured provider cannot return a usable response."""


class ApiCallLimitError(ExtractionProviderError):
    """Raised before a request would exceed the configured API-call limit."""


class OpenAIConceptExtractor:
    """Extract concepts through an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_api_calls: int = 100,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ExtractorConfigurationError("Model identifier must not be empty")
        if timeout <= 0:
            raise ExtractorConfigurationError("API timeout must be greater than zero")
        if max_retries < 0:
            raise ExtractorConfigurationError("API retries must not be negative")
        if max_api_calls < 1:
            raise ExtractorConfigurationError("API call limit must be greater than zero")
        if client is None and not api_key:
            raise ExtractorConfigurationError("OPENAI_API_KEY is required")

        self.model = model
        self.max_api_calls = max_api_calls
        self.api_calls = 0
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        max_api_calls: int | None = None,
        client: Any | None = None,
    ) -> "OpenAIConceptExtractor":
        """Build an extractor from environment variables and optional overrides."""

        resolved_model = model or os.getenv("KGA_MODEL")
        if not resolved_model:
            raise ExtractorConfigurationError(
                "A model is required via --model or KGA_MODEL"
            )
        try:
            timeout = float(os.getenv("KGA_API_TIMEOUT_SECONDS", "30"))
            max_retries = int(os.getenv("KGA_API_MAX_RETRIES", "2"))
            resolved_call_limit = (
                max_api_calls
                if max_api_calls is not None
                else int(os.getenv("KGA_MAX_API_CALLS", "100"))
            )
        except ValueError as exc:
            raise ExtractorConfigurationError(
                "Timeout, retry, and API-call settings must be numeric"
            ) from exc

        return cls(
            model=resolved_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=timeout,
            max_retries=max_retries,
            max_api_calls=resolved_call_limit,
            client=client,
        )

    def extract(self, passage: DocumentPassage) -> list[Concept]:
        if self.api_calls >= self.max_api_calls:
            raise ApiCallLimitError(
                f"API call limit reached ({self.max_api_calls}); no request was sent"
            )

        self.api_calls += 1
        try:
            completion = self._client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _passage_prompt(passage)},
                ],
                response_format=ConceptExtraction,
            )
        except (OpenAIError, ValueError, TypeError) as exc:
            raise ExtractionProviderError(
                f"Concept extraction request failed for {passage.source_path} "
                f"lines {passage.start_line}-{passage.end_line}: {exc}"
            ) from exc

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise ExtractionProviderError(
                f"Provider refused concept extraction: {message.refusal}"
            )
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise ExtractionProviderError(
                "Provider returned no validated structured concept response"
            )
        if not isinstance(parsed, ConceptExtraction):
            try:
                parsed = ConceptExtraction.model_validate(parsed)
            except ValueError as exc:
                raise ExtractionProviderError(
                    "Provider response failed concept schema validation"
                ) from exc
        return [concept.model_copy(deep=True) for concept in parsed.concepts]


def _passage_prompt(passage: DocumentPassage) -> str:
    numbered_lines = "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(
            passage.text.split("\n"), start=passage.start_line
        )
    )
    return (
        f"Source path: {passage.source_path}\n"
        f"Allowed line range: {passage.start_line}-{passage.end_line}\n"
        "The numeric prefixes below are metadata; do not include them in evidence text.\n"
        "BEGIN PASSAGE\n"
        f"{numbered_lines}\n"
        "END PASSAGE"
    )
