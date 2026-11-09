"""Provider-independent concept extractor contracts and test implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from kga.models import Concept, DocumentPassage


PassageKey = tuple[str, int, int]


class ConceptExtractor(Protocol):
    """An extractor that discovers concepts in one passage at a time."""

    def extract(self, passage: DocumentPassage) -> list[Concept]:
        """Return concepts supported by the supplied passage."""
        ...


class FakeConceptExtractor:
    """Return preconfigured concepts for exact passage locations.

    This deterministic implementation is intended for tests and local pipeline
    development. Returned models are copied so callers cannot mutate fixtures.
    """

    def __init__(self, results: Mapping[PassageKey, Sequence[Concept]] | None = None):
        self._results = {
            key: tuple(concepts) for key, concepts in (results or {}).items()
        }

    def extract(self, passage: DocumentPassage) -> list[Concept]:
        key = (passage.source_path, passage.start_line, passage.end_line)
        return [concept.model_copy(deep=True) for concept in self._results.get(key, ())]
