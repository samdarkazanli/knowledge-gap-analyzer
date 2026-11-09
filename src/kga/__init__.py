"""Knowledge Gap Analyzer."""

from kga.corpus import CorpusLoadError, load_corpus
from kga.extraction import (
    EvidenceValidationError,
    deduplicate_concepts,
    extract_concepts,
    normalize_concept_name,
    verify_evidence,
)
from kga.extractors import ConceptExtractor, FakeConceptExtractor
from kga.models import (
    CategoryStatistics,
    Concept,
    ConceptCategory,
    ConceptExtraction,
    ConceptReport,
    ConceptType,
    CorpusDocument,
    DocumentPassage,
    DocumentFormat,
    ExecutionSummary,
    SourcePassage,
)
from kga.openai_extractor import OpenAIConceptExtractor
from kga.passages import segment_corpus, segment_document

__all__ = [
    "CategoryStatistics",
    "Concept",
    "ConceptCategory",
    "ConceptExtraction",
    "ConceptReport",
    "ConceptType",
    "CorpusDocument",
    "CorpusLoadError",
    "DocumentPassage",
    "DocumentFormat",
    "ExecutionSummary",
    "EvidenceValidationError",
    "ConceptExtractor",
    "FakeConceptExtractor",
    "OpenAIConceptExtractor",
    "SourcePassage",
    "deduplicate_concepts",
    "extract_concepts",
    "normalize_concept_name",
    "load_corpus",
    "segment_corpus",
    "segment_document",
    "verify_evidence",
]
