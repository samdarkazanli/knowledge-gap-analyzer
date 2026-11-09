"""Command-line interface for KGA concept extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kga.corpus import CorpusLoadError, load_corpus
from kga.extraction import EvidenceValidationError, extract_concepts
from kga.openai_extractor import (
    ExtractionProviderError,
    ExtractorConfigurationError,
    OpenAIConceptExtractor,
)
from kga.passages import segment_corpus
from kga.reporting import build_summary, export_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kga", description="Discover concepts in an internal documentation corpus."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--model", help="Override the KGA_MODEL environment variable")
    parser.add_argument("--max-documents", type=_positive_int)
    parser.add_argument("--max-api-calls", type=_positive_int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect planned work without initializing a client or calling an API.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        documents = load_corpus(args.corpus)
        if args.max_documents is not None:
            documents = documents[: args.max_documents]
        passages = segment_corpus(documents)

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "documents": len(documents),
                        "passages": len(passages),
                        "would_exceed_api_call_limit": (
                            args.max_api_calls is not None
                            and len(passages) > args.max_api_calls
                        ),
                    },
                    indent=2,
                )
            )
            return 0

        extractor = OpenAIConceptExtractor.from_env(
            model=args.model, max_api_calls=args.max_api_calls
        )
        concepts = extract_concepts(documents, extractor)
        summary = build_summary(
            concepts,
            documents_processed=len(documents),
            passages_processed=len(passages),
            api_calls=extractor.api_calls,
        )
        export_reports(args.output, concepts, summary)
        print(summary.model_dump_json(indent=2))
        return 0
    except (
        CorpusLoadError,
        EvidenceValidationError,
        ExtractionProviderError,
        ExtractorConfigurationError,
        OSError,
    ) as exc:
        print(f"kga: error: {exc}", file=sys.stderr)
        return 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed
