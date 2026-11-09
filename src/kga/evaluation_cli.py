"""CLI for full-corpus answerability evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

from kga.corpus import CorpusLoadError, load_corpus
from kga.evaluation import build_evaluation_summary, evaluate_questions
from kga.evaluation_provenance import build_evaluation_run
from kga.evaluation_reporting import (
    QuestionInventoryError,
    export_evaluation_reports,
    load_question_inventory,
)
from kga.openai_evaluator import (
    EVALUATOR_VERSION,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    EvaluatorConfigurationError,
    OpenAIQuestionEvaluator,
    build_evaluation_prompt,
    estimate_prompt_tokens,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kga-evaluate",
        description="Evaluate generated questions against a complete documentation corpus.",
    )
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--model", help="Override the KGA_MODEL environment variable")
    parser.add_argument("--max-questions", type=_positive_int)
    parser.add_argument("--max-api-calls", type=_positive_int)
    parser.add_argument(
        "--context-window-tokens",
        type=_positive_int,
        help="Override KGA_MODEL_CONTEXT_TOKENS (default: 128000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect full-corpus request sizes without initializing a client.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        questions = load_question_inventory(args.questions)
        if args.max_questions is not None:
            questions = questions[: args.max_questions]
        documents = load_corpus(args.corpus)

        if args.dry_run:
            print(json.dumps(_dry_run_summary(args, questions, documents), indent=2))
            return 0

        evaluator = OpenAIQuestionEvaluator.from_env(
            model=args.model,
            max_api_calls=args.max_api_calls,
            context_window_tokens=args.context_window_tokens,
        )
        started_at = datetime.now(timezone.utc)
        results = evaluate_questions(questions, documents, evaluator)
        completed_at = datetime.now(timezone.utc)
        summary = build_evaluation_summary(results, api_calls=evaluator.api_calls)
        evaluation_run = build_evaluation_run(
            documents=documents,
            questions=questions,
            summary=summary,
            model=evaluator.model,
            evaluator_version=EVALUATOR_VERSION,
            prompt_version=PROMPT_VERSION,
            started_at=started_at,
            completed_at=completed_at,
            configuration={
                "max_api_calls": evaluator.max_api_calls,
                "context_window_tokens": evaluator.context_window_tokens,
                "output_token_reserve": evaluator.output_token_reserve,
                "full_corpus": True,
            },
        )
        export_evaluation_reports(
            args.output, results, summary, evaluation_run=evaluation_run
        )
        print(summary.model_dump_json(indent=2))
        return 0
    except (
        CorpusLoadError,
        QuestionInventoryError,
        EvaluatorConfigurationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"kga-evaluate: error: {exc}", file=sys.stderr)
        return 1


def _dry_run_summary(args, questions, documents) -> dict[str, object]:
    try:
        context_window = args.context_window_tokens or int(
            os.getenv("KGA_MODEL_CONTEXT_TOKENS", "128000")
        )
        output_reserve = int(os.getenv("KGA_EVALUATION_OUTPUT_TOKENS", "4096"))
    except ValueError as exc:
        raise EvaluatorConfigurationError(
            "Context and output token settings must be numeric"
        ) from exc
    estimates = [
        estimate_prompt_tokens(SYSTEM_PROMPT, build_evaluation_prompt(item, documents))
        for item in questions
    ]
    largest = max(estimates, default=0)
    return {
        "dry_run": True,
        "questions": len(questions),
        "documents": len(documents),
        "largest_estimated_input_tokens": largest,
        "output_token_reserve": output_reserve,
        "context_window_tokens": context_window,
        "would_exceed_context_window": largest + output_reserve > context_window,
        "would_exceed_api_call_limit": (
            args.max_api_calls is not None
            and len(questions) > args.max_api_calls
        ),
        "maximum_api_calls_with_retries": len(questions) * 2,
        "could_exceed_api_call_limit_with_retries": (
            args.max_api_calls is not None
            and len(questions) * 2 > args.max_api_calls
        ),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
