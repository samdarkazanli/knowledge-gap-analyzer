"""Offline joins and deterministic knowledge-gap report generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from kga.formal_ids import stable_id
from kga.formal_models import (
    EvaluationRun,
    FormalConcept,
    FormalEvidence,
    FormalEvaluation,
    FormalKnowledgeModel,
    FormalModelManifest,
    FormalQuestion,
    KnowledgeRequirement,
    RequirementType,
)
from kga.knowledge_model import ModelIntegrityError, validate_referential_integrity
from kga.models import AnswerabilityStatus, QuestionEvaluation, SourcePassage
from kga.report_models import (
    CoverageCounts,
    CoverageGroup,
    EvaluationAnomaly,
    GapSummary,
    GapType,
    KnowledgeGap,
    ReportManifest,
    RequirementCoverage,
    RequirementCoverageStatus,
)


REPORT_SCHEMA_VERSION = "0.2.2"
SUPPORTED_MODEL_SCHEMA_VERSIONS = {"0.2.3"}


class ReportValidationError(ValueError):
    def __init__(self, diagnostics: Sequence[str]) -> None:
        self.diagnostics = list(diagnostics)
        super().__init__("; ".join(self.diagnostics))


def load_reporting_inputs(
    model_directory: str | Path,
    evaluation_directory: str | Path,
    *,
    strict: bool = False,
) -> tuple[FormalKnowledgeModel, list[QuestionEvaluation], EvaluationRun | None, list[str], dict[str, str]]:
    model_path = Path(model_directory)
    evaluation_path = Path(evaluation_directory)
    required = {
        "formal-concepts.json": list[FormalConcept],
        "formal-evidence.json": list[FormalEvidence],
        "questions-enriched.json": list[FormalQuestion],
        "knowledge-requirements.json": list[KnowledgeRequirement],
        "model-manifest.json": FormalModelManifest,
    }
    missing = [str(model_path / name) for name in required if not (model_path / name).is_file()]
    answerability = evaluation_path / "answerability.json"
    if not answerability.is_file():
        missing.append(str(answerability))
    if missing:
        raise ReportValidationError([f"Missing required artifact: {item}" for item in missing])

    try:
        loaded = {
            name: TypeAdapter(annotation).validate_python(_read_json(model_path / name))
            for name, annotation in required.items()
        }
        evaluations = TypeAdapter(list[QuestionEvaluation]).validate_python(
            _read_json(answerability)
        )
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ReportValidationError([f"Invalid report input: {exc}"]) from exc

    model = FormalKnowledgeModel(
        concepts=loaded["formal-concepts.json"],
        evidence=loaded["formal-evidence.json"],
        questions=loaded["questions-enriched.json"],
        requirements=loaded["knowledge-requirements.json"],
        manifest=loaded["model-manifest.json"],
    )
    diagnostics: list[str] = []
    if model.manifest.model_schema_version not in SUPPORTED_MODEL_SCHEMA_VERSIONS:
        diagnostics.append(
            f"Unsupported formal model schema {model.manifest.model_schema_version!r}; "
            f"supported: {', '.join(sorted(SUPPORTED_MODEL_SCHEMA_VERSIONS))}"
        )
    try:
        validate_referential_integrity(model)
    except ModelIntegrityError as exc:
        diagnostics.extend(exc.diagnostics)
    _validate_manifest_counts(model, diagnostics)
    _validate_evaluations(model, evaluations, diagnostics)

    run: EvaluationRun | None = None
    run_path = evaluation_path / "evaluation-run.json"
    warnings: list[str] = []
    if run_path.is_file():
        try:
            run = EvaluationRun.model_validate(_read_json(run_path))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            diagnostics.append(f"Invalid evaluation provenance: {exc}")
        if run is not None and run.question_set_id != model.manifest.question_set_id:
            diagnostics.append(
                "Evaluation question-set identity does not match formal model: "
                f"{run.question_set_id} != {model.manifest.question_set_id}"
            )
        formal_evaluations_path = evaluation_path / "formal-evaluations.json"
        if formal_evaluations_path.is_file():
            try:
                formal_evaluations = TypeAdapter(list[FormalEvaluation]).validate_python(
                    _read_json(formal_evaluations_path)
                )
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                diagnostics.append(f"Invalid formal evaluation records: {exc}")
            else:
                run_ids = {item.evaluation_run_id for item in formal_evaluations}
                if run is not None and run_ids != {run.evaluation_run_id}:
                    diagnostics.append(
                        "Formal evaluation records do not belong exclusively to the declared evaluation run"
                    )
                formal_results = {item.result.question_id: item.result for item in formal_evaluations}
                if len(formal_results) != len(formal_evaluations):
                    diagnostics.append("Formal evaluation records contain duplicate question IDs")
                if formal_results != {item.question_id: item for item in evaluations}:
                    diagnostics.append("Formal evaluation records do not match answerability.json")
    else:
        message = "Evaluation run provenance is unavailable; legacy evaluation input accepted"
        (diagnostics if strict else warnings).append(message)
    if diagnostics:
        raise ReportValidationError(diagnostics)

    paths = [model_path / name for name in required] + [answerability]
    if run_path.is_file():
        paths.append(run_path)
        formal_evaluations_path = evaluation_path / "formal-evaluations.json"
        if formal_evaluations_path.is_file():
            paths.append(formal_evaluations_path)
    fingerprints = {
        f"{('model' if path.parent == model_path else 'evaluation')}/{path.name}": _fingerprint(path)
        for path in sorted(paths, key=lambda item: (item.parent.name, item.name))
    }
    return model, evaluations, run, warnings, fingerprints


def build_report(
    model: FormalKnowledgeModel,
    evaluations: Sequence[QuestionEvaluation],
    *,
    evaluation_run: EvaluationRun | None = None,
) -> tuple[list[RequirementCoverage], list[KnowledgeGap], GapSummary]:
    by_question = {item.question_id: item for item in evaluations}
    questions = {item.question_id: item for item in model.questions}
    concepts = {item.concept_id: item for item in model.concepts}
    coverage: list[RequirementCoverage] = []
    gaps: list[KnowledgeGap] = []
    anomalies: list[EvaluationAnomaly] = []
    for requirement in sorted(model.requirements, key=lambda item: item.requirement_id):
        relevant = [by_question[item] for item in requirement.question_ids if item in by_question]
        item = _aggregate_requirement(requirement, relevant, evaluation_run)
        coverage.append(item)
        if item.evaluation_status is RequirementCoverageStatus.EVALUATION_ERROR:
            anomalies.append(EvaluationAnomaly(
                requirement_id=requirement.requirement_id,
                question_ids=requirement.question_ids,
                anomaly_type="evaluation_error",
                explanation=item.explanation,
            ))
        gap_type = _gap_type(item.evaluation_status)
        if gap_type is not None:
            concept = concepts[requirement.concept_id]
            question_values = [questions[qid].question for qid in requirement.question_ids]
            gaps.append(KnowledgeGap(
                gap_id=stable_id("gap", {
                    "requirement_id": requirement.requirement_id,
                    "gap_type": gap_type.value,
                }),
                requirement_id=requirement.requirement_id,
                concept_id=requirement.concept_id,
                concept_name=concept.name,
                concept_category=concept.category,
                concept_type=concept.concept_type,
                dimension=requirement.dimension,
                requirement_type=requirement.requirement_type,
                question_ids=requirement.question_ids,
                questions=question_values,
                gap_type=gap_type,
                description=_gap_description(gap_type, question_values),
                evaluation_status=AnswerabilityStatus(item.evaluation_status.value),
                evaluation_explanation=item.explanation,
                supporting_evidence=item.supporting_evidence,
                conflicting_evidence=item.conflicting_evidence,
                confidence=item.confidence,
                recommended_documentation_action=_documentation_action(
                    requirement.requirement_type, concept.name, question_values[0]
                ),
            ))
    summary = _build_summary(coverage, gaps, anomalies, concepts)
    return coverage, sorted(gaps, key=lambda item: item.gap_id), summary


def export_report(
    output_directory: str | Path,
    coverage: Sequence[RequirementCoverage],
    gaps: Sequence[KnowledgeGap],
    summary: GapSummary,
    manifest: ReportManifest,
) -> None:
    output = Path(output_directory)
    payloads = {
        "requirement-coverage.json": _json_bytes(list(coverage)),
        "knowledge-gaps.json": _json_bytes(list(gaps)),
        "gap-summary.json": _json_bytes(summary),
        "knowledge-gap-report.md": render_markdown(coverage, gaps, summary).encode("utf-8"),
        "report-manifest.json": _json_bytes(manifest),
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary: list[tuple[Path, Path]] = []
    try:
        for name, data in payloads.items():
            target = output / name
            temp = output / f".{name}.tmp"
            temp.write_bytes(data)
            temporary.append((temp, target))
        for temp, target in temporary:
            os.replace(temp, target)
    finally:
        for temp, _ in temporary:
            if temp.exists():
                temp.unlink()


def make_manifest(
    model: FormalKnowledgeModel,
    coverage: Sequence[RequirementCoverage],
    gaps: Sequence[KnowledgeGap],
    summary: GapSummary,
    fingerprints: dict[str, str],
    warnings: Sequence[str],
    *,
    strict: bool,
    evaluation_run: EvaluationRun | None,
) -> ReportManifest:
    formal_identity = stable_id("fmodel", {
        "concept_inventory_id": model.manifest.concept_inventory_id,
        "question_set_id": model.manifest.question_set_id,
        "model_schema_version": model.manifest.model_schema_version,
    })
    return ReportManifest(
        report_schema_version=REPORT_SCHEMA_VERSION,
        formal_model_identity=formal_identity,
        formal_model_schema_version=model.manifest.model_schema_version,
        evaluation_run_identity=(evaluation_run.evaluation_run_id if evaluation_run else None),
        input_artifact_fingerprints=fingerprints,
        input_question_set_identity=model.manifest.question_set_id,
        requirement_count=len(coverage),
        evaluated_requirement_count=summary.counts.evaluated_requirements,
        gap_count=len(gaps),
        evaluation_anomaly_count=len(summary.evaluation_anomalies),
        reporting_configuration={"strict": strict},
        validation_warnings=list(warnings),
        generated_at=datetime.now(timezone.utc),
    )


def render_markdown(
    coverage: Sequence[RequirementCoverage],
    gaps: Sequence[KnowledgeGap],
    summary: GapSummary,
) -> str:
    counts = summary.counts
    lines = [
        "# Knowledge Gap Report", "", "## Executive Summary", "",
        f"This report assesses {counts.total_requirements} identified knowledge requirements against the supplied evaluation corpus.",
        f"It found {len(gaps)} explicit, partial, conflicting, or uncertain coverage gaps and {len(summary.evaluation_anomalies)} evaluation anomalies.",
        "It measures coverage of identified requirements, not exhaustive organizational knowledge completeness.", "",
        "## Evaluation Coverage", "",
        f"- Evaluated requirements with usable results: {counts.evaluated_requirements}/{counts.total_requirements} ({counts.evaluation_coverage_percent:.2f}%)",
        f"- Not evaluated: {counts.not_evaluated_requirements}",
        f"- Evaluation errors: {counts.evaluation_error_requirements}", "",
        "## Answerability Distribution", "",
        f"- Answered: {counts.answered_requirements}",
        f"- Partially answered: {counts.partially_answered_requirements}",
        f"- Unanswered: {counts.unanswered_requirements}",
        f"- Conflicting evidence: {counts.conflicting_evidence_requirements}",
        f"- Uncertain: {counts.uncertain_requirements}",
        f"- Answerability coverage: {counts.answerability_coverage_percent:.2f}% of requirements with usable evaluations", "",
        "## Knowledge Gaps by Concept", "",
    ]
    by_concept: dict[str, list[KnowledgeGap]] = defaultdict(list)
    for gap in gaps:
        by_concept[gap.concept_id].append(gap)
    if not by_concept:
        lines.extend(["No reportable knowledge gaps were identified.", ""])
    for concept_id in sorted(by_concept, key=lambda key: (by_concept[key][0].concept_name.casefold(), key)):
        items = sorted(by_concept[concept_id], key=lambda item: (item.dimension.value, item.requirement_type.value, item.gap_id))
        first = items[0]
        lines.extend([f"### {first.concept_name}", "", f"Category: `{first.concept_category.value}` · Type: `{first.concept_type.value}`", ""])
        for gap in items:
            lines.extend([
                f"- **{gap.gap_type.value}** — `{gap.dimension.value}` / `{gap.requirement_type.value}`",
                f"  - Question: {gap.questions[0]}",
                f"  - Finding: {gap.description}",
                f"  - Evaluation explanation: {gap.evaluation_explanation}",
                f"  - Source references: {_source_references(gap)}",
                f"  - Recommended action: {gap.recommended_documentation_action}",
            ])
        lines.append("")
    lines.extend(["## Knowledge Gaps by Dimension", ""])
    for group in summary.by_dimension:
        gap_count = sum(1 for item in gaps if item.dimension.value == group.key)
        lines.append(f"- `{group.key}`: {gap_count} gaps across {group.counts.total_requirements} requirements")
    lines.extend(["", "## Conflicting or Uncertain Knowledge", ""])
    special = [item for item in gaps if item.gap_type in {GapType.CONFLICTING_KNOWLEDGE, GapType.UNCERTAIN_COVERAGE}]
    lines.extend([f"- {item.concept_name}: {item.questions[0]}" for item in special] or ["None identified."])
    lines.extend(["", "## Evaluation Anomalies", ""])
    lines.extend([f"- `{item.requirement_id}`: {item.explanation}" for item in summary.evaluation_anomalies] or ["None identified."])
    lines.extend([
        "", "## Methodology and Limitations", "",
        "Evaluations are joined to formal requirements only through stable question IDs. Missing evaluations are reported as `not_evaluated`; they are not treated as unanswered. Evaluation errors are anomalies, not knowledge gaps.",
        "When multiple questions assess one requirement, matching statuses are aggregated. Disagreement is reported conservatively as uncertain coverage; any technical evaluation error remains an anomaly.",
        "The report reflects answerability against the evaluated corpus and does not claim that knowledge is absent from the organization or universally false.", "",
    ])
    return "\n".join(lines)


def _aggregate_requirement(requirement: KnowledgeRequirement, evaluations: list[QuestionEvaluation], run: EvaluationRun | None) -> RequirementCoverage:
    missing = len(requirement.question_ids) - len(evaluations)
    if missing:
        status = RequirementCoverageStatus.NOT_EVALUATED
        explanation = f"{missing} of {len(requirement.question_ids)} question evaluations are missing; requirement coverage was not inferred."
        answer = None
        confidence = None
    elif any(item.status is AnswerabilityStatus.EVALUATION_ERROR for item in evaluations):
        status = RequirementCoverageStatus.EVALUATION_ERROR
        explanation = _join_explanations(evaluations)
        answer = None
        confidence = min(item.confidence for item in evaluations)
    elif len({item.status for item in evaluations}) > 1:
        status = RequirementCoverageStatus.UNCERTAIN
        status_list = ", ".join(sorted({item.status.value for item in evaluations}))
        explanation = f"Question evaluations disagree at requirement level ({status_list}); manual review is required. " + _join_explanations(evaluations)
        answer = None
        confidence = min(item.confidence for item in evaluations)
    else:
        status = RequirementCoverageStatus(evaluations[0].status.value)
        explanation = _join_explanations(evaluations)
        answers = [f"{item.question_id}: {item.answer}" for item in evaluations if item.answer]
        answer = answers[0].split(": ", 1)[1] if len(evaluations) == 1 and answers else ("\n".join(answers) or None)
        confidence = min(item.confidence for item in evaluations)
    return RequirementCoverage(
        requirement_id=requirement.requirement_id,
        concept_id=requirement.concept_id,
        question_ids=requirement.question_ids,
        dimension=requirement.dimension,
        requirement_type=requirement.requirement_type,
        description=requirement.description,
        evaluation_status=status,
        answer=answer,
        explanation=explanation,
        confidence=confidence,
        supporting_evidence=_unique_passages(item for ev in evaluations for item in ev.supporting_evidence),
        conflicting_evidence=_unique_passages(item for ev in evaluations for item in ev.conflicting_evidence),
        evaluation_ids=[stable_id("eval", {"evaluation_run_id": run.evaluation_run_id, "question_id": item.question_id}) for item in evaluations] if run else [],
        evaluation_run_id=run.evaluation_run_id if run else None,
        question_evaluations=sorted(evaluations, key=lambda item: item.question_id),
    )


def _validate_manifest_counts(model: FormalKnowledgeModel, diagnostics: list[str]) -> None:
    expected = {"concept_count": len(model.concepts), "question_count": len(model.questions), "requirement_count": len(model.requirements), "evidence_count": len(model.evidence)}
    for field, value in expected.items():
        if getattr(model.manifest, field) != value:
            diagnostics.append(f"Model manifest {field}={getattr(model.manifest, field)} does not match loaded count {value}")


def _validate_evaluations(model: FormalKnowledgeModel, evaluations: Sequence[QuestionEvaluation], diagnostics: list[str]) -> None:
    formal = {item.question_id: item for item in model.questions}
    seen: set[str] = set()
    for evaluation in evaluations:
        if evaluation.question_id in seen:
            diagnostics.append(f"Duplicate evaluation record for question {evaluation.question_id}")
            continue
        seen.add(evaluation.question_id)
        question = formal.get(evaluation.question_id)
        if question is None:
            diagnostics.append(f"Evaluation references unknown question {evaluation.question_id}")
            continue
        comparisons = (("question", evaluation.question, question.question), ("dimension", evaluation.dimension, question.dimension), ("concept name", evaluation.concept_name, question.concept_name), ("concept category", evaluation.concept_category, question.concept_category), ("concept type", evaluation.concept_type, question.concept_type))
        for label, actual, expected in comparisons:
            if actual != expected:
                diagnostics.append(f"Evaluation {evaluation.question_id} {label} does not match formal question")


def _build_summary(coverage: Sequence[RequirementCoverage], gaps: Sequence[KnowledgeGap], anomalies: Sequence[EvaluationAnomaly], concepts: dict[str, FormalConcept]) -> GapSummary:
    def groups(values: dict[str, list[RequirementCoverage]], labels: dict[str, str] | None = None) -> list[CoverageGroup]:
        return [CoverageGroup(key=key, label=(labels or {}).get(key, key), counts=_counts(items)) for key, items in sorted(values.items())]
    concept_groups: dict[str, list[RequirementCoverage]] = defaultdict(list)
    category_groups: dict[str, list[RequirementCoverage]] = defaultdict(list)
    type_groups: dict[str, list[RequirementCoverage]] = defaultdict(list)
    dimension_groups: dict[str, list[RequirementCoverage]] = defaultdict(list)
    requirement_groups: dict[str, list[RequirementCoverage]] = defaultdict(list)
    for item in coverage:
        concept = concepts[item.concept_id]
        concept_groups[item.concept_id].append(item)
        category_groups[concept.category.value].append(item)
        type_groups[concept.concept_type.value].append(item)
        dimension_groups[item.dimension.value].append(item)
        requirement_groups[item.requirement_type.value].append(item)
    gap_counts = {kind: sum(item.gap_type is kind for item in gaps) for kind in GapType}
    return GapSummary(counts=_counts(coverage), gap_counts_by_type=gap_counts, by_concept=groups(concept_groups, {key: item.name for key, item in concepts.items()}), by_concept_category=groups(category_groups), by_concept_type=groups(type_groups), by_dimension=groups(dimension_groups), by_requirement_type=groups(requirement_groups), evaluation_anomalies=list(anomalies))


def _counts(items: Sequence[RequirementCoverage]) -> CoverageCounts:
    total = len(items)
    counts = {status: sum(item.evaluation_status is status for item in items) for status in RequirementCoverageStatus}
    usable = total - counts[RequirementCoverageStatus.NOT_EVALUATED] - counts[RequirementCoverageStatus.EVALUATION_ERROR]
    return CoverageCounts(
        total_requirements=total,
        evaluated_requirements=usable,
        not_evaluated_requirements=counts[RequirementCoverageStatus.NOT_EVALUATED],
        answered_requirements=counts[RequirementCoverageStatus.ANSWERED],
        partially_answered_requirements=counts[RequirementCoverageStatus.PARTIALLY_ANSWERED],
        unanswered_requirements=counts[RequirementCoverageStatus.UNANSWERED],
        conflicting_evidence_requirements=counts[RequirementCoverageStatus.CONFLICTING_EVIDENCE],
        uncertain_requirements=counts[RequirementCoverageStatus.UNCERTAIN],
        evaluation_error_requirements=counts[RequirementCoverageStatus.EVALUATION_ERROR],
        evaluation_coverage_percent=round(100 * usable / total, 2) if total else 0,
        answerability_coverage_percent=round(100 * counts[RequirementCoverageStatus.ANSWERED] / usable, 2) if usable else 0,
    )


def _gap_type(status: RequirementCoverageStatus) -> GapType | None:
    return {RequirementCoverageStatus.UNANSWERED: GapType.EXPLICIT_GAP, RequirementCoverageStatus.PARTIALLY_ANSWERED: GapType.PARTIAL_GAP, RequirementCoverageStatus.CONFLICTING_EVIDENCE: GapType.CONFLICTING_KNOWLEDGE, RequirementCoverageStatus.UNCERTAIN: GapType.UNCERTAIN_COVERAGE}.get(status)


def _gap_description(kind: GapType, questions: Sequence[str]) -> str:
    subject = questions[0]
    return {
        GapType.EXPLICIT_GAP: f"The evaluated corpus does not establish the knowledge required to answer: {subject}",
        GapType.PARTIAL_GAP: f"The evaluated corpus provides incomplete information for: {subject}",
        GapType.CONFLICTING_KNOWLEDGE: f"The evaluated corpus contains materially conflicting information for: {subject}",
        GapType.UNCERTAIN_COVERAGE: f"The evaluation does not permit a reliable coverage determination for: {subject}",
    }[kind]


def _documentation_action(kind: RequirementType, concept: str, question: str) -> str:
    templates = {
        RequirementType.OWNERSHIP: f"Document the responsible team for {concept}.",
        RequirementType.DEPENDENCIES: f"Document the systems and dependencies associated with {concept}.",
        RequirementType.MONITORING: f"Document how {concept} is monitored.",
        RequirementType.MAINTENANCE: f"Document how {concept} is maintained.",
        RequirementType.DEPLOYMENT: f"Document how {concept} is deployed.",
        RequirementType.INCIDENT_CAUSE: f"Document the confirmed cause of {concept}.",
        RequirementType.INCIDENT_DIAGNOSIS: f"Document how {concept} was diagnosed.",
        RequirementType.INCIDENT_RESOLUTION: f"Document how {concept} was resolved.",
        RequirementType.INCIDENT_IMPACT: f"Document the impact of {concept}.",
        RequirementType.PURPOSE: f"Document the purpose of {concept}.",
        RequirementType.IDENTIFICATION: f"Document how {concept} is identified.",
    }
    return templates.get(kind, f"Review and complete documentation addressing: {question}")


def _unique_passages(items: Iterable[SourcePassage]) -> list[SourcePassage]:
    values: dict[tuple[str, int, int, str], SourcePassage] = {}
    for item in items:
        values[(item.source_path, item.start_line, item.end_line, item.text)] = item
    return [values[key] for key in sorted(values)]


def _join_explanations(items: Sequence[QuestionEvaluation]) -> str:
    return " ".join(f"{item.question_id}: {item.explanation}" for item in sorted(items, key=lambda value: value.question_id))


def _source_references(gap: KnowledgeGap) -> str:
    evidence = gap.supporting_evidence + gap.conflicting_evidence
    if not evidence:
        return "None cited by the evaluation"
    return ", ".join(sorted({f"`{item.source_path}:{item.start_line}-{item.end_line}`" for item in evidence}))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    else:
        value = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in value]
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
