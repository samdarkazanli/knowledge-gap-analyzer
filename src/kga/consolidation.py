"""Conservative offline consolidation of V0.2.2 knowledge-gap findings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from kga.consolidation_models import (
    ConsolidatedGap,
    ConsolidatedGapInventory,
    ConsolidationConfig,
    ConsolidationMethod,
    ConsolidationSummary,
    ProvisionalPriority,
    RemediationPlan,
    RemediationProposal,
)
from kga.formal_ids import stable_id
from kga.formal_models import RequirementType
from kga.models import AnswerabilityStatus, SourcePassage
from kga.report_models import (
    GapSummary,
    GapType,
    KnowledgeGap,
    ReportManifest,
    RequirementCoverage,
)


CONSOLIDATION_SCHEMA_VERSION = "0.2.3"
SUPPORTED_REPORT_SCHEMA_VERSIONS = {"0.2.2"}


class ConsolidationValidationError(ValueError):
    def __init__(self, diagnostics: Sequence[str]) -> None:
        self.diagnostics = list(diagnostics)
        super().__init__("; ".join(self.diagnostics))


class ConsolidationInputs:
    def __init__(
        self,
        gaps: list[KnowledgeGap],
        coverage: list[RequirementCoverage],
        summary: GapSummary,
        manifest: ReportManifest,
        fingerprints: dict[str, str],
    ) -> None:
        self.gaps = gaps
        self.coverage = coverage
        self.summary = summary
        self.manifest = manifest
        self.fingerprints = fingerprints


def load_consolidation_inputs(report_directory: str | Path) -> ConsolidationInputs:
    report = Path(report_directory)
    definitions: dict[str, Any] = {
        "knowledge-gaps.json": list[KnowledgeGap],
        "requirement-coverage.json": list[RequirementCoverage],
        "gap-summary.json": GapSummary,
        "report-manifest.json": ReportManifest,
    }
    missing = [str(report / name) for name in definitions if not (report / name).is_file()]
    if missing:
        raise ConsolidationValidationError(
            [f"Missing required report artifact: {item}" for item in missing]
        )
    try:
        values = {
            name: TypeAdapter(annotation).validate_python(_read_json(report / name))
            for name, annotation in definitions.items()
        }
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ConsolidationValidationError([f"Invalid report artifact: {exc}"]) from exc
    inputs = ConsolidationInputs(
        gaps=values["knowledge-gaps.json"],
        coverage=values["requirement-coverage.json"],
        summary=values["gap-summary.json"],
        manifest=values["report-manifest.json"],
        fingerprints={
            name: _fingerprint(report / name) for name in sorted(definitions)
        },
    )
    diagnostics = validate_consolidation_inputs(inputs)
    if diagnostics:
        raise ConsolidationValidationError(diagnostics)
    return inputs


def load_consolidation_config(path: str | Path | None) -> ConsolidationConfig:
    if path is None:
        return ConsolidationConfig()
    try:
        return ConsolidationConfig.model_validate(_read_json(Path(path)))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ConsolidationValidationError([f"Invalid consolidation config: {exc}"]) from exc


def validate_consolidation_inputs(inputs: ConsolidationInputs) -> list[str]:
    diagnostics: list[str] = []
    manifest = inputs.manifest
    if manifest.report_schema_version not in SUPPORTED_REPORT_SCHEMA_VERSIONS:
        diagnostics.append(
            f"Unsupported gap report schema {manifest.report_schema_version!r}; "
            f"supported: {', '.join(sorted(SUPPORTED_REPORT_SCHEMA_VERSIONS))}"
        )
    if manifest.gap_count != len(inputs.gaps):
        diagnostics.append(
            f"Report manifest gap_count={manifest.gap_count} does not match "
            f"knowledge-gaps.json count {len(inputs.gaps)}"
        )
    if inputs.summary.counts.total_requirements != manifest.requirement_count:
        diagnostics.append("Gap summary requirement count does not match report manifest")
    if sum(inputs.summary.gap_counts_by_type.values()) != len(inputs.gaps):
        diagnostics.append("Gap summary type counts do not match knowledge-gaps.json")

    coverage_by_requirement: dict[str, RequirementCoverage] = {}
    for item in inputs.coverage:
        if item.requirement_id in coverage_by_requirement:
            diagnostics.append(
                f"Duplicate requirement coverage record: {item.requirement_id}"
            )
        coverage_by_requirement[item.requirement_id] = item
    if len(inputs.coverage) != manifest.requirement_count:
        diagnostics.append(
            f"Requirement coverage count {len(inputs.coverage)} does not match "
            f"manifest count {manifest.requirement_count}"
        )

    seen_gaps: set[str] = set()
    expected_status = {
        GapType.EXPLICIT_GAP: AnswerabilityStatus.UNANSWERED,
        GapType.PARTIAL_GAP: AnswerabilityStatus.PARTIALLY_ANSWERED,
        GapType.CONFLICTING_KNOWLEDGE: AnswerabilityStatus.CONFLICTING_EVIDENCE,
        GapType.UNCERTAIN_COVERAGE: AnswerabilityStatus.UNCERTAIN,
    }
    for gap in inputs.gaps:
        if gap.gap_id in seen_gaps:
            diagnostics.append(f"Duplicate source gap ID: {gap.gap_id}")
        seen_gaps.add(gap.gap_id)
        coverage = coverage_by_requirement.get(gap.requirement_id)
        if coverage is None:
            diagnostics.append(
                f"Gap {gap.gap_id} references missing requirement coverage "
                f"{gap.requirement_id}"
            )
            continue
        comparisons = (
            ("concept_id", gap.concept_id, coverage.concept_id),
            ("dimension", gap.dimension, coverage.dimension),
            ("requirement_type", gap.requirement_type, coverage.requirement_type),
            ("question_ids", sorted(gap.question_ids), sorted(coverage.question_ids)),
            ("evaluation_status", gap.evaluation_status.value, coverage.evaluation_status.value),
        )
        for field, actual, expected in comparisons:
            if actual != expected:
                diagnostics.append(
                    f"Gap {gap.gap_id} {field} does not match requirement "
                    f"{gap.requirement_id}"
                )
        if gap.evaluation_status is not expected_status[gap.gap_type]:
            diagnostics.append(
                f"Gap {gap.gap_id} has incompatible gap type and evaluation status"
            )
    return diagnostics


def consolidate_gaps(
    inputs: ConsolidationInputs,
    config: ConsolidationConfig | None = None,
) -> tuple[ConsolidatedGapInventory, RemediationPlan, ConsolidationSummary]:
    diagnostics = validate_consolidation_inputs(inputs)
    if diagnostics:
        raise ConsolidationValidationError(diagnostics)
    policy = config or ConsolidationConfig()
    coverage = {item.requirement_id: item for item in inputs.coverage}
    groups: dict[tuple[str, str, str, str, str, tuple[str, ...]], list[KnowledgeGap]] = defaultdict(list)
    for gap in inputs.gaps:
        key = (
            gap.requirement_id,
            gap.concept_id,
            gap.dimension.value,
            gap.requirement_type.value,
            gap.gap_type.value,
            tuple(sorted(gap.question_ids)),
        )
        groups[key].append(gap)

    consolidated: list[ConsolidatedGap] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda item: item.gap_id)
        consolidated.append(_consolidate_group(members, coverage[members[0].requirement_id]))
    consolidated.sort(key=lambda item: item.consolidated_gap_id)
    _validate_complete_membership(inputs.gaps, consolidated)

    remediations = [_remediation(item, policy) for item in consolidated]
    remediations.sort(key=lambda item: item.remediation_id)
    inventory = ConsolidatedGapInventory(
        schema_version=CONSOLIDATION_SCHEMA_VERSION,
        consolidated_gaps=consolidated,
    )
    plan = RemediationPlan(
        schema_version=CONSOLIDATION_SCHEMA_VERSION,
        remediations=remediations,
    )
    method_counts = {
        method: sum(item.consolidation_method is method for item in consolidated)
        for method in ConsolidationMethod
    }
    priority_counts = {
        priority: sum(item.provisional_priority is priority for item in remediations)
        for priority in ProvisionalPriority
    }
    original_count = len(inputs.gaps)
    consolidated_count = len(consolidated)
    input_identity = stable_id("greport", {
        "report_schema_version": inputs.manifest.report_schema_version,
        "formal_model_identity": inputs.manifest.formal_model_identity,
        "question_set_identity": inputs.manifest.input_question_set_identity,
        "artifact_fingerprints": inputs.fingerprints,
    })
    summary = ConsolidationSummary(
        schema_version=CONSOLIDATION_SCHEMA_VERSION,
        input_report_identity=input_identity,
        input_report_schema_version=inputs.manifest.report_schema_version,
        input_artifact_fingerprints=inputs.fingerprints,
        original_gap_count=original_count,
        consolidated_gap_count=consolidated_count,
        merged_gap_count=original_count - consolidated_count,
        singleton_gap_count=method_counts[ConsolidationMethod.SINGLETON],
        remediation_count=len(remediations),
        priority_distribution=priority_counts,
        consolidation_method_distribution=method_counts,
        validation_warnings=list(inputs.manifest.validation_warnings),
    )
    return inventory, plan, summary


def export_consolidation(
    output_directory: str | Path,
    inventory: ConsolidatedGapInventory,
    plan: RemediationPlan,
    summary: ConsolidationSummary,
) -> None:
    output = Path(output_directory)
    payloads = {
        "consolidated-gaps.json": _json_bytes(inventory),
        "remediation-plan.json": _json_bytes(plan),
        "remediation-plan.md": render_remediation_markdown(inventory, plan, summary).encode("utf-8"),
        "consolidation-summary.json": _json_bytes(summary),
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary: list[tuple[Path, Path]] = []
    try:
        for name, payload in payloads.items():
            target = output / name
            temp = output / f".{name}.tmp"
            temp.write_bytes(payload)
            temporary.append((temp, target))
        for temp, target in temporary:
            os.replace(temp, target)
    finally:
        for temp, _ in temporary:
            if temp.exists():
                temp.unlink()


def render_remediation_markdown(
    inventory: ConsolidatedGapInventory,
    plan: RemediationPlan,
    summary: ConsolidationSummary,
) -> str:
    remediation_by_gap = {item.consolidated_gap_id: item for item in plan.remediations}
    lines = [
        "# Knowledge Gap Remediation Plan", "", "## Summary", "",
        f"- Original gap findings: {summary.original_gap_count}",
        f"- Consolidated gap groups: {summary.consolidated_gap_count}",
        f"- Duplicate representations merged: {summary.merged_gap_count}",
        f"- Singleton groups: {summary.singleton_gap_count}",
        f"- Remediation proposals: {summary.remediation_count}", "",
        "Consolidation removes duplicate representations only; it does not eliminate knowledge gaps.",
        "Priorities are provisional policy outputs and do not infer business criticality.", "",
        "## Priority Distribution", "",
    ]
    for priority in ProvisionalPriority:
        lines.append(f"- `{priority.value}`: {summary.priority_distribution[priority]}")
    lines.extend(["", "## Remediation Proposals", ""])
    for gap in sorted(inventory.consolidated_gaps, key=lambda item: (item.title.casefold(), item.consolidated_gap_id)):
        remediation = remediation_by_gap[gap.consolidated_gap_id]
        lines.extend([
            f"### {gap.title}", "",
            f"- Consolidated gap: `{gap.consolidated_gap_id}`",
            f"- Status: `{gap.gap_status.value}`",
            f"- Dimension/type: `{gap.knowledge_dimension.value}` / `{gap.requirement_type.value}`",
            f"- Source findings: {', '.join(f'`{item}`' for item in gap.source_gap_ids)}",
            f"- Questions: {'; '.join(gap.questions)}",
            f"- Recommended action: {remediation.recommended_action}",
            f"- Suggested document: {remediation.suggested_document_type}",
            f"- Provisional priority: `{remediation.provisional_priority.value}` — {remediation.priority_rationale}",
            f"- Evidence references: {_evidence_references(gap)}", "",
        ])
    lines.extend([
        "## Methodology and Limitations", "",
        "Automatic consolidation requires the same formal requirement plus compatible concept, dimension, requirement type, gap status, and question membership. Similar wording, shared dimensions, or matching concept names are not treated as equivalence.",
        "Recommendations describe information to collect and document; they do not supply missing answers. All proposals require human validation before operational use.",
        "The plan inherits the provenance and limitations of the input V0.2.2 report and its evaluated corpus.", "",
    ])
    return "\n".join(lines)


def _consolidate_group(
    members: Sequence[KnowledgeGap], coverage: RequirementCoverage
) -> ConsolidatedGap:
    first = members[0]
    source_ids = sorted(item.gap_id for item in members)
    method = (
        ConsolidationMethod.SAME_FORMAL_REQUIREMENT
        if len(members) > 1 else ConsolidationMethod.SINGLETON
    )
    identifier = stable_id("cgap", {
        "requirement_id": first.requirement_id,
        "gap_type": first.gap_type.value,
        "source_gap_ids": source_ids,
    })
    label = first.requirement_type.value.replace("_", " ")
    return ConsolidatedGap(
        consolidated_gap_id=identifier,
        title=f"{first.concept_name}: {label} documentation gap",
        description=(
            f"The evaluated corpus has {first.gap_type.value.replace('_', ' ')} "
            f"coverage for the {label} requirement concerning {first.concept_name}."
        ),
        knowledge_dimension=first.dimension,
        requirement_type=first.requirement_type,
        gap_status=first.gap_type,
        concept_ids=[first.concept_id],
        concept_names=[first.concept_name],
        requirement_ids=[first.requirement_id],
        question_ids=sorted(first.question_ids),
        questions=sorted({question for item in members for question in item.questions}),
        source_gap_ids=source_ids,
        evaluation_statuses=sorted(
            {item.evaluation_status for item in members}, key=lambda item: item.value
        ),
        evaluation_explanations=sorted(
            {item.evaluation_explanation for item in members}
        ),
        evaluation_ids=sorted(coverage.evaluation_ids),
        evaluation_run_ids=([coverage.evaluation_run_id] if coverage.evaluation_run_id else []),
        supporting_evidence=_unique_evidence(
            evidence for item in members for evidence in item.supporting_evidence
        ),
        conflicting_evidence=_unique_evidence(
            evidence for item in members for evidence in item.conflicting_evidence
        ),
        missing_information=sorted({question for item in members for question in item.questions}),
        consolidation_method=method,
        requires_human_validation=(
            len(members) > 1
            or first.gap_type in {GapType.CONFLICTING_KNOWLEDGE, GapType.UNCERTAIN_COVERAGE}
        ),
    )


def _remediation(
    gap: ConsolidatedGap, config: ConsolidationConfig
) -> RemediationProposal:
    concept = gap.concept_names[0]
    action_templates = {
        RequirementType.OWNERSHIP: f"Identify and document the responsible team, escalation path, and ownership boundary for {concept}.",
        RequirementType.DEPENDENCIES: f"Verify and document the systems and dependencies associated with {concept}.",
        RequirementType.MONITORING: f"Define and document how {concept} is monitored, including relevant signals and response ownership.",
        RequirementType.MAINTENANCE: f"Define and document the maintenance procedure and prerequisites for {concept}.",
        RequirementType.DEPLOYMENT: f"Define and document the deployment process and validation steps for {concept}.",
        RequirementType.PURPOSE: f"Verify and document the purpose and intended use of {concept}.",
        RequirementType.IDENTIFICATION: f"Verify and document how {concept} is identified in the internal environment.",
        RequirementType.TROUBLESHOOTING: f"Document a verified troubleshooting approach for {concept}.",
        RequirementType.INCIDENT_IMPACT: f"Collect and document the verified impact information for {concept}.",
        RequirementType.INCIDENT_CAUSE: f"Collect and document the confirmed cause evidence for {concept}, distinguishing it from hypotheses.",
        RequirementType.INCIDENT_DIAGNOSIS: f"Document the verified diagnostic path for {concept}.",
        RequirementType.INCIDENT_RESOLUTION: f"Document the verified resolution and validation steps for {concept}.",
        RequirementType.PROCEDURE_STEPS: f"Document the verified steps for {concept}.",
        RequirementType.PROCEDURE_PREREQUISITES: f"Document the prerequisites and validation conditions for {concept}.",
        RequirementType.CONFIGURATION_VALUES: f"Verify and document the supported configuration values for {concept}.",
        RequirementType.CONFIGURATION_CHANGE: f"Document how changes to {concept} should be performed and validated safely.",
        RequirementType.SAFETY_RATIONALE: f"Document the verified rationale and risks associated with {concept}.",
        RequirementType.PROCEDURE_RATIONALE: f"Document the verified operational rationale for {concept}.",
        RequirementType.RESOLUTION: f"Document the verified remediation and validation steps for {concept}.",
    }
    document_types = {
        RequirementType.OWNERSHIP: "service ownership record",
        RequirementType.DEPENDENCIES: "system dependency reference",
        RequirementType.MONITORING: "monitoring and alerting guide",
        RequirementType.MAINTENANCE: "operations runbook",
        RequirementType.DEPLOYMENT: "deployment runbook",
        RequirementType.TROUBLESHOOTING: "troubleshooting guide",
        RequirementType.INCIDENT_IMPACT: "incident review",
        RequirementType.INCIDENT_CAUSE: "incident review",
        RequirementType.INCIDENT_DIAGNOSIS: "incident review",
        RequirementType.INCIDENT_RESOLUTION: "incident review",
        RequirementType.PROCEDURE_STEPS: "operations runbook",
        RequirementType.PROCEDURE_PREREQUISITES: "operations runbook",
        RequirementType.CONFIGURATION_VALUES: "configuration reference",
        RequirementType.CONFIGURATION_CHANGE: "configuration change guide",
    }
    priority = config.priority_by_gap_type[gap.gap_status]
    basis = {
        GapType.CONFLICTING_KNOWLEDGE: "conflicting documented knowledge requires reconciliation before it can be used reliably",
        GapType.EXPLICIT_GAP: "the evaluated corpus does not establish this modeled requirement",
        GapType.PARTIAL_GAP: "some relevant documentation exists but does not fully establish the requirement",
        GapType.UNCERTAIN_COVERAGE: "coverage could not be classified reliably and requires review",
    }[gap.gap_status]
    rationale = (
        f"The configured gap-type policy assigns {priority.value} priority because "
        f"{basis}; no business criticality was inferred."
    )
    return RemediationProposal(
        remediation_id=stable_id("rem", {"consolidated_gap_id": gap.consolidated_gap_id}),
        consolidated_gap_id=gap.consolidated_gap_id,
        recommended_action=action_templates.get(
            gap.requirement_type,
            f"Review and complete documentation addressing: {gap.questions[0]}",
        ),
        suggested_document_type=document_types.get(
            gap.requirement_type, "technical documentation"
        ),
        information_to_collect=list(gap.missing_information),
        provisional_priority=priority,
        priority_rationale=rationale,
        requires_human_validation=True,
    )


def _validate_complete_membership(
    source: Sequence[KnowledgeGap], consolidated: Sequence[ConsolidatedGap]
) -> None:
    source_ids = sorted(item.gap_id for item in source)
    represented = sorted(
        gap_id for item in consolidated for gap_id in item.source_gap_ids
    )
    if represented != source_ids:
        raise ConsolidationValidationError(
            ["Every original gap must be represented exactly once"]
        )


def _unique_evidence(items: Iterable[SourcePassage]) -> list[SourcePassage]:
    values = {
        (item.source_path, item.start_line, item.end_line, item.text): item
        for item in items
    }
    return [values[key] for key in sorted(values)]


def _evidence_references(gap: ConsolidatedGap) -> str:
    evidence = gap.supporting_evidence + gap.conflicting_evidence
    if not evidence:
        return "None cited by the source evaluations"
    return ", ".join(sorted({
        f"`{item.source_path}:{item.start_line}-{item.end_line}`" for item in evidence
    }))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
