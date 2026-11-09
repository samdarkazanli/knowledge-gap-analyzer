import json
from pathlib import Path

import pytest

from kga.consolidation import (
    ConsolidationInputs,
    ConsolidationValidationError,
    consolidate_gaps,
    export_consolidation,
    load_consolidation_inputs,
)
from kga.consolidation_models import ConsolidationMethod
from kga.consolidation_cli import main
from kga.gap_reporting import build_report, export_report, make_manifest
from kga.models import AnswerabilityStatus
from kga.report_models import GapType
from kga.report_models import RequirementCoverageStatus
from test_gap_reporting import _evaluation, _model


def _inputs(status=AnswerabilityStatus.UNANSWERED, *, warnings=None):
    model = _model()
    evaluation = _evaluation(model.questions[0], status)
    coverage, gaps, summary = build_report(model, [evaluation])
    manifest = make_manifest(
        model, coverage, gaps, summary, {"fixture": "a" * 64}, warnings or [],
        strict=False, evaluation_run=None,
    )
    return ConsolidationInputs(gaps, coverage, summary, manifest, {"knowledge-gaps.json": "b" * 64})


def _with_duplicate(inputs):
    duplicate = inputs.gaps[0].model_copy(
        update={"gap_id": "gap_" + "f" * 24}, deep=True
    )
    summary = inputs.summary.model_copy(deep=True)
    summary.gap_counts_by_type[duplicate.gap_type] += 1
    manifest = inputs.manifest.model_copy(update={"gap_count": 2}, deep=True)
    return ConsolidationInputs(
        [duplicate, inputs.gaps[0]], inputs.coverage, summary, manifest,
        inputs.fingerprints,
    )


def test_equivalent_findings_for_same_formal_requirement_are_consolidated():
    inputs = _with_duplicate(_inputs())
    inventory, plan, summary = consolidate_gaps(inputs)
    gap = inventory.consolidated_gaps[0]
    assert len(inventory.consolidated_gaps) == 1
    assert gap.consolidation_method is ConsolidationMethod.SAME_FORMAL_REQUIREMENT
    assert gap.source_gap_ids == sorted(item.gap_id for item in inputs.gaps)
    assert summary.merged_gap_count == 1
    assert len(plan.remediations) == 1


def test_same_dimension_but_different_subjects_remain_separate():
    inputs = _inputs()
    first = inputs.gaps[0]
    original_coverage = next(
        item for item in inputs.coverage if item.requirement_id == first.requirement_id
    )
    second_requirement = original_coverage.model_copy(update={
        "requirement_id": "kr_" + "e" * 24,
        "concept_id": "concept_" + "e" * 24,
        "question_ids": ["q_" + "e" * 16],
    }, deep=True)
    inputs.coverage.append(second_requirement)
    second = first.model_copy(update={
        "gap_id": "gap_" + "e" * 24,
        "requirement_id": second_requirement.requirement_id,
        "concept_id": second_requirement.concept_id,
        "requirement_type": second_requirement.requirement_type,
        "question_ids": second_requirement.question_ids,
        "questions": ["A different missing subject?"],
    }, deep=True)
    summary = inputs.summary.model_copy(deep=True)
    summary.counts.total_requirements += 1
    summary.gap_counts_by_type[first.gap_type] += 1
    manifest = inputs.manifest.model_copy(update={"gap_count": 2, "requirement_count": len(inputs.coverage)}, deep=True)
    changed = ConsolidationInputs([first, second], inputs.coverage, summary, manifest, inputs.fingerprints)
    inventory, _, _ = consolidate_gaps(changed)
    assert len(inventory.consolidated_gaps) == 2


def test_same_concept_with_different_missing_knowledge_remains_separate():
    model = _model()
    questions = [
        item for item in model.questions if item.dimension.value == "relationships"
    ]
    evaluations = [_evaluation(item, AnswerabilityStatus.UNANSWERED) for item in questions]
    coverage, gaps, summary = build_report(model, evaluations)
    manifest = make_manifest(model, coverage, gaps, summary, {}, [], strict=False, evaluation_run=None)
    inputs = ConsolidationInputs(gaps, coverage, summary, manifest, {})
    inventory, _, _ = consolidate_gaps(inputs)
    selected = [item for item in inventory.consolidated_gaps if item.knowledge_dimension.value == "relationships"]
    assert len(selected) == 2
    assert len({item.requirement_type for item in selected}) == 2


def test_ambiguous_concept_equivalence_does_not_merge():
    inputs = _inputs()
    first = inputs.gaps[0]
    original_coverage = next(
        item for item in inputs.coverage if item.requirement_id == first.requirement_id
    )
    coverage = original_coverage.model_copy(update={
        "requirement_id": "kr_" + "d" * 24,
        "concept_id": "concept_" + "d" * 24,
        "question_ids": ["q_" + "d" * 16],
        "evaluation_status": RequirementCoverageStatus.UNANSWERED,
    }, deep=True)
    inputs.coverage.append(coverage)
    second = first.model_copy(update={
        "gap_id": "gap_" + "c" * 24,
        "requirement_id": coverage.requirement_id,
        "concept_id": coverage.concept_id,
        "question_ids": coverage.question_ids,
        "requirement_type": coverage.requirement_type,
    }, deep=True)
    summary = inputs.summary.model_copy(deep=True)
    summary.counts.total_requirements += 1
    summary.gap_counts_by_type[first.gap_type] += 1
    manifest = inputs.manifest.model_copy(update={"gap_count": 2, "requirement_count": len(inputs.coverage)}, deep=True)
    changed = ConsolidationInputs([first, second], inputs.coverage, summary, manifest, inputs.fingerprints)
    inventory, _, _ = consolidate_gaps(changed)
    assert len(inventory.consolidated_gaps) == 2


def test_singleton_is_valid_and_every_source_gap_is_represented_once():
    inputs = _inputs()
    inventory, _, summary = consolidate_gaps(inputs)
    assert inventory.consolidated_gaps[0].consolidation_method is ConsolidationMethod.SINGLETON
    represented = [item for gap in inventory.consolidated_gaps for item in gap.source_gap_ids]
    assert represented == [inputs.gaps[0].gap_id]
    assert summary.singleton_gap_count == 1


def test_stable_ids_and_outputs_do_not_depend_on_input_order(tmp_path):
    inputs = _with_duplicate(_inputs())
    first = consolidate_gaps(inputs)
    inputs.gaps.reverse()
    second = consolidate_gaps(inputs)
    assert first == second
    one, two = tmp_path / "one", tmp_path / "two"
    export_consolidation(one, *first)
    export_consolidation(two, *second)
    for name in ["consolidated-gaps.json", "remediation-plan.json", "remediation-plan.md", "consolidation-summary.json"]:
        assert (one / name).read_bytes() == (two / name).read_bytes()


def test_remediation_requests_information_without_inventing_answers():
    inventory, plan, _ = consolidate_gaps(_inputs())
    remediation = plan.remediations[0]
    gap = inventory.consolidated_gaps[0]
    assert remediation.information_to_collect == gap.questions
    assert "Platform team owns" not in remediation.recommended_action
    assert remediation.requires_human_validation is True
    assert "no business criticality was inferred" in remediation.priority_rationale


def test_broken_requirement_reference_is_explicit_and_cli_writes_nothing(tmp_path):
    inputs = _inputs()
    report = tmp_path / "report"
    export_report(report, inputs.coverage, inputs.gaps, inputs.summary, inputs.manifest)
    path = report / "knowledge-gaps.json"
    values = json.loads(path.read_text())
    values[0]["requirement_id"] = "kr_" + "0" * 24
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ConsolidationValidationError, match="missing requirement"):
        load_consolidation_inputs(report)
    output = tmp_path / "output"
    assert main(["--report", str(report), "--output", str(output)]) == 1
    assert not output.exists()


def test_consolidation_cli_writes_expected_outputs(tmp_path):
    inputs = _inputs()
    report = tmp_path / "report"
    output = tmp_path / "output"
    export_report(report, inputs.coverage, inputs.gaps, inputs.summary, inputs.manifest)
    assert main(["--report", str(report), "--output", str(output)]) == 0
    assert {item.name for item in output.iterdir()} == {
        "consolidated-gaps.json", "remediation-plan.json",
        "remediation-plan.md", "consolidation-summary.json",
    }


def test_legacy_provenance_warning_is_preserved():
    warning = "Evaluation run provenance is unavailable; legacy evaluation input accepted"
    _, _, summary = consolidate_gaps(_inputs(warnings=[warning]))
    assert summary.validation_warnings == [warning]


def test_different_gap_semantics_for_same_requirement_are_not_merged():
    inputs = _inputs()
    first = inputs.gaps[0]
    second = first.model_copy(update={
        "gap_id": "gap_" + "b" * 24,
        "gap_type": GapType.UNCERTAIN_COVERAGE,
        "evaluation_status": AnswerabilityStatus.UNCERTAIN,
    }, deep=True)
    # This contradictory duplicate cannot match one requirement-coverage record.
    inputs.gaps = [first, second]
    inputs.manifest = inputs.manifest.model_copy(update={"gap_count": 2})
    inputs.summary.gap_counts_by_type[first.gap_type] = 1
    inputs.summary.gap_counts_by_type[second.gap_type] = 1
    with pytest.raises(ConsolidationValidationError, match="evaluation_status"):
        consolidate_gaps(inputs)
