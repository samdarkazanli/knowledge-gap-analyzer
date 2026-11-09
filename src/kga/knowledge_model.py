"""Deterministic construction and validation of KGA's formal knowledge model."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import json
from pathlib import Path

from kga.formal_ids import (
    concept_id,
    concept_inventory_id,
    evidence_id,
    question_set_id,
    stable_id,
)
from kga.formal_models import (
    FormalConcept,
    FormalEvidence,
    FormalKnowledgeModel,
    FormalModelManifest,
    FormalQuestion,
    KnowledgeRequirement,
    RequirementType,
)
from kga.models import Concept, ConceptType, KnowledgeQuestion, QuestionDimension


MODEL_SCHEMA_VERSION = "0.2.3"


class ModelIntegrityError(ValueError):
    """Raised with actionable diagnostics for broken formal references."""

    def __init__(self, diagnostics: Sequence[str]) -> None:
        self.diagnostics = list(diagnostics)
        super().__init__("; ".join(self.diagnostics))


def build_formal_knowledge_model(
    concepts: Sequence[Concept], questions: Sequence[KnowledgeQuestion]
) -> FormalKnowledgeModel:
    """Project legacy inventories into stable formal entities and validate them."""

    _reject_duplicate_question_ids(questions)
    formal_concepts, formal_evidence, concept_lookup = _formalize_concepts(concepts)
    grouped: dict[
        tuple[str, QuestionDimension, RequirementType, str],
        list[tuple[KnowledgeQuestion, str]],
    ] = defaultdict(list)

    for question in questions:
        matching = concept_lookup.get(_concept_key_from_question(question), [])
        if not matching:
            raise ModelIntegrityError(
                [
                    f"Question {question.question_id} references no concept matching "
                    f"{question.concept_normalized_name!r}, "
                    f"{question.concept_category.value}, {question.concept_type.value}"
                ]
            )
        if len(matching) > 1:
            ids = ", ".join(item.concept_id for item in matching)
            raise ModelIntegrityError(
                [f"Question {question.question_id} has ambiguous concept references: {ids}"]
            )
        concept = matching[0]
        _validate_question_evidence(question, concept)
        requirement_type = classify_requirement_type(question)
        semantic_key = _normalized_question(question.question)
        grouped[
            (concept.concept_id, question.dimension, requirement_type, semantic_key)
        ].append((question, concept.concept_id))

    requirements: list[KnowledgeRequirement] = []
    question_requirement: dict[str, str] = {}
    for key in sorted(
        grouped,
        key=lambda item: (item[0], item[1].value, item[2].value, item[3]),
    ):
        concept_ref, dimension, requirement_type, semantic_key = key
        group = grouped[key]
        question_ids = sorted(item.question_id for item, _ in group)
        requirement_ref = stable_id(
            "kr",
            {
                "concept_id": concept_ref,
                "dimension": dimension.value,
                "requirement_type": requirement_type.value,
                "semantic_key": semantic_key,
            },
        )
        source_evidence = _unique_evidence(
            evidence for question, _ in group for evidence in question.evidence
        )
        requirement = KnowledgeRequirement(
            requirement_id=requirement_ref,
            concept_id=concept_ref,
            dimension=dimension,
            requirement_type=requirement_type,
            description=f"Knowledge needed to answer: {group[0][0].question}",
            question_ids=question_ids,
            source_evidence=source_evidence,
        )
        requirements.append(requirement)
        for question_id_value in question_ids:
            if question_id_value in question_requirement:
                raise ModelIntegrityError(
                    [f"Question {question_id_value} belongs to multiple requirements"]
                )
            question_requirement[question_id_value] = requirement_ref

    enriched_questions = [
        FormalQuestion(
            **question.model_dump(),
            concept_id=_single_concept_id(question, concept_lookup),
            requirement_id=question_requirement[question.question_id],
        )
        for question in questions
    ]
    manifest = FormalModelManifest(
        model_schema_version=MODEL_SCHEMA_VERSION,
        concept_inventory_id=concept_inventory_id(concepts),
        question_set_id=question_set_id(questions),
        concept_count=len(formal_concepts),
        question_count=len(enriched_questions),
        requirement_count=len(requirements),
        evidence_count=len(formal_evidence),
        validation_passed=True,
        validation_errors=[],
    )
    model = FormalKnowledgeModel(
        concepts=formal_concepts,
        evidence=formal_evidence,
        questions=enriched_questions,
        requirements=requirements,
        manifest=manifest,
    )
    validate_referential_integrity(model)
    return model


def classify_requirement_type(question: KnowledgeQuestion) -> RequirementType:
    """Conservatively map atomic generated questions to specific knowledge needs."""

    text = _normalized_question(question.question)
    if question.dimension is QuestionDimension.IDENTIFICATION:
        return RequirementType.IDENTIFICATION
    if question.dimension is QuestionDimension.IMPACT:
        return RequirementType.INCIDENT_IMPACT
    if question.dimension is QuestionDimension.CAUSE:
        return RequirementType.INCIDENT_CAUSE
    if question.dimension is QuestionDimension.DIAGNOSIS:
        return RequirementType.INCIDENT_DIAGNOSIS
    if question.dimension is QuestionDimension.RESOLUTION:
        if question.concept_type is ConceptType.INCIDENT:
            return RequirementType.INCIDENT_RESOLUTION
        return RequirementType.RESOLUTION
    if question.dimension is QuestionDimension.TROUBLESHOOTING:
        return RequirementType.TROUBLESHOOTING
    if question.dimension is QuestionDimension.RELATIONSHIPS:
        if " team owns " in f" {text} ":
            return RequirementType.OWNERSHIP
        return RequirementType.DEPENDENCIES
    if question.dimension is QuestionDimension.OPERATIONS:
        if " deployed" in text:
            return RequirementType.DEPLOYMENT
        if " monitored" in text:
            return RequirementType.MONITORING
        if " maintained" in text:
            return RequirementType.MAINTENANCE
        if "values are valid" in text:
            return RequirementType.CONFIGURATION_VALUES
        if "changed safely" in text:
            return RequirementType.CONFIGURATION_CHANGE
        if "prerequisites" in text:
            return RequirementType.PROCEDURE_PREREQUISITES
        if "steps are documented" in text:
            return RequirementType.PROCEDURE_STEPS
        return RequirementType.OTHER
    if question.dimension is QuestionDimension.PURPOSE:
        if text.startswith("why should") and "avoided" in text:
            return RequirementType.SAFETY_RATIONALE
        if text.startswith("why is") and "recommended" in text:
            return RequirementType.PROCEDURE_RATIONALE
        return RequirementType.PURPOSE
    return RequirementType.OTHER


def validate_referential_integrity(model: FormalKnowledgeModel) -> None:
    diagnostics: list[str] = []
    concept_ids = _unique_ids(
        "concept", [item.concept_id for item in model.concepts], diagnostics
    )
    evidence_ids = _unique_ids(
        "evidence", [item.evidence_id for item in model.evidence], diagnostics
    )
    question_ids = _unique_ids(
        "question", [item.question_id for item in model.questions], diagnostics
    )
    requirement_ids = _unique_ids(
        "requirement", [item.requirement_id for item in model.requirements], diagnostics
    )

    for concept in model.concepts:
        for evidence_ref in concept.evidence_ids:
            if evidence_ref not in evidence_ids:
                diagnostics.append(
                    f"Concept {concept.concept_id} references missing evidence {evidence_ref}"
                )
    requirement_membership: dict[str, int] = defaultdict(int)
    for requirement in model.requirements:
        if requirement.concept_id not in concept_ids:
            diagnostics.append(
                f"Requirement {requirement.requirement_id} references missing concept "
                f"{requirement.concept_id}"
            )
        for question_ref in requirement.question_ids:
            requirement_membership[question_ref] += 1
            if question_ref not in question_ids:
                diagnostics.append(
                    f"Requirement {requirement.requirement_id} references missing question "
                    f"{question_ref}"
                )
    for question in model.questions:
        if question.concept_id not in concept_ids:
            diagnostics.append(
                f"Question {question.question_id} references missing concept "
                f"{question.concept_id}"
            )
        if question.requirement_id not in requirement_ids:
            diagnostics.append(
                f"Question {question.question_id} references missing requirement "
                f"{question.requirement_id}"
            )
        if requirement_membership.get(question.question_id, 0) != 1:
            diagnostics.append(
                f"Question {question.question_id} must belong to exactly one requirement"
            )
    if diagnostics:
        raise ModelIntegrityError(diagnostics)


def export_formal_knowledge_model(
    output_directory: str | Path, model: FormalKnowledgeModel
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "knowledge-requirements.json", model.requirements)
    _write_json(output_path / "questions-enriched.json", model.questions)
    _write_json(output_path / "formal-concepts.json", model.concepts)
    _write_json(output_path / "formal-evidence.json", model.evidence)
    _write_json(output_path / "model-manifest.json", model.manifest)


def _formalize_concepts(
    concepts: Sequence[Concept],
) -> tuple[
    list[FormalConcept],
    list[FormalEvidence],
    dict[tuple[str, object, object], list[FormalConcept]],
]:
    formal_concepts: list[FormalConcept] = []
    evidence_by_id: dict[str, FormalEvidence] = {}
    lookup: dict[tuple[str, object, object], list[FormalConcept]] = defaultdict(list)
    seen_concept_ids: set[str] = set()
    for concept in concepts:
        concept_ref = concept_id(concept)
        if concept_ref in seen_concept_ids:
            raise ModelIntegrityError([f"Duplicate concept ID: {concept_ref}"])
        seen_concept_ids.add(concept_ref)
        evidence_refs: list[str] = []
        for evidence in concept.evidence:
            evidence_ref = evidence_id(evidence)
            evidence_refs.append(evidence_ref)
            evidence_by_id.setdefault(
                evidence_ref,
                FormalEvidence(
                    evidence_id=evidence_ref,
                    source_path=evidence.source_path,
                    start_line=evidence.start_line,
                    end_line=evidence.end_line,
                    text=evidence.text,
                ),
            )
        formal = FormalConcept(
            concept_id=concept_ref,
            name=concept.name,
            normalized_name=concept.normalized_name,
            category=concept.category,
            concept_type=concept.concept_type,
            description=concept.description,
            evidence_ids=sorted(evidence_refs),
        )
        formal_concepts.append(formal)
        lookup[_concept_key_from_concept(concept)].append(formal)
    return (
        sorted(formal_concepts, key=lambda item: item.concept_id),
        sorted(evidence_by_id.values(), key=lambda item: item.evidence_id),
        lookup,
    )


def _validate_question_evidence(
    question: KnowledgeQuestion, concept: FormalConcept
) -> None:
    concept_evidence = set(concept.evidence_ids)
    missing = [
        evidence_id(item)
        for item in question.evidence
        if evidence_id(item) not in concept_evidence
    ]
    if missing:
        raise ModelIntegrityError(
            [
                f"Question {question.question_id} contains evidence not linked to "
                f"concept {concept.concept_id}: {', '.join(missing)}"
            ]
        )


def _single_concept_id(question, lookup) -> str:
    return lookup[_concept_key_from_question(question)][0].concept_id


def _concept_key_from_concept(concept: Concept) -> tuple[str, object, object]:
    return (concept.normalized_name, concept.category, concept.concept_type)


def _concept_key_from_question(question: KnowledgeQuestion) -> tuple[str, object, object]:
    return (
        question.concept_normalized_name,
        question.concept_category,
        question.concept_type,
    )


def _reject_duplicate_question_ids(questions: Sequence[KnowledgeQuestion]) -> None:
    diagnostics: list[str] = []
    _unique_ids("question", [item.question_id for item in questions], diagnostics)
    if diagnostics:
        raise ModelIntegrityError(diagnostics)


def _unique_ids(label: str, values: Sequence[str], diagnostics: list[str]) -> set[str]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            diagnostics.append(f"Duplicate {label} ID: {value}")
        seen.add(value)
    return seen


def _unique_evidence(passages) -> list:
    unique = {}
    for item in passages:
        unique.setdefault(evidence_id(item), item.model_copy(deep=True))
    return [unique[key] for key in sorted(unique)]


def _normalized_question(value: str) -> str:
    return " ".join(value.casefold().split())


def _write_json(path: Path, value) -> None:
    if isinstance(value, list):
        payload = [item.model_dump(mode="json") for item in value]
    else:
        payload = value.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
