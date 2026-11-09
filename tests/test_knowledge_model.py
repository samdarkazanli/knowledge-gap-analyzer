import pytest

from kga.formal_models import RequirementType
from kga.knowledge_model import (
    ModelIntegrityError,
    build_formal_knowledge_model,
    validate_referential_integrity,
)
from kga.models import ConceptType
from kga.question_generators import TemplateQuestionGenerator, generate_questions
from test_question_generators import make_concept


def application_inventory():
    concept = make_concept("Atlas", ConceptType.APPLICATION)
    questions = generate_questions([concept], TemplateQuestionGenerator())
    return concept, questions


def test_builds_formal_entities_and_preserves_question_and_evidence_provenance():
    concept, questions = application_inventory()

    model = build_formal_knowledge_model([concept], questions)

    assert model.manifest.validation_passed is True
    assert model.manifest.concept_count == 1
    assert model.manifest.question_count == len(questions)
    assert model.manifest.requirement_count == len(questions)
    assert {item.question_id for item in model.questions} == {
        item.question_id for item in questions
    }
    assert all(item.concept_id == model.concepts[0].concept_id for item in model.questions)
    assert all(item.requirement_id for item in model.questions)
    assert all(
        requirement.source_evidence == concept.evidence
        for requirement in model.requirements
    )
    validate_referential_integrity(model)


def test_same_dimension_ownership_and_dependencies_are_distinct_requirements():
    concept, questions = application_inventory()
    relationship_questions = [
        item for item in questions if item.dimension.value == "relationships"
    ]

    model = build_formal_knowledge_model([concept], relationship_questions)

    assert len(model.requirements) == 2
    assert {item.requirement_type for item in model.requirements} == {
        RequirementType.OWNERSHIP,
        RequirementType.DEPENDENCIES,
    }


def test_conservative_grouping_merges_exact_equivalence_only():
    concept, questions = application_inventory()
    original = next(item for item in questions if "systems interact" in item.question)
    exact_duplicate = original.model_copy(
        update={"question_id": "q_1111111111111111"}, deep=True
    )
    different_fact = original.model_copy(
        update={
            "question_id": "q_2222222222222222",
            "question": "Which systems depend on Atlas?",
        },
        deep=True,
    )

    model = build_formal_knowledge_model(
        [concept], [original, exact_duplicate, different_fact]
    )

    assert len(model.requirements) == 2
    assert sorted(len(item.question_ids) for item in model.requirements) == [1, 2]
    assert len({item.requirement_id for item in model.questions}) == 2


def test_duplicate_question_identifier_is_rejected():
    concept, questions = application_inventory()

    with pytest.raises(ModelIntegrityError, match="Duplicate question ID"):
        build_formal_knowledge_model([concept], [questions[0], questions[0]])


def test_question_with_missing_concept_is_rejected():
    concept, questions = application_inventory()
    broken = questions[0].model_copy(
        update={"concept_normalized_name": "missing"}, deep=True
    )

    with pytest.raises(ModelIntegrityError, match="references no concept"):
        build_formal_knowledge_model([concept], [broken])


def test_question_with_evidence_not_linked_to_concept_is_rejected():
    concept, questions = application_inventory()
    foreign = questions[0].model_copy(deep=True)
    foreign.evidence[0] = foreign.evidence[0].model_copy(
        update={"source_path": "foreign.txt"}
    )

    with pytest.raises(ModelIntegrityError, match="evidence not linked"):
        build_formal_knowledge_model([concept], [foreign])


def test_broken_requirement_reference_is_rejected():
    concept, questions = application_inventory()
    model = build_formal_knowledge_model([concept], [questions[0]])
    broken_question = model.questions[0].model_copy(
        update={"requirement_id": "kr_ffffffffffffffffffffffff"}
    )
    broken_model = model.model_copy(update={"questions": [broken_question]}, deep=True)

    with pytest.raises(ModelIntegrityError, match="missing requirement"):
        validate_referential_integrity(broken_model)
