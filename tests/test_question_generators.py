from kga.models import (
    Concept,
    ConceptCategory,
    ConceptType,
    QuestionDimension,
    SourcePassage,
)
from kga.question_generators import TemplateQuestionGenerator, generate_questions
from kga.question_generators import ProcedureKind, classify_procedure


def make_concept(
    name: str,
    concept_type: ConceptType,
    *,
    category: ConceptCategory = ConceptCategory.COMPANY_SPECIFIC,
    source_path: str = "systems/internal.txt",
    line: int = 1,
    description: str | None = None,
) -> Concept:
    return Concept(
        name=name,
        normalized_name=name.casefold(),
        category=category,
        concept_type=concept_type,
        description=description,
        evidence=[
            SourcePassage(
                source_path=source_path,
                start_line=line,
                end_line=line,
                text=name,
            )
        ],
    )


def test_generates_type_specific_questions_for_supported_company_concepts():
    concepts = [
        make_concept("Atlas", ConceptType.APPLICATION),
        make_concept("orders-primary", ConceptType.IDENTIFIER),
        make_concept("INC-204", ConceptType.INCIDENT),
        make_concept("retry-window", ConceptType.CONFIGURATION),
        make_concept("regional failover", ConceptType.PROCEDURE),
    ]

    questions = generate_questions(concepts, TemplateQuestionGenerator())

    dimensions_by_concept = {
        concept.name: {q.dimension for q in questions if q.concept_name == concept.name}
        for concept in concepts
    }
    assert dimensions_by_concept["Atlas"] == {
        QuestionDimension.IDENTIFICATION,
        QuestionDimension.PURPOSE,
        QuestionDimension.RELATIONSHIPS,
        QuestionDimension.OPERATIONS,
        QuestionDimension.TROUBLESHOOTING,
    }
    assert dimensions_by_concept["orders-primary"] == {
        QuestionDimension.IDENTIFICATION,
        QuestionDimension.RELATIONSHIPS,
    }
    assert dimensions_by_concept["INC-204"] == {
        QuestionDimension.IDENTIFICATION,
        QuestionDimension.IMPACT,
        QuestionDimension.CAUSE,
        QuestionDimension.DIAGNOSIS,
        QuestionDimension.RESOLUTION,
    }
    assert dimensions_by_concept["retry-window"] == {
        QuestionDimension.PURPOSE,
        QuestionDimension.OPERATIONS,
        QuestionDimension.TROUBLESHOOTING,
    }
    assert dimensions_by_concept["regional failover"] == {
        QuestionDimension.PURPOSE,
        QuestionDimension.OPERATIONS,
    }


def test_incident_questions_are_atomic():
    questions = generate_questions(
        [make_concept("INC-001", ConceptType.INCIDENT)],
        TemplateQuestionGenerator(),
    )

    assert [(item.dimension, item.question) for item in questions] == [
        (QuestionDimension.IDENTIFICATION, "What happened in INC-001?"),
        (QuestionDimension.IMPACT, "What impact did INC-001 have?"),
        (QuestionDimension.CAUSE, "What was the confirmed cause of INC-001?"),
        (QuestionDimension.DIAGNOSIS, "How was INC-001 diagnosed?"),
        (QuestionDimension.RESOLUTION, "How was INC-001 resolved?"),
    ]


def test_named_runbook_generates_document_questions():
    concept = make_concept(
        "Elasticsearch ILM Rollover Runbook", ConceptType.PROCEDURE
    )

    questions = generate_questions([concept], TemplateQuestionGenerator())

    assert classify_procedure(concept) is ProcedureKind.NAMED_PROCEDURE
    assert "What steps are documented in Elasticsearch ILM Rollover Runbook?" in {
        item.question for item in questions
    }


def test_operational_instruction_generates_a_natural_rationale_question():
    concept = make_concept(
        "Start ILM problems with `_ilm/explain`",
        ConceptType.PROCEDURE,
        description="The troubleshooting practice stated in the engineering notes.",
    )

    questions = generate_questions([concept], TemplateQuestionGenerator())

    assert classify_procedure(concept) is ProcedureKind.OPERATIONAL_INSTRUCTION
    assert [item.question for item in questions] == [
        "Why is starting ILM problems with `_ilm/explain` recommended?"
    ]
    assert all("performing Start" not in item.question for item in questions)


def test_remediation_action_focuses_on_prerequisites_and_outcome():
    concept = make_concept(
        "Removal of old test indices after validation",
        ConceptType.PROCEDURE,
        description="Incident response action performed after validation.",
    )

    questions = generate_questions([concept], TemplateQuestionGenerator())

    assert classify_procedure(concept) is ProcedureKind.REMEDIATION_ACTION
    assert [item.question for item in questions] == [
        "What prerequisites must be satisfied before removing old test indices?",
        "What outcome should removing old test indices produce?",
    ]


def test_safety_recommendation_explains_why_an_action_is_avoided():
    concept = make_concept(
        "Avoid deleting indices after rollover failure", ConceptType.PROCEDURE
    )

    questions = generate_questions([concept], TemplateQuestionGenerator())

    assert classify_procedure(concept) is ProcedureKind.SAFETY_RECOMMENDATION
    assert [item.question for item in questions] == [
        "Why should deleting indices after rollover failure be avoided?"
    ]


def test_generated_questions_reject_known_compound_forms():
    concepts = [
        make_concept("Atlas", ConceptType.APPLICATION),
        make_concept("INC-001", ConceptType.INCIDENT),
        make_concept("retry-window", ConceptType.CONFIGURATION),
        make_concept("Regional Failover Runbook", ConceptType.PROCEDURE),
    ]

    questions = generate_questions(concepts, TemplateQuestionGenerator())

    forbidden = (
        "cause, diagnostic path, and resolution",
        "prerequisites and steps",
        "operated, monitored, and maintained",
        "what happened in inc-001, and what was its impact",
    )
    assert all(
        phrase not in question.question.casefold()
        for question in questions
        for phrase in forbidden
    )


def test_suppresses_inappropriate_questions_instead_of_filling_a_quota():
    concepts = [
        make_concept(
            "HTTP",
            ConceptType.ACRONYM,
            category=ConceptCategory.GENERAL_TECHNICAL,
        ),
        make_concept(
            "possibly-internal",
            ConceptType.APPLICATION,
            category=ConceptCategory.UNCERTAIN,
        ),
        make_concept("internal acronym", ConceptType.ACRONYM),
        make_concept("miscellaneous note", ConceptType.OTHER),
        make_concept("operational state", ConceptType.OPERATIONAL_CONCEPT),
    ]

    assert generate_questions(concepts, TemplateQuestionGenerator()) == []


def test_questions_preserve_concept_identity_and_all_source_evidence():
    concept = make_concept(
        "Atlas", ConceptType.APPLICATION, source_path="apps/atlas.md", line=4
    )
    concept.evidence.append(
        SourcePassage(
            source_path="runbooks/atlas.txt",
            start_line=10,
            end_line=11,
            text="Atlas on-call procedure",
        )
    )

    questions = generate_questions([concept], TemplateQuestionGenerator())

    assert questions
    for question in questions:
        assert question.concept_name == "Atlas"
        assert question.concept_normalized_name == "atlas"
        assert question.concept_category is ConceptCategory.COMPANY_SPECIFIC
        assert question.concept_type is ConceptType.APPLICATION
        assert question.evidence == concept.evidence
        assert question.evidence is not concept.evidence


def test_deduplicates_questions_and_merges_distinct_evidence():
    first = make_concept("Atlas", ConceptType.IDENTIFIER, source_path="one.txt")
    duplicate = make_concept(
        "ATLAS", ConceptType.IDENTIFIER, source_path="two.txt", line=2
    )
    duplicate.normalized_name = "atlas"

    questions = generate_questions([first, duplicate], TemplateQuestionGenerator())

    assert len(questions) == 2
    assert all(len(question.evidence) == 2 for question in questions)
    assert all(
        {item.source_path for item in question.evidence} == {"one.txt", "two.txt"}
        for question in questions
    )


def test_question_ids_are_stable_and_different_concepts_are_not_merged():
    first = make_concept("Atlas", ConceptType.IDENTIFIER)
    second = make_concept("Atlas", ConceptType.IDENTIFIER)
    second.normalized_name = "atlas-service"
    generator = TemplateQuestionGenerator()

    first_run = generate_questions([first], generator)
    repeated_run = generate_questions([first], generator)
    separate_concepts = generate_questions([first, second], generator)

    assert [item.question_id for item in first_run] == [
        item.question_id for item in repeated_run
    ]
    assert len(separate_concepts) == 4
    assert len({item.question_id for item in separate_concepts}) == 4
