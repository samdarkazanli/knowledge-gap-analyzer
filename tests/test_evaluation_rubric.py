from hashlib import sha256

from kga.evaluation import FakeQuestionEvaluator, evaluate_questions
from kga.models import (
    AnswerabilityStatus,
    ConceptCategory,
    ConceptType,
    CorpusDocument,
    DocumentFormat,
    EvaluationDecision,
    KnowledgeQuestion,
    QuestionDimension,
    SourcePassage,
)


def document(content: str) -> CorpusDocument:
    return CorpusDocument(
        source_path="inventory.txt",
        format=DocumentFormat.TEXT,
        content=content,
        sha256=sha256(content.encode("utf-8")).hexdigest(),
    )


def question(
    question_id: str,
    text: str,
    concept_name: str,
    dimension: QuestionDimension,
) -> KnowledgeQuestion:
    return KnowledgeQuestion(
        question_id=question_id,
        question=text,
        dimension=dimension,
        concept_name=concept_name,
        concept_normalized_name=concept_name.casefold(),
        concept_category=ConceptCategory.COMPANY_SPECIFIC,
        concept_type=ConceptType.APPLICATION,
        evidence=[
            SourcePassage(
                source_path="inventory.txt",
                start_line=1,
                end_line=1,
                text="System:",
            )
        ],
    )


def test_equivalent_associations_are_consistently_partial_not_interactions():
    first = question(
        "q_1000000000000001",
        "Which systems interact with payments?",
        "payments",
        QuestionDimension.RELATIONSHIPS,
    )
    second = question(
        "q_1000000000000002",
        "Which systems interact with customer-search?",
        "customer-search",
        QuestionDimension.RELATIONSHIPS,
    )
    corpus = document(
        "System: Elasticsearch\nApplication: payments\n"
        "System: Elasticsearch\nApplication: customer-search"
    )
    decisions = {
        first.question_id: EvaluationDecision(
            status=AnswerabilityStatus.PARTIALLY_ANSWERED,
            answer=(
                "Elasticsearch is associated with payments, but the corpus does "
                "not explicitly establish an interaction."
            ),
            explanation="Adjacent metadata establishes association only.",
            supporting_evidence=[
                SourcePassage(
                    source_path="inventory.txt",
                    start_line=1,
                    end_line=2,
                    text="System: Elasticsearch\nApplication: payments",
                )
            ],
            confidence=0.95,
        ),
        second.question_id: EvaluationDecision(
            status=AnswerabilityStatus.PARTIALLY_ANSWERED,
            answer=(
                "Elasticsearch is associated with customer-search, but the corpus "
                "does not explicitly establish an interaction."
            ),
            explanation="Adjacent metadata establishes association only.",
            supporting_evidence=[
                SourcePassage(
                    source_path="inventory.txt",
                    start_line=3,
                    end_line=4,
                    text="System: Elasticsearch\nApplication: customer-search",
                )
            ],
            confidence=0.95,
        ),
    }

    results = evaluate_questions(
        [first, second], [corpus], FakeQuestionEvaluator(decisions)
    )

    assert [item.status for item in results] == [
        AnswerabilityStatus.PARTIALLY_ANSWERED,
        AnswerabilityStatus.PARTIALLY_ANSWERED,
    ]
    assert all("does not explicitly establish" in item.answer for item in results)


def test_application_name_alone_does_not_establish_purpose():
    item = question(
        "q_1000000000000003",
        "What purpose does payments serve?",
        "payments",
        QuestionDimension.PURPOSE,
    )
    corpus = document("Application: payments")

    result = evaluate_questions([item], [corpus], FakeQuestionEvaluator())[0]

    assert result.status is AnswerabilityStatus.UNANSWERED
    assert result.answer is None


def test_incident_action_is_only_partial_general_maintenance_knowledge():
    item = question(
        "q_1000000000000004",
        "How is payments maintained?",
        "payments",
        QuestionDimension.OPERATIONS,
    )
    corpus = document("During INC-001, the write alias was corrected.")
    decision = EvaluationDecision(
        status=AnswerabilityStatus.PARTIALLY_ANSWERED,
        answer=(
            "A write alias was corrected during INC-001, but this does not establish "
            "the application's general maintenance process."
        ),
        explanation="The evidence is incident-specific.",
        supporting_evidence=[
            SourcePassage(
                source_path="inventory.txt",
                start_line=1,
                end_line=1,
                text="During INC-001, the write alias was corrected.",
            )
        ],
        confidence=0.9,
    )

    result = evaluate_questions(
        [item], [corpus], FakeQuestionEvaluator({item.question_id: decision})
    )[0]

    assert result.status is AnswerabilityStatus.PARTIALLY_ANSWERED
    assert "general maintenance process" in result.answer


def test_preliminary_and_later_confirmed_causes_are_not_automatically_conflicting():
    item = question(
        "q_1000000000000005",
        "What was the confirmed cause of INC-001?",
        "INC-001",
        QuestionDimension.CAUSE,
    )
    corpus = document(
        "Preliminary hypothesis: disk pressure.\n"
        "Final finding: an alias configuration error caused INC-001."
    )
    decision = EvaluationDecision(
        status=AnswerabilityStatus.ANSWERED,
        answer="The confirmed cause was an alias configuration error.",
        explanation="The final finding supersedes the preliminary hypothesis.",
        supporting_evidence=[
            SourcePassage(
                source_path="inventory.txt",
                start_line=2,
                end_line=2,
                text="Final finding: an alias configuration error caused INC-001.",
            )
        ],
        confidence=0.98,
    )

    result = evaluate_questions(
        [item], [corpus], FakeQuestionEvaluator({item.question_id: decision})
    )[0]

    assert result.status is AnswerabilityStatus.ANSWERED
    assert result.conflicting_evidence == []
