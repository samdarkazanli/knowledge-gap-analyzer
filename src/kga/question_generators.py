"""Provider-independent, atomic knowledge-question generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from hashlib import sha256
import re
from typing import Protocol

from kga.models import (
    Concept,
    ConceptCategory,
    ConceptType,
    KnowledgeQuestion,
    QuestionDimension,
)


QuestionTemplate = tuple[QuestionDimension, str]


class QuestionGenerator(Protocol):
    """Generate relevant questions for one extracted concept.

    Future implementations may use an LLM while retaining this contract.
    """

    def generate(self, concept: Concept) -> list[KnowledgeQuestion]:
        """Return questions grounded in the supplied concept and evidence."""
        ...


class ProcedureKind(str, Enum):
    """Generic linguistic forms used to select natural procedure questions."""

    NAMED_PROCEDURE = "named_procedure"
    OPERATIONAL_INSTRUCTION = "operational_instruction"
    REMEDIATION_ACTION = "remediation_action"
    SAFETY_RECOMMENDATION = "safety_recommendation"


TEMPLATES: Mapping[ConceptType, Sequence[QuestionTemplate]] = {
    ConceptType.APPLICATION: (
        (QuestionDimension.IDENTIFICATION, "What is {name}?"),
        (QuestionDimension.PURPOSE, "What purpose does {name} serve?"),
        (QuestionDimension.RELATIONSHIPS, "Which systems interact with {name}?"),
        (QuestionDimension.RELATIONSHIPS, "Which team owns {name}?"),
        (QuestionDimension.OPERATIONS, "How is {name} deployed?"),
        (QuestionDimension.OPERATIONS, "How is {name} monitored?"),
        (QuestionDimension.OPERATIONS, "How is {name} maintained?"),
        (
            QuestionDimension.TROUBLESHOOTING,
            "How should an engineer diagnose a failure involving {name}?",
        ),
    ),
    ConceptType.IDENTIFIER: (
        (QuestionDimension.IDENTIFICATION, "What does {name} identify?"),
        (QuestionDimension.RELATIONSHIPS, "What is {name} associated with?"),
    ),
    ConceptType.INCIDENT: (
        (QuestionDimension.IDENTIFICATION, "What happened in {name}?"),
        (QuestionDimension.IMPACT, "What impact did {name} have?"),
        (QuestionDimension.CAUSE, "What was the confirmed cause of {name}?"),
        (QuestionDimension.DIAGNOSIS, "How was {name} diagnosed?"),
        (QuestionDimension.RESOLUTION, "How was {name} resolved?"),
    ),
    ConceptType.CONFIGURATION: (
        (
            QuestionDimension.PURPOSE,
            "What behavior does the {name} configuration control?",
        ),
        (QuestionDimension.OPERATIONS, "What values are valid for {name}?"),
        (QuestionDimension.OPERATIONS, "How can {name} be changed safely?"),
        (
            QuestionDimension.TROUBLESHOOTING,
            "What symptom indicates that {name} is incorrect?",
        ),
    ),
}


_NAMED_PROCEDURE_MARKERS = re.compile(
    r"\b(runbook|playbook|procedure|workflow|process|guide)\b", re.IGNORECASE
)
_SAFETY_OPENING = re.compile(
    r"^(avoid|never|do not|don't|must not|should not)\b", re.IGNORECASE
)
_REMEDIATION_OPENING = re.compile(
    r"^(removal|deletion|restoration|replacement|restart|rollback|recovery|"
    r"cleanup|clean-up|remediation|repair|correction)\b",
    re.IGNORECASE,
)
_INSTRUCTION_OPENING = re.compile(
    r"^(start|begin|check|use|set|verify|inspect|review|monitor|run|perform|"
    r"create|update|enable|disable|restart|remove|delete|restore|replace|"
    r"retry|validate|confirm|ensure|configure)\b",
    re.IGNORECASE,
)


class TemplateQuestionGenerator:
    """Generate deterministic atomic questions for supported internal concepts."""

    def generate(self, concept: Concept) -> list[KnowledgeQuestion]:
        if concept.category is not ConceptCategory.COMPANY_SPECIFIC:
            return []

        if concept.concept_type is ConceptType.PROCEDURE:
            specifications = _procedure_questions(concept)
        else:
            specifications = [
                (dimension, template.format(name=concept.name))
                for dimension, template in TEMPLATES.get(concept.concept_type, ())
            ]
        return [
            _build_question(concept, dimension, question)
            for dimension, question in specifications
        ]


def classify_procedure(concept: Concept) -> ProcedureKind:
    """Classify a procedure by generic linguistic and metadata cues."""

    name = " ".join(concept.name.split())
    context_parts = [concept.description or ""]
    context_parts.extend(item.text for item in concept.evidence)
    context = " ".join(context_parts)

    if _SAFETY_OPENING.search(name) or re.search(
        r"\b(safety recommendation|safety warning|unsafe|must be avoided)\b",
        context,
        re.IGNORECASE,
    ):
        return ProcedureKind.SAFETY_RECOMMENDATION
    if _REMEDIATION_OPENING.search(name) or re.search(
        r"\b(remediation|response action|resolution action|corrective action)\b",
        context,
        re.IGNORECASE,
    ):
        return ProcedureKind.REMEDIATION_ACTION
    if _NAMED_PROCEDURE_MARKERS.search(name) or re.search(
        r"\b(named procedure|documented procedure|runbook|playbook)\b",
        context,
        re.IGNORECASE,
    ):
        return ProcedureKind.NAMED_PROCEDURE
    if _INSTRUCTION_OPENING.search(name) or re.search(
        r"\b(operational instruction|troubleshooting practice|recommended step)\b",
        context,
        re.IGNORECASE,
    ):
        return ProcedureKind.OPERATIONAL_INSTRUCTION
    return ProcedureKind.NAMED_PROCEDURE


def generate_questions(
    concepts: Sequence[Concept], generator: QuestionGenerator
) -> list[KnowledgeQuestion]:
    """Generate in inventory order and deduplicate only within one concept."""

    questions: list[KnowledgeQuestion] = []
    seen: dict[tuple[str, str], KnowledgeQuestion] = {}
    for concept in concepts:
        for question in generator.generate(concept):
            key = (
                question.concept_normalized_name.casefold().strip(),
                " ".join(question.question.casefold().split()),
            )
            existing = seen.get(key)
            if existing is None:
                seen[key] = question
                questions.append(question)
                continue
            _merge_evidence(existing, question)
    return questions


def _procedure_questions(concept: Concept) -> list[QuestionTemplate]:
    kind = classify_procedure(concept)
    name = concept.name
    if kind is ProcedureKind.NAMED_PROCEDURE:
        return [
            (QuestionDimension.PURPOSE, f"What purpose does {name} serve?"),
            (QuestionDimension.OPERATIONS, f"What steps are documented in {name}?"),
        ]
    if kind is ProcedureKind.OPERATIONAL_INSTRUCTION:
        action = _as_gerund_phrase(name)
        return [(QuestionDimension.PURPOSE, f"Why is {action} recommended?")]
    if kind is ProcedureKind.REMEDIATION_ACTION:
        action = _remediation_action(name)
        return [
            (
                QuestionDimension.OPERATIONS,
                f"What prerequisites must be satisfied before {action}?",
            ),
            (
                QuestionDimension.RESOLUTION,
                f"What outcome should {action} produce?",
            ),
        ]
    action = _safety_action(name)
    return [(QuestionDimension.PURPOSE, f"Why should {action} be avoided?")]


def _as_gerund_phrase(instruction: str) -> str:
    words = instruction.strip().rstrip(".?").split(maxsplit=1)
    if not words:
        return "following the instruction"
    verb = words[0].lower()
    if verb.endswith("ing"):
        gerund = verb
    elif verb.endswith("ie"):
        gerund = verb[:-2] + "ying"
    elif verb.endswith("e") and not verb.endswith("ee"):
        gerund = verb[:-1] + "ing"
    elif verb.endswith("c"):
        gerund = verb + "king"
    else:
        gerund = verb + "ing"
    return " ".join([gerund, *words[1:]])


def _remediation_action(name: str) -> str:
    phrase = re.sub(r"\s+after\s+.+$", "", name.strip(), flags=re.IGNORECASE)
    conversions = (
        (r"^removal of\s+", "removing "),
        (r"^deletion of\s+", "deleting "),
        (r"^restoration of\s+", "restoring "),
        (r"^replacement of\s+", "replacing "),
        (r"^restart of\s+", "restarting "),
        (r"^rollback of\s+", "rolling back "),
        (r"^recovery of\s+", "recovering "),
        (r"^(cleanup|clean-up) of\s+", "cleaning up "),
    )
    for pattern, replacement in conversions:
        converted, count = re.subn(
            pattern, replacement, phrase, count=1, flags=re.IGNORECASE
        )
        if count:
            return converted
    if _INSTRUCTION_OPENING.search(phrase):
        return _as_gerund_phrase(phrase)
    return f"carrying out {phrase}"


def _safety_action(name: str) -> str:
    phrase = _SAFETY_OPENING.sub("", name.strip(), count=1).strip()
    return _as_gerund_phrase(phrase)


def _merge_evidence(existing: KnowledgeQuestion, duplicate: KnowledgeQuestion) -> None:
    evidence_keys = {
        (item.source_path, item.start_line, item.end_line, item.text)
        for item in existing.evidence
    }
    for evidence in duplicate.evidence:
        key = (evidence.source_path, evidence.start_line, evidence.end_line, evidence.text)
        if key not in evidence_keys:
            existing.evidence.append(evidence.model_copy(deep=True))
            evidence_keys.add(key)


def _build_question(
    concept: Concept, dimension: QuestionDimension, question: str
) -> KnowledgeQuestion:
    identity = f"{concept.normalized_name}\0{dimension.value}\0{question}"
    question_id = f"q_{sha256(identity.encode('utf-8')).hexdigest()[:16]}"
    return KnowledgeQuestion(
        question_id=question_id,
        question=question,
        dimension=dimension,
        concept_name=concept.name,
        concept_normalized_name=concept.normalized_name,
        concept_category=concept.category,
        concept_type=concept.concept_type,
        evidence=[item.model_copy(deep=True) for item in concept.evidence],
    )
