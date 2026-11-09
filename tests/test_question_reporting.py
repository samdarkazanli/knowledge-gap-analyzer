import csv
import json

import pytest

from kga.models import ConceptType
from kga.question_generators import TemplateQuestionGenerator, generate_questions
from kga.question_reporting import (
    ConceptInventoryError,
    export_question_reports,
    load_concept_inventory,
)
from test_question_generators import make_concept


def test_loads_inventory_and_exports_json_and_csv_with_source_references(tmp_path):
    concept = make_concept(
        "orders-primary",
        ConceptType.IDENTIFIER,
        source_path="systems/orders.md",
        line=7,
    )
    inventory = tmp_path / "concepts.json"
    inventory.write_text(
        json.dumps([concept.model_dump(mode="json")]), encoding="utf-8"
    )

    concepts = load_concept_inventory(inventory)
    questions = generate_questions(concepts, TemplateQuestionGenerator())
    export_question_reports(tmp_path / "reports", questions)

    json_report = json.loads(
        (tmp_path / "reports" / "questions.json").read_text(encoding="utf-8")
    )
    with (tmp_path / "reports" / "questions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_report = list(csv.DictReader(handle))

    assert len(json_report) == 2
    assert json_report[0]["concept_normalized_name"] == "orders-primary"
    assert json_report[0]["evidence"][0] == {
        "source_path": "systems/orders.md",
        "start_line": 7,
        "end_line": 7,
        "text": "orders-primary",
    }
    assert len(csv_report) == 2
    assert csv_report[0]["source_path"] == "systems/orders.md"
    assert csv_report[0]["concept_category"] == "company_specific"


def test_rejects_invalid_inventory(tmp_path):
    inventory = tmp_path / "concepts.json"
    inventory.write_text('{"not": "a concept list"}', encoding="utf-8")

    with pytest.raises(ConceptInventoryError, match="expected schema"):
        load_concept_inventory(inventory)


def test_loads_v013_inventory_without_question_fields(tmp_path):
    concept = make_concept("legacy-service", ConceptType.APPLICATION)
    legacy_payload = concept.model_dump(mode="json")
    assert "question_id" not in legacy_payload
    inventory = tmp_path / "concepts-v013.json"
    inventory.write_text(json.dumps([legacy_payload]), encoding="utf-8")

    loaded = load_concept_inventory(inventory)
    questions = generate_questions(loaded, TemplateQuestionGenerator())

    assert loaded == [concept]
    assert questions
    assert all(item.concept_name == "legacy-service" for item in questions)
