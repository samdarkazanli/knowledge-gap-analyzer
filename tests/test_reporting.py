import csv
import json

from kga.models import ConceptCategory
from kga.reporting import build_summary, export_reports
from test_extraction import make_concept


def test_builds_statistics_and_exports_reports(tmp_path):
    company_concept = make_concept()
    general_concept = make_concept(name="Kafka", normalized_name="kafka")
    general_concept.category = ConceptCategory.GENERAL_TECHNICAL
    concepts = [company_concept, general_concept]
    summary = build_summary(
        concepts,
        documents_processed=3,
        passages_processed=4,
        api_calls=4,
    )

    export_reports(tmp_path, concepts, summary)

    concepts_json = json.loads((tmp_path / "concepts.json").read_text())
    summary_json = json.loads((tmp_path / "execution-summary.json").read_text())
    with (tmp_path / "concepts.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(concepts_json) == 2
    assert summary_json == {
        "documents_processed": 3,
        "passages_processed": 4,
        "api_calls": 4,
        "concepts_discovered": 2,
        "concepts_by_category": {
            "general_technical": 1,
            "company_specific": 1,
            "uncertain": 0,
        },
    }
    assert [row["name"] for row in rows] == ["PGW", "Kafka"]
