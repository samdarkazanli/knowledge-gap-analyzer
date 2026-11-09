from hashlib import sha256

import pytest

from kga.corpus import CorpusLoadError, load_corpus
from kga.models import DocumentFormat


def test_loads_supported_documents_recursively_in_stable_order(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "nested" / "a.md").write_text("# Heading\n", encoding="utf-8")
    (tmp_path / "nested" / "b.adoc").write_text("= Title\n", encoding="utf-8")
    (tmp_path / "nested" / "c.asciidoc").write_text("content", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")

    documents = load_corpus(tmp_path)

    assert [document.source_path for document in documents] == [
        "nested/a.md",
        "nested/b.adoc",
        "nested/c.asciidoc",
        "z.txt",
    ]
    assert [document.format for document in documents] == [
        DocumentFormat.MARKDOWN,
        DocumentFormat.ASCIIDOC,
        DocumentFormat.ASCIIDOC,
        DocumentFormat.TEXT,
    ]
    assert documents[0].content == "# Heading\n"
    assert documents[0].sha256 == sha256(b"# Heading\n").hexdigest()


def test_loads_a_single_supported_file(tmp_path):
    source = tmp_path / "notes.TXT"
    source.write_text("notes", encoding="utf-8")

    [document] = load_corpus(source)

    assert document.source_path == "notes.TXT"
    assert document.format is DocumentFormat.TEXT


def test_ignores_unsupported_file(tmp_path):
    source = tmp_path / "data.json"
    source.write_text("{}", encoding="utf-8")

    assert load_corpus(source) == []


def test_rejects_invalid_utf8(tmp_path):
    source = tmp_path / "broken.md"
    source.write_bytes(b"\xff")

    with pytest.raises(CorpusLoadError, match="not valid UTF-8: broken.md"):
        load_corpus(tmp_path)


def test_rejects_missing_path(tmp_path):
    with pytest.raises(CorpusLoadError, match="does not exist"):
        load_corpus(tmp_path / "missing")
