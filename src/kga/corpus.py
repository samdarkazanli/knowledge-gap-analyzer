"""Load supported documents from a local corpus."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from kga.models import CorpusDocument, DocumentFormat


SUPPORTED_FORMATS = {
    ".md": DocumentFormat.MARKDOWN,
    ".txt": DocumentFormat.TEXT,
    ".adoc": DocumentFormat.ASCIIDOC,
    ".asciidoc": DocumentFormat.ASCIIDOC,
}


class CorpusLoadError(ValueError):
    """Raised when a corpus path or document cannot be loaded safely."""


def load_corpus(path: str | Path) -> list[CorpusDocument]:
    """Recursively load supported UTF-8 documents in stable path order.

    If ``path`` names a file, its source path is its filename. For a directory,
    source paths are relative to that directory. Unsupported files are ignored.
    """

    corpus_path = Path(path)
    if not corpus_path.exists():
        raise CorpusLoadError(f"Corpus path does not exist: {corpus_path}")

    if corpus_path.is_file():
        candidates = [corpus_path]
        root = corpus_path.parent
    elif corpus_path.is_dir():
        candidates = sorted(
            (item for item in corpus_path.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(corpus_path).as_posix(),
        )
        root = corpus_path
    else:
        raise CorpusLoadError(f"Corpus path is not a regular file or directory: {corpus_path}")

    documents: list[CorpusDocument] = []
    for document_path in candidates:
        document_format = SUPPORTED_FORMATS.get(document_path.suffix.lower())
        if document_format is None:
            continue
        try:
            content = document_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            relative_path = document_path.relative_to(root).as_posix()
            raise CorpusLoadError(f"Document is not valid UTF-8: {relative_path}") from exc
        except OSError as exc:
            relative_path = document_path.relative_to(root).as_posix()
            raise CorpusLoadError(f"Could not read document: {relative_path}") from exc

        documents.append(
            CorpusDocument(
                source_path=document_path.relative_to(root).as_posix(),
                format=document_format,
                content=content,
                sha256=sha256(content.encode("utf-8")).hexdigest(),
            )
        )
    return documents
