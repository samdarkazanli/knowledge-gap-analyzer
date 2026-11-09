"""File-based declaration and corpus validation of documentation changes."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from pydantic import ValidationError

from kga.corpus import SUPPORTED_FORMATS, CorpusLoadError, load_corpus
from kga.documentation_tasks import TaskSet, TaskValidationError, load_task_set
from kga.formal_ids import corpus_snapshot_id, stable_id
from kga.remediation_models import (
    ChangeManifest, ChangeValidationResult, DocumentationChange,
    DocumentationChangeInventory, DocumentationChangeStatus,
    VerifiedChangedFile, VerifiedDocumentationChange,
)

CHANGE_SCHEMA_VERSION = "0.2.5"


class ChangeValidationError(ValueError):
    pass


class ChangeSet:
    def __init__(self, inventory: DocumentationChangeInventory, manifest: ChangeManifest,
                 fingerprints: dict[str, str]) -> None:
        self.inventory = inventory
        self.manifest = manifest
        self.fingerprints = fingerprints


def initialize_changes(tasks: TaskSet, output_directory: str | Path) -> ChangeManifest:
    output = Path(output_directory)
    if output.exists():
        raise ChangeValidationError(f"Change workspace already exists; refusing to overwrite: {output}")
    changes = [DocumentationChange(change_id=stable_id("change", {"task_id": task.task_id}), task_id=task.task_id)
               for task in sorted(tasks.inventory.tasks, key=lambda item: item.task_id)]
    inventory = DocumentationChangeInventory(schema_version=CHANGE_SCHEMA_VERSION, changes=changes)
    identity = stable_id("changeset", {"task_set_identity": tasks.manifest.task_set_identity,
                                       "change_ids": [item.change_id for item in changes]})
    manifest = ChangeManifest(
        change_schema_version=CHANGE_SCHEMA_VERSION, change_set_identity=identity,
        task_set_identity=tasks.manifest.task_set_identity,
        task_artifact_fingerprints=tasks.fingerprints, change_count=len(changes),
        initialized_at=datetime.now(timezone.utc),
    )
    _create_directory(output, {
        "documentation-changes.json": _json_bytes(inventory),
        "change-manifest.json": _json_bytes(manifest),
        "change-guide.md": _guide().encode("utf-8"),
    })
    return manifest


def load_change_set(directory: str | Path) -> ChangeSet:
    root = Path(directory)
    required = ["documentation-changes.json", "change-manifest.json", "change-guide.md"]
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise ChangeValidationError("; ".join(f"Missing change artifact: {item}" for item in missing))
    try:
        inventory = DocumentationChangeInventory.model_validate(json.loads((root / "documentation-changes.json").read_text(encoding="utf-8")))
        manifest = ChangeManifest.model_validate(json.loads((root / "change-manifest.json").read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ChangeValidationError(f"Invalid change artifact: {exc}") from exc
    fingerprints = {name: sha256((root / name).read_bytes()).hexdigest() for name in sorted(required)}
    return ChangeSet(inventory, manifest, fingerprints)


def validate_changes(tasks: TaskSet, changes: ChangeSet, corpus_directory: str | Path) -> ChangeValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if changes.inventory.schema_version != CHANGE_SCHEMA_VERSION or changes.manifest.change_schema_version != CHANGE_SCHEMA_VERSION:
        errors.append("Unsupported or inconsistent change schema version")
    if changes.manifest.task_set_identity != tasks.manifest.task_set_identity:
        errors.append("Change workspace task-set identity does not match tasks")
    if changes.manifest.task_artifact_fingerprints != tasks.fingerprints:
        errors.append("Change workspace task artifact fingerprints do not match tasks")
    task_ids = {item.task_id for item in tasks.inventory.tasks}
    records: dict[str, DocumentationChange] = {}
    change_ids: set[str] = set()
    for item in changes.inventory.changes:
        if item.task_id in records:
            errors.append(f"Duplicate change record for task {item.task_id}")
        if item.change_id in change_ids:
            errors.append(f"Duplicate change ID {item.change_id}")
        records[item.task_id] = item
        change_ids.add(item.change_id)
        if item.task_id not in task_ids:
            errors.append(f"Change {item.change_id} references unknown task {item.task_id}")
        if item.change_id != stable_id("change", {"task_id": item.task_id}):
            errors.append(f"Change ID does not match task {item.task_id}")
    missing = sorted(task_ids - set(records))
    if missing:
        errors.append("Missing change records for tasks: " + ", ".join(missing))
    if changes.manifest.change_count != len(changes.inventory.changes):
        errors.append("Change manifest count does not match inventory")
    expected_set = stable_id("changeset", {"task_set_identity": tasks.manifest.task_set_identity,
                                           "change_ids": sorted(change_ids)})
    if changes.manifest.change_set_identity != expected_set:
        errors.append("Change-set identity does not match change records")
    try:
        documents = load_corpus(corpus_directory)
    except CorpusLoadError as exc:
        errors.append(str(exc)); documents = []
    corpus_id = corpus_snapshot_id(documents) if documents else None
    corpus_root = Path(corpus_directory).resolve()
    verified: list[VerifiedDocumentationChange] = []
    for item in sorted(changes.inventory.changes, key=lambda value: value.change_id):
        files: list[VerifiedChangedFile] = []
        files_valid = item.status is DocumentationChangeStatus.DECLARED
        if item.status is DocumentationChangeStatus.PENDING:
            warnings.append(f"Change for task {item.task_id} is pending")
        for declared in item.changed_files:
            candidate = (corpus_root / declared.source_path).resolve()
            try:
                candidate.relative_to(corpus_root)
            except ValueError:
                errors.append(f"Changed file escapes corpus root: {declared.source_path}"); files_valid = False; continue
            if candidate.suffix.lower() not in SUPPORTED_FORMATS:
                errors.append(f"Changed file is not a supported corpus document: {declared.source_path}"); files_valid = False
            elif not candidate.is_file():
                errors.append(f"Changed file does not exist in updated corpus: {declared.source_path}"); files_valid = False
            else:
                files.append(VerifiedChangedFile(source_path=declared.source_path, sha256=sha256(candidate.read_bytes()).hexdigest()))
        verified.append(VerifiedDocumentationChange(
            change_id=item.change_id, task_id=item.task_id, declared_status=item.status,
            files_verified=files_valid, changed_files=files,
        ))
    return ChangeValidationResult(
        schema_version=CHANGE_SCHEMA_VERSION, validation_passed=not errors,
        validation_errors=errors, validation_warnings=warnings,
        corpus_snapshot_id=corpus_id, verified_changes=verified,
    )


def load_and_validate(tasks_directory: str | Path, changes_directory: str | Path,
                      corpus_directory: str | Path) -> tuple[TaskSet, ChangeSet, ChangeValidationResult]:
    try:
        tasks = load_task_set(tasks_directory)
    except TaskValidationError as exc:
        raise ChangeValidationError(str(exc)) from exc
    changes = load_change_set(changes_directory)
    return tasks, changes, validate_changes(tasks, changes, corpus_directory)


def _guide() -> str:
    return """# Documentation Change Guide

Each task has one pending change record. To declare documentation work, set `status` to `declared`, list corpus-relative POSIX paths in `changed_files`, and provide `contributor`, timezone-aware `recorded_at`, and `description`.

A declaration does not complete the task or resolve its knowledge gap. `kga-changes-validate` verifies only that declared paths are safe, exist within the supplied updated corpus, and have reproducible content fingerprints. Knowledge resolution is determined separately by `kga-verify` from answerability evaluations.
"""


def _create_directory(output: Path, payloads: dict[str, bytes]) -> None:
    if output.exists():
        raise ChangeValidationError(f"Change workspace already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, payload in payloads.items(): (temp / name).write_bytes(payload)
        os.replace(temp, output)
    finally:
        if temp.exists():
            for item in temp.iterdir(): item.unlink()
            temp.rmdir()


def _json_bytes(value) -> bytes:
    return (json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
