"""Small, safe JSON persistence helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def read_json_object(path: Path) -> JsonObject:
    """Read a JSON object from disk.

    Args:
        path: JSON file to read.

    Returns:
        Parsed top-level object.

    Raises:
        ValueError: If the top-level JSON value is not an object.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a top-level JSON object")
    return cast(JsonObject, payload)


def write_json_atomic(path: Path, payload: JsonObject) -> None:
    """Atomically replace a JSON file.

    A temporary file is created in the destination directory, flushed, and
    moved into place with :func:`os.replace`. Readers therefore never observe
    a partially written benchmark result.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable top-level object.

    Raises:
        OSError: If the temporary file cannot be written or replaced.
        TypeError: If ``payload`` contains a non-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
