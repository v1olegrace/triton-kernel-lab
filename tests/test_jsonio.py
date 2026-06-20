"""Unit tests for atomic JSON persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from tklab.harness.jsonio import JsonObject, read_json_object, write_json_atomic


def test_json_round_trip(tmp_path: Path) -> None:
    """Persist and load a nested JSON object."""
    path = tmp_path / "result.json"
    payload: JsonObject = {
        "schema_version": 1,
        "rows": [{"size": 128, "gbps": 42.0}],
    }
    write_json_atomic(path, payload)
    assert read_json_object(path) == payload


def test_read_json_object_rejects_top_level_array(tmp_path: Path) -> None:
    """Reject JSON payloads that do not have object semantics."""
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level"):
        read_json_object(path)
