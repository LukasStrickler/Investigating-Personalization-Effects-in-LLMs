from __future__ import annotations

from importlib import import_module
from typing import Any

csv_schema: Any = import_module("inference.experiments.csv_schema")


def test_json_prompt_and_response_round_trip() -> None:
    prompt = {
        "instruction": "Return JSON",
        "input": 'Comma, quote " and newline\ninside',
        "params": {"temperature": 0.1, "tags": ["a", "b"]},
    }
    response = {
        "result": {"ok": True, "items": [1, 2, 3]},
        "note": "Line1\nLine2",
    }

    serialized_prompt = csv_schema.serialize_prompt_content(prompt)
    serialized_response = csv_schema.serialize_response_content(response)

    assert csv_schema.deserialize_prompt_content(serialized_prompt) == prompt
    assert csv_schema.deserialize_response_content(serialized_response) == response


def test_cell_identity_is_deterministic_from_prompt_and_alias() -> None:
    prompt = {"question": "What is 2+2?"}

    first_id = csv_schema.compute_cell_id(prompt, "openai_gpt4")
    second_id = csv_schema.compute_cell_id(prompt, "openai_gpt4")
    different_alias_id = csv_schema.compute_cell_id(prompt, "anthropic_claude")

    assert first_id == second_id
    assert first_id != different_alias_id


def test_matrix_header_is_prompt_id_then_alias_columns() -> None:
    headers = csv_schema.build_matrix_headers(["model_a", "model_b"])
    assert headers == ["prompt_id", "model_a", "model_b"]


def test_matrix_cell_csv_round_trip() -> None:
    original = csv_schema.MatrixCell(
        status=csv_schema.CellStatus.SUCCESS,
        response={"message": 'hello,"world"\nnext', "structured": {"x": 1}},
    )

    serialized = original.to_csv_cell()
    parsed = csv_schema.MatrixCell.from_csv_cell(serialized)

    assert parsed == original
    assert csv_schema.MatrixCell.from_csv_cell("") is None
