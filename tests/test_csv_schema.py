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


def test_matrix_header_is_prompt_id_prompt_metadata_then_alias_columns() -> None:
    headers = csv_schema.build_matrix_headers(["model_a", "model_b"])
    assert headers == ["prompt_id", "prompt", "prompt_metadata", "model_a", "model_b"]


def test_canonical_prompt_spec_normalizes_to_messages() -> None:
    # str -> single user message
    assert csv_schema.canonical_prompt_spec("hello") == {
        "messages": [{"role": "user", "content": "hello"}]
    }
    # dict with system + user -> system then user
    assert csv_schema.canonical_prompt_spec({"system": "S", "user": "U"}) == {
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    }
    # dict with messages only -> unchanged structure
    assert csv_schema.canonical_prompt_spec({"messages": [{"role": "user", "content": "x"}]}) == {
        "messages": [{"role": "user", "content": "x"}]
    }
    # metadata key is excluded from the canonical form
    assert csv_schema.canonical_prompt_spec(
        {"messages": [{"role": "user", "content": "x"}], "metadata": {"history_id": "h1"}}
    ) == {"messages": [{"role": "user", "content": "x"}]}


def test_prompt_metadata_is_excluded_from_prompt_id() -> None:
    bare = {"messages": [{"role": "user", "content": "x"}]}
    tagged = {"messages": [{"role": "user", "content": "x"}], "metadata": {"history_id": "h1"}}

    pid_bare = csv_schema.compute_prompt_id(csv_schema.canonical_prompt_spec(bare))
    pid_tagged = csv_schema.compute_prompt_id(csv_schema.canonical_prompt_spec(tagged))

    assert pid_bare == pid_tagged


def test_compute_prompt_id_for_spec_default_excludes_metadata() -> None:
    bare = {"messages": [{"role": "user", "content": "x"}]}
    tagged_a = {"messages": [{"role": "user", "content": "x"}], "metadata": {"iteration": 0}}
    tagged_b = {"messages": [{"role": "user", "content": "x"}], "metadata": {"iteration": 1}}

    # Default: metadata ignored, same content -> same id
    assert csv_schema.compute_prompt_id_for_spec(bare) == csv_schema.compute_prompt_id_for_spec(
        tagged_a
    )
    assert csv_schema.compute_prompt_id_for_spec(tagged_a) == csv_schema.compute_prompt_id_for_spec(
        tagged_b
    )


def test_compute_prompt_id_for_spec_with_metadata_distinguishes_iterations() -> None:
    bare = {"messages": [{"role": "user", "content": "x"}]}
    tagged_a = {"messages": [{"role": "user", "content": "x"}], "metadata": {"iteration": 0}}
    tagged_b = {"messages": [{"role": "user", "content": "x"}], "metadata": {"iteration": 1}}

    pid_bare = csv_schema.compute_prompt_id_for_spec(bare, include_metadata=True)
    pid_a = csv_schema.compute_prompt_id_for_spec(tagged_a, include_metadata=True)
    pid_b = csv_schema.compute_prompt_id_for_spec(tagged_b, include_metadata=True)

    # No metadata vs with-metadata should still differ
    assert pid_bare != pid_a
    # Each iteration gets a distinct id
    assert pid_a != pid_b
    # Same metadata reproduces same id
    assert pid_a == csv_schema.compute_prompt_id_for_spec(
        {"messages": [{"role": "user", "content": "x"}], "metadata": {"iteration": 0}},
        include_metadata=True,
    )


def test_extract_prompt_metadata() -> None:
    metadata = {"history_id": "h1", "true_gender": "Female"}
    spec = {"messages": [{"role": "user", "content": "x"}], "metadata": metadata}

    assert csv_schema.extract_prompt_metadata(spec) == metadata
    assert csv_schema.extract_prompt_metadata({"messages": []}) is None
    assert csv_schema.extract_prompt_metadata({"messages": [], "metadata": {}}) is None
    assert csv_schema.extract_prompt_metadata({"messages": [], "metadata": "bad"}) is None
    assert csv_schema.extract_prompt_metadata("plain string") is None


def test_prompt_metadata_serialization_round_trip() -> None:
    metadata = {"history_id": "h1", "nested": {"k": [1, 2]}}

    serialized = csv_schema.serialize_prompt_metadata(metadata)
    assert csv_schema.deserialize_prompt_metadata(serialized) == metadata

    # absent/empty metadata maps to the empty cell, and back to None
    assert csv_schema.serialize_prompt_metadata(None) == ""
    assert csv_schema.serialize_prompt_metadata({}) == ""
    assert csv_schema.deserialize_prompt_metadata("") is None
    assert csv_schema.deserialize_prompt_metadata("   ") is None


def test_matrix_cell_csv_round_trip() -> None:
    original = csv_schema.MatrixCell(
        status=csv_schema.CellStatus.SUCCESS,
        response={"message": 'hello,"world"\nnext', "structured": {"x": 1}},
    )

    serialized = original.to_csv_cell()
    parsed = csv_schema.MatrixCell.from_csv_cell(serialized)

    assert parsed == original
    assert csv_schema.MatrixCell.from_csv_cell("") is None
