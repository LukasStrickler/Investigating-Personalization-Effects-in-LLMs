"""Unit tests for generate_backgrounds.combination — CSV loading and indicator combo logic."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from generate_backgrounds.combination import IndicatorCombo, load_combinations


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dimension_value", "Indicator_name", "Indicator_value"])
        writer.writerows(rows)


class TestScalarOnly:
    """Dimension values where every indicator appears exactly once."""

    def test_single_value_single_indicator(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(csv_path, [("Elite", "Income", "90000")])

        combos = load_combinations(csv_path, "Social_Status")

        assert len(combos) == 1
        assert combos[0].dimension == "Social_Status"
        assert combos[0].dimension_value == "Elite"
        assert combos[0].indicators == {"Income": "90000"}

    def test_multiple_scalars_produce_one_combo(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(
            csv_path,
            [
                ("Elite", "Income", "90000"),
                ("Elite", "Savings", "150000"),
                ("Elite", "Education", "PhD"),
            ],
        )

        combos = load_combinations(csv_path, "Social_Status")

        assert len(combos) == 1
        assert combos[0].indicators == {
            "Education": "PhD",
            "Income": "90000",
            "Savings": "150000",
        }

    def test_multiple_dimension_values_each_scalar(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(
            csv_path,
            [
                ("Young", "Age_range", "18-25"),
                ("Old", "Age_range", "65+"),
            ],
        )

        combos = load_combinations(csv_path, "Age")

        assert len(combos) == 2
        values = {c.dimension_value for c in combos}
        assert values == {"Young", "Old"}


class TestListIndicators:
    """Dimension values where some indicators appear multiple times (lists)."""

    def test_single_list_indicator_produces_one_combo_per_value(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(
            csv_path,
            [
                ("Elite", "Occupation", "CEO"),
                ("Elite", "Occupation", "Barrister"),
                ("Elite", "Occupation", "Surgeon"),
            ],
        )

        combos = load_combinations(csv_path, "Social_Status")

        assert len(combos) == 3
        occupations = {c.indicators["Occupation"] for c in combos}
        assert occupations == {"CEO", "Barrister", "Surgeon"}

    def test_scalar_plus_list_crosses_correctly(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(
            csv_path,
            [
                ("Elite", "Income", "90000"),
                ("Elite", "Savings", "150000"),
                ("Elite", "Occupation", "CEO"),
                ("Elite", "Occupation", "Barrister"),
            ],
        )

        combos = load_combinations(csv_path, "Social_Status")

        assert len(combos) == 2
        for c in combos:
            assert c.indicators["Income"] == "90000"
            assert c.indicators["Savings"] == "150000"
            assert c.indicators["Occupation"] in {"CEO", "Barrister"}

    def test_two_list_indicators_produce_cartesian_product(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(
            csv_path,
            [
                ("Elite", "Occupation", "CEO"),
                ("Elite", "Occupation", "Barrister"),
                ("Elite", "Hobby", "Golf"),
                ("Elite", "Hobby", "Sailing"),
                ("Elite", "Hobby", "Polo"),
            ],
        )

        combos = load_combinations(csv_path, "Social_Status")

        # 2 occupations × 3 hobbies = 6 combos
        assert len(combos) == 6
        pairs = {(c.indicators["Occupation"], c.indicators["Hobby"]) for c in combos}
        assert len(pairs) == 6


class TestCombinationIds:
    """IDs are deterministic and unique."""

    def test_same_input_produces_same_id(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(csv_path, [("Elite", "Income", "90000")])

        combos1 = load_combinations(csv_path, "Social_Status")
        combos2 = load_combinations(csv_path, "Social_Status")

        assert combos1[0].combination_id == combos2[0].combination_id

    def test_different_values_produce_different_ids(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(
            csv_path,
            [
                ("Elite", "Income", "90000"),
                ("Poor", "Income", "15000"),
            ],
        )

        combos = load_combinations(csv_path, "Social_Status")

        assert combos[0].combination_id != combos[1].combination_id

    def test_all_ids_unique_within_list_indicators(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(
            csv_path,
            [
                ("Elite", "Occupation", "CEO"),
                ("Elite", "Occupation", "Barrister"),
                ("Elite", "Occupation", "Surgeon"),
            ],
        )

        combos = load_combinations(csv_path, "Social_Status")
        ids = [c.combination_id for c in combos]

        assert len(set(ids)) == len(ids)


class TestEdgeCases:
    """Whitespace handling, empty files, BOM."""

    def test_whitespace_is_stripped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        path = csv_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("Dimension_value,Indicator_name,Indicator_value\n")
            f.write("  Elite , Income ,  90000  \n")

        combos = load_combinations(csv_path, "Social_Status")

        assert combos[0].dimension_value == "Elite"
        assert combos[0].indicators == {"Income": "90000"}

    def test_empty_csv_returns_no_combos(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        _write_csv(csv_path, [])

        combos = load_combinations(csv_path, "Social_Status")

        assert combos == []

    def test_utf8_bom_is_handled(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dim.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Dimension_value", "Indicator_name", "Indicator_value"])
            writer.writerow(["Elite", "Income", "90000"])

        combos = load_combinations(csv_path, "Social_Status")

        assert len(combos) == 1
        assert combos[0].indicators == {"Income": "90000"}
