"""Load dimension value mapping CSVs and generate all indicator combinations."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndicatorCombo:
    """One fully-resolved combination of indicator values for a single dimension value."""

    dimension: str
    dimension_value: str
    indicators: dict[str, str]
    combination_id: str


def _combination_id(dimension: str, dimension_value: str, indicators: dict[str, str]) -> str:
    payload = json.dumps(
        {"dimension": dimension, "dimension_value": dimension_value, "indicators": indicators},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_combinations(csv_path: Path, dimension: str) -> list[IndicatorCombo]:
    """Read a dimension CSV and return all indicator combinations.

    The CSV must have columns: Dimension_value, Indicator_name, Indicator_value.

    Algorithm:
    - Group rows by Dimension_value.
    - Within each group, split indicator names into scalars (appear once) and
      lists (appear more than once).
    - Take the Cartesian product over list indicator values while holding
      scalars fixed, yielding one IndicatorCombo per product element.
    """
    groups: dict[str, dict[str, list[str]]] = {}

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dim_value = row["Dimension_value"].strip()
            ind_name = row["Indicator_name"].strip()
            ind_value = row["Indicator_value"].strip()

            if dim_value not in groups:
                groups[dim_value] = {}
            if ind_name not in groups[dim_value]:
                groups[dim_value][ind_name] = []
            groups[dim_value][ind_name].append(ind_value)

    combos: list[IndicatorCombo] = []

    for dim_value, indicator_map in groups.items():
        scalars = {name: values[0] for name, values in indicator_map.items() if len(values) == 1}
        lists = {name: values for name, values in indicator_map.items() if len(values) > 1}

        if not lists:
            indicators = dict(sorted(scalars.items()))
            combo_id = _combination_id(dimension, dim_value, indicators)
            combos.append(
                IndicatorCombo(
                    dimension=dimension,
                    dimension_value=dim_value,
                    indicators=indicators,
                    combination_id=combo_id,
                )
            )
        else:
            list_names = sorted(lists.keys())
            list_values = [lists[name] for name in list_names]
            for product_tuple in itertools.product(*list_values):
                indicators = dict(sorted(scalars.items()))
                for name, value in zip(list_names, product_tuple):
                    indicators[name] = value
                indicators = dict(sorted(indicators.items()))
                combo_id = _combination_id(dimension, dim_value, indicators)
                combos.append(
                    IndicatorCombo(
                        dimension=dimension,
                        dimension_value=dim_value,
                        indicators=indicators,
                        combination_id=combo_id,
                    )
                )

    return combos
