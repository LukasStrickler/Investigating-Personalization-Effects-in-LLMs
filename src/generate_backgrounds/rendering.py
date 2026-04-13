"""Load dimension prompt templates and render indicator values into prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path


class TemplateRenderError(Exception):
    pass


def load_templates(templates_path: Path) -> dict[str, str]:
    """Load dimension_templates.json and return {dimension_key: template_string}."""
    with open(templates_path, encoding="utf-8") as f:
        return json.load(f)


def render_template(template: str, indicators: dict[str, str]) -> str:
    """Replace all <IndicatorName> placeholders with their indicator values.

    Raises TemplateRenderError if any placeholder has no matching key in indicators.
    """
    placeholders = re.findall(r"<([^>]+)>", template)
    missing = [p for p in placeholders if p not in indicators]
    if missing:
        raise TemplateRenderError(
            f"Template placeholders not found in indicators: {missing}. "
            f"Available keys: {sorted(indicators.keys())}"
        )
    return re.sub(r"<([^>]+)>", lambda m: indicators[m.group(1)], template)


def find_dimension_csv(mapping_dir: Path, dimension_key: str) -> Path | None:
    """Resolve the CSV file for a dimension by convention.

    Convention: <mapping_dir>/<dimension_key_lowercase>.csv
    E.g. "Social_Status" -> mapping_dir / "social_status.csv"
    """
    filename = dimension_key.lower() + ".csv"
    candidate = mapping_dir / filename
    return candidate if candidate.exists() else None


def discover_dimensions(mapping_dir: Path, templates: dict[str, str]) -> list[str]:
    """Return dimension keys that have both a template and a CSV file.

    Returns keys in the order they appear in templates.
    """
    return [key for key in templates if find_dimension_csv(mapping_dir, key) is not None]
