"""Configuration for the persona linear-probing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent


@dataclass
class ModelConfig:
    """Configuration for the frozen LLM whose representations are probed."""

    model_name: str = str(_HERE / "models" / "SmolLM2-360M-Instruct")
    device_map: str = "auto"
    torch_dtype: str = "float16"
    max_seq_length: int = 2048
    use_flash_attention: bool = False
    hf_token: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for loading generated persona conversation histories."""

    personas_file: str = str(
        _REPO_ROOT / "src" / "generate_backgrounds" / "data" / "personas" / "personas.jsonl"
    )
    data_dir: str = str(_HERE / "data")
    attributes: List[str] = field(default_factory=lambda: ["Gender"])
    include_partial: bool = False
    # Optional cap per joint persona group (e.g. Female + Germany). None uses all.
    samples_per_group: Optional[int] = None
    test_size: float = 0.2
    seed: int = 42
    # "full" retains every message; "gender-turn-only" removes other
    # dimensions (especially gender-coded names in the Region turn).
    context_mode: str = "full"


@dataclass
class ProbeConfig:
    """Configuration for the linear probing classifiers."""

    layers: Optional[List[int]] = None
    token_position: str = "last"
    logistic_C: float = 1.0
    max_iter: int = 1000
    cv_folds: int = 5


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    results_dir: str = str(_HERE / "results")
    plots_dir: str = str(_HERE / "plots")
    verbose: bool = True
