"""
Configuration for the Linear Probing Pipeline.

Adjust these settings for your model, dataset, and compute environment.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the LLM to probe."""
    # Model identifier on Hugging Face Hub (or local path)
    # Other tested models: "google/gemma-4-E4B", "google/gemma-4-E2B", "meta-llama/Llama-3.2-1B-Instruct"
    model_name: str = "google/gemma-4-E2B"
    # Device map for model loading ("auto", "cpu", "cuda:0", etc.)
    device_map: str = "auto"
    # Load in reduced precision to save memory
    torch_dtype: str = "float16"  # "float16", "bfloat16", "float32"
    # Maximum sequence length for tokenization
    max_seq_length: int = 1024
    # Use Flash Attention 2 if available
    use_flash_attention: bool = False
    # HuggingFace access token (set via env var HF_TOKEN if needed)
    hf_token: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for dataset generation."""
    # Directory for storing generated datasets
    data_dir: str = "data"
    # Number of conversations to generate per expertise level
    samples_per_class: int = 50
    # Expertise levels to classify
    expertise_levels: List[str] = field(default_factory=lambda: [
        "novice", "intermediate", "expert",
    ])
    # Train/test split ratio
    test_size: float = 0.2
    # Random seed
    seed: int = 42


@dataclass
class ProbeConfig:
    """Configuration for the linear probing classifiers."""
    # Which layers to probe — None means all layers
    layers: Optional[List[int]] = None
    # Token position to extract: "last" (last user token) or "mean" (mean pool)
    token_position: str = "last"
    # Regularization strength for Logistic Regression (smaller = stronger reg.)
    logistic_C: float = 1.0
    # Maximum iterations for solver convergence
    max_iter: int = 1000
    # Number of cross-validation folds
    cv_folds: int = 5
    # Evaluation mode: "split" = single train/test split, "repeated_kfold" = repeated stratified k-fold
    eval_mode: str = "repeated_kfold"
    # Number of repeats for repeated k-fold
    n_repeats: int = 5


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    # Directory for results (metrics, plots)
    results_dir: str = "results"
    plots_dir: str = "plots"
    # Verbosity
    verbose: bool = True
