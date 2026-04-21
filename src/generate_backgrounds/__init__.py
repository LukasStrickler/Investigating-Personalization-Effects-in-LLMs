"""
generate_backgrounds — pipeline for generating LLM persona conversation histories.

Public API:
    BackgroundPipeline   — orchestrates all three generation phases
    GenerationConfig     — configuration dataclass
    BackgroundRecord     — one LLM prompt+response pair for a single indicator combo
    ConversationHistory  — assembled multi-turn history for a full persona
    PipelineResult       — summary returned by BackgroundPipeline.run()
"""

from .pipeline import (
    AssemblyResult,
    BackgroundPipeline,
    BackgroundRecord,
    ConversationHistory,
    DimensionResult,
    GenerationConfig,
    PipelineResult,
)

__all__ = [
    "BackgroundPipeline",
    "GenerationConfig",
    "BackgroundRecord",
    "ConversationHistory",
    "DimensionResult",
    "AssemblyResult",
    "PipelineResult",
]
