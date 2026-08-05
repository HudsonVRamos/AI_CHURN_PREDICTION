# Módulo comum com utilitários compartilhados
"""Utilitários, configuração e helpers compartilhados entre módulos."""

from src.common.models import (
    ChurnPattern,
    ExecutionSummary,
    ExplainabilityResult,
    FeatureContribution,
    FeatureVector,
    ModelVersion,
    PredictionResult,
)

__all__ = [
    "FeatureVector",
    "PredictionResult",
    "ExplainabilityResult",
    "FeatureContribution",
    "ModelVersion",
    "ExecutionSummary",
    "ChurnPattern",
]

from src.common.config import (
    ConfigurationError,
    Settings,
    get_settings,
    reset_settings_cache,
)

__all__ = [
    "ConfigurationError",
    "Settings",
    "get_settings",
    "reset_settings_cache",
]
