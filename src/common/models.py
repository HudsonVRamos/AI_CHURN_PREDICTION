"""Modelos de dados compartilhados da plataforma de predição de churn.

Define os Pydantic models para todas as entidades do sistema:
FeatureVector, PredictionResult, ExplainabilityResult, FeatureContribution,
ModelVersion, ExecutionSummary e ChurnPattern.

Validação de tipos e ranges conforme especificação dos requisitos 3.5, 10.4, 11.3, 13.1.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class FeatureVector(BaseModel):
    """Vetor de features comportamentais agregadas por assinante.

    Contém métricas de engagement, qualidade, comportamento e tendências
    derivadas das sessões de visualização extraídas da NPAW.

    Validates: Requirement 3.5
    """

    user_id: str = Field(..., min_length=1, description="ID único do assinante")
    version: int = Field(..., ge=1, description="Versão do feature vector (auto-incremento)")
    generated_at: str = Field(..., description="Timestamp de geração (ISO 8601)")
    observation_start: str = Field(..., description="Início do período de observação (ISO 8601)")
    observation_end: str = Field(..., description="Fim do período de observação (ISO 8601)")

    # Engagement
    total_sessions: int = Field(..., ge=0, description="Total de sessões no período")
    total_viewing_hours: float = Field(
        ..., ge=0.0, description="Total de horas de visualização efetiva"
    )
    avg_session_duration_min: float = Field(
        ..., ge=0.0, description="Duração média de sessão em minutos"
    )
    sessions_per_week: float = Field(..., ge=0.0, description="Frequência de sessões por semana")
    distinct_channels: int = Field(..., ge=0, description="Canais distintos assistidos")

    # Quality
    avg_happiness_score: float = Field(
        ..., ge=0.0, le=10.0, description="Score médio de satisfação (0-10)"
    )
    avg_buffer_ratio: float = Field(..., ge=0.0, description="Ratio médio de buffering")
    error_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Taxa de erro (0.0-1.0)"
    )
    avg_bitrate: float = Field(..., ge=0.0, description="Bitrate médio em bps")

    # Behavioral
    pct_episode: float = Field(
        ..., ge=0.0, le=100.0, description="Percentual de sessões EPISODE"
    )
    pct_sport: float = Field(
        ..., ge=0.0, le=100.0, description="Percentual de sessões SPORT"
    )
    pct_live: float = Field(
        ..., ge=0.0, le=100.0, description="Percentual de sessões LIVE"
    )
    pct_show: float = Field(
        ..., ge=0.0, le=100.0, description="Percentual de sessões SHOW"
    )
    distinct_devices: int = Field(..., ge=0, description="Dispositivos distintos utilizados")
    avg_pause_count: float = Field(..., ge=0.0, description="Média de pausas por sessão")
    avg_seek_count: float = Field(..., ge=0.0, description="Média de seeks por sessão")

    # Trends (nullable - None se período < 4 semanas)
    viewing_time_trend: Optional[float] = Field(
        default=None, description="Tendência de tempo de visualização (horas/semana)"
    )
    error_rate_trend: Optional[float] = Field(
        default=None, description="Tendência de taxa de erro"
    )
    session_frequency_trend: Optional[float] = Field(
        default=None, description="Tendência de frequência de sessões"
    )

    @model_validator(mode="after")
    def validate_content_percentages_sum(self) -> "FeatureVector":
        """Valida que os percentuais de conteúdo somam 100% (com tolerância de 0.01)."""
        total = self.pct_episode + self.pct_sport + self.pct_live + self.pct_show
        if abs(total - 100.0) > 0.01:
            raise ValueError(
                f"Percentuais de conteúdo devem somar 100%, mas somam {total:.2f}%"
            )
        return self


class PredictionResult(BaseModel):
    """Resultado de predição de churn para um assinante.

    Produzido pelo ML_Pipeline (SageMaker Batch Transform).

    Validates: Requirement 10.4
    """

    user_id: str = Field(..., min_length=1, description="ID do assinante")
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilidade de churn (0.0-1.0)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Grau de confiança da predição (0.0-1.0)"
    )
    risk_tier: Literal["Low", "Medium", "High"] = Field(
        ..., description="Classificação de risco: Low (0-0.30), Medium (0.31-0.60), High (0.61-1.0)"
    )
    model_version: str = Field(..., min_length=1, description="Versão do modelo utilizado")
    feature_version: int = Field(..., ge=1, description="Versão do Feature Vector utilizado")
    timestamp: str = Field(..., description="Timestamp da predição (ISO 8601)")

    @model_validator(mode="after")
    def validate_risk_tier_consistency(self) -> "PredictionResult":
        """Valida que o risk_tier é consistente com o churn_probability."""
        prob = self.churn_probability
        tier = self.risk_tier
        if prob <= 0.30 and tier != "Low":
            raise ValueError(
                f"churn_probability={prob:.2f} deveria ser 'Low', mas é '{tier}'"
            )
        if 0.30 < prob <= 0.60 and tier != "Medium":
            raise ValueError(
                f"churn_probability={prob:.2f} deveria ser 'Medium', mas é '{tier}'"
            )
        if prob > 0.60 and tier != "High":
            raise ValueError(
                f"churn_probability={prob:.2f} deveria ser 'High', mas é '{tier}'"
            )
        return self


class FeatureContribution(BaseModel):
    """Contribuição de uma feature individual para a predição.

    Produzido pelo Explainability_Engine (SHAP).

    Validates: Requirement 11.3
    """

    feature_name: str = Field(..., min_length=1, description="Nome da feature")
    contribution_weight: float = Field(
        ..., description="Peso de contribuição (signed: positivo empurra para churn)"
    )
    normalized_impact: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Impacto normalizado na predição (-1.0 a 1.0)",
    )


class ExplainabilityResult(BaseModel):
    """Resultado de explicabilidade para uma predição individual.

    Contém as top features com maior impacto na classificação.

    Validates: Requirement 11.3
    """

    user_id: str = Field(..., min_length=1, description="ID do assinante")
    top_features: List[FeatureContribution] = Field(
        ..., description="Top features com maior impacto"
    )
    base_value: float = Field(..., description="Valor base do modelo (SHAP base value)")
    prediction_value: float = Field(..., description="Valor da predição (SHAP output)")

    @field_validator("top_features")
    @classmethod
    def validate_top_features_not_empty(
        cls, v: List[FeatureContribution],
    ) -> List[FeatureContribution]:
        """Valida que existe pelo menos uma feature contribution."""
        if len(v) == 0:
            raise ValueError("top_features deve conter pelo menos uma feature")
        return v


class ModelVersion(BaseModel):
    """Metadados de uma versão de modelo registrada no SageMaker Model Registry.

    Validates: Requirement 15.2
    """

    model_package_arn: str = Field(
        ..., min_length=1, description="ARN do Model Package no SageMaker"
    )
    algorithm: Literal["xgboost", "lightgbm", "catboost"] = Field(
        ..., description="Algoritmo utilizado no treinamento"
    )
    training_date: str = Field(..., description="Data de treinamento (ISO 8601)")
    dataset_version: str = Field(
        ..., min_length=1, description="Referência à versão do dataset usado no treino"
    )
    metrics: Dict[str, float] = Field(
        ..., description="Métricas de avaliação (precision, recall, f1, roc_auc)"
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics_keys(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Valida que todas as métricas obrigatórias estão presentes e dentro do range."""
        required_keys = {"precision", "recall", "f1", "roc_auc"}
        missing = required_keys - set(v.keys())
        if missing:
            raise ValueError(f"Métricas obrigatórias ausentes: {missing}")
        for key in required_keys:
            value = v[key]
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Métrica '{key}' deve estar entre 0.0 e 1.0, mas é {value}"
                )
        return v


class ExecutionSummary(BaseModel):
    """Resumo de uma execução do pipeline de predição.

    Validates: Requirements 8.6, 17.1
    """

    execution_id: str = Field(
        ..., min_length=1, description="ID único da execução (UUID)"
    )
    start_time: str = Field(..., description="Timestamp de início (ISO 8601)")
    end_time: str = Field(..., description="Timestamp de fim (ISO 8601)")
    mode: Literal["train", "predict"] = Field(
        ..., description="Modo de execução: train ou predict"
    )
    model_version: str = Field(..., min_length=1, description="Versão do modelo utilizado")
    users_processed: int = Field(
        ..., ge=0, description="Quantidade de usuários processados com sucesso"
    )
    users_failed: int = Field(
        ..., ge=0, description="Quantidade de usuários com falha"
    )
    status: Literal["running", "completed", "failed"] = Field(
        ..., description="Status da execução"
    )
    output_s3_path: str = Field(
        ..., min_length=1, description="Path S3 dos resultados de saída"
    )


class ChurnPattern(BaseModel):
    """Padrão de churn identificado na população (estatísticas agregadas).

    Modelo leve para armazenar importância de features na população.

    Validates: Requirement 13.1
    """

    feature_name: str = Field(..., min_length=1, description="Nome da feature")
    population_mean: float = Field(..., description="Média da feature na população")
    population_std: float = Field(
        ..., ge=0.0, description="Desvio padrão na população"
    )
    importance_rank: int = Field(
        ..., ge=1, description="Ranking de importância (1 = mais importante)"
    )
