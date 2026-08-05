"""Property-based tests para graceful degradation do BedrockExplainer.

Valida que quando o AWS Bedrock falhar (qualquer tipo de exceção),
a predição permanece válida com explanation = None. O sistema
NUNCA propaga exceções do Bedrock para o caller.

**Validates: Requirements 12.6**
**Property 5: Graceful Degradation** — Se Bedrock falhar, prediction
permanece válida com explanation = null.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.common.models import FeatureContribution, PredictionResult
from src.explanations.bedrock_explainer import BedrockExplainer


# --- Estratégias ---

# Probabilidade de churn: float entre 0.0 e 1.0
churn_probability_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Confiança: float entre 0.0 e 1.0
confidence_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Contribution weight: float signed
contribution_weight_strategy = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)

# Normalized impact: entre -1.0 e 1.0
normalized_impact_strategy = st.floats(
    min_value=-1.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Feature name válido
feature_name_strategy = st.sampled_from([
    "total_sessions",
    "total_viewing_hours",
    "avg_session_duration_min",
    "sessions_per_week",
    "distinct_channels",
    "avg_happiness_score",
    "avg_buffer_ratio",
    "error_rate",
    "avg_bitrate",
    "pct_episode",
])

# Estratégia para gerar uma lista de FeatureContributions
feature_contribution_strategy = st.builds(
    FeatureContribution,
    feature_name=feature_name_strategy,
    contribution_weight=contribution_weight_strategy,
    normalized_impact=normalized_impact_strategy,
)

top_features_strategy = st.lists(
    feature_contribution_strategy,
    min_size=1,
    max_size=10,
)

# Tipos de exceção que podem ocorrer quando o Bedrock falha
exception_types_strategy = st.sampled_from([
    RuntimeError,
    TimeoutError,
    ValueError,
    ConnectionError,
    OSError,
    IOError,
    TypeError,
    KeyError,
    AttributeError,
    Exception,
])

# Mensagens de erro aleatórias
error_message_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
    ),
    min_size=1,
    max_size=100,
)


# --- Property Tests ---


class TestBedrockExplainerAlwaysReturnsStringOrNone:
    """Verifica que generate_explanation SEMPRE retorna str ou None.

    **Validates: Requirements 12.6**

    Nunca levanta exceção, nunca retorna tipo inesperado.
    """

    @given(
        churn_probability=churn_probability_strategy,
        confidence=confidence_strategy,
        top_features=top_features_strategy,
    )
    @settings(max_examples=200, deadline=None)
    @patch("src.explanations.bedrock_explainer.time.sleep")
    def test_generate_explanation_always_returns_string_or_none(
        self,
        mock_sleep: MagicMock,
        churn_probability: float,
        confidence: float,
        top_features: list[FeatureContribution],
    ) -> None:
        """generate_explanation() SEMPRE retorna str ou None.

        **Validates: Requirements 12.6**

        Propriedade: para qualquer combinação de inputs válidos,
        o retorno é SEMPRE uma string (explicação) ou None
        (Bedrock indisponível). Nunca levanta exceção, nunca retorna
        outro tipo.
        """
        # Cliente que falha com erro genérico
        client = MagicMock()
        client.invoke_model.side_effect = RuntimeError("Bedrock falhou")
        explainer = BedrockExplainer(client=client)

        user_feature_values = {
            f.feature_name: 1.0 for f in top_features
        }
        population_stats = {
            f.feature_name: {"mean": 5.0, "std": 2.0}
            for f in top_features
        }

        # NUNCA deve levantar exceção
        result = explainer.generate_explanation(
            user_id="user-test-001",
            churn_probability=churn_probability,
            confidence=confidence,
            top_features=top_features,
            user_feature_values=user_feature_values,
            population_stats=population_stats,
        )

        # Resultado é SEMPRE str ou None
        assert result is None or isinstance(result, str), (
            f"generate_explanation retornou tipo inesperado: "
            f"{type(result)} — valor: {result}"
        )


class TestBedrockFailureReturnsNone:
    """Verifica que qualquer exceção do Bedrock resulta em None.

    **Validates: Requirements 12.6**

    Para qualquer tipo de exceção, generate_explanation
    captura graciosamente e retorna None.
    """

    @given(
        exception_type=exception_types_strategy,
        error_message=error_message_strategy,
        churn_probability=churn_probability_strategy,
        confidence=confidence_strategy,
        top_features=top_features_strategy,
    )
    @settings(max_examples=200, deadline=None)
    @patch("src.explanations.bedrock_explainer.time.sleep")
    def test_any_exception_results_in_none(
        self,
        mock_sleep: MagicMock,
        exception_type: type,
        error_message: str,
        churn_probability: float,
        confidence: float,
        top_features: list[FeatureContribution],
    ) -> None:
        """Qualquer exceção do Bedrock resulta em retorno None.

        **Validates: Requirements 12.6**

        Propriedade: para qualquer tipo de exceção (RuntimeError,
        TimeoutError, ValueError, etc.) levantada pelo cliente
        Bedrock, generate_explanation() NUNCA propaga a exceção —
        sempre retorna None.
        """
        assume(len(error_message.strip()) > 0)

        # Cliente que levanta o tipo de exceção gerado
        client = MagicMock()
        client.invoke_model.side_effect = exception_type(error_message)
        explainer = BedrockExplainer(client=client)

        user_feature_values = {
            f.feature_name: 1.0 for f in top_features
        }
        population_stats = {
            f.feature_name: {"mean": 5.0, "std": 2.0}
            for f in top_features
        }

        # NÃO deve levantar exceção
        result = explainer.generate_explanation(
            user_id="user-test-002",
            churn_probability=churn_probability,
            confidence=confidence,
            top_features=top_features,
            user_feature_values=user_feature_values,
            population_stats=population_stats,
        )

        # Resultado DEVE ser None quando Bedrock falha
        assert result is None, (
            f"Esperado None quando Bedrock levanta "
            f"{exception_type.__name__}('{error_message}'), "
            f"mas obteve: {result}"
        )


class TestPredictionValidRegardlessOfBedrock:
    """Verifica que PredictionResult é válido independente do Bedrock.

    **Validates: Requirements 12.6**

    Um PredictionResult pode ser criado com todos os campos obrigatórios
    mesmo quando não há explicação disponível (explanation = None).
    """

    @given(
        churn_probability=churn_probability_strategy,
        confidence=confidence_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_prediction_result_valid_without_explanation(
        self,
        churn_probability: float,
        confidence: float,
    ) -> None:
        """PredictionResult é válido sem explicação do Bedrock.

        **Validates: Requirements 12.6**

        Propriedade: para qualquer probabilidade de churn e confiança
        válidas, é possível criar um PredictionResult completo e válido
        mesmo quando a explicação do Bedrock não está disponível.
        A predição NÃO depende do Bedrock para ser válida.
        """
        # Determinar risk_tier consistente com churn_probability
        if churn_probability <= 0.30:
            risk_tier = "Low"
        elif churn_probability <= 0.60:
            risk_tier = "Medium"
        else:
            risk_tier = "High"

        # PredictionResult é válido SEM qualquer explicação
        prediction = PredictionResult(
            user_id="user-graceful-001",
            churn_probability=churn_probability,
            confidence=confidence,
            risk_tier=risk_tier,
            model_version="arn:aws:sagemaker:us-east-1:123:model/v1",
            feature_version=1,
            timestamp="2024-01-15T10:00:00+00:00",
        )

        # Todos os campos obrigatórios estão presentes e válidos
        assert prediction.user_id == "user-graceful-001"
        assert prediction.churn_probability == churn_probability
        assert prediction.confidence == confidence
        assert prediction.risk_tier == risk_tier
        assert prediction.model_version is not None
        assert prediction.feature_version >= 1
        assert prediction.timestamp is not None

        # PredictionResult NÃO possui campo de explicação —
        # ela vive separada, exatamente para permitir degradation
        assert not hasattr(prediction, "explanation"), (
            "PredictionResult não deveria ter campo 'explanation' — "
            "a explicação vive separada para permitir graceful "
            "degradation"
        )
        assert not hasattr(prediction, "bedrock_explanation"), (
            "PredictionResult não deveria ter campo "
            "'bedrock_explanation' — separação de concerns"
        )

    @given(
        churn_probability=churn_probability_strategy,
        confidence=confidence_strategy,
        top_features=top_features_strategy,
    )
    @settings(max_examples=200, deadline=None)
    @patch("src.explanations.bedrock_explainer.time.sleep")
    def test_prediction_fields_unaffected_by_bedrock_failure(
        self,
        mock_sleep: MagicMock,
        churn_probability: float,
        confidence: float,
        top_features: list[FeatureContribution],
    ) -> None:
        """Campos da predição não são afetados pela falha do Bedrock.

        **Validates: Requirements 12.6**

        Propriedade: quando o Bedrock falha e retorna None,
        todos os campos da PredictionResult permanecem intactos.
        A indisponibilidade do Bedrock NÃO invalida, corrompe ou
        altera a predição.
        """
        # Determinar risk_tier
        if churn_probability <= 0.30:
            risk_tier = "Low"
        elif churn_probability <= 0.60:
            risk_tier = "Medium"
        else:
            risk_tier = "High"

        # Criar predição ANTES de tentar Bedrock
        prediction = PredictionResult(
            user_id="user-graceful-002",
            churn_probability=churn_probability,
            confidence=confidence,
            risk_tier=risk_tier,
            model_version="arn:aws:sagemaker:us-east-1:123:model/v2",
            feature_version=3,
            timestamp="2024-02-01T14:30:00+00:00",
        )

        # Bedrock falha
        client = MagicMock()
        client.invoke_model.side_effect = TimeoutError("timeout")
        explainer = BedrockExplainer(client=client)

        user_feature_values = {
            f.feature_name: 1.0 for f in top_features
        }
        population_stats = {
            f.feature_name: {"mean": 5.0, "std": 2.0}
            for f in top_features
        }

        explanation = explainer.generate_explanation(
            user_id="user-graceful-002",
            churn_probability=churn_probability,
            confidence=confidence,
            top_features=top_features,
            user_feature_values=user_feature_values,
            population_stats=population_stats,
        )

        # Bedrock retornou None (graceful degradation)
        assert explanation is None

        # A predição permanece COMPLETAMENTE VÁLIDA e inalterada
        assert prediction.user_id == "user-graceful-002"
        assert prediction.churn_probability == churn_probability
        assert prediction.confidence == confidence
        assert prediction.risk_tier == risk_tier
        assert prediction.model_version == (
            "arn:aws:sagemaker:us-east-1:123:model/v2"
        )
        assert prediction.feature_version == 3
        assert prediction.timestamp == "2024-02-01T14:30:00+00:00"
