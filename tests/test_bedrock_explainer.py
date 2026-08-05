"""Testes para o BedrockExplainer.

Valida a geração de explicações em PT-BR via AWS Bedrock,
incluindo retry, timeout e graceful degradation.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import (
    ClientError,
    ReadTimeoutError,
)

from src.common.models import FeatureContribution
from src.explanations.bedrock_explainer import BedrockExplainer


# --- Fixtures ---


@pytest.fixture
def sample_features() -> list[FeatureContribution]:
    """Features de exemplo para testes."""
    return [
        FeatureContribution(
            feature_name="sessions_per_week",
            contribution_weight=0.35,
            normalized_impact=0.8,
        ),
        FeatureContribution(
            feature_name="avg_happiness_score",
            contribution_weight=-0.25,
            normalized_impact=-0.6,
        ),
        FeatureContribution(
            feature_name="total_viewing_hours",
            contribution_weight=0.20,
            normalized_impact=0.5,
        ),
    ]


@pytest.fixture
def sample_user_values() -> dict:
    """Valores de features do assinante para testes."""
    return {
        "sessions_per_week": 1.2,
        "avg_happiness_score": 4.5,
        "total_viewing_hours": 3.0,
    }


@pytest.fixture
def sample_population_stats() -> dict:
    """Estatísticas populacionais para testes."""
    return {
        "sessions_per_week": {"mean": 5.3, "std": 2.1},
        "avg_happiness_score": {"mean": 7.2, "std": 1.5},
        "total_viewing_hours": {"mean": 15.0, "std": 8.0},
    }


@pytest.fixture
def mock_bedrock_client() -> MagicMock:
    """Cliente Bedrock mockado com resposta padrão."""
    client = MagicMock()
    response_body = json.dumps({
        "content": [
            {
                "type": "text",
                "text": (
                    "O assinante apresenta risco elevado de cancelamento. "
                    "A frequência de uso semanal está significativamente "
                    "abaixo da média da população."
                ),
            }
        ]
    })
    client.invoke_model.return_value = {
        "body": BytesIO(response_body.encode("utf-8")),
    }
    return client


@pytest.fixture
def explainer(mock_bedrock_client: MagicMock) -> BedrockExplainer:
    """BedrockExplainer com cliente mockado."""
    return BedrockExplainer(client=mock_bedrock_client)


# --- Testes de Configuração ---


class TestBedrockExplainerConfig:
    """Testes de configuração e constantes."""

    def test_model_id_is_claude_3_haiku(self) -> None:
        """Verifica que o modelo configurado é Claude 3 Haiku."""
        assert "claude-3-haiku" in BedrockExplainer.MODEL_ID

    def test_timeout_is_60_seconds(self) -> None:
        """Verifica que o timeout é 60 segundos (R12.6)."""
        assert BedrockExplainer.TIMEOUT_SECONDS == 60

    def test_max_retries_is_2(self) -> None:
        """Verifica que o número máximo de retentativas é 2."""
        assert BedrockExplainer.MAX_RETRIES == 2

    def test_retry_interval_is_5_seconds(self) -> None:
        """Verifica que o intervalo entre retentativas é 5 segundos."""
        assert BedrockExplainer.RETRY_INTERVAL_SECONDS == 5


# --- Testes de Geração de Explicação ---


class TestGenerateExplanation:
    """Testes do método generate_explanation."""

    def test_returns_explanation_on_success(
        self,
        explainer: BedrockExplainer,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica que retorna explicação quando Bedrock responde."""
        result = explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )
        assert result is not None
        assert "assinante" in result.lower()

    def test_invokes_bedrock_with_correct_model(
        self,
        explainer: BedrockExplainer,
        mock_bedrock_client: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica que o modelo correto é invocado."""
        explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        assert call_kwargs["modelId"] == BedrockExplainer.MODEL_ID

    def test_sends_all_required_data_in_prompt(
        self,
        explainer: BedrockExplainer,
        mock_bedrock_client: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica que todos os dados requeridos são enviados (R12.1)."""
        explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        prompt = body["messages"][0]["content"]

        # R12.1: churn_probability, confidence, top_features, valores, stats
        assert "75.0%" in prompt  # churn_probability
        assert "85.0%" in prompt  # confidence
        assert "sessions_per_week" in prompt
        assert "avg_happiness_score" in prompt
        assert "1.2" in prompt  # valor do assinante
        assert "5.3" in prompt  # média da população

    def test_prompt_instructs_not_to_calculate_probability(
        self,
        explainer: BedrockExplainer,
        mock_bedrock_client: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica instrução de NÃO calcular probabilidade (R12.4)."""
        explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        prompt = body["messages"][0]["content"]

        assert "NÃO calcule" in prompt or "NÃO calcul" in prompt

    def test_prompt_instructs_not_to_override_ml_result(
        self,
        explainer: BedrockExplainer,
        mock_bedrock_client: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica instrução de NÃO alterar resultado do ML (R12.5)."""
        explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        prompt = body["messages"][0]["content"]

        assert "NÃO altere" in prompt or "NÃO alter" in prompt

    def test_prompt_is_in_portuguese(
        self,
        explainer: BedrockExplainer,
        mock_bedrock_client: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica que o prompt é em PT-BR (R12.7)."""
        explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        call_kwargs = mock_bedrock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        prompt = body["messages"][0]["content"]

        assert "português brasileiro" in prompt.lower()


# --- Testes de Graceful Degradation ---


class TestGracefulDegradation:
    """Testes de graceful degradation (R12.6)."""

    @patch("src.explanations.bedrock_explainer.time.sleep")
    def test_returns_none_on_timeout(
        self,
        mock_sleep: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica retorno None quando Bedrock dá timeout (R12.6)."""
        client = MagicMock()
        client.invoke_model.side_effect = ReadTimeoutError(
            endpoint_url="https://bedrock.us-east-1.amazonaws.com"
        )
        explainer = BedrockExplainer(client=client)

        result = explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        assert result is None

    @patch("src.explanations.bedrock_explainer.time.sleep")
    def test_returns_none_on_client_error(
        self,
        mock_sleep: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica retorno None quando Bedrock retorna erro."""
        client = MagicMock()
        client.invoke_model.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ServiceUnavailableException",
                    "Message": "Service unavailable",
                }
            },
            operation_name="InvokeModel",
        )
        explainer = BedrockExplainer(client=client)

        result = explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        assert result is None

    @patch("src.explanations.bedrock_explainer.time.sleep")
    def test_retries_twice_before_returning_none(
        self,
        mock_sleep: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica que faz 2 tentativas antes de retornar None."""
        client = MagicMock()
        client.invoke_model.side_effect = ReadTimeoutError(
            endpoint_url="https://bedrock.us-east-1.amazonaws.com"
        )
        explainer = BedrockExplainer(client=client)

        explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        assert client.invoke_model.call_count == 2

    @patch("src.explanations.bedrock_explainer.time.sleep")
    def test_waits_5_seconds_between_retries(
        self,
        mock_sleep: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica intervalo de 5s entre retentativas."""
        client = MagicMock()
        client.invoke_model.side_effect = ReadTimeoutError(
            endpoint_url="https://bedrock.us-east-1.amazonaws.com"
        )
        explainer = BedrockExplainer(client=client)

        explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        mock_sleep.assert_called_once_with(5)

    @patch("src.explanations.bedrock_explainer.time.sleep")
    def test_succeeds_on_second_attempt(
        self,
        mock_sleep: MagicMock,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica sucesso na segunda tentativa após falha na primeira."""
        client = MagicMock()
        response_body = json.dumps({
            "content": [{"type": "text", "text": "Explicação gerada."}]
        })

        client.invoke_model.side_effect = [
            ReadTimeoutError(
                endpoint_url="https://bedrock.us-east-1.amazonaws.com"
            ),
            {"body": BytesIO(response_body.encode("utf-8"))},
        ]
        explainer = BedrockExplainer(client=client)

        result = explainer.generate_explanation(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        assert result == "Explicação gerada."
        assert client.invoke_model.call_count == 2


# --- Testes do Prompt ---


class TestBuildPrompt:
    """Testes de construção do prompt."""

    def test_prompt_contains_risk_classification(
        self,
        explainer: BedrockExplainer,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica que o prompt contém classificação de risco (R12.3)."""
        prompt = explainer._build_prompt(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        assert "Alto" in prompt

    def test_prompt_medium_risk(
        self,
        explainer: BedrockExplainer,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica classificação Médio para prob entre 0.31 e 0.60."""
        prompt = explainer._build_prompt(
            user_id="user-123",
            churn_probability=0.45,
            confidence=0.80,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        assert "Médio" in prompt

    def test_prompt_low_risk(
        self,
        explainer: BedrockExplainer,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica classificação Baixo para prob <= 0.30."""
        prompt = explainer._build_prompt(
            user_id="user-123",
            churn_probability=0.20,
            confidence=0.90,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        assert "Baixo" in prompt

    def test_prompt_includes_population_stats(
        self,
        explainer: BedrockExplainer,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica comparação com estatísticas populacionais (R12.1)."""
        prompt = explainer._build_prompt(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        # Verifica média e desvio padrão
        assert "5.3" in prompt  # mean sessions_per_week
        assert "2.1" in prompt  # std sessions_per_week

    def test_prompt_includes_feature_direction(
        self,
        explainer: BedrockExplainer,
        sample_features: list[FeatureContribution],
        sample_user_values: dict,
        sample_population_stats: dict,
    ) -> None:
        """Verifica que indica direção da contribuição da feature."""
        prompt = explainer._build_prompt(
            user_id="user-123",
            churn_probability=0.75,
            confidence=0.85,
            top_features=sample_features,
            user_feature_values=sample_user_values,
            population_stats=sample_population_stats,
        )

        assert "aumenta" in prompt
        assert "diminui" in prompt


# --- Testes de Classificação de Risco ---


class TestClassifyRisk:
    """Testes do método _classify_risk."""

    @pytest.mark.parametrize(
        "probability,expected",
        [
            (0.0, "Baixo"),
            (0.15, "Baixo"),
            (0.30, "Baixo"),
            (0.31, "Médio"),
            (0.45, "Médio"),
            (0.60, "Médio"),
            (0.61, "Alto"),
            (0.80, "Alto"),
            (1.0, "Alto"),
        ],
    )
    def test_risk_classification(
        self, probability: float, expected: str
    ) -> None:
        """Verifica classificação correta para cada faixa."""
        assert BedrockExplainer._classify_risk(probability) == expected
