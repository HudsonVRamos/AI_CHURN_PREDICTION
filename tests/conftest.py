"""Fixtures compartilhadas para testes end-to-end do pipeline de churn.

Fornece dados sintéticos e mocks reutilizáveis via pytest fixtures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.synthetic_users import (
    ACTIVE_USER_IDS,
    CHURNED_USER_IDS,
    generate_active_users,
    generate_all_users,
    generate_churned_users,
)
from tests.fixtures.mock_responses import (
    MockBedrockResponses,
    MockNPAWResponses,
    MockSageMakerResponses,
)


# --- Data de referência fixa para reprodutibilidade ---

REFERENCE_DATE = datetime(2024, 6, 1, tzinfo=timezone.utc)


# --- Fixtures de IDs ---


@pytest.fixture
def churned_user_ids() -> list[str]:
    """Lista de 10 user IDs de usuários churned."""
    return CHURNED_USER_IDS.copy()


@pytest.fixture
def active_user_ids() -> list[str]:
    """Lista de 10 user IDs de usuários ativos."""
    return ACTIVE_USER_IDS.copy()


@pytest.fixture
def all_user_ids() -> list[str]:
    """Lista de todos os 20 user IDs (churned + active)."""
    return CHURNED_USER_IDS + ACTIVE_USER_IDS


# --- Fixtures de sessões sintéticas ---


@pytest.fixture
def churned_sessions() -> dict[str, list[dict[str, Any]]]:
    """Sessões sintéticas para 10 churned users."""
    return generate_churned_users(REFERENCE_DATE)


@pytest.fixture
def active_sessions() -> dict[str, list[dict[str, Any]]]:
    """Sessões sintéticas para 10 active users."""
    return generate_active_users(REFERENCE_DATE)


@pytest.fixture
def all_sessions() -> dict[str, list[dict[str, Any]]]:
    """Sessões sintéticas para todos os 20 users."""
    return generate_all_users(REFERENCE_DATE)


# --- Fixtures de mocks de serviços ---


@pytest.fixture
def mock_npaw() -> MockNPAWResponses:
    """Mock da API NPAW com sessões pré-geradas."""
    return MockNPAWResponses()


@pytest.fixture
def mock_bedrock() -> MockBedrockResponses:
    """Mock do AWS Bedrock com explicações fixas em PT-BR."""
    return MockBedrockResponses()


@pytest.fixture
def mock_sagemaker() -> MockSageMakerResponses:
    """Mock do SageMaker com predições determinísticas."""
    return MockSageMakerResponses()


# --- Fixtures de dados derivados para validação ---


@pytest.fixture
def expected_predictions(mock_sagemaker: MockSageMakerResponses) -> list[dict[str, Any]]:
    """Predições esperadas para todos os 20 users."""
    all_ids = CHURNED_USER_IDS + ACTIVE_USER_IDS
    return mock_sagemaker.predict_batch(all_ids)


@pytest.fixture
def reference_date() -> datetime:
    """Data de referência fixa para geração de dados."""
    return REFERENCE_DATE


# --- Fixture de modelo pré-treinado (mock XGBoost) ---


@pytest.fixture
def mock_trained_model() -> MagicMock:
    """Modelo XGBoost mock para testes de inferência.

    Simula a interface do XGBoost Booster para uso em testes
    sem necessidade de treinamento real.
    """
    model = MagicMock()
    model.predict = MagicMock(
        side_effect=lambda dmatrix: [0.75] * dmatrix.num_row()
        if hasattr(dmatrix, "num_row")
        else [0.75]
    )
    model.get_dump = MagicMock(return_value=["tree_0", "tree_1", "tree_2"])
    model.save_model = MagicMock()
    model.load_model = MagicMock()
    model.num_features = MagicMock(return_value=22)
    return model


# --- Fixture de patch do boto3 para Bedrock ---


@pytest.fixture
def patched_bedrock_client(mock_bedrock: MockBedrockResponses):
    """Patch do cliente boto3 Bedrock para retornar respostas mock.

    Uso:
        def test_explanation(patched_bedrock_client):
            # boto3.client("bedrock-runtime") já está mockado
            ...
    """
    import io

    def mock_invoke_model(**kwargs):
        # Extrair risk_tier do prompt (simplificado)
        body = kwargs.get("body", "{}")
        if isinstance(body, str):
            import json
            body_data = json.loads(body)
        else:
            body_data = body

        # Default: High risk explanation
        response = mock_bedrock.invoke_model(
            user_id="mock-user",
            churn_probability=0.85,
            risk_tier="High",
        )

        return {
            "body": io.BytesIO(response["body"].encode("utf-8")),
            "contentType": "application/json",
        }

    mock_client = MagicMock()
    mock_client.invoke_model = MagicMock(side_effect=mock_invoke_model)

    with patch("boto3.client") as patched:
        patched.return_value = mock_client
        yield mock_client
