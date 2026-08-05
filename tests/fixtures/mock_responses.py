"""Mock responses para serviços externos (NPAW, Bedrock, SageMaker).

Fornece respostas determinísticas para testes end-to-end sem dependência
de serviços reais.

Validates: Requirements 10.5, 12.6
"""

from __future__ import annotations

import json
import random
from typing import Any

from tests.fixtures.synthetic_users import (
    ACTIVE_USER_IDS,
    CHURNED_USER_IDS,
    generate_all_users,
)


class MockNPAWResponses:
    """Mock de respostas da API NPAW para testes.

    Simula o endpoint GET /sky_brazil/rawdata com paginação.
    """

    BATCH_SIZE = 100

    def __init__(self) -> None:
        self._user_sessions = generate_all_users()

    def get_sessions(
        self, user_id: str, offset: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        """Simula resposta da NPAW API para um user_id com paginação.

        Retorna formato compatível com a API NPAW:
        {
            "status": "ok",
            "data": {
                "values": [...sessions...],
                "total": N
            }
        }
        """
        sessions = self._user_sessions.get(user_id, [])
        total = len(sessions)
        page = sessions[offset: offset + limit]

        return {
            "status": "ok",
            "data": {
                "values": page,
                "total": total,
            },
        }

    def get_error_response(self, status_code: int = 500) -> dict[str, Any]:
        """Simula resposta de erro da NPAW API."""
        return {
            "status": "error",
            "error": {
                "code": status_code,
                "message": f"Internal Server Error ({status_code})",
            },
        }

    def get_auth_error(self) -> dict[str, Any]:
        """Simula erro de autenticação (401)."""
        return {
            "status": "error",
            "error": {
                "code": 401,
                "message": "Invalid API key",
            },
        }

    def get_empty_response(self) -> dict[str, Any]:
        """Simula resposta sem dados."""
        return {
            "status": "ok",
            "data": {
                "values": [],
                "total": 0,
            },
        }


class MockBedrockResponses:
    """Mock de respostas do AWS Bedrock para testes.

    Gera explicações fixas em PT-BR para cada nível de risco.
    Validates: Requirement 12.6 (graceful degradation quando Bedrock falha)
    """

    # Explicações fixas por tier de risco
    EXPLANATIONS: dict[str, str] = {
        "High": (
            "Este assinante apresenta alto risco de cancelamento. "
            "Os principais fatores identificados são: redução significativa no tempo "
            "de visualização nas últimas semanas, aumento na taxa de erros de reprodução, "
            "e diminuição na diversidade de conteúdos assistidos. "
            "O padrão comportamental é consistente com assinantes que cancelaram "
            "nos últimos 6 meses."
        ),
        "Medium": (
            "Este assinante apresenta risco moderado de cancelamento. "
            "Observa-se uma leve redução na frequência de uso, porém o engajamento "
            "com conteúdos esportivos se mantém estável. "
            "Recomenda-se acompanhamento dos indicadores nas próximas semanas."
        ),
        "Low": (
            "Este assinante apresenta baixo risco de cancelamento. "
            "O padrão de uso é consistente e estável, com alta diversidade "
            "de conteúdos e boa qualidade de experiência. "
            "Não foram identificados sinais de desengajamento."
        ),
    }

    def invoke_model(
        self,
        user_id: str,
        churn_probability: float,
        risk_tier: str,
        top_features: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Simula resposta do Bedrock (Claude 3 Haiku).

        Retorna formato compatível com a API Bedrock InvokeModel.
        """
        explanation = self.EXPLANATIONS.get(risk_tier, self.EXPLANATIONS["Low"])

        # Formato de resposta do Bedrock (Claude)
        response_body = {
            "id": f"msg_mock_{user_id[:8]}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": explanation,
                }
            ],
            "model": "anthropic.claude-3-haiku-20240307-v1:0",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 350,
                "output_tokens": 120,
            },
        }

        return {
            "statusCode": 200,
            "body": json.dumps(response_body),
        }

    def invoke_model_timeout(self) -> dict[str, Any]:
        """Simula timeout do Bedrock (>60s).

        Usado para testar graceful degradation (Requirement 12.6).
        """
        raise TimeoutError("Bedrock invocation exceeded 60 seconds timeout")

    def invoke_model_error(self) -> dict[str, Any]:
        """Simula erro do Bedrock (ThrottlingException)."""
        raise Exception("ThrottlingException: Rate exceeded")

    def get_explanation_text(self, risk_tier: str) -> str:
        """Retorna o texto da explicação para o tier de risco dado."""
        return self.EXPLANATIONS.get(risk_tier, self.EXPLANATIONS["Low"])


class MockSageMakerResponses:
    """Mock de respostas do SageMaker para testes de inferência.

    Gera predições determinísticas baseadas no user_id (churned → alta probabilidade,
    active → baixa probabilidade).

    Validates: Requirement 10.5 (inferência determinística)
    """

    MODEL_VERSION = "churn-model-v1.0.0-test"
    MODEL_PACKAGE_ARN = (
        "arn:aws:sagemaker:us-east-1:123456789012:model-package/"
        "churn-prediction-models/1"
    )

    # Métricas do modelo pré-treinado (mock)
    MODEL_METRICS: dict[str, float] = {
        "precision": 0.85,
        "recall": 0.82,
        "f1": 0.83,
        "roc_auc": 0.91,
    }

    def predict_batch(
        self, user_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Simula Batch Transform do SageMaker.

        Churned users recebem churn_probability entre 0.65-0.95 (High).
        Active users recebem churn_probability entre 0.05-0.25 (Low).
        """
        predictions: list[dict[str, Any]] = []

        for user_id in user_ids:
            # Seed determinístico baseado no user_id para reprodutibilidade
            seed = int(user_id.replace("-", "")[:8], 16) % 10000
            rng = random.Random(seed)

            if user_id in CHURNED_USER_IDS:
                churn_prob = round(rng.uniform(0.65, 0.95), 4)
                confidence = round(rng.uniform(0.75, 0.95), 4)
            elif user_id in ACTIVE_USER_IDS:
                churn_prob = round(rng.uniform(0.05, 0.25), 4)
                confidence = round(rng.uniform(0.80, 0.95), 4)
            else:
                # Usuário desconhecido — risco médio
                churn_prob = round(rng.uniform(0.30, 0.60), 4)
                confidence = round(rng.uniform(0.50, 0.70), 4)

            # Determinar risk_tier
            if churn_prob <= 0.30:
                risk_tier = "Low"
            elif churn_prob <= 0.60:
                risk_tier = "Medium"
            else:
                risk_tier = "High"

            predictions.append({
                "user_id": user_id,
                "churn_probability": churn_prob,
                "confidence": confidence,
                "risk_tier": risk_tier,
                "model_version": self.MODEL_VERSION,
            })

        return predictions

    def predict_single(self, user_id: str) -> dict[str, Any]:
        """Predição para um único user (conveniência)."""
        return self.predict_batch([user_id])[0]

    def get_model_info(self) -> dict[str, Any]:
        """Retorna metadados do modelo mock."""
        return {
            "model_package_arn": self.MODEL_PACKAGE_ARN,
            "algorithm": "xgboost",
            "training_date": "2024-05-15T10:00:00Z",
            "dataset_version": "features-v42",
            "metrics": self.MODEL_METRICS,
        }

    def get_shap_values(self, user_id: str) -> dict[str, Any]:
        """Gera SHAP values mock para um usuário.

        Retorna top 10 features com contribuição simulada.
        """
        seed = int(user_id.replace("-", "")[:8], 16) % 10000
        rng = random.Random(seed)

        is_churned = user_id in CHURNED_USER_IDS

        feature_names = [
            "sessions_per_week",
            "avg_happiness_score",
            "error_rate",
            "total_viewing_hours",
            "viewing_time_trend",
            "avg_session_duration_min",
            "distinct_channels",
            "avg_buffer_ratio",
            "pct_sport",
            "session_frequency_trend",
        ]

        contributions = []
        for feat in feature_names:
            if is_churned:
                # Churned: features negativas empurram para churn
                weight = round(rng.uniform(0.01, 0.25), 4)
                if feat in ("sessions_per_week", "avg_happiness_score", "total_viewing_hours"):
                    weight = round(rng.uniform(0.10, 0.30), 4)
            else:
                # Active: features empurram para não-churn
                weight = round(rng.uniform(-0.25, -0.01), 4)
                if feat in ("sessions_per_week", "avg_happiness_score", "total_viewing_hours"):
                    weight = round(rng.uniform(-0.30, -0.10), 4)

            normalized = round(max(-1.0, min(1.0, weight * 3)), 4)

            contributions.append({
                "feature_name": feat,
                "contribution_weight": weight,
                "normalized_impact": normalized,
            })

        # Ordenar por abs(weight) descendente
        contributions.sort(key=lambda c: abs(c["contribution_weight"]), reverse=True)

        return {
            "user_id": user_id,
            "top_features": contributions,
            "base_value": 0.35,
            "prediction_value": rng.uniform(0.65, 0.95) if is_churned else rng.uniform(0.05, 0.25),
        }
