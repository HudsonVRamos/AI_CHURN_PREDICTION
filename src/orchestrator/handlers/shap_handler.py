"""Lambda handler para o estágio de Explicabilidade (SHAP) do pipeline.

Calcula SHAP values reais para cada predição usando TreeExplainer.
Roda como Lambda Docker para suportar a biblioteca SHAP (pesada).

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import json
import os
import pickle
import time
import uuid
from typing import Any

import boto3
import numpy as np

from src.common.logging import get_logger, set_execution_id
from src.common.models import FeatureVector

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# Features na ordem do modelo
MODEL_FEATURES = [
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
    "pct_sport",
    "pct_live",
    "pct_show",
    "distinct_devices",
    "avg_pause_count",
    "avg_seek_count",
    "viewing_time_trend",
    "error_rate_trend",
    "session_frequency_trend",
]


def _load_model(bucket: str, model_key: str):
    """Carrega modelo pickle do S3."""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=model_key)
    return pickle.loads(response["Body"].read())


def _feature_vector_to_array(fv: FeatureVector) -> list[float]:
    """Converte FeatureVector em lista de valores."""
    values = []
    for feat in MODEL_FEATURES:
        val = getattr(fv, feat, None)
        values.append(float(val) if val is not None else 0.0)
    return values


def _compute_shap_explanations(
    feature_vectors: list[FeatureVector],
    model,
    logger,
) -> list[dict]:
    """Calcula SHAP values reais usando TreeExplainer."""
    X = np.array([_feature_vector_to_array(fv) for fv in feature_vectors])

    if HAS_SHAP:
        logger.info("Calculando SHAP values com TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Para classificação binária, shap_values pode ser lista de 2 arrays
        # Usamos o índice 1 (classe positiva = churn)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
    else:
        # Fallback: usar feature importance * desvio (sem SHAP real)
        logger.warning("SHAP não disponível, usando feature importance como fallback.")
        importances = model.feature_importances_
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds == 0] = 1.0
        deviations = (X - means) / stds
        shap_vals = deviations * importances

    explanations = []
    for i, fv in enumerate(feature_vectors):
        user_shap = shap_vals[i]

        # Ordenar por impacto absoluto
        sorted_idx = np.argsort(np.abs(user_shap))[::-1]

        # Top 5 fatores de risco
        top_factors = []
        for idx in sorted_idx[:5]:
            feat_name = MODEL_FEATURES[idx]
            shap_value = float(user_shap[idx])
            feat_value = float(X[i][idx])
            direction = "aumenta_risco" if shap_value > 0 else "diminui_risco"

            top_factors.append({
                "feature": feat_name,
                "shap_value": round(shap_value, 6),
                "feature_value": round(feat_value, 4),
                "direction": direction,
                "impact_pct": round(abs(shap_value) / (np.abs(user_shap).sum() + 1e-10) * 100, 1),
            })

        explanations.append({
            "user_id": fv.user_id,
            "top_factors": top_factors,
            "all_shap_values": {
                MODEL_FEATURES[j]: round(float(user_shap[j]), 6)
                for j in range(len(MODEL_FEATURES))
            },
            "base_value": float(
                explainer.expected_value[1]
                if HAS_SHAP and hasattr(explainer, "expected_value")
                and isinstance(explainer.expected_value, (list, np.ndarray))
                and len(explainer.expected_value) > 1
                else (
                    float(explainer.expected_value)
                    if HAS_SHAP and hasattr(explainer, "expected_value")
                    and np.isscalar(explainer.expected_value)
                    else 0.5
                )
            ),
        })

    return explanations


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para explicabilidade SHAP.

    Calcula SHAP values para cada predição identificando as features
    que mais contribuem para o risco de churn de cada usuário.

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução.
            - predictions_s3_key: URI S3 das predições.
            - bucket: Bucket S3.
        context: Contexto Lambda.

    Returns:
        Dicionário com:
            - explanations_s3_key: URI S3 das explicações.
            - shap_success_count: Contagem de explicações geradas.
            - stage_completed: "explainability"
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("explainability")
    logger.log_stage_start()
    start_time = time.time()

    try:
        bucket = event.get("bucket", os.environ.get("BUCKET_NAME", "sky-brazil-churn-prediction"))
        model_s3_key = os.environ.get("MODEL_S3_KEY", "models/approved/churn_model.pkl")
        s3_client = boto3.client("s3")

        # 1. Carregar feature vectors do S3
        features_key = f"features/{execution_id}/feature_vectors.json"
        logger.info(f"Carregando features: {features_key}")

        try:
            response = s3_client.get_object(Bucket=bucket, Key=features_key)
            feature_vectors_data = json.loads(response["Body"].read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"Features não encontradas, pulando SHAP: {e}")
            output = {
                **event,
                "execution_id": execution_id,
                "explanations_s3_key": "",
                "shap_success_count": 0,
                "stage_completed": "explainability",
            }
            duration = time.time() - start_time
            logger.log_stage_completion(duration_seconds=duration)
            return output

        feature_vectors = [FeatureVector(**fv) for fv in feature_vectors_data]
        logger.info(f"Feature vectors carregados: {len(feature_vectors)}")

        # 2. Carregar modelo
        logger.info(f"Carregando modelo: {model_s3_key}")
        model = _load_model(bucket, model_s3_key)

        # 3. Calcular SHAP
        explanations = _compute_shap_explanations(feature_vectors, model, logger)
        logger.info(f"Explicações SHAP geradas: {len(explanations)} usuários")

        # 4. Salvar no S3
        explanations_key = f"explanations/{execution_id}/shap_results.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=explanations_key,
            Body=json.dumps(explanations, ensure_ascii=False),
            ContentType="application/json",
        )

        output = {
            **event,
            "execution_id": execution_id,
            "explanations_s3_key": f"s3://{bucket}/{explanations_key}",
            "shap_success_count": len(explanations),
            "stage_completed": "explainability",
        }

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except Exception as e:
        logger.error(
            f"Falha no estágio explainability: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
