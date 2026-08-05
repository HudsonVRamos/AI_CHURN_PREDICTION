"""Módulo de explicabilidade via SHAP (SHapley Additive exPlanations).

Calcula a importância de cada feature para uma predição individual usando
SHAP TreeExplainer (otimizado para modelos tree-based: XGBoost, LightGBM, CatBoost).

Para cada assinante, retorna as top N features com maior impacto na classificação,
incluindo contribution_weight (signed float) e normalized_impact (-1.0 a 1.0).

Se o cálculo SHAP falhar para um usuário, a predição permanece válida e o resultado
de explicabilidade é None (degradação graciosa).

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.common.logging import get_logger
from src.common.models import (
    ExplainabilityResult,
    FeatureContribution,
    FeatureVector,
)
from src.ml.batch_inference import FEATURE_COLUMNS

logger = get_logger("explainability")


class SHAPExplainer:
    """Calcula importância de features usando SHAP TreeExplainer.

    Utiliza o algoritmo SHAP para decompor cada predição em contribuições
    individuais por feature. O TreeExplainer é otimizado para modelos
    baseados em árvore (XGBoost, LightGBM, CatBoost), oferecendo
    cálculo exato e eficiente dos SHAP values.

    Attributes:
        TOP_FEATURES_DEFAULT: Número padrão de top features retornadas.
    """

    TOP_FEATURES_DEFAULT = 10

    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        top_n: int | None = None,
    ) -> None:
        """Inicializa o SHAP Explainer com TreeExplainer.

        Args:
            model: Modelo treinado (XGBoost, LightGBM ou CatBoost).
            training_data: DataFrame com dados de treino para o background
                do TreeExplainer (deve conter as colunas FEATURE_COLUMNS).
            top_n: Número de top features a retornar (default: 10).
        """
        self._model = model
        self._training_data = training_data
        self._top_n = top_n or self.TOP_FEATURES_DEFAULT
        self._explainer = shap.TreeExplainer(model, data=training_data)

        logger.info(
            f"SHAPExplainer inicializado: "
            f"top_n={self._top_n}, "
            f"training_samples={len(training_data)}"
        )

    def explain(
        self,
        feature_vector: FeatureVector,
    ) -> ExplainabilityResult | None:
        """Calcula SHAP values para uma predição individual.

        Converte o FeatureVector em DataFrame row, calcula os SHAP values
        e retorna as top N features com maior impacto absoluto.

        Se o cálculo falhar (ex.: dados incompatíveis, erro numérico),
        retorna None e a predição permanece válida (R11.6).

        Args:
            feature_vector: Feature vector do assinante.

        Returns:
            ExplainabilityResult com top features, ou None se falhar.
        """
        try:
            # Converter FeatureVector em DataFrame row
            input_df = self._feature_vector_to_dataframe(feature_vector)

            # Calcular SHAP values
            shap_values = self._explainer.shap_values(input_df)

            # shap_values pode ser array 2D (1 amostra x N features)
            if isinstance(shap_values, list):
                # Para classificação binária, usar classe positiva (churn)
                values = np.array(shap_values[1]).flatten()
            else:
                values = np.array(shap_values).flatten()

            # Base value do explainer
            base_value = self._get_base_value()

            # Prediction value = base_value + sum(shap_values)
            prediction_value = float(base_value + np.sum(values))

            # Selecionar top N features por impacto absoluto
            top_features = self._select_top_features(values)

            result = ExplainabilityResult(
                user_id=feature_vector.user_id,
                top_features=top_features,
                base_value=float(base_value),
                prediction_value=prediction_value,
            )

            logger.info(
                f"SHAP calculado para user={feature_vector.user_id}: "
                f"top_features={len(top_features)}, "
                f"prediction_value={prediction_value:.4f}"
            )

            return result

        except Exception as e:
            # R11.6: degradação graciosa — predição permanece válida
            logger.error(
                f"Falha ao calcular SHAP para user={feature_vector.user_id}: "
                f"{type(e).__name__}: {e}"
            )
            return None

    def explain_batch(
        self,
        feature_vectors: list[FeatureVector],
    ) -> list[ExplainabilityResult | None]:
        """Calcula SHAP values para um batch de predições.

        Processa cada feature vector individualmente. Se o cálculo falhar
        para um usuário específico, retorna None naquela posição (R11.6).

        Args:
            feature_vectors: Lista de feature vectors dos assinantes.

        Returns:
            Lista de ExplainabilityResult (ou None para falhas individuais),
            na mesma ordem dos inputs.
        """
        logger.info(
            f"Iniciando SHAP batch: {len(feature_vectors)} usuários"
        )

        results: list[ExplainabilityResult | None] = []
        success_count = 0
        failure_count = 0

        for fv in feature_vectors:
            result = self.explain(fv)
            results.append(result)
            if result is not None:
                success_count += 1
            else:
                failure_count += 1

        logger.info(
            f"SHAP batch concluído: "
            f"sucesso={success_count}, falhas={failure_count}, "
            f"total={len(feature_vectors)}"
        )

        return results

    def _feature_vector_to_dataframe(
        self,
        feature_vector: FeatureVector,
    ) -> pd.DataFrame:
        """Converte um FeatureVector em DataFrame row para SHAP.

        Extrai apenas os campos numéricos na ordem definida por
        FEATURE_COLUMNS (mesma ordem utilizada no treinamento).
        Valores None (trends) são substituídos por 0.0.

        Args:
            feature_vector: Feature vector a converter.

        Returns:
            DataFrame com uma linha e colunas em FEATURE_COLUMNS.
        """
        row_data: dict[str, float] = {}
        for col in FEATURE_COLUMNS:
            value = getattr(feature_vector, col)
            row_data[col] = float(value) if value is not None else 0.0

        return pd.DataFrame([row_data], columns=FEATURE_COLUMNS)

    def _get_base_value(self) -> float:
        """Extrai o base value (expected value) do explainer.

        Returns:
            Float com o valor base do modelo.
        """
        expected_value = self._explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            # Classificação binária: usar classe positiva (índice 1)
            return float(expected_value[1])
        return float(expected_value)

    def _select_top_features(
        self,
        shap_values: np.ndarray,
    ) -> list[FeatureContribution]:
        """Seleciona as top N features com maior impacto absoluto.

        Ordena os SHAP values por valor absoluto (decrescente) e retorna
        as top N como FeatureContribution com contribution_weight (raw SHAP)
        e normalized_impact (normalizado para [-1.0, 1.0]).

        Args:
            shap_values: Array de SHAP values (1D, uma por feature).

        Returns:
            Lista de FeatureContribution ordenada por impacto absoluto.
        """
        # Índices ordenados por impacto absoluto (decrescente)
        abs_values = np.abs(shap_values)
        sorted_indices = np.argsort(abs_values)[::-1]

        # Top N features
        top_indices = sorted_indices[:self._top_n]

        # Normalizar: dividir pelo max absoluto para range [-1.0, 1.0]
        max_abs = float(abs_values.max())
        if max_abs == 0.0:
            max_abs = 1.0  # Evitar divisão por zero

        contributions: list[FeatureContribution] = []
        for idx in top_indices:
            feature_name = FEATURE_COLUMNS[idx]
            raw_value = float(shap_values[idx])
            normalized = raw_value / max_abs

            # Clamp para garantir range [-1.0, 1.0]
            normalized = max(-1.0, min(1.0, normalized))

            contributions.append(
                FeatureContribution(
                    feature_name=feature_name,
                    contribution_weight=round(raw_value, 6),
                    normalized_impact=round(normalized, 6),
                )
            )

        return contributions
