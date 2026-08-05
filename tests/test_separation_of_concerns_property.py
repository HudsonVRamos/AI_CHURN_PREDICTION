"""Property-based tests para separação de responsabilidades.

Valida que o SHAP (Explainability_Engine) NUNCA gera texto explicativo,
apenas pesos numéricos. A responsabilidade de gerar explicações em
linguagem natural pertence exclusivamente ao AWS Bedrock.

**Validates: Requirements 10.6, 12.4, 12.5**
**Property 3: Separation of Concerns** — SHAP nunca gera texto
explicativo, apenas pesos numéricos. Bedrock nunca calcula score de churn.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.common.models import (
    ExplainabilityResult,
    FeatureContribution,
    PredictionResult,
)
from src.ml.batch_inference import FEATURE_COLUMNS


# --- Estratégias ---

# SHAP values aleatórios (um por feature)
shap_values_strategy = st.lists(
    st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    min_size=len(FEATURE_COLUMNS),
    max_size=len(FEATURE_COLUMNS),
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

# Feature name válido (de FEATURE_COLUMNS)
feature_name_strategy = st.sampled_from(FEATURE_COLUMNS)

# Top N (1 a 19 features)
top_n_strategy = st.integers(min_value=1, max_value=len(FEATURE_COLUMNS))


# --- Property Tests ---


class TestSHAPOutputIsNumericOnly:
    """Verifica que _select_top_features produz APENAS dados numéricos.

    **Validates: Requirements 10.6, 12.4, 12.5**

    SHAP nunca gera texto explicativo — apenas feature_name (identificador),
    contribution_weight (float) e normalized_impact (float).
    """

    @given(shap_values=shap_values_strategy)
    @settings(max_examples=200, deadline=None)
    def test_select_top_features_output_is_strictly_numeric(
        self, shap_values: list[float]
    ) -> None:
        """_select_top_features retorna SOMENTE dados numéricos, sem texto.

        **Validates: Requirements 10.6, 12.4, 12.5**

        Propriedade: para qualquer array de SHAP values gerado
        aleatoriamente, o output contém apenas:
        - feature_name: string identificadora (coluna do modelo)
        - contribution_weight: float
        - normalized_impact: float

        Nenhum campo de texto explicativo deve existir.
        """
        from unittest.mock import MagicMock, patch
        import pandas as pd

        values_array = np.array(shap_values)

        # Garantir que pelo menos um valor não é zero
        # para evitar edge case de divisão
        assume(np.any(values_array != 0.0))

        # Criar SHAPExplainer com mock
        mock_model = MagicMock()
        training_data = pd.DataFrame(
            np.zeros((10, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        mock_explainer_instance = MagicMock()

        with patch(
            "src.explainability.shap_explainer.shap.TreeExplainer",
            return_value=mock_explainer_instance,
        ):
            from src.explainability.shap_explainer import SHAPExplainer

            explainer = SHAPExplainer(
                model=mock_model,
                training_data=training_data,
                top_n=10,
            )

        # Chamar _select_top_features diretamente
        contributions = explainer._select_top_features(values_array)

        # Verificar que cada contribuição é puramente numérica
        for contrib in contributions:
            assert isinstance(contrib, FeatureContribution)

            # contribution_weight DEVE ser numérico (float)
            assert isinstance(
                contrib.contribution_weight, (int, float)
            ), (
                f"contribution_weight não é numérico: "
                f"tipo={type(contrib.contribution_weight)}, "
                f"valor={contrib.contribution_weight}"
            )

            # normalized_impact DEVE ser numérico (float)
            assert isinstance(
                contrib.normalized_impact, (int, float)
            ), (
                f"normalized_impact não é numérico: "
                f"tipo={type(contrib.normalized_impact)}, "
                f"valor={contrib.normalized_impact}"
            )

            # feature_name DEVE ser um nome de coluna válido
            # (identificador, NÃO texto explicativo)
            assert contrib.feature_name in FEATURE_COLUMNS, (
                f"feature_name inválido: '{contrib.feature_name}' "
                f"não está em FEATURE_COLUMNS"
            )

            # Verificar que NÃO há frases/texto natural no feature_name
            # (nomes de coluna são snake_case curtos, sem espaços)
            assert " " not in contrib.feature_name, (
                f"feature_name contém espaços (parece texto): "
                f"'{contrib.feature_name}'"
            )


class TestExplainabilityResultHasNoTextExplanation:
    """Verifica que ExplainabilityResult NÃO possui campo de explicação textual.

    **Validates: Requirements 10.6, 12.4, 12.5**

    A geração de texto explicativo é responsabilidade exclusiva do
    AWS Bedrock. O SHAP produz apenas dados numéricos estruturados.
    """

    @given(
        user_id=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=20,
        ),
        base_value=st.floats(
            min_value=-5.0,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        prediction_value=st.floats(
            min_value=-5.0,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        weight=contribution_weight_strategy,
        impact=normalized_impact_strategy,
        feature_name=feature_name_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_explainability_result_has_no_text_explanation_field(
        self,
        user_id: str,
        base_value: float,
        prediction_value: float,
        weight: float,
        impact: float,
        feature_name: str,
    ) -> None:
        """ExplainabilityResult não contém campo de explicação textual.

        **Validates: Requirements 10.6, 12.4, 12.5**

        Propriedade: para qualquer combinação válida de inputs,
        o ExplainabilityResult NÃO possui atributos que armazenem
        texto explicativo (explanation, description, summary, text, etc.).
        """
        contrib = FeatureContribution(
            feature_name=feature_name,
            contribution_weight=weight,
            normalized_impact=impact,
        )

        result = ExplainabilityResult(
            user_id=user_id,
            top_features=[contrib],
            base_value=base_value,
            prediction_value=prediction_value,
        )

        # Campos que indicariam texto explicativo (violação de SoC)
        text_fields = [
            "explanation",
            "description",
            "summary",
            "text",
            "narrative",
            "nl_explanation",
            "natural_language",
            "reason",
            "insight",
            "interpretation",
        ]

        for field_name in text_fields:
            assert not hasattr(result, field_name), (
                f"ExplainabilityResult NÃO deveria ter campo "
                f"'{field_name}' — texto é responsabilidade do Bedrock"
            )

        # Verificar que os campos existentes são apenas numéricos/estruturais
        model_fields = ExplainabilityResult.model_fields
        for name, field_info in model_fields.items():
            # Campos permitidos: user_id (identificador), top_features (lista),
            # base_value (float), prediction_value (float)
            assert name in (
                "user_id",
                "top_features",
                "base_value",
                "prediction_value",
            ), (
                f"Campo inesperado em ExplainabilityResult: '{name}'. "
                f"Se for textual, viola separação de responsabilidades."
            )


class TestPredictionResultHasNoTextExplanation:
    """Verifica que PredictionResult NÃO possui campo de explicação textual.

    **Validates: Requirements 10.6, 12.4, 12.5**

    ML inference produz apenas dados numéricos. Texto é gerado pelo Bedrock.
    """

    @given(
        churn_prob=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=200, deadline=None)
    def test_prediction_result_has_no_text_explanation_field(
        self, churn_prob: float
    ) -> None:
        """PredictionResult do batch inference não contém texto explicativo.

        **Validates: Requirements 10.6, 12.4, 12.5**

        Propriedade: para qualquer probabilidade de churn, o
        PredictionResult contém apenas dados numéricos e identificadores.
        Nenhum campo textual explicativo deve existir.
        """
        # Determinar risk_tier consistente
        if churn_prob <= 0.30:
            risk_tier = "Low"
        elif churn_prob <= 0.60:
            risk_tier = "Medium"
        else:
            risk_tier = "High"

        result = PredictionResult(
            user_id="test-user-001",
            churn_probability=churn_prob,
            confidence=abs(churn_prob - 0.5) * 2.0,
            risk_tier=risk_tier,
            model_version="arn:aws:sagemaker:us-east-1:123:model/v1",
            feature_version=1,
            timestamp="2024-01-15T10:00:00+00:00",
        )

        # Campos que indicariam texto explicativo (violação de SoC)
        text_fields = [
            "explanation",
            "description",
            "summary",
            "text",
            "narrative",
            "nl_explanation",
            "natural_language",
            "reason",
            "insight",
            "interpretation",
        ]

        for field_name in text_fields:
            assert not hasattr(result, field_name), (
                f"PredictionResult NÃO deveria ter campo "
                f"'{field_name}' — texto é responsabilidade do Bedrock"
            )


class TestFeatureContributionValuesAreNumeric:
    """Verifica que FeatureContribution contém apenas valores numéricos.

    **Validates: Requirements 10.6, 12.4, 12.5**

    contribution_weight e normalized_impact são SEMPRE numéricos,
    feature_name é SEMPRE um nome de coluna válido de FEATURE_COLUMNS.
    """

    @given(
        feature_name=feature_name_strategy,
        weight=contribution_weight_strategy,
        impact=normalized_impact_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_feature_contribution_values_are_always_numeric(
        self, feature_name: str, weight: float, impact: float
    ) -> None:
        """FeatureContribution tem valores estritamente numéricos.

        **Validates: Requirements 10.6, 12.4, 12.5**

        Propriedade: para qualquer combinação de feature_name,
        contribution_weight e normalized_impact gerados aleatoriamente,
        os valores são sempre numéricos e o feature_name é sempre
        um nome de coluna válido (sem texto explicativo).
        """
        contrib = FeatureContribution(
            feature_name=feature_name,
            contribution_weight=weight,
            normalized_impact=impact,
        )

        # contribution_weight é numérico
        assert isinstance(contrib.contribution_weight, (int, float)), (
            f"contribution_weight deveria ser numérico, "
            f"mas é {type(contrib.contribution_weight)}"
        )

        # normalized_impact é numérico
        assert isinstance(contrib.normalized_impact, (int, float)), (
            f"normalized_impact deveria ser numérico, "
            f"mas é {type(contrib.normalized_impact)}"
        )

        # feature_name é um identificador válido de FEATURE_COLUMNS
        assert contrib.feature_name in FEATURE_COLUMNS, (
            f"feature_name '{contrib.feature_name}' não é uma coluna "
            f"válida — pode ser texto explicativo indevido"
        )

        # feature_name NÃO contém espaços (é identificador, não frase)
        assert " " not in contrib.feature_name, (
            f"feature_name contém espaços: '{contrib.feature_name}'"
        )

        # feature_name NÃO contém caracteres de pontuação de texto
        for char in [".", ",", "!", "?", ":", ";"]:
            assert char not in contrib.feature_name, (
                f"feature_name contém pontuação '{char}': "
                f"'{contrib.feature_name}' — parece texto"
            )

        # FeatureContribution NÃO tem campos de texto explicativo
        text_fields = [
            "explanation",
            "description",
            "summary",
            "text",
            "reason",
        ]
        for field in text_fields:
            assert not hasattr(contrib, field), (
                f"FeatureContribution NÃO deveria ter '{field}'"
            )

    @given(shap_values=shap_values_strategy, top_n=top_n_strategy)
    @settings(max_examples=200, deadline=None)
    def test_all_contributions_from_select_top_features_are_numeric(
        self, shap_values: list[float], top_n: int
    ) -> None:
        """Todas as contribuições de _select_top_features são numéricas.

        **Validates: Requirements 10.6, 12.4, 12.5**

        Propriedade: para qualquer array de SHAP values e qualquer
        valor de top_n, _select_top_features retorna SOMENTE
        FeatureContributions com dados numéricos — sem texto.
        """
        from unittest.mock import MagicMock, patch
        import pandas as pd

        values_array = np.array(shap_values)
        assume(np.any(values_array != 0.0))

        mock_model = MagicMock()
        training_data = pd.DataFrame(
            np.zeros((10, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        mock_explainer_instance = MagicMock()

        with patch(
            "src.explainability.shap_explainer.shap.TreeExplainer",
            return_value=mock_explainer_instance,
        ):
            from src.explainability.shap_explainer import SHAPExplainer

            explainer = SHAPExplainer(
                model=mock_model,
                training_data=training_data,
                top_n=top_n,
            )

        contributions = explainer._select_top_features(values_array)

        # Verificar quantidade retornada
        expected_count = min(top_n, len(FEATURE_COLUMNS))
        assert len(contributions) == expected_count, (
            f"Esperado {expected_count} contribuições, "
            f"obteve {len(contributions)}"
        )

        # Cada contribuição deve ser puramente numérica
        for contrib in contributions:
            # Tipo correto
            assert isinstance(contrib, FeatureContribution)

            # Valores numéricos
            assert isinstance(contrib.contribution_weight, (int, float))
            assert isinstance(contrib.normalized_impact, (int, float))

            # feature_name é identificador válido
            assert contrib.feature_name in FEATURE_COLUMNS

            # normalized_impact dentro do range
            assert -1.0 <= contrib.normalized_impact <= 1.0, (
                f"normalized_impact fora do range: "
                f"{contrib.normalized_impact}"
            )
