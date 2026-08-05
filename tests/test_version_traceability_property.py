"""Property-based tests para rastreabilidade de versão.

Valida que cada PredictionResult registra exatamente qual versão de modelo
e features foi usada, garantindo auditoria e rastreabilidade completa.

**Validates: Requirements 15.4, 17.1**
**Property 4: Version Traceability** — Cada predição registra exatamente
qual versão de modelo e features foi usada.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.common.models import PredictionResult


# --- Estratégias ---

# model_version: strings não-vazias (mínimo 1 caractere)
model_version_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip() != "")

# feature_version: inteiros >= 1
feature_version_strategy = st.integers(min_value=1, max_value=1_000_000)

# churn_probability: floats entre 0.0 e 1.0
churn_probability_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# confidence: floats entre 0.0 e 1.0
confidence_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)


def _determine_risk_tier(prob: float) -> str:
    """Determina o risk_tier consistente com a probabilidade."""
    if prob <= 0.30:
        return "Low"
    elif prob <= 0.60:
        return "Medium"
    else:
        return "High"


# --- Property Tests ---


class TestVersionTraceabilityModelVersion:
    """Testes de propriedade para rastreabilidade de model_version."""

    @given(
        model_version=model_version_strategy,
        feature_version=feature_version_strategy,
        churn_probability=churn_probability_strategy,
        confidence=confidence_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_prediction_result_always_has_non_empty_model_version(
        self,
        model_version: str,
        feature_version: int,
        churn_probability: float,
        confidence: float,
    ) -> None:
        """Toda PredictionResult sempre contém um model_version não-vazio.

        **Validates: Requirements 15.4, 17.1**

        Propriedade: para qualquer combinação válida de inputs, o
        PredictionResult armazena um model_version com pelo menos 1
        caractere, garantindo rastreabilidade do modelo usado.
        """
        risk_tier = _determine_risk_tier(churn_probability)

        prediction = PredictionResult(
            user_id="test-user-001",
            churn_probability=churn_probability,
            confidence=confidence,
            risk_tier=risk_tier,
            model_version=model_version,
            feature_version=feature_version,
            timestamp="2024-01-15T10:00:00Z",
        )

        assert prediction.model_version is not None, (
            "model_version não pode ser None"
        )
        assert len(prediction.model_version) >= 1, (
            f"model_version deve ter pelo menos 1 caractere, "
            f"mas tem {len(prediction.model_version)}"
        )
        assert prediction.model_version.strip() != "", (
            f"model_version não pode ser uma string vazia ou apenas espaços: "
            f"'{prediction.model_version}'"
        )


class TestVersionTraceabilityFeatureVersion:
    """Testes de propriedade para rastreabilidade de feature_version."""

    @given(
        model_version=model_version_strategy,
        feature_version=feature_version_strategy,
        churn_probability=churn_probability_strategy,
        confidence=confidence_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_prediction_result_always_has_positive_feature_version(
        self,
        model_version: str,
        feature_version: int,
        churn_probability: float,
        confidence: float,
    ) -> None:
        """Toda PredictionResult sempre contém feature_version >= 1.

        **Validates: Requirements 15.4, 17.1**

        Propriedade: para qualquer combinação válida de inputs, o
        PredictionResult armazena um feature_version positivo (>= 1),
        garantindo rastreabilidade da versão de features usada.
        """
        risk_tier = _determine_risk_tier(churn_probability)

        prediction = PredictionResult(
            user_id="test-user-001",
            churn_probability=churn_probability,
            confidence=confidence,
            risk_tier=risk_tier,
            model_version=model_version,
            feature_version=feature_version,
            timestamp="2024-01-15T10:00:00Z",
        )

        assert prediction.feature_version is not None, (
            "feature_version não pode ser None"
        )
        assert prediction.feature_version >= 1, (
            f"feature_version deve ser >= 1, "
            f"mas é {prediction.feature_version}"
        )


class TestVersionTraceabilityImmutability:
    """Testes de propriedade para imutabilidade das versões registradas."""

    @given(
        model_version=model_version_strategy,
        feature_version=feature_version_strategy,
        churn_probability=churn_probability_strategy,
        confidence=confidence_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_prediction_result_versions_are_never_mutated(
        self,
        model_version: str,
        feature_version: int,
        churn_probability: float,
        confidence: float,
    ) -> None:
        """model_version e feature_version não são mutados após criação.

        **Validates: Requirements 15.4, 17.1**

        Propriedade: ao criar um PredictionResult com versões específicas,
        os valores retornados são exatamente os mesmos fornecidos na
        construção. As versões são registradas fielmente.
        """
        risk_tier = _determine_risk_tier(churn_probability)

        prediction = PredictionResult(
            user_id="test-user-001",
            churn_probability=churn_probability,
            confidence=confidence,
            risk_tier=risk_tier,
            model_version=model_version,
            feature_version=feature_version,
            timestamp="2024-01-15T10:00:00Z",
        )

        assert prediction.model_version == model_version, (
            f"model_version foi mutado: "
            f"esperado='{model_version}', obtido='{prediction.model_version}'"
        )
        assert prediction.feature_version == feature_version, (
            f"feature_version foi mutado: "
            f"esperado={feature_version}, obtido={prediction.feature_version}"
        )
