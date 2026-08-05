"""Property-based tests para inferência determinística.

Valida que para o mesmo conjunto de inputs e modelo, a inferência
sempre produz o mesmo resultado. Testa as funções puras do
BatchInferenceProcessor que garantem determinismo:
- _compute_confidence: mesma probabilidade → mesma confiança
- _determine_risk_tier: mesma probabilidade → mesmo tier

**Validates: Requirements 10.5**
**Property 1: Deterministic Inference** — Mesmo FeatureVector + mesmo
modelo = mesma predição.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.ml.batch_inference import BatchInferenceProcessor, DEFAULT_RISK_THRESHOLDS


# --- Estratégias ---

# Probabilidades válidas entre 0.0 e 1.0
probability_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Offset simétrico: distância de 0.5 (0.0 a 0.5)
symmetric_offset_strategy = st.floats(
    min_value=0.0,
    max_value=0.5,
    allow_nan=False,
    allow_infinity=False,
)

# Pares de probabilidades para testar monotonicidade
monotonic_pair_strategy = st.tuples(
    st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
)


# --- Property Tests ---


class TestDeterministicConfidence:
    """Testes de propriedade para _compute_confidence."""

    @given(prob=probability_strategy)
    @settings(max_examples=200, deadline=None)
    def test_same_probability_always_returns_same_confidence(
        self, prob: float
    ) -> None:
        """Para qualquer probabilidade válida, _compute_confidence retorna
        sempre o mesmo valor quando chamada múltiplas vezes.

        **Validates: Requirements 10.5**

        Propriedade: _compute_confidence é uma função pura e determinística.
        f(x) chamada N vezes com o mesmo x → mesmo resultado.
        """
        result1 = BatchInferenceProcessor._compute_confidence(prob)
        result2 = BatchInferenceProcessor._compute_confidence(prob)
        result3 = BatchInferenceProcessor._compute_confidence(prob)

        assert result1 == result2 == result3, (
            f"_compute_confidence não é determinística para prob={prob}: "
            f"resultados={result1}, {result2}, {result3}"
        )

    @given(prob=probability_strategy)
    @settings(max_examples=200, deadline=None)
    def test_confidence_matches_formula(self, prob: float) -> None:
        """Para qualquer probabilidade válida, confidence = |prob - 0.5| * 2.

        **Validates: Requirements 10.5**

        Propriedade: a implementação segue exatamente a fórmula definida.
        """
        result = BatchInferenceProcessor._compute_confidence(prob)
        expected = abs(prob - 0.5) * 2.0

        assert result == pytest.approx(expected, abs=1e-10), (
            f"Confiança incorreta para prob={prob}: "
            f"esperado={expected}, obtido={result}"
        )

    @given(offset=symmetric_offset_strategy)
    @settings(max_examples=200, deadline=None)
    def test_confidence_is_symmetric_around_half(
        self, offset: float
    ) -> None:
        """A função de confiança é simétrica: f(0.5 + x) == f(0.5 - x).

        **Validates: Requirements 10.5**

        Propriedade: probabilidades equidistantes de 0.5 (por cima e por
        baixo) produzem a mesma confiança. Isso garante que o modelo
        trata com igual certeza tanto churn alto quanto churn baixo.
        """
        prob_above = 0.5 + offset
        prob_below = 0.5 - offset

        confidence_above = BatchInferenceProcessor._compute_confidence(
            prob_above
        )
        confidence_below = BatchInferenceProcessor._compute_confidence(
            prob_below
        )

        assert confidence_above == pytest.approx(
            confidence_below, abs=1e-10
        ), (
            f"Confiança não simétrica: "
            f"f(0.5 + {offset}) = {confidence_above}, "
            f"f(0.5 - {offset}) = {confidence_below}"
        )

    @given(pair=monotonic_pair_strategy)
    @settings(max_examples=200, deadline=None)
    def test_confidence_monotonically_increasing_from_half(
        self, pair: tuple[float, float]
    ) -> None:
        """Maior distância de 0.5 → maior confiança (monotonicidade).

        **Validates: Requirements 10.5**

        Propriedade: se |a - 0.5| > |b - 0.5|, então
        confidence(a) >= confidence(b). A confiança cresce
        monotonicamente com a distância do ponto de incerteza máxima.
        """
        a, b = pair
        dist_a = abs(a - 0.5)
        dist_b = abs(b - 0.5)

        conf_a = BatchInferenceProcessor._compute_confidence(a)
        conf_b = BatchInferenceProcessor._compute_confidence(b)

        if dist_a > dist_b:
            assert conf_a >= conf_b, (
                f"Monotonicidade violada: dist({a})={dist_a:.6f} > "
                f"dist({b})={dist_b:.6f}, mas conf({a})={conf_a:.6f} < "
                f"conf({b})={conf_b:.6f}"
            )
        elif dist_a < dist_b:
            assert conf_a <= conf_b, (
                f"Monotonicidade violada: dist({a})={dist_a:.6f} < "
                f"dist({b})={dist_b:.6f}, mas conf({a})={conf_a:.6f} > "
                f"conf({b})={conf_b:.6f}"
            )
        else:
            # Distâncias iguais → confiança igual
            assert conf_a == pytest.approx(conf_b, abs=1e-10)


class TestDeterministicRiskTier:
    """Testes de propriedade para _determine_risk_tier."""

    @given(prob=probability_strategy)
    @settings(max_examples=200, deadline=None)
    def test_same_probability_always_returns_same_tier(
        self, prob: float
    ) -> None:
        """Para qualquer probabilidade válida, _determine_risk_tier retorna
        sempre o mesmo tier.

        **Validates: Requirements 10.5**

        Propriedade: _determine_risk_tier é determinística — mesma
        probabilidade sempre mapeia para o mesmo tier de risco.
        """
        processor = BatchInferenceProcessor(
            sagemaker_pipeline=None,
            bucket="test-bucket",
            s3_client=None,
            dynamodb_resource=None,
        )

        result1 = processor._determine_risk_tier(prob)
        result2 = processor._determine_risk_tier(prob)
        result3 = processor._determine_risk_tier(prob)

        assert result1 == result2 == result3, (
            f"_determine_risk_tier não é determinística para prob={prob}: "
            f"resultados={result1}, {result2}, {result3}"
        )

    @given(prob=probability_strategy)
    @settings(max_examples=200, deadline=None)
    def test_risk_tier_follows_threshold_rules(
        self, prob: float
    ) -> None:
        """O tier retornado é consistente com os thresholds definidos.

        **Validates: Requirements 10.5**

        Propriedade: para os thresholds padrão (low_max=0.30,
        medium_max=0.60), o mapeamento é:
        - prob <= 0.30 → "Low"
        - 0.30 < prob <= 0.60 → "Medium"
        - prob > 0.60 → "High"
        """
        processor = BatchInferenceProcessor(
            sagemaker_pipeline=None,
            bucket="test-bucket",
            s3_client=None,
            dynamodb_resource=None,
        )

        tier = processor._determine_risk_tier(prob)

        low_max = DEFAULT_RISK_THRESHOLDS["low_max"]
        medium_max = DEFAULT_RISK_THRESHOLDS["medium_max"]

        if prob <= low_max:
            assert tier == "Low", (
                f"prob={prob:.6f} <= {low_max} deveria ser 'Low', "
                f"mas é '{tier}'"
            )
        elif prob <= medium_max:
            assert tier == "Medium", (
                f"{low_max} < prob={prob:.6f} <= {medium_max} "
                f"deveria ser 'Medium', mas é '{tier}'"
            )
        else:
            assert tier == "High", (
                f"prob={prob:.6f} > {medium_max} deveria ser 'High', "
                f"mas é '{tier}'"
            )

    @given(prob=probability_strategy)
    @settings(max_examples=200, deadline=None)
    def test_risk_tier_returns_valid_value(
        self, prob: float
    ) -> None:
        """O tier retornado é sempre um dos três valores válidos.

        **Validates: Requirements 10.5**

        Propriedade: para qualquer probabilidade entre 0.0 e 1.0,
        o resultado é sempre "Low", "Medium" ou "High".
        """
        processor = BatchInferenceProcessor(
            sagemaker_pipeline=None,
            bucket="test-bucket",
            s3_client=None,
            dynamodb_resource=None,
        )

        tier = processor._determine_risk_tier(prob)

        assert tier in {"Low", "Medium", "High"}, (
            f"Tier inválido para prob={prob}: '{tier}'. "
            f"Esperado: 'Low', 'Medium' ou 'High'"
        )
