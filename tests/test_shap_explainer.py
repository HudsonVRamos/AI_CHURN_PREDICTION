"""Testes unitários para o módulo src.explainability.shap_explainer.

Valida:
- Inicialização do SHAPExplainer com TreeExplainer
- explain() para predição individual (top N features, normalização)
- explain_batch() para múltiplas predições
- Degradação graciosa: SHAP falha → retorna None (R11.6)
- contribution_weight (signed float) e normalized_impact (-1.0 a 1.0)
- Top N features default (10) e customizado

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.common.models import (
    ExplainabilityResult,
    FeatureContribution,
    FeatureVector,
)
from src.explainability.shap_explainer import SHAPExplainer
from src.ml.batch_inference import FEATURE_COLUMNS


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def sample_feature_vector() -> FeatureVector:
    """FeatureVector de exemplo para testes."""
    return FeatureVector(
        user_id="user-001",
        version=1,
        generated_at="2024-01-15T10:00:00+00:00",
        observation_start="2023-07-15T00:00:00+00:00",
        observation_end="2024-01-15T00:00:00+00:00",
        total_sessions=120,
        total_viewing_hours=85.5,
        avg_session_duration_min=42.5,
        sessions_per_week=4.8,
        distinct_channels=12,
        avg_happiness_score=7.2,
        avg_buffer_ratio=0.02,
        error_rate=0.05,
        avg_bitrate=5000000.0,
        pct_episode=40.0,
        pct_sport=25.0,
        pct_live=20.0,
        pct_show=15.0,
        distinct_devices=3,
        avg_pause_count=2.1,
        avg_seek_count=1.5,
        viewing_time_trend=0.3,
        error_rate_trend=-0.01,
        session_frequency_trend=0.2,
    )


@pytest.fixture
def sample_feature_vectors() -> list[FeatureVector]:
    """Lista de FeatureVectors para testes de batch."""
    base_kwargs = {
        "version": 1,
        "generated_at": "2024-01-15T10:00:00+00:00",
        "observation_start": "2023-07-15T00:00:00+00:00",
        "observation_end": "2024-01-15T00:00:00+00:00",
        "total_sessions": 100,
        "total_viewing_hours": 50.0,
        "avg_session_duration_min": 30.0,
        "sessions_per_week": 4.0,
        "distinct_channels": 8,
        "avg_happiness_score": 6.5,
        "avg_buffer_ratio": 0.03,
        "error_rate": 0.08,
        "avg_bitrate": 4000000.0,
        "pct_episode": 50.0,
        "pct_sport": 20.0,
        "pct_live": 15.0,
        "pct_show": 15.0,
        "distinct_devices": 2,
        "avg_pause_count": 1.5,
        "avg_seek_count": 1.0,
        "viewing_time_trend": 0.1,
        "error_rate_trend": 0.0,
        "session_frequency_trend": -0.1,
    }
    return [
        FeatureVector(user_id=f"user-{i:03d}", **base_kwargs)
        for i in range(1, 4)
    ]


@pytest.fixture
def training_data() -> pd.DataFrame:
    """DataFrame de treino para o TreeExplainer (background data)."""
    rng = np.random.default_rng(42)
    n_samples = 50
    data = {}
    for col in FEATURE_COLUMNS:
        data[col] = rng.random(n_samples) * 10.0
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


@pytest.fixture
def mock_shap_values() -> np.ndarray:
    """SHAP values simulados para 19 features."""
    # Valores com sinais mistos para testar normalização
    return np.array([
        0.15, -0.08, 0.22, 0.05, -0.03,
        0.30, -0.12, 0.18, -0.01, 0.07,
        -0.04, 0.02, -0.06, 0.09, -0.11,
        0.01, -0.25, 0.03, -0.02
    ])


@pytest.fixture
def mock_explainer(mock_shap_values):
    """Mock do shap.TreeExplainer."""
    explainer = MagicMock()
    # shap_values retorna array 1D (regressão) por padrão
    explainer.shap_values.return_value = mock_shap_values.reshape(1, -1)
    explainer.expected_value = 0.5
    return explainer


@pytest.fixture
def shap_explainer(training_data, mock_explainer):
    """SHAPExplainer com TreeExplainer mockado."""
    mock_model = MagicMock()
    with patch("src.explainability.shap_explainer.shap.TreeExplainer",
               return_value=mock_explainer):
        explainer = SHAPExplainer(
            model=mock_model,
            training_data=training_data,
        )
    # Substituir o explainer interno pelo mock
    explainer._explainer = mock_explainer
    return explainer


# ---------------------------------------------------------------
# Testes de inicialização
# ---------------------------------------------------------------


class TestSHAPExplainerInit:
    """Testes da inicialização do SHAPExplainer."""

    def test_top_features_default(self, training_data):
        """TOP_FEATURES_DEFAULT deve ser 10."""
        assert SHAPExplainer.TOP_FEATURES_DEFAULT == 10

    def test_inicializa_com_tree_explainer(self, training_data):
        """Deve criar um shap.TreeExplainer internamente."""
        mock_model = MagicMock()
        with patch("src.explainability.shap_explainer.shap.TreeExplainer") as mock_te:
            mock_te.return_value = MagicMock()
            explainer = SHAPExplainer(
                model=mock_model,
                training_data=training_data,
            )
            mock_te.assert_called_once_with(mock_model, data=training_data)

    def test_top_n_customizado(self, training_data):
        """Deve aceitar top_n customizado."""
        mock_model = MagicMock()
        with patch("src.explainability.shap_explainer.shap.TreeExplainer") as mock_te:
            mock_te.return_value = MagicMock()
            explainer = SHAPExplainer(
                model=mock_model,
                training_data=training_data,
                top_n=5,
            )
            assert explainer._top_n == 5

    def test_top_n_none_usa_default(self, training_data):
        """top_n=None deve usar TOP_FEATURES_DEFAULT (10)."""
        mock_model = MagicMock()
        with patch("src.explainability.shap_explainer.shap.TreeExplainer") as mock_te:
            mock_te.return_value = MagicMock()
            explainer = SHAPExplainer(
                model=mock_model,
                training_data=training_data,
                top_n=None,
            )
            assert explainer._top_n == 10


# ---------------------------------------------------------------
# Testes de explain()
# ---------------------------------------------------------------


class TestExplain:
    """Testes do método explain() para predição individual."""

    def test_retorna_explainability_result(
        self, shap_explainer, sample_feature_vector
    ):
        """explain() deve retornar ExplainabilityResult."""
        result = shap_explainer.explain(sample_feature_vector)
        assert isinstance(result, ExplainabilityResult)

    def test_user_id_no_resultado(
        self, shap_explainer, sample_feature_vector
    ):
        """Resultado deve conter o user_id do input."""
        result = shap_explainer.explain(sample_feature_vector)
        assert result.user_id == "user-001"

    def test_top_features_tem_contribuicoes(
        self, shap_explainer, sample_feature_vector
    ):
        """Resultado deve conter lista de FeatureContribution."""
        result = shap_explainer.explain(sample_feature_vector)
        assert len(result.top_features) > 0
        for fc in result.top_features:
            assert isinstance(fc, FeatureContribution)

    def test_top_features_default_10(
        self, shap_explainer, sample_feature_vector
    ):
        """Default deve retornar 10 top features."""
        result = shap_explainer.explain(sample_feature_vector)
        assert len(result.top_features) == 10

    def test_top_features_customizado(
        self, training_data, mock_explainer, sample_feature_vector
    ):
        """top_n customizado deve retornar N features."""
        mock_model = MagicMock()
        with patch("src.explainability.shap_explainer.shap.TreeExplainer",
                   return_value=mock_explainer):
            explainer = SHAPExplainer(
                model=mock_model,
                training_data=training_data,
                top_n=5,
            )
        explainer._explainer = mock_explainer
        result = explainer.explain(sample_feature_vector)
        assert len(result.top_features) == 5

    def test_features_ordenadas_por_impacto_absoluto(
        self, shap_explainer, sample_feature_vector
    ):
        """Top features devem estar ordenadas por impacto absoluto decrescente."""
        result = shap_explainer.explain(sample_feature_vector)
        abs_impacts = [abs(fc.normalized_impact) for fc in result.top_features]
        # Verificar que está em ordem decrescente
        for i in range(len(abs_impacts) - 1):
            assert abs_impacts[i] >= abs_impacts[i + 1]

    def test_contribution_weight_signed(
        self, shap_explainer, sample_feature_vector
    ):
        """contribution_weight deve ser signed (pode ser positivo ou negativo)."""
        result = shap_explainer.explain(sample_feature_vector)
        weights = [fc.contribution_weight for fc in result.top_features]
        # Com nossos mock SHAP values, devemos ter positivos e negativos
        has_positive = any(w > 0 for w in weights)
        has_negative = any(w < 0 for w in weights)
        assert has_positive
        assert has_negative

    def test_normalized_impact_entre_menos_um_e_um(
        self, shap_explainer, sample_feature_vector
    ):
        """normalized_impact deve estar no range [-1.0, 1.0]."""
        result = shap_explainer.explain(sample_feature_vector)
        for fc in result.top_features:
            assert -1.0 <= fc.normalized_impact <= 1.0

    def test_normalized_impact_max_absoluto_eh_um(
        self, shap_explainer, sample_feature_vector
    ):
        """A feature com maior impacto deve ter normalized_impact = ±1.0."""
        result = shap_explainer.explain(sample_feature_vector)
        max_impact = max(abs(fc.normalized_impact) for fc in result.top_features)
        assert abs(max_impact - 1.0) < 1e-6

    def test_base_value_presente(
        self, shap_explainer, sample_feature_vector
    ):
        """Resultado deve conter base_value do modelo."""
        result = shap_explainer.explain(sample_feature_vector)
        assert isinstance(result.base_value, float)

    def test_prediction_value_presente(
        self, shap_explainer, sample_feature_vector
    ):
        """Resultado deve conter prediction_value (base + sum SHAP)."""
        result = shap_explainer.explain(sample_feature_vector)
        assert isinstance(result.prediction_value, float)

    def test_feature_names_validos(
        self, shap_explainer, sample_feature_vector
    ):
        """feature_name deve ser um nome de coluna válido (FEATURE_COLUMNS)."""
        result = shap_explainer.explain(sample_feature_vector)
        for fc in result.top_features:
            assert fc.feature_name in FEATURE_COLUMNS


# ---------------------------------------------------------------
# Testes de explain() com classificação binária
# ---------------------------------------------------------------


class TestExplainBinaryClassification:
    """Testes para modelos de classificação binária (retornam lista de arrays)."""

    def test_shap_values_lista_usa_classe_positiva(
        self, training_data, sample_feature_vector
    ):
        """Para classificação binária (lista), usar classe 1 (churn)."""
        mock_model = MagicMock()
        mock_exp = MagicMock()
        # Simular output de classificação binária: [classe_0, classe_1]
        class_0_values = np.zeros((1, len(FEATURE_COLUMNS)))
        class_1_values = np.array([[
            0.15, -0.08, 0.22, 0.05, -0.03,
            0.30, -0.12, 0.18, -0.01, 0.07,
            -0.04, 0.02, -0.06, 0.09, -0.11,
            0.01, -0.25, 0.03, -0.02
        ]])
        mock_exp.shap_values.return_value = [class_0_values, class_1_values]
        mock_exp.expected_value = [0.3, 0.7]

        with patch("src.explainability.shap_explainer.shap.TreeExplainer",
                   return_value=mock_exp):
            explainer = SHAPExplainer(
                model=mock_model,
                training_data=training_data,
            )
        explainer._explainer = mock_exp

        result = explainer.explain(sample_feature_vector)
        assert result is not None
        # base_value deve ser da classe positiva (0.7)
        assert abs(result.base_value - 0.7) < 1e-6


# ---------------------------------------------------------------
# Testes de degradação graciosa (R11.6)
# ---------------------------------------------------------------


class TestGracefulDegradation:
    """Testes da degradação graciosa quando SHAP falha."""

    def test_retorna_none_quando_shap_falha(
        self, shap_explainer, sample_feature_vector
    ):
        """Se SHAP falhar, deve retornar None (R11.6)."""
        # Forçar exceção no shap_values
        shap_explainer._explainer.shap_values.side_effect = RuntimeError(
            "Erro numérico interno"
        )
        result = shap_explainer.explain(sample_feature_vector)
        assert result is None

    def test_retorna_none_quando_value_error(
        self, shap_explainer, sample_feature_vector
    ):
        """ValueError no SHAP → retorna None."""
        shap_explainer._explainer.shap_values.side_effect = ValueError(
            "Feature mismatch"
        )
        result = shap_explainer.explain(sample_feature_vector)
        assert result is None

    def test_retorna_none_quando_type_error(
        self, shap_explainer, sample_feature_vector
    ):
        """TypeError no SHAP → retorna None."""
        shap_explainer._explainer.shap_values.side_effect = TypeError(
            "Tipo incompatível"
        )
        result = shap_explainer.explain(sample_feature_vector)
        assert result is None

    def test_nao_propaga_excecao(
        self, shap_explainer, sample_feature_vector
    ):
        """Exceções NÃO devem propagar — explain() nunca levanta exceção."""
        shap_explainer._explainer.shap_values.side_effect = Exception(
            "Erro inesperado"
        )
        # Não deve levantar exceção
        result = shap_explainer.explain(sample_feature_vector)
        assert result is None


# ---------------------------------------------------------------
# Testes de explain_batch()
# ---------------------------------------------------------------


class TestExplainBatch:
    """Testes do método explain_batch()."""

    def test_retorna_lista_mesma_ordem(
        self, shap_explainer, sample_feature_vectors
    ):
        """Batch deve retornar resultados na mesma ordem dos inputs."""
        results = shap_explainer.explain_batch(sample_feature_vectors)
        assert len(results) == len(sample_feature_vectors)
        for i, result in enumerate(results):
            if result is not None:
                assert result.user_id == sample_feature_vectors[i].user_id

    def test_batch_com_falhas_parciais(
        self, shap_explainer, sample_feature_vectors
    ):
        """Se SHAP falhar para um user no batch, apenas esse é None."""
        call_count = [0]
        original_shap_values = shap_explainer._explainer.shap_values

        def side_effect(input_df):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Falha para user 2")
            return original_shap_values(input_df)

        shap_explainer._explainer.shap_values = MagicMock(
            side_effect=side_effect
        )

        results = shap_explainer.explain_batch(sample_feature_vectors)

        # User 1 e 3 devem ter resultado, user 2 deve ser None
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

    def test_batch_todos_sucesso(
        self, shap_explainer, sample_feature_vectors
    ):
        """Se todos funcionarem, nenhum resultado é None."""
        results = shap_explainer.explain_batch(sample_feature_vectors)
        for result in results:
            assert result is not None
            assert isinstance(result, ExplainabilityResult)

    def test_batch_vazio_retorna_lista_vazia(self, shap_explainer):
        """Batch com lista vazia deve retornar lista vazia."""
        results = shap_explainer.explain_batch([])
        assert results == []


# ---------------------------------------------------------------
# Testes de _feature_vector_to_dataframe
# ---------------------------------------------------------------


class TestFeatureVectorToDataframe:
    """Testes da conversão FeatureVector → DataFrame."""

    def test_retorna_dataframe_com_colunas_corretas(
        self, shap_explainer, sample_feature_vector
    ):
        """DataFrame deve ter as mesmas colunas que FEATURE_COLUMNS."""
        df = shap_explainer._feature_vector_to_dataframe(sample_feature_vector)
        assert list(df.columns) == FEATURE_COLUMNS

    def test_retorna_uma_linha(
        self, shap_explainer, sample_feature_vector
    ):
        """DataFrame deve ter exatamente 1 linha."""
        df = shap_explainer._feature_vector_to_dataframe(sample_feature_vector)
        assert len(df) == 1

    def test_trends_none_viram_zero(self, shap_explainer):
        """Trends com None devem ser convertidas para 0.0."""
        fv = FeatureVector(
            user_id="user-null-trends",
            version=1,
            generated_at="2024-01-15T10:00:00+00:00",
            observation_start="2023-12-15T00:00:00+00:00",
            observation_end="2024-01-15T00:00:00+00:00",
            total_sessions=20,
            total_viewing_hours=10.0,
            avg_session_duration_min=30.0,
            sessions_per_week=5.0,
            distinct_channels=3,
            avg_happiness_score=7.0,
            avg_buffer_ratio=0.01,
            error_rate=0.02,
            avg_bitrate=3000000.0,
            pct_episode=100.0,
            pct_sport=0.0,
            pct_live=0.0,
            pct_show=0.0,
            distinct_devices=1,
            avg_pause_count=1.0,
            avg_seek_count=0.5,
            viewing_time_trend=None,
            error_rate_trend=None,
            session_frequency_trend=None,
        )
        df = shap_explainer._feature_vector_to_dataframe(fv)
        assert df["viewing_time_trend"].iloc[0] == 0.0
        assert df["error_rate_trend"].iloc[0] == 0.0
        assert df["session_frequency_trend"].iloc[0] == 0.0

    def test_valores_numericos_corretos(
        self, shap_explainer, sample_feature_vector
    ):
        """Valores numéricos devem ser preservados corretamente."""
        df = shap_explainer._feature_vector_to_dataframe(sample_feature_vector)
        assert df["total_sessions"].iloc[0] == 120.0
        assert df["avg_happiness_score"].iloc[0] == 7.2
        assert df["error_rate"].iloc[0] == 0.05


# ---------------------------------------------------------------
# Testes de _select_top_features
# ---------------------------------------------------------------


class TestSelectTopFeatures:
    """Testes da seleção de top N features."""

    def test_retorna_top_n_features(self, shap_explainer, mock_shap_values):
        """Deve retornar exatamente top_n features."""
        features = shap_explainer._select_top_features(mock_shap_values)
        assert len(features) == 10

    def test_ordenado_por_abs_impacto_decrescente(
        self, shap_explainer, mock_shap_values
    ):
        """Features devem estar ordenadas por |normalized_impact| decrescente."""
        features = shap_explainer._select_top_features(mock_shap_values)
        abs_impacts = [abs(f.normalized_impact) for f in features]
        for i in range(len(abs_impacts) - 1):
            assert abs_impacts[i] >= abs_impacts[i + 1]

    def test_normalizacao_max_eh_um(self, shap_explainer, mock_shap_values):
        """A feature com maior SHAP absoluto deve ter |normalized| = 1.0."""
        features = shap_explainer._select_top_features(mock_shap_values)
        max_norm = max(abs(f.normalized_impact) for f in features)
        assert abs(max_norm - 1.0) < 1e-6

    def test_shap_values_todos_zero(self, shap_explainer):
        """Se todos SHAP = 0, normalized_impact deve ser 0.0 (sem divisão por zero)."""
        zeros = np.zeros(len(FEATURE_COLUMNS))
        features = shap_explainer._select_top_features(zeros)
        for f in features:
            assert f.contribution_weight == 0.0
            assert f.normalized_impact == 0.0

    def test_contribution_weight_preserva_sinal(
        self, shap_explainer, mock_shap_values
    ):
        """contribution_weight deve manter o sinal original do SHAP value."""
        features = shap_explainer._select_top_features(mock_shap_values)
        # O maior absoluto é 0.30 (índice 5 = avg_happiness_score)
        # Deve estar na primeira posição
        top_feature = features[0]
        assert top_feature.contribution_weight > 0  # 0.30 é positivo

    def test_menos_features_que_top_n(self, training_data):
        """Se tiver menos features que top_n, retornar todas disponíveis."""
        mock_model = MagicMock()
        mock_exp = MagicMock()
        mock_exp.expected_value = 0.5

        with patch("src.explainability.shap_explainer.shap.TreeExplainer",
                   return_value=mock_exp):
            explainer = SHAPExplainer(
                model=mock_model,
                training_data=training_data,
                top_n=50,  # Maior que o número de features (19)
            )
        explainer._explainer = mock_exp

        shap_vals = np.random.randn(len(FEATURE_COLUMNS))
        features = explainer._select_top_features(shap_vals)
        # Deve retornar todas as 19 features (não pode ultrapassar)
        assert len(features) == len(FEATURE_COLUMNS)
