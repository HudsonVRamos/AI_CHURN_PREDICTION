"""Testes unitários para o Feature Store (DynamoDB).

Usa moto para simular DynamoDB local.
Valida: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from src.common.models import FeatureVector
from src.store.feature_store import FeatureStore


def _create_table(dynamodb_resource, table_name: str):
    """Cria tabela DynamoDB simulada com o schema do Feature Store."""
    dynamodb_resource.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "user_id", "KeyType": "HASH"},
            {"AttributeName": "version", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "version", "AttributeType": "N"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


def _make_feature_vector(
    user_id: str = "user-001",
    version: int = 1,
    generated_at: str = "2024-01-15T10:00:00Z",
    observation_start: str = "2023-07-15T00:00:00Z",
    observation_end: str = "2024-01-15T00:00:00Z",
) -> FeatureVector:
    """Cria um FeatureVector de teste com valores padrão válidos."""
    return FeatureVector(
        user_id=user_id,
        version=version,
        generated_at=generated_at,
        observation_start=observation_start,
        observation_end=observation_end,
        total_sessions=100,
        total_viewing_hours=50.5,
        avg_session_duration_min=30.3,
        sessions_per_week=4.0,
        distinct_channels=8,
        avg_happiness_score=7.5,
        avg_buffer_ratio=0.02,
        error_rate=0.05,
        avg_bitrate=5000000.0,
        pct_episode=40.0,
        pct_sport=20.0,
        pct_live=25.0,
        pct_show=15.0,
        distinct_devices=3,
        avg_pause_count=2.1,
        avg_seek_count=1.5,
        viewing_time_trend=0.5,
        error_rate_trend=-0.01,
        session_frequency_trend=0.2,
    )


@pytest.fixture
def dynamo_setup():
    """Fixture que cria um DynamoDB mock com a tabela do Feature Store."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table_name = "test_feature_store"
        _create_table(dynamodb, table_name)

        store = FeatureStore(
            table_name=table_name,
            dynamodb_resource=dynamodb,
        )
        yield store


class TestFeatureStoreStore:
    """Testes para o método store()."""

    def test_store_retorna_versao_1_para_primeiro_registro(self, dynamo_setup):
        """R9.2: Primeiro store para um user retorna versão 1."""
        store = dynamo_setup
        fv = _make_feature_vector(user_id="user-001")

        version = store.store(fv)

        assert version == 1

    def test_store_auto_incrementa_versao(self, dynamo_setup):
        """R9.2: Versões são auto-incrementadas por user."""
        store = dynamo_setup
        fv1 = _make_feature_vector(user_id="user-001")
        fv2 = _make_feature_vector(
            user_id="user-001", generated_at="2024-02-15T10:00:00Z"
        )

        v1 = store.store(fv1)
        v2 = store.store(fv2)

        assert v1 == 1
        assert v2 == 2

    def test_store_versao_independente_por_user(self, dynamo_setup):
        """R9.2: Versionamento é independente por user_id."""
        store = dynamo_setup
        fv_a = _make_feature_vector(user_id="user-A")
        fv_b = _make_feature_vector(user_id="user-B")

        v_a = store.store(fv_a)
        v_b = store.store(fv_b)

        assert v_a == 1
        assert v_b == 1

    def test_store_persiste_metadados_corretamente(self, dynamo_setup):
        """R9.3: FeatureVector armazenado contém timestamp, window e user_id."""
        store = dynamo_setup
        fv = _make_feature_vector(
            user_id="user-meta",
            generated_at="2024-03-01T08:30:00Z",
            observation_start="2023-09-01T00:00:00Z",
            observation_end="2024-03-01T00:00:00Z",
        )

        store.store(fv)
        result = store.get_latest("user-meta")

        assert result is not None
        assert result.user_id == "user-meta"
        assert result.generated_at == "2024-03-01T08:30:00Z"
        assert result.observation_start == "2023-09-01T00:00:00Z"
        assert result.observation_end == "2024-03-01T00:00:00Z"


class TestFeatureStoreGetLatest:
    """Testes para o método get_latest()."""

    def test_get_latest_retorna_none_se_user_nao_existe(self, dynamo_setup):
        """R9.6: Retorna None para user sem dados."""
        store = dynamo_setup

        result = store.get_latest("user-inexistente")

        assert result is None

    def test_get_latest_retorna_versao_mais_recente(self, dynamo_setup):
        """R9.6: Retorna a versão mais recente sem exigir número."""
        store = dynamo_setup
        fv1 = _make_feature_vector(
            user_id="user-X", generated_at="2024-01-01T00:00:00Z"
        )
        fv2 = _make_feature_vector(
            user_id="user-X", generated_at="2024-02-01T00:00:00Z"
        )
        fv3 = _make_feature_vector(
            user_id="user-X", generated_at="2024-03-01T00:00:00Z"
        )

        store.store(fv1)
        store.store(fv2)
        store.store(fv3)

        latest = store.get_latest("user-X")

        assert latest is not None
        assert latest.version == 3
        assert latest.generated_at == "2024-03-01T00:00:00Z"


class TestFeatureStoreGetVersion:
    """Testes para o método get_version()."""

    def test_get_version_retorna_versao_especifica(self, dynamo_setup):
        """R9.4: Query por user_id + version number."""
        store = dynamo_setup
        fv1 = _make_feature_vector(
            user_id="user-V", generated_at="2024-01-01T00:00:00Z"
        )
        fv2 = _make_feature_vector(
            user_id="user-V", generated_at="2024-02-01T00:00:00Z"
        )

        store.store(fv1)
        store.store(fv2)

        result = store.get_version("user-V", 1)

        assert result is not None
        assert result.version == 1
        assert result.generated_at == "2024-01-01T00:00:00Z"

    def test_get_version_retorna_none_para_versao_inexistente(self, dynamo_setup):
        """R9.4: Retorna None para versão que não existe."""
        store = dynamo_setup
        fv = _make_feature_vector(user_id="user-V2")
        store.store(fv)

        result = store.get_version("user-V2", 999)

        assert result is None


class TestFeatureStoreGetHistory:
    """Testes para o método get_history()."""

    def test_get_history_retorna_todas_versoes(self, dynamo_setup):
        """R9.4: Retorna todas as versões de um user."""
        store = dynamo_setup
        for i in range(1, 4):
            fv = _make_feature_vector(
                user_id="user-H",
                generated_at=f"2024-0{i}-01T00:00:00Z",
            )
            store.store(fv)

        history = store.get_history("user-H")

        assert len(history) == 3
        assert history[0].version == 1
        assert history[1].version == 2
        assert history[2].version == 3

    def test_get_history_com_filtro_de_data(self, dynamo_setup):
        """R9.4: Query por user_id + date range."""
        store = dynamo_setup
        fv_jan = _make_feature_vector(
            user_id="user-D", generated_at="2024-01-15T00:00:00Z"
        )
        fv_feb = _make_feature_vector(
            user_id="user-D", generated_at="2024-02-15T00:00:00Z"
        )
        fv_mar = _make_feature_vector(
            user_id="user-D", generated_at="2024-03-15T00:00:00Z"
        )

        store.store(fv_jan)
        store.store(fv_feb)
        store.store(fv_mar)

        # Filtra a partir de fevereiro
        history = store.get_history("user-D", from_date="2024-02-01T00:00:00Z")

        assert len(history) == 2
        assert history[0].generated_at == "2024-02-15T00:00:00Z"
        assert history[1].generated_at == "2024-03-15T00:00:00Z"

    def test_get_history_retorna_lista_vazia_se_user_nao_existe(self, dynamo_setup):
        """Retorna lista vazia para user sem histórico."""
        store = dynamo_setup

        history = store.get_history("user-fantasma")

        assert history == []


class TestFeatureStoreImutabilidade:
    """Testes de imutabilidade (R9.5)."""

    def test_versoes_anteriores_preservadas(self, dynamo_setup):
        """R9.5: Versões anteriores não são sobrescritas."""
        store = dynamo_setup

        fv1 = _make_feature_vector(
            user_id="user-I",
            generated_at="2024-01-01T00:00:00Z",
        )
        fv2 = _make_feature_vector(
            user_id="user-I",
            generated_at="2024-02-01T00:00:00Z",
        )

        store.store(fv1)
        store.store(fv2)

        # A versão 1 deve continuar intacta
        v1 = store.get_version("user-I", 1)
        assert v1 is not None
        assert v1.generated_at == "2024-01-01T00:00:00Z"

        # A versão 2 é a nova
        v2 = store.get_version("user-I", 2)
        assert v2 is not None
        assert v2.generated_at == "2024-02-01T00:00:00Z"


class TestFeatureStoreSerialization:
    """Testes de serialização/desserialização de features."""

    def test_features_numericas_preservadas(self, dynamo_setup):
        """Valores numéricos são preservados após round-trip DynamoDB."""
        store = dynamo_setup
        fv = _make_feature_vector(user_id="user-S")

        store.store(fv)
        result = store.get_latest("user-S")

        assert result is not None
        assert result.total_sessions == 100
        assert result.total_viewing_hours == pytest.approx(50.5)
        assert result.avg_session_duration_min == pytest.approx(30.3)
        assert result.avg_happiness_score == pytest.approx(7.5)
        assert result.error_rate == pytest.approx(0.05)
        assert result.pct_episode == pytest.approx(40.0)
        assert result.distinct_devices == 3

    def test_trends_none_preservados(self, dynamo_setup):
        """Trends None são preservados corretamente."""
        store = dynamo_setup
        fv = FeatureVector(
            user_id="user-T",
            version=1,
            generated_at="2024-01-01T00:00:00Z",
            observation_start="2023-12-01T00:00:00Z",
            observation_end="2024-01-01T00:00:00Z",
            total_sessions=10,
            total_viewing_hours=5.0,
            avg_session_duration_min=30.0,
            sessions_per_week=3.0,
            distinct_channels=2,
            avg_happiness_score=8.0,
            avg_buffer_ratio=0.01,
            error_rate=0.1,
            avg_bitrate=3000000.0,
            pct_episode=50.0,
            pct_sport=20.0,
            pct_live=20.0,
            pct_show=10.0,
            distinct_devices=1,
            avg_pause_count=1.0,
            avg_seek_count=0.5,
            viewing_time_trend=None,
            error_rate_trend=None,
            session_frequency_trend=None,
        )

        store.store(fv)
        result = store.get_latest("user-T")

        assert result is not None
        assert result.viewing_time_trend is None
        assert result.error_rate_trend is None
        assert result.session_frequency_trend is None
