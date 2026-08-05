"""Property-based tests para integridade de dados (Data Integrity).

Valida que Feature Vectors são persistidos corretamente no Feature Store,
garantindo que store() retorna versão positiva, dados são imediatamente
recuperáveis, e campos-chave sobrevivem ao round-trip de serialização.

**Validates: Requirements 9.1, 17.4**
**Property 6: Data Integrity** — Features persistidas ANTES da inferência;
resultados persistidos ANTES do relatório.
"""

from __future__ import annotations

import boto3
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from moto import mock_aws

from src.common.models import FeatureVector
from src.store.feature_store import FeatureStore


# --- Estratégias para gerar FeatureVectors válidos ---

safe_float = st.integers(
    min_value=0, max_value=100000
).map(lambda x: x / 100.0)

safe_float_small = st.integers(
    min_value=0, max_value=100
).map(lambda x: x / 100.0)

safe_float_signed = st.integers(
    min_value=-1000, max_value=1000
).map(lambda x: x / 100.0)


@st.composite
def feature_vector_strategy(draw):
    """Gera um FeatureVector válido com valores aleatórios.

    Usa inteiros mapeados para floats para evitar problemas com
    Decimal e valores subnormais no DynamoDB.
    """
    user_id = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=3,
            max_size=15,
        )
    )

    # Gerar percentuais que somam 100%
    parts = draw(st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=4,
        max_size=4,
    ))
    total = sum(parts) or 1
    pct_episode = round(parts[0] / total * 100, 2)
    pct_sport = round(parts[1] / total * 100, 2)
    pct_live = round(parts[2] / total * 100, 2)
    pct_show = round(100.0 - pct_episode - pct_sport - pct_live, 2)
    if pct_show < 0.0:
        pct_show = 0.0
        pct_live = round(100.0 - pct_episode - pct_sport, 2)

    # Datas de observação variáveis
    month_start = draw(st.integers(min_value=1, max_value=6))
    month_end = draw(st.integers(min_value=7, max_value=12))
    observation_start = f"2023-{month_start:02d}-01T00:00:00Z"
    observation_end = f"2023-{month_end:02d}-01T00:00:00Z"

    generated_at = draw(
        st.integers(min_value=1, max_value=12).map(
            lambda m: f"2024-{m:02d}-15T10:00:00Z"
        )
    )

    total_sessions = draw(st.integers(min_value=1, max_value=5000))

    return FeatureVector(
        user_id=user_id,
        version=1,  # Ignorado pelo store (auto-increment)
        generated_at=generated_at,
        observation_start=observation_start,
        observation_end=observation_end,
        total_sessions=total_sessions,
        total_viewing_hours=draw(safe_float),
        avg_session_duration_min=draw(
            st.integers(min_value=0, max_value=30000).map(
                lambda x: x / 100.0
            )
        ),
        sessions_per_week=draw(
            st.integers(min_value=0, max_value=10000).map(
                lambda x: x / 100.0
            )
        ),
        distinct_channels=draw(st.integers(min_value=0, max_value=50)),
        avg_happiness_score=draw(
            st.integers(min_value=0, max_value=1000).map(
                lambda x: x / 100.0
            )
        ),
        avg_buffer_ratio=draw(safe_float_small),
        error_rate=draw(safe_float_small),
        avg_bitrate=draw(
            st.integers(min_value=0, max_value=50000000).map(float)
        ),
        pct_episode=pct_episode,
        pct_sport=pct_sport,
        pct_live=pct_live,
        pct_show=pct_show,
        distinct_devices=draw(st.integers(min_value=0, max_value=20)),
        avg_pause_count=draw(
            st.integers(min_value=0, max_value=5000).map(
                lambda x: x / 100.0
            )
        ),
        avg_seek_count=draw(
            st.integers(min_value=0, max_value=5000).map(
                lambda x: x / 100.0
            )
        ),
        viewing_time_trend=draw(
            st.one_of(st.none(), safe_float_signed)
        ),
        error_rate_trend=draw(
            st.one_of(
                st.none(),
                st.integers(min_value=-100, max_value=100).map(
                    lambda x: x / 100.0
                ),
            )
        ),
        session_frequency_trend=draw(
            st.one_of(st.none(), safe_float_signed)
        ),
    )


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


# --- Property Tests ---


@given(fv=feature_vector_strategy())
@settings(max_examples=200, deadline=None)
def test_store_always_returns_positive_version(fv: FeatureVector) -> None:
    """FeatureStore.store() sempre retorna um número de versão >= 1.

    **Validates: Requirements 9.1, 17.4**

    Propriedade: para qualquer FeatureVector válido, store() retorna
    um inteiro positivo (>= 1), garantindo que dados são efetivamente
    persistidos antes de serem usados na inferência.
    """
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table_name = "test_data_integrity_version"
        _create_table(dynamodb, table_name)
        store = FeatureStore(
            table_name=table_name,
            dynamodb_resource=dynamodb,
        )

        version = store.store(fv)

        assert isinstance(version, int), (
            f"store() retornou tipo {type(version)}, esperado int"
        )
        assert version >= 1, (
            f"store() retornou versão {version}, esperado >= 1. "
            f"user_id='{fv.user_id}'"
        )


@given(fv=feature_vector_strategy())
@settings(max_examples=200, deadline=None)
def test_stored_data_immediately_retrievable(fv: FeatureVector) -> None:
    """Após store(), get_version() com a versão retornada sempre tem sucesso.

    **Validates: Requirements 9.1, 17.4**

    Propriedade: para qualquer FeatureVector armazenado, o dado é
    imediatamente recuperável via get_version(user_id, version).
    Isto garante que a persistência ocorre ANTES de qualquer
    etapa subsequente (inferência, relatórios).
    """
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table_name = "test_data_integrity_retrieval"
        _create_table(dynamodb, table_name)
        store = FeatureStore(
            table_name=table_name,
            dynamodb_resource=dynamodb,
        )

        version = store.store(fv)
        retrieved = store.get_version(fv.user_id, version)

        assert retrieved is not None, (
            f"get_version('{fv.user_id}', {version}) retornou None "
            f"imediatamente após store(). Dados não foram persistidos."
        )
        assert isinstance(retrieved, FeatureVector), (
            f"get_version retornou tipo {type(retrieved)}, "
            f"esperado FeatureVector"
        )


@given(fv=feature_vector_strategy())
@settings(max_examples=100, deadline=None)
def test_stored_feature_vector_never_corrupted(fv: FeatureVector) -> None:
    """Campos-chave do FeatureVector sobrevivem ao round-trip store/retrieve.

    **Validates: Requirements 9.1, 17.4**

    Propriedade: para qualquer FeatureVector, os campos user_id,
    total_sessions, observation_start e observation_end são idênticos
    após o ciclo store() → get_version(). Nenhuma corrupção de dados
    ocorre durante a serialização/desserialização DynamoDB.
    """
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table_name = "test_data_integrity_roundtrip"
        _create_table(dynamodb, table_name)
        store = FeatureStore(
            table_name=table_name,
            dynamodb_resource=dynamodb,
        )

        version = store.store(fv)
        retrieved = store.get_version(fv.user_id, version)

        assert retrieved is not None, (
            "Dado não recuperável após store()"
        )

        # Campos de identidade e metadados
        assert retrieved.user_id == fv.user_id, (
            f"user_id corrompido: esperado '{fv.user_id}', "
            f"obteve '{retrieved.user_id}'"
        )
        assert retrieved.observation_start == fv.observation_start, (
            f"observation_start corrompido: esperado "
            f"'{fv.observation_start}', obteve "
            f"'{retrieved.observation_start}'"
        )
        assert retrieved.observation_end == fv.observation_end, (
            f"observation_end corrompido: esperado "
            f"'{fv.observation_end}', obteve "
            f"'{retrieved.observation_end}'"
        )
        assert retrieved.generated_at == fv.generated_at, (
            f"generated_at corrompido: esperado "
            f"'{fv.generated_at}', obteve '{retrieved.generated_at}'"
        )

        # Campos numéricos inteiros (devem ser exatos)
        assert retrieved.total_sessions == fv.total_sessions, (
            f"total_sessions corrompido: esperado "
            f"{fv.total_sessions}, obteve {retrieved.total_sessions}"
        )
        assert retrieved.distinct_channels == fv.distinct_channels, (
            f"distinct_channels corrompido: esperado "
            f"{fv.distinct_channels}, obteve "
            f"{retrieved.distinct_channels}"
        )
        assert retrieved.distinct_devices == fv.distinct_devices, (
            f"distinct_devices corrompido: esperado "
            f"{fv.distinct_devices}, obteve "
            f"{retrieved.distinct_devices}"
        )

        # Campos float (usar approx para tolerância de Decimal)
        assert retrieved.total_viewing_hours == pytest.approx(
            fv.total_viewing_hours, abs=0.01
        ), (
            f"total_viewing_hours corrompido: esperado "
            f"{fv.total_viewing_hours}, obteve "
            f"{retrieved.total_viewing_hours}"
        )
        assert retrieved.avg_session_duration_min == pytest.approx(
            fv.avg_session_duration_min, abs=0.01
        ), (
            f"avg_session_duration_min corrompido"
        )
        assert retrieved.sessions_per_week == pytest.approx(
            fv.sessions_per_week, abs=0.01
        ), (
            f"sessions_per_week corrompido"
        )
        assert retrieved.avg_happiness_score == pytest.approx(
            fv.avg_happiness_score, abs=0.01
        ), (
            f"avg_happiness_score corrompido"
        )
        assert retrieved.error_rate == pytest.approx(
            fv.error_rate, abs=0.01
        ), (
            f"error_rate corrompido"
        )
        assert retrieved.avg_bitrate == pytest.approx(
            fv.avg_bitrate, abs=0.01
        ), (
            f"avg_bitrate corrompido"
        )
        assert retrieved.pct_episode == pytest.approx(
            fv.pct_episode, abs=0.01
        ), (
            f"pct_episode corrompido"
        )
        assert retrieved.pct_sport == pytest.approx(
            fv.pct_sport, abs=0.01
        ), (
            f"pct_sport corrompido"
        )
        assert retrieved.pct_live == pytest.approx(
            fv.pct_live, abs=0.01
        ), (
            f"pct_live corrompido"
        )
        assert retrieved.pct_show == pytest.approx(
            fv.pct_show, abs=0.01
        ), (
            f"pct_show corrompido"
        )
        assert retrieved.avg_pause_count == pytest.approx(
            fv.avg_pause_count, abs=0.01
        ), (
            f"avg_pause_count corrompido"
        )
        assert retrieved.avg_seek_count == pytest.approx(
            fv.avg_seek_count, abs=0.01
        ), (
            f"avg_seek_count corrompido"
        )

        # Trends (nullable) — verificar preservação de None
        if fv.viewing_time_trend is None:
            assert retrieved.viewing_time_trend is None, (
                f"viewing_time_trend deveria ser None, "
                f"obteve {retrieved.viewing_time_trend}"
            )
        else:
            assert retrieved.viewing_time_trend == pytest.approx(
                fv.viewing_time_trend, abs=0.01
            ), (
                f"viewing_time_trend corrompido"
            )

        if fv.error_rate_trend is None:
            assert retrieved.error_rate_trend is None, (
                f"error_rate_trend deveria ser None, "
                f"obteve {retrieved.error_rate_trend}"
            )
        else:
            assert retrieved.error_rate_trend == pytest.approx(
                fv.error_rate_trend, abs=0.01
            ), (
                f"error_rate_trend corrompido"
            )

        if fv.session_frequency_trend is None:
            assert retrieved.session_frequency_trend is None, (
                f"session_frequency_trend deveria ser None, "
                f"obteve {retrieved.session_frequency_trend}"
            )
        else:
            assert retrieved.session_frequency_trend == pytest.approx(
                fv.session_frequency_trend, abs=0.01
            ), (
                f"session_frequency_trend corrompido"
            )
