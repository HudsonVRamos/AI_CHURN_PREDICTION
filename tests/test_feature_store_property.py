"""Property-based tests para imutabilidade do Feature Store.

Valida que store() sempre cria nova versão, nunca sobrescreve.
Usa hypothesis para gerar sequências arbitrárias de store() e verificar
que todas as versões são preservadas intactas.

**Validates: Requirements 9.5**
**Property 2: Immutable Storage** — Verificar que store() sempre cria
nova versão, nunca sobrescreve.
"""

from __future__ import annotations

import boto3
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from moto import mock_aws

from src.common.models import FeatureVector
from src.store.feature_store import FeatureStore


# --- Estratégias para gerar FeatureVectors ---

# Floats seguros para DynamoDB: usamos inteiros divididos para evitar
# valores subnormais que causam decimal.Underflow na serialização.
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
def feature_vector_strategy(draw, user_id=None):
    """Gera um FeatureVector válido com valores aleatórios."""
    uid = user_id or draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=3,
            max_size=10,
        )
    )

    # Gerar percentuais que somam 100% usando inteiros e dividindo
    parts = draw(st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=4,
        max_size=4,
    ))
    total = sum(parts) or 1  # evitar divisão por zero
    pct_episode = round(parts[0] / total * 100, 2)
    pct_sport = round(parts[1] / total * 100, 2)
    pct_live = round(parts[2] / total * 100, 2)
    pct_show = round(100.0 - pct_episode - pct_sport - pct_live, 2)
    if pct_show < 0.0:
        pct_show = 0.0
        # Ajustar para garantir soma = 100
        pct_live = round(100.0 - pct_episode - pct_sport, 2)

    generated_at = draw(
        st.integers(min_value=1, max_value=12).map(
            lambda m: f"2024-{m:02d}-15T10:00:00Z"
        )
    )

    return FeatureVector(
        user_id=uid,
        version=1,  # Ignorado pelo store (auto-increment)
        generated_at=generated_at,
        observation_start="2023-07-01T00:00:00Z",
        observation_end="2024-01-01T00:00:00Z",
        total_sessions=draw(st.integers(min_value=1, max_value=5000)),
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


@st.composite
def store_sequence_strategy(draw, min_stores=2, max_stores=8):
    """Gera uma sequência de FeatureVectors para o mesmo user_id."""
    user_id = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=3,
            max_size=10,
        )
    )
    count = draw(st.integers(min_value=min_stores, max_value=max_stores))
    vectors = []
    for _ in range(count):
        fv = draw(feature_vector_strategy(user_id=user_id))
        vectors.append(fv)
    return vectors


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


@given(vectors=store_sequence_strategy(min_stores=2, max_stores=8))
@settings(max_examples=100, deadline=None)
def test_all_versions_preserved_after_multiple_stores(
    vectors: list[FeatureVector],
) -> None:
    """Após N stores para o mesmo user, get_history retorna N itens.

    **Validates: Requirements 9.5**

    Propriedade: para qualquer sequência de N store() calls para o mesmo
    user_id, get_history() retorna exatamente N Feature Vectors.
    Nenhuma versão é perdida ou sobrescrita.
    """
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table_name = "test_immutability"
        _create_table(dynamodb, table_name)
        store = FeatureStore(
            table_name=table_name,
            dynamodb_resource=dynamodb,
        )

        user_id = vectors[0].user_id

        for fv in vectors:
            store.store(fv)

        history = store.get_history(user_id)

        assert len(history) == len(vectors), (
            f"Esperado {len(vectors)} versões no histórico, "
            f"mas encontrou {len(history)}. "
            f"user_id='{user_id}'"
        )


@given(vectors=store_sequence_strategy(min_stores=2, max_stores=8))
@settings(max_examples=100, deadline=None)
def test_each_store_returns_unique_incrementing_version(
    vectors: list[FeatureVector],
) -> None:
    """Cada store() retorna uma versão única e incrementante.

    **Validates: Requirements 9.5**

    Propriedade: para qualquer sequência de store() calls, as versões
    retornadas são estritamente crescentes (1, 2, 3, ..., N).
    Nenhuma versão é reutilizada.
    """
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table_name = "test_versions"
        _create_table(dynamodb, table_name)
        store = FeatureStore(
            table_name=table_name,
            dynamodb_resource=dynamodb,
        )

        versions = []
        for fv in vectors:
            version = store.store(fv)
            versions.append(version)

        # Verificar que versões são únicas
        assert len(set(versions)) == len(versions), (
            f"Versões duplicadas detectadas: {versions}"
        )

        # Verificar que versões são estritamente crescentes
        for i in range(1, len(versions)):
            assert versions[i] > versions[i - 1], (
                f"Versão {versions[i]} não é maior que "
                f"versão anterior {versions[i - 1]}. "
                f"Sequência: {versions}"
            )

        # Verificar sequência exata 1, 2, ..., N
        expected = list(range(1, len(vectors) + 1))
        assert versions == expected, (
            f"Versões não seguem sequência esperada. "
            f"Esperado: {expected}, obtido: {versions}"
        )


@given(vectors=store_sequence_strategy(min_stores=2, max_stores=6))
@settings(max_examples=80, deadline=None)
def test_previous_versions_unchanged_after_new_store(
    vectors: list[FeatureVector],
) -> None:
    """Versões anteriores permanecem inalteradas após novos stores.

    **Validates: Requirements 9.5**

    Propriedade: get_version(user_id, v) retorna os mesmos dados antes
    e depois de um store() subsequente. Dados armazenados são imutáveis.
    """
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table_name = "test_immutable_data"
        _create_table(dynamodb, table_name)
        store = FeatureStore(
            table_name=table_name,
            dynamodb_resource=dynamodb,
        )

        user_id = vectors[0].user_id
        stored_snapshots: list[tuple[int, FeatureVector]] = []

        for fv in vectors:
            version = store.store(fv)

            # Capturar snapshot imediatamente após store
            snapshot = store.get_version(user_id, version)
            assert snapshot is not None
            stored_snapshots.append((version, snapshot))

        # Após todos os stores, verificar que cada versão
        # permanece idêntica ao snapshot capturado
        for version, original_snapshot in stored_snapshots:
            current = store.get_version(user_id, version)
            assert current is not None, (
                f"Versão {version} desapareceu após stores subsequentes"
            )

            # Comparar campos essenciais
            assert current.user_id == original_snapshot.user_id
            assert current.version == original_snapshot.version
            assert current.generated_at == original_snapshot.generated_at
            assert (
                current.observation_start
                == original_snapshot.observation_start
            )
            assert (
                current.observation_end
                == original_snapshot.observation_end
            )
            assert (
                current.total_sessions
                == original_snapshot.total_sessions
            )
            assert current.total_viewing_hours == pytest.approx(
                original_snapshot.total_viewing_hours
            )
            assert current.avg_session_duration_min == pytest.approx(
                original_snapshot.avg_session_duration_min
            )
            assert current.error_rate == pytest.approx(
                original_snapshot.error_rate
            )
            assert current.pct_episode == pytest.approx(
                original_snapshot.pct_episode
            )
            assert current.pct_sport == pytest.approx(
                original_snapshot.pct_sport
            )
            assert current.pct_live == pytest.approx(
                original_snapshot.pct_live
            )
            assert current.pct_show == pytest.approx(
                original_snapshot.pct_show
            )
            assert (
                current.distinct_devices
                == original_snapshot.distinct_devices
            )
