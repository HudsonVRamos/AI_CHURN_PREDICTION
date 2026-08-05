"""Property-based tests para ingestão de User IDs.

Valida que IDs válidos (UUID v4) nunca são perdidos durante a deduplicação.

**Validates: Requirements 1.1, 1.2, 1.6**
**Property 2: Immutable Storage** — aplicar ao contexto de ingestão:
IDs válidos nunca são perdidos na deduplicação.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.extractors.ingestion import ingest_user_ids


# Estratégia: gera listas de UUID v4 como strings
uuid_v4_strategy = st.uuids(version=4).map(str)


@given(
    uuids=st.lists(uuid_v4_strategy, min_size=1, max_size=100),
)
@settings(max_examples=200, deadline=None)
def test_valid_uuids_never_lost_in_deduplication(uuids: list[str]) -> None:
    """Para qualquer conjunto de UUIDs v4 válidos, todos os IDs únicos aparecem no resultado.

    **Validates: Requirements 1.1, 1.2, 1.6**

    Propriedade: nenhum UUID válido é perdido durante a deduplicação.
    O número de valid_ids deve ser igual ao número de UUIDs únicos no input.
    """
    result = ingest_user_ids(uuids)

    # Calcular o conjunto de UUIDs únicos (case-insensitive, normalizado)
    unique_input = set(u.lower() for u in uuids)

    # Todos os IDs únicos devem estar presentes no resultado
    assert len(result.valid_ids) == len(unique_input), (
        f"Esperado {len(unique_input)} IDs únicos, obteve {len(result.valid_ids)}"
    )

    # Cada UUID único do input deve existir no resultado (nenhum perdido)
    result_set = set(result.valid_ids)
    for uid in unique_input:
        assert uid in result_set, f"UUID válido perdido na deduplicação: {uid}"

    # Nenhum ID inválido deve ser reportado (todos são UUID v4 válidos)
    assert result.invalid_ids == [], (
        f"UUIDs v4 válidos foram incorretamente marcados como inválidos: {result.invalid_ids}"
    )


@given(
    uuids=st.lists(uuid_v4_strategy, min_size=1, max_size=100),
)
@settings(max_examples=200, deadline=None)
def test_deduplication_count_matches_unique_ids(uuids: list[str]) -> None:
    """O count de valid_ids é igual ao número de UUIDs únicos no input.

    **Validates: Requirements 1.1, 1.2, 1.6**

    Propriedade: duplicates_removed + len(valid_ids) == len(input)
    considerando que todos os inputs são válidos.
    """
    result = ingest_user_ids(uuids)

    unique_count = len(set(u.lower() for u in uuids))
    total_input = len(uuids)
    expected_duplicates = total_input - unique_count

    assert result.duplicates_removed == expected_duplicates, (
        f"Esperado {expected_duplicates} duplicatas removidas, obteve {result.duplicates_removed}"
    )
    assert len(result.valid_ids) == unique_count, (
        f"Esperado {unique_count} IDs válidos, obteve {len(result.valid_ids)}"
    )


@given(
    uuids=st.lists(uuid_v4_strategy, min_size=1, max_size=50),
)
@settings(max_examples=200, deadline=None)
def test_ingest_is_idempotent(uuids: list[str]) -> None:
    """Rodar ingest_user_ids duas vezes com o mesmo input produz o mesmo resultado.

    **Validates: Requirements 1.1, 1.2, 1.6**

    Propriedade: a função é idempotente — resultado estável para o mesmo input.
    """
    result1 = ingest_user_ids(uuids)
    result2 = ingest_user_ids(uuids)

    assert result1.valid_ids == result2.valid_ids, (
        "Resultados diferem entre execuções com o mesmo input"
    )
    assert result1.invalid_ids == result2.invalid_ids
    assert result1.duplicates_removed == result2.duplicates_removed


@given(
    uuids=st.lists(uuid_v4_strategy, min_size=2, max_size=50),
)
@settings(max_examples=200, deadline=None)
def test_no_valid_uuid_lost_with_duplicates(uuids: list[str]) -> None:
    """Mesmo com duplicatas presentes, nenhum UUID válido único é perdido.

    **Validates: Requirements 1.1, 1.2, 1.6**

    Propriedade: se adicionarmos duplicatas ao input, o conjunto de valid_ids
    permanece o mesmo (apenas duplicates_removed muda).
    """
    # Criar input com duplicatas adicionais (duplicar o primeiro elemento)
    input_with_extra_dupes = uuids + [uuids[0]]

    result_original = ingest_user_ids(uuids)
    result_with_dupes = ingest_user_ids(input_with_extra_dupes)

    # O conjunto de IDs válidos deve ser o mesmo
    assert set(result_original.valid_ids) == set(result_with_dupes.valid_ids), (
        "Adicionar duplicatas alterou o conjunto de IDs válidos"
    )

    # Nenhum UUID do input original pode ter sido perdido
    for uid in result_original.valid_ids:
        assert uid in result_with_dupes.valid_ids, (
            f"UUID perdido ao adicionar duplicatas: {uid}"
        )
