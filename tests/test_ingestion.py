"""Testes unitários para o módulo de ingestão de User IDs.

Cobre os requisitos 1.1-1.6:
- Validação de UUID v4
- Aceitação de CSV, JSON e array direto
- Deduplicação
- Rejeição de IDs inválidos com continuidade
- Erro quando todos os IDs são inválidos ou lista vazia
- Limites de 1 a 50.000 IDs
"""

from __future__ import annotations

import json
import uuid

import pytest

from src.extractors.ingestion import (
    IngestionError,
    IngestionResult,
    ingest_user_ids,
    is_valid_uuid_v4,
)


# =========================================================================
# Fixtures
# =========================================================================


def _generate_uuid_v4() -> str:
    """Gera um UUID v4 válido."""
    return str(uuid.uuid4())


@pytest.fixture
def valid_uuids() -> list[str]:
    """Lista de 5 UUIDs v4 válidos."""
    return [_generate_uuid_v4() for _ in range(5)]


@pytest.fixture
def invalid_ids() -> list[str]:
    """Lista de IDs inválidos (não UUID v4)."""
    return [
        "not-a-uuid",
        "12345",
        "",
        "550e8400-e29b-11d4-a716-446655440000",  # UUID v1 (versão 1, não 4)
        "xyz-abc-123",
        "00000000-0000-0000-0000-000000000000",  # UUID nil (versão 0)
    ]


# =========================================================================
# Testes de is_valid_uuid_v4
# =========================================================================


class TestIsValidUuidV4:
    """Testes para validação de UUID v4."""

    def test_uuid_v4_valido(self) -> None:
        """UUID v4 gerado pelo módulo uuid deve ser válido."""
        uid = str(uuid.uuid4())
        assert is_valid_uuid_v4(uid) is True

    def test_uuid_v4_uppercase(self) -> None:
        """UUID v4 em uppercase deve ser válido."""
        uid = str(uuid.uuid4()).upper()
        assert is_valid_uuid_v4(uid) is True

    def test_uuid_v4_com_espacos(self) -> None:
        """UUID v4 com espaços ao redor deve ser válido."""
        uid = f"  {uuid.uuid4()}  "
        assert is_valid_uuid_v4(uid) is True

    def test_uuid_v1_invalido(self) -> None:
        """UUID v1 não deve passar validação de v4."""
        # UUID v1 tem '1' na posição de versão
        uid = "550e8400-e29b-11d4-a716-446655440000"
        assert is_valid_uuid_v4(uid) is False

    def test_string_vazia(self) -> None:
        """String vazia não é UUID v4 válido."""
        assert is_valid_uuid_v4("") is False

    def test_string_aleatoria(self) -> None:
        """String aleatória não é UUID v4 válido."""
        assert is_valid_uuid_v4("hello-world") is False

    def test_uuid_nil(self) -> None:
        """UUID nil (todos zeros) não é UUID v4."""
        assert is_valid_uuid_v4("00000000-0000-0000-0000-000000000000") is False


# =========================================================================
# Testes de ingest_user_ids com array direto
# =========================================================================


class TestIngestArrayDireto:
    """Testes para ingestão via lista (array) direta."""

    def test_lista_valida(self, valid_uuids: list[str]) -> None:
        """Lista de UUIDs válidos retorna todos como valid_ids."""
        result = ingest_user_ids(valid_uuids)
        assert len(result.valid_ids) == 5
        assert result.invalid_ids == []
        assert result.duplicates_removed == 0

    def test_lista_com_invalidos(self, valid_uuids: list[str]) -> None:
        """IDs inválidos são rejeitados, válidos continuam."""
        mixed = valid_uuids + ["invalid-id", "another-bad"]
        result = ingest_user_ids(mixed)
        assert len(result.valid_ids) == 5
        assert len(result.invalid_ids) == 2
        assert "invalid-id" in result.invalid_ids

    def test_lista_vazia_levanta_erro(self) -> None:
        """Lista vazia deve levantar IngestionError."""
        with pytest.raises(IngestionError, match="Nenhum ID fornecido"):
            ingest_user_ids([])

    def test_todos_invalidos_levanta_erro(self, invalid_ids: list[str]) -> None:
        """Lista com todos IDs inválidos deve levantar IngestionError."""
        # Remover string vazia pois será filtrada antes
        ids_sem_vazio = [uid for uid in invalid_ids if uid.strip()]
        with pytest.raises(IngestionError, match="Nenhum User ID válido"):
            ingest_user_ids(ids_sem_vazio)

    def test_deduplicacao(self) -> None:
        """IDs duplicados devem ser removidos."""
        uid = _generate_uuid_v4()
        result = ingest_user_ids([uid, uid, uid])
        assert len(result.valid_ids) == 1
        assert result.duplicates_removed == 2

    def test_deduplicacao_case_insensitive(self) -> None:
        """Deduplicação deve ser case-insensitive."""
        uid = _generate_uuid_v4()
        result = ingest_user_ids([uid.lower(), uid.upper()])
        assert len(result.valid_ids) == 1
        assert result.duplicates_removed == 1

    def test_normalizacao_lowercase(self) -> None:
        """IDs válidos devem ser normalizados para lowercase."""
        uid = _generate_uuid_v4().upper()
        result = ingest_user_ids([uid])
        assert result.valid_ids[0] == uid.lower()

    def test_limite_maximo_excedido(self) -> None:
        """Exceder 50.000 IDs válidos deve levantar IngestionError."""
        # Gerar 50.001 UUIDs
        ids = [str(uuid.uuid4()) for _ in range(50_001)]
        with pytest.raises(IngestionError, match="excede o limite"):
            ingest_user_ids(ids)

    def test_limite_maximo_exato(self) -> None:
        """Exatamente 50.000 IDs válidos deve funcionar."""
        ids = [str(uuid.uuid4()) for _ in range(50_000)]
        result = ingest_user_ids(ids)
        assert len(result.valid_ids) == 50_000

    def test_uuid_v1_rejeitado_no_fluxo(self) -> None:
        """UUID v1 deve ser rejeitado no fluxo completo de ingestão."""
        uid_v4 = _generate_uuid_v4()
        uid_v1 = "550e8400-e29b-11d4-a716-446655440000"
        result = ingest_user_ids([uid_v4, uid_v1])
        assert len(result.valid_ids) == 1
        assert uid_v1 in result.invalid_ids


# =========================================================================
# Testes de ingest_user_ids com CSV
# =========================================================================


class TestIngestCSV:
    """Testes para ingestão via conteúdo CSV."""

    def test_csv_valido(self, valid_uuids: list[str]) -> None:
        """CSV com coluna user_id deve extrair IDs corretamente."""
        csv_content = "user_id\n" + "\n".join(valid_uuids)
        result = ingest_user_ids(csv_content, source_format="csv")
        assert len(result.valid_ids) == 5

    def test_csv_header_case_insensitive(self) -> None:
        """Header 'User_ID' deve ser aceito (case insensitive)."""
        uid = _generate_uuid_v4()
        csv_content = f"User_ID\n{uid}"
        result = ingest_user_ids(csv_content, source_format="csv")
        assert len(result.valid_ids) == 1

    def test_csv_sem_coluna_user_id(self) -> None:
        """CSV sem coluna user_id deve levantar IngestionError."""
        csv_content = "id,name\n123,test"
        with pytest.raises(IngestionError, match="Coluna 'user_id' não encontrada"):
            ingest_user_ids(csv_content, source_format="csv")

    def test_csv_vazio(self) -> None:
        """CSV vazio deve levantar IngestionError."""
        with pytest.raises(IngestionError, match="vazio"):
            ingest_user_ids("", source_format="csv")

    def test_csv_com_linhas_vazias(self) -> None:
        """CSV com linhas vazias deve ignorá-las."""
        uid = _generate_uuid_v4()
        csv_content = f"user_id\n{uid}\n\n\n"
        result = ingest_user_ids(csv_content, source_format="csv")
        assert len(result.valid_ids) == 1


# =========================================================================
# Testes de ingest_user_ids com JSON
# =========================================================================


class TestIngestJSON:
    """Testes para ingestão via conteúdo JSON."""

    def test_json_valido(self, valid_uuids: list[str]) -> None:
        """JSON com chave user_ids deve extrair IDs corretamente."""
        json_content = json.dumps({"user_ids": valid_uuids})
        result = ingest_user_ids(json_content, source_format="json")
        assert len(result.valid_ids) == 5

    def test_json_sem_chave_user_ids(self) -> None:
        """JSON sem chave user_ids deve levantar IngestionError."""
        json_content = json.dumps({"ids": ["abc"]})
        with pytest.raises(IngestionError, match="Chave 'user_ids' não encontrada"):
            ingest_user_ids(json_content, source_format="json")

    def test_json_invalido(self) -> None:
        """JSON malformado deve levantar IngestionError."""
        with pytest.raises(IngestionError, match="JSON inválido"):
            ingest_user_ids("{invalid", source_format="json")

    def test_json_user_ids_nao_lista(self) -> None:
        """JSON com user_ids não sendo lista deve levantar IngestionError."""
        json_content = json.dumps({"user_ids": "not-a-list"})
        with pytest.raises(IngestionError, match="deve ser uma lista"):
            ingest_user_ids(json_content, source_format="json")

    def test_json_com_valores_none(self) -> None:
        """JSON com valores None na lista deve ignorá-los."""
        uid = _generate_uuid_v4()
        json_content = json.dumps({"user_ids": [uid, None, ""]})
        result = ingest_user_ids(json_content, source_format="json")
        assert len(result.valid_ids) == 1


# =========================================================================
# Testes de auto-detecção de formato
# =========================================================================


class TestAutoDetect:
    """Testes para detecção automática de formato."""

    def test_detecta_json(self, valid_uuids: list[str]) -> None:
        """Conteúdo JSON deve ser detectado automaticamente."""
        json_content = json.dumps({"user_ids": valid_uuids})
        result = ingest_user_ids(json_content)
        assert len(result.valid_ids) == 5

    def test_detecta_csv(self) -> None:
        """Conteúdo CSV deve ser detectado automaticamente."""
        uid = _generate_uuid_v4()
        csv_content = f"user_id\n{uid}"
        result = ingest_user_ids(csv_content)
        assert len(result.valid_ids) == 1

    def test_formato_nao_detectado(self) -> None:
        """Conteúdo irreconhecível deve levantar IngestionError."""
        with pytest.raises(IngestionError, match="Não foi possível detectar"):
            ingest_user_ids("random text without structure")


# =========================================================================
# Testes do IngestionResult
# =========================================================================


class TestIngestionResult:
    """Testes para o dataclass de resultado."""

    def test_resultado_completo(self) -> None:
        """IngestionResult deve conter todas as informações."""
        uid_valid = _generate_uuid_v4()
        result = ingest_user_ids([uid_valid, "invalid-id", uid_valid])
        assert isinstance(result, IngestionResult)
        assert len(result.valid_ids) == 1
        assert len(result.invalid_ids) == 1
        assert result.duplicates_removed == 1
