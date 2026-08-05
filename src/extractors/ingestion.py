"""Módulo de ingestão de listas de User IDs.

Aceita entrada via CSV (coluna "user_id"), JSON (chave "user_ids") ou array direto.
Valida UUID v4, deduplica, e retorna apenas IDs válidos.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass

from src.common.logging import get_logger

logger = get_logger("extraction")

# Regex para validação de UUID v4
# Aceita lowercase e uppercase; o dígito de versão deve ser '4'
# e o primeiro dígito do campo clock_seq deve ser 8, 9, a ou b.
UUID_V4_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Limites de IDs válidos aceitos
MIN_VALID_IDS = 1
MAX_VALID_IDS = 50_000


class IngestionError(Exception):
    """Erro levantado quando a ingestão falha (nenhum ID válido ou lista vazia)."""

    pass


@dataclass
class IngestionResult:
    """Resultado da ingestão de IDs de usuários.

    Attributes:
        valid_ids: Lista de UUIDs v4 válidos e únicos.
        invalid_ids: Lista de IDs que falharam na validação.
        duplicates_removed: Quantidade de duplicatas removidas.
    """

    valid_ids: list[str]
    invalid_ids: list[str]
    duplicates_removed: int


def is_valid_uuid_v4(value: str) -> bool:
    """Verifica se uma string é um UUID v4 válido.

    Args:
        value: String a ser validada.

    Returns:
        True se for UUID v4 válido, False caso contrário.
    """
    return bool(UUID_V4_REGEX.match(value.strip()))


def _parse_csv(content: str) -> list[str]:
    """Extrai IDs de conteúdo CSV com coluna 'user_id'.

    Args:
        content: Conteúdo CSV como string.

    Returns:
        Lista de strings extraídas da coluna user_id.

    Raises:
        IngestionError: Se a coluna 'user_id' não for encontrada.
    """
    reader = csv.DictReader(io.StringIO(content))

    if reader.fieldnames is None:
        raise IngestionError("CSV vazio ou sem cabeçalho.")

    # Normalizar nomes das colunas para busca case-insensitive
    fieldnames_lower = [f.strip().lower() for f in reader.fieldnames]
    if "user_id" not in fieldnames_lower:
        raise IngestionError(
            f"Coluna 'user_id' não encontrada no CSV. "
            f"Colunas disponíveis: {reader.fieldnames}"
        )

    # Encontrar o nome real da coluna
    col_index = fieldnames_lower.index("user_id")
    col_name = reader.fieldnames[col_index]

    ids: list[str] = []
    for row in reader:
        value = row.get(col_name, "")
        if value and value.strip():
            ids.append(value.strip())

    return ids


def _parse_json(content: str) -> list[str]:
    """Extrai IDs de conteúdo JSON com chave 'user_ids'.

    Args:
        content: Conteúdo JSON como string.

    Returns:
        Lista de strings extraídas da chave user_ids.

    Raises:
        IngestionError: Se o JSON for inválido ou a chave 'user_ids' não existir.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise IngestionError(f"JSON inválido: {e}") from e

    if not isinstance(data, dict):
        raise IngestionError(
            "JSON deve ser um objeto com a chave 'user_ids'."
        )

    if "user_ids" not in data:
        raise IngestionError(
            f"Chave 'user_ids' não encontrada no JSON. "
            f"Chaves disponíveis: {list(data.keys())}"
        )

    user_ids = data["user_ids"]
    if not isinstance(user_ids, list):
        raise IngestionError(
            "'user_ids' deve ser uma lista de strings."
        )

    # Converter todos os elementos para string e filtrar vazios
    return [str(uid).strip() for uid in user_ids if uid is not None and str(uid).strip()]


def ingest_user_ids(
    source: str | list[str],
    source_format: str | None = None,
) -> IngestionResult:
    """Ingere uma lista de User IDs a partir de CSV, JSON ou array direto.

    Valida UUID v4, deduplica e retorna resultado estruturado.

    Args:
        source: Conteúdo CSV (string), conteúdo JSON (string), ou lista de IDs.
        source_format: Formato da fonte - "csv", "json", ou None (auto-detect para
                       listas diretas). Se source for list, este parâmetro é ignorado.

    Returns:
        IngestionResult com IDs válidos, inválidos e contagem de duplicatas.

    Raises:
        IngestionError: Se nenhum ID válido for encontrado, lista vazia,
                        ou se exceder o limite de 50.000 IDs.
    """
    raw_ids: list[str]

    # Determinar a fonte e extrair IDs brutos
    if isinstance(source, list):
        raw_ids = [str(uid).strip() for uid in source if uid is not None and str(uid).strip()]
    elif source_format == "csv":
        raw_ids = _parse_csv(source)
    elif source_format == "json":
        raw_ids = _parse_json(source)
    else:
        # Auto-detect: tenta JSON primeiro, depois CSV
        raw_ids = _auto_detect_and_parse(source)

    # Verificar lista vazia
    if not raw_ids:
        raise IngestionError(
            "Nenhum ID fornecido. É necessário ao menos 1 User ID válido."
        )

    # Deduplicar preservando ordem
    seen: set[str] = set()
    unique_ids: list[str] = []
    for uid in raw_ids:
        uid_lower = uid.lower()
        if uid_lower not in seen:
            seen.add(uid_lower)
            unique_ids.append(uid)

    duplicates_removed = len(raw_ids) - len(unique_ids)

    if duplicates_removed > 0:
        logger.info(
            f"Deduplicação: {duplicates_removed} IDs duplicados removidos.",
            extra={"duplicates_removed": duplicates_removed},
        )

    # Validar UUID v4
    valid_ids: list[str] = []
    invalid_ids: list[str] = []

    for uid in unique_ids:
        if is_valid_uuid_v4(uid):
            valid_ids.append(uid.lower())  # Normalizar para lowercase
        else:
            invalid_ids.append(uid)
            logger.warning(
                f"ID inválido rejeitado (não é UUID v4): {uid}",
                extra={"invalid_id": uid},
            )

    # Verificar se há IDs válidos (R1.4)
    if not valid_ids:
        raise IngestionError(
            "Nenhum User ID válido encontrado. "
            f"Todos os {len(invalid_ids)} IDs fornecidos são inválidos. "
            "É necessário ao menos 1 User ID em formato UUID v4."
        )

    # Verificar limite máximo (R1.1)
    if len(valid_ids) > MAX_VALID_IDS:
        raise IngestionError(
            f"Número de IDs válidos ({len(valid_ids)}) excede o limite "
            f"máximo de {MAX_VALID_IDS}. Reduza a lista de entrada."
        )

    logger.info(
        f"Ingestão concluída: {len(valid_ids)} IDs válidos, "
        f"{len(invalid_ids)} inválidos, {duplicates_removed} duplicatas removidas.",
        extra={
            "valid_count": len(valid_ids),
            "invalid_count": len(invalid_ids),
            "duplicates_removed": duplicates_removed,
        },
    )

    return IngestionResult(
        valid_ids=valid_ids,
        invalid_ids=invalid_ids,
        duplicates_removed=duplicates_removed,
    )


def _auto_detect_and_parse(content: str) -> list[str]:
    """Tenta detectar automaticamente o formato do conteúdo (JSON ou CSV).

    Args:
        content: Conteúdo como string.

    Returns:
        Lista de IDs extraídos.

    Raises:
        IngestionError: Se o formato não puder ser detectado.
    """
    content_stripped = content.strip()

    # Tenta JSON primeiro (começa com '{' ou '[')
    if content_stripped.startswith("{") or content_stripped.startswith("["):
        try:
            return _parse_json(content_stripped)
        except IngestionError:
            pass

    # Tenta CSV (tem header com 'user_id')
    try:
        return _parse_csv(content_stripped)
    except IngestionError:
        pass

    raise IngestionError(
        "Não foi possível detectar o formato da entrada. "
        "Use source_format='csv' ou source_format='json', "
        "ou forneça uma lista direta de IDs."
    )
