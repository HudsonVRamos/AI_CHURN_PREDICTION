"""Lambda handler para o estágio de Extração NPAW do pipeline.

Responsável por extrair sessões de visualização de cada usuário
via API NPAW com rate limiting e concorrência controlada.

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import boto3

from src.common.logging import get_logger, set_execution_id
from src.extractors.npaw_extractor import (
    NPAWExtractor,
    NPAWAuthenticationError,
    NPAWExtractorError,
)


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para o estágio de extração de dados NPAW.

    Extrai sessões de visualização para os User IDs fornecidos no evento.
    Persiste os dados brutos em S3 antes de retornar (R17.4).

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução do pipeline.
            - valid_user_ids: Lista de IDs válidos do estágio anterior.
            - from_date: Data início da extração (ex: '2024-01-01').
            - to_date (opcional): Data fim da extração.
            - npaw_account_code: Código da conta NPAW.
            - npaw_api_key: Chave de API da NPAW.
        context: Contexto Lambda.

    Returns:
        Dicionário com event enriquecido com:
            - extracted_data_s3_prefix: Prefixo S3 dos dados extraídos.
            - users_extracted: Contagem de usuários com dados.
            - users_without_data: Contagem de usuários sem dados.
            - stage_completed: "extraction"

    Raises:
        NPAWAuthenticationError: Se a API NPAW retornar 401/403.
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("extraction")
    logger.log_stage_start()
    start_time = time.time()

    try:
        user_ids = event.get("valid_user_ids", [])
        from_date = event.get("from_date", "last6months")
        to_date = event.get("to_date")
        account_code = event.get("npaw_account_code", "sky_brazil")
        api_key = event.get("npaw_api_key", "")

        extractor = NPAWExtractor(
            account_code=account_code,
            api_key=api_key,
        )

        # Executar extração assíncrona
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                extractor.extract_batch(
                    user_ids=user_ids,
                    from_date=from_date,
                    to_date=to_date,
                )
            )
        finally:
            loop.close()

        # Persistir dados brutos em S3 ANTES do próximo estágio (R17.4)
        s3_prefix = f"raw_data/{execution_id}"
        s3_client = boto3.client("s3")
        bucket = event.get("bucket", "sky-brazil-churn-prediction")

        users_extracted = 0
        users_without_data = 0

        for user_id, sessions in results.items():
            if sessions:
                s3_key = f"{s3_prefix}/{user_id}.json"
                s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=json.dumps(sessions, ensure_ascii=False),
                    ContentType="application/json",
                )
                users_extracted += 1
            else:
                users_without_data += 1

        output = {
            **event,
            "execution_id": execution_id,
            "extracted_data_s3_prefix": f"s3://{bucket}/{s3_prefix}",
            "extracted_sessions": {
                uid: sessions
                for uid, sessions in results.items()
                if sessions
            },
            "users_extracted": users_extracted,
            "users_without_data": users_without_data,
            "stage_completed": "extraction",
        }

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except NPAWAuthenticationError as e:
        logger.critical(
            f"Erro de autenticação NPAW - pipeline abortado: {e}",
            extra={"error_type": "NPAWAuthenticationError"},
        )
        raise

    except Exception as e:
        logger.error(
            f"Falha no estágio extraction: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
