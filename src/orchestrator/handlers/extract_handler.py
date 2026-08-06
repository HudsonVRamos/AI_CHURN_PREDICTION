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
        account_code = event.get("npaw_account_code", "sky_brazil")
        api_key = event.get("npaw_api_key", "")

        # Período pode ser:
        # 1. Global: from_date/to_date no evento (aplica para todos)
        # 2. Por user: user_dates = {"user_id": {"from_date": "...", "to_date": "..."}}
        # 3. Default: calcula automaticamente (últimos N meses a partir de hoje)
        global_from_date = event.get("from_date")
        global_to_date = event.get("to_date")
        user_dates = event.get("user_dates", {})  # dict: user_id -> {from_date, to_date}

        # Se nenhuma data fornecida, calcular default
        if not global_from_date and not user_dates:
            from datetime import datetime, timedelta, timezone
            time_window_months = int(os.environ.get("TIME_WINDOW_MONTHS", "6"))
            now = datetime.now(timezone.utc)
            from_dt = now - timedelta(days=time_window_months * 30)
            global_from_date = from_dt.strftime("%Y-%m-%d")
            global_to_date = now.strftime("%Y-%m-%d")

        extractor = NPAWExtractor(
            account_code=account_code,
            api_key=api_key,
        )

        # Executar extração assíncrona (com datas por user quando disponível)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results: dict[str, list] = {}

            async def _extract_all():
                for user_id in user_ids:
                    # Determinar datas para este user
                    ud = user_dates.get(user_id, {})
                    u_from = ud.get("from_date", global_from_date or "last6months")
                    u_to = ud.get("to_date", global_to_date)

                    try:
                        sessions = await extractor.extract_user_sessions(
                            user_id=user_id,
                            from_date=u_from,
                            to_date=u_to,
                        )
                        results[user_id] = sessions
                    except Exception as e:
                        logger.warning(
                            f"Falha na extração de {user_id}: {e}",
                            extra={"user_id": user_id, "error": str(e)},
                        )
                        results[user_id] = []

            loop.run_until_complete(_extract_all())
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
