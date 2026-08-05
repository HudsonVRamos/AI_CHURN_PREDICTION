"""Extrator de dados de sessões da API NPAW.

Implementa o cliente assíncrono para extrair sessões de visualização
por usuário, com paginação automática, rate limiting, concorrência limitada,
retry com backoff exponencial e logging de progresso.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 8.2, 8.3, 8.4
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from src.common.logging import get_logger

logger = get_logger("extraction")


class NPAWExtractorError(Exception):
    """Erro base do extrator NPAW."""

    pass


class NPAWAuthenticationError(NPAWExtractorError):
    """Erro de autenticação na API NPAW (401/403)."""

    pass


class NPAWExtractor:
    """Extrai sessões de usuários da API NPAW com rate limiting e concorrência.

    Realiza queries paginadas ao endpoint rawdata da NPAW, respeitando
    limites de taxa configuráveis e extraindo até MAX_SESSIONS_PER_USER
    sessões por usuário.

    Attributes:
        BASE_URL: URL base da API NPAW.
        BATCH_SIZE: Quantidade de registros por request (paginação).
        MAX_SESSIONS_PER_USER: Limite máximo de sessões extraídas por user.
        MAX_CONCURRENT: Máximo de usuários processados em paralelo.
    """

    BASE_URL = "https://api.npaw.com"
    BATCH_SIZE = 100
    MAX_SESSIONS_PER_USER = 5000
    MAX_CONCURRENT = 5

    def __init__(
        self,
        account_code: str,
        api_key: str,
        rate_limit_seconds: float = 1.0,
        max_concurrent: int | None = None,
        batch_size: int | None = None,
        max_sessions_per_user: int | None = None,
        base_url: str | None = None,
    ) -> None:
        """Inicializa o extrator NPAW.

        Args:
            account_code: Código da conta NPAW (ex: 'sky_brazil').
            api_key: Chave de API para autenticação via header.
            rate_limit_seconds: Intervalo mínimo entre chamadas
                (default: 1.0s).
            max_concurrent: Máximo de users em paralelo (default: 5).
            batch_size: Registros por página (default: 100).
            max_sessions_per_user: Limite de sessões por user (default: 5000).
            base_url: URL base da API (default: https://api.npaw.com).
        """
        if not account_code:
            raise ValueError("account_code não pode ser vazio")
        if not api_key:
            raise ValueError("api_key não pode ser vazio")

        self._account_code = account_code
        self._api_key = api_key
        self._rate_limit_seconds = rate_limit_seconds
        self._max_concurrent = max_concurrent or self.MAX_CONCURRENT
        self._batch_size = batch_size or self.BATCH_SIZE
        self._max_sessions_per_user = (
            max_sessions_per_user or self.MAX_SESSIONS_PER_USER
        )
        self._base_url = base_url or self.BASE_URL

        # Semáforo para limitar concorrência
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        # Controle de rate limiting (timestamp da última chamada)
        self._last_request_time: float = 0.0
        self._rate_lock = asyncio.Lock()

    @property
    def endpoint_url(self) -> str:
        """URL completa do endpoint rawdata."""
        return f"{self._base_url}/{self._account_code}/rawdata"

    def _build_headers(self) -> dict[str, str]:
        """Constrói headers de autenticação para a API."""
        return {"npaw-api-key": self._api_key}

    def _build_params(
        self,
        user_id: str,
        from_date: str,
        to_date: str | None,
        offset: int,
    ) -> dict[str, str]:
        """Constrói parâmetros de query para a requisição.

        Args:
            user_id: ID do usuário para filtro.
            from_date: Data de início (ex: '2024-01-01' ou 'last6months').
            to_date: Data de fim (opcional).
            offset: Offset para paginação.

        Returns:
            Dicionário com os parâmetros de query.
        """
        filter_json = json.dumps([
            {"name": "uf", "rules": {"user_id": [user_id]}}
        ])

        params: dict[str, str] = {
            "fromDate": from_date,
            "filter": filter_json,
            "limit": str(self._batch_size),
            "offset": str(offset),
            "orderBy": "end_at",
            "orderDirection": "desc",
        }

        if to_date:
            params["toDate"] = to_date

        return params

    async def _wait_rate_limit(self) -> None:
        """Aguarda o intervalo de rate limiting entre chamadas."""
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait_time = self._rate_limit_seconds - elapsed

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self._last_request_time = time.monotonic()

    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        user_id: str,
        from_date: str,
        to_date: str | None,
        offset: int,
    ) -> list[dict[str, Any]]:
        """Busca uma página de sessões da API NPAW com retry automático.

        Implementa retry com backoff exponencial (2s, 4s, 8s) para erros
        de servidor (5xx) e timeouts. Erros de autenticação (401/403) são
        propagados imediatamente sem retry.

        Args:
            session: Sessão HTTP aiohttp.
            user_id: ID do usuário.
            from_date: Data de início.
            to_date: Data de fim (opcional).
            offset: Offset para paginação.

        Returns:
            Lista de sessões retornadas pela API.

        Raises:
            NPAWAuthenticationError: Se a API retornar 401 ou 403.
            NPAWExtractorError: Após 3 tentativas falhas ou erro HTTP 4xx.
        """
        max_retries = 3
        backoff_delays = [2.0, 4.0, 8.0]

        params = self._build_params(user_id, from_date, to_date, offset)

        for attempt in range(max_retries + 1):
            await self._wait_rate_limit()

            logger.debug(
                f"Requisição NPAW: user_id={user_id}, offset={offset}"
                + (f", tentativa={attempt + 1}" if attempt > 0 else ""),
                extra={
                    "user_id": user_id,
                    "offset": offset,
                    "attempt": attempt + 1,
                },
            )

            try:
                async with session.get(
                    self.endpoint_url,
                    headers=self._build_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    # Auth errors: abortar imediatamente, sem retry
                    if response.status in (401, 403):
                        raise NPAWAuthenticationError(
                            f"Erro de autenticação NPAW (HTTP {response.status}) "
                            f"para user_id={user_id}"
                        )

                    # Server errors (5xx): retry com backoff
                    if response.status >= 500:
                        body = await response.text()
                        if attempt < max_retries:
                            delay = backoff_delays[attempt]
                            logger.warning(
                                f"Erro HTTP {response.status} da NPAW para "
                                f"user_id={user_id}. Retry em {delay}s "
                                f"(tentativa {attempt + 1}/{max_retries})",
                                extra={
                                    "user_id": user_id,
                                    "http_status": response.status,
                                    "attempt": attempt + 1,
                                    "retry_delay": delay,
                                },
                            )
                            await asyncio.sleep(delay)
                            continue
                        # Esgotou retries
                        raise NPAWExtractorError(
                            f"Erro HTTP {response.status} da API NPAW após "
                            f"{max_retries} tentativas: {body}"
                        )

                    # Client errors (4xx exceto 401/403): falhar imediatamente
                    if response.status >= 400:
                        body = await response.text()
                        raise NPAWExtractorError(
                            f"Erro HTTP {response.status} da API NPAW: {body}"
                        )

                    data = await response.json()

                # Extrair sessões da estrutura de resposta da NPAW
                return self._parse_sessions(data)

            except aiohttp.ServerTimeoutError:
                if attempt < max_retries:
                    delay = backoff_delays[attempt]
                    logger.warning(
                        f"Timeout do servidor NPAW para user_id={user_id}. "
                        f"Retry em {delay}s "
                        f"(tentativa {attempt + 1}/{max_retries})",
                        extra={
                            "user_id": user_id,
                            "attempt": attempt + 1,
                            "retry_delay": delay,
                        },
                    )
                    await asyncio.sleep(delay)
                    continue
                raise NPAWExtractorError(
                    f"Timeout do servidor NPAW após {max_retries} tentativas "
                    f"para user_id={user_id}"
                )

            except asyncio.TimeoutError:
                if attempt < max_retries:
                    delay = backoff_delays[attempt]
                    logger.warning(
                        f"Timeout na requisição NPAW para user_id={user_id}. "
                        f"Retry em {delay}s "
                        f"(tentativa {attempt + 1}/{max_retries})",
                        extra={
                            "user_id": user_id,
                            "attempt": attempt + 1,
                            "retry_delay": delay,
                        },
                    )
                    await asyncio.sleep(delay)
                    continue
                raise NPAWExtractorError(
                    f"Timeout na API NPAW após {max_retries} tentativas "
                    f"para user_id={user_id}"
                )

        # Fallback (não deveria chegar aqui)
        raise NPAWExtractorError(
            f"Falha inesperada ao buscar dados NPAW para user_id={user_id}"
        )

    def _parse_sessions(self, response_data: Any) -> list[dict[str, Any]]:
        """Extrai a lista de sessões da resposta da API NPAW.

        A API retorna dados na estrutura:
        {"data": [{"values": [...sessões...]}]}

        Args:
            response_data: JSON decodificado da resposta.

        Returns:
            Lista de dicionários com os dados de cada sessão.
        """
        if not response_data:
            return []

        data = response_data.get("data")
        if not data or not isinstance(data, list) or len(data) == 0:
            return []

        values = data[0].get("values")
        if not values or not isinstance(values, list):
            return []

        return values

    async def extract_user_sessions(
        self,
        user_id: str,
        from_date: str,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extrai todas as sessões de um usuário com paginação automática.

        Realiza requests paginados até que:
        - A API retorne menos que BATCH_SIZE registros (última página)
        - O total de sessões atinja MAX_SESSIONS_PER_USER

        Args:
            user_id: ID do usuário (UUID).
            from_date: Data de início (ex: '2024-01-01', 'last6months').
            to_date: Data de fim (opcional, ex: '2024-07-01').

        Returns:
            Lista com todas as sessões extraídas para o usuário.

        Raises:
            NPAWAuthenticationError: Se a API retornar 401/403.
            NPAWExtractorError: Para outros erros de comunicação.
        """
        all_sessions: list[dict[str, Any]] = []
        offset = 0

        logger.info(
            f"Iniciando extração para user_id={user_id}",
            extra={
                "user_id": user_id,
                "from_date": from_date,
                "to_date": to_date,
            },
        )

        async with aiohttp.ClientSession() as session:
            while len(all_sessions) < self._max_sessions_per_user:
                page = await self._fetch_page(
                    session, user_id, from_date, to_date, offset
                )

                if not page:
                    break

                all_sessions.extend(page)
                offset += self._batch_size

                # Parar se retornou menos que o batch (última página)
                if len(page) < self._batch_size:
                    break

        # Truncar no limite máximo caso a última página tenha excedido
        if len(all_sessions) > self._max_sessions_per_user:
            all_sessions = all_sessions[: self._max_sessions_per_user]

        logger.info(
            f"Extração concluída para user_id={user_id}: "
            f"{len(all_sessions)} sessões",
            extra={
                "user_id": user_id,
                "sessions_count": len(all_sessions),
            },
        )

        # Log warning se não há dados para o usuário (R2.7)
        if len(all_sessions) == 0:
            logger.warning(
                f"Sem dados de sessão para user_id={user_id} "
                f"no período {from_date} a {to_date or 'atual'}",
                extra={
                    "user_id": user_id,
                    "from_date": from_date,
                    "to_date": to_date,
                },
            )

        return all_sessions

    async def extract_batch(
        self,
        user_ids: list[str],
        from_date: str,
        to_date: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Extrai sessões para múltiplos usuários com concorrência limitada.

        Processa até MAX_CONCURRENT usuários em paralelo, usando
        asyncio.Semaphore para controlar concorrência. Loga progresso
        a cada 50 usuários processados ou 60 segundos (R8.2).

        Args:
            user_ids: Lista de IDs de usuários para extração.
            from_date: Data de início para todos os usuários.
            to_date: Data de fim (opcional).

        Returns:
            Dicionário mapeando user_id -> lista de sessões.
            Usuários com erro ou sem dados terão lista vazia.

        Raises:
            NPAWAuthenticationError: Se a API retornar 401/403
                (aborta processamento de todos os usuários).
        """
        results: dict[str, list[dict[str, Any]]] = {}
        total_users = len(user_ids)
        processed_count = 0
        failed_count = 0
        last_progress_time = time.monotonic()
        progress_interval_users = 50
        progress_interval_seconds = 60.0

        logger.info(
            f"Iniciando extração em batch para {total_users} usuários",
            extra={
                "total_users": total_users,
                "max_concurrent": self._max_concurrent,
            },
        )

        # Lock para proteger contadores compartilhados
        counter_lock = asyncio.Lock()

        async def _log_progress_if_needed() -> None:
            """Loga progresso se atingiu threshold de 50 users ou 60s."""
            nonlocal last_progress_time
            now = time.monotonic()
            elapsed = now - last_progress_time

            should_log = (
                processed_count % progress_interval_users == 0
                or elapsed >= progress_interval_seconds
            )

            if should_log and processed_count > 0:
                remaining = total_users - processed_count
                logger.info(
                    f"Extração em progresso: {processed_count}/{total_users} "
                    f"usuários processados, {failed_count} falhas",
                    extra={
                        "processed": processed_count,
                        "total": total_users,
                        "remaining": remaining,
                        "failed": failed_count,
                    },
                )
                last_progress_time = now

        async def _extract_with_semaphore(uid: str) -> None:
            """Extrai sessões de um user respeitando o semáforo."""
            nonlocal processed_count, failed_count

            async with self._semaphore:
                try:
                    sessions = await self.extract_user_sessions(
                        uid, from_date, to_date
                    )
                    results[uid] = sessions
                except NPAWAuthenticationError:
                    # Propagar erros de autenticação (abortar tudo)
                    raise
                except NPAWExtractorError as e:
                    logger.warning(
                        f"Erro ao extrair user_id={uid}: {e}",
                        extra={"user_id": uid, "error": str(e)},
                    )
                    results[uid] = []
                    async with counter_lock:
                        failed_count += 1

                async with counter_lock:
                    processed_count += 1
                    await _log_progress_if_needed()

        # Criar tasks para todos os usuários
        tasks = [
            asyncio.create_task(_extract_with_semaphore(uid))
            for uid in user_ids
        ]

        try:
            await asyncio.gather(*tasks)
        except NPAWAuthenticationError:
            # Cancelar tasks pendentes ao detectar erro de autenticação
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Aguardar cancelamento antes de propagar
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        logger.info(
            f"Extração em batch concluída: "
            f"{len(results)} usuários processados",
            extra={
                "total_users": total_users,
                "users_with_data": sum(
                    1 for s in results.values() if len(s) > 0
                ),
                "users_without_data": sum(
                    1 for s in results.values() if len(s) == 0
                ),
                "failed": failed_count,
            },
        )

        return results
