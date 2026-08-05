"""Testes para o módulo src/extractors/npaw_extractor.py.

Verifica:
- Construção correta de parâmetros e headers
- Paginação automática (para no batch < 100 ou max 5000)
- Rate limiting entre chamadas
- Concorrência limitada via semáforo
- Tratamento de erros HTTP (401/403 vs 5xx)
- Retry com backoff exponencial para 5xx e timeout
- Parsing da estrutura de resposta da NPAW
- Logging de progresso a cada 50 users ou 60s
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.extractors.npaw_extractor import (
    NPAWAuthenticationError,
    NPAWExtractor,
    NPAWExtractorError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor() -> NPAWExtractor:
    """Extrator com configurações padrão para testes."""
    return NPAWExtractor(
        account_code="sky_brazil",
        api_key="test-api-key-123",
        rate_limit_seconds=0.0,  # Sem delay nos testes
    )


@pytest.fixture
def extractor_with_rate_limit() -> NPAWExtractor:
    """Extrator com rate limiting ativo para testar delays."""
    return NPAWExtractor(
        account_code="sky_brazil",
        api_key="test-api-key-123",
        rate_limit_seconds=0.1,
    )


def _make_npaw_response(sessions: list[dict], empty: bool = False) -> dict:
    """Cria uma resposta mock no formato da API NPAW."""
    if empty:
        return {"data": [{"values": []}]}
    return {"data": [{"values": sessions}]}


def _make_sessions(count: int, base_id: int = 0) -> list[dict]:
    """Gera N sessões fictícias."""
    return [
        {
            "user_id": "test-user-id",
            "token": f"session-{base_id + i}",
            "effective_time": 60000 * (i + 1),
            "end_at": f"2024-07-{15 - i:02d} 10:00:00",
            "content_channel": f"Channel_{i % 3}",
            "happiness_score": 7.5,
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Testes de inicialização
# ---------------------------------------------------------------------------


class TestNPAWExtractorInit:
    """Testes de inicialização e validação de parâmetros."""

    def test_init_com_parametros_validos(self) -> None:
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="my-key",
        )
        assert extractor._account_code == "sky_brazil"
        assert extractor._api_key == "my-key"
        assert extractor._rate_limit_seconds == 1.0
        assert extractor._max_concurrent == 5
        assert extractor._batch_size == 100
        assert extractor._max_sessions_per_user == 5000

    def test_init_com_parametros_customizados(self) -> None:
        extractor = NPAWExtractor(
            account_code="other_account",
            api_key="key-2",
            rate_limit_seconds=2.0,
            max_concurrent=10,
            batch_size=50,
            max_sessions_per_user=1000,
            base_url="https://custom.api.com",
        )
        assert extractor._rate_limit_seconds == 2.0
        assert extractor._max_concurrent == 10
        assert extractor._batch_size == 50
        assert extractor._max_sessions_per_user == 1000
        assert extractor._base_url == "https://custom.api.com"

    def test_init_rejeita_account_code_vazio(self) -> None:
        with pytest.raises(ValueError, match="account_code"):
            NPAWExtractor(account_code="", api_key="key")

    def test_init_rejeita_api_key_vazia(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            NPAWExtractor(account_code="sky", api_key="")

    def test_endpoint_url(self, extractor: NPAWExtractor) -> None:
        assert extractor.endpoint_url == (
            "https://api.npaw.com/sky_brazil/rawdata"
        )


# ---------------------------------------------------------------------------
# Testes de construção de parâmetros
# ---------------------------------------------------------------------------


class TestBuildParams:
    """Testes para construção de parâmetros de request."""

    def test_params_basicos(self, extractor: NPAWExtractor) -> None:
        params = extractor._build_params(
            user_id="user-123",
            from_date="2024-01-01",
            to_date=None,
            offset=0,
        )
        assert params["fromDate"] == "2024-01-01"
        assert params["limit"] == "100"
        assert params["offset"] == "0"
        assert params["orderBy"] == "end_at"
        assert params["orderDirection"] == "desc"
        assert "toDate" not in params

        # Verificar filtro JSON
        filter_parsed = json.loads(params["filter"])
        assert filter_parsed == [
            {"name": "uf", "rules": {"user_id": ["user-123"]}}
        ]

    def test_params_com_to_date(self, extractor: NPAWExtractor) -> None:
        params = extractor._build_params(
            user_id="user-123",
            from_date="2024-01-01",
            to_date="2024-07-01",
            offset=200,
        )
        assert params["toDate"] == "2024-07-01"
        assert params["offset"] == "200"

    def test_headers_contêm_api_key(self, extractor: NPAWExtractor) -> None:
        headers = extractor._build_headers()
        assert headers == {"npaw-api-key": "test-api-key-123"}


# ---------------------------------------------------------------------------
# Testes de parsing de resposta
# ---------------------------------------------------------------------------


class TestParseSessions:
    """Testes para parsing da estrutura de resposta da NPAW."""

    def test_parse_resposta_com_sessoes(
        self, extractor: NPAWExtractor
    ) -> None:
        sessions = _make_sessions(3)
        response = _make_npaw_response(sessions)
        result = extractor._parse_sessions(response)
        assert len(result) == 3
        assert result[0]["token"] == "session-0"

    def test_parse_resposta_vazia(self, extractor: NPAWExtractor) -> None:
        response = _make_npaw_response([], empty=True)
        result = extractor._parse_sessions(response)
        assert result == []

    def test_parse_none(self, extractor: NPAWExtractor) -> None:
        assert extractor._parse_sessions(None) == []

    def test_parse_sem_data(self, extractor: NPAWExtractor) -> None:
        assert extractor._parse_sessions({}) == []
        assert extractor._parse_sessions({"data": []}) == []

    def test_parse_sem_values(self, extractor: NPAWExtractor) -> None:
        assert extractor._parse_sessions({"data": [{}]}) == []
        assert extractor._parse_sessions({"data": [{"values": None}]}) == []


# ---------------------------------------------------------------------------
# Testes de extração com mock HTTP
# ---------------------------------------------------------------------------


class TestExtractUserSessions:
    """Testes de extração de sessões para um único usuário."""

    @pytest.mark.asyncio
    async def test_extrai_pagina_unica(
        self, extractor: NPAWExtractor
    ) -> None:
        """Quando a API retorna < batch_size, para na primeira página."""
        sessions = _make_sessions(50)
        mock_response = _make_npaw_response(sessions)

        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 200
            mock_ctx.json = AsyncMock(return_value=mock_response)
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)

            MockSession.return_value = mock_session_instance

            result = await extractor.extract_user_sessions(
                user_id="user-abc", from_date="last6months"
            )

        assert len(result) == 50
        # Apenas 1 chamada (< batch_size, não pagina)
        mock_session_instance.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_pagina_multiplas_paginas(
        self, extractor: NPAWExtractor
    ) -> None:
        """Quando retorna batch_size exato, busca próxima página."""
        page1 = _make_sessions(100, base_id=0)
        page2 = _make_sessions(30, base_id=100)

        responses = [
            _make_npaw_response(page1),
            _make_npaw_response(page2),
        ]
        call_count = 0

        async def mock_json():
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 200
            mock_ctx.json = mock_json
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)

            MockSession.return_value = mock_session_instance

            result = await extractor.extract_user_sessions(
                user_id="user-abc", from_date="2024-01-01"
            )

        assert len(result) == 130
        assert mock_session_instance.get.call_count == 2

    @pytest.mark.asyncio
    async def test_respeita_max_sessions_per_user(self) -> None:
        """Para ao atingir o limite máximo de sessões por user."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
            max_sessions_per_user=150,
            batch_size=100,
        )

        page1 = _make_sessions(100, base_id=0)
        page2 = _make_sessions(100, base_id=100)

        responses = [
            _make_npaw_response(page1),
            _make_npaw_response(page2),
        ]
        call_count = 0

        async def mock_json():
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 200
            mock_ctx.json = mock_json
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)

            MockSession.return_value = mock_session_instance

            result = await extractor.extract_user_sessions(
                user_id="user-abc", from_date="last6months"
            )

        # Deve truncar para max_sessions_per_user=150
        assert len(result) == 150

    @pytest.mark.asyncio
    async def test_erro_autenticacao_401(
        self, extractor: NPAWExtractor
    ) -> None:
        """HTTP 401 lança NPAWAuthenticationError."""
        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 401
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)

            MockSession.return_value = mock_session_instance

            with pytest.raises(NPAWAuthenticationError):
                await extractor.extract_user_sessions(
                    user_id="user-abc", from_date="last6months"
                )

    @pytest.mark.asyncio
    async def test_erro_servidor_5xx(
        self, extractor: NPAWExtractor
    ) -> None:
        """HTTP 5xx retries 3x com backoff e depois lança NPAWExtractorError."""
        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 500
            mock_ctx.text = AsyncMock(return_value="Internal Server Error")
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)

            MockSession.return_value = mock_session_instance

            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(
                    NPAWExtractorError, match="após 3 tentativas"
                ):
                    await extractor.extract_user_sessions(
                        user_id="user-abc", from_date="last6months"
                    )

            # Deve ter feito 4 chamadas (1 inicial + 3 retries)
            assert mock_session_instance.get.call_count == 4

    @pytest.mark.asyncio
    async def test_resposta_sem_dados(
        self, extractor: NPAWExtractor
    ) -> None:
        """API retornando array vazio resulta em lista vazia."""
        empty_response = _make_npaw_response([], empty=True)

        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 200
            mock_ctx.json = AsyncMock(return_value=empty_response)
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)

            MockSession.return_value = mock_session_instance

            result = await extractor.extract_user_sessions(
                user_id="user-abc", from_date="last6months"
            )

        assert result == []


# ---------------------------------------------------------------------------
# Testes de rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Testes de controle de rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_aplica_delay(
        self, extractor_with_rate_limit: NPAWExtractor
    ) -> None:
        """Rate limiting deve aguardar entre chamadas consecutivas."""
        extractor = extractor_with_rate_limit

        start = time.monotonic()

        # Simular 3 chamadas com rate limit de 0.1s
        await extractor._wait_rate_limit()
        await extractor._wait_rate_limit()
        await extractor._wait_rate_limit()

        elapsed = time.monotonic() - start
        # Esperar pelo menos 2 * 0.1s = 0.2s (entre 2ª e 3ª chamada)
        assert elapsed >= 0.18  # Margem para variação do SO


# ---------------------------------------------------------------------------
# Testes de extração em batch
# ---------------------------------------------------------------------------


class TestExtractBatch:
    """Testes para extração de múltiplos usuários em paralelo."""

    @pytest.mark.asyncio
    async def test_batch_extrai_multiplos_users(self) -> None:
        """Extrai sessões para vários users em paralelo."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
            max_concurrent=2,
        )

        sessions_u1 = _make_sessions(10, base_id=0)
        sessions_u2 = _make_sessions(5, base_id=10)

        async def mock_extract(
            user_id: str, from_date: str, to_date=None
        ):
            if user_id == "user-1":
                return sessions_u1
            elif user_id == "user-2":
                return sessions_u2
            return []

        with patch.object(
            extractor, "extract_user_sessions", side_effect=mock_extract
        ):
            results = await extractor.extract_batch(
                user_ids=["user-1", "user-2"],
                from_date="2024-01-01",
            )

        assert len(results["user-1"]) == 10
        assert len(results["user-2"]) == 5

    @pytest.mark.asyncio
    async def test_batch_erro_auth_aborta_todos(self) -> None:
        """Erro de autenticação aborta processamento de todos os users."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="bad-key",
            rate_limit_seconds=0.0,
        )

        async def mock_extract(user_id: str, from_date: str, to_date=None):
            raise NPAWAuthenticationError("HTTP 401")

        with patch.object(
            extractor, "extract_user_sessions", side_effect=mock_extract
        ):
            with pytest.raises(NPAWAuthenticationError):
                await extractor.extract_batch(
                    user_ids=["user-1", "user-2", "user-3"],
                    from_date="2024-01-01",
                )

    @pytest.mark.asyncio
    async def test_batch_erro_individual_nao_aborta(self) -> None:
        """Erro não-auth em um user não impede os demais."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
        )

        async def mock_extract(user_id: str, from_date: str, to_date=None):
            if user_id == "user-bad":
                raise NPAWExtractorError("HTTP 500")
            return _make_sessions(3)

        with patch.object(
            extractor, "extract_user_sessions", side_effect=mock_extract
        ):
            results = await extractor.extract_batch(
                user_ids=["user-ok", "user-bad"],
                from_date="2024-01-01",
            )

        assert len(results["user-ok"]) == 3
        assert results["user-bad"] == []

    @pytest.mark.asyncio
    async def test_batch_concorrencia_limitada(self) -> None:
        """Verifica que não excede MAX_CONCURRENT usuários simultâneos."""
        max_concurrent = 2
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
            max_concurrent=max_concurrent,
        )

        concurrent_count = 0
        max_observed_concurrent = 0

        async def mock_extract(user_id: str, from_date: str, to_date=None):
            nonlocal concurrent_count, max_observed_concurrent
            concurrent_count += 1
            max_observed_concurrent = max(
                max_observed_concurrent, concurrent_count
            )
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return _make_sessions(1)

        with patch.object(
            extractor, "extract_user_sessions", side_effect=mock_extract
        ):
            await extractor.extract_batch(
                user_ids=[f"user-{i}" for i in range(6)],
                from_date="2024-01-01",
            )

        assert max_observed_concurrent <= max_concurrent


# ---------------------------------------------------------------------------
# Testes de retry com backoff exponencial
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    """Testes para retry com backoff exponencial em erros 5xx e timeout."""

    @pytest.mark.asyncio
    async def test_retry_5xx_sucesso_na_segunda_tentativa(self) -> None:
        """Retry após 5xx: sucesso na 2ª tentativa não lança erro."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
        )

        sessions = _make_sessions(10)
        success_response = _make_npaw_response(sessions)

        call_count = 0

        class MockContext:
            def __init__(self):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    self.status = 500
                else:
                    self.status = 200

            async def text(self):
                return "Server Error"

            async def json(self):
                return success_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("aiohttp.ClientSession") as MockSession:
            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(
                side_effect=lambda *a, **kw: MockContext()
            )
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = mock_session_instance

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await extractor.extract_user_sessions(
                    user_id="user-abc", from_date="last6months"
                )

            assert len(result) == 10
            # Verifica que o backoff de 2s foi usado na 1ª retry
            mock_sleep.assert_called_once_with(2.0)

    @pytest.mark.asyncio
    async def test_retry_5xx_backoff_exponencial(self) -> None:
        """Verifica delays de backoff: 2s, 4s, 8s."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
        )

        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 503
            mock_ctx.text = AsyncMock(return_value="Service Unavailable")
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = mock_session_instance

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(NPAWExtractorError):
                    await extractor.extract_user_sessions(
                        user_id="user-abc", from_date="last6months"
                    )

            # Deve ter chamado sleep com 2.0, 4.0, 8.0
            sleep_calls = [
                call.args[0] for call in mock_sleep.call_args_list
            ]
            assert sleep_calls == [2.0, 4.0, 8.0]

    @pytest.mark.asyncio
    async def test_retry_timeout_asyncio(self) -> None:
        """asyncio.TimeoutError causa retry com backoff."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
        )

        with patch("aiohttp.ClientSession") as MockSession:
            mock_session_instance = AsyncMock()

            # Simular timeout em todas as tentativas
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_session_instance.get = MagicMock(return_value=mock_ctx)

            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = mock_session_instance

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(
                    NPAWExtractorError, match="Timeout na API NPAW"
                ):
                    await extractor.extract_user_sessions(
                        user_id="user-abc", from_date="last6months"
                    )

            # 3 sleeps para os 3 retries
            assert mock_sleep.call_count == 3
            sleep_calls = [
                call.args[0] for call in mock_sleep.call_args_list
            ]
            assert sleep_calls == [2.0, 4.0, 8.0]

    @pytest.mark.asyncio
    async def test_retry_server_timeout_error(self) -> None:
        """aiohttp.ServerTimeoutError (subclasse de asyncio.TimeoutError)
        causa retry com backoff e mensagem específica."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
        )

        with patch("aiohttp.ClientSession") as MockSession:
            mock_session_instance = AsyncMock()

            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(
                side_effect=aiohttp.ServerTimeoutError(
                    "Connection timeout"
                )
            )
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_session_instance.get = MagicMock(return_value=mock_ctx)

            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = mock_session_instance

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(
                    NPAWExtractorError,
                    match="Timeout do servidor NPAW",
                ):
                    await extractor.extract_user_sessions(
                        user_id="user-abc", from_date="last6months"
                    )

            # 3 sleeps com backoff exponencial
            assert mock_sleep.call_count == 3

    @pytest.mark.asyncio
    async def test_auth_401_nao_faz_retry(self) -> None:
        """HTTP 401 levanta NPAWAuthenticationError sem retry."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
        )

        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 401
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = mock_session_instance

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(NPAWAuthenticationError):
                    await extractor.extract_user_sessions(
                        user_id="user-abc", from_date="last6months"
                    )

            # Apenas 1 chamada (sem retry)
            assert mock_session_instance.get.call_count == 1
            # Sem sleep de backoff
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_auth_403_nao_faz_retry(self) -> None:
        """HTTP 403 levanta NPAWAuthenticationError sem retry."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
        )

        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 403
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = mock_session_instance

            with pytest.raises(NPAWAuthenticationError):
                await extractor.extract_user_sessions(
                    user_id="user-abc", from_date="last6months"
                )

            assert mock_session_instance.get.call_count == 1


# ---------------------------------------------------------------------------
# Testes de logging de progresso
# ---------------------------------------------------------------------------


class TestProgressLogging:
    """Testes para logging de progresso a cada 50 users ou 60s."""

    @pytest.mark.asyncio
    async def test_log_progresso_a_cada_50_users(self) -> None:
        """Loga progresso ao processar 50 e 100 usuários."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
            max_concurrent=10,
        )

        async def mock_extract(user_id: str, from_date: str, to_date=None):
            return _make_sessions(1)

        with patch.object(
            extractor, "extract_user_sessions", side_effect=mock_extract
        ):
            with patch(
                "src.extractors.npaw_extractor.logger"
            ) as mock_logger:
                await extractor.extract_batch(
                    user_ids=[f"user-{i}" for i in range(100)],
                    from_date="2024-01-01",
                )

            # Verificar que houve logs de progresso com a mensagem correta
            info_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Extração em progresso" in str(call)
            ]
            # Deve ter logado ao menos no 50º user
            assert len(info_calls) >= 1

    @pytest.mark.asyncio
    async def test_log_progresso_contem_contagens(self) -> None:
        """Mensagem de progresso contém processed, total e falhas."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
            max_concurrent=50,
        )

        async def mock_extract(user_id: str, from_date: str, to_date=None):
            return _make_sessions(1)

        with patch.object(
            extractor, "extract_user_sessions", side_effect=mock_extract
        ):
            with patch(
                "src.extractors.npaw_extractor.logger"
            ) as mock_logger:
                await extractor.extract_batch(
                    user_ids=[f"user-{i}" for i in range(50)],
                    from_date="2024-01-01",
                )

            # Verificar que mensagem de progresso contém as informações
            info_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Extração em progresso" in str(call)
            ]
            if info_calls:
                msg = str(info_calls[0])
                assert "usuários processados" in msg
                assert "falhas" in msg

    @pytest.mark.asyncio
    async def test_log_progresso_com_falhas(self) -> None:
        """Contagem de falhas é registrada no progresso."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
            max_concurrent=50,
        )

        call_count = 0

        async def mock_extract(user_id: str, from_date: str, to_date=None):
            nonlocal call_count
            call_count += 1
            # Simular falha em 10% dos users
            if call_count % 10 == 0:
                raise NPAWExtractorError("HTTP 500")
            return _make_sessions(1)

        with patch.object(
            extractor, "extract_user_sessions", side_effect=mock_extract
        ):
            results = await extractor.extract_batch(
                user_ids=[f"user-{i}" for i in range(50)],
                from_date="2024-01-01",
            )

        # Deve ter 5 falhas (50 / 10 = 5)
        failed_users = [uid for uid, s in results.items() if len(s) == 0]
        assert len(failed_users) == 5


# ---------------------------------------------------------------------------
# Testes de empty data handling
# ---------------------------------------------------------------------------


class TestEmptyDataHandling:
    """Testes para logging de warning quando user não tem dados."""

    @pytest.mark.asyncio
    async def test_sem_dados_loga_warning(self) -> None:
        """User sem sessões gera log warning com user_id e date range."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
        )

        empty_response = _make_npaw_response([], empty=True)

        with patch("aiohttp.ClientSession") as MockSession:
            mock_ctx = AsyncMock()
            mock_ctx.status = 200
            mock_ctx.json = AsyncMock(return_value=empty_response)
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session_instance = AsyncMock()
            mock_session_instance.get = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = mock_session_instance

            with patch(
                "src.extractors.npaw_extractor.logger"
            ) as mock_logger:
                result = await extractor.extract_user_sessions(
                    user_id="user-empty",
                    from_date="2024-01-01",
                    to_date="2024-07-01",
                )

        assert result == []
        # Verificar que logou warning sobre dados vazios
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "Sem dados de sessão" in str(call)
        ]
        assert len(warning_calls) == 1
        msg = str(warning_calls[0])
        assert "user-empty" in msg
        assert "2024-01-01" in msg

    @pytest.mark.asyncio
    async def test_batch_user_sem_dados_nao_aborta(self) -> None:
        """User sem dados no batch não impede outros de serem processados."""
        extractor = NPAWExtractor(
            account_code="sky_brazil",
            api_key="key",
            rate_limit_seconds=0.0,
        )

        async def mock_extract(user_id: str, from_date: str, to_date=None):
            if user_id == "user-empty":
                return []
            return _make_sessions(5)

        with patch.object(
            extractor, "extract_user_sessions", side_effect=mock_extract
        ):
            results = await extractor.extract_batch(
                user_ids=["user-ok", "user-empty", "user-ok2"],
                from_date="2024-01-01",
            )

        assert len(results["user-ok"]) == 5
        assert results["user-empty"] == []
        assert len(results["user-ok2"]) == 5
