"""Testes unitários para o FeatureEngineer.

Validates: Requirements 3.1, 3.2, 3.3, 3.5
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

import pytest

from src.features.feature_engineer import FeatureEngineer


def _make_session(
    effective_time: int = 1800000,  # 30 min em ms
    end_at: Optional[str] = None,
    content_channel: str = "HBO",
    happiness_score: Optional[float] = 7.5,
    buffer_ratio: Optional[float] = 0.02,
    error_code: Optional[str] = None,
    avg_bitrate: Optional[float] = 5000000.0,
    content_type: str = "EPISODE",
    device_model: str = "Samsung TV",
    pause_count: Optional[int] = 3,
    seek_count: Optional[int] = 2,
) -> dict:
    """Helper para criar sessões de teste."""
    if end_at is None:
        end_at = datetime.now(timezone.utc).isoformat()

    return {
        "effective_time": effective_time,
        "end_at": end_at,
        "content_channel": content_channel,
        "happiness_score": happiness_score,
        "buffer_ratio": buffer_ratio,
        "error_code": error_code,
        "avg_bitrate": avg_bitrate,
        "content_type": content_type,
        "device": {"device_model": device_model},
        "pause_count": pause_count,
        "seek_count": seek_count,
    }


def _make_sessions(count: int = 10, **kwargs) -> list[dict]:
    """Gera N sessões com valores padrão."""
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    sessions = []
    for i in range(count):
        end_at = (base_time + timedelta(days=i)).isoformat()
        sessions.append(_make_session(end_at=end_at, **kwargs))
    return sessions


class TestFeatureEngineerMinSessions:
    """Testa o comportamento com sessões insuficientes (R3.5)."""

    def test_returns_none_with_less_than_min_sessions(self):
        """Deve retornar None se user tem < 5 sessões."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=4)
        result = engineer.compute("user_1", sessions)
        assert result is None

    def test_returns_none_with_zero_sessions(self):
        """Deve retornar None se user tem 0 sessões."""
        engineer = FeatureEngineer()
        result = engineer.compute("user_1", [])
        assert result is None

    def test_returns_feature_vector_with_exactly_min_sessions(self):
        """Deve retornar FeatureVector com exatamente 5 sessões."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=5)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.user_id == "user_1"


class TestEngagementFeatures:
    """Testa features de engagement (R3.1)."""

    def test_total_sessions(self):
        """total_sessions = contagem de sessões."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=10)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.total_sessions == 10

    def test_total_viewing_hours(self):
        """total_viewing_hours = soma de effective_time (ms) / 3600000."""
        engineer = FeatureEngineer()
        # 5 sessões de 3600000 ms (1 hora cada) = 5 horas
        sessions = _make_sessions(count=5, effective_time=3600000)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.total_viewing_hours == 5.0

    def test_avg_session_duration_min(self):
        """avg_session_duration_min = média de effective_time (ms) / 60000."""
        engineer = FeatureEngineer()
        # 5 sessões de 1800000 ms (30 min) = média 30 min
        sessions = _make_sessions(count=5, effective_time=1800000)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_session_duration_min == 30.0

    def test_sessions_per_week(self):
        """sessions_per_week = total_sessions / weeks_in_period."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 7 sessões distribuídas em 7 dias (dias 0..6 = 6 dias de range = 6/7 semanas, min 1.0)
        # Com min 1.0 semana: 7 sessões / 1.0 semana = 7.0
        sessions = []
        for i in range(7):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        # 7 sessões, range = 6 dias = 0.857 semanas -> clamp a 1.0 -> 7/1.0 = 7.0
        assert result.sessions_per_week == pytest.approx(7.0, rel=0.01)

    def test_distinct_channels(self):
        """distinct_channels = contagem de content_channel únicos."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        channels = ["HBO", "ESPN", "FOX", "HBO", "ESPN"]
        sessions = []
        for i, ch in enumerate(channels):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, content_channel=ch))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.distinct_channels == 3  # HBO, ESPN, FOX


class TestQualityFeatures:
    """Testa features de qualidade (R3.2)."""

    def test_avg_happiness_score(self):
        """avg_happiness_score = média de happiness_score (0-10, excluindo null/negativos)."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        scores = [8.0, 6.0, 7.0, 9.0, 5.0]
        sessions = []
        for i, score in enumerate(scores):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, happiness_score=score))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_happiness_score == pytest.approx(7.0, rel=0.01)

    def test_happiness_score_excludes_negatives(self):
        """Scores negativos são excluídos do cálculo."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        scores = [8.0, -1.0, 6.0, -5.0, 10.0]
        sessions = []
        for i, score in enumerate(scores):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, happiness_score=score))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        # Apenas 8.0, 6.0, 10.0 são válidos (0-10) -> média 8.0
        assert result.avg_happiness_score == pytest.approx(8.0, rel=0.01)

    def test_avg_buffer_ratio(self):
        """avg_buffer_ratio = média de buffer_ratio."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ratios = [0.01, 0.02, 0.03, 0.04, 0.05]
        sessions = []
        for i, ratio in enumerate(ratios):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, buffer_ratio=ratio))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_buffer_ratio == pytest.approx(0.03, rel=0.01)

    def test_error_rate(self):
        """error_rate = count(sessões com error_code não vazio) / total_sessions."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 2 de 5 sessões com erro
        errors = ["ERR_001", None, "ERR_002", None, None]
        sessions = []
        for i, err in enumerate(errors):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, error_code=err))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.error_rate == pytest.approx(0.4, rel=0.01)

    def test_error_rate_empty_string_not_counted(self):
        """Strings vazias em error_code não contam como erro."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        errors = ["", None, "", None, "ERR_001"]
        sessions = []
        for i, err in enumerate(errors):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, error_code=err))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        # Apenas 1 sessão com erro real
        assert result.error_rate == pytest.approx(0.2, rel=0.01)

    def test_avg_bitrate(self):
        """avg_bitrate = média de avg_bitrate (bps)."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=5, avg_bitrate=4000000.0)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_bitrate == pytest.approx(4000000.0, rel=0.01)


class TestBehavioralFeatures:
    """Testa features comportamentais (R3.3)."""

    def test_content_type_percentages(self):
        """Percentuais de content_type devem somar 100%."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 3 EPISODE, 1 SPORT, 1 LIVE = total known 5
        types = ["EPISODE", "EPISODE", "EPISODE", "SPORT", "LIVE"]
        sessions = []
        for i, ct in enumerate(types):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, content_type=ct))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.pct_episode == pytest.approx(60.0, rel=0.01)
        assert result.pct_sport == pytest.approx(20.0, rel=0.01)
        assert result.pct_live == pytest.approx(20.0, rel=0.01)
        assert result.pct_show == pytest.approx(0.0, abs=0.01)
        # Soma deve ser 100%
        total_pct = result.pct_episode + result.pct_sport + result.pct_live + result.pct_show
        assert total_pct == pytest.approx(100.0, abs=0.01)

    def test_distinct_devices_nested(self):
        """distinct_devices com formato aninhado device.device_model."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        devices = ["Samsung TV", "iPhone 15", "Samsung TV", "iPad Pro", "iPhone 15"]
        sessions = []
        for i, dev in enumerate(devices):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, device_model=dev))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.distinct_devices == 3  # Samsung TV, iPhone 15, iPad Pro

    def test_avg_pause_count(self):
        """avg_pause_count = média de pause_count."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pauses = [2, 4, 6, 8, 10]
        sessions = []
        for i, p in enumerate(pauses):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, pause_count=p))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_pause_count == pytest.approx(6.0, rel=0.01)

    def test_avg_seek_count(self):
        """avg_seek_count = média de seek_count."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        seeks = [1, 3, 5, 7, 9]
        sessions = []
        for i, s in enumerate(seeks):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at, seek_count=s))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_seek_count == pytest.approx(5.0, rel=0.01)


class TestDefaultValues:
    """Testa fallback para 0.0 em dados faltantes (R3.5)."""

    def test_null_happiness_scores_default_zero(self):
        """Quando todos happiness_score são None, default = 0.0."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=5, happiness_score=None)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_happiness_score == 0.0

    def test_null_buffer_ratio_default_zero(self):
        """Quando todos buffer_ratio são None, default = 0.0."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=5, buffer_ratio=None)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_buffer_ratio == 0.0

    def test_null_bitrate_default_zero(self):
        """Quando todos avg_bitrate são None, default = 0.0."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=5, avg_bitrate=None)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_bitrate == 0.0

    def test_null_pause_seek_default_zero(self):
        """Quando pause_count e seek_count são None, default = 0.0."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=5, pause_count=None, seek_count=None)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.avg_pause_count == 0.0
        assert result.avg_seek_count == 0.0


class TestTrendFeatures:
    """Testa features de tendência (R3.4, R3.7)."""

    def test_trends_none_when_period_less_than_4_weeks(self):
        """Trends devem ser None quando período < 4 semanas (R3.7)."""
        engineer = FeatureEngineer()
        # Sessões em 2 semanas (14 dias) -> weeks_in_period = 2.0
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sessions = []
        for i in range(7):
            end_at = (base_time + timedelta(days=i * 2)).isoformat()
            sessions.append(_make_session(end_at=end_at))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.viewing_time_trend is None
        assert result.error_rate_trend is None
        assert result.session_frequency_trend is None

    def test_trends_computed_when_period_gte_4_weeks(self):
        """Trends devem ser calculados quando período >= 4 semanas (R3.4)."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 5 sessões espalhadas em 5 semanas (1 por semana)
        sessions = []
        for i in range(5):
            end_at = (base_time + timedelta(weeks=i)).isoformat()
            sessions.append(_make_session(end_at=end_at))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.viewing_time_trend is not None
        assert result.error_rate_trend is not None
        assert result.session_frequency_trend is not None

    def test_viewing_time_trend_positive_slope(self):
        """viewing_time_trend positivo quando horas aumentam ao longo das semanas."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Semana 0: 1h, Semana 1: 2h, Semana 2: 3h, Semana 3: 4h, Semana 4: 5h
        sessions = []
        for week in range(5):
            # effective_time em ms: (week+1) * 3600000 = (week+1) horas
            end_at = (base_time + timedelta(weeks=week)).isoformat()
            sessions.append(_make_session(
                end_at=end_at,
                effective_time=(week + 1) * 3600000,
            ))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        # Slope positivo (1 hora a mais por semana)
        assert result.viewing_time_trend is not None
        assert result.viewing_time_trend > 0.0
        # Slope = 1.0 (1h a mais por semana)
        assert result.viewing_time_trend == pytest.approx(1.0, rel=0.01)

    def test_viewing_time_trend_negative_slope(self):
        """viewing_time_trend negativo quando horas diminuem."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Semana 0: 5h, Semana 1: 4h, Semana 2: 3h, Semana 3: 2h, Semana 4: 1h
        sessions = []
        for week in range(5):
            end_at = (base_time + timedelta(weeks=week)).isoformat()
            sessions.append(_make_session(
                end_at=end_at,
                effective_time=(5 - week) * 3600000,
            ))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.viewing_time_trend is not None
        assert result.viewing_time_trend < 0.0
        # Slope = -1.0
        assert result.viewing_time_trend == pytest.approx(-1.0, rel=0.01)

    def test_error_rate_trend_increasing(self):
        """error_rate_trend positivo quando erros aumentam no final."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sessions = []
        # Semana 0 e 1: sem erros (2 sessões cada)
        for day in range(4):
            end_at = (base_time + timedelta(days=day * 4)).isoformat()
            sessions.append(_make_session(end_at=end_at, error_code=None))
        # Semana 3 e 4: com erros (1 sessão com erro em cada)
        for day in range(4):
            week_offset = 3 * 7 + day * 4
            end_at = (base_time + timedelta(days=week_offset)).isoformat()
            sessions.append(_make_session(end_at=end_at, error_code="ERR_001"))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.error_rate_trend is not None
        # Últimas 2 semanas: todos com erro -> 1.0
        # Primeiras 2 semanas: nenhum com erro -> 0.0
        # Trend = 1.0 - 0.0 = 1.0
        assert result.error_rate_trend > 0.0

    def test_session_frequency_trend_decreasing(self):
        """session_frequency_trend negativo quando frequência diminui."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sessions = []
        # Semanas 0 e 1: 4 sessões cada (8 total nas 2 primeiras)
        for i in range(4):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at))
        for i in range(4):
            end_at = (base_time + timedelta(days=7 + i)).isoformat()
            sessions.append(_make_session(end_at=end_at))
        # Semanas 3 e 4: 1 sessão cada (2 total nas 2 últimas)
        end_at = (base_time + timedelta(days=21)).isoformat()
        sessions.append(_make_session(end_at=end_at))
        end_at = (base_time + timedelta(days=28)).isoformat()
        sessions.append(_make_session(end_at=end_at))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.session_frequency_trend is not None
        # Primeiras 2 semanas: 4 sessões/semana
        # Últimas 2 semanas: 1 sessão/semana
        # Trend = 1 - 4 = -3.0
        assert result.session_frequency_trend < 0.0

    def test_trends_default_zero_for_constant_data(self):
        """Trends = 0.0 quando dados são constantes ao longo das semanas."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 1 sessão por semana, mesma duração, sem erros
        sessions = []
        for week in range(5):
            end_at = (base_time + timedelta(weeks=week)).isoformat()
            sessions.append(_make_session(
                end_at=end_at,
                effective_time=3600000,  # 1h constante
                error_code=None,
            ))
        result = engineer.compute("user_1", sessions)
        assert result is not None
        # Slope = 0 (constante)
        assert result.viewing_time_trend == pytest.approx(0.0, abs=0.001)
        # Error rate igual nas 2 primeiras e últimas semanas
        assert result.error_rate_trend == pytest.approx(0.0, abs=0.001)
        # Frequência igual
        assert result.session_frequency_trend == pytest.approx(0.0, abs=0.001)


class TestManySessionsUser:
    """Testa cenário com user com muitas sessões (10+) (R3.1, R3.2, R3.3)."""

    def test_user_with_many_sessions_produces_valid_feature_vector(self):
        """User com 50 sessões diversificadas gera feature vector completo."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sessions = []
        content_types = ["EPISODE", "SPORT", "LIVE", "SHOW"]
        channels = ["HBO", "ESPN", "FOX", "Disney+", "Globo"]
        devices = ["Samsung TV", "iPhone 15", "iPad Pro", "Chromecast"]

        for i in range(50):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append({
                "effective_time": 1800000 + (i * 60000),
                "end_at": end_at,
                "content_channel": channels[i % len(channels)],
                "happiness_score": 5.0 + (i % 5),
                "buffer_ratio": 0.01 * (i % 10),
                "error_code": "ERR_001" if i % 10 == 0 else None,
                "avg_bitrate": 3000000.0 + (i * 100000),
                "content_type": content_types[i % len(content_types)],
                "device": {"device_model": devices[i % len(devices)]},
                "pause_count": i % 8,
                "seek_count": i % 5,
            })

        result = engineer.compute("user_heavy", sessions)
        assert result is not None
        assert result.total_sessions == 50
        assert result.distinct_channels == 5
        assert result.distinct_devices == 4
        assert result.total_viewing_hours > 0
        assert result.avg_session_duration_min > 0
        # Percentuais somam 100%
        total_pct = (
            result.pct_episode + result.pct_sport
            + result.pct_live + result.pct_show
        )
        assert total_pct == pytest.approx(100.0, abs=0.01)
        # Error rate: 5 sessões com erro (i=0,10,20,30,40) de 50
        assert result.error_rate == pytest.approx(0.1, rel=0.01)

    def test_user_with_many_sessions_sessions_per_week(self):
        """sessions_per_week calculado corretamente com muitas sessões."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 20 sessões distribuídas em 28 dias exatos (4 semanas)
        sessions = []
        for i in range(20):
            end_at = (base_time + timedelta(days=i * 28 // 20)).isoformat()
            sessions.append(_make_session(end_at=end_at))

        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.sessions_per_week > 0
        # O período real: dia 0 a dia 26 = 26 dias = 26/7 semanas = 3.714
        # 20 sessões / 3.714 semanas ≈ 5.38
        assert result.sessions_per_week == pytest.approx(
            20 / (26.0 / 7.0), rel=0.01
        )


class TestShortPeriodTrends:
    """Testa que trends = None quando período < 4 semanas (R3.7)."""

    def test_trends_none_when_period_less_than_4_weeks(self):
        """Com sessões em período < 4 semanas, trends devem ser None."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 10 sessões em apenas 14 dias (2 semanas)
        sessions = []
        for i in range(10):
            end_at = (base_time + timedelta(days=i % 14)).isoformat()
            sessions.append(_make_session(end_at=end_at))

        result = engineer.compute("user_short_period", sessions)
        assert result is not None
        # Período < 4 semanas: trends devem ser None
        assert result.viewing_time_trend is None
        assert result.error_rate_trend is None
        assert result.session_frequency_trend is None

    def test_engagement_quality_behavioral_computed_despite_short_period(self):
        """Features de engagement/quality/behavioral são calculadas mesmo com período curto."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # 7 sessões em 7 dias
        sessions = []
        for i in range(7):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append(_make_session(end_at=end_at))

        result = engineer.compute("user_short", sessions)
        assert result is not None
        # Engagement features são computadas
        assert result.total_sessions == 7
        assert result.total_viewing_hours > 0
        assert result.avg_session_duration_min > 0
        # Quality features são computadas
        assert result.avg_happiness_score > 0
        assert result.avg_buffer_ratio >= 0
        # Behavioral features são computadas
        assert result.distinct_devices >= 1


class TestNullFieldsVariations:
    """Testa combinações de campos null/missing nas sessões (R3.5)."""

    def test_all_fields_null_produces_defaults(self):
        """Sessões com todos os campos opcionais null produzem defaults 0.0."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sessions = []
        for i in range(5):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append({
                "effective_time": None,
                "end_at": end_at,
                "content_channel": None,
                "happiness_score": None,
                "buffer_ratio": None,
                "error_code": None,
                "avg_bitrate": None,
                "content_type": None,
                "device": {"device_model": None},
                "pause_count": None,
                "seek_count": None,
            })

        result = engineer.compute("user_nulls", sessions)
        assert result is not None
        assert result.avg_happiness_score == 0.0
        assert result.avg_buffer_ratio == 0.0
        assert result.avg_bitrate == 0.0
        assert result.avg_pause_count == 0.0
        assert result.avg_seek_count == 0.0
        assert result.avg_session_duration_min == 0.0
        assert result.distinct_channels == 0
        assert result.distinct_devices == 0

    def test_mixed_null_and_valid_fields(self):
        """Sessões com mix de campos null e válidos calculam corretamente."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sessions = []
        for i in range(6):
            end_at = (base_time + timedelta(days=i)).isoformat()
            # Sessões pares: happiness=8.0, ímpares: None
            happiness = 8.0 if i % 2 == 0 else None
            # Sessões pares: buffer=0.05, ímpares: None
            buffer = 0.05 if i % 2 == 0 else None
            sessions.append({
                "effective_time": 3600000,
                "end_at": end_at,
                "content_channel": "HBO",
                "happiness_score": happiness,
                "buffer_ratio": buffer,
                "error_code": None,
                "avg_bitrate": 5000000.0,
                "content_type": "EPISODE",
                "device": {"device_model": "Samsung TV"},
                "pause_count": 2,
                "seek_count": 1,
            })

        result = engineer.compute("user_mixed", sessions)
        assert result is not None
        # Apenas as 3 sessões com happiness válido são consideradas
        assert result.avg_happiness_score == pytest.approx(8.0, rel=0.01)
        # Apenas as 3 sessões com buffer válido
        assert result.avg_buffer_ratio == pytest.approx(0.05, rel=0.01)
        # Bitrate todas válidas
        assert result.avg_bitrate == pytest.approx(5000000.0, rel=0.01)

    def test_missing_device_key_entirely(self):
        """Sessões sem a chave 'device' não quebram o cálculo."""
        engineer = FeatureEngineer()
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sessions = []
        for i in range(5):
            end_at = (base_time + timedelta(days=i)).isoformat()
            sessions.append({
                "effective_time": 1800000,
                "end_at": end_at,
                "content_channel": "HBO",
                "happiness_score": 7.0,
                "buffer_ratio": 0.01,
                "error_code": None,
                "avg_bitrate": 4000000.0,
                "content_type": "EPISODE",
                # Sem chave 'device' nem 'device_model'
                "pause_count": 2,
                "seek_count": 1,
            })

        result = engineer.compute("user_no_device", sessions)
        assert result is not None
        assert result.distinct_devices == 0


class TestFeatureVectorMetadata:
    """Testa metadados do FeatureVector gerado."""

    def test_user_id_is_set(self):
        """O user_id no resultado deve corresponder ao input."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=5)
        result = engineer.compute("user_abc", sessions)
        assert result is not None
        assert result.user_id == "user_abc"

    def test_observation_period_is_set(self):
        """observation_start e observation_end devem ser preenchidos."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=5)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.observation_start is not None
        assert result.observation_end is not None
        # Devem ser parseable como ISO 8601
        datetime.fromisoformat(result.observation_start)
        datetime.fromisoformat(result.observation_end)

    def test_generated_at_is_set(self):
        """generated_at deve ser preenchido com timestamp atual."""
        engineer = FeatureEngineer()
        sessions = _make_sessions(count=5)
        result = engineer.compute("user_1", sessions)
        assert result is not None
        assert result.generated_at is not None
        datetime.fromisoformat(result.generated_at)
