"""Property-based tests para Feature Engineer.

Valida que Feature Vectors produzidos são completos e consistentes:
todas as features presentes, percentuais somam 100%, ranges corretos.

**Validates: Requirements 3.1, 3.2, 3.3, 3.5**
**Property 6: Data Integrity** — Feature Vectors produzidos são completos
e consistentes (todas as features presentes, percentuais somam 100%).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.features.feature_engineer import FeatureEngineer


# --- Estratégias para gerar sessões sintéticas ---

content_type_strategy = st.sampled_from(["EPISODE", "SPORT", "LIVE", "SHOW"])

error_code_strategy = st.one_of(
    st.none(),
    st.text(min_size=0, max_size=10),
)

happiness_score_strategy = st.one_of(
    st.none(),
    st.floats(min_value=-5.0, max_value=15.0, allow_nan=False, allow_infinity=False),
)

effective_time_strategy = st.integers(min_value=0, max_value=10_000_000)


def session_strategy(base_date: datetime, day_offset: int):
    """Estratégia para gerar uma sessão sintética com data fixa."""
    end_at = (base_date + timedelta(days=day_offset)).isoformat()

    return st.fixed_dictionaries({
        "effective_time": effective_time_strategy,
        "end_at": st.just(end_at),
        "content_channel": st.text(min_size=1, max_size=5),
        "happiness_score": happiness_score_strategy,
        "buffer_ratio": st.one_of(
            st.none(),
            st.floats(
                min_value=0.0, max_value=1.0,
                allow_nan=False, allow_infinity=False,
            ),
        ),
        "error_code": error_code_strategy,
        "avg_bitrate": st.one_of(
            st.none(),
            st.floats(
                min_value=0.0, max_value=50_000_000.0,
                allow_nan=False, allow_infinity=False,
            ),
        ),
        "content_type": content_type_strategy,
        "device": st.fixed_dictionaries({
            "device_model": st.text(min_size=1, max_size=10),
        }),
        "pause_count": st.one_of(
            st.none(),
            st.integers(min_value=0, max_value=100),
        ),
        "seek_count": st.one_of(
            st.none(),
            st.integers(min_value=0, max_value=100),
        ),
    })


@st.composite
def sessions_strategy(draw, min_size=5, max_size=50):
    """Gera uma lista de sessões válidas com datas espalhadas."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    sessions = []
    for i in range(count):
        session = draw(session_strategy(base_date, day_offset=i))
        sessions.append(session)
    return sessions


# --- Property Tests ---

engineer = FeatureEngineer()


@given(sessions=sessions_strategy(min_size=5, max_size=50))
@settings(max_examples=200, deadline=None)
def test_feature_vector_is_complete(sessions: list[dict]) -> None:
    """Para qualquer conjunto de ≥5 sessões, o feature vector é completo.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.5**

    Propriedade: todas as features obrigatórias estão presentes e não-None
    (exceto trend fields que podem ser None).
    """
    result = engineer.compute("test_user", sessions)

    # Com ≥ 5 sessões, resultado nunca deve ser None
    assert result is not None, (
        f"FeatureVector não deveria ser None com {len(sessions)} sessões"
    )

    # Campos obrigatórios de metadata
    assert result.user_id == "test_user"
    assert result.version is not None
    assert result.generated_at is not None
    assert result.observation_start is not None
    assert result.observation_end is not None

    # Features de engagement (R3.1) - nunca None
    assert result.total_sessions is not None
    assert result.total_viewing_hours is not None
    assert result.avg_session_duration_min is not None
    assert result.sessions_per_week is not None
    assert result.distinct_channels is not None

    # Features de qualidade (R3.2) - nunca None
    assert result.avg_happiness_score is not None
    assert result.avg_buffer_ratio is not None
    assert result.error_rate is not None
    assert result.avg_bitrate is not None

    # Features comportamentais (R3.3) - nunca None
    assert result.pct_episode is not None
    assert result.pct_sport is not None
    assert result.pct_live is not None
    assert result.pct_show is not None
    assert result.distinct_devices is not None
    assert result.avg_pause_count is not None
    assert result.avg_seek_count is not None


@given(sessions=sessions_strategy(min_size=5, max_size=50))
@settings(max_examples=200, deadline=None)
def test_content_type_percentages_sum_to_100(sessions: list[dict]) -> None:
    """Percentuais de content type sempre somam 100%.

    **Validates: Requirements 3.3, 3.5**

    Propriedade: pct_episode + pct_sport + pct_live + pct_show == 100.0
    (com tolerância de 0.01).
    """
    result = engineer.compute("test_user", sessions)
    assert result is not None

    total_pct = (
        result.pct_episode
        + result.pct_sport
        + result.pct_live
        + result.pct_show
    )
    assert abs(total_pct - 100.0) <= 0.01, (
        f"Percentuais de content type somam {total_pct:.4f}%, "
        f"esperado 100.0% (±0.01). "
        f"Valores: episode={result.pct_episode}, sport={result.pct_sport}, "
        f"live={result.pct_live}, show={result.pct_show}"
    )


@given(sessions=sessions_strategy(min_size=5, max_size=50))
@settings(max_examples=200, deadline=None)
def test_error_rate_between_0_and_1(sessions: list[dict]) -> None:
    """Error rate está sempre entre 0.0 e 1.0.

    **Validates: Requirements 3.2, 3.5**

    Propriedade: 0.0 <= error_rate <= 1.0
    """
    result = engineer.compute("test_user", sessions)
    assert result is not None

    assert 0.0 <= result.error_rate <= 1.0, (
        f"error_rate={result.error_rate} está fora do range [0.0, 1.0]"
    )


@given(sessions=sessions_strategy(min_size=5, max_size=50))
@settings(max_examples=200, deadline=None)
def test_happiness_score_between_0_and_10(sessions: list[dict]) -> None:
    """Happiness score médio está sempre entre 0.0 e 10.0.

    **Validates: Requirements 3.2, 3.5**

    Propriedade: 0.0 <= avg_happiness_score <= 10.0
    """
    result = engineer.compute("test_user", sessions)
    assert result is not None

    assert 0.0 <= result.avg_happiness_score <= 10.0, (
        f"avg_happiness_score={result.avg_happiness_score} "
        f"está fora do range [0.0, 10.0]"
    )


@given(sessions=sessions_strategy(min_size=5, max_size=50))
@settings(max_examples=200, deadline=None)
def test_non_negative_features(sessions: list[dict]) -> None:
    """Features que representam quantidades nunca são negativas.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.5**

    Propriedade: total_sessions, total_viewing_hours, avg_session_duration_min,
    sessions_per_week, distinct_channels, avg_buffer_ratio, avg_bitrate,
    distinct_devices, avg_pause_count, avg_seek_count >= 0
    """
    result = engineer.compute("test_user", sessions)
    assert result is not None

    non_negative_fields = [
        ("total_sessions", result.total_sessions),
        ("total_viewing_hours", result.total_viewing_hours),
        ("avg_session_duration_min", result.avg_session_duration_min),
        ("sessions_per_week", result.sessions_per_week),
        ("distinct_channels", result.distinct_channels),
        ("avg_buffer_ratio", result.avg_buffer_ratio),
        ("avg_bitrate", result.avg_bitrate),
        ("distinct_devices", result.distinct_devices),
        ("avg_pause_count", result.avg_pause_count),
        ("avg_seek_count", result.avg_seek_count),
        ("pct_episode", result.pct_episode),
        ("pct_sport", result.pct_sport),
        ("pct_live", result.pct_live),
        ("pct_show", result.pct_show),
    ]

    for field_name, value in non_negative_fields:
        assert value >= 0, (
            f"{field_name}={value} é negativo, mas deveria ser >= 0"
        )


@given(sessions=sessions_strategy(min_size=5, max_size=50))
@settings(max_examples=200, deadline=None)
def test_function_never_loses_data(sessions: list[dict]) -> None:
    """Se sessões >= 5, o resultado é sempre not None.

    **Validates: Requirements 3.5**

    Propriedade: com ≥ MIN_SESSIONS sessões, compute() SEMPRE
    retorna um FeatureVector válido (nunca None).
    """
    assume(len(sessions) >= 5)

    result = engineer.compute("test_user", sessions)
    assert result is not None, (
        f"compute() retornou None com {len(sessions)} sessões "
        f"(mínimo exigido: {FeatureEngineer.MIN_SESSIONS})"
    )
    # Verificar que total_sessions reflete a contagem real
    assert result.total_sessions == len(sessions), (
        f"total_sessions={result.total_sessions} difere de "
        f"len(sessions)={len(sessions)}"
    )
