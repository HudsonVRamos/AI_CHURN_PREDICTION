"""Gerador de usuários sintéticos para testes end-to-end.

Produz 10 churned users + 10 active users com sessões sintéticas que refletem
padrões comportamentais distintos:
- Churned: menor engajamento, mais erros, tendência declinante
- Active: engajamento consistente, altos happiness scores

Validates: Requirements 10.5, 12.6
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

# UUIDs fixos para reprodutibilidade nos testes
CHURNED_USER_IDS: list[str] = [
    "a1b2c3d4-1111-4aaa-8000-000000000001",
    "a1b2c3d4-1111-4aaa-8000-000000000002",
    "a1b2c3d4-1111-4aaa-8000-000000000003",
    "a1b2c3d4-1111-4aaa-8000-000000000004",
    "a1b2c3d4-1111-4aaa-8000-000000000005",
    "a1b2c3d4-1111-4aaa-8000-000000000006",
    "a1b2c3d4-1111-4aaa-8000-000000000007",
    "a1b2c3d4-1111-4aaa-8000-000000000008",
    "a1b2c3d4-1111-4aaa-8000-000000000009",
    "a1b2c3d4-1111-4aaa-8000-000000000010",
]

ACTIVE_USER_IDS: list[str] = [
    "b2c3d4e5-2222-4bbb-9000-000000000001",
    "b2c3d4e5-2222-4bbb-9000-000000000002",
    "b2c3d4e5-2222-4bbb-9000-000000000003",
    "b2c3d4e5-2222-4bbb-9000-000000000004",
    "b2c3d4e5-2222-4bbb-9000-000000000005",
    "b2c3d4e5-2222-4bbb-9000-000000000006",
    "b2c3d4e5-2222-4bbb-9000-000000000007",
    "b2c3d4e5-2222-4bbb-9000-000000000008",
    "b2c3d4e5-2222-4bbb-9000-000000000009",
    "b2c3d4e5-2222-4bbb-9000-000000000010",
]

# Constantes de geração
CONTENT_TYPES = ["EPISODE", "SPORT", "LIVE", "SHOW"]
DEVICES = ["SmartTV-Samsung", "SmartTV-LG", "Mobile-iOS", "Mobile-Android", "Web-Chrome", "Roku"]
CHANNELS = [
    "Sky Sports", "HBO", "ESPN", "Fox", "TNT", "Discovery",
    "National Geographic", "Telecine", "Megapix", "Multishow",
]


def _random_iso_timestamp(base: datetime, offset_hours: int = 0) -> str:
    """Gera timestamp ISO 8601 a partir de base + offset."""
    dt = base + timedelta(hours=offset_hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_sessions(
    user_id: str,
    num_sessions: int,
    start_date: datetime,
    end_date: datetime,
    *,
    is_churned: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """Gera sessões sintéticas para um usuário.

    Churned users:
    - happiness_score mais baixo (2-6)
    - error_rate mais alto (~15-30%)
    - sessões mais curtas
    - tendência declinante (menos sessões no final do período)

    Active users:
    - happiness_score alto (6-10)
    - error_rate baixo (~0-5%)
    - sessões mais longas e consistentes
    - distribuição uniforme ao longo do período
    """
    rng = random.Random(seed)
    sessions: list[dict[str, Any]] = []

    total_span = (end_date - start_date).total_seconds()

    for i in range(num_sessions):
        # Distribuição temporal: churned concentra no início, active distribui uniformemente
        if is_churned:
            # Peso decrescente: mais sessões no início do período
            weight = 1.0 - (i / num_sessions) * 0.7
            time_offset = rng.uniform(0, total_span * weight)
        else:
            time_offset = rng.uniform(0, total_span)

        session_start = start_date + timedelta(seconds=time_offset)
        session_end = session_start + timedelta(
            minutes=rng.uniform(5, 45) if is_churned else rng.uniform(20, 120)
        )

        # Métricas de qualidade diferenciadas
        if is_churned:
            happiness = round(rng.uniform(2.0, 6.0), 1)
            buffer_ratio = round(rng.uniform(0.02, 0.15), 4)
            has_error = rng.random() < 0.20  # 20% erro
            avg_bitrate = round(rng.uniform(800_000, 3_000_000), 0)
        else:
            happiness = round(rng.uniform(6.5, 10.0), 1)
            buffer_ratio = round(rng.uniform(0.0, 0.03), 4)
            has_error = rng.random() < 0.03  # 3% erro
            avg_bitrate = round(rng.uniform(3_000_000, 8_000_000), 0)

        effective_ms = int((session_end - session_start).total_seconds() * 1000)
        content_type = rng.choice(CONTENT_TYPES)
        device = rng.choice(DEVICES if not is_churned else DEVICES[:3])
        channel = rng.choice(CHANNELS if not is_churned else CHANNELS[:4])

        session = {
            "session_id": str(uuid.uuid4()),
            "user_id": user_id,
            "start_at": session_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_at": session_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "effective_time": effective_ms,
            "happiness_score": happiness,
            "buffer_ratio": buffer_ratio,
            "error_code": f"ERR_{rng.randint(100, 999)}" if has_error else "",
            "avg_bitrate": avg_bitrate,
            "content_type": content_type,
            "content_channel": channel,
            "device": {
                "device_model": device,
                "device_type": device.split("-")[0],
            },
            "pause_count": rng.randint(0, 3) if is_churned else rng.randint(0, 8),
            "seek_count": rng.randint(0, 2) if is_churned else rng.randint(0, 5),
        }
        sessions.append(session)

    # Ordenar por end_at descendente (como a API NPAW retorna)
    sessions.sort(key=lambda s: s["end_at"], reverse=True)
    return sessions


def generate_churned_users(
    reference_date: datetime | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Gera sessões para 10 churned users.

    Retorna dict mapeando user_id → lista de sessões.
    Churned users têm 20-50 sessões (menor engajamento) ao longo de 5 meses.
    """
    if reference_date is None:
        reference_date = datetime(2024, 6, 1, tzinfo=timezone.utc)

    result: dict[str, list[dict[str, Any]]] = {}

    for idx, user_id in enumerate(CHURNED_USER_IDS):
        seed = 1000 + idx
        rng = random.Random(seed)
        num_sessions = rng.randint(20, 50)
        # Período de 5 meses antes da data de referência (cancelamento)
        start_date = reference_date - timedelta(days=150)
        end_date = reference_date - timedelta(days=7)  # última sessão ~1 semana antes de cancelar

        sessions = _generate_sessions(
            user_id=user_id,
            num_sessions=num_sessions,
            start_date=start_date,
            end_date=end_date,
            is_churned=True,
            seed=seed,
        )
        result[user_id] = sessions

    return result


def generate_active_users(
    reference_date: datetime | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Gera sessões para 10 active users.

    Retorna dict mapeando user_id → lista de sessões.
    Active users têm 50-100 sessões (alto engajamento) ao longo de 6 meses.
    """
    if reference_date is None:
        reference_date = datetime(2024, 6, 1, tzinfo=timezone.utc)

    result: dict[str, list[dict[str, Any]]] = {}

    for idx, user_id in enumerate(ACTIVE_USER_IDS):
        seed = 2000 + idx
        rng = random.Random(seed)
        num_sessions = rng.randint(50, 100)
        # Período de 6 meses, com sessões até a data de referência
        start_date = reference_date - timedelta(days=180)
        end_date = reference_date

        sessions = _generate_sessions(
            user_id=user_id,
            num_sessions=num_sessions,
            start_date=start_date,
            end_date=end_date,
            is_churned=False,
            seed=seed,
        )
        result[user_id] = sessions

    return result


def generate_all_users(
    reference_date: datetime | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Gera sessões para todos os 20 usuários (10 churned + 10 active)."""
    churned = generate_churned_users(reference_date)
    active = generate_active_users(reference_date)
    return {**churned, **active}
