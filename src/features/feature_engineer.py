"""Feature Engineer para o pipeline de predição de churn.

Transforma sessões brutas extraídas da NPAW em Feature Vectors
com métricas de engagement, qualidade e comportamento.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.common.logging import get_logger
from src.common.models import FeatureVector

logger = get_logger("feature-engineering")


class FeatureEngineer:
    """Calcula features comportamentais a partir de sessões brutas NPAW.

    Atributos de classe:
        MIN_SESSIONS: Mínimo de sessões para gerar feature vector (R3.5).
        MIN_WEEKS_FOR_TRENDS: Mínimo de semanas para cálculo de tendências.
    """

    MIN_SESSIONS = 5
    MIN_WEEKS_FOR_TRENDS = 4

    def compute(self, user_id: str, sessions: list[dict]) -> Optional[FeatureVector]:
        """Transforma sessões brutas em feature vector.

        Args:
            user_id: Identificador único do assinante.
            sessions: Lista de dicts com dados de sessão da NPAW.

        Returns:
            FeatureVector populado ou None se < MIN_SESSIONS.
        """
        if len(sessions) < self.MIN_SESSIONS:
            logger.warning(
                f"Usuário {user_id} possui {len(sessions)} sessões "
                f"(mínimo: {self.MIN_SESSIONS}). Ignorando.",
                extra={"user_id": user_id, "session_count": len(sessions)},
            )
            return None

        total_sessions = len(sessions)

        # Calcular período de observação
        observation_start, observation_end = self._compute_observation_period(sessions)
        weeks_in_period = self._compute_weeks_in_period(observation_start, observation_end)

        # Features de engagement (R3.1)
        engagement = self._compute_engagement_features(sessions, total_sessions, weeks_in_period)

        # Features de qualidade (R3.2)
        quality = self._compute_quality_features(sessions, total_sessions)

        # Features comportamentais (R3.3)
        behavioral = self._compute_behavioral_features(sessions, total_sessions)

        # Trends (R3.4, R3.7)
        trends = self._compute_trend_features(sessions, weeks_in_period)

        now_iso = datetime.now(timezone.utc).isoformat()

        feature_vector = FeatureVector(
            user_id=user_id,
            version=1,  # Versão será definida pelo FeatureStore ao persistir
            generated_at=now_iso,
            observation_start=observation_start,
            observation_end=observation_end,
            # Engagement
            total_sessions=engagement["total_sessions"],
            total_viewing_hours=engagement["total_viewing_hours"],
            avg_session_duration_min=engagement["avg_session_duration_min"],
            sessions_per_week=engagement["sessions_per_week"],
            distinct_channels=engagement["distinct_channels"],
            # Quality
            avg_happiness_score=quality["avg_happiness_score"],
            avg_buffer_ratio=quality["avg_buffer_ratio"],
            error_rate=quality["error_rate"],
            avg_bitrate=quality["avg_bitrate"],
            # Behavioral
            pct_episode=behavioral["pct_episode"],
            pct_sport=behavioral["pct_sport"],
            pct_live=behavioral["pct_live"],
            pct_show=behavioral["pct_show"],
            distinct_devices=behavioral["distinct_devices"],
            avg_pause_count=behavioral["avg_pause_count"],
            avg_seek_count=behavioral["avg_seek_count"],
            # Trends (R3.4, R3.7)
            viewing_time_trend=trends["viewing_time_trend"],
            error_rate_trend=trends["error_rate_trend"],
            session_frequency_trend=trends["session_frequency_trend"],
        )

        logger.info(
            f"Feature vector gerado para usuário {user_id}",
            extra={"user_id": user_id, "total_sessions": total_sessions},
        )

        return feature_vector

    def _compute_observation_period(self, sessions: list[dict]) -> tuple[str, str]:
        """Calcula início e fim do período de observação baseado nas sessões.

        Usa o campo 'end_at' das sessões para determinar o range temporal.

        Returns:
            Tupla (observation_start, observation_end) em formato ISO 8601.
        """
        timestamps: list[datetime] = []
        for session in sessions:
            end_at = session.get("end_at")
            if end_at:
                try:
                    ts = datetime.fromisoformat(end_at.replace("Z", "+00:00"))
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    continue

        if not timestamps:
            # Fallback: usar datetime atual
            now = datetime.now(timezone.utc)
            return now.isoformat(), now.isoformat()

        start = min(timestamps)
        end = max(timestamps)
        return start.isoformat(), end.isoformat()

    def _compute_weeks_in_period(self, start_iso: str, end_iso: str) -> float:
        """Calcula o número de semanas no período de observação.

        Returns:
            Número de semanas (mínimo 1.0 para evitar divisão por zero).
        """
        try:
            start = datetime.fromisoformat(start_iso)
            end = datetime.fromisoformat(end_iso)
            delta_days = (end - start).total_seconds() / 86400.0
            weeks = delta_days / 7.0
            return max(weeks, 1.0)
        except (ValueError, TypeError):
            return 1.0

    def _compute_engagement_features(
        self, sessions: list[dict], total_sessions: int, weeks_in_period: float
    ) -> dict:
        """Calcula features de engagement (R3.1).

        - total_sessions: contagem de sessões
        - total_viewing_hours: soma de effective_time (ms) / 3600000
        - avg_session_duration_min: média de effective_time (ms) / 60000
        - sessions_per_week: total_sessions / weeks_in_observation_period
        - distinct_channels: contagem de valores únicos de content_channel
        """
        total_effective_ms = 0.0
        valid_duration_count = 0
        channels: set[str] = set()

        for session in sessions:
            # effective_time em milissegundos
            effective_time = session.get("effective_time")
            if effective_time is not None and effective_time >= 0:
                total_effective_ms += float(effective_time)
                valid_duration_count += 1

            # Canais distintos
            channel = session.get("content_channel")
            if channel:
                channels.add(channel)

        total_viewing_hours = total_effective_ms / 3_600_000.0

        if valid_duration_count > 0:
            avg_session_duration_min = (total_effective_ms / valid_duration_count) / 60_000.0
        else:
            avg_session_duration_min = 0.0  # R3.5: default 0.0 para dados faltantes

        sessions_per_week = total_sessions / weeks_in_period

        return {
            "total_sessions": total_sessions,
            "total_viewing_hours": round(total_viewing_hours, 4),
            "avg_session_duration_min": round(avg_session_duration_min, 4),
            "sessions_per_week": round(sessions_per_week, 4),
            "distinct_channels": len(channels),
        }

    def _compute_quality_features(self, sessions: list[dict], total_sessions: int) -> dict:
        """Calcula features de qualidade (R3.2).

        - avg_happiness_score: média de happiness_score (0-10, excluindo null/negativos)
        - avg_buffer_ratio: média de buffer_ratio
        - error_rate: count(sessões com error_code não vazio) / total_sessions
        - avg_bitrate: média de avg_bitrate (bps)
        """
        happiness_scores: list[float] = []
        buffer_ratios: list[float] = []
        bitrates: list[float] = []
        error_count = 0

        for session in sessions:
            # Happiness score: 0-10, excluir null e negativos
            happiness = session.get("happiness_score")
            if happiness is not None:
                try:
                    h_val = float(happiness)
                    if 0.0 <= h_val <= 10.0:
                        happiness_scores.append(h_val)
                except (ValueError, TypeError):
                    pass

            # Buffer ratio
            buffer = session.get("buffer_ratio")
            if buffer is not None:
                try:
                    b_val = float(buffer)
                    if b_val >= 0.0:
                        buffer_ratios.append(b_val)
                except (ValueError, TypeError):
                    pass

            # Error code: considerar como erro se não-vazio e não-None
            error_code = session.get("error_code")
            if error_code is not None and str(error_code).strip() != "":
                error_count += 1

            # Bitrate médio
            bitrate = session.get("avg_bitrate")
            if bitrate is not None:
                try:
                    br_val = float(bitrate)
                    if br_val >= 0.0:
                        bitrates.append(br_val)
                except (ValueError, TypeError):
                    pass

        # Médias com fallback 0.0 (R3.5)
        avg_happiness = (
            sum(happiness_scores) / len(happiness_scores) if happiness_scores else 0.0
        )
        avg_buffer = sum(buffer_ratios) / len(buffer_ratios) if buffer_ratios else 0.0
        error_rate = error_count / total_sessions if total_sessions > 0 else 0.0
        avg_bitrate = sum(bitrates) / len(bitrates) if bitrates else 0.0

        return {
            "avg_happiness_score": round(avg_happiness, 4),
            "avg_buffer_ratio": round(avg_buffer, 4),
            "error_rate": round(error_rate, 4),
            "avg_bitrate": round(avg_bitrate, 4),
        }

    def _compute_behavioral_features(self, sessions: list[dict], total_sessions: int) -> dict:
        """Calcula features comportamentais (R3.3).

        - pct_episode/sport/live/show: % de sessões por content_type, soma = 100%
        - distinct_devices: valores únicos de device.device_model
        - avg_pause_count: média de pause_count
        - avg_seek_count: média de seek_count
        """
        content_type_counts: dict[str, int] = {
            "EPISODE": 0,
            "SPORT": 0,
            "LIVE": 0,
            "SHOW": 0,
        }
        devices: set[str] = set()
        pause_counts: list[float] = []
        seek_counts: list[float] = []

        for session in sessions:
            # Content type
            content_type = session.get("content_type")
            if content_type and content_type.upper() in content_type_counts:
                content_type_counts[content_type.upper()] += 1

            # Device model (campo aninhado: device.device_model)
            device = session.get("device")
            if isinstance(device, dict):
                device_model = device.get("device_model")
                if device_model:
                    devices.add(device_model)
            else:
                # Suporte para formato flat (device_model direto)
                device_model = session.get("device_model")
                if device_model:
                    devices.add(device_model)

            # Pause count
            pause = session.get("pause_count")
            if pause is not None:
                try:
                    pause_counts.append(float(pause))
                except (ValueError, TypeError):
                    pass

            # Seek count
            seek = session.get("seek_count")
            if seek is not None:
                try:
                    seek_counts.append(float(seek))
                except (ValueError, TypeError):
                    pass

        # Percentuais de content type (soma = 100%)
        known_type_total = sum(content_type_counts.values())

        if known_type_total > 0:
            pct_episode = (content_type_counts["EPISODE"] / known_type_total) * 100.0
            pct_sport = (content_type_counts["SPORT"] / known_type_total) * 100.0
            pct_live = (content_type_counts["LIVE"] / known_type_total) * 100.0
            pct_show = (content_type_counts["SHOW"] / known_type_total) * 100.0
        else:
            # R3.5: Se nenhum content_type válido, distribuir igualmente para somar 100%
            pct_episode = 25.0
            pct_sport = 25.0
            pct_live = 25.0
            pct_show = 25.0

        # Médias com fallback 0.0 (R3.5)
        avg_pause = sum(pause_counts) / len(pause_counts) if pause_counts else 0.0
        avg_seek = sum(seek_counts) / len(seek_counts) if seek_counts else 0.0

        return {
            "pct_episode": round(pct_episode, 4),
            "pct_sport": round(pct_sport, 4),
            "pct_live": round(pct_live, 4),
            "pct_show": round(pct_show, 4),
            "distinct_devices": len(devices),
            "avg_pause_count": round(avg_pause, 4),
            "avg_seek_count": round(avg_seek, 4),
        }

    def _compute_trend_features(
        self, sessions: list[dict], weeks_in_period: float
    ) -> dict[str, Optional[float]]:
        """Calcula features de tendência temporal (R3.4, R3.7).

        - viewing_time_trend: slope da regressão linear do total de horas/semana
        - error_rate_trend: error_rate(últimas 2 semanas) - error_rate(primeiras 2 semanas)
        - session_frequency_trend: sessions/week(últimas 2) - sessions/week(primeiras 2)

        Args:
            sessions: Lista de sessões com campo end_at.
            weeks_in_period: Duração do período de observação em semanas.

        Returns:
            Dict com as 3 trend features (None se período < MIN_WEEKS_FOR_TRENDS).
        """
        # R3.7: Se período < 4 semanas, todas as trends são None
        if weeks_in_period < self.MIN_WEEKS_FOR_TRENDS:
            return {
                "viewing_time_trend": None,
                "error_rate_trend": None,
                "session_frequency_trend": None,
            }

        # Agrupar sessões por número da semana (semana 0, 1, 2, ...)
        weekly_data = self._group_sessions_by_week(sessions)

        if not weekly_data:
            return {
                "viewing_time_trend": 0.0,
                "error_rate_trend": 0.0,
                "session_frequency_trend": 0.0,
            }

        # Ordenar semanas cronologicamente
        sorted_weeks = sorted(weekly_data.keys())
        num_weeks = len(sorted_weeks)

        # viewing_time_trend: slope linear de horas de visualização por semana
        viewing_time_trend = self._calc_viewing_time_trend(
            weekly_data, sorted_weeks
        )

        # error_rate_trend: error_rate(últimas 2 sem) - error_rate(primeiras 2 sem)
        error_rate_trend = self._calc_error_rate_trend(
            weekly_data, sorted_weeks, num_weeks
        )

        # session_frequency_trend: freq(últimas 2 sem) - freq(primeiras 2 sem)
        session_frequency_trend = self._calc_session_frequency_trend(
            weekly_data, sorted_weeks, num_weeks
        )

        return {
            "viewing_time_trend": round(viewing_time_trend, 4),
            "error_rate_trend": round(error_rate_trend, 4),
            "session_frequency_trend": round(session_frequency_trend, 4),
        }

    def _group_sessions_by_week(
        self, sessions: list[dict]
    ) -> dict[int, list[dict]]:
        """Agrupa sessões por semana relativa (baseado em end_at).

        Semana 0 = primeira semana do período de observação.
        """
        # Encontrar o timestamp mais antigo para usar como referência
        timestamps: list[tuple[datetime, dict]] = []
        for session in sessions:
            end_at = session.get("end_at")
            if end_at:
                try:
                    ts = datetime.fromisoformat(
                        end_at.replace("Z", "+00:00")
                    )
                    timestamps.append((ts, session))
                except (ValueError, TypeError):
                    continue

        if not timestamps:
            return {}

        # Ordenar por timestamp
        timestamps.sort(key=lambda x: x[0])
        start_ts = timestamps[0][0]

        # Agrupar por semana relativa
        weekly: dict[int, list[dict]] = defaultdict(list)
        for ts, session in timestamps:
            delta_days = (ts - start_ts).total_seconds() / 86400.0
            week_num = int(delta_days / 7.0)
            weekly[week_num].append(session)

        return dict(weekly)

    def _calc_viewing_time_trend(
        self, weekly_data: dict[int, list[dict]], sorted_weeks: list[int]
    ) -> float:
        """Calcula slope da regressão linear do viewing time semanal (horas/semana).

        Usa np.polyfit grau 1 sobre as horas totais por semana.
        """
        week_indices: list[float] = []
        weekly_hours: list[float] = []

        for week_num in sorted_weeks:
            sessions_in_week = weekly_data[week_num]
            total_ms = 0.0
            for s in sessions_in_week:
                eff = s.get("effective_time")
                if eff is not None and eff >= 0:
                    total_ms += float(eff)
            hours = total_ms / 3_600_000.0
            week_indices.append(float(week_num))
            weekly_hours.append(hours)

        if len(week_indices) < 2:
            return 0.0

        # Regressão linear: y = slope*x + intercept
        coeffs = np.polyfit(week_indices, weekly_hours, 1)
        slope = float(coeffs[0])
        return slope

    def _calc_error_rate_trend(
        self,
        weekly_data: dict[int, list[dict]],
        sorted_weeks: list[int],
        num_weeks: int,
    ) -> float:
        """Calcula tendência da taxa de erro.

        error_rate(últimas 2 semanas) - error_rate(primeiras 2 semanas)
        """
        if num_weeks < 2:
            return 0.0

        # Primeiras 2 semanas
        first_weeks = sorted_weeks[:2]
        # Últimas 2 semanas
        last_weeks = sorted_weeks[-2:]

        first_error_rate = self._compute_error_rate_for_weeks(
            weekly_data, first_weeks
        )
        last_error_rate = self._compute_error_rate_for_weeks(
            weekly_data, last_weeks
        )

        return last_error_rate - first_error_rate

    def _calc_session_frequency_trend(
        self,
        weekly_data: dict[int, list[dict]],
        sorted_weeks: list[int],
        num_weeks: int,
    ) -> float:
        """Calcula tendência de frequência de sessões.

        sessions/week(últimas 2 semanas) - sessions/week(primeiras 2 semanas)
        """
        if num_weeks < 2:
            return 0.0

        # Primeiras 2 semanas
        first_weeks = sorted_weeks[:2]
        # Últimas 2 semanas
        last_weeks = sorted_weeks[-2:]

        first_freq = self._compute_sessions_per_week_for_weeks(
            weekly_data, first_weeks
        )
        last_freq = self._compute_sessions_per_week_for_weeks(
            weekly_data, last_weeks
        )

        return last_freq - first_freq

    def _compute_error_rate_for_weeks(
        self, weekly_data: dict[int, list[dict]], weeks: list[int]
    ) -> float:
        """Calcula error_rate para um conjunto de semanas."""
        total_sessions = 0
        error_count = 0

        for week_num in weeks:
            sessions_in_week = weekly_data.get(week_num, [])
            for session in sessions_in_week:
                total_sessions += 1
                error_code = session.get("error_code")
                if (
                    error_code is not None
                    and str(error_code).strip() != ""
                ):
                    error_count += 1

        if total_sessions == 0:
            return 0.0
        return error_count / total_sessions

    def _compute_sessions_per_week_for_weeks(
        self, weekly_data: dict[int, list[dict]], weeks: list[int]
    ) -> float:
        """Calcula média de sessões por semana para um conjunto de semanas."""
        if not weeks:
            return 0.0

        total_sessions = 0
        for week_num in weeks:
            total_sessions += len(weekly_data.get(week_num, []))

        return total_sessions / len(weeks)
