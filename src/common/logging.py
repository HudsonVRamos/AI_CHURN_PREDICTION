"""Módulo de logging estruturado em JSON compatível com AWS CloudWatch.

Fornece logging com campos padronizados (timestamp, execution_id, level, stage, message)
e suporte a propagação de contexto de execução entre estágios do pipeline.

Stages válidos do pipeline:
- extraction
- feature-engineering
- ml-inference
- explainability
- bedrock-explanation
- report-generation
- dashboard

Requirements: 8.1, 8.5, 17.3
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any


# Estágios válidos do pipeline (correspondem a log groups no CloudWatch)
PIPELINE_STAGES = [
    "extraction",
    "feature-engineering",
    "ml-inference",
    "explainability",
    "bedrock-explanation",
    "report-generation",
    "dashboard",
]

# Contexto de execução compartilhado via thread-local
_context = threading.local()


def set_execution_id(execution_id: str) -> None:
    """Define o execution_id para o contexto atual (propagado entre estágios).

    Args:
        execution_id: UUID que identifica a execução do pipeline.
    """
    _context.execution_id = execution_id


def get_execution_id() -> str:
    """Retorna o execution_id do contexto atual.

    Se nenhum execution_id foi definido, gera um novo UUID v4.

    Returns:
        String com o UUID da execução.
    """
    if not hasattr(_context, "execution_id") or _context.execution_id is None:
        _context.execution_id = str(uuid.uuid4())
    return _context.execution_id


def reset_execution_id() -> None:
    """Remove o execution_id do contexto (útil para testes)."""
    _context.execution_id = None


class JSONFormatter(logging.Formatter):
    """Formatter que produz logs em JSON compatível com CloudWatch.

    Campos produzidos:
    - timestamp: ISO 8601 com timezone UTC
    - execution_id: UUID da execução do pipeline
    - level: nível do log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - stage: estágio do pipeline que gerou o log
    - message: mensagem do log

    Campos adicionais são incluídos quando passados via extra dict.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Formata o LogRecord como JSON estruturado."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_id": get_execution_id(),
            "level": record.levelname,
            "stage": getattr(record, "stage", "unknown"),
            "message": record.getMessage(),
        }

        # Incluir campos extras (excluindo atributos internos do logging)
        _internal_attrs = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "filename", "module", "pathname", "levelno", "levelname",
            "msecs", "thread", "threadName", "process", "processName",
            "taskName", "stage", "message",
        }

        for key, value in record.__dict__.items():
            if key not in _internal_attrs and not key.startswith("_"):
                log_entry[key] = value

        # Incluir exceção se presente
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class StageLogger:
    """Logger para um estágio específico do pipeline.

    Wrapper sobre logging.Logger que injeta automaticamente o campo 'stage'
    em todas as mensagens de log.
    """

    def __init__(self, logger: logging.Logger, stage: str) -> None:
        self._logger = logger
        self._stage = stage

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Emite log com stage e campos extras."""
        extra = kwargs.pop("extra", {})
        extra["stage"] = self._stage
        self._logger.log(level, message, extra=extra, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log nível DEBUG."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log nível INFO."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log nível WARNING."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log nível ERROR."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log nível CRITICAL."""
        self._log(logging.CRITICAL, message, **kwargs)

    def log_stage_start(self) -> None:
        """Loga início de execução do estágio com timestamp."""
        self.info(f"Início do estágio: {self._stage}")

    def log_stage_completion(self, duration_seconds: float | None = None) -> None:
        """Loga conclusão do estágio com timestamp e duração opcional."""
        extra: dict[str, Any] = {}
        if duration_seconds is not None:
            extra["duration_seconds"] = round(duration_seconds, 3)
        self.info(
            f"Estágio concluído: {self._stage}",
            extra=extra,
        )

    @property
    def stage(self) -> str:
        """Retorna o nome do estágio."""
        return self._stage


def get_logger(stage: str, level: int = logging.INFO) -> StageLogger:
    """Obtém um logger estruturado para um estágio específico do pipeline.

    Args:
        stage: Nome do estágio do pipeline (ex: 'extraction', 'ml-inference').
        level: Nível mínimo de log (default: INFO).

    Returns:
        StageLogger configurado com JSON formatter para CloudWatch.
    """
    logger_name = f"churn_prediction.{stage}"
    logger = logging.getLogger(logger_name)

    # Evitar adicionar handlers duplicados
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    logger.setLevel(level)

    return StageLogger(logger=logger, stage=stage)
