"""Módulo de configuração e validação da plataforma de churn prediction.

Responsável por:
- Carregar configurações do arquivo config/settings.yaml
- Permitir overrides via variáveis de ambiente (prefixo CHURN_)
- Validar valores obrigatórios e ranges
- Falhar imediatamente (fail-fast) com mensagens descritivas

Variáveis de ambiente suportadas (sobrescrevem valores do YAML):
- CHURN_NPAW_API_KEY: chave de API da NPAW (obrigatória)
- CHURN_NPAW_ACCOUNT_CODE: código da conta NPAW
- CHURN_NPAW_BASE_URL: URL base da API NPAW
- CHURN_NPAW_RATE_LIMIT_SECONDS: intervalo entre chamadas NPAW
- CHURN_NPAW_MAX_CONCURRENT_REQUESTS: máximo de requests simultâneos
- CHURN_NPAW_BATCH_SIZE: tamanho do batch de extração
- CHURN_NPAW_MAX_SESSIONS_PER_USER: máximo de sessões por usuário
- CHURN_OBSERVATION_TIME_WINDOW_MONTHS: janela de observação (1-24)
- CHURN_OBSERVATION_MIN_SESSIONS: mínimo de sessões para análise (1-10000)
- CHURN_OBSERVATION_MIN_WEEKS_FOR_TRENDS: semanas mínimas para trends
- CHURN_BEDROCK_REGION: região AWS do Bedrock
- CHURN_BEDROCK_MODEL_ID: ID do modelo Bedrock
- CHURN_BEDROCK_TIMEOUT_SECONDS: timeout do Bedrock em segundos
- CHURN_BEDROCK_MAX_RETRIES: máximo de retries do Bedrock
- CHURN_SAGEMAKER_REGION: região AWS do SageMaker
- CHURN_SAGEMAKER_ALGORITHM: algoritmo de ML (xgboost|lightgbm|catboost)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class ConfigurationError(Exception):
    """Erro de configuração que impede a inicialização do sistema.

    Contém informações sobre o parâmetro inválido ou ausente,
    sua fonte esperada, e o valor/range aceitável quando aplicável.
    """

    pass


# ---------------------------------------------------------------------------
# Sub-modelos de configuração
# ---------------------------------------------------------------------------


class NPAWConfig(BaseModel):
    """Configuração do conector NPAW."""

    account_code: str = Field(min_length=1, description="Código da conta NPAW")
    api_key: str = Field(min_length=1, description="API key da NPAW")
    base_url: str = Field(
        default="https://api.npaw.com", min_length=1, description="URL base da API NPAW"
    )
    rate_limit_seconds: float = Field(default=1.0, ge=0.0, description="Intervalo entre chamadas")
    max_concurrent_requests: int = Field(
        default=5, ge=1, le=50, description="Máximo de requests simultâneos"
    )
    batch_size: int = Field(default=100, ge=1, le=1000, description="Tamanho do batch")
    max_sessions_per_user: int = Field(
        default=5000, ge=1, le=100000, description="Máximo de sessões por usuário"
    )


class ObservationConfig(BaseModel):
    """Configuração da janela de observação."""

    time_window_months: int = Field(
        default=6, ge=1, le=24, description="Janela de observação em meses (1-24)"
    )
    min_sessions: int = Field(
        default=5, ge=1, le=10000, description="Mínimo de sessões para análise válida (1-10000)"
    )
    min_weeks_for_trends: int = Field(
        default=4, ge=1, le=52, description="Semanas mínimas para cálculo de trends"
    )


class SageMakerConfig(BaseModel):
    """Configuração do Amazon SageMaker."""

    region: str = Field(min_length=1, description="Região AWS do SageMaker")
    algorithm: str = Field(default="xgboost", description="Algoritmo de ML")
    training_instance: str = Field(default="ml.m5.xlarge", description="Tipo de instância treino")
    batch_instance: str = Field(default="ml.m5.large", description="Tipo de instância batch")
    model_package_group: str = Field(
        default="churn-prediction-models", description="Grupo de pacotes de modelo"
    )

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        allowed = ("xgboost", "lightgbm", "catboost")
        if v.lower() not in allowed:
            raise ValueError(
                f"Algoritmo '{v}' inválido. Valores aceitos: {', '.join(allowed)}"
            )
        return v.lower()


class BedrockConfig(BaseModel):
    """Configuração do AWS Bedrock."""

    region: str = Field(min_length=1, description="Região AWS do Bedrock")
    model_id: str = Field(min_length=1, description="ID do modelo Bedrock")
    timeout_seconds: int = Field(default=60, ge=1, le=300, description="Timeout em segundos")
    max_retries: int = Field(default=2, ge=0, le=10, description="Máximo de retries")
    language: str = Field(default="pt-BR", description="Idioma das explicações")


class ExplainabilityConfig(BaseModel):
    """Configuração de explicabilidade."""

    top_features: int = Field(default=10, ge=1, le=100, description="Top features a exibir")
    method: str = Field(default="shap", description="Método de explicabilidade")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = ("shap", "lime")
        if v.lower() not in allowed:
            raise ValueError(f"Método '{v}' inválido. Valores aceitos: {', '.join(allowed)}")
        return v.lower()


class RiskThresholds(BaseModel):
    """Thresholds para classificação de risco."""

    low_max: int = Field(default=30, ge=0, le=100, description="Limite superior do risco baixo")
    medium_max: int = Field(
        default=60, ge=0, le=100, description="Limite superior do risco médio"
    )

    @model_validator(mode="after")
    def validate_thresholds_order(self) -> "RiskThresholds":
        if self.low_max >= self.medium_max:
            raise ValueError(
                f"low_max ({self.low_max}) deve ser menor que medium_max ({self.medium_max})"
            )
        return self


class PredictionConfig(BaseModel):
    """Configuração de predição."""

    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)
    batch_size: int = Field(default=1000, ge=1, le=100000, description="Tamanho do batch")


class MonitoringConfig(BaseModel):
    """Configuração de monitoramento."""

    drift_threshold_std: float = Field(
        default=2.0, ge=0.1, le=10.0, description="Threshold de drift em desvios-padrão"
    )
    max_inference_time_ms: int = Field(
        default=5000, ge=100, le=60000, description="Tempo máximo de inferência (ms)"
    )
    max_failure_rate_pct: float = Field(
        default=5.0, ge=0.0, le=100.0, description="Taxa máxima de falha (%)"
    )


class DashboardConfig(BaseModel):
    """Configuração do dashboard."""

    port: int = Field(default=8501, ge=1, le=65535, description="Porta do dashboard")
    refresh_on_new_execution: bool = Field(
        default=True, description="Atualizar ao detectar nova execução"
    )


class ReportsConfig(BaseModel):
    """Configuração de relatórios."""

    formats: list[str] = Field(default=["json", "markdown"], description="Formatos de saída")
    output_bucket: str = Field(min_length=1, description="Bucket S3 de saída")
    output_prefix: str = Field(default="reports", description="Prefixo S3")


# ---------------------------------------------------------------------------
# Modelo principal de Settings
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    """Configuração central da plataforma de churn prediction.

    Carrega valores de config/settings.yaml e permite overrides via
    variáveis de ambiente com prefixo CHURN_.

    Uso:
        settings = Settings.load()
        # ou via helper cacheado:
        settings = get_settings()
    """

    npaw: NPAWConfig
    observation: ObservationConfig
    sagemaker: SageMakerConfig
    bedrock: BedrockConfig
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    reports: ReportsConfig

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        """Carrega configurações do YAML e variáveis de ambiente.

        A ordem de prioridade é:
        1. Variáveis de ambiente (maior prioridade)
        2. Arquivo YAML (menor prioridade)

        Args:
            config_path: Caminho para o arquivo YAML. Se None, usa
                        config/settings.yaml relativo à raiz do projeto.

        Returns:
            Instância validada de Settings.

        Raises:
            ConfigurationError: Se valor obrigatório estiver ausente ou inválido.
        """
        if config_path is None:
            config_path = _find_config_file()

        config_path = Path(config_path)

        # Carregar YAML
        yaml_data = _load_yaml(config_path)

        # Aplicar overrides de variáveis de ambiente
        merged_data = _apply_env_overrides(yaml_data)

        # Validar e criar instância
        try:
            return cls.model_validate(merged_data)
        except ValidationError as e:
            raise _convert_validation_error(e, config_path)

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Carrega configurações a partir de um dicionário (útil para testes).

        Args:
            data: Dicionário com a configuração completa.

        Returns:
            Instância validada de Settings.

        Raises:
            ConfigurationError: Se valor obrigatório estiver ausente ou inválido.
        """
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise _convert_validation_error(e)


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

# Mapeamento: variável de ambiente -> caminho no dicionário de config
ENV_MAPPING: dict[str, tuple[str, ...]] = {
    "CHURN_NPAW_API_KEY": ("npaw", "api_key"),
    "CHURN_NPAW_ACCOUNT_CODE": ("npaw", "account_code"),
    "CHURN_NPAW_BASE_URL": ("npaw", "base_url"),
    "CHURN_NPAW_RATE_LIMIT_SECONDS": ("npaw", "rate_limit_seconds"),
    "CHURN_NPAW_MAX_CONCURRENT_REQUESTS": ("npaw", "max_concurrent_requests"),
    "CHURN_NPAW_BATCH_SIZE": ("npaw", "batch_size"),
    "CHURN_NPAW_MAX_SESSIONS_PER_USER": ("npaw", "max_sessions_per_user"),
    "CHURN_OBSERVATION_TIME_WINDOW_MONTHS": ("observation", "time_window_months"),
    "CHURN_OBSERVATION_MIN_SESSIONS": ("observation", "min_sessions"),
    "CHURN_OBSERVATION_MIN_WEEKS_FOR_TRENDS": ("observation", "min_weeks_for_trends"),
    "CHURN_BEDROCK_REGION": ("bedrock", "region"),
    "CHURN_BEDROCK_MODEL_ID": ("bedrock", "model_id"),
    "CHURN_BEDROCK_TIMEOUT_SECONDS": ("bedrock", "timeout_seconds"),
    "CHURN_BEDROCK_MAX_RETRIES": ("bedrock", "max_retries"),
    "CHURN_SAGEMAKER_REGION": ("sagemaker", "region"),
    "CHURN_SAGEMAKER_ALGORITHM": ("sagemaker", "algorithm"),
}


def _find_config_file() -> Path:
    """Localiza o arquivo de configuração settings.yaml.

    Busca em:
    1. config/settings.yaml relativo ao diretório de trabalho atual
    2. config/settings.yaml relativo à raiz do projeto (2 níveis acima de src/)
    """
    # Opção 1: relativo ao CWD
    cwd_path = Path.cwd() / "config" / "settings.yaml"
    if cwd_path.exists():
        return cwd_path

    # Opção 2: relativo ao módulo (raiz do projeto)
    module_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
    if module_path.exists():
        return module_path

    raise ConfigurationError(
        "Arquivo de configuração não encontrado. "
        "Esperado em: config/settings.yaml (relativo ao diretório de trabalho) "
        f"ou {module_path}"
    )


def _load_yaml(config_path: Path) -> dict[str, Any]:
    """Carrega e parseia o arquivo YAML."""
    if not config_path.exists():
        raise ConfigurationError(
            f"Arquivo de configuração não encontrado: {config_path}. "
            "Verifique se o arquivo config/settings.yaml existe."
        )

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Erro ao parsear arquivo de configuração {config_path}: {e}"
        )

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Arquivo de configuração {config_path} deve conter um mapeamento YAML "
            f"no nível raiz. Tipo encontrado: {type(data).__name__}"
        )

    return data


def _apply_env_overrides(yaml_data: dict[str, Any]) -> dict[str, Any]:
    """Aplica overrides de variáveis de ambiente sobre os dados do YAML.

    Variáveis de ambiente com prefixo CHURN_ sobrescrevem valores
    correspondentes do arquivo YAML.
    """
    data = _deep_copy_dict(yaml_data)

    for env_var, path in ENV_MAPPING.items():
        value = os.environ.get(env_var)
        if value is not None:
            _set_nested(data, path, value)

    return data


def _deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Cópia profunda de dicionário (sem import copy para simplicidade)."""
    result: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _deep_copy_dict(value)
        elif isinstance(value, list):
            result[key] = value.copy()
        else:
            result[key] = value
    return result


def _set_nested(d: dict[str, Any], path: tuple[str, ...], value: str) -> None:
    """Define um valor em um dicionário aninhado criando níveis intermediários."""
    for key in path[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[path[-1]] = value


def _convert_validation_error(
    error: ValidationError, config_path: Path | None = None
) -> ConfigurationError:
    """Converte erros de validação Pydantic em mensagens claras para o usuário.

    Formata cada erro incluindo:
    - Nome do parâmetro (ex: npaw.api_key)
    - Fonte esperada (variável de ambiente ou arquivo YAML)
    - Valor inválido e range aceitável quando aplicável
    """
    messages: list[str] = []

    for err in error.errors():
        # Montar o caminho do parâmetro (ex: npaw.api_key)
        param_path = ".".join(str(loc) for loc in err["loc"])
        err_type = err["type"]
        err_msg = err["msg"]

        # Determinar a fonte esperada (env var correspondente ou YAML)
        env_var = _find_env_var_for_path(err["loc"])
        if env_var:
            source = f"variável de ambiente '{env_var}' ou arquivo de configuração"
        elif config_path:
            source = f"arquivo de configuração '{config_path}'"
        else:
            source = "arquivo de configuração ou variável de ambiente"

        # Formatar mensagem baseada no tipo de erro
        if err_type == "missing":
            messages.append(
                f"Parâmetro obrigatório ausente: '{param_path}'. "
                f"Fonte esperada: {source}."
            )
        elif err_type in ("value_error", "string_too_short"):
            messages.append(
                f"Parâmetro '{param_path}' com valor inválido. "
                f"{err_msg}. Fonte: {source}."
            )
        elif "greater_than_equal" in err_type or "less_than_equal" in err_type:
            input_val = err.get("input", "N/A")
            messages.append(
                f"Parâmetro '{param_path}' com valor fora do range permitido: "
                f"valor={input_val}. {err_msg}. Fonte: {source}."
            )
        else:
            input_val = err.get("input", "")
            val_info = f" (valor fornecido: '{input_val}')" if input_val != "" else ""
            messages.append(
                f"Parâmetro '{param_path}' inválido{val_info}. "
                f"{err_msg}. Fonte: {source}."
            )

    full_message = "Erro de configuração - falha na inicialização:\n" + "\n".join(
        f"  - {m}" for m in messages
    )
    return ConfigurationError(full_message)


def _find_env_var_for_path(loc: tuple[str | int, ...]) -> str | None:
    """Encontra a variável de ambiente correspondente a um caminho de configuração."""
    # Converter loc para tupla de strings (sem índices numéricos)
    str_path = tuple(str(part) for part in loc if isinstance(part, str))
    for env_var, env_path in ENV_MAPPING.items():
        if env_path == str_path:
            return env_var
    return None


# ---------------------------------------------------------------------------
# Singleton / cache
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    """Retorna a instância de Settings cacheada (singleton).

    Na primeira chamada, carrega e valida as configurações.
    Chamadas subsequentes retornam a mesma instância.

    Args:
        config_path: Caminho opcional para o arquivo YAML.

    Returns:
        Instância validada de Settings.

    Raises:
        ConfigurationError: Se a configuração estiver inválida ou incompleta.
    """
    return Settings.load(config_path)


def reset_settings_cache() -> None:
    """Limpa o cache de settings (útil para testes)."""
    get_settings.cache_clear()
