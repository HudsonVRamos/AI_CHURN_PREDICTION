# Technical Design Document

## Overview

Plataforma de predição de churn para Sky Brazil. Extrai dados comportamentais via NPAW API, treina modelos supervisionados (XGBoost/LightGBM/CatBoost) no Amazon SageMaker, calcula explicabilidade via SHAP, gera explicações em linguagem natural via AWS Bedrock (Claude 3 Haiku), e apresenta resultados em dashboard interativo (Streamlit/ECS Fargate).

**Pipeline principal:**
1. Ingestão de IDs (CSV/JSON) → 2. Extração NPAW → 3. Feature Engineering → 4. Feature Store (DynamoDB) → 5. SageMaker Training/Batch Transform → 6. SHAP Explainability → 7. Bedrock NL Explanation → 8. Reports (S3) → 9. Dashboard

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CHURN PREDICTION PLATFORM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  INPUT   │──▶│   EXTRATOR   │──▶│   FEATURE    │──▶│  FEATURE STORE │  │
│  │ (CSV/JSON)│   │    NPAW      │   │  ENGINEER    │   │   (DynamoDB)   │  │
│  └──────────┘   └──────────────┘   └──────────────┘   └───────┬────────┘  │
│                                                                 │           │
│                                                                 ▼           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      AMAZON SAGEMAKER                                 │  │
│  │  ┌────────────┐   ┌─────────────────┐   ┌─────────────────────────┐ │  │
│  │  │  TRAINING  │   │ MODEL REGISTRY  │   │    BATCH TRANSFORM      │ │  │
│  │  │ (XGBoost/  │──▶│  (Versioning)   │──▶│  (Bulk Inference)       │ │  │
│  │  │  LightGBM) │   └─────────────────┘   └───────────┬─────────────┘ │  │
│  │  └────────────┘                                      │               │  │
│  └──────────────────────────────────────────────────────┼───────────────┘  │
│                                                          │                  │
│                                                          ▼                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────────┐   │
│  │    SHAP      │◀──│  PREDICTIONS │   │       AWS BEDROCK            │   │
│  │ (Explainer)  │   │   (Results)  │──▶│  (NL Explanation - PT-BR)    │   │
│  └──────┬───────┘   └──────┬───────┘   └──────────────┬───────────────┘   │
│         │                   │                          │                    │
│         ▼                   ▼                          ▼                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      REPORT GENERATOR                                 │  │
│  │              (JSON + Markdown) → S3 Bucket                            │  │
│  └──────────────────────────────────────────────────────┬───────────────┘  │
│                                                          │                  │
│                                                          ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         DASHBOARD                                     │  │
│  │              (Streamlit / React on ECS Fargate)                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MONITORING & LOGGING                                │  │
│  │         CloudWatch Logs + Metrics + Alarms                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## AWS Services Used

| Serviço | Função | Requisitos Cobertos |
|---------|--------|---------------------|
| Amazon SageMaker | Treino ML, Batch Transform, Model Registry | R10, R15 |
| Amazon S3 | Storage de dados, modelos, relatórios | R9, R10, R13 |
| Amazon DynamoDB | Feature Store, resultados de predição | R9, R11 |
| AWS Bedrock | Explicações em linguagem natural | R4, R12 |
| Amazon ECS Fargate | Dashboard web | R14 |
| AWS Secrets Manager | Credenciais (NPAW API key) | R7 |
| Amazon CloudWatch | Logs, métricas, alarmes | R8, R16, R17 |
| AWS Step Functions | Orquestração do pipeline | R2-R13 |
| Amazon EventBridge | Agendamento (cron semanal) | R2 |

## Components and Interfaces

### 1. Data Extractor (`src/extractors/npaw_extractor.py`)

```python
class NPAWExtractor:
    """Extrai sessões de usuários da API NPAW com retry e rate limiting."""
    
    BASE_URL = "https://api.npaw.com"
    BATCH_SIZE = 100
    MAX_SESSIONS_PER_USER = 5000
    MAX_CONCURRENT = 5
    
    def __init__(self, account_code: str, api_key: str, rate_limit_seconds: float = 1.0):
        ...
    
    async def extract_user_sessions(
        self, user_id: str, from_date: str, to_date: str | None = None
    ) -> list[dict]:
        """Extrai todas as sessões de um user com paginação."""
        ...
    
    async def extract_batch(
        self, user_ids: list[str], from_date: str, to_date: str | None = None
    ) -> dict[str, list[dict]]:
        """Extrai sessões para múltiplos users com concorrência limitada."""
        ...
```

**Relates to:** Requirement 2

### 2. Feature Engineer (`src/features/feature_engineer.py`)

```python
@dataclass
class FeatureVector:
    user_id: str
    version: int
    generated_at: str  # ISO 8601
    observation_start: str
    observation_end: str
    
    # Engagement
    total_sessions: int
    total_viewing_hours: float
    avg_session_duration_min: float
    sessions_per_week: float
    distinct_channels: int
    
    # Quality
    avg_happiness_score: float
    avg_buffer_ratio: float
    error_rate: float
    avg_bitrate: float
    
    # Behavioral
    pct_episode: float
    pct_sport: float
    pct_live: float
    pct_show: float
    distinct_devices: int
    avg_pause_count: float
    avg_seek_count: float
    
    # Trends (nullable)
    viewing_time_trend: float | None
    error_rate_trend: float | None
    session_frequency_trend: float | None


class FeatureEngineer:
    MIN_SESSIONS = 5
    MIN_WEEKS_FOR_TRENDS = 4
    
    def compute(self, user_id: str, sessions: list[dict]) -> FeatureVector | None:
        """Transforma sessões brutas em feature vector."""
        ...
```

**Relates to:** Requirement 3

### 3. Feature Store (`src/store/feature_store.py`)

```python
class FeatureStore:
    """Armazenamento versionado de Feature Vectors no DynamoDB."""
    
    TABLE_NAME = "churn_feature_store"
    
    # DynamoDB Schema:
    # PK: user_id (String)
    # SK: version (Number) - auto-increment per user
    # Attributes: generated_at, observation_start, observation_end, features (Map)
    
    def store(self, feature_vector: FeatureVector) -> int:
        """Armazena feature vector e retorna a versão atribuída."""
        ...
    
    def get_latest(self, user_id: str) -> FeatureVector | None:
        """Retorna a versão mais recente das features de um user."""
        ...
    
    def get_version(self, user_id: str, version: int) -> FeatureVector | None:
        """Retorna uma versão específica."""
        ...
    
    def get_history(self, user_id: str, from_date: str = None) -> list[FeatureVector]:
        """Retorna todas as versões de um user (com filtro de data opcional)."""
        ...
```

**Relates to:** Requirement 9

### 4. ML Pipeline (`src/ml/sagemaker_pipeline.py`)

```python
class SageMakerMLPipeline:
    """Pipeline de treino e inferência no Amazon SageMaker."""
    
    SUPPORTED_ALGORITHMS = ["xgboost", "lightgbm", "catboost"]
    
    def train(
        self,
        training_data_s3: str,  # S3 path do dataset (features + labels)
        algorithm: str = "xgboost",
        hyperparameters: dict | None = None,
    ) -> ModelVersion:
        """Treina modelo no SageMaker e registra no Model Registry."""
        # 1. Upload training data to S3
        # 2. Create SageMaker Training Job
        # 3. Evaluate on test set (Precision, Recall, F1, ROC AUC)
        # 4. Register in Model Registry
        ...
    
    def predict_batch(
        self,
        feature_vectors_s3: str,  # S3 path dos features para inferência
        model_version: str | None = None,  # None = usar modelo aprovado
    ) -> str:
        """Executa Batch Transform e retorna S3 path dos resultados."""
        ...
    
    def get_active_model(self) -> ModelVersion:
        """Retorna o modelo atualmente aprovado no Model Registry."""
        ...


@dataclass
class ModelVersion:
    model_package_arn: str
    algorithm: str
    training_date: str
    dataset_version: str
    metrics: dict  # precision, recall, f1, roc_auc
```

**Relates to:** Requirements 10, 15

### 5. Explainability Engine (`src/explainability/shap_explainer.py`)

```python
class SHAPExplainer:
    """Calcula importância de features usando SHAP."""
    
    TOP_FEATURES_DEFAULT = 10
    
    def __init__(self, model, training_data: pd.DataFrame):
        self.explainer = shap.TreeExplainer(model)
        self.training_data = training_data
    
    def explain(self, feature_vector: FeatureVector) -> ExplainabilityResult:
        """Calcula SHAP values para uma predição individual."""
        ...
    
    def explain_batch(self, feature_vectors: list[FeatureVector]) -> list[ExplainabilityResult]:
        """SHAP values para um batch de predições."""
        ...


@dataclass
class FeatureContribution:
    feature_name: str
    contribution_weight: float  # signed: positive pushes toward churn
    normalized_impact: float    # -1.0 to 1.0


@dataclass
class ExplainabilityResult:
    user_id: str
    top_features: list[FeatureContribution]
    base_value: float
    prediction_value: float
```

**Relates to:** Requirement 11

### 6. Bedrock Explanation Generator (`src/explanations/bedrock_explainer.py`)

```python
class BedrockExplainer:
    """Gera explicações em linguagem natural via AWS Bedrock."""
    
    MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"  # Custo-benefício
    TIMEOUT_SECONDS = 60
    MAX_RETRIES = 2
    
    def generate_explanation(
        self,
        user_id: str,
        churn_probability: float,
        confidence: float,
        top_features: list[FeatureContribution],
        user_feature_values: dict,
        population_stats: dict,  # mean, median, std per feature
    ) -> str | None:
        """Gera explicação em PT-BR. Retorna None se Bedrock indisponível."""
        ...
    
    def _build_prompt(self, ...) -> str:
        """Monta o prompt estruturado para o Bedrock."""
        # Instrui o modelo a:
        # 1. Explicar em português brasileiro
        # 2. Usar apenas os dados fornecidos
        # 3. NÃO calcular probabilidade (já fornecida)
        # 4. Comparar o assinante com a população
        # 5. Ser objetivo e factual
        ...
```

**Relates to:** Requirement 12

### 7. Report Generator (`src/reports/report_generator.py`)

```python
class ReportGenerator:
    """Gera relatórios em JSON e Markdown."""
    
    def generate_individual_report(
        self,
        user_id: str,
        prediction: PredictionResult,
        explainability: ExplainabilityResult,
        explanation: str | None,
    ) -> dict:
        ...
    
    def generate_executive_report(
        self,
        predictions: list[PredictionResult],
        explainabilities: list[ExplainabilityResult],
    ) -> dict:
        ...
    
    def export_json(self, report: dict, output_path: str) -> None:
        ...
    
    def export_markdown(self, report: dict, output_path: str) -> None:
        ...
```

**Relates to:** Requirements 6, 13

### 8. Dashboard (`src/dashboard/app.py`)

**Tech Stack:** Streamlit (Python) deployed on ECS Fargate

```python
# Streamlit app structure
# Pages:
#   1. Overview (KPIs, risk distribution, top factors)
#   2. Subscriber Detail (search by user_id, show all metrics)
#   3. History (temporal evolution, drift indicators)
#   4. Export (filtered data download)

class DashboardDataSource:
    """Reads prediction results from S3/DynamoDB for dashboard display."""
    
    def get_latest_execution(self) -> ExecutionSummary:
        ...
    
    def get_predictions(self, risk_level: str = None, period: str = None) -> list[dict]:
        ...
    
    def get_subscriber_detail(self, user_id: str) -> dict:
        ...
    
    def get_history(self, user_id: str) -> list[dict]:
        ...
```

**Relates to:** Requirement 14

### 9. Pipeline Orchestrator (`src/orchestrator/step_functions.py`)

```python
# AWS Step Functions state machine definition
PIPELINE_DEFINITION = {
    "StartAt": "Ingestion",
    "States": {
        "Ingestion": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:ingest-user-lists",
            "Next": "Extraction"
        },
        "Extraction": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:extract-npaw-data",
            "Next": "FeatureEngineering"
        },
        "FeatureEngineering": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:compute-features",
            "Next": "StoreFeatures"
        },
        "StoreFeatures": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:store-features",
            "Next": "ChooseMode"
        },
        "ChooseMode": {
            "Type": "Choice",
            "Choices": [
                {"Variable": "$.mode", "StringEquals": "train", "Next": "Training"},
                {"Variable": "$.mode", "StringEquals": "predict", "Next": "BatchPredict"}
            ]
        },
        "Training": {
            "Type": "Task",
            "Resource": "arn:aws:sagemaker:...:training-job",
            "Next": "EvaluateModel"
        },
        "EvaluateModel": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:evaluate-model",
            "Next": "RegisterModel"
        },
        "RegisterModel": {
            "Type": "Task",
            "Resource": "arn:aws:sagemaker:...:model-registry",
            "End": True
        },
        "BatchPredict": {
            "Type": "Task",
            "Resource": "arn:aws:sagemaker:...:batch-transform",
            "Next": "Explainability"
        },
        "Explainability": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:compute-shap",
            "Next": "BedrockExplanations"
        },
        "BedrockExplanations": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:generate-explanations",
            "Next": "GenerateReports"
        },
        "GenerateReports": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:generate-reports",
            "End": True
        }
    }
}
```

**Relates to:** Requirements 8, 16, 17

## Data Models

### DynamoDB Tables

**Table: `churn_feature_store`**
```
PK: user_id (String)
SK: version (Number)
Attributes:
  - generated_at (String, ISO 8601)
  - observation_start (String)
  - observation_end (String)
  - features (Map) - all feature values
  - label (Number, optional) - 0=active, 1=churned (for training)
```

**Table: `churn_predictions`**
```
PK: execution_id (String)
SK: user_id (String)
Attributes:
  - churn_probability (Number)
  - confidence (Number)
  - risk_tier (String: Low|Medium|High)
  - model_version (String)
  - feature_version (Number)
  - timestamp (String, ISO 8601)
  - shap_results (Map)
  - bedrock_explanation (String, nullable)
  - explanation_status (String: available|unavailable|pending)
```

**Table: `churn_executions`**
```
PK: execution_id (String)
Attributes:
  - start_time (String)
  - end_time (String)
  - mode (String: train|predict)
  - model_version (String)
  - users_processed (Number)
  - users_failed (Number)
  - status (String: running|completed|failed)
  - output_s3_path (String)
```

### S3 Bucket Structure

```
s3://sky-brazil-churn-prediction/
├── input/
│   ├── churned_users/         # CSVs com IDs de churned
│   └── active_users/          # CSVs com IDs de ativos
├── raw_data/
│   └── {execution_id}/
│       └── {user_id}.json     # Sessões brutas da NPAW
├── features/
│   └── {execution_id}/
│       ├── training_data.csv  # Features + labels para treino
│       └── inference_data.csv # Features para inferência
├── models/
│   └── {model_version}/
│       ├── model.tar.gz       # Artefato do modelo
│       ├── hyperparameters.json
│       └── metrics.json
├── predictions/
│   └── {execution_id}/
│       ├── results.json       # Predições + SHAP
│       └── explanations.json  # Explicações Bedrock
├── reports/
│   └── {execution_id}/
│       ├── executive_report.json
│       ├── executive_report.md
│       ├── high_risk_users.json
│       └── high_risk_users.md
└── monitoring/
    └── drift_reports/
```

## Configuration

### `config/settings.yaml`
```yaml
npaw:
  account_code: "sky_brazil"
  base_url: "https://api.npaw.com"
  rate_limit_seconds: 1.0
  max_concurrent_requests: 5
  batch_size: 100
  max_sessions_per_user: 5000

observation:
  time_window_months: 6
  min_sessions: 5
  min_weeks_for_trends: 4

sagemaker:
  region: "us-east-1"
  algorithm: "xgboost"  # xgboost | lightgbm | catboost
  training_instance: "ml.m5.xlarge"
  batch_instance: "ml.m5.large"
  model_package_group: "churn-prediction-models"

bedrock:
  region: "us-east-1"
  model_id: "anthropic.claude-3-haiku-20240307-v1:0"
  timeout_seconds: 60
  max_retries: 2
  language: "pt-BR"

explainability:
  top_features: 10
  method: "shap"  # shap | lime

prediction:
  risk_thresholds:
    low_max: 30
    medium_max: 60
  batch_size: 1000

monitoring:
  drift_threshold_std: 2.0
  max_inference_time_ms: 5000
  max_failure_rate_pct: 5.0

dashboard:
  port: 8501
  refresh_on_new_execution: true

reports:
  formats: ["json", "markdown"]
  output_bucket: "sky-brazil-churn-prediction"
  output_prefix: "reports"
```

## Security

| Aspecto | Implementação |
|---------|--------------|
| NPAW API Key | AWS Secrets Manager (`churn-prediction/npaw-api-key`) |
| AWS Access | IAM Roles (SageMaker Execution Role, Lambda Execution Role) |
| Data at rest | S3 SSE-S3, DynamoDB encryption enabled |
| Data in transit | HTTPS para NPAW API, VPC endpoints para serviços AWS |
| Dashboard access | ALB + Cognito authentication |
| Least privilege | Cada Lambda/serviço com IAM role mínima |

## Monitoring & Alerting

### CloudWatch Metrics (Custom Namespace: `ChurnPrediction`)

| Metric | Unit | Alarm Threshold |
|--------|------|-----------------|
| `InferenceTime` | Milliseconds | > 5000ms |
| `PredictionCount` | Count | - |
| `PredictionFailures` | Count | > 5% of batch |
| `FeatureDriftDetected` | Count | > 0 |
| `BedrockTimeout` | Count | > 10% of requests |
| `ExtractionErrors` | Count | > 20% of users |

### CloudWatch Log Groups

```
/churn-prediction/extraction
/churn-prediction/feature-engineering
/churn-prediction/ml-inference
/churn-prediction/explainability
/churn-prediction/bedrock-explanation
/churn-prediction/report-generation
/churn-prediction/dashboard
```

## Technology Stack Summary

| Layer | Technology | Justification |
|-------|------------|---------------|
| Language | Python 3.11+ | Ecosystem ML (scikit-learn, SHAP, boto3) |
| ML Framework | XGBoost/LightGBM (SageMaker built-in) | Performant, deterministic, tree-based (ideal for tabular data) |
| Explainability | SHAP (TreeExplainer) | Gold standard for tree models, fast |
| NLP Explanation | AWS Bedrock (Claude 3 Haiku) | Low cost, Portuguese, no infra to manage |
| Orchestration | AWS Step Functions | Visual workflow, retries built-in, serverless |
| Compute | AWS Lambda + SageMaker | Serverless for pipeline, managed for ML |
| Storage | S3 + DynamoDB | S3 for bulk, DynamoDB for fast lookups |
| Dashboard | Streamlit on ECS Fargate | Python native, fast to build, low cost |
| Monitoring | CloudWatch | Native AWS, no extra tooling |
| IaC | AWS CDK (Python) | Same language as app, type-safe |


## Correctness Properties

### Property 1: Deterministic Inference
Para o mesmo Feature_Vector e modelo, a predição deve ser idêntica (seed fixo no modelo).
**Validates: Requirements 10.5**

### Property 2: Immutable Storage
Feature Vectors e predições armazenadas nunca são sobrescritos; novas versões são sempre criadas.
**Validates: Requirements 9.5**

### Property 3: Separation of Concerns
Bedrock NUNCA calcula score de churn; o ML model NUNCA gera texto explicativo.
**Validates: Requirements 10.6, 12.4, 12.5**

### Property 4: Version Traceability
Cada predição registra exatamente qual versão de modelo e features foi usada.
**Validates: Requirements 15.4, 17.1**

### Property 5: Graceful Degradation
Se Bedrock falhar, a predição permanece válida com explanation = null.
**Validates: Requirements 12.6**

### Property 6: Data Integrity
Feature Vectors são persistidos ANTES da inferência; resultados são persistidos ANTES da geração de relatório.
**Validates: Requirements 9.1, 17.4**

## Error Handling

| Cenário | Comportamento | Retry |
|---------|--------------|-------|
| NPAW Auth Error (401/403) | Aborta todo o pipeline | Não |
| NPAW Server Error (5xx) | Retry com backoff para aquele user | 3x (2s, 4s, 8s) |
| NPAW Timeout (>60s) | Retry com backoff | 3x |
| NPAW sem dados para user | Skip user, log warning | Não |
| SageMaker Training falha | Aborta treino, notifica | Não |
| SageMaker Batch Transform falha | Retry entire batch | 1x |
| SHAP falha para um user | Prediction válida, SHAP = null | Não |
| Bedrock Timeout (>60s) | Retry | 2x (5s intervalo) |
| Bedrock Error | Prediction válida, explanation = null | 2x |
| DynamoDB throttle | Retry com backoff | 5x |
| Report generation falha | Dados preservados em S3, log CRITICAL | Não |

## Testing Strategy

| Tipo | Ferramenta | Cobertura |
|------|------------|-----------|
| Unit Tests | pytest | Feature Engineer, validações, transformações |
| Integration Tests | pytest + moto | NPAW mock, DynamoDB local, S3 local |
| ML Model Tests | pytest | Determinismo, métricas mínimas (ROC AUC > 0.7) |
| Contract Tests | pytest | Schema do Feature_Vector, schema dos relatórios |
| End-to-End | Step Functions local | Pipeline completo com dados sintéticos |
| Load Tests | locust (opcional) | Dashboard sob carga, API extraction throttle |

**Dados de teste:**
- Fixture com 10 churned + 10 active users (sessões sintéticas)
- Modelo pré-treinado para testes de inferência
- Mock do Bedrock com respostas fixas para testes determinísticos
