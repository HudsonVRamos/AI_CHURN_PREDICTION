# Implementation Plan: AI Churn Prediction Platform

## Overview

Plataforma de predição de churn para Sky Brazil utilizando Python 3.11+, Amazon SageMaker, AWS Bedrock (Claude 3 Haiku), DynamoDB, S3, Step Functions, EventBridge e Streamlit. A implementação segue a pipeline: Ingestão → Extração NPAW → Feature Engineering → Feature Store → SageMaker Train/Predict → SHAP → Bedrock Explanation → Reports → Dashboard. Toda infraestrutura via AWS CDK (Python).

## Tasks

- [x] 1. Estrutura do projeto, configuração e interfaces base
  - [x] 1.1 Criar estrutura de diretórios e arquivos de configuração do projeto
    - Criar diretórios: `src/extractors/`, `src/features/`, `src/store/`, `src/ml/`, `src/explainability/`, `src/explanations/`, `src/reports/`, `src/dashboard/`, `src/orchestrator/`, `src/common/`, `config/`, `tests/`, `infra/`
    - Criar `pyproject.toml` com dependências (boto3, pandas, numpy, shap, xgboost, lightgbm, streamlit, pydantic, pyyaml, pytest, moto)
    - Criar `config/settings.yaml` com todas as configurações conforme o design
    - Criar `src/common/__init__.py`, `src/__init__.py`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 1.2 Implementar módulo de configuração e validação (`src/common/config.py`)
    - Classe `Settings` que lê `config/settings.yaml` e variáveis de ambiente
    - Validação de valores obrigatórios com mensagens claras (nome do parâmetro + fonte esperada)
    - Environment variables sobrescrevem valores do YAML
    - Falha imediata (fail-fast) com mensagem descritiva se valor ausente ou inválido
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 1.3 Implementar módulo de logging estruturado (`src/common/logging.py`)
    - Logger JSON compatível com CloudWatch (campos: timestamp, execution_id, level, stage, message)
    - Suporte a contexto de execução (execution_id propagado)
    - Níveis: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - _Requirements: 8.1, 8.5, 17.3_

  - [x] 1.4 Definir modelos de dados e interfaces (`src/common/models.py`)
    - Dataclasses/Pydantic models para: `FeatureVector`, `PredictionResult`, `ExplainabilityResult`, `FeatureContribution`, `ModelVersion`, `ExecutionSummary`, `ChurnPattern`
    - Validação de tipos e ranges nos modelos
    - _Requirements: 3.5, 10.4, 11.3, 13.1_

  - [x] 1.5 Escrever testes unitários para módulo de configuração
    - Testar carregamento de YAML, override por env vars, validação de valores inválidos, mensagens de erro
    - _Requirements: 7.5, 7.6_

- [x] 2. Ingestão de listas de usuários
  - [x] 2.1 Implementar módulo de ingestão (`src/extractors/ingestion.py`)
    - Aceitar CSV (coluna "user_id"), JSON (chave "user_ids"), ou array direto
    - Validar UUID v4 para cada ID
    - Deduplicar IDs
    - Rejeitar IDs inválidos com log warning e continuar com os válidos
    - Retornar erro se todos os IDs forem inválidos ou lista vazia
    - Limitar entre 1 e 50.000 IDs válidos
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 2.2 Escrever property test para validação de UUIDs
    - **Property 2: Immutable Storage** — aplicar ao contexto de ingestão: IDs válidos nunca são perdidos na deduplicação
    - **Validates: Requirements 1.1, 1.2, 1.6**

  - [x] 2.3 Escrever testes unitários para ingestão
    - Testar CSV válido/inválido, JSON válido/inválido, IDs duplicados, lista vazia, UUID inválido
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 3. Extração de dados da NPAW
  - [x] 3.1 Implementar cliente NPAW com retry e rate limiting (`src/extractors/npaw_extractor.py`)
    - Classe `NPAWExtractor` com autenticação via API key (Secrets Manager ou env var)
    - Query ao endpoint `GET /sky_brazil/rawdata` com filtros por user_id
    - Paginação com offset (batch de 100, máximo 5000 sessões por user)
    - Rate limiting configurável (default 1s entre chamadas)
    - Concorrência limitada (default 5 users em paralelo) com asyncio/semáforo
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.8_

  - [x] 3.2 Implementar tratamento de erros e retry para NPAW
    - Auth error (401/403): abortar todo o pipeline
    - Server error (5xx) / timeout (>60s): retry 3x com backoff exponencial (2s, 4s, 8s)
    - Sem dados para user: log warning, skip user
    - Logging de progresso a cada 50 users ou 60s
    - _Requirements: 2.5, 2.6, 2.7, 8.2, 8.3, 8.4_

  - [x] 3.3 Escrever testes de integração para NPAW extractor (com mocks)
    - Mock de respostas HTTP (sucesso, 401, 5xx, timeout, dados vazios)
    - Verificar paginação, retry, rate limiting
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6, 2.7_

- [x] 4. Checkpoint - Verificar ingestão e extração
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Engenharia de Features
  - [x] 5.1 Implementar Feature Engineer (`src/features/feature_engineer.py`)
    - Classe `FeatureEngineer` com método `compute(user_id, sessions) -> FeatureVector | None`
    - Calcular features de engagement: total_sessions, total_viewing_hours, avg_session_duration_min, sessions_per_week, distinct_channels
    - Calcular features de qualidade: avg_happiness_score, avg_buffer_ratio, error_rate, avg_bitrate
    - Calcular features comportamentais: pct_episode/sport/live/show, distinct_devices, avg_pause_count, avg_seek_count
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 5.2 Implementar cálculo de trend features
    - Calcular trends somente se observação ≥ 4 semanas: viewing_time_trend, error_rate_trend, session_frequency_trend
    - Setar trends como None se período < 4 semanas
    - Excluir users com < 5 sessões (retornar None)
    - Default 0.0 para features não computáveis por dados faltantes
    - _Requirements: 3.4, 3.5, 3.6, 3.7_

  - [x] 5.3 Escrever property test para Feature Engineer
    - **Property 6: Data Integrity** — Feature Vectors produzidos são completos e consistentes (todas as features presentes, percentuais somam 100%)
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.5**

  - [x] 5.4 Escrever testes unitários para Feature Engineer
    - Testar com sessões sintéticas: user com muitas sessões, user com < 5 sessões, user com período < 4 semanas, campos null
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 6. Feature Store (DynamoDB)
  - [x] 6.1 Implementar Feature Store (`src/store/feature_store.py`)
    - Classe `FeatureStore` com métodos: `store()`, `get_latest()`, `get_version()`, `get_history()`
    - Schema DynamoDB: PK=user_id, SK=version (auto-increment)
    - Versionamento imutável (nunca sobrescrever)
    - Suporte a query por user_id + version ou date range
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 6.2 Escrever property test para imutabilidade do Feature Store
    - **Property 2: Immutable Storage** — Verificar que store() sempre cria nova versão, nunca sobrescreve
    - **Validates: Requirements 9.5**

  - [x] 6.3 Escrever testes de integração para Feature Store (moto/DynamoDB local)
    - Testar store, get_latest, get_version, get_history, versionamento
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [x] 7. Checkpoint - Verificar Feature Engineering e Store
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Pipeline de Machine Learning (SageMaker)
  - [x] 8.1 Implementar SageMaker ML Pipeline (`src/ml/sagemaker_pipeline.py`)
    - Classe `SageMakerMLPipeline` com métodos: `train()`, `predict_batch()`, `get_active_model()`
    - Suporte a XGBoost, LightGBM, CatBoost (configurável)
    - Split estratificado: train 70%, validation 15%, test 15%
    - Upload de training data para S3, criação de Training Job
    - Registro no Model Registry com métricas (Precision, Recall, F1, ROC AUC)
    - _Requirements: 10.1, 10.2, 10.3, 10.7, 10.8, 10.10_

  - [x] 8.2 Implementar Batch Transform e inferência determinística
    - SageMaker Batch Transform para scoring em lote
    - Garantir seed fixo para inferência determinística
    - Cada predição retorna: churn_probability, confidence, timestamp, model_version
    - Armazenar resultados no DynamoDB (tabela churn_predictions)
    - _Requirements: 10.4, 10.5, 10.6, 10.9_

  - [x] 8.3 Implementar Model Registry e ciclo de vida (`src/ml/model_registry.py`)
    - Integração com SageMaker Model Registry
    - Armazenar artefatos em S3 (model.tar.gz, hyperparameters.json, metrics.json)
    - Modelo ativo via approval status (configurável)
    - Suporte a rollback
    - Impedir deleção de modelos usados em produção
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_

  - [x] 8.4 Escrever property test para inferência determinística
    - **Property 1: Deterministic Inference** — Mesmo FeatureVector + mesmo modelo = mesma predição
    - **Validates: Requirements 10.5**

  - [x] 8.5 Escrever testes unitários para ML Pipeline
    - Testar preparação de dados, split estratificado, formato de saída, métricas
    - _Requirements: 10.1, 10.4, 10.7, 10.8_

- [x] 9. Explainability Engine (SHAP)
  - [x] 9.1 Implementar SHAP Explainer (`src/explainability/shap_explainer.py`)
    - Classe `SHAPExplainer` com TreeExplainer
    - Métodos: `explain()` e `explain_batch()`
    - Retornar top N features (default 10) com contribution_weight e normalized_impact
    - Se SHAP falhar para um user: prediction válida, SHAP = null
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

  - [x] 9.2 Escrever property test para separação de responsabilidades
    - **Property 3: Separation of Concerns** — SHAP nunca gera texto explicativo, apenas pesos numéricos
    - **Validates: Requirements 10.6, 12.4, 12.5**

  - [x] 9.3 Escrever testes unitários para SHAP Explainer
    - Testar com modelo mockado, verificar formato de saída, top features, fallback em caso de erro
    - _Requirements: 11.1, 11.2, 11.3, 11.6_

- [x] 10. Explicações via AWS Bedrock
  - [x] 10.1 Implementar Bedrock Explainer (`src/explanations/bedrock_explainer.py`)
    - Classe `BedrockExplainer` com Claude 3 Haiku
    - Método `generate_explanation()` recebe: user_id, churn_probability, confidence, top_features, user_feature_values, population_stats
    - Prompt estruturado em PT-BR, instrui a NÃO calcular probabilidade
    - Timeout 60s, retry 2x com intervalo 5s
    - Retornar None se Bedrock indisponível (graceful degradation)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_

  - [x] 10.2 Escrever property test para graceful degradation
    - **Property 5: Graceful Degradation** — Se Bedrock falhar, prediction permanece válida com explanation = null
    - **Validates: Requirements 12.6**

  - [x] 10.3 Escrever testes unitários para Bedrock Explainer
    - Testar geração de prompt, resposta válida, timeout, retry, fallback para null
    - _Requirements: 12.1, 12.2, 12.4, 12.6, 12.7_

- [x] 11. Checkpoint - Verificar ML Pipeline, SHAP e Bedrock
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Geração de Relatórios
  - [x] 12.1 Implementar Report Generator (`src/reports/report_generator.py`)
    - Classe `ReportGenerator` com métodos: `generate_individual_report()`, `generate_executive_report()`, `export_json()`, `export_markdown()`
    - Relatório individual: user_id, churn_probability, confidence, risk_tier, top_features, explanation
    - Relatório executivo: total analisados, distribuição por risco, média de churn, top fatores
    - Incluir metadata: model_version, feature_version, período, timestamp ISO 8601
    - Upload para S3 na estrutura definida no design
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 13.1, 13.2, 13.3, 13.4, 13.5_

  - [x] 12.2 Escrever testes unitários para Report Generator
    - Testar geração JSON/Markdown, report sem high-risk users, explanation indisponível
    - _Requirements: 6.1, 6.4, 6.5, 6.7, 13.3, 13.5_

- [x] 13. (Removido — Pattern Analyzer e Churn Predictor substituídos por SageMaker ML Pipeline)
  - _Os Requirements 4 e 5 foram substituídos pelos Requirements 10, 11 e 12._
  - _A predição agora é feita pelo SageMaker (task 8), explicabilidade pelo SHAP (task 9), e explicação em linguagem natural pelo Bedrock (task 10)._

- [x] 14. Pipeline Orchestrator (Step Functions)
  - [x] 14.1 Implementar orquestrador Step Functions (`src/orchestrator/step_functions.py`)
    - Definir state machine com estados: Ingestion → Extraction → FeatureEngineering → StoreFeatures → ChooseMode → Training/BatchPredict → Explainability → BedrockExplanations → GenerateReports
    - Choice state para modo train vs predict
    - Error handling e retry por estado
    - _Requirements: 8.1, 8.4, 16.1, 17.1, 17.3_

  - [x] 14.2 Implementar Lambda handlers para cada estágio do pipeline
    - Criar handlers em `src/orchestrator/handlers/`: `ingest_handler.py`, `extract_handler.py`, `feature_handler.py`, `store_handler.py`, `predict_handler.py`, `shap_handler.py`, `bedrock_handler.py`, `report_handler.py`
    - Cada handler loga início/fim com execution_id e timestamps
    - Persistir resultados ANTES do próximo estágio (data integrity)
    - _Requirements: 8.1, 8.3, 8.4, 17.3, 17.4_

  - [x] 14.3 Escrever property test para version traceability
    - **Property 4: Version Traceability** — Cada predição registra exatamente qual versão de modelo e features foi usada
    - **Validates: Requirements 15.4, 17.1**

  - [x] 14.4 Escrever property test para data integrity
    - **Property 6: Data Integrity** — Features persistidas ANTES da inferência; resultados persistidos ANTES do relatório
    - **Validates: Requirements 9.1, 17.4**

- [x] 15. Checkpoint - Verificar orquestração e handlers
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Monitoramento e Alertas
  - [x] 16.1 Implementar módulo de monitoramento (`src/common/monitoring.py`)
    - Publicar métricas custom no CloudWatch (namespace: ChurnPrediction)
    - Métricas: InferenceTime, PredictionCount, PredictionFailures, FeatureDriftDetected, BedrockTimeout, ExtractionErrors
    - Detecção de data drift (comparar features atuais vs distribuição de treino, alerta se shift > 2 std)
    - _Requirements: 16.1, 16.2, 16.3_

  - [x] 16.2 Implementar alertas e logging aprimorado
    - Alerta se inference time > 5s (configurável)
    - Alerta se failure rate > 5% (configurável)
    - Log timing separado para ML inference e Bedrock explanation
    - Audit trail: vincular cada predição a feature_version, model_version, explainability, explanation
    - _Requirements: 16.4, 16.5, 17.1, 17.2, 17.5_

  - [x] 16.3 Escrever testes unitários para monitoramento
    - Testar publicação de métricas, detecção de drift, geração de alertas
    - _Requirements: 16.1, 16.2, 16.4, 16.5_

- [x] 17. Dashboard Analítico (Streamlit)
  - [x] 17.1 Implementar data source do dashboard (`src/dashboard/data_source.py`)
    - Classe `DashboardDataSource` que lê de S3/DynamoDB
    - Métodos: `get_latest_execution()`, `get_predictions()`, `get_subscriber_detail()`, `get_history()`
    - Suporte a filtros por período, risk_level, user_id
    - _Requirements: 14.1, 14.3, 14.4, 14.6_

  - [x] 17.2 Implementar páginas do dashboard Streamlit (`src/dashboard/app.py`)
    - Página Overview: KPIs (total analisados, high/medium/low risk, média churn probability)
    - Página Charts: distribuição de risco (pie), histograma de scores, top fatores (bar), evolução temporal (line)
    - Página Subscriber Detail: busca por user_id, exibir score, confidence, top features, explanation
    - Página Export: download filtrado em JSON/Markdown
    - Refresh automático após nova execução
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_

  - [x] 17.3 Escrever testes unitários para data source do dashboard
    - Testar queries, filtros, formatação de dados
    - _Requirements: 14.1, 14.3, 14.6_

- [x] 18. Infraestrutura AWS CDK
  - [x] 18.1 Implementar stack CDK principal (`infra/stacks/churn_prediction_stack.py`)
    - S3 Bucket com estrutura de prefixos (input/, raw_data/, features/, models/, predictions/, reports/)
    - DynamoDB tables: churn_feature_store, churn_predictions, churn_executions
    - Secrets Manager para NPAW API key
    - IAM Roles (least privilege) para Lambda, SageMaker, ECS
    - VPC endpoints para serviços AWS
    - _Requirements: 7.1, 9.1, 10.10, 15.1_

  - [x] 18.2 Implementar stack CDK para Step Functions e EventBridge
    - Step Functions state machine com definição completa
    - Lambda functions para cada handler do pipeline
    - EventBridge rule para agendamento semanal (cron)
    - S3 event trigger para início do pipeline via upload
    - _Requirements: 8.1, 14.6, 16.1, 17.3_

  - [x] 18.3 Implementar stack CDK para ECS Fargate (Dashboard)
    - ECS Fargate service com Streamlit
    - ALB + Cognito para autenticação
    - CloudWatch Log Groups para cada estágio
    - CloudWatch Alarms para métricas customizadas
    - _Requirements: 14.7, 16.3, 16.4, 16.5_

  - [x] 18.4 Escrever testes de snapshot/assertion para CDK stacks
    - Testar que recursos esperados existem nas stacks
    - _Requirements: 7.1, 9.1, 14.7_

- [x] 19. Integração e wiring final
  - [x] 19.1 Integrar todos os componentes no pipeline completo
    - Garantir que cada handler invoca os componentes corretos na sequência
    - Verificar fluxo de dados entre estágios (S3 paths, DynamoDB keys)
    - Configurar variáveis de ambiente nos Lambdas
    - Validar que dados são persistidos antes de cada estágio subsequente
    - _Requirements: 8.1, 9.1, 17.3, 17.4_

  - [x] 19.2 Criar fixtures e dados de teste para pipeline end-to-end
    - 10 churned users + 10 active users (sessões sintéticas)
    - Modelo pré-treinado para testes de inferência
    - Mock de Bedrock com respostas fixas
    - _Requirements: 10.5, 12.6_

  - [x] 19.3 Escrever testes end-to-end do pipeline
    - Testar pipeline completo com dados sintéticos (Step Functions local)
    - Verificar que todos os artefatos são gerados corretamente
    - _Requirements: 8.1, 8.6, 17.1, 17.4_

- [x] 20. Checkpoint final - Verificar integração completa
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marcadas com `*` são opcionais e podem ser ignoradas para um MVP mais rápido
- Cada task referencia requisitos específicos para rastreabilidade
- Checkpoints garantem validação incremental
- Property tests validam propriedades universais de correção
- Unit tests validam exemplos específicos e edge cases
- Toda infraestrutura é definida via AWS CDK (Python) — sem criação manual de recursos
- O dashboard usa Streamlit por ser nativo Python e rápido de construir
- Bedrock é usado como componente não-crítico (graceful degradation se indisponível)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4"] },
    { "id": 2, "tasks": ["1.5", "2.1", "3.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "3.2"] },
    { "id": 4, "tasks": ["3.3", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "5.4", "6.1"] },
    { "id": 6, "tasks": ["6.2", "6.3", "8.1"] },
    { "id": 7, "tasks": ["8.2", "8.3"] },
    { "id": 8, "tasks": ["8.4", "8.5", "9.1"] },
    { "id": 9, "tasks": ["9.2", "9.3", "10.1"] },
    { "id": 10, "tasks": ["10.2", "10.3", "12.1"] },
    { "id": 11, "tasks": ["12.2", "14.1"] },
    { "id": 12, "tasks": ["14.2", "14.3", "14.4"] },
    { "id": 14, "tasks": ["16.1", "17.1"] },
    { "id": 15, "tasks": ["16.2", "16.3", "17.2"] },
    { "id": 16, "tasks": ["17.3", "18.1"] },
    { "id": 17, "tasks": ["18.2", "18.3"] },
    { "id": 18, "tasks": ["18.4", "19.1"] },
    { "id": 19, "tasks": ["19.2"] },
    { "id": 20, "tasks": ["19.3"] }
  ]
}
```
