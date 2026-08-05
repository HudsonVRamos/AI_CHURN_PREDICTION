# Requirements Document

## Introduction

Plataforma de predição de churn para a plataforma de streaming Sky Brazil. O sistema extrai dados comportamentais de assinantes via API NPAW, armazena features versionadas, treina modelos supervisionados de Machine Learning (XGBoost/LightGBM/CatBoost) para predizer a probabilidade de churn, utiliza técnicas de Explainable AI (SHAP) para identificar os fatores que contribuem para cada predição, e gera explicações em linguagem natural via AWS Bedrock.

O sistema é responsável por:
- Predizer a probabilidade de churn dos assinantes
- Explicar os fatores comportamentais que contribuíram para a predição
- Gerar relatórios técnicos para as equipes de Produto, BI e Marketing
- Apresentar resultados em um dashboard analítico interativo

O sistema **não** é responsável por:
- Recomendar campanhas de retenção, descontos ou ações de marketing

## Glossary

- **System**: O sistema AI Churn Prediction como um todo
- **Data_Extractor**: Componente responsável por extrair dados de sessões de usuários da API NPAW
- **Feature_Engineer**: Componente que transforma dados brutos de sessões em features comportamentais agregadas por usuário
- **Report_Generator**: Componente que produz relatórios com scores de risco e insights acionáveis
- **NPAW_API**: API REST da NPAW que fornece dados de sessões de vídeo (endpoint: `GET https://api.npaw.com/sky_brazil/rawdata`)
- **AWS_Bedrock**: Serviço de IA generativa da AWS utilizado para geração de explicações em linguagem natural sobre as predições
- **Churned_User**: Assinante que já cancelou sua assinatura na plataforma
- **Active_User**: Assinante com assinatura ativa na plataforma
- **Session**: Um registro individual de visualização de vídeo na NPAW, contendo métricas de engajamento, qualidade, erros e dispositivo
- **Feature_Vector**: Conjunto de métricas comportamentais agregadas que representam o perfil de uso de um assinante
- **Churn_Probability**: Valor numérico de 0.0 a 1.0 gerado pelo modelo de ML que indica a probabilidade de um assinante cancelar
- **Risk_Tier**: Classificação de risco derivada da Churn_Probability: Low (0-0.30), Medium (0.31-0.60), High (0.61-1.0)
- **Feature_Store**: Componente de armazenamento versionado de Feature Vectors para reutilização e auditoria
- **ML_Pipeline**: Pipeline de Machine Learning supervisionado no Amazon SageMaker para treinamento e inferência do modelo de churn
- **Explainability_Engine**: Componente que utiliza técnicas de Explainable AI (SHAP) para calcular a importância de cada feature na predição
- **Dashboard**: Interface web interativa para visualização dos resultados de predição
- **Model_Registry**: Amazon SageMaker Model Registry, componente que gerencia o ciclo de vida e versões dos modelos treinados
- **SageMaker**: Amazon SageMaker, serviço gerenciado de ML da AWS utilizado para treino, inferência e versionamento de modelos
- **Batch_Transform**: SageMaker Batch Transform, modo de inferência em lote para processar múltiplos assinantes de uma vez

## Requirements

### Requirement 1: Ingestão de Listas de Usuários

**User Story:** As a data analyst, I want to provide lists of churned and active user IDs to the system, so that it can process both groups for ML model training and churn prediction.

#### Acceptance Criteria

1. WHEN a list of Churned_User IDs is provided, THE System SHALL validate that each ID is a non-empty string in UUID v4 format and accept lists containing between 1 and 50,000 valid User IDs
2. WHEN a list of Active_User IDs is provided, THE System SHALL validate that each ID is a non-empty string in UUID v4 format and accept lists containing between 1 and 50,000 valid User IDs
3. IF a provided User ID is not in valid UUID format, THEN THE System SHALL reject that ID, log a warning identifying the invalid ID, and continue processing the remaining valid IDs
4. IF all User IDs in a provided list are invalid or the list is empty, THEN THE System SHALL return an error indicating that at least one valid User ID is required
5. THE System SHALL accept lists of User IDs via CSV file (single column with header "user_id", one ID per row), JSON file (array of strings under a "user_ids" key), or direct array input
6. IF duplicate User IDs are found within the same list, THEN THE System SHALL deduplicate them and process each unique ID only once

### Requirement 2: Extração de Dados da NPAW

**User Story:** As a data analyst, I want the system to extract behavioral data from NPAW for each user, so that I have complete viewing history for analysis.

#### Acceptance Criteria

1. WHEN a valid User ID is provided, THE Data_Extractor SHALL query the NPAW_API endpoint `GET https://api.npaw.com/sky_brazil/rawdata` with the filter `[{"name":"uf","rules":{"user_id":["<USER_ID>"]}}]` and authenticate using the `npaw-api-key` header
2. THE Data_Extractor SHALL request sessions ordered by `end_at` in descending order with a batch size of 100 records per request
3. WHEN the NPAW_API returns exactly the batch size number of records (100), THE Data_Extractor SHALL paginate using the `offset` parameter, incrementing by 100, until fewer than 100 records are returned or a maximum of 5000 sessions per user is reached
4. THE Data_Extractor SHALL extract data for a configurable time window (default: 6 months prior to the extraction date for active users, or 6 months prior to cancellation date for churned users) using the `fromDate` and `toDate` parameters
5. IF the NPAW_API returns an authentication error (HTTP 401 or 403), THEN THE Data_Extractor SHALL stop processing all users immediately and report the authentication failure
6. IF the NPAW_API returns a server error (HTTP 5xx) or a request timeout (exceeding 60 seconds), THEN THE Data_Extractor SHALL retry the request up to 3 times with exponential backoff (2s, 4s, 8s) before marking that user as failed
7. IF the NPAW_API returns no data for a User ID (empty `values` array), THEN THE Data_Extractor SHALL log a warning including the User ID and date range, and skip that user without stopping the batch
8. THE Data_Extractor SHALL wait a configurable interval between API calls (default: 1 second) to avoid overloading the NPAW_API, and SHALL support concurrent extraction of up to 5 users in parallel

### Requirement 3: Engenharia de Features Comportamentais

**User Story:** As a data scientist, I want raw session data transformed into meaningful behavioral features per user, so that patterns can be identified by the AI model.

#### Acceptance Criteria

1. WHEN raw session data is available for a user, THE Feature_Engineer SHALL compute the following engagement features: total sessions count, total effective viewing time (in hours, derived from the sum of `effective_time` in milliseconds), average session duration (in minutes, based on `effective_time` per session), viewing frequency (sessions per week, calculated as total sessions divided by the number of weeks in the observation period), and count of distinct `content_channel` values watched
2. WHEN raw session data is available for a user, THE Feature_Engineer SHALL compute the following quality features: average `happiness_score` (scale 0–10, excluding sessions where `happiness_score` is null or negative), average `buffer_ratio`, error rate (count of sessions where `error_code` is non-empty divided by total sessions, expressed as a value between 0.0 and 1.0), and average `avg_bitrate` (in bps)
3. WHEN raw session data is available for a user, THE Feature_Engineer SHALL compute the following behavioral features: content type distribution (percentage of sessions for each of the four categories EPISODE, SPORT, LIVE, SHOW, summing to 100%), distinct device count (based on unique `device.device_model` values), average `pause_count` per session, and average `seek_count` per session
4. WHEN raw session data is available for a user with an observation period of at least 4 weeks, THE Feature_Engineer SHALL compute the following trend features: change in weekly viewing time (linear slope of weekly total `effective_time` over the observation period, expressed as hours per week), change in error rate (difference between the error rate in the last 2 weeks and the first 2 weeks of the observation period), and change in session frequency (difference between sessions per week in the last 2 weeks and the first 2 weeks)
5. THE Feature_Engineer SHALL produce one Feature_Vector per user containing all computed features, using a default value of 0.0 for any feature that cannot be computed due to missing or null source data fields
6. IF a user has fewer than 5 sessions in the observation period, THEN THE Feature_Engineer SHALL flag that user as having insufficient data and exclude the user from prediction
7. IF a user has an observation period shorter than 4 weeks, THEN THE Feature_Engineer SHALL compute engagement, quality, and behavioral features but SHALL set all trend features to null in the Feature_Vector

### Requirement 4: (Substituído pelo Requirement 10)

**User Story:** As a data scientist, I want the system to identify churn patterns, so that predictions are based on data-driven insights.

#### Acceptance Criteria

1. THIS REQUIREMENT HAS BEEN SUPERSEDED by Requirement 10 (Pipeline de Machine Learning). The identification of churn-related patterns is now performed by the supervised ML model during training, which learns feature importance from labeled data (churned vs active users). The explicit Pattern_Analyzer component via Bedrock was removed in favor of the deterministic, reproducible SageMaker approach.

### Requirement 5: (Substituído pelos Requirements 10 + 11)

**User Story:** As a retention manager, I want the system to score active subscribers for churn risk, so that I can identify who needs intervention.

#### Acceptance Criteria

1. THIS REQUIREMENT HAS BEEN SUPERSEDED by Requirements 10 (SageMaker ML inference via Batch Transform), 11 (SHAP explainability), and 12 (Bedrock natural language explanation). The scoring of active users against churn patterns via Bedrock was replaced by deterministic ML inference with SHAP-based feature importance and optional Bedrock explanation in Portuguese.

### Requirement 6: Geração de Relatório Básico

> **Nota:** Este requisito é mantido como funcionalidade mínima de relatório. O Requirement 13 (Relatórios Aprimorados) expande e substitui esta funcionalidade com relatórios mais completos baseados no output do ML + SHAP + Bedrock.

**User Story:** As a retention manager, I want a clear report with at-risk subscribers, so that my team can intervene before customers cancel.

#### Acceptance Criteria

1. WHEN all Active_Users have been scored by the ML_Pipeline, THE Report_Generator SHALL produce a summary report containing: total users analyzed, distribution by risk tier (count and percentage for Low, Medium, High), and the top behavioral factors identified by the Explainability_Engine
2. WHEN all Active_Users have been scored, THE Report_Generator SHALL produce a detailed list of High-risk users sorted by churn_probability descending, containing for each user: User ID, churn_probability, confidence, risk_tier, and top features from SHAP
3. THE Report_Generator SHALL output the report in JSON format for programmatic consumption
4. IF the output configuration specifies Markdown format, THEN THE Report_Generator SHALL additionally produce a human-readable Markdown version of the report
5. THE Report_Generator SHALL include a timestamp in ISO 8601 format and metadata identifying: the analysis period start and end dates, number of churned users used for training, and model version
6. IF no Active_Users are classified as High risk, THEN THE Report_Generator SHALL produce the summary report with zero High-risk count and omit the detailed High-risk user list

### Requirement 7: Configuração e Credenciais

**User Story:** As a system administrator, I want to configure API keys, time windows, and thresholds externally, so that the system can be adapted without code changes.

#### Acceptance Criteria

1. THE System SHALL read the NPAW API key from an environment variable or AWS Secrets Manager (never hardcoded), checking the environment variable first and falling back to AWS Secrets Manager if the environment variable is not set
2. THE System SHALL read the AWS Bedrock configuration (region, model ID) from a configuration file or environment variables, with environment variables taking precedence over configuration file values when both are present
3. THE System SHALL allow configuration of the observation time window (default: 6 months) via configuration file, accepting values between 1 and 24 months expressed as an integer number of months
4. THE System SHALL allow configuration of the minimum sessions threshold for valid analysis (default: 5 sessions) via configuration file, accepting integer values between 1 and 10000
5. IF a required configuration value is missing, THEN THE System SHALL fail at startup with an error message that includes the name of the missing configuration parameter and the expected source (environment variable name or configuration file path)
6. IF a configuration value is present but invalid (non-numeric value for numeric fields, out-of-range value, or empty string), THEN THE System SHALL fail at startup with an error message that includes the parameter name, the invalid value provided, and the acceptable format or range

### Requirement 8: Logging e Observabilidade

**User Story:** As a system administrator, I want comprehensive logging throughout the pipeline, so that I can monitor progress, debug failures, and audit results.

#### Acceptance Criteria

1. THE System SHALL log the start and completion of each pipeline stage (ingestion, extraction, feature engineering, ML inference, explainability, Bedrock explanation, report generation) with timestamps and a unique execution ID that correlates all log entries of the same pipeline run
2. THE System SHALL log progress during data extraction at intervals of no more than every 50 users processed or every 60 seconds (whichever occurs first), including: number of users processed, number of users remaining, and number of users failed
3. IF any stage of the pipeline fails for an individual user, THEN THE System SHALL log the error at ERROR level with full context (stage name, user ID, error message, stack trace) and continue processing the remaining users in the batch
4. IF a pipeline stage fails due to a systemic error (e.g., loss of API connectivity or resource exhaustion) that prevents processing any further users, THEN THE System SHALL log the error at CRITICAL level and abort the current stage while preserving all results processed up to the point of failure
5. THE System SHALL use structured logging (JSON format) compatible with AWS CloudWatch, including the fields: timestamp, execution_id, level (DEBUG, INFO, WARNING, ERROR, CRITICAL), stage, and message
6. THE System SHALL log a final execution summary at INFO level containing: total execution time in seconds, users processed successfully, users failed, and output file locations


### Requirement 9: Armazenamento de Features (Feature Store)

**User Story:** As a data scientist, I want to store all generated behavioral features so they can be reused, audited, and versioned over time.

#### Acceptance Criteria

1. THE System SHALL store all Feature_Vectors in the Feature_Store before the prediction stage executes
2. EACH stored Feature_Vector SHALL contain a unique version identifier (auto-incremented per user)
3. EACH stored Feature_Vector SHALL include: generation timestamp (ISO 8601), observation time window used (start and end dates), and the subscriber's User ID
4. THE System SHALL allow querying previous versions of features for the same subscriber by User ID and version number or date range
5. THE System SHALL NOT overwrite previous versions of features; all versions SHALL be retained and immutable once stored
6. THE Feature_Store SHALL support retrieval of the latest Feature_Vector for a given User ID without requiring the version number

### Requirement 10: Pipeline de Machine Learning (Amazon SageMaker)

**User Story:** As a data scientist, I want to use a supervised Machine Learning model on Amazon SageMaker to calculate churn probability, enabling objective and reproducible measurements.

#### Acceptance Criteria

1. THE ML_Pipeline SHALL train a supervised model on Amazon SageMaker using Feature_Vectors of both Active_Users (label=0) and Churned_Users (label=1)
2. THE ML_Pipeline SHALL support, at minimum, the following algorithms: XGBoost (SageMaker built-in), LightGBM (SageMaker built-in), and CatBoost (via SageMaker custom container)
3. THE active algorithm SHALL be configurable via configuration file without code changes
4. EACH prediction SHALL return: churn probability (float between 0.0 and 1.0), confidence degree (float between 0.0 and 1.0), inference timestamp (ISO 8601), and model version used
5. FOR the same set of input features and model version, THE ML_Pipeline SHALL always produce the same prediction result (deterministic inference)
6. THE ML_Pipeline SHALL NOT generate natural language explanations (that responsibility belongs to AWS_Bedrock)
7. THE ML_Pipeline SHALL compute and expose the following model evaluation metrics after training: Precision, Recall, F1-Score, and ROC AUC
8. THE ML_Pipeline SHALL split the training data into train (70%), validation (15%), and test (15%) sets using stratified sampling to maintain class balance
9. THE ML_Pipeline SHALL use SageMaker Batch Transform for scoring Active_Users in bulk (non-real-time inference)
10. THE ML_Pipeline SHALL store training artifacts (model files, hyperparameters, metrics) in Amazon S3 with a path structure that includes the model version

### Requirement 11: Explicabilidade da Predição

**User Story:** As a product analyst, I want to understand which factors influenced each prediction so that I can understand subscriber behavior.

#### Acceptance Criteria

1. AFTER each prediction, THE Explainability_Engine SHALL calculate the importance of each feature used by the model
2. THE Explainability_Engine SHALL automatically identify the top features with the highest impact on the classification (configurable limit, default: top 10)
3. FOR EACH subscriber prediction, THE Explainability_Engine SHALL store: feature name, contribution weight (signed float indicating positive or negative impact), and normalized impact on the prediction (value between -1.0 and 1.0)
4. THE Explainability_Engine SHALL support SHAP (SHapley Additive exPlanations) or an equivalent Explainable AI technique
5. THE explainability information SHALL be available for use in report generation and dashboard visualization
6. IF the explainability computation fails for a specific subscriber, THE prediction result SHALL remain valid and the report SHALL indicate that feature importance could not be computed for that subscriber

### Requirement 12: Explicação dos Resultados utilizando AWS Bedrock

**User Story:** As a business analyst, I want to receive a natural language explanation about the factors that led a subscriber to present high churn risk.

#### Acceptance Criteria

1. THE System SHALL send to AWS_Bedrock for explanation generation: churn probability, confidence degree, top features responsible for the prediction (from Explainability_Engine), feature values for the subscriber, and statistical comparison with the analyzed population (mean, median, standard deviation for each relevant feature)
2. AWS_Bedrock SHALL generate a natural language explanation based exclusively on the information received in the prompt
3. AWS_Bedrock SHALL produce an executive summary for subscribers classified as Medium or High Risk
4. AWS_Bedrock SHALL NOT calculate the churn probability (the ML model is solely responsible for that)
5. AWS_Bedrock SHALL NOT alter or override the result produced by the Machine Learning model
6. IF AWS_Bedrock is unavailable or fails to respond within 60 seconds, THE prediction SHALL remain valid and THE report SHALL indicate that the natural language explanation could not be generated, with the field set to null
7. THE natural language explanation SHALL be generated in Portuguese (Brazilian)

### Requirement 13: Relatórios Aprimorados

**User Story:** As a product or BI analyst, I want to receive complete reports containing prediction, evidence, and explanations to facilitate churn factor analysis.

#### Acceptance Criteria

1. THE individual subscriber report SHALL contain: subscriber User ID, churn probability (0.0-1.0), confidence degree, risk classification (Low/Medium/High), top factors responsible for the prediction (from Explainability_Engine with feature names, weights, and values), and natural language explanation (from AWS_Bedrock)
2. THE executive report SHALL contain: total subscribers analyzed, distribution by risk level (count and percentage for Low, Medium, High), average churn probability across all analyzed subscribers, and top behavioral factors identified in the population (aggregated feature importance)
3. THE System SHALL generate reports in the following formats: JSON (for programmatic consumption) and Markdown (for human reading)
4. ALL reports SHALL include: model version, feature version, analysis period (start and end dates), and execution timestamp (ISO 8601)
5. IF the natural language explanation is unavailable for a subscriber, THE report SHALL include all other fields and indicate explanation_status as "unavailable"

### Requirement 14: Dashboard Analítico

**User Story:** As a product, BI, or marketing analyst, I want to visualize prediction results in an interactive dashboard to facilitate subscriber behavior analysis.

#### Acceptance Criteria

1. THE Dashboard SHALL display general indicators: total subscribers analyzed, count of High-Risk subscribers, count of Medium-Risk subscribers, count of Low-Risk subscribers, and average churn probability
2. THE Dashboard SHALL display charts showing: risk level distribution (pie/donut chart), churn score distribution (histogram), top behavioral factors influencing predictions (bar chart), historical evolution of subscriber count by risk level (line chart), and temporal evolution of average churn probability (line chart)
3. THE Dashboard SHALL support filters by: time period (date range selector), risk level (Low, Medium, High, or All), and subscriber User ID (search)
4. WHEN a subscriber is selected in the Dashboard, THE Dashboard SHALL display: churn score, confidence degree, top features responsible for classification (with values and weights), behavioral metric values, temporal evolution of key metrics, and the natural language explanation generated by AWS_Bedrock
5. THE Dashboard SHALL allow export of filtered results in JSON and Markdown formats
6. THE Dashboard SHALL refresh data after each new pipeline execution without requiring manual intervention
7. THE Dashboard SHALL be accessible via web browser without requiring local software installation

### Requirement 15: Gerenciamento do Ciclo de Vida do Modelo (SageMaker Model Registry)

**User Story:** As a data scientist, I want to control the versions of trained models to ensure traceability and auditability.

#### Acceptance Criteria

1. THE SageMaker Model_Registry SHALL store all versions of trained models with their serialized artifacts in Amazon S3
2. EACH model version SHALL contain: algorithm used (XGBoost, LightGBM, or CatBoost), training date (ISO 8601), dataset version (reference to Feature_Store version used for training), and evaluation metrics (Precision, Recall, F1-Score, ROC AUC)
3. THE active model used for predictions SHALL be configurable via SageMaker Model Registry approval status or configuration file without code changes
4. EACH prediction execution SHALL record which model version (SageMaker Model Package ARN) was used in the prediction results and logs
5. THE System SHALL support rolling back to a previous model version by changing the approved model in SageMaker Model Registry
6. THE Model_Registry SHALL NOT allow deletion of model versions that have been used in production predictions (SageMaker Model Package with InferenceSpecification)

### Requirement 16: Monitoramento das Predições

**User Story:** As a system administrator, I want to continuously monitor the Machine Learning pipeline performance.

#### Acceptance Criteria

1. THE System SHALL monitor and expose: average inference time (milliseconds per prediction), total predictions executed (counter), total prediction failures (counter), and churn score distribution (histogram of the last N predictions, configurable, default 1000)
2. THE System SHALL detect significant changes in feature distribution over time (data drift) by comparing current feature statistics against the training data distribution, alerting when any feature's mean shifts by more than 2 standard deviations
3. THE System SHALL expose all monitoring metrics in a format compatible with AWS CloudWatch
4. THE System SHALL generate an alert when the average inference time exceeds a configurable threshold (default: 5 seconds)
5. THE System SHALL generate an alert when the prediction failure rate exceeds a configurable threshold (default: 5% of predictions in a batch)

### Requirement 17: Logging Aprimorado

**User Story:** As a system administrator, I want complete traceability of all predictions executed.

#### Acceptance Criteria

1. EACH pipeline execution SHALL record: execution_id (unique UUID), model version, feature version, prediction timestamp (ISO 8601), and AWS Bedrock model ID used for explanation generation
2. THE System SHALL record the time spent: on ML model inference (milliseconds), and on AWS Bedrock explanation generation (milliseconds), as separate timing entries in the execution log
3. EACH pipeline stage (extraction, feature engineering, ML inference, explainability, Bedrock explanation, report generation) SHALL produce independent log streams that can be queried separately
4. IF a failure occurs during report generation, THE prediction results and explainability data SHALL NOT be lost; they SHALL be persisted before the report generation stage begins
5. THE System SHALL maintain an audit trail linking each prediction to its: input Feature_Vector version, model version, explainability results, and generated explanation
