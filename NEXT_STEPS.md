# Próximos Passos - AI Churn Prediction

## Status Atual (06/08/2026)

### ✅ Pipeline COMPLETO e Funcionando End-to-End!

O pipeline roda todos os 8 estágios com sucesso:
1. ✅ **Ingestion** — Lê CSV do S3, valida UUIDs v4/v5
2. ✅ **Extraction** — Busca sessões NPAW com API key do Secrets Manager
3. ✅ **Feature Engineering** — Calcula 19 features comportamentais
4. ✅ **Store Features** — Persiste no DynamoDB
5. ✅ **Batch Predict** — Modelo local (GradientBoosting) carregado do S3
6. ✅ **Explainability** — Feature importance (SHAP Docker em progresso)
7. ✅ **Bedrock Explanations** — Geração de texto explicativo
8. ✅ **Generate Reports** — Relatório final

### Modelo de ML Treinado
- **Algoritmo**: GradientBoosting (sklearn)
- **F1 Score**: 0.9412 | **ROC AUC**: 0.9821
- **Dataset**: 273 registros (166 churn + 107 ativos)
- **Artefato**: `s3://sky-brazil-churn-prediction/models/approved/churn_model.pkl`
- **Top features**: total_viewing_hours (45.6%), pct_sport (17.3%), total_sessions (10.2%)

### Infraestrutura Deployada
- ✅ S3 Bucket (`sky-brazil-churn-prediction`)
- ✅ 3 DynamoDB tables (feature_store, predictions, executions)
- ✅ Secrets Manager com NPAW API key
- ✅ IAM Roles (Lambda, SageMaker, ECS, CodeBuild)
- ✅ 8 Lambda functions com Layer (sklearn, numpy, pandas, aiohttp, pydantic)
- ✅ Step Functions state machine (`churn-prediction-pipeline`)
- ✅ EventBridge cron (segunda 08:00 UTC)
- ✅ S3 trigger (upload em `input/` → dispara pipeline)
- ✅ ECR Repository (`churn-pipeline-shap`)
- ✅ CodeBuild Project (`churn-shap-image-builder`)

---

## Em Progresso

### SHAP Docker Lambda (CodeBuild)
- Build da imagem Docker via CodeBuild (sem Docker Desktop)
- Dockerfile com gcc + shap + sklearn no ECR
- Quando pronto, atualizar Lambda `churn-pipeline-shap` com imagem ECR
- CDK preparado para referenciar imagem ECR existente

---

## Próximos Passos

### 1. Completar SHAP Docker (quando build passar)
```powershell
# Verificar status do build
aws codebuild batch-get-builds --ids "churn-shap-image-builder:51fc1631-ef0d-409d-9962-152e095ce71d" --region us-east-1 --query "builds[0].buildStatus"

# Se SUCCEEDED, atualizar Lambda
aws lambda update-function-code --function-name churn-pipeline-shap --image-uri 761018874615.dkr.ecr.us-east-1.amazonaws.com/churn-pipeline-shap:latest --region us-east-1

# Deploy CDK com DockerImageFunction from ECR
cdk deploy ChurnPredictionStack --require-approval never
```

### 2. Aumentar Dataset de Treinamento
- Rodar extração com todos os 714 cancelados + 800 ativos
- Retreinar modelo com dataset maior (~1000+ registros)
- Esperar F1 > 0.90 confirmado em cross-validation

### 3. Testar Pipeline com Dados Reais (Produção)
- Converter planilha completa: `python scripts/convert-xlsx-to-input.py "docs/20260729 Cancelados may-26.xlsx" input_full.csv`
- Upload: `aws s3 cp input_full.csv s3://sky-brazil-churn-prediction/input/`
- Monitorar execução e verificar predições no DynamoDB

### 4. Monitoramento e Alertas
- Criar alarme CloudWatch para falhas no Step Functions
- Dashboard com métricas do pipeline (duração, erros, contagens)

### 5. Retraining Automático
- Cron mensal para retreinar com novos dados de cancelamento
- Comparar métricas do novo modelo vs. anterior
- Auto-approve se F1 >= threshold

---

## Comandos Úteis

```powershell
# Executar pipeline (upload CSV)
aws s3 cp input_test.csv s3://sky-brazil-churn-prediction/input/ --region us-east-1

# Ver execuções
aws stepfunctions list-executions --state-machine-arn "arn:aws:states:us-east-1:761018874615:stateMachine:churn-prediction-pipeline" --max-results 5 --region us-east-1

# Ver predições no DynamoDB
aws dynamodb scan --table-name churn_predictions --region us-east-1 --max-items 5

# Retreinar modelo
python scripts/fast_training_extract.py --max 500
python scripts/train_local.py --data data/training_data_sample.csv

# Build imagem SHAP via CodeBuild
python scripts/build_shap_zip.py  # (criar ZIP com paths Unix)
aws s3 cp shap-source.zip s3://sky-brazil-churn-prediction/codebuild/
aws codebuild start-build --project-name churn-shap-image-builder --region us-east-1

# Deploy
cdk deploy ChurnPredictionStack --require-approval never
```

---

## Conta AWS
- Account: 761018874615
- Região: us-east-1
- Stack: ChurnPredictionStack
- State Machine: churn-prediction-pipeline
- ECR: churn-pipeline-shap
- CodeBuild: churn-shap-image-builder
