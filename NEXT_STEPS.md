# Próximos Passos - AI Churn Prediction

## Status Atual (05/08/2026)

### O que está deployado e funcionando:
- ✅ S3 Bucket (`sky-brazil-churn-prediction`)
- ✅ 3 DynamoDB tables (feature_store, predictions, executions)
- ✅ Secrets Manager com NPAW API key configurada
- ✅ IAM Roles (Lambda, SageMaker, ECS)
- ✅ 8 Lambda functions (com Layer de dependências: pydantic, aiohttp, pyyaml)
- ✅ Step Functions state machine (`churn-prediction-pipeline`)
- ✅ EventBridge cron (segunda 08:00 UTC)
- ✅ S3 trigger (upload em `input/` → dispara pipeline)
- ✅ CloudWatch Log Groups (7 estágios)

### O que funciona no trigger:
- Upload para S3 `input/` → Lambda S3Trigger é invocada → Step Functions inicia
- Lambda de ingestão importa corretamente (Layer com pydantic OK)

---

## Problema a Resolver (Próxima Sessão)

### 1. Ingest Handler não lê CSV do S3 quando trigger é S3

**Causa:** O S3 trigger envia o input como:
```json
{"mode": "predict", "source": "s3", "trigger_key": "input/test_final.csv"}
```

O `ingest_handler` tenta parsear `"s3"` como lista de IDs ou CSV — e falha.

**Solução:** Ajustar `ingest_handler.py` para:
1. Detectar quando `source == "s3"` ou quando `trigger_key` está presente
2. Baixar o CSV de `s3://sky-brazil-churn-prediction/{trigger_key}`
3. Passar o conteúdo CSV para `ingest_user_ids(content, source_format="csv")`

Código aproximado:
```python
source = event.get("source", [])
trigger_key = event.get("trigger_key")

if trigger_key:
    # Trigger S3: baixar o CSV do bucket
    s3_client = boto3.client("s3")
    bucket = os.environ.get("BUCKET_NAME", "sky-brazil-churn-prediction")
    response = s3_client.get_object(Bucket=bucket, Key=trigger_key)
    source = response["Body"].read().decode("utf-8")
    source_format = "csv"  # CSV por padrão quando vem do S3
```

---

### 2. Extract Handler precisa buscar API key do Secrets Manager

O `extract_handler` atualmente espera `npaw_api_key` no evento, mas em produção
deve buscar do Secrets Manager automaticamente.

**Solução:**
```python
if not api_key:
    secrets_client = boto3.client("secretsmanager")
    response = secrets_client.get_secret_value(SecretId="churn-prediction/npaw-api-key")
    api_key = response["SecretString"]
```

---

### 3. UUIDs da planilha são v5 (não v4)

Os UUIDs da Sky Brazil (ex: `9e527027-b330-5d8f-aaa2-bf6653bd6eec`) são UUID v5
(dígito de versão é `5`). A validação atual aceita apenas UUID v4.

**Solução:** Relaxar o regex de `UUID_V4_REGEX` para aceitar v4 E v5:
```python
UUID_VALID_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[45][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
```

---

### 4. Testar pipeline end-to-end com 10 users reais

Depois de corrigir itens 1-3:
1. Upload CSV: `aws s3 cp input_test.csv s3://sky-brazil-churn-prediction/input/`
2. Verificar Step Functions: `aws stepfunctions list-executions ...`
3. Ver logs: `aws logs tail /aws/lambda/churn-pipeline-ingest --since 5m`
4. Verificar se NPAW retorna dados para os users

---

### 5. Estágios seguintes do pipeline (depois que ingest + extract funcionarem)

- Feature Engineering: pode precisar de numpy na Layer (ou usar Lambda com mais memória)
- ML Inference: precisa de modelo treinado primeiro (pode mockar no início)
- SHAP: precisa de numpy + shap na Layer (pode ultrapassar 250MB → usar Docker Lambda)
- Bedrock: pode funcionar direto (só boto3 necessário)
- Reports: deve funcionar com a Layer atual

---

## Comandos Úteis

```powershell
# Ver execuções do pipeline
aws stepfunctions list-executions --state-machine-arn "arn:aws:states:us-east-1:761018874615:stateMachine:churn-prediction-pipeline" --max-results 5 --region us-east-1

# Ver erro da última execução
aws stepfunctions get-execution-history --execution-arn "ARN_AQUI" --region us-east-1 --query "events[?type=='LambdaFunctionFailed'].lambdaFunctionFailedEventDetails"

# Ver logs de uma Lambda
aws logs tail /aws/lambda/churn-pipeline-ingest --since 5m --region us-east-1

# Redeploy rápido (após editar código)
cd infra; cdk deploy ChurnPredictionStack --require-approval never

# Destruir tudo
.\scripts\destroy-all.ps1 -Force

# Converter planilha para CSV
python scripts/convert-xlsx-to-input.py "docs/20260729 Cancelados may-26.xlsx" output.csv --max 100
```

---

## Conta AWS
- Account: 761018874615
- Região: us-east-1
- Stack: ChurnPredictionStack
- State Machine: churn-prediction-pipeline
