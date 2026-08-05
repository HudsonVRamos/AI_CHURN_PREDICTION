#!/bin/bash
#
# Destrói TODOS os recursos AWS criados pelo CDK.
# Uso: ./scripts/destroy-all.sh [--force] [--region us-east-1]
#

set -e

REGION="${REGION:-us-east-1}"
FORCE=false

for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
        --region) shift; REGION="$1" ;;
    esac
done

echo "╔══════════════════════════════════════════════════╗"
echo "║  ATENÇÃO: DESTRUIÇÃO TOTAL DE RECURSOS AWS      ║"
echo "║  Região: $REGION                                ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

if [ "$FORCE" != "true" ]; then
    read -p "Tem certeza que deseja DESTRUIR todos os recursos? (sim/não): " confirm
    if [ "$confirm" != "sim" ]; then
        echo "Operação cancelada."
        exit 0
    fi
fi

echo "[1/6] Verificando credenciais AWS..."
aws sts get-caller-identity --region "$REGION" || { echo "Falha na autenticação"; exit 1; }

echo "[2/6] Esvaziando bucket S3..."
aws s3 rm "s3://sky-brazil-churn-prediction" --recursive --region "$REGION" 2>/dev/null || true

echo "[3/6] Destruindo DashboardStack..."
cd infra && cdk destroy DashboardStack --force 2>/dev/null || true

echo "[4/6] Destruindo OrchestrationStack..."
cdk destroy OrchestrationStack --force 2>/dev/null || true

echo "[5/6] Destruindo ChurnPredictionStack..."
cdk destroy ChurnPredictionStack --force 2>/dev/null || true
cd ..

echo "[6/6] Limpeza de recursos residuais..."
for lg in extraction feature-engineering ml-inference explainability bedrock-explanation report-generation dashboard; do
    aws logs delete-log-group --log-group-name "/churn-prediction/$lg" --region "$REGION" 2>/dev/null || true
done

aws secretsmanager delete-secret \
    --secret-id "churn-prediction/npaw-api-key" \
    --force-delete-without-recovery \
    --region "$REGION" 2>/dev/null || true

echo ""
echo "✓ Destruição completa. Verifique o console AWS para recursos residuais."
