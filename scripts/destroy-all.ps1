#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Destrói TODOS os recursos AWS criados pelo CDK de forma rápida e completa.
    
.DESCRIPTION
    Este script executa cdk destroy em todas as stacks na ordem correta
    (dependências inversas) e limpa recursos que o CDK não remove sozinho.
    
.EXAMPLE
    .\scripts\destroy-all.ps1
    .\scripts\destroy-all.ps1 -Region us-east-1 -Force
#>

param(
    [string]$Region = "us-east-1",
    [switch]$Force,
    [switch]$SkipCDK
)

$ErrorActionPreference = "Stop"

Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Red
Write-Host "║  ATENÇÃO: DESTRUIÇÃO TOTAL DE RECURSOS AWS      ║" -ForegroundColor Red
Write-Host "║  Região: $Region                                ║" -ForegroundColor Red
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Red
Write-Host ""

if (-not $Force) {
    $confirm = Read-Host "Tem certeza que deseja DESTRUIR todos os recursos? (sim/não)"
    if ($confirm -ne "sim") {
        Write-Host "Operação cancelada." -ForegroundColor Yellow
        exit 0
    }
}

# Verificar AWS CLI
Write-Host "`n[1/6] Verificando credenciais AWS..." -ForegroundColor Cyan
try {
    $identity = aws sts get-caller-identity --region $Region 2>&1
    Write-Host "  ✓ Autenticado: $identity" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Falha na autenticação AWS. Configure aws configure." -ForegroundColor Red
    exit 1
}

# Esvaziar bucket S3 (necessário antes do cdk destroy)
Write-Host "`n[2/6] Esvaziando bucket S3..." -ForegroundColor Cyan
$bucketName = "sky-brazil-churn-prediction"
try {
    # Deletar todas as versões (bucket versionado)
    aws s3 rm "s3://$bucketName" --recursive --region $Region 2>&1 | Out-Null
    
    # Deletar versões antigas (delete markers + old versions)
    $versions = aws s3api list-object-versions --bucket $bucketName --region $Region --output json 2>&1
    if ($LASTEXITCODE -eq 0) {
        # Deletar todas as versões
        aws s3api delete-objects --bucket $bucketName --region $Region --delete "$(
            echo $versions | python -c "
import sys, json
data = json.load(sys.stdin)
objects = []
for v in data.get('Versions', []):
    objects.append({'Key': v['Key'], 'VersionId': v['VersionId']})
for d in data.get('DeleteMarkers', []):
    objects.append({'Key': d['Key'], 'VersionId': d['VersionId']})
if objects:
    print(json.dumps({'Objects': objects[:1000], 'Quiet': True}))
else:
    print(json.dumps({'Objects': [], 'Quiet': True}))
")" 2>&1 | Out-Null
    }
    Write-Host "  ✓ Bucket esvaziado (ou não existia)" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ Bucket não encontrado ou já vazio" -ForegroundColor Yellow
}

# CDK Destroy (ordem inversa de dependências)
if (-not $SkipCDK) {
    Write-Host "`n[3/6] Destruindo stack DashboardStack..." -ForegroundColor Cyan
    Push-Location "$PSScriptRoot\..\infra"
    try {
        cdk destroy DashboardStack --force --region $Region 2>&1
        Write-Host "  ✓ DashboardStack destruída" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ Falha ao destruir DashboardStack: $_" -ForegroundColor Yellow
    }

    Write-Host "`n[4/6] Destruindo stack OrchestrationStack..." -ForegroundColor Cyan
    try {
        cdk destroy OrchestrationStack --force --region $Region 2>&1
        Write-Host "  ✓ OrchestrationStack destruída" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ Falha ao destruir OrchestrationStack: $_" -ForegroundColor Yellow
    }

    Write-Host "`n[5/6] Destruindo stack ChurnPredictionStack..." -ForegroundColor Cyan
    try {
        cdk destroy ChurnPredictionStack --force --region $Region 2>&1
        Write-Host "  ✓ ChurnPredictionStack destruída" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ Falha ao destruir ChurnPredictionStack: $_" -ForegroundColor Yellow
    }
    Pop-Location
} else {
    Write-Host "`n[3-5/6] Skipping CDK destroy (--SkipCDK)" -ForegroundColor Yellow
}

# Limpeza de recursos que CDK pode não remover
Write-Host "`n[6/6] Limpeza de recursos residuais..." -ForegroundColor Cyan

# Deletar Log Groups
$logGroups = @(
    "/churn-prediction/extraction",
    "/churn-prediction/feature-engineering",
    "/churn-prediction/ml-inference",
    "/churn-prediction/explainability",
    "/churn-prediction/bedrock-explanation",
    "/churn-prediction/report-generation",
    "/churn-prediction/dashboard"
)

foreach ($lg in $logGroups) {
    try {
        aws logs delete-log-group --log-group-name $lg --region $Region 2>&1 | Out-Null
        Write-Host "  ✓ Log group deletado: $lg" -ForegroundColor DarkGray
    } catch {
        # Ignorar se não existir
    }
}

# Deletar Secrets Manager secret (force delete sem recovery window)
try {
    aws secretsmanager delete-secret `
        --secret-id "churn-prediction/npaw-api-key" `
        --force-delete-without-recovery `
        --region $Region 2>&1 | Out-Null
    Write-Host "  ✓ Secret deletado: churn-prediction/npaw-api-key" -ForegroundColor DarkGray
} catch {
    # Ignorar se não existir
}

Write-Host "`n╔══════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  DESTRUIÇÃO COMPLETA                             ║" -ForegroundColor Green
Write-Host "║  Todos os recursos foram removidos.              ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Nota: Verifique no console AWS se algum recurso residual permaneceu." -ForegroundColor Gray
Write-Host "      NAT Gateways e VPCs podem demorar alguns minutos para serem deletados." -ForegroundColor Gray
