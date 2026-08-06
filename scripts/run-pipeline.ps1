#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Inicia o pipeline de churn prediction com user IDs e período configuráveis.

.DESCRIPTION
    Edite as variáveis abaixo e execute. O script inicia o Step Functions
    e mostra o progresso.

.EXAMPLE
    .\scripts\run-pipeline.ps1
#>

# ╔══════════════════════════════════════════════════════════════╗
# ║  EDITE AQUI                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

# Período de extração (formato: YYYY-MM-DD)
$FROM_DATE = "2024-01-01"
$TO_DATE   = "2024-12-31"

# Modo: "predict" (inferência) ou "train" (treinamento)
$MODE = "predict"

# User IDs para analisar (um por linha)
$USER_IDS = @(
    "9e527027-b330-5d8f-aaa2-bf6653bd6eec"
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  NÃO EDITE ABAIXO                                           ║
# ╚══════════════════════════════════════════════════════════════╝

$REGION = "us-east-1"
$STATE_MACHINE_ARN = "arn:aws:states:us-east-1:761018874615:stateMachine:churn-prediction-pipeline"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  AI CHURN PREDICTION - Sky Brazil                ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Modo:       $MODE"
Write-Host "  Período:    $FROM_DATE → $TO_DATE"
Write-Host "  Usuários:   $($USER_IDS.Count)"
Write-Host ""

# Montar input JSON
$inputObj = @{
    mode      = $MODE
    source    = $USER_IDS
    from_date = $FROM_DATE
    to_date   = $TO_DATE
}
$inputJson = $inputObj | ConvertTo-Json -Compress

Write-Host "[1/3] Iniciando pipeline..." -ForegroundColor Yellow
$executionResult = aws stepfunctions start-execution `
    --state-machine-arn $STATE_MACHINE_ARN `
    --input $inputJson `
    --region $REGION 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Erro ao iniciar: $executionResult" -ForegroundColor Red
    exit 1
}

$execution = $executionResult | ConvertFrom-Json
$executionArn = $execution.executionArn
$executionName = $executionArn.Split(":")[-1]

Write-Host "  ✓ Execução iniciada: $executionName" -ForegroundColor Green
Write-Host ""

Write-Host "[2/3] Acompanhando execução..." -ForegroundColor Yellow
Write-Host "  ARN: $executionArn"
Write-Host ""

# Polling do status
$maxWait = 600  # 10 minutos
$elapsed = 0
$interval = 10

while ($elapsed -lt $maxWait) {
    $statusResult = aws stepfunctions describe-execution `
        --execution-arn $executionArn `
        --region $REGION 2>&1 | ConvertFrom-Json

    $status = $statusResult.status
    $elapsed += $interval

    switch ($status) {
        "RUNNING" {
            Write-Host "  ⏳ Status: RUNNING ($($elapsed)s)" -ForegroundColor DarkGray
            Start-Sleep -Seconds $interval
        }
        "SUCCEEDED" {
            Write-Host ""
            Write-Host "  ✓ Pipeline concluído com SUCESSO!" -ForegroundColor Green
            Write-Host ""
            Write-Host "[3/3] Resultado:" -ForegroundColor Yellow
            
            $output = $statusResult.output | ConvertFrom-Json
            Write-Host "  Usuários processados: $($output.users_count)" -ForegroundColor White
            Write-Host "  Predições geradas:    $($output.predictions_count)" -ForegroundColor White
            
            if ($output.report_s3_paths) {
                Write-Host ""
                Write-Host "  Relatórios:" -ForegroundColor Cyan
                $output.report_s3_paths.PSObject.Properties | ForEach-Object {
                    Write-Host "    $($_.Name): $($_.Value)" -ForegroundColor DarkCyan
                }
            }
            
            Write-Host ""
            Write-Host "  Tempo total: $($elapsed)s"
            exit 0
        }
        "FAILED" {
            Write-Host ""
            Write-Host "  ✗ Pipeline FALHOU!" -ForegroundColor Red
            Write-Host "  Erro: $($statusResult.error)" -ForegroundColor Red
            Write-Host "  Causa: $($statusResult.cause)" -ForegroundColor Red
            Write-Host ""
            Write-Host "  Para ver logs detalhados:" -ForegroundColor Yellow
            Write-Host "  aws logs tail /aws/lambda/churn-pipeline-ingest --since 5m --region $REGION"
            exit 1
        }
        "TIMED_OUT" {
            Write-Host ""
            Write-Host "  ✗ Pipeline TIMEOUT!" -ForegroundColor Red
            exit 1
        }
        "ABORTED" {
            Write-Host ""
            Write-Host "  ⚠ Pipeline ABORTADO" -ForegroundColor Yellow
            exit 1
        }
    }
}

Write-Host ""
Write-Host "  ⚠ Timeout aguardando conclusão (${maxWait}s). Pipeline ainda pode estar rodando." -ForegroundColor Yellow
Write-Host "  Verifique: aws stepfunctions describe-execution --execution-arn $executionArn --region $REGION"
