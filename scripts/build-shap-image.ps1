#!/usr/bin/env pwsh
# Build e push da imagem Docker para Lambda SHAP usando AWS CodeBuild
# Não precisa de Docker Desktop local!
#
# Pré-requisito: projeto CodeBuild criado (feito automaticamente pelo script)
#
# Uso:
#   .\scripts\build-shap-image.ps1

$ErrorActionPreference = "Stop"
$REGION = "us-east-1"
$ACCOUNT_ID = "761018874615"
$ECR_REPO = "churn-pipeline-shap"
$IMAGE_TAG = "latest"
$PROJECT_NAME = "churn-shap-image-builder"

Write-Host "🐳 Build da imagem SHAP Lambda via CodeBuild" -ForegroundColor Cyan
Write-Host "   (Sem Docker Desktop necessário!)" -ForegroundColor Gray

# 1. Criar repositório ECR se não existir
Write-Host "`n📦 Verificando repositório ECR..." -ForegroundColor Cyan
$repoExists = aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION 2>$null
if (-not $repoExists) {
    Write-Host "  Criando repositório ECR: $ECR_REPO"
    aws ecr create-repository --repository-name $ECR_REPO --region $REGION --image-scanning-configuration scanOnPush=true | Out-Null
    Write-Host "  ✅ Criado"
} else {
    Write-Host "  ✅ Já existe"
}

# 2. Criar zip do source para CodeBuild
Write-Host "`n📁 Preparando source..." -ForegroundColor Cyan
$tempDir = "$env:TEMP\shap-build-$(Get-Random)"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

# Copiar Dockerfile
Copy-Item "infra\docker\shap\Dockerfile" "$tempDir\Dockerfile"

# Copiar src/
Copy-Item -Recurse "src" "$tempDir\src"

# Criar buildspec.yml
$buildspec = @"
version: 0.2
phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
  build:
    commands:
      - echo Building Docker image...
      - docker build -t $ECR_REPO`:$IMAGE_TAG .
      - docker tag $ECR_REPO`:$IMAGE_TAG $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO`:$IMAGE_TAG
  post_build:
    commands:
      - echo Pushing Docker image...
      - docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO`:$IMAGE_TAG
      - echo Done!
"@
$buildspec | Out-File -FilePath "$tempDir\buildspec.yml" -Encoding utf8

# Criar zip
$zipPath = "$env:TEMP\shap-source.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath }
Compress-Archive -Path "$tempDir\*" -DestinationPath $zipPath
Write-Host "  ✅ Source zip: $zipPath ($([math]::Round((Get-Item $zipPath).Length / 1KB, 1)) KB)"

# 3. Upload zip para S3
$s3Key = "codebuild/shap-source.zip"
Write-Host "`n☁️  Upload source para S3..." -ForegroundColor Cyan
aws s3 cp $zipPath "s3://sky-brazil-churn-prediction/$s3Key" --region $REGION | Out-Null
Write-Host "  ✅ s3://sky-brazil-churn-prediction/$s3Key"

# 4. Criar/atualizar projeto CodeBuild
Write-Host "`n🔨 Configurando CodeBuild project..." -ForegroundColor Cyan
$projectExists = aws codebuild batch-get-projects --names $PROJECT_NAME --region $REGION --query "projects[0].name" --output text 2>$null

$projectConfig = @"
{
    "name": "$PROJECT_NAME",
    "source": {
        "type": "S3",
        "location": "sky-brazil-churn-prediction/$s3Key"
    },
    "artifacts": {
        "type": "NO_ARTIFACTS"
    },
    "environment": {
        "type": "LINUX_CONTAINER",
        "image": "aws/codebuild/standard:7.0",
        "computeType": "BUILD_GENERAL1_MEDIUM",
        "privilegedMode": true
    },
    "serviceRole": "arn:aws:iam::${ACCOUNT_ID}:role/ChurnPredictionStack-CodeBuildRole"
}
"@

if ($projectExists -eq $PROJECT_NAME) {
    Write-Host "  Projeto já existe, atualizando..."
    $projectConfig | aws codebuild update-project --cli-input-json file:///dev/stdin --region $REGION 2>$null | Out-Null
} else {
    Write-Host "  Criando projeto CodeBuild..."
    # Criar role para CodeBuild se não existir
    $roleArn = "arn:aws:iam::${ACCOUNT_ID}:role/ChurnPredictionStack-CodeBuildRole"
    
    $trustPolicy = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"codebuild.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    aws iam create-role --role-name "ChurnPredictionStack-CodeBuildRole" --assume-role-policy-document $trustPolicy --region $REGION 2>$null
    aws iam attach-role-policy --role-name "ChurnPredictionStack-CodeBuildRole" --policy-arn "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser" 2>$null
    aws iam attach-role-policy --role-name "ChurnPredictionStack-CodeBuildRole" --policy-arn "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess" 2>$null
    aws iam attach-role-policy --role-name "ChurnPredictionStack-CodeBuildRole" --policy-arn "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess" 2>$null
    
    Start-Sleep -Seconds 10  # Esperar propagação da role

    aws codebuild create-project `
        --name $PROJECT_NAME `
        --source "type=S3,location=sky-brazil-churn-prediction/$s3Key" `
        --artifacts "type=NO_ARTIFACTS" `
        --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:7.0,computeType=BUILD_GENERAL1_MEDIUM,privilegedMode=true" `
        --service-role $roleArn `
        --region $REGION | Out-Null
}
Write-Host "  ✅ Projeto configurado"

# 5. Iniciar build
Write-Host "`n🚀 Iniciando build..." -ForegroundColor Cyan
$buildId = aws codebuild start-build --project-name $PROJECT_NAME --region $REGION --query "build.id" --output text
Write-Host "  Build ID: $buildId"

# 6. Aguardar conclusão
Write-Host "`n⏳ Aguardando build (pode levar 3-5 min)..." -ForegroundColor Cyan
while ($true) {
    $status = aws codebuild batch-get-builds --ids $buildId --region $REGION --query "builds[0].buildStatus" --output text
    if ($status -eq "SUCCEEDED") {
        Write-Host "  ✅ Build SUCCEEDED!" -ForegroundColor Green
        break
    }
    if ($status -eq "FAILED" -or $status -eq "FAULT" -or $status -eq "STOPPED") {
        Write-Host "  ❌ Build $status" -ForegroundColor Red
        $logs = aws codebuild batch-get-builds --ids $buildId --region $REGION --query "builds[0].logs.deepLink" --output text
        Write-Host "  Logs: $logs"
        exit 1
    }
    Write-Host "  Status: $status..." -NoNewline
    Start-Sleep -Seconds 15
    Write-Host ""
}

# 7. Atualizar Lambda para usar a nova imagem
$imageUri = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
Write-Host "`n🔄 Atualizando Lambda churn-pipeline-shap..." -ForegroundColor Cyan
aws lambda update-function-code `
    --function-name "churn-pipeline-shap" `
    --image-uri $imageUri `
    --region $REGION | Out-Null
Write-Host "  ✅ Lambda atualizada: $imageUri"

# Cleanup
Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
Remove-Item $zipPath -ErrorAction SilentlyContinue

Write-Host "`n🏁 Concluído! Lambda SHAP deployada com imagem Docker." -ForegroundColor Green
Write-Host "   Image: $imageUri" -ForegroundColor Gray
