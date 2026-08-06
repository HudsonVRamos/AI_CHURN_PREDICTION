#!/usr/bin/env pwsh
# Build da Lambda Layer com dependências Python
# Gera: infra/layers/dependencies/python/

$ErrorActionPreference = "Stop"

$layerDir = "$PSScriptRoot/dependencies/python"

Write-Host "Limpando diretório anterior..." -ForegroundColor Cyan
if (Test-Path "$PSScriptRoot/dependencies") {
    Remove-Item -Recurse -Force "$PSScriptRoot/dependencies"
}
New-Item -ItemType Directory -Path $layerDir -Force | Out-Null

Write-Host "Instalando dependências na layer..." -ForegroundColor Cyan
pip install `
    pydantic `
    pyyaml `
    aiohttp `
    numpy `
    pandas `
    boto3 `
    --target $layerDir `
    --platform manylinux2014_x86_64 `
    --only-binary=:all: `
    --python-version 3.11 `
    --quiet 2>&1

Write-Host "Layer construída em: $layerDir" -ForegroundColor Green
$size = (Get-ChildItem -Recurse $layerDir | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "Tamanho: $([math]::Round($size, 1)) MB" -ForegroundColor Gray
