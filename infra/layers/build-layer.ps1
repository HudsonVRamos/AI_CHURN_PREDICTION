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
# Nota: boto3/botocore NÃO incluídos pois já vêm no runtime Lambda Python 3.11
pip install `
    pydantic `
    pyyaml `
    aiohttp `
    numpy `
    pandas `
    scikit-learn `
    --target $layerDir `
    --platform manylinux2014_x86_64 `
    --only-binary=:all: `
    --python-version 3.11 `
    --quiet 2>&1

Write-Host "Layer construída em: $layerDir" -ForegroundColor Green

# Limpar arquivos desnecessários para economizar espaço
Write-Host "Limpando arquivos desnecessários..." -ForegroundColor Cyan
Get-ChildItem -Recurse $layerDir -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse "$layerDir\*.dist-info" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse $layerDir -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
# Remover testes unitários do sklearn e pandas (NÃO remover tests do numpy - são módulos internos)
Get-ChildItem -Recurse "$layerDir\sklearn" -Directory -Filter "tests" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse "$layerDir\pandas" -Directory -Filter "tests" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
# Remover datasets do sklearn (dados pesados que não usamos)
if (Test-Path "$layerDir\sklearn\datasets\data") { Remove-Item -Recurse -Force "$layerDir\sklearn\datasets\data" }
if (Test-Path "$layerDir\sklearn\datasets\descr") { Remove-Item -Recurse -Force "$layerDir\sklearn\datasets\descr" }
if (Test-Path "$layerDir\sklearn\datasets\images") { Remove-Item -Recurse -Force "$layerDir\sklearn\datasets\images" }

$size = (Get-ChildItem -Recurse $layerDir | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "Tamanho: $([math]::Round($size, 1)) MB" -ForegroundColor Gray
