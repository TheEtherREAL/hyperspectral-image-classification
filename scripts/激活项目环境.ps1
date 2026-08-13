$projectRoot = Split-Path -Parent $PSScriptRoot
$venvActivate = Join-Path $projectRoot '.venv\Scripts\Activate.ps1'

if (-not (Test-Path -LiteralPath $venvActivate)) {
    throw "Project environment not found: $venvActivate"
}

& $venvActivate
Set-Location -LiteralPath $projectRoot
Write-Host "高光谱项目环境已激活 / HSI environment activated: $projectRoot"
