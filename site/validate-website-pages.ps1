$ErrorActionPreference = 'Stop'
$checker = Join-Path (Split-Path $PSScriptRoot -Parent) 'scripts/check-site.py'
& python $checker
exit $LASTEXITCODE
