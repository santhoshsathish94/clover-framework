# Moves the datasets the human chose to delete (2026-11-01) into CloverAI\_removed, restarts the Studio on the
# Omni-only code and asks one question through the real chat API. Deleting _removed is a separate step, after this passes.
$ErrorActionPreference = 'Stop'
$b = Join-Path $env:USERPROFILE 'CloverAI'
$hold = Join-Path $b '_removed'
New-Item -ItemType Directory -Force $hold | Out-Null
foreach ($rel in 'data\hybridagent-8b', 'data\lfm3', 'data\ncm', 'data\omni\gguf\LFM3-Omni-2B-Q4_K_M.gguf') {
    $p = Join-Path $b $rel
    if (Test-Path $p) { Move-Item $p (Join-Path $hold ($rel -replace '\\', '_')); "moved $rel" } else { "absent $rel" }
}
& (Join-Path $b 'tests\restart_studio.ps1')
$url = 'http://127.0.0.1:8765'
$h = @{ Origin = $url }
$sid = 'removal-check-' + (Get-Date -Format 'HHmmss')
$body = @{ prompt = 'What is 17 times 23? Answer with the number only.'; session_id = $sid } | ConvertTo-Json
$r = Invoke-RestMethod -Method Post -TimeoutSec 180 -ContentType 'application/json' -Headers $h -Body $body "$url/api/chat"
"reply: " + ([string]$r.reply).Trim()
"model: " + (Invoke-RestMethod -TimeoutSec 5 "$url/api/status?passive=1").model
$st = Invoke-RestMethod -TimeoutSec 5 "$url/api/settings"
"settings model: $($st.settings.model), gpu_available: $($st.gpu_available), image_model: $($st.settings.image_model)"
Invoke-RestMethod -Method Post -TimeoutSec 10 -ContentType 'application/json' -Headers $h -Body (@{ session_id = $sid; delete = $true } | ConvertTo-Json) "$url/api/session/forget" | Out-Null
"test chat deleted"
Get-ChildItem $hold | ForEach-Object { $s = if ($_.PSIsContainer) { (Get-ChildItem $_.FullName -Recurse -File -Force | Measure-Object Length -Sum).Sum } else { $_.Length }; "{0,-40} {1,8:N0} MB" -f $_.Name, ($s / 1MB) }
