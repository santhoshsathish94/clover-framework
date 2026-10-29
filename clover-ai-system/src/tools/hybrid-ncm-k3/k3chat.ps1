param(
    [switch]$Reset,
    [switch]$Fast1Cand,
    [switch]$EnsureServerOnly,
    [switch]$BatchMode,
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$PromptArgs
)

$ErrorActionPreference = "Stop"
$LocalUrl = "http://127.0.0.1:8765"

function Ensure-K3TunnelAndServer {
    try {
        $h = Invoke-RestMethod -Uri "$LocalUrl/api/health" -Method Get -TimeoutSec 2
        if ($h.status -eq "ready") { return $h }
    } catch {}

    Write-Host "[Clover K3] Starting local SSH tunnel (127.0.0.1:8765 -> gpu:8765) and verifying GPU daemon..." -ForegroundColor Cyan
    # Ensure remote daemon is running on gpu
    & ssh -o BatchMode=yes -o ConnectTimeout=10 gpu 'if ! curl -s http://127.0.0.1:8765/api/health | grep -q ready; then cd ~/merge_lf && export NCM_REPO=/home/usergpu/ncm/repo NCM_ART=/home/usergpu/ncm/artifacts/tinystories PYTHONPATH=/home/usergpu/ncm/repo:/home/usergpu/plan2text:/home/usergpu/merge_lf:/home/usergpu/hybridagent && setsid nohup /home/usergpu/ncm/venv/bin/python k3_chat_server.py --port 8765 > /home/usergpu/merge_lf/k3_chat_server.log 2>&1 < /dev/null & for i in $(seq 1 30); do sleep 1; if curl -s http://127.0.0.1:8765/api/health | grep -q ready; then break; fi; done; fi' | Out-Null

    # Start background ssh port-forward on Windows if port 8765 is not listening locally
    $conn = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
    if (-not $conn) {
        Start-Process -FilePath "ssh" -ArgumentList "-N -L 8765:127.0.0.1:8765 -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes gpu" -WindowStyle Hidden
        for ($i = 0; $i -lt 20; $i++) {
            try {
                $h = Invoke-RestMethod -Uri "$LocalUrl/api/health" -Method Get -TimeoutSec 1
                if ($h.status -eq "ready") { return $h }
            } catch {}
        }
    }
    return Invoke-RestMethod -Uri "$LocalUrl/api/health" -Method Get -TimeoutSec 5
}

$health = Ensure-K3TunnelAndServer
if ($EnsureServerOnly) {
    Write-Host "[Clover K3] Server & tunnel ready at $LocalUrl (model=$($health.model), history=$($health.history_len) turns)" -ForegroundColor Green
    exit 0
}

if ($Reset) {
    $r = Invoke-RestMethod -Uri "$LocalUrl/api/reset" -Method Post -ContentType "application/json" -Body "{}"
    Write-Host "[Clover K3] Conversation memory cleared (history_len=$($r.history_len))." -ForegroundColor Yellow
    if (-not $PromptArgs) { exit 0 }
}

$cands = if ($Fast1Cand) { 1 } else { 8 }

function Send-OnePrompt([string]$text) {
    $body = @{ prompt = $text; cands = $cands; remember = $true } | ConvertTo-Json -Compress
    $r = Invoke-RestMethod -Uri "$LocalUrl/api/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
    $reply = if ($cands -eq 1) { $r.reply_1cand } else { $r.reply_mbr8 }
    $nTok  = if ($cands -eq 1) { $r.n_k3_tokens_1cand } else { $r.n_k3_tokens_mbr8 }
    $ncmMs = if ($cands -eq 1) { $r.ncm_1cand_ms } else { $r.ncm_mbr8_ms }
    $totMs = if ($cands -eq 1) { $r.total_1cand_ms } else { $r.total_mbr8_ms }
    $e2eTps = [math]::Round($nTok / ($totMs / 1000.0), 2)
    $ncmTps = [math]::Round($nTok / ($ncmMs / 1000.0), 1)
    $planStr = ($r.plans -join " || ")

    Write-Host "`n[Turn $($r.history_len)] " -NoNewline -ForegroundColor Cyan
    Write-Host $reply -ForegroundColor White
    Write-Host ("  +-- Total: {0:N2}s ({1} tok/s E2E) | NCM: {2} ms ({3} tok/s) | Tokens: {4} K3 tok | Latent: {5} steps" -f ($totMs/1000.0), $e2eTps, $ncmMs, $ncmTps, $nTok, $r.latent_steps) -ForegroundColor DarkGray
    Write-Host ("  +-- Plan : {0}" -f $planStr) -ForegroundColor DarkGray
}

if ($BatchMode -and $PromptArgs -and $PromptArgs.Count -gt 0) {
    $body = @{ prompts = $PromptArgs; cands = 1 } | ConvertTo-Json -Compress
    $res = Invoke-RestMethod -Uri "$LocalUrl/api/batch" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
    Write-Host "`n=== Clubbed Batch ($($res.n_prompts) prompts, $($res.n_windows) windows) ===" -ForegroundColor Cyan
    for ($i = 0; $i -lt $res.results.Count; $i++) {
        $item = $res.results[$i]
        Write-Host ("[{0}] {1} ({2} K3 tok)" -f ($i+1), $item.reply, $item.n_k3_tokens) -ForegroundColor White
    }
    Write-Host ("  +-- Total: {0}s ({1}s/prompt) | E2E: {2} tok/s | NCM: {3} ms ({4} tok/s)" -f $res.total_wall_s, $res.effective_s_per_prompt, $res.e2e_tok_per_s, $res.ncm_ms, $res.ncm_stage_tok_per_s) -ForegroundColor DarkGray
    exit 0
}

if ($PromptArgs -and $PromptArgs.Count -gt 0) {
    $p = ($PromptArgs -join " ").Trim()
    Send-OnePrompt $p
    exit 0
}

Write-Host "=========================================================================" -ForegroundColor DarkCyan
Write-Host " Clover Kimi K3 -- VS Code Interactive Chat (Zero-Reload Hot GPU Server)" -ForegroundColor Cyan
Write-Host " Endpoint : $LocalUrl (Web UI + OpenAI /v1/chat/completions)" -ForegroundColor Gray
Write-Host " Commands : type your prompt, 'reset' to clear memory, '1cand'/'mbr8' to" -ForegroundColor Gray
Write-Host "            switch decode mode, or 'exit' to quit." -ForegroundColor Gray
Write-Host "=========================================================================`n" -ForegroundColor DarkCyan

while ($true) {
    $modeLabel = if ($cands -eq 1) { "1-cand" } else { "MBR8" }
    Write-Host -NoNewline "K3 ($modeLabel)> " -ForegroundColor Green
    $line = [Console]::ReadLine()
    if ($null -eq $line) { break }
    $line = $line.Trim()
    if (-not $line) { continue }
    if ($line -in @("exit", "quit", "q")) { break }
    if ($line -in @("reset", "clear")) {
        $r = Invoke-RestMethod -Uri "$LocalUrl/api/reset" -Method Post -ContentType "application/json" -Body "{}"
        Write-Host "  [Memory cleared]" -ForegroundColor Yellow
        continue
    }
    if ($line -eq "1cand") { $cands = 1; Write-Host "  [Switched to 1-candidate fast mode]" -ForegroundColor Yellow; continue }
    if ($line -eq "mbr8")  { $cands = 8; Write-Host "  [Switched to MBR8 8-candidate mode]" -ForegroundColor Yellow; continue }
    Send-OnePrompt $line
    Write-Host ""
}
