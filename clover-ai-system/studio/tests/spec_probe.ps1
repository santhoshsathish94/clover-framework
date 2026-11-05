# Can a drafter speed up LFM3 writing on the Iris Xe? Two facts that do not depend on which drafter:
#  1. how long LFM3 takes to check a block of N drafted tokens at once, against writing them one at a time;
#  2. whether llama-server's speculative decoding runs at all on LFM3's hybrid (convolution + attention) memory.
# LFM3 drafts for itself in part 2, so no speed-up is expected there; it only tests that the path works.
# Windows PowerShell turns a native program's stderr into errors, so "Stop" would end the script at llama-bench's banner.
$ErrorActionPreference = "Continue"
$b = Join-Path $env:USERPROFILE "CloverAI"
$vk = Join-Path $b "tools\llama.cpp\vulkan"
$m = Join-Path $b "data\lfm3\gguf\LFM3-1.7B-QAD-Q4_0.gguf"
$out = Join-Path $b "verify\spec-probe"
New-Item -ItemType Directory -Force $out | Out-Null
"== 1. llama-bench, Iris Xe, Q4_0: one batch of N tokens (pp N) and writing (tg 64)"
& (Join-Path $vk "llama-bench.exe") -m $m -ngl 99 -p "1,2,4,8,16,32" -n 0 -r 3 -o md 2>$null
& (Join-Path $vk "llama-bench.exe") -m $m -ngl 99 -p 0 -n 64 -r 2 -o md 2>$null

"== 2. llama-server with a draft model (LFM3 drafting for itself)"
$key = [guid]::NewGuid().ToString("N")
$env:LLAMA_API_KEY = $key
$log = Join-Path $out "server.log"
$srvArgs = @("-m", $m, "-md", $m, "-ngl", "99", "-ngld", "99", "--draft-max", "8", "--draft-min", "1",
             "--host", "127.0.0.1", "--port", "8899", "-c", "4096", "-np", "1", "--no-webui")
$p = Start-Process -FilePath (Join-Path $vk "llama-server.exe") -ArgumentList $srvArgs -PassThru -WindowStyle Hidden `
     -RedirectStandardError $log -RedirectStandardOutput (Join-Path $out "server.out")
try {
    $ok = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Seconds 1
        if ($p.HasExited) { break }
        try { if ((Invoke-RestMethod -Uri "http://127.0.0.1:8899/health" -TimeoutSec 2).status -eq "ok") { $ok = $true; break } } catch { }
    }
    "server up: $ok  exited: $($p.HasExited)"
    if ($ok) {
        $body = @{ prompt = "Explain in a few sentences why the sky is blue."; n_predict = 96; temperature = 0; cache_prompt = $false } | ConvertTo-Json
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8899/completion" -Method Post -ContentType "application/json" -Body $body `
             -Headers @{ Authorization = "Bearer $key" } -TimeoutSec 300
        "timings: " + ($r.timings | ConvertTo-Json -Compress)
        "text: " + $r.content.Substring(0, [Math]::Min(160, $r.content.Length)).Replace("`n", " ")
    }
} finally {
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue
}
"== server log lines about drafting"
Select-String -Path $log -Pattern "draft|specul|recurrent|hybrid|not supported|error|failed" | Select-Object -First 30 | ForEach-Object { $_.Line }
