# Speculative decoding on the Iris Xe with llama.cpp b12012, measured back to back on the same prompts (greedy).
# Under greedy decoding every configuration must produce the same text as "none"; the script checks that.
#   none          plain LFM3
#   ngram-mod     drafts from text already in the conversation (no extra model)
#   ngram-simple  same idea, simpler lookup
#   draft-simple  a draft model; LFM3 drafts for itself here, only to see whether the path runs on its hybrid memory
$ErrorActionPreference = "Continue"
$b = Join-Path $env:USERPROFILE "CloverAI"
$vk = Join-Path $b "tools\llama.cpp\vulkan"
$m = Join-Path $b "data\lfm3\gguf\LFM3-1.7B-QAD-Q4_0.gguf"
$out = Join-Path $b "verify\spec-probe"
New-Item -ItemType Directory -Force $out | Out-Null

$code = @'
def summarize(orders):
    x = 0
    count = 0
    for order in orders:
        if order["status"] == "paid":
            x += order["amount"]
            count += 1
    average = x / count if count else 0
    return {"total": x, "count": count, "average": average}
'@
$prompts = @(
    @{ name = "rewrite"; text = "Rewrite this Python function with the variable x renamed to total. Reply with only the code.`n`n$code" },
    @{ name = "explain"; text = "In four sentences, explain why the sky is blue." }
)
$configs = @(
    @{ name = "none"; args = @() },
    @{ name = "ngram-mod"; args = @("--spec-type", "ngram-mod") },
    @{ name = "ngram-simple"; args = @("--spec-type", "ngram-simple") },
    @{ name = "draft-simple"; args = @("--spec-type", "draft-simple", "-md", $m, "-ngld", "99", "--spec-draft-n-max", "8") }
)
$reference = @{}
foreach ($c in $configs) {
    $key = [guid]::NewGuid().ToString("N")
    $env:LLAMA_API_KEY = $key
    $log = Join-Path $out ("server-" + $c.name + ".log")
    $srvArgs = @("-m", $m, "-ngl", "99", "--host", "127.0.0.1", "--port", "8899", "-c", "4096", "-np", "1", "--no-webui", "-ub", "256") + $c.args
    $p = Start-Process -FilePath (Join-Path $vk "llama-server.exe") -ArgumentList $srvArgs -PassThru -WindowStyle Hidden `
         -RedirectStandardError $log -RedirectStandardOutput (Join-Path $out ("server-" + $c.name + ".out"))
    try {
        $ok = $false
        for ($i = 0; $i -lt 120; $i++) {
            Start-Sleep -Seconds 1
            if ($p.HasExited) { break }
            try { if ((Invoke-RestMethod -Uri "http://127.0.0.1:8899/health" -TimeoutSec 2).status -eq "ok") { $ok = $true; break } } catch { }
        }
        if (-not $ok) {
            "{0,-13} did not start (exited {1}): {2}" -f $c.name, $p.HasExited, ((Get-Content $log -Tail 3) -join " | ")
            continue
        }
        $h = @{ Authorization = "Bearer $key" }
        foreach ($q in $prompts) {
            $tpl = Invoke-RestMethod -Uri "http://127.0.0.1:8899/apply-template" -Method Post -ContentType "application/json" -Headers $h `
                   -Body (@{ messages = @(@{ role = "user"; content = $q.text }) } | ConvertTo-Json -Depth 5)
            $body = @{ prompt = $tpl.prompt; n_predict = 320; temperature = 0; cache_prompt = $false } | ConvertTo-Json
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:8899/completion" -Method Post -ContentType "application/json" -Headers $h `
                 -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 600
            $t = $r.timings
            $rk = $q.name
            if ($c.name -eq "none") { $reference[$rk] = $r.content }
            $same = if ($reference.ContainsKey($rk)) { $reference[$rk] -eq $r.content } else { "n/a" }
            "{0,-13} {1,-8} write {2,6:N1} tok/s  tokens {3,4}  drafted {4,4}  accepted {5,4}  same-as-none {6}" -f `
                $c.name, $q.name, $t.predicted_per_second, $t.predicted_n, $t.draft_n, $t.draft_n_accepted, $same
        }
    } finally {
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
        Remove-Item Env:LLAMA_API_KEY -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}
