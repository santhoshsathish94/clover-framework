# Builds CloverAIStudio.exe and CloverAIStudio-Setup.exe as clean native .NET Framework 4.8 x64 GUI executables
# using Microsoft's built-in csc.exe compiler (no IExpress / wextract.exe wrapper and no .ps1 scripts).
$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
$base = Split-Path -Parent $src
$build = Join-Path $env:TEMP 'clover-setup-build'
$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path $csc)) { throw "csc.exe not found at $csc" }

$fw = [IO.Path]::GetFullPath((Join-Path $base '..\..\..\clover-collective-wt'))
if (-not (Test-Path (Join-Path $fw 'clover-ai-system\mobile'))) {
    $fw = [IO.Path]::GetFullPath((Join-Path $base '..\..\..\clover-framework'))
}
if (Test-Path (Join-Path $fw 'clover-ai-system\mobile')) {
    foreach ($f in 'semantics.py', 'personal.py', 'collective.py', 'collective_gateway.py', 'collective_evaluator.py', 'core_registry.py', 'core_sync.py', 'test_collective.py') {
        $p = Join-Path $fw "clover-ai-system\mobile\$f"
        if (Test-Path $p) { Copy-Item $p (Join-Path $base "clover\$f") -Force }
    }
    $proto = Join-Path $fw 'research\semantic-interface\prototype'
    if (Test-Path (Join-Path $proto 'csi_service.py')) {
        Copy-Item (Join-Path $proto 'csi_service.py') (Join-Path $base 'clover\csi_service.py') -Force
        New-Item -ItemType Directory -Force (Join-Path $base 'clover\csi') | Out-Null
        Copy-Item (Join-Path $proto 'csi\*.py') (Join-Path $base 'clover\csi') -Force
    }
    New-Item -ItemType Directory -Force (Join-Path $base 'clover\fused') | Out-Null
    & git -C $fw archive --format=zip -o (Join-Path $base 'clover\fused\framework.zip') HEAD `
        AGENTS.md `
        clover-ai-system/mobile/s23_on_device_clover.py `
        clover-ai-system/mobile/lfm3_cpu_engine.c `
        clover-ai-system/mobile/clover_cycle.py `
        clover-ai-system/mobile/clover_git_memory_repo.py `
        clover-ai-system/mobile/local_terminal.py `
        clover-ai-system/mobile/memory_index.py `
        clover-ai-system/mobile/mobile_vault.py `
        clover-ai-system/mobile/personal.py `
        clover-ai-system/mobile/collective.py `
        clover-ai-system/mobile/collective_gateway.py `
        clover-ai-system/mobile/collective_evaluator.py `
        clover-ai-system/mobile/core_registry.py `
        clover-ai-system/mobile/core_sync.py `
        clover-ai-system/mobile/semantics.py `
        clover-ai-system/mobile/web_search.py `
        clover-ai-system/src/models/lfm3-1.7b `
        clover-ai-system/src/tools/hybrid-ncm-k3/clover_hyphen_runtime.py `
        clover-ai-system/src/tools/hybrid-ncm-k3/k3_memory_graph.py `
        clover-ai-system/src/tools/hybrid-ncm-k3/k3_ontology_graph.py `
        clover-ai-system/src/tools/hybrid-ncm-k3/k3_generative_compiler.py `
        clover-ai-system/src/tools/hybrid-ncm-k3/memory-knowledge-graph.py `
        clover-ai-system/src/tools/hybrid-ncm-k3/symbolic-ontology-graph.py `
        clover-ai-system/src/tools/hybrid-ncm-k3/generative-code-compiler.py `
        clover-ai-system/src/tools/hybrid-ncm-k3/live-web-search-tool.py `
        clover-ai-system/src/tools/hybrid-ncm-k3/live_web_search.py `
        clover-ai-system/src/tools/hybrid-ncm-k3/memory_tree
}

if (Test-Path $build) { Remove-Item $build -Recurse -Force }
New-Item -ItemType Directory -Force $build | Out-Null

$ico = Join-Path $src 'clover.ico'
$cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert -ErrorAction SilentlyContinue |
        Where-Object { $_.Subject -like '*Clover*' -and $_.NotAfter -gt (Get-Date) } |
        Select-Object -First 1
if (-not $cert) {
    $cert = New-SelfSignedCertificate `
        -Type CodeSigningCert `
        -Subject "CN=Clover Framework, O=Clover Framework" `
        -KeyAlgorithm RSA `
        -KeyLength 2048 `
        -HashAlgorithm SHA256 `
        -CertStoreLocation "Cert:\CurrentUser\My" `
        -NotAfter (Get-Date).AddYears(5)
}
$cerPath = Join-Path $src 'clover-codesign.cer'
Export-Certificate -Cert $cert -FilePath $cerPath -Force | Out-Null
Import-Certificate -FilePath $cerPath -CertStoreLocation "Cert:\CurrentUser\TrustedPublisher" | Out-Null

$launcherExe = Join-Path $build 'CloverAIStudio.exe'
& $csc /nologo /target:winexe /platform:x64 /optimize+ "/win32icon:$ico" `
    /r:System.Windows.Forms.dll,System.Drawing.dll `
    "/out:$launcherExe" (Join-Path $src 'Launcher.cs')
if ($LASTEXITCODE -or -not (Test-Path $launcherExe)) { throw "csc Launcher.cs failed ($LASTEXITCODE)" }
Set-AuthenticodeSignature -FilePath $launcherExe -Certificate $cert -HashAlgorithm SHA256 | Out-Null
$exe = Get-Item $launcherExe
"EXE $($exe.Length) bytes"

$stage = Join-Path $build 'app'
New-Item -ItemType Directory -Force (Join-Path $stage 'studio') | Out-Null
Copy-Item $exe.FullName $stage
& robocopy (Join-Path $base 'clover') (Join-Path $stage 'clover') /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
Copy-Item (Join-Path $src 'clover.ico'), (Join-Path $src 'requirements.lock') (Join-Path $stage 'studio')
$vsix = Get-ChildItem (Join-Path $src 'dist') -Filter 'clover-local-*.vsix' |
        Sort-Object { [version]($_.BaseName -replace '^clover-local-', '') } -Descending | Select-Object -First 1
if (-not $vsix) { throw 'no VS Code extension package in studio\dist' }
Copy-Item $vsix.FullName (Join-Path $stage 'studio')

$pay = Join-Path $build 'payload'
New-Item -ItemType Directory -Force $pay | Out-Null
$appZip = Join-Path $pay 'app.zip'
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($stage, $appZip, [IO.Compression.CompressionLevel]::Optimal, $false)
"PAYLOAD " + ((Get-ChildItem $stage -Recurse -File).Count) + " files, app.zip " + (Get-Item $appZip).Length + " bytes"

$outDir = Join-Path $base 'setup'
New-Item -ItemType Directory -Force $outDir | Out-Null
$target = Join-Path $outDir 'CloverAIStudio-Setup.exe'
if (Test-Path $target) { Remove-Item $target -Force }
& $csc /nologo /target:winexe /platform:x64 /optimize+ "/win32icon:$ico" `
    /r:System.Windows.Forms.dll,System.Drawing.dll,System.IO.Compression.dll,System.IO.Compression.FileSystem.dll,System.Web.Extensions.dll `
    "/resource:$appZip,CloverAIStudio.app.zip" `
    "/out:$target" (Join-Path $src 'Setup.cs')
if ($LASTEXITCODE -or -not (Test-Path $target)) { throw "csc Setup.cs failed ($LASTEXITCODE)" }
Set-AuthenticodeSignature -FilePath $target -Certificate $cert -HashAlgorithm SHA256 | Out-Null
if ((Test-Path (Join-Path $fw 'download\laptop')) -and ($fw -like '*clover-collective-wt*')) {
    Copy-Item $target (Join-Path $fw 'download\laptop\CloverAIStudio-Setup.exe') -Force
    Copy-Item $cerPath (Join-Path $fw 'download\laptop\CloverFramework-CodeSign.cer') -Force
}
"SETUP $target " + (Get-Item $target).Length + " bytes"
