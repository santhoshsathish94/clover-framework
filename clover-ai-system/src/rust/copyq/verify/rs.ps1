# Helper: copy files from this folder to the box verify dir, optionally run a local .sh there, write all output to a private file.
# usage: rs.ps1 -Out out.txt [-Put a.cpp,b.sh] [-Run script.sh]
param([Parameter(Mandatory=$true)][string]$Out, [string]$Put = '', [string]$Run = '')
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$files = @($Put -split '[,; ]' | Where-Object { $_ })
$o = Join-Path $here $Out
"" | Set-Content -Encoding ascii $o
foreach ($f in $files) {
    $r = scp -o BatchMode=yes -o ConnectTimeout=20 (Join-Path $here $f) "gpu:rustmig/copyq/verify/$f" 2>&1
    "put $f exit=$LASTEXITCODE $r" | Add-Content -Encoding ascii $o
    ssh -o BatchMode=yes -o ConnectTimeout=20 gpu "cd ~/rustmig/copyq/verify && sed -i 's/\r$//' '$f'" 2>&1 | Add-Content -Encoding ascii $o
}
if ($Run) {
    ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 gpu "cd ~/rustmig/copyq/verify && bash $Run" 2>&1 | Add-Content -Encoding ascii $o
    "run $Run exit=$LASTEXITCODE" | Add-Content -Encoding ascii $o
}
"ENDOFHELPER" | Add-Content -Encoding ascii $o
