param([string[]]$Files, [string]$Cmd, [string]$Out = "")
# Copies the named files of this directory to ~/rustmig/slotcache/verify on the box, then runs $Cmd there; output goes to $Out (a local file) and is printed.
$d = Split-Path -Parent $MyInvocation.MyCommand.Path
foreach ($f in $Files) { scp -q -o BatchMode=yes (Join-Path $d $f) "gpu:rustmig/slotcache/verify/$f" }
$r = ssh -o BatchMode=yes -o ConnectTimeout=20 gpu "cd ~/rustmig/slotcache/verify; sed -i 's/\r$//' *.sh *.cpp *.inc 2>/dev/null; $Cmd" 2>&1
if ($Out) { $r | Out-File -Encoding ascii $Out }
$r
