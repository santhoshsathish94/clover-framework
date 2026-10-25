param([string]$Files = "", [string]$Cmd = "", [string]$Out = "")
# Copies the named files (comma separated) of this directory to ~/rustmig/planner/verify on the box, then runs $Cmd there; output goes to $Out (a local file) and is printed.
$d = Split-Path -Parent $MyInvocation.MyCommand.Path
ssh -o BatchMode=yes -o ConnectTimeout=20 gpu "mkdir -p ~/rustmig/planner/verify" 2>&1 | Out-Null
foreach ($f in ($Files -split ',')) { if ($f) { scp -q -o BatchMode=yes (Join-Path $d $f) "gpu:rustmig/planner/verify/$f" } }
$r = ssh -o BatchMode=yes -o ConnectTimeout=20 gpu "cd ~/rustmig/planner/verify; sed -i 's/\r$//' *.sh *.cpp 2>/dev/null; chmod +x *.sh; $Cmd" 2>&1
if ($Out) { $r | Out-File -Encoding ascii $Out }
$r
