param([string]$Cmd = "", [string]$Out = "", [switch]$NoCopy)
# Copies the crate (Cargo.toml, smoke_rust.c, src/, include/, ref/, verify/ - not target/ and not verify/out) to ~/rustmig/supply on the box, strips CRs from the text files, then runs $Cmd there;
# output goes to $Out (a local file, absolute path) and is printed. Run it from your own terminal: the shared one is used by other agents.
$d = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$so = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20')
if (-not $NoCopy) {
  scp -q @so "$d\Cargo.toml" "$d\smoke_rust.c" "gpu:rustmig/supply/"
  foreach ($sub in 'src', 'include', 'ref', 'verify') { scp -q -r @so "$d\$sub" "gpu:rustmig/supply/" }
}
$strip = "find . -path ./target -prune -o -path ./verify/out -prune -o -type f \( -name '*.sh' -o -name '*.rs' -o -name '*.c' -o -name '*.cpp' -o -name '*.h' -o -name '*.toml' -o -name '*.inc' \) -print0 | xargs -0 sed -i 's/\r$//'"
$r = ssh @so gpu "cd ~/rustmig/supply; $strip; chmod +x ref/*.sh verify/*.sh 2>/dev/null; $Cmd" 2>&1
if ($Out) { $r | Out-File -Encoding ascii $Out }
$r
