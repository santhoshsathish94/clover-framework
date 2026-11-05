# Read-only inventory of Clover records on the laptop: names, counts, sizes, dates, no contents.
param([Parameter(Mandatory)][string]$B)  # the Studio's CloverAI folder
$ErrorActionPreference = 'SilentlyContinue'
"time " + (Get-Date -Format o)
"== top level"
Get-ChildItem $B -Directory | ForEach-Object {
  $f = Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue
  "{0,-28} files={1,7} MB={2,9:N1} newest={3}" -f $_.Name, $f.Count, (($f | Measure-Object Length -Sum).Sum / 1MB), (($f | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime)
}
$R = Join-Path $B 'clover-ai-studio'
"== clover-ai-studio tree (dirs, depth 3)"
Get-ChildItem $R -Directory -Recurse -Depth 2 | Where-Object { $_.FullName -notmatch '\\.git(\\|$)' } | ForEach-Object {
  $f = Get-ChildItem $_.FullName -File
  "{0,-70} files={1,6} KB={2,10:N0}" -f $_.FullName.Substring($R.Length), $f.Count, (($f | Measure-Object Length -Sum).Sum / 1KB)
}
"== clover-ai-studio root files"
Get-ChildItem $R -File | ForEach-Object { "{0,-40} {1,10} {2}" -f $_.Name, $_.Length, $_.LastWriteTime }
"== git (read-only)"
git -C $R rev-list --count HEAD
git -C $R log -1 --format='%ci' --reverse
git -C $R log --format='%ci' | Select-Object -Last 1
git -C $R log -1 --format='%ci'
"== extensions under clover-ai-studio"
Get-ChildItem $R -Recurse -File | Where-Object { $_.FullName -notmatch '\\.git\\' } | Group-Object Extension | Sort-Object Count -Descending | ForEach-Object { "{0,-10} {1,6}" -f $_.Name, $_.Count }
"== other record-like folders"
foreach ($d in 'logs','verify','memory','data\memory','config','workspaces','media','studio','tmp') {
  $p = Join-Path $B $d
  if (Test-Path $p) {
    $f = Get-ChildItem $p -Recurse -File
    "{0,-14} files={1,7} MB={2,9:N1}" -f $d, $f.Count, (($f | Measure-Object Length -Sum).Sum / 1MB)
    $f | Group-Object { $_.Directory.FullName.Substring($B.Length) } | Sort-Object Count -Descending | Select-Object -First 12 | ForEach-Object { "    {0,-60} {1,5}" -f $_.Name, $_.Count }
  }
}
"== studio server running?"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*server.py*' } | ForEach-Object { "proc " + $_.ProcessId + " started " + $_.CreationDate }
