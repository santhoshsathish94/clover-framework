# Copies the crate (sources, header, ref, verify) from the Windows tree to ~/rustmig/loader on the box (ssh alias gpu) with ONE tar and ONE ssh connection, then strips CRs from
# scripts and C/C++/python sources there (the .inc files are extracted untouched; md5 compared with the local files). Usage: pwsh verify/sync.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$files = @('Cargo.toml', 'smoke_rust.c', 'smoke_rust.sh', 'include/loader.h') +
    (Get-ChildItem "$root\src" -File | ForEach-Object { "src/$($_.Name)" }) +
    (Get-ChildItem "$root\ref" -File | Where-Object { $_.Extension -in '.cpp', '.c', '.sh', '.py', '.inc' } | ForEach-Object { "ref/$($_.Name)" }) +
    (Get-ChildItem "$root\verify" -File | Where-Object { $_.Extension -in '.cpp', '.c', '.sh', '.py', '.inc', '.h' } | ForEach-Object { "verify/$($_.Name)" }) +
    (Get-ChildItem "$root\verify\mutants" -File | Where-Object { $_.Extension -in '.c', '.sh', '.py' } | ForEach-Object { "verify/mutants/$($_.Name)" })
$tar = Join-Path $env:TEMP 'loader-sync.tar'
if (Test-Path $tar) { Remove-Item $tar }
Push-Location $root
tar -cf $tar @files
Pop-Location
scp -q $tar gpu:/tmp/loader-sync.tar
ssh gpu 'mkdir -p ~/rustmig/loader && cd ~/rustmig/loader && tar -xf /tmp/loader-sync.tar && rm -f /tmp/loader-sync.tar && sed -i "s/\r$//" verify/*.sh verify/*.c verify/*.cpp verify/*.py verify/mutants/*.sh verify/mutants/*.py verify/mutants/*.c ref/*.sh ref/*.c ref/*.cpp ref/*.py smoke_rust.c smoke_rust.sh 2>/dev/null; chmod +x verify/*.sh verify/mutants/*.sh ref/*.sh smoke_rust.sh 2>/dev/null; md5sum ref/loader_a.inc ref/loader_b.inc'
Get-FileHash -Algorithm MD5 "$root\ref\loader_a.inc", "$root\ref\loader_b.inc" | ForEach-Object { "$($_.Hash.ToLower())  local $(Split-Path -Leaf $_.Path)" }
