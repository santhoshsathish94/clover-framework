param(
    [ValidateSet('Plan', 'Move', 'Verify')][string]$Action = 'Plan',
    [string]$Selected = 'all'
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$record = Join-Path $root 'DATASET-CENTRALIZATION-WINDOWS'
$planPath = Join-Path $record 'plan.json'
$journal = Join-Path $record 'journal.jsonl'
$groups = @('client', 'server', 'normalization') + @(1..92 | ForEach-Object { "tansformers/transformer-$_" })

function Assert-Absent([string]$Path) {
    if (Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue) { throw "Destination exists: $Path" }
}

function Get-Inventory([string]$Directory) {
    $base = Get-Item -LiteralPath $Directory -Force
    if (!$base.PSIsContainer -or ($base.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Not a real dataset directory: $Directory"
    }
    $entries = @($base) + @(Get-ChildItem -LiteralPath $Directory -Force -Recurse)
    foreach ($entry in $entries | Sort-Object FullName) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Nested link: $($entry.FullName)" }
        $hash = $null
        $length = 0
        if (!$entry.PSIsContainer) {
            $hash = (Get-FileHash -LiteralPath $entry.FullName -Algorithm SHA256).Hash
            $length = $entry.Length
        }
        [ordered]@{
            Relative = [IO.Path]::GetRelativePath($Directory, $entry.FullName)
            Directory = [bool]$entry.PSIsContainer
            Length = $length
            Modified = $entry.LastWriteTimeUtc.Ticks.ToString()
            Created = $entry.CreationTimeUtc.Ticks.ToString()
            Hash = $hash
        }
    }
}

function Assert-Inventory($Unit, [string]$Directory) {
    $actual = @(Get-Inventory $Directory) | ConvertTo-Json -Depth 6 -Compress
    $expected = @($Unit.Entries) | ConvertTo-Json -Depth 6 -Compress
    if ($actual -cne $expected) { throw "Dataset contents or metadata changed: $($Unit.Group)" }
}

function Get-CodeInventory {
    @(Get-ChildItem -LiteralPath (Join-Path $root 'code') -File -Force -Recurse |
        Where-Object { $_.FullName -notmatch '[\\/]bin[\\/]dataset[\\/]' } |
        Sort-Object FullName | ForEach-Object {
            [ordered]@{ Path = $_.FullName; Length = $_.Length; Modified = $_.LastWriteTimeUtc.Ticks.ToString() }
        })
}

function Write-Journal([string]$Group, [string]$State) {
    $line = ([ordered]@{ Group = $Group; State = $State } | ConvertTo-Json -Compress) + "`n"
    $bytes = [Text.Encoding]::UTF8.GetBytes($line)
    $stream = [IO.File]::Open($journal, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try {
        $null = $stream.Seek(0, [IO.SeekOrigin]::End)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    } finally { $stream.Dispose() }
}

function Assert-Junction($Unit) {
    $link = Get-Item -LiteralPath $Unit.Source -Force
    if ($link.LinkType -ne 'Junction' -or
        [IO.Path]::GetFullPath($link.Target) -ne [IO.Path]::GetFullPath($Unit.Target)) {
        throw "Runtime junction mismatch: $($Unit.Source)"
    }
    foreach ($entry in $Unit.Entries | Where-Object { !$_.Directory }) {
        $path = Join-Path $Unit.Source $entry.Relative
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -cne $entry.Hash) {
            throw "File differs through runtime junction: $path"
        }
    }
}

if ($Action -eq 'Plan') {
    Assert-Absent $record
    $units = foreach ($group in $groups) {
        $source = Join-Path $root "code/$group/bin/dataset"
        $target = Join-Path $root "dataset/$group"
        Assert-Absent $target
        if ([IO.Path]::GetPathRoot($source) -ne [IO.Path]::GetPathRoot($target)) { throw "Different volumes: $group" }
        [ordered]@{ Group = $group; Source = $source; Target = $target; Entries = @(Get-Inventory $source) }
    }
    $plan = [ordered]@{ Root = $root; Units = @($units); Code = @(Get-CodeInventory) }
    $null = New-Item -ItemType Directory -Path $record
    [IO.File]::WriteAllText($planPath, ($plan | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    "PASS: planned $($units.Count) local dataset directories; no payload moved"
    return
}

$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json -AsHashtable
if ($plan.Root -ne $root -or $plan.Units.Count -ne 95) { throw 'Wrong or incomplete relocation plan' }
$completed = @()
if (Test-Path -LiteralPath $journal) {
    $completed = @(Get-Content -LiteralPath $journal | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object State -eq 'MOVED_AND_VERIFIED' | ForEach-Object Group)
}

if ($Action -eq 'Move') {
    foreach ($unit in $plan.Units) {
        if ($Selected -ne 'all' -and $Selected -ne $unit.Group) { continue }
        if ($completed -contains $unit.Group) {
            Assert-Inventory $unit $unit.Target
            Assert-Junction $unit
            continue
        }
        Assert-Absent $unit.Target
        Assert-Inventory $unit $unit.Source
        $parent = Split-Path $unit.Target -Parent
        if (!(Test-Path -LiteralPath $parent)) { $null = New-Item -ItemType Directory -Path $parent }
        Write-Journal $unit.Group 'STARTED'
        Move-Item -LiteralPath $unit.Source -Destination $unit.Target
        try {
            $null = New-Item -ItemType Junction -Path $unit.Source -Target $unit.Target
        } catch {
            if (!(Get-Item -LiteralPath $unit.Source -Force -ErrorAction SilentlyContinue)) {
                Move-Item -LiteralPath $unit.Target -Destination $unit.Source
            }
            throw
        }
        Assert-Inventory $unit $unit.Target
        Assert-Junction $unit
        Write-Journal $unit.Group 'MOVED_AND_VERIFIED'
        "PASS: $($unit.Group) moved; all file hashes and runtime junction verified"
    }
} else {
    if ($completed.Count -ne 95) { throw 'Incomplete move count' }
    foreach ($unit in $plan.Units) {
        Assert-Inventory $unit $unit.Target
        Assert-Junction $unit
    }
    $before = @($plan.Code) | ConvertTo-Json -Depth 5 -Compress
    $after = @(Get-CodeInventory) | ConvertTo-Json -Depth 5 -Compress
    if ($before -cne $after) { throw 'Code or binaries changed during relocation' }
    'PASS: 95 local central datasets and runtime junctions; all dataset hashes, code and binaries unchanged'
}