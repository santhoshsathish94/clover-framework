param([ValidateSet('Plan', 'Move', 'Recover', 'Verify')][string]$Action = 'Plan')
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$source = Join-Path $root 'code'
$target = Join-Path $source 'distrubuted-hosting'
$record = Join-Path $root 'HOSTING-MOVE-WINDOWS'
$packages = @('client', 'normalization', 'pipeline', 'server', 'tansformers')

function Get-Entries([string]$Base) {
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Base)
    $entries = while ($pending.Count) {
        $directory = $pending.Pop()
        foreach ($entry in Get-ChildItem -LiteralPath $directory -Force) {
            $relative = [IO.Path]::GetRelativePath($Base, $entry.FullName)
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                if ($entry.LinkType -ne 'Junction' -or $entry.Name -ne 'dataset' -or
                    !$entry.Target.StartsWith((Join-Path $root 'dataset'), [StringComparison]::OrdinalIgnoreCase)) {
                    throw "Unexpected code link: $($entry.FullName)"
                }
                if (!(Test-Path -LiteralPath $entry.Target -PathType Container)) { throw "Broken junction: $relative" }
                [ordered]@{ Path = $relative; Kind = 'Junction'; Target = $entry.Target }
            } elseif ($entry.PSIsContainer) {
                $pending.Push($entry.FullName)
                [ordered]@{ Path = $relative; Kind = 'Directory'; Created = $entry.CreationTimeUtc.Ticks.ToString() }
            } else {
                [ordered]@{
                    Path = $relative; Kind = 'File'; Length = $entry.Length
                    Created = $entry.CreationTimeUtc.Ticks.ToString(); Modified = $entry.LastWriteTimeUtc.Ticks.ToString()
                    Hash = (Get-FileHash -LiteralPath $entry.FullName -Algorithm SHA256).Hash
                }
            }
        }
    }
    @($entries | Sort-Object { $_.Path })
}

function Write-State([string]$Package, [string]$State) {
    [ordered]@{ Package = $Package; State = $State } | ConvertTo-Json -Compress |
        Add-Content -LiteralPath (Join-Path $record 'journal.jsonl') -Encoding utf8
}

if ($Action -eq 'Plan') {
    if ((Test-Path -LiteralPath $record) -or (Test-Path -LiteralPath $target)) { throw 'Occupied move destination or plan' }
    $names = @(Get-ChildItem -LiteralPath $source -Force | Sort-Object Name | ForEach-Object Name)
    if (($names -join '|') -ne ($packages -join '|')) { throw 'Unexpected code entries' }
    $entries = @(Get-Entries $source)
    if (@($entries | Where-Object { $_.Kind -eq 'Junction' }).Count -ne 95) { throw 'Expected 95 dataset junctions' }
    $null = New-Item -ItemType Directory -Path $record
    [ordered]@{ Root = $root; Entries = $entries } | ConvertTo-Json -Depth 7 |
        Set-Content -LiteralPath (Join-Path $record 'plan.json') -Encoding utf8
    "PASS: five local packages, $(@($entries | Where-Object { $_.Kind -eq 'File' }).Count) hashed files, 95 junctions; nothing moved"
    return
}

$plan = Get-Content -LiteralPath (Join-Path $record 'plan.json') -Raw | ConvertFrom-Json -AsHashtable
if ($plan.Root -ne $root) { throw 'Wrong relocation root' }
if ($Action -eq 'Move') {
    if (Test-Path -LiteralPath $target) { throw 'Hosting directory already exists; inspect interrupted move' }
    $actual = @(Get-Entries $source) | ConvertTo-Json -Depth 7 -Compress
    if ($actual -cne ($plan.Entries | ConvertTo-Json -Depth 7 -Compress)) { throw 'Code changed after planning' }
    $null = New-Item -ItemType Directory -Path $target
    foreach ($package in $packages) {
        Write-State $package 'STARTED'
        [IO.Directory]::Move((Join-Path $source $package), (Join-Path $target $package))
        Write-State $package 'MOVED'
    }
}
if ($Action -eq 'Recover') {
    $dataPlan = Get-Content -LiteralPath (Join-Path $root 'DATASET-CENTRALIZATION-WINDOWS/plan.json') -Raw | ConvertFrom-Json
    foreach ($unit in $dataPlan.Units) {
        $bin = Join-Path $target "$($unit.Group)/bin/dataset"
        if ((Get-Item -LiteralPath $bin -Force).LinkType -eq 'Junction') { continue }
        if ($unit.Group -notlike 'tansformers/transformer-*') { throw "Unexpected recovery group: $($unit.Group)" }
        $present = @(Get-ChildItem -LiteralPath $bin -Force -Recurse)
        if (@($present | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Unexpected recovery link' }
        $files = @($unit.Entries | Where-Object { !$_.Directory })
        if (@($present | Where-Object { !$_.PSIsContainer }).Count -ne $files.Count) { throw 'Unexpected recovery file count' }
        if (@(Get-ChildItem -LiteralPath $unit.Target -Force).Count) { throw "Central recovery destination not empty: $($unit.Target)" }
        foreach ($entry in $files) {
            $path = Join-Path $bin $entry.Relative
            if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -cne $entry.Hash) { throw "Recovery hash mismatch: $path" }
        }
        Write-State $unit.Group 'RETURNING_DATA_TO_CENTRAL'
        foreach ($entry in $unit.Entries | Where-Object Directory | Sort-Object { $_.Relative.Length }) {
            $null = [IO.Directory]::CreateDirectory((Join-Path $unit.Target $entry.Relative))
        }
        foreach ($entry in $files) {
            $destination = Join-Path $unit.Target $entry.Relative
            [IO.File]::Move((Join-Path $bin $entry.Relative), $destination)
            if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -cne $entry.Hash) { throw "Restored hash mismatch: $destination" }
        }
        foreach ($entry in $present | Where-Object PSIsContainer | Sort-Object { $_.FullName.Length } -Descending) {
            [IO.Directory]::Delete($entry.FullName, $false)
        }
        [IO.Directory]::Delete($bin, $false)
        $null = New-Item -ItemType Junction -Path $bin -Target $unit.Target
        foreach ($entry in $unit.Entries | Where-Object Directory | Sort-Object { $_.Relative.Length } -Descending) {
            $path = Join-Path $unit.Target $entry.Relative
            [IO.Directory]::SetCreationTimeUtc($path, [datetime]::new([long]$entry.Created, [DateTimeKind]::Utc))
            [IO.Directory]::SetLastWriteTimeUtc($path, [datetime]::new([long]$entry.Modified, [DateTimeKind]::Utc))
        }
        Write-State $unit.Group 'CENTRAL_DATA_AND_JUNCTION_RESTORED'
    }
    foreach ($entry in $plan.Entries | Where-Object { $_.Kind -eq 'Directory' }) {
        [IO.Directory]::SetCreationTimeUtc((Join-Path $target $entry.Path), [datetime]::new([long]$entry.Created, [DateTimeKind]::Utc))
    }
}
$actual = @(Get-Entries $target) | ConvertTo-Json -Depth 7 -Compress
if ($actual -cne ($plan.Entries | ConvertTo-Json -Depth 7 -Compress)) { throw 'Moved code or junctions differ from inventory' }
$remaining = @(Get-ChildItem -LiteralPath $source -Force)
if ($remaining.Count -ne 1 -or $remaining[0].Name -ne 'distrubuted-hosting') { throw 'Old code entries remain' }
if ($Action -in @('Move', 'Recover')) { Write-State 'all' 'VERIFIED' }
'PASS: only distrubuted-hosting under code; five packages, all file hashes and 95 dataset junctions unchanged'