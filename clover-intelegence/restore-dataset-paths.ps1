param(
    [ValidateSet('Plan', 'Move', 'Verify')][string]$Action = 'Plan',
    [string]$Selected = 'all'
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$data = Join-Path $root 'dataset'
$code = Join-Path $root 'code/distrubuted-hosting'
$record = Join-Path $root 'DATASET-ORIGINAL-PATHS-WINDOWS'
$packages = @('normalization','client','server') + @(1..92 | ForEach-Object { "tansformers/transformer-$_" })

function Get-Units {
    [ordered]@{ Group='normalization'; Source='normalization/leaves.json'; Target='leaves.json' }
    foreach ($entry in @('inputs','outputs','tiktoken.model','vocabulary.bin')) {
        [ordered]@{ Group='client'; Source="client/$entry"; Target=$entry }
    }
    [ordered]@{ Group='server'; Source='server/trunk-0-manifest.json'; Target='trunk-0/manifest.json' }
    [ordered]@{ Group='server'; Source='server/trunk-0-qkv-manifest.json'; Target='operators/trunk-0-qkv/manifest.json' }
    [ordered]@{ Group='server'; Source='server/qkv-all-layer-0-manifest.json'; Target='operators/trunk-0-qkv/qkv-all-layer-0-manifest.json' }
    foreach ($layer in 1..92) {
        $group = "tansformers/transformer-$layer"
        foreach ($entry in @("trunk-$layer","root-$layer","operators/qkv-all/layer-$layer")) {
            [ordered]@{ Group=$group; Source="$group/$entry"; Target=$entry }
        }
    }
}

function Assert-Absent([string]$Path) {
    if (Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue) { throw "Occupied: $Path" }
}

function Get-Inventory([string]$Path) {
    $base = Get-Item -LiteralPath $Path -Force
    if ($base.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Payload is a link: $Path" }
    $entries = @($base)
    if ($base.PSIsContainer) { $entries += @(Get-ChildItem -LiteralPath $Path -Force -Recurse) }
    foreach ($entry in $entries | Sort-Object FullName) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Nested payload link: $($entry.FullName)" }
        $hash = $null; $length = 0
        if (!$entry.PSIsContainer) {
            $hash = (Get-FileHash -LiteralPath $entry.FullName -Algorithm SHA256).Hash
            $length = $entry.Length
        }
        [ordered]@{
            Relative=[IO.Path]::GetRelativePath($Path,$entry.FullName); Directory=[bool]$entry.PSIsContainer
            Length=$length; Created=$entry.CreationTimeUtc.Ticks.ToString(); Modified=$entry.LastWriteTimeUtc.Ticks.ToString(); Hash=$hash
        }
    }
}

function Assert-Inventory($Unit,[string]$Path) {
    $actual = @(Get-Inventory $Path) | ConvertTo-Json -Depth 6 -Compress
    $expected = @($Unit.Entries) | ConvertTo-Json -Depth 6 -Compress
    if ($actual -cne $expected) { throw "Payload differs: $Path" }
}

function Write-Journal([string]$Name,[string]$State) {
    $bytes = [Text.Encoding]::UTF8.GetBytes(([ordered]@{ Name=$Name; State=$State } | ConvertTo-Json -Compress) + "`n")
    $stream = [IO.File]::Open((Join-Path $record 'journal.jsonl'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try { $null=$stream.Seek(0,[IO.SeekOrigin]::End); $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) }
    finally { $stream.Dispose() }
}

function Get-CodeFiles {
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($code)
    $files = while ($pending.Count) {
        foreach ($entry in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
            if ($entry.PSIsContainer) { $pending.Push($entry.FullName) }
            else { [ordered]@{ Path=$entry.FullName; Length=$entry.Length; Modified=$entry.LastWriteTimeUtc.Ticks.ToString() } }
        }
    }
    @($files | Sort-Object { $_.Path })
}

function Assert-Junction([string]$Path,[string]$Target) {
    $link = Get-Item -LiteralPath $Path -Force
    if ($link.LinkType -ne 'Junction' -or [IO.Path]::GetFullPath($link.Target) -ne [IO.Path]::GetFullPath($Target) -or
        !(Test-Path -LiteralPath $Target -PathType Container)) { throw "Bad mapping: $Path" }
}

function Assert-Bin([string]$Group) {
    $bin = Join-Path $code "$Group/bin/dataset"
    if ($Group -eq 'server') {
        if (@(Get-ChildItem -LiteralPath $bin -Force).Count -ne 2) { throw 'Server mapping count differs' }
        Assert-Junction (Join-Path $bin 'trunk-0') (Join-Path $data 'trunk-0')
        Assert-Junction (Join-Path $bin 'trunk-0-qkv') (Join-Path $data 'operators/trunk-0-qkv')
    } else { Assert-Junction $bin $data }
}

if ($Action -eq 'Plan') {
    Assert-Absent $record
    $units = foreach ($unit in Get-Units) {
        $source = Join-Path $data $unit.Source
        Assert-Absent (Join-Path $data $unit.Target)
        $unit.Entries = @(Get-Inventory $source)
        $unit
    }
    foreach ($group in $packages) { Assert-Junction (Join-Path $code "$group/bin/dataset") (Join-Path $data $group) }
    $moving = @{}
    foreach ($unit in $units) {
        foreach ($entry in $unit.Entries | Where-Object { !$_.Directory }) {
            $moving[[IO.Path]::GetFullPath((Join-Path (Join-Path $data $unit.Source) $entry.Relative))] = $true
        }
    }
    $unchanged = @(Get-ChildItem -LiteralPath $data -File -Force -Recurse | Where-Object { !$moving.ContainsKey($_.FullName) } | ForEach-Object {
        [ordered]@{ Path=$_.FullName; Hash=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
    })
    $plan = [ordered]@{ Root=$root; Units=@($units); Code=@(Get-CodeFiles); Unchanged=$unchanged }
    $null = New-Item -ItemType Directory -Path $record
    [IO.File]::WriteAllText((Join-Path $record 'plan.json'),($plan | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
    "PASS: $($units.Count) local original-path moves planned; no payload moved"
    return
}

$plan = Get-Content -LiteralPath (Join-Path $record 'plan.json') -Raw | ConvertFrom-Json -AsHashtable
if ($plan.Root -ne $root -or $plan.Units.Count -ne 284) { throw 'Invalid original-path plan' }
$completed = @()
$journal = Join-Path $record 'journal.jsonl'
if (Test-Path -LiteralPath $journal) {
    $completed = @(Get-Content -LiteralPath $journal | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object State -eq 'PACKAGE_VERIFIED' | ForEach-Object Name)
}
if ($Action -eq 'Move') {
    foreach ($group in $packages) {
        if ($Selected -ne 'all' -and $Selected -ne $group) { continue }
        if ($completed -contains $group) { Assert-Bin $group; continue }
        foreach ($unit in $plan.Units | Where-Object { $_.Group -eq $group }) {
            $source = Join-Path $data $unit.Source; $target = Join-Path $data $unit.Target
            Assert-Absent $target
            Assert-Inventory $unit $source
            $null = [IO.Directory]::CreateDirectory((Split-Path $target -Parent))
            Write-Journal $unit.Source 'STARTED'
            if ((Get-Item -LiteralPath $source).PSIsContainer) { [IO.Directory]::Move($source,$target) }
            else { [IO.File]::Move($source,$target) }
            Assert-Inventory $unit $target
            Write-Journal $unit.Target 'MOVED_AND_VERIFIED'
        }
        $bin = Join-Path $code "$group/bin/dataset"
        Assert-Junction $bin (Join-Path $data $group)
        [IO.Directory]::Delete($bin,$false)
        if ($group -eq 'server') {
            $null = [IO.Directory]::CreateDirectory($bin)
            $null = New-Item -ItemType Junction -Path (Join-Path $bin 'trunk-0') -Target (Join-Path $data 'trunk-0')
            $null = New-Item -ItemType Junction -Path (Join-Path $bin 'trunk-0-qkv') -Target (Join-Path $data 'operators/trunk-0-qkv')
        } else { $null = New-Item -ItemType Junction -Path $bin -Target $data }
        Assert-Bin $group
        Write-Journal $group 'PACKAGE_VERIFIED'
        "PASS: $group original real dataset paths and bin-only mappings"
    }
} else {
    if ($completed.Count -ne 95) { throw 'Incomplete package count' }
    foreach ($unit in $plan.Units) {
        Assert-Absent (Join-Path $data $unit.Source)
        Assert-Inventory $unit (Join-Path $data $unit.Target)
    }
    foreach ($group in $packages) { Assert-Bin $group }
    foreach ($entry in Get-ChildItem -LiteralPath $data -Force -Recurse) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Link remains in dataset: $($entry.FullName)" }
    }
    foreach ($entry in $plan.Unchanged) {
        if ((Get-FileHash -LiteralPath $entry.Path -Algorithm SHA256).Hash -cne $entry.Hash) { throw "Unrelated file changed: $($entry.Path)" }
    }
    if ((@(Get-CodeFiles) | ConvertTo-Json -Depth 6 -Compress) -cne (@($plan.Code) | ConvertTo-Json -Depth 6 -Compress)) { throw 'Code or binaries changed' }
    'PASS: 284 original real paths, all file hashes/timestamps, 95 bin mappings; no links inside dataset; code unchanged'
}