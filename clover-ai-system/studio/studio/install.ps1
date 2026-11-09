param([switch]$Quiet, [string]$Base = '', [string]$Components = 'core,csi,c2c3', [string]$Reuse = '',
      [switch]$NoShortcuts)
# Clover AI Studio setup. Installs everything on this computer from the publishers, each file checked against the
# SHA-256 its publisher lists (clover\launch\manifest.json, studio\requirements.lock):
#   1. checks the computer against the minimum (the laptop Clover was built on: 16 GB, 8+ threads with AVX2, 64-bit
#      Windows 10 2004+, 25 GB free) and the Visual C++ runtime llama.cpp needs (Microsoft's installer if missing),
#   2. the app (this package), uv, Python 3.12 and the locked Python packages (torch, spaCy + en_core_web_trf, ...),
#   3. llama.cpp, MinGit, ffmpeg, the LFM3-Omni-2B model, CSI's dictionary and lemmatiser, the embedder, the picture
#      models and Clover's precomputed C2/C3 parts (the C1 + C2 + C3 engine) - clover\setup_fetch.py,
#   4. reads the hardware (clover\hardware.py) so the Studio picks GPU or CPU, context and threads for this computer,
#   5. shortcuts, the VS Code extension and an uninstall entry (not with -NoShortcuts, for a second test install).
# Run again to repair or update: what is already there with the right hash is kept. -Reuse <folder>: copy files with
# the same hash from another Clover install instead of downloading them.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Base) { $Base = Join-Path $env:USERPROFILE 'CloverAI' }
$Base = [IO.Path]::GetFullPath($Base)
$version = '5.0.2'
$log = Join-Path $env:TEMP 'CloverAIStudio-setup.log'
function Log([string]$m) { Add-Content -Path $log -Value ((Get-Date -Format 's') + ' ' + $m) }
function Remove-Tree([string]$p) {  # PS 5.1's Remove-Item -Recurse can stop halfway; rmdir does not (nor follow junctions)
    for ($i = 0; $i -lt 3 -and (Test-Path $p); $i++) {
        if ($i) { Start-Sleep -Seconds 2 }
        & cmd.exe /d /c "rmdir /s /q `"$p`"" 2>&1 | ForEach-Object { Log "$_" }
    }
    if (Test-Path $p) { throw "could not remove $p (a program may still have files open in it)" }
}
$ui = $null  # the setup window's shared state, when there is a window
function Step([string]$m, [int]$n) {
    Log $m; Write-Host ''; Write-Host "== $m" -ForegroundColor Green
    if ($ui) {
        if ($ui.Cancel) { throw 'stopped' }
        $ui.Step = $n; $ui.StepText = $m; $ui.Item = ''; $ui.ItemDone = 0.0; $ui.ItemTotal = 0.0; $ui.Detail = ''
    }
}
function Show-Event([string]$j) {  # setup_fetch's progress events (CLOVER_PROGRESS=1) -> the window
    try { $e = $j | ConvertFrom-Json } catch { return }
    switch ($e.ev) {
        'plan' { $ui.Plan = [double]$e.total; $ui.Done = 0.0 }
        'item' {
            if ($e.state -eq 'checking') {
                $ui.Item = $e.name; $ui.ItemState = 'checking'; $ui.ItemDone = 0.0; $ui.ItemTotal = [double]$e.size
                $ui.ItemIndex = $e.i; $ui.Items = $e.n
            } elseif ($e.state -eq 'getting') {
                $ui.ItemState = 'getting'; if ($e.size) { $ui.ItemTotal = [double]$e.size }
            } else {
                $ui.Done = [double]$ui.Done + [double]$ui.ItemTotal; $ui.ItemDone = 0.0; $ui.ItemState = $e.state
            }
        }
        'bytes' { $ui.ItemDone = [double]$e.done; if ($e.total) { $ui.ItemTotal = [double]$e.total } }
    }
}
function Run([string]$exe, [string[]]$argv, [string]$what) {
    $ErrorActionPreference = 'Continue'  # tools write progress to stderr; the exit code decides
    & $exe @argv 2>&1 | ForEach-Object {
        $s = "$_"
        if ($s.StartsWith('@@')) { if ($ui) { Show-Event $s.Substring(2) }; return }
        Write-Host "   $s"; Log $s
        if ($ui -and $s.Trim()) { $ui.Detail = $s.Trim() }
    }
    $rc = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($ui -and $ui.Cancel) { throw 'stopped' }
    if ($rc -ne 0) { throw "$what failed (exit $rc); see $log" }
}

# ---------------------------------------------------------------- the setup window
# The app's look (cream, clover green). WinForms on its own STA thread, fed through a synchronized table; Cancel stops
# the running tool. The console is hidden once the window shows; without a window everything goes to the console.
$windowScript = {
    try {
        Add-Type -AssemblyName System.Windows.Forms, System.Drawing, System.IO.Compression.FileSystem
        Add-Type -Namespace CloverUi -Name Native -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
        [void][CloverUi.Native]::SetProcessDPIAware()
        [Windows.Forms.Application]::EnableVisualStyles()
        function Rgb([string]$h) { [Drawing.ColorTranslator]::FromHtml($h) }
        $cream = Rgb '#F7F3E9'; $surface = Rgb '#FFFDF8'; $fill = Rgb '#ECE6D7'; $green = Rgb '#176B2C'
        $ink = Rgb '#1B271F'; $ink2 = Rgb '#39473D'; $muted = Rgb '#687469'; $faint = Rgb '#98A198'; $warn = Rgb '#85540F'
        $dot = [string][char]0x00B7
        $f = New-Object Windows.Forms.Form
        $f.SuspendLayout()
        $f.AutoScaleDimensions = New-Object Drawing.SizeF(96, 96); $f.AutoScaleMode = 'Dpi'
        $f.Text = 'Clover AI Studio Setup'; $f.ClientSize = New-Object Drawing.Size(600, 470)
        $f.StartPosition = 'CenterScreen'; $f.FormBorderStyle = 'FixedSingle'; $f.MaximizeBox = $false
        $f.BackColor = $cream; $f.ForeColor = $ink; $f.Font = New-Object Drawing.Font('Segoe UI', 10)
        $logo = $null
        try {  # the Clover icon, straight from the app package
            $z = [IO.Compression.ZipFile]::OpenRead($ui.Zip)
            foreach ($en in $z.Entries) {
                $name = $en.FullName -replace '\\', '/'
                if ($name -in 'clover/web/icon.png', 'studio/clover.ico') {
                    $ms = New-Object IO.MemoryStream; $st = $en.Open(); $st.CopyTo($ms); $st.Close(); $ms.Position = 0
                    if ($name -like '*.png') { $logo = [Drawing.Image]::FromStream($ms) } else { $f.Icon = New-Object Drawing.Icon($ms) }
                }
            }
            $z.Dispose()
        } catch { }
        function Lbl([string]$t, [int]$x, [int]$y, [int]$w, [int]$h, [float]$size, [string]$face, $color) {
            $l = New-Object Windows.Forms.Label
            $l.Text = $t; $l.Location = New-Object Drawing.Point($x, $y); $l.Size = New-Object Drawing.Size($w, $h)
            $l.Font = New-Object Drawing.Font($face, $size); $l.ForeColor = $color; $l.AutoEllipsis = $true
            $f.Controls.Add($l); $l
        }
        if ($logo) {
            $pb = New-Object Windows.Forms.PictureBox; $pb.Image = $logo; $pb.SizeMode = 'Zoom'
            $pb.Location = New-Object Drawing.Point(32, 30); $pb.Size = New-Object Drawing.Size(44, 44); $f.Controls.Add($pb)
        }
        $title = Lbl 'Installing Clover AI Studio' 90 26 480 32 15 'Segoe UI Semibold' $ink
        $sub = Lbl "Version $($ui.Version). Everything comes from its publisher and is checked before it is used." 90 60 480 40 9.5 'Segoe UI' $muted
        $names = 'Checking this computer', 'Installing the app', 'Python and its packages', 'Models, tools and data', 'Reading the hardware', 'Shortcuts and VS Code'
        $rows = @()
        for ($i = 0; $i -lt $names.Count; $i++) {
            $mark = Lbl ([string][char]0x25CB) 36 (114 + 28 * $i) 24 26 11 'Segoe UI Symbol' $faint
            $text = Lbl $names[$i] 62 (114 + 28 * $i) 500 26 10.5 'Segoe UI' $faint
            $rows += , @($mark, $text, '')
        }
        $overall = Lbl 'Starting' 32 292 330 24 10 'Segoe UI Semibold' $ink2
        $overallR = Lbl '' 362 292 206 24 10 'Segoe UI Semibold' $ink2; $overallR.TextAlign = 'TopRight'
        $msg = Lbl '' 32 292 536 116 10 'Segoe UI' $ink2; $msg.Visible = $false; $msg.AutoEllipsis = $false
        $bars = @()
        foreach ($b in @(@(318, 10), @(362, 6))) {
            $p = New-Object Windows.Forms.Panel
            $p.Location = New-Object Drawing.Point(32, $b[0]); $p.Size = New-Object Drawing.Size(536, $b[1]); $p.BackColor = $fill
            [Windows.Forms.Panel].GetProperty('DoubleBuffered', [Reflection.BindingFlags]'NonPublic,Instance').SetValue($p, $true, $null)
            $f.Controls.Add($p); $bars += $p
        }
        $item = Lbl '' 32 336 380 24 9.5 'Segoe UI' $ink2
        $itemR = Lbl '' 412 336 156 24 9.5 'Segoe UI' $ink2; $itemR.TextAlign = 'TopRight'
        $detail = Lbl '' 32 374 300 22 9 'Segoe UI' $muted
        $detailR = Lbl '' 332 374 236 22 9 'Segoe UI' $muted; $detailR.TextAlign = 'TopRight'
        $btn1 = New-Object Windows.Forms.Button; $btn1.Text = 'Cancel'
        $btn1.Location = New-Object Drawing.Point(388, 414); $btn1.Size = New-Object Drawing.Size(180, 36)
        $btn2 = New-Object Windows.Forms.Button; $btn2.Text = 'Show details'; $btn2.Visible = $false
        $btn2.Location = New-Object Drawing.Point(236, 414); $btn2.Size = New-Object Drawing.Size(140, 36)
        foreach ($b in $btn1, $btn2) {
            $b.FlatStyle = 'Flat'; $b.FlatAppearance.BorderColor = $fill; $b.BackColor = $surface; $b.ForeColor = $ink
            $b.Font = New-Object Drawing.Font('Segoe UI Semibold', 10); $b.Cursor = 'Hand'; $f.Controls.Add($b)
        }
        $f.ResumeLayout($false)

        $v = @{ p1 = 0.0; p2 = 0.0; m = 0.0; indet = $true; item = $null; state = 'running'; samples = New-Object Collections.ArrayList }
        $bars[0].Add_Paint({ param($s, $e)
            $w = [int]($s.Width * [math]::Min(1.0, [math]::Max(0.0, $v.p1)))
            if ($w -gt 0) { $br = New-Object Drawing.SolidBrush($green); $e.Graphics.FillRectangle($br, 0, 0, $w, $s.Height); $br.Dispose() }
        })
        $bars[1].Add_Paint({ param($s, $e)
            $br = New-Object Drawing.SolidBrush($green)
            if ($v.indet) {
                $bw = [int]($s.Width * 0.24); $x = [int](($s.Width + $bw) * $v.m) - $bw
                $e.Graphics.FillRectangle($br, $x, 0, $bw, $s.Height)
            } else {
                $w = [int]($s.Width * [math]::Min(1.0, [math]::Max(0.0, $v.p2)))
                if ($w -gt 0) { $e.Graphics.FillRectangle($br, 0, 0, $w, $s.Height) }
            }
            $br.Dispose()
        })
        function Size-Text([double]$b) { if ($b -ge 1GB) { '{0:N1} GB' -f ($b / 1GB) } else { '{0:N0} MB' -f ($b / 1MB) } }
        function Stop-Tools {  # the tool setup is running (uv, Python) and anything it started
            Get-CimInstance Win32_Process -Filter "ParentProcessId=$PID" -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -ne 'conhost.exe' } |
                ForEach-Object { & taskkill.exe /PID $_.ProcessId /T /F 2>&1 | Out-Null }
        }
        $tick = {
            $step = [int]$ui.Step; $state = $ui.State
            for ($i = 0; $i -lt $rows.Count; $i++) {
                $n = $i + 1
                $k = if ($state -eq 'done' -or $n -lt $step) { 'done' } elseif ($n -eq $step) { if ($state -eq 'running') { 'now' } else { 'stop' } } else { 'todo' }
                if ($rows[$i][2] -eq $k) { continue }
                $rows[$i][2] = $k
                $mk = $rows[$i][0]; $tx = $rows[$i][1]
                switch ($k) {
                    'done' { $mk.Text = [string][char]0x2713; $mk.ForeColor = $green; $tx.ForeColor = $ink2; $tx.Font = New-Object Drawing.Font('Segoe UI', 10.5) }
                    'now' { $mk.Text = [string][char]0x25CF; $mk.ForeColor = $green; $tx.ForeColor = $ink; $tx.Font = New-Object Drawing.Font('Segoe UI Semibold', 10.5) }
                    'stop' { $mk.Text = [string][char]0x25CF; $mk.ForeColor = $warn; $tx.ForeColor = $warn; $tx.Font = New-Object Drawing.Font('Segoe UI Semibold', 10.5) }
                    'todo' { $mk.Text = [string][char]0x25CB; $mk.ForeColor = $faint; $tx.ForeColor = $faint; $tx.Font = New-Object Drawing.Font('Segoe UI', 10.5) }
                }
            }
            if ($state -ne 'running') {
                if ($state -eq 'autoclose') { $ui.State = 'done'; $f.Close(); return }
                if ($v.state -ne $state) {
                    $v.state = $state
                    foreach ($c in @($overall, $overallR, $item, $itemR, $detail, $detailR) + $bars) { $c.Visible = $false }
                    $msg.Text = ([string]$ui.Message) -replace "`n`n", "`n"; $msg.Visible = $true; $btn2.Visible = $true; $btn1.Enabled = $true
                    if ($state -eq 'done') {
                        $title.Text = 'Clover AI Studio is ready'
                        $btn1.Text = 'Start Clover AI Studio'; $btn1.BackColor = $green; $btn1.ForeColor = [Drawing.Color]::White
                        $btn1.FlatAppearance.BorderColor = $green; $btn2.Text = 'Close'
                    } else {
                        $title.Text = 'Setup stopped'; $title.ForeColor = $warn; $btn1.Text = 'Close'; $btn2.Text = 'Show details'
                    }
                    $f.Activate()
                }
                return
            }
            # the overall bar: the first steps are quick, the data is most of the time
            $now = [DateTime]::UtcNow
            $cur = [double]$ui.Done + [double]$ui.ItemDone
            $plan = [double]$ui.Plan
            $frac = if ($plan -gt 0) { [math]::Min(1.0, $cur / $plan) } else { 0.0 }
            $target = if ($step -le 3) { 0.03 * $step } elseif ($step -eq 4) { 0.10 + 0.85 * $frac } else { 0.95 + 0.02 * ($step - 4) }
            $v.p1 += ($target - $v.p1) * 0.15
            $bars[0].Invalidate()
            $speed = ''
            if ($step -eq 4 -and $plan -gt 0) {
                [void]$v.samples.Add(@($now, $cur))
                while ($v.samples.Count -gt 2 -and ($now - $v.samples[0][0]).TotalSeconds -gt 8) { $v.samples.RemoveAt(0) }
                $first = $v.samples[0]; $span = ($now - $first[0]).TotalSeconds
                $rate = if ($span -ge 1.5) { ($cur - $first[1]) / $span } else { 0 }
                $overall.Text = 'Models, tools and data'
                $overallR.Text = "$([int]($frac * 100))%  $dot  $(Size-Text $cur) of $(Size-Text $plan)"
                if ($rate -gt 50KB) {
                    $left = ($plan - $cur) / $rate
                    $eta = if ($left -lt 60) { 'under a minute left' } elseif ($left -lt 3600) { "about $([math]::Ceiling($left / 60)) min left" } else { 'about {0:N1} h left' -f ($left / 3600) }
                    $speed = "$('{0:N1}' -f ($rate / 1MB)) MB/s  $dot  $eta"
                }
            } else {
                $overall.Text = $ui.StepText; $overallR.Text = ''
            }
            $detailR.Text = $speed
            # the item bar: the file being fetched, or a moving band while a step has no measure
            $name = [string]$ui.Item; $tot = [double]$ui.ItemTotal
            if ($step -eq 4 -and $name) {
                if ($v.item -ne $name) { $v.item = $name; $v.p2 = 0.0 }
                if ($ui.ItemState -eq 'checking') { $item.Text = "Checking $name"; $itemR.Text = ''; $v.indet = $true }
                elseif ($tot -gt 0) {
                    $item.Text = $name; $itemR.Text = "$(Size-Text ([double]$ui.ItemDone)) of $(Size-Text $tot)"
                    $v.indet = $false; $v.p2 += ([double]$ui.ItemDone / $tot - $v.p2) * 0.3
                } else { $item.Text = $name; $itemR.Text = ''; $v.indet = $true }
                $detail.Text = if ($ui.Items) { "Item $($ui.ItemIndex) of $($ui.Items)" } else { '' }
            } else {
                $item.Text = ''; $itemR.Text = ''; $v.indet = $true
                $detail.Text = [string]$ui.Detail
            }
            $v.m = ($v.m + 0.012) % 1.0
            $bars[1].Invalidate()
        }
        $timer = New-Object Windows.Forms.Timer; $timer.Interval = 50
        $timer.Add_Tick({ try { & $tick } catch { } })
        $btn1.Add_Click({
            if ($ui.State -eq 'running') {
                $r = [Windows.Forms.MessageBox]::Show($f, "Stop setup now?`n`nNothing is lost: run setup again later and it continues where it stopped.",
                    'Clover AI Studio Setup', 'YesNo', 'Question')
                if ($r -eq 'Yes' -and $ui.State -eq 'running') { $ui.Cancel = $true; $btn1.Enabled = $false; $btn1.Text = 'Stopping'; Stop-Tools }
            } else {
                if ($ui.State -eq 'done') { $ui.Start = $true }
                $f.Close()
            }
        })
        $btn2.Add_Click({ if ($ui.State -eq 'done') { $f.Close() } else { Start-Process notepad.exe -ArgumentList "`"$($ui.Log)`"" } })
        $f.Add_FormClosing({ param($s, $e) if ($ui.State -eq 'running') { $e.Cancel = $true; if ($btn1.Enabled) { $btn1.PerformClick() } } })
        $f.Add_Shown({ $ui.Shown = $true; $f.Activate(); $timer.Start() })
        [Windows.Forms.Application]::Run($f)
    } catch {
        $ui.Error = $_.Exception.Message
    } finally {
        $ui.Closed = $true
    }
}
function Start-SetupWindow {
    $t = [hashtable]::Synchronized(@{ Step = 0; StepText = 'Starting'; Plan = 0.0; Done = 0.0; Item = ''; ItemState = ''
        ItemDone = 0.0; ItemTotal = 0.0; ItemIndex = 0; Items = 0; Detail = ''; State = 'running'; Message = ''; Error = ''
        Cancel = $false; Shown = $false; Closed = $false; Start = $false; Log = $log; Version = $version
        Zip = (Join-Path $here 'app.zip') })
    $rs = [runspacefactory]::CreateRunspace(); $rs.ApartmentState = 'STA'; $rs.ThreadOptions = 'ReuseThread'; $rs.Open()
    $rs.SessionStateProxy.SetVariable('ui', $t)
    $ps = [powershell]::Create(); $ps.Runspace = $rs; [void]$ps.AddScript($windowScript)
    $t.PS = $ps; [void]$ps.BeginInvoke()
    $until = (Get-Date).AddSeconds(20)
    while (-not $t.Shown -and -not $t.Closed -and (Get-Date) -lt $until) { Start-Sleep -Milliseconds 100 }
    if (-not $t.Shown) { Log "the setup window did not open: $($t.Error)"; return $null }
    try {  # the window shows the progress; the console stays behind it, hidden
        Add-Type -Namespace CloverSetup -Name ConsoleWindow -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
'@
        [void][CloverSetup.ConsoleWindow]::ShowWindow([CloverSetup.ConsoleWindow]::GetConsoleWindow(), 0)
    } catch { }
    $t
}
function Finish([string]$state, [string]$msg) {  # the end of setup, in the window if there is one
    Log $msg
    if ($Quiet) { Write-Output $msg; return }
    if ($ui -and -not $ui.Closed) {
        $ui.Message = $msg; $ui.State = $state
        while (-not $ui.Closed) { Start-Sleep -Milliseconds 200 }
        if ($ui.Start) { Start-Process (Join-Path $Base 'CloverAIStudio.exe') -WorkingDirectory $Base }
        return
    }
    Add-Type -AssemblyName System.Windows.Forms
    [void][System.Windows.Forms.MessageBox]::Show($msg, 'Clover AI Studio Setup', 'OK', $(if ($state -eq 'done') { 'Information' } else { 'Error' }))
}
try { $host.UI.RawUI.WindowTitle = 'Clover AI Studio Setup' } catch { }
Log "setup $version start, base=$Base components=$Components reuse=$Reuse"
Write-Host "Clover AI Studio $version setup into $Base" -ForegroundColor Green
Write-Host 'Everything comes from its publisher and is checked before it is used. This takes a while the first time.'
$hasVenv = Test-Path (Join-Path $Base 'venv\Scripts\pythonw.exe')
if (-not $Quiet -and -not $hasVenv) { try { $ui = Start-SetupWindow } catch { Log "the setup window did not open: $($_.Exception.Message)"; $ui = $null } }

try {
    # ---------------------------------------------------------------- 1. this computer
    Step 'Checking this computer' 1
    Add-Type -Namespace Clover -Name Cpu -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint f);'
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    $ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
    $threads = [Environment]::ProcessorCount
    $build = [Environment]::OSVersion.Version.Build
    $avx2 = [Clover.Cpu]::IsProcessorFeaturePresent(40)
    New-Item -ItemType Directory -Force $Base | Out-Null
    $freeGb = [math]::Round((Get-PSDrive ([IO.Path]::GetPathRoot($Base).Substring(0, 1))).Free / 1GB, 1)
    $existing = Test-Path (Join-Path $Base 'venv\Scripts\python.exe')
    $low = @()
    if ($arch -ne 'AMD64') { $low += "a 64-bit Intel or AMD processor (this one is $arch)" }
    if ($build -lt 19041) { $low += "Windows 10 version 2004 or later (this is build $build)" }
    if ($ramGb -lt 15.0) { $low += "16 GB of memory (this computer has $ramGb GB)" }
    if ($threads -lt 8) { $low += "8 processor threads (this computer has $threads)" }
    if (-not $avx2) { $low += 'a processor with AVX2' }
    $needGb = if ($existing) { 8 } else { 25 }
    if ($freeGb -lt $needGb) { $low += "$needGb GB free on the drive of $Base ($freeGb GB free)" }
    Write-Host "   $threads threads, AVX2 $avx2, $ramGb GB memory, Windows build $build, $freeGb GB free"
    if ($low.Count) {
        Finish 'failed' ("Clover AI Studio needs at least the computer it was built on. This one is missing:`n`n- " + ($low -join "`n- "))
        exit 2
    }

    # llama.cpp and stable-diffusion.cpp are built with Visual C++: they need its runtime (Microsoft, signed)
    $vc = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue
    if (-not $vc -or $vc.Installed -ne 1) {
        Step 'Installing the Microsoft Visual C++ runtime (Windows asks for permission)' 1
        $vcExe = Join-Path $env:TEMP 'vc_redist.x64.exe'
        & curl.exe -sSfL --retry 5 -o $vcExe 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
        if ($LASTEXITCODE) { throw "could not download the Visual C++ runtime (curl exit $LASTEXITCODE)" }
        $sig = Get-AuthenticodeSignature $vcExe
        if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
            throw "the Visual C++ runtime installer is not signed by Microsoft ($($sig.Status)); not run"
        }
        if ($Quiet) { throw 'the Visual C++ runtime is missing and a quiet setup cannot ask Windows for permission' }
        $p = Start-Process $vcExe -ArgumentList '/install', '/quiet', '/norestart' -Verb RunAs -Wait -PassThru
        Log "vc_redist exit $($p.ExitCode)"
        if ($p.ExitCode -notin 0, 1638, 3010) { throw "the Visual C++ runtime did not install (exit $($p.ExitCode))" }
    }

    # ---------------------------------------------------------------- 2. the app, uv, Python, packages
    Step 'Installing the app' 2
    try {  # a running Studio keeps its files open
        Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8765/api/shutdown' -ContentType 'application/json' -Body '{}' -TimeoutSec 5 | Out-Null
        Start-Sleep -Seconds 3
    } catch { }
    # a Studio window still open keeps its launcher running, and the launcher keeps CloverAIStudio.exe open
    $launcher = Join-Path $Base 'CloverAIStudio.exe'
    Get-Process CloverAIStudio -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $launcher } | ForEach-Object {
        Log "stopping the open Studio's launcher (pid $($_.Id))"; Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    if (Test-Path (Join-Path $Base 'clover\server.py')) {  # an incomplete app is not worth a backup slot
        $bk = Join-Path $Base "backup\clover-$stamp.zip"
        New-Item -ItemType Directory -Force (Split-Path $bk) | Out-Null
        Compress-Archive -Path (Join-Path $Base 'clover') -DestinationPath $bk
        Write-Host "   previous app kept in $bk"
        Get-ChildItem (Split-Path $bk) -Filter 'clover-*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 2 |
            Remove-Item -Force
    }
    # unpacked whole first, then swapped in: the app and studio folders hold only what this package ships (the user's
    # data lives elsewhere in $Base). Expand-Archive is not used: on an error it deletes what it unpacked and hides why.
    $new = Join-Path $Base 'setup-new'
    Remove-Tree $new
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory((Join-Path $here 'app.zip'), $new)
    foreach ($d in 'clover', 'studio') { Remove-Tree (Join-Path $Base $d); Move-Item (Join-Path $new $d) (Join-Path $Base $d) }
    Get-ChildItem $new -File | ForEach-Object { Copy-Item $_.FullName (Join-Path $Base $_.Name) -Force }
    Remove-Tree $new
    $man = Get-Content (Join-Path $Base 'clover\launch\manifest.json') -Raw | ConvertFrom-Json

    Step "uv $($man.uv.version) (installs Python and the packages)" 3
    $uvDir = Join-Path $Base 'tools\uv'
    $uv = Join-Path $uvDir 'uv.exe'
    $haveUv = $false
    if (Test-Path $uv) {
        $ErrorActionPreference = 'Continue'
        $haveUv = ((& $uv --version 2>$null | Out-String) -match [regex]::Escape($man.uv.version))
        $ErrorActionPreference = 'Stop'
    }
    if (-not $haveUv) {
        $zip = Join-Path $Base 'downloads\uv-x86_64-pc-windows-msvc.zip'
        New-Item -ItemType Directory -Force (Split-Path $zip), $uvDir | Out-Null
        if (-not ((Test-Path $zip) -and ((Get-FileHash $zip -Algorithm SHA256).Hash -ieq $man.uv.sha256))) {
            & curl.exe -sSfL --retry 5 --retry-delay 5 -o $zip $man.uv.url
            if ($LASTEXITCODE) { throw "could not download uv (curl exit $LASTEXITCODE)" }
            if ((Get-FileHash $zip -Algorithm SHA256).Hash -ine $man.uv.sha256) { Remove-Item $zip; throw 'uv does not match its published SHA-256' }
        }
        Expand-Archive -Path $zip -DestinationPath $uvDir -Force
        Remove-Item $zip -Force
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Base 'runtime\python'
    $env:UV_CACHE_DIR = Join-Path $Base 'runtime\uv-cache'
    $env:UV_NO_CONFIG = '1'
    $env:UV_PYTHON_DOWNLOADS = 'automatic'

    Step "Python $($man.python)" 3
    # inside the install folder only: no python3.12.exe in the user's bin folder, no entry in the Windows registry
    Run $uv @('python', 'install', '--no-bin', '--no-registry', $man.python) 'installing Python'
    $py = Join-Path $Base 'venv\Scripts\python.exe'
    $pyVer = ''
    if (Test-Path $py) {
        $ErrorActionPreference = 'Continue'
        $pyVer = (& $py -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null | Out-String).Trim()
        $ErrorActionPreference = 'Stop'
    }
    if ($pyVer -ne $man.python) {
        Remove-Tree (Join-Path $Base 'venv')
        Run $uv @('venv', '--python', $man.python, (Join-Path $Base 'venv')) 'making the Python environment'
    }

    Step 'Python packages (locked; each file checked against its SHA-256)' 3
    $reqLock = Join-Path $Base 'studio\requirements.lock'
    $reqStamp = Join-Path $Base 'config\requirements.sha256'
    $reqHash = (Get-FileHash $reqLock -Algorithm SHA256).Hash
    $havePkgs = (Test-Path $reqStamp) -and ((Get-Content $reqStamp -Raw).Trim() -ieq $reqHash)
    if (-not $havePkgs) {
        $ErrorActionPreference = 'Continue'
        $impOk = (& $py -c "import numpy, openpyxl, pypdf, docx, pptx, PIL; print('OK')" 2>$null | Out-String).Trim()
        $ErrorActionPreference = 'Stop'
        if ($impOk -eq 'OK') {
            New-Item -ItemType Directory -Force (Split-Path $reqStamp) | Out-Null
            Set-Content -Path $reqStamp -Value $reqHash -Encoding Ascii
            $havePkgs = $true
        }
    }
    if (-not $havePkgs) {
        Run $uv @('pip', 'sync', '--python', $py, '--require-hashes', $reqLock) 'installing the Python packages'
        New-Item -ItemType Directory -Force (Split-Path $reqStamp) | Out-Null
        Set-Content -Path $reqStamp -Value $reqHash -Encoding Ascii
        $ErrorActionPreference = 'Continue'  # the download cache is not needed once the packages are installed
        & $uv cache clean 2>&1 | ForEach-Object { Log "$_" }
        $ErrorActionPreference = 'Stop'
    }

    # ---------------------------------------------------------------- 3. models, tools, data
    Step 'Models, tools and data' 4
    $env:PYTHONIOENCODING = 'utf-8'
    if ($Quiet) {
        $fetchArgs = @((Join-Path $Base 'clover\setup_fetch.py'), '--base', $Base, '--components', $Components)
        if ($Reuse) { $fetchArgs += @('--reuse', $Reuse) }
        Run $py $fetchArgs 'getting the models and tools'
        $dl = Join-Path $Base 'downloads'
        if ((Test-Path $dl) -and -not (Get-ChildItem $dl -Force)) { Remove-Item $dl -Force }
        $inst = Get-Content (Join-Path $Base 'config\installed.json') -Raw | ConvertFrom-Json
        $problems = @()
        foreach ($c in $inst.components.PSObject.Properties) { if (-not $c.Value.ok) { $problems += "$($c.Name): " + ($c.Value.problems -join '; ') } }
    } else {
        # Ensure MinGit is unpacked so server.py's repository starts cleanly, then walk the user through web/setup.html
        if (-not (Test-Path (Join-Path $Base 'tools\git\cmd\git.exe')) -and -not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
            $gitItem = $man.archives | Where-Object { $_.dest -eq 'tools/git' } | Select-Object -First 1
            if ($gitItem) {
                $gz = Join-Path $Base 'downloads\MinGit.zip'
                $gd = Join-Path $Base 'tools\git'
                New-Item -ItemType Directory -Force (Split-Path $gz), $gd | Out-Null
                & curl.exe -sSfL --retry 5 -o $gz $gitItem.url
                if (-not $LASTEXITCODE -and ((Get-FileHash $gz -Algorithm SHA256).Hash -ieq $gitItem.sha256)) {
                    [IO.Compression.ZipFile]::ExtractToDirectory($gz, $gd)
                    Set-Content -Path (Join-Path $gd '.clover-archive') -Value $gitItem.sha256 -Encoding Ascii
                }
                Remove-Item $gz -Force -ErrorAction SilentlyContinue
            }
        }
        $problems = @()
        New-Item -ItemType Directory -Force (Join-Path $Base 'config') | Out-Null
        Set-Content -Path (Join-Path $Base 'config\setup-walkthrough.flag') -Value $version -Encoding Ascii
    }

    # ---------------------------------------------------------------- 4. this computer's settings
    Step 'Reading the hardware' 5
    $ErrorActionPreference = 'Continue'
    $hwJson = & $py (Join-Path $Base 'clover\hardware.py') $Base 2>$null | Out-String
    $ErrorActionPreference = 'Stop'
    Log $hwJson
    $hw = $hwJson | ConvertFrom-Json
    $runs = if ($hw.chosen.engine -eq 'gpu') { "the GPU ($($hw.chosen.gpu))" } else { 'the CPU (no GPU llama.cpp can use was found)' }
    Write-Host "   Clover will run on $runs, context $($hw.chosen.ctx) tokens"

    # ---------------------------------------------------------------- 5. shortcuts, VS Code, uninstall entry
    $extMsg = ''
    if (-not $NoShortcuts) {
        Step 'Shortcuts and the VS Code extension' 6
        $exe = Join-Path $Base 'CloverAIStudio.exe'
        $ws = New-Object -ComObject WScript.Shell
        function New-Link([string]$path, [string]$arguments, [string]$desc) {
            $l = $ws.CreateShortcut($path)
            $l.TargetPath = $exe; $l.Arguments = $arguments; $l.WorkingDirectory = $Base
            $l.IconLocation = "$exe,0"; $l.Description = $desc; $l.Save()
        }
        $programs = [Environment]::GetFolderPath('Programs')
        $desktop = [Environment]::GetFolderPath('Desktop')
        New-Link (Join-Path $programs 'Clover AI Studio.lnk') '' 'Clover AI Studio (on this computer)'
        New-Link (Join-Path $programs 'Clover Engine for VS Code.lnk') '--engine-only' 'Start the Clover engine for VS Code'
        New-Link (Join-Path $desktop 'Clover AI Studio.lnk') '' 'Clover AI Studio (on this computer)'
        $extMsg = 'VS Code was not found, so the VS Code extension was not installed.'
        $code = @("$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd", "$env:ProgramFiles\Microsoft VS Code\bin\code.cmd") |
                Where-Object { Test-Path $_ } | Select-Object -First 1
        $vsix = Get-ChildItem (Join-Path $Base 'studio') -Filter 'clover-local-*.vsix' -ErrorAction SilentlyContinue |
                Sort-Object { [version]($_.BaseName -replace '^clover-local-', '') } -Descending | Select-Object -First 1
        if ($code -and $vsix) {
            $ErrorActionPreference = 'Continue'
            $out = & $code --install-extension $vsix.FullName --force 2>&1 | Out-String
            $rc = $LASTEXITCODE
            $ErrorActionPreference = 'Stop'
            Log "code --install-extension rc=$rc $out"
            $extMsg = if ($rc -eq 0) { 'The VS Code extension is installed (reload open VS Code windows).' }
                      else { "The VS Code extension could not be installed (exit $rc)." }
        }
        $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CloverAIStudio'
        New-Item -Path $key -Force | Out-Null
        $props = @{
            DisplayName = 'Clover AI Studio'; DisplayVersion = $version; Publisher = 'Clover'; DisplayIcon = "$exe,0"
            InstallLocation = $Base; NoModify = 1; NoRepair = 1
            UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Base\studio\uninstall.ps1`""
        }
        foreach ($k in $props.Keys) { Set-ItemProperty -Path $key -Name $k -Value $props[$k] }
    }

    Log 'SETUP_DONE'
    if (-not $Quiet) {
        if ($ui -and -not $ui.Closed) {
            $ui.State = 'autoclose'
            for ($w = 0; $w -lt 20 -and -not $ui.Closed; $w++) { Start-Sleep -Milliseconds 100 }
        }
        Start-Process (Join-Path $Base 'CloverAIStudio.exe') -WorkingDirectory $Base
        exit 0
    }
    $note = if ($problems.Count) { "`n`nNot everything could be fetched (run setup again to retry):`n- " + ($problems -join "`n- ") } else { '' }
    $start = if ($NoShortcuts) { "Start it with $Base\CloverAIStudio.exe." } else { 'Start it from the Start menu or the desktop shortcut.' }
    Finish 'done' ("Clover AI Studio $version is installed in $Base.`n`nIt runs on $runs.`n`n$start$note" +
         $(if ($extMsg) { "`n`n$extMsg" } else { '' }))
    exit 0
} catch {
    if ($ui -and $ui.Cancel) {
        Log 'SETUP_STOPPED by the user'
        Finish 'failed' 'Setup was stopped. Nothing is lost: run setup again and it continues where it stopped.'
        exit 3
    }
    Log ('SETUP_FAILED ' + $_.Exception.Message)
    Log ($_.InvocationInfo.PositionMessage + "`n" + $_.ScriptStackTrace)
    Finish 'failed' ("Setup failed: " + $_.Exception.Message + "`n`nDetails: $log`n`nRun setup again to continue where it stopped.")
    exit 1
}
