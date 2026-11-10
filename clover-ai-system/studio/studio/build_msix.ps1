# Builds the Microsoft Store MSIX package (SanthoshNarayanan.CloverAIStudio_5.0.3.0_x64.msix),
# the Partner Center .msixupload bundle (clover-ai-studio-beta.msixupload),
# and copies the standalone installer (clover-ai-studio-beta-setup.exe) into release/beta/windows/.
$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
$base = Split-Path -Parent $src
$wt = [IO.Path]::GetFullPath((Join-Path $base '..\..'))
if (-not (Test-Path (Join-Path $wt 'release\beta\windows'))) {
    $wt = [IO.Path]::GetFullPath((Join-Path $base '..\..\..\clover-collective-wt'))
}
$outWin = Join-Path $wt 'release\beta\windows'
New-Item -ItemType Directory -Force $outWin | Out-Null

# 1. Ensure CloverAIStudio.exe and CloverAIStudio-Setup.exe are freshly built
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $src 'build_setup.ps1')
if ($LASTEXITCODE) { throw "build_setup.ps1 failed ($LASTEXITCODE)" }

$makeappx = Get-ChildItem "$env:TEMP\winsdk-buildtools\bin\*\x64\makeappx.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
$signtool = Get-ChildItem "$env:TEMP\winsdk-buildtools\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $makeappx) { throw "makeappx.exe not found in $env:TEMP\winsdk-buildtools" }
if (-not $signtool) { throw "signtool.exe not found in $env:TEMP\winsdk-buildtools" }

$msixBuild = Join-Path $env:TEMP 'clover-msix-stage'
if (Test-Path $msixBuild) { Remove-Item $msixBuild -Recurse -Force }
$layout = Join-Path $msixBuild 'layout'
$assets = Join-Path $layout 'Assets'
New-Item -ItemType Directory -Force $assets | Out-Null

# 2. Generate crisp 5-petal yellow Clover flower MSIX & Store visual assets using the canonical brand master icon
Add-Type -AssemblyName System.Drawing
$yellowIconPack = Join-Path $wt 'assets\clover-yellow-flower-icon-pack'
$brandMasterPath = Join-Path $yellowIconPack 'brand\clover-app-icon-master.png'
$brandCirclePath = Join-Path $yellowIconPack 'brand\clover-circle.png'

function New-CloverAsset([string]$path, [int]$w, [int]$h, [bool]$transparentBg = $false) {
    $bmp = New-Object System.Drawing.Bitmap($w, $h, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

    if ($transparentBg) {
        $g.Clear([System.Drawing.Color]::Transparent)
    } else {
        $bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
            (New-Object System.Drawing.Point(0, 0)),
            (New-Object System.Drawing.Point($w, $h)),
            [System.Drawing.Color]::FromArgb(255, 14, 17, 23),
            [System.Drawing.Color]::FromArgb(255, 22, 26, 35)
        )
        $g.FillRectangle($bgBrush, 0, 0, $w, $h)
        $bgBrush.Dispose()
    }

    if ((Test-Path $brandMasterPath) -and -not $transparentBg -and ($w -eq $h)) {
        $srcImg = [System.Drawing.Image]::FromFile($brandMasterPath)
        $g.DrawImage($srcImg, 0, 0, $w, $h)
        $srcImg.Dispose()
    } elseif (Test-Path $brandCirclePath) {
        $srcImg = [System.Drawing.Image]::FromFile($brandCirclePath)
        $side = [int]([Math]::Min($w, $h) * $(if ($transparentBg) { 0.92 } else { 0.78 }))
        $dx = [int](($w - $side) / 2)
        $dy = [int](($h - $side) / 2)
        $g.DrawImage($srcImg, $dx, $dy, $side, $side)
        $srcImg.Dispose()
    } else {
        $cx = $w / 2.0
        $cy = $h / 2.0
        $scale = [Math]::Min($w, $h) * 0.34
        $colors = @(
            [System.Drawing.Color]::FromArgb(255, 255, 240, 120),
            [System.Drawing.Color]::FromArgb(255, 248, 207, 30),
            [System.Drawing.Color]::FromArgb(255, 255, 240, 120),
            [System.Drawing.Color]::FromArgb(255, 216, 150, 6),
            [System.Drawing.Color]::FromArgb(255, 248, 207, 30)
        )
        for ($i = 0; $i -lt 5; $i++) {
            $state = $g.Save()
            $g.TranslateTransform($cx, $cy)
            $g.RotateTransform($i * 72.0)
            $leafW = [float]($scale * 0.54)
            $leafH = [float]($scale * 0.88)
            $brush = New-Object System.Drawing.SolidBrush($colors[$i])
            $g.FillEllipse($brush, [float](-$leafW / 2.0), [float](-$leafH - ($scale * 0.06)), $leafW, $leafH)
            $brush.Dispose()
            $g.Restore($state)
        }
        $hubR = [float]([Math]::Max(2.0, $scale * 0.14))
        $hubBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 240, 120))
        $g.FillEllipse($hubBrush, [float]($cx - $hubR), [float]($cy - $hubR), [float]($hubR * 2.0), [float]($hubR * 2.0))
        $hubBrush.Dispose()
    }

    $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}

New-CloverAsset (Join-Path $assets 'StoreLogo.png') 50 50 $false
New-CloverAsset (Join-Path $assets 'Square44x44Logo.png') 44 44 $false
New-CloverAsset (Join-Path $assets 'Square44x44Logo.targetsize-24_altform-unplated.png') 24 24 $true
New-CloverAsset (Join-Path $assets 'Square71x71Logo.png') 71 71 $false
New-CloverAsset (Join-Path $assets 'Square150x150Logo.png') 150 150 $false
New-CloverAsset (Join-Path $assets 'Wide310x150Logo.png') 310 150 $false
New-CloverAsset (Join-Path $assets 'Square310x310Logo.png') 310 310 $false
New-CloverAsset (Join-Path $assets 'SplashScreen.png') 620 300 $false

# Also generate Google Play (512x512 icon, 1024x500 feature graphic) and Microsoft Store (300x300 icon) assets
$gpScreens = Join-Path $wt 'release\beta\store\google-play\screenshots'
$msScreens = Join-Path $wt 'release\beta\store\microsoft\screenshots'
New-Item -ItemType Directory -Force $gpScreens, $msScreens | Out-Null
$gpYellow512 = Join-Path $yellowIconPack 'mobile\android\ic_launcher-512.png'
if (Test-Path $gpYellow512) {
    Copy-Item $gpYellow512 (Join-Path $gpScreens 'app-icon-512x512.png') -Force
} else {
    New-CloverAsset (Join-Path $gpScreens 'app-icon-512x512.png') 512 512 $false
}
New-CloverAsset (Join-Path $msScreens 'app-icon-300x300.png') 300 300 $false

function New-FeatureGraphic([string]$path, [int]$w, [int]$h) {
    $bmp = New-Object System.Drawing.Bitmap($w, $h, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $bg = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
        (New-Object System.Drawing.Point(0, 0)),
        (New-Object System.Drawing.Point($w, $h)),
        [System.Drawing.Color]::FromArgb(255, 14, 17, 23),
        [System.Drawing.Color]::FromArgb(255, 24, 28, 38)
    )
    $g.FillRectangle($bg, 0, 0, $w, $h)
    $bg.Dispose()
    $cx = $w * 0.18
    $cy = $h / 2.0
    if (Test-Path $brandCirclePath) {
        $srcImg = [System.Drawing.Image]::FromFile($brandCirclePath)
        $side = [int]($h * 0.64)
        $g.DrawImage($srcImg, [int]($cx - ($side / 2.0)), [int]($cy - ($side / 2.0)), $side, $side)
        $srcImg.Dispose()
    } else {
        $sc = $h * 0.30
        $cols = @(
            [System.Drawing.Color]::FromArgb(255, 255, 240, 120),
            [System.Drawing.Color]::FromArgb(255, 248, 207, 30),
            [System.Drawing.Color]::FromArgb(255, 255, 240, 120),
            [System.Drawing.Color]::FromArgb(255, 216, 150, 6),
            [System.Drawing.Color]::FromArgb(255, 248, 207, 30)
        )
        for ($i = 0; $i -lt 5; $i++) {
            $st = $g.Save()
            $g.TranslateTransform($cx, $cy)
            $g.RotateTransform($i * 72.0)
            $lw = [float]($sc * 0.54)
            $lh = [float]($sc * 0.88)
            $b = New-Object System.Drawing.SolidBrush($cols[$i])
            $g.FillEllipse($b, [float](-$lw / 2.0), [float](-$lh - ($sc * 0.06)), $lw, $lh)
            $b.Dispose()
            $g.Restore($st)
        }
        $hr = [float]([Math]::Max(3.0, $sc * 0.14))
        $hb = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 240, 120))
        $g.FillEllipse($hb, [float]($cx - $hr), [float]($cy - $hr), [float]($hr * 2.0), [float]($hr * 2.0))
        $hb.Dispose()
    }
    $f1 = New-Object System.Drawing.Font("Segoe UI Semibold", 36, [System.Drawing.FontStyle]::Bold)
    $f2 = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Regular)
    $f3 = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Regular)
    $tb1 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 248, 250, 252))
    $tb2 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 248, 207, 30))
    $tb3 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 160, 174, 192))
    $g.DrawString("Clover AI Studio", $f1, $tb1, [float]($w * 0.34), [float]($h * 0.27))
    $g.DrawString("Local-First Multimodal AI Assistant & Workspace", $f2, $tb2, [float]($w * 0.345), [float]($h * 0.49))
    $g.DrawString("7-Layer Architecture  ·  AGENTS.md Clover Cycles  ·  Zero Cloud Lock-In", $f3, $tb3, [float]($w * 0.345), [float]($h * 0.63))
    $f1.Dispose(); $f2.Dispose(); $f3.Dispose(); $tb1.Dispose(); $tb2.Dispose(); $tb3.Dispose()
    $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}
New-FeatureGraphic (Join-Path $gpScreens 'feature-graphic-1024x500.png') 1024 500

# 3. Copy application files into MSIX layout
$setupBuild = Join-Path $env:TEMP 'clover-setup-build'
Copy-Item (Join-Path $setupBuild 'CloverAIStudio.exe') (Join-Path $layout 'CloverAIStudio.exe') -Force
Copy-Item (Join-Path $base 'setup\CloverAIStudio-Setup.exe') (Join-Path $layout 'CloverAIStudio-Setup.exe') -Force
& robocopy (Join-Path $setupBuild 'app\clover') (Join-Path $layout 'clover') /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy clover failed ($LASTEXITCODE)" }
& robocopy (Join-Path $setupBuild 'app\studio') (Join-Path $layout 'studio') /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy studio failed ($LASTEXITCODE)" }

# 4. Write AppxManifest.xml with exact reserved Microsoft Partner Center identity
$manifestXml = @'
<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap rescap">

  <Identity
    Name="SanthoshNarayanan.CloverAIStudio"
    Publisher="CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF"
    Version="5.0.3.0"
    ProcessorArchitecture="x64" />

  <Properties>
    <DisplayName>Clover AI Studio</DisplayName>
    <PublisherDisplayName>Santhosh Narayanan</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
    <Description>Local-first multimodal AI assistant and developer workspace powered by the Clover 7-layer architecture.</Description>
  </Properties>

  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.19041.0" MaxVersionTested="10.0.26100.0" />
  </Dependencies>

  <Resources>
    <Resource Language="en-us" />
  </Resources>

  <Applications>
    <Application Id="CloverAIStudio" Executable="CloverAIStudio.exe" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="Clover AI Studio"
        Description="Clover AI Studio — Local-first multimodal AI assistant and developer workspace"
        BackgroundColor="#0E1117"
        Square150x150Logo="Assets\Square150x150Logo.png"
        Square44x44Logo="Assets\Square44x44Logo.png">
        <uap:DefaultTile Wide310x150Logo="Assets\Wide310x150Logo.png" Square310x310Logo="Assets\Square310x310Logo.png" Square71x71Logo="Assets\Square71x71Logo.png">
          <uap:ShowNameOnTiles>
            <uap:ShowOn Tile="square150x150Logo" />
            <uap:ShowOn Tile="wide310x150Logo" />
            <uap:ShowOn Tile="square310x310Logo" />
          </uap:ShowNameOnTiles>
        </uap:DefaultTile>
        <uap:SplashScreen Image="Assets\SplashScreen.png" BackgroundColor="#0E1117" />
      </uap:VisualElements>
    </Application>
  </Applications>

  <Capabilities>
    <Capability Name="internetClient" />
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
'@
[IO.File]::WriteAllText((Join-Path $layout 'AppxManifest.xml'), $manifestXml, (New-Object System.Text.UTF8Encoding($false)))

# 5. Build .msix using makeappx.exe
$msixName = 'SanthoshNarayanan.CloverAIStudio_5.0.3.0_x64.msix'
$msixPath = Join-Path $outWin $msixName
$msixBetaPath = Join-Path $outWin 'clover-ai-studio-beta.msix'
$msixUploadPath = Join-Path $outWin 'clover-ai-studio-beta.msixupload'
$exeBetaPath = Join-Path $outWin 'clover-ai-studio-beta-setup.exe'

foreach ($p in @($msixPath, $msixBetaPath, $msixUploadPath, $exeBetaPath)) {
    if (Test-Path $p) { Remove-Item $p -Force }
}

& $makeappx pack /d $layout /p $msixPath /o /nv
if ($LASTEXITCODE -or -not (Test-Path $msixPath)) { throw "makeappx pack failed ($LASTEXITCODE)" }

# Build .msixupload (Partner Center upload archive containing the .msix package)
$uploadStage = Join-Path $msixBuild 'upload'
New-Item -ItemType Directory -Force $uploadStage | Out-Null
Copy-Item $msixPath (Join-Path $uploadStage $msixName) -Force
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($uploadStage, $msixUploadPath, [IO.Compression.CompressionLevel]::Optimal, $false)

# Sign the local .msix with a test certificate matching CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF for local validation
$storeCert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert -ErrorAction SilentlyContinue |
             Where-Object { $_.Subject -eq 'CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF' -and $_.NotAfter -gt (Get-Date) } |
             Select-Object -First 1
if (-not $storeCert) {
    $storeCert = New-SelfSignedCertificate `
        -Type Custom `
        -Subject "CN=A0DCEEB8-49DB-4CD8-8D07-BC439E69B9AF" `
        -KeyUsage DigitalSignature `
        -FriendlyName "Clover AI Studio Store Test Cert" `
        -CertStoreLocation "Cert:\CurrentUser\My" `
        -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}") `
        -NotAfter (Get-Date).AddYears(5)
}
$pfxPass = ConvertTo-SecureString -String "clover-beta-test" -Force -AsPlainText
$pfxPath = Join-Path $msixBuild 'store-test.pfx'
Export-PfxCertificate -Cert $storeCert -FilePath $pfxPath -Password $pfxPass | Out-Null
& $signtool sign /fd SHA256 /f $pfxPath /p "clover-beta-test" $msixPath | Out-Null
Copy-Item $msixPath $msixBetaPath -Force
Copy-Item (Join-Path $base 'setup\CloverAIStudio-Setup.exe') $exeBetaPath -Force

Get-FileHash -Algorithm SHA256 $exeBetaPath, $msixBetaPath, $msixUploadPath, $msixPath |
    ForEach-Object { "$($_.Hash.ToLower())  $(Split-Path $_.Path -Leaf)" } |
    Set-Content -Path (Join-Path $outWin 'SHA256SUMS.txt') -Encoding ascii

Get-ChildItem $outWin | Select-Object Name, Length
Get-Content (Join-Path $outWin 'SHA256SUMS.txt')
