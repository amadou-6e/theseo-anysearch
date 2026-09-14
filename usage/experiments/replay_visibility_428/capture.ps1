param(
    [Parameter(Mandatory = $true)][string]$Trajectory,
    [string]$OutputDirectory = '',
    [ValidateSet('AgentFront', 'WallFront')][string]$View = 'AgentFront',
    [int]$WallFrontOrbitX = -196
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ReplayCapture428 {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out RECT rect);
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr window, int x, int y, int width, int height, bool repaint);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint x, uint y, uint data, UIntPtr extra);
}
'@

function Get-ReplayRect([IntPtr]$Window) {
    $rect = [ReplayCapture428+RECT]::new()
    if (-not [ReplayCapture428]::GetWindowRect($Window, [ref]$rect)) { throw 'Cannot read replay window.' }
    return $rect
}

function Click-Replay([IntPtr]$Window, [int]$X, [int]$Y) {
    $rect = Get-ReplayRect $Window
    [System.Windows.Forms.Cursor]::Position = [System.Drawing.Point]::new($rect.Left + $X, $rect.Top + $Y)
    [ReplayCapture428]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)
    [ReplayCapture428]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)
}

function Snapshot-Replay([IntPtr]$Window, [string]$Path) {
    $rect = Get-ReplayRect $Window
    if (($rect.Right - $rect.Left) -ne 1216 -or ($rect.Bottom - $rect.Top) -ne 799) {
        throw 'Expected a 1216x799 window at 100% display scaling.'
    }
    $display = [System.Windows.Forms.Screen]::FromHandle($Window).Bounds
    if ($display.Right -ge ($rect.Right + 16)) {
        $park = [System.Drawing.Point]::new($rect.Right + 12, $rect.Top + 20)
    } elseif ($display.Bottom -ge ($rect.Bottom + 16)) {
        $park = [System.Drawing.Point]::new($rect.Left + 20, $rect.Bottom + 12)
    } else {
        throw 'Need desktop space outside the replay window to park the pointer.'
    }
    [System.Windows.Forms.Cursor]::Position = $park
    Start-Sleep -Milliseconds 100
    $bitmap = [System.Drawing.Bitmap]::new(1216, 799)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, [System.Drawing.Size]::new(1216, 799))
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

function Count-AgentPixels([string]$Path) {
    $bitmap = [System.Drawing.Bitmap]::new($Path)
    try {
        $yellow = 0
        $mutedYellow = 0
        $gray = 0
        $badge = 0
        for ($x = 700; $x -le 780; $x++) {
            for ($y = 380; $y -le 460; $y++) {
                $c = $bitmap.GetPixel($x, $y)
                if ($c.R -gt 210 -and $c.G -gt 180 -and $c.B -lt 80) { $yellow++ }
                if ($c.R -gt ($c.B + 15) -and $c.G -gt ($c.B + 10)) { $mutedYellow++ }
                if ($c.R -gt 85 -and $c.R -lt 170 -and
                    [Math]::Abs([int]$c.R - [int]$c.G) -lt 15 -and
                    [Math]::Abs([int]$c.G - [int]$c.B) -lt 25) { $gray++ }
            }
        }
        for ($x = 290; $x -le 500; $x++) {
            for ($y = 70; $y -le 95; $y++) {
                $c = $bitmap.GetPixel($x, $y)
                if ($c.R -gt 220 -and $c.G -gt 170 -and $c.B -lt 130) { $badge++ }
            }
        }
        return [pscustomobject]@{ yellow = $yellow; muted_yellow = $mutedYellow; gray = $gray; badge = $badge }
    } finally { $bitmap.Dispose() }
}

$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$trajectoryPath = [IO.Path]::GetFullPath($Trajectory)
$binary = Join-Path $repo 'theseo_anysearch\core\target\debug\voxel-replay.exe'
if (-not (Test-Path -LiteralPath $binary)) { throw 'Build voxel-replay first.' }
if (-not (Test-Path -LiteralPath $trajectoryPath)) { throw 'Trajectory not found.' }
$trajectoryHash = (Get-FileHash -LiteralPath $trajectoryPath -Algorithm SHA256).Hash
if ($trajectoryHash -ne 'FDA5337188B31AD10A50F0218B7C597DD555710F3B65D213EE9288D02DFB221A') {
    throw "Wrong stage-11 trajectory: $trajectoryHash"
}
$episode = Get-Content -LiteralPath $trajectoryPath -Raw | ConvertFrom-Json
$stepIndex = if ($View -eq 'AgentFront') { 3711 } else { 3713 }
$expectedX = $stepIndex + 2
$step = $episode.episode.steps[$stepIndex]
if ($step.cursor_x -ne $expectedX -or $step.cursor_y -ne 1024 -or $step.cursor_z -ne 256) {
    throw "Step $stepIndex cursor does not match the frozen gate-crossing frame."
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repo ('runtime\replay-visibility-428\' + $View + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
$output = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $output -Force | Out-Null
$process = Start-Process -FilePath $binary -ArgumentList ('"' + $trajectoryPath + '"') -WorkingDirectory $repo -PassThru
try {
    $window = [IntPtr]::Zero
    for ($attempt = 0; $attempt -lt 80; $attempt++) {
        Start-Sleep -Milliseconds 250
        $process.Refresh()
        if ($process.HasExited) { throw 'Replayer exited before its window appeared.' }
        if ($process.MainWindowHandle -ne [IntPtr]::Zero -and
            $process.MainWindowTitle.StartsWith('Voxel Replay --')) {
            $window = $process.MainWindowHandle
            break
        }
    }
    if ($window -eq [IntPtr]::Zero) { throw 'Timed out waiting for replay window.' }
    if (-not [ReplayCapture428]::MoveWindow($window, 0, 0, 1216, 799, $true)) { throw 'Cannot size window.' }
    [ReplayCapture428]::SetForegroundWindow($window) | Out-Null
    Start-Sleep -Milliseconds 300
    Click-Replay $window 144 346
    [System.Windows.Forms.SendKeys]::SendWait('^a')
    [System.Windows.Forms.SendKeys]::SendWait('{BACKSPACE}')
    if ($View -eq 'AgentFront') {
        [System.Windows.Forms.SendKeys]::SendWait('3711{ENTER}')
    } else {
        [System.Windows.Forms.SendKeys]::SendWait('3713{ENTER}')
    }
    Start-Sleep -Milliseconds 300

    # At AgentFront, orbit toward +X: the final gate wall is behind the agent.
    # At WallFront, orbit toward -X with an oblique view: solid wall voxels
    # should occlude the agent, while x-ray shows its hidden location.
    $orbitX = if ($View -eq 'AgentFront') { 87 } else { $WallFrontOrbitX }
    $orbitY = if ($View -eq 'AgentFront') { 73 } else { 0 }
    $rect = Get-ReplayRect $window
    [System.Windows.Forms.Cursor]::Position = [System.Drawing.Point]::new($rect.Left + 700, $rect.Top + 400)
    [ReplayCapture428]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)
    for ($drag = 1; $drag -le 12; $drag++) {
        [System.Windows.Forms.Cursor]::Position = [System.Drawing.Point]::new(
            $rect.Left + 700 + [int]($orbitX * $drag / 12),
            $rect.Top + 400 + [int]($orbitY * $drag / 12))
        Start-Sleep -Milliseconds 35
    }
    [ReplayCapture428]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 300
    [System.Windows.Forms.Cursor]::Position = [System.Drawing.Point]::new($rect.Left + 738, $rect.Top + 420)
    for ($notch = 0; $notch -lt 5; $notch++) {
        [ReplayCapture428]::mouse_event(0x0800, 0, 0, 120, [UIntPtr]::Zero)
        Start-Sleep -Milliseconds 90
    }

    $normalPath = Join-Path $output 'normal.png'
    $diagnosticPath = Join-Path $output 'diagnostic.png'
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        Snapshot-Replay $window $normalPath
        $normal = Count-AgentPixels $normalPath
        if ($normal.gray -gt 4000) { break }
    }
    if ($normal.gray -le 4000) { throw 'Gate geometry did not finish rendering.' }
    # The large compiled-world trajectory has no trail, so Show trail is absent.
    # Reveal hidden agent is the first checkbox under Render diagnostics.
    Click-Replay $window 22 662
    Start-Sleep -Milliseconds 750
    Snapshot-Replay $window $diagnosticPath
    $diagnostic = Count-AgentPixels $diagnosticPath
    $manifest = [ordered]@{
        issue = 428
        view = $View
        orbit_drag_pixels = @($orbitX, $orbitY)
        captured_at = (Get-Date).ToUniversalTime().ToString('o')
        source_sha = (git -C $repo rev-parse HEAD).Trim()
        trajectory_sha256 = $trajectoryHash
        world_identity_sha256 = 'ed3f7cf6a2ab67d5d9bb82537bfa8fa50638a599cabc7ded2213fd4c3cc20c05'
        binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
        step = $stepIndex
        cursor = @($expectedX, 1024, 256)
        wall_x = 3712
        window_pixels = @(1216, 799)
        pixel_roi = @(700, 380, 780, 460)
        normal = [ordered]@{ sha256 = (Get-FileHash $normalPath -Algorithm SHA256).Hash; yellow_pixels = $normal.yellow; muted_yellow_pixels = $normal.muted_yellow; gray_pixels = $normal.gray; badge_pixels = $normal.badge }
        diagnostic = [ordered]@{ sha256 = (Get-FileHash $diagnosticPath -Algorithm SHA256).Hash; yellow_pixels = $diagnostic.yellow; muted_yellow_pixels = $diagnostic.muted_yellow; gray_pixels = $diagnostic.gray; badge_pixels = $diagnostic.badge }
        fixed = if ($View -eq 'AgentFront') {
            $normal.yellow -ge 300 -and $diagnostic.yellow -ge 300 -and $normal.gray -gt 4000 -and
                $normal.badge -eq 0 -and $diagnostic.badge -ge 20
        } else {
            $normal.yellow -eq 0 -and $normal.muted_yellow -eq 0 -and $diagnostic.yellow -ge 150 -and $normal.gray -gt 4000 -and
                $normal.badge -eq 0 -and $diagnostic.badge -ge 20
        }
    }
    $manifestPath = Join-Path $output 'capture.json'
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Output $manifestPath
    if (-not $manifest.fixed) {
        throw "Visibility assertion failed: normal=$($normal.yellow), diagnostic=$($diagnostic.yellow), badge=$($diagnostic.badge)"
    }
} finally {
    if ($process -and -not $process.HasExited) { $process.CloseMainWindow() | Out-Null }
}
