param(
    [Parameter(Mandatory = $true)][string]$Trajectory,
    [string]$OutputDirectory = ''
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
        $gray = 0
        for ($x = 700; $x -le 780; $x++) {
            for ($y = 380; $y -le 460; $y++) {
                $c = $bitmap.GetPixel($x, $y)
                if ($c.R -gt 210 -and $c.G -gt 180 -and $c.B -lt 80) { $yellow++ }
                if ($c.R -gt 85 -and $c.R -lt 170 -and
                    [Math]::Abs([int]$c.R - [int]$c.G) -lt 15 -and
                    [Math]::Abs([int]$c.G - [int]$c.B) -lt 25) { $gray++ }
            }
        }
        return [pscustomobject]@{ yellow = $yellow; gray = $gray }
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
$step = $episode.episode.steps[3711]
if ($step.cursor_x -ne 3713 -or $step.cursor_y -ne 1024 -or $step.cursor_z -ne 256) {
    throw 'Step 3711 cursor does not match the frozen gate-crossing frame.'
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repo ('runtime\replay-visibility-428\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
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
    [System.Windows.Forms.SendKeys]::SendWait('3711{ENTER}')
    Start-Sleep -Milliseconds 300

    # Orbit roughly toward +X. At this camera, the final gate wall is behind the agent.
    $rect = Get-ReplayRect $window
    [System.Windows.Forms.Cursor]::Position = [System.Drawing.Point]::new($rect.Left + 700, $rect.Top + 400)
    [ReplayCapture428]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)
    for ($drag = 1; $drag -le 12; $drag++) {
        [System.Windows.Forms.Cursor]::Position = [System.Drawing.Point]::new(
            $rect.Left + 700 + [int](87 * $drag / 12),
            $rect.Top + 400 + [int](73 * $drag / 12))
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
    $xrayPath = Join-Path $output 'xray.png'
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        Snapshot-Replay $window $normalPath
        $normal = Count-AgentPixels $normalPath
        if ($normal.gray -gt 4000) { break }
    }
    if ($normal.gray -le 4000) { throw 'Gate geometry did not finish rendering.' }
    Click-Replay $window 22 568
    Start-Sleep -Milliseconds 750
    Snapshot-Replay $window $xrayPath
    $xray = Count-AgentPixels $xrayPath
    $manifest = [ordered]@{
        issue = 428
        captured_at = (Get-Date).ToUniversalTime().ToString('o')
        source_sha = (git -C $repo rev-parse HEAD).Trim()
        trajectory_sha256 = $trajectoryHash
        world_identity_sha256 = 'ed3f7cf6a2ab67d5d9bb82537bfa8fa50638a599cabc7ded2213fd4c3cc20c05'
        binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
        step = 3711
        cursor = @(3713, 1024, 256)
        wall_x = 3712
        window_pixels = @(1216, 799)
        pixel_roi = @(700, 380, 780, 460)
        normal = [ordered]@{ sha256 = (Get-FileHash $normalPath -Algorithm SHA256).Hash; yellow_pixels = $normal.yellow; gray_pixels = $normal.gray }
        xray = [ordered]@{ sha256 = (Get-FileHash $xrayPath -Algorithm SHA256).Hash; yellow_pixels = $xray.yellow; gray_pixels = $xray.gray }
        fixed = ($normal.yellow -ge 300 -and $normal.yellow -eq $xray.yellow -and $normal.gray -gt 4000)
    }
    $manifestPath = Join-Path $output 'capture.json'
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Output $manifestPath
    if (-not $manifest.fixed) { throw "Visibility assertion failed: normal=$($normal.yellow), x-ray=$($xray.yellow)" }
} finally {
    if ($process -and -not $process.HasExited) { $process.CloseMainWindow() | Out-Null }
}
