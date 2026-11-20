# MJPEG-in-Matroska player / desktop streamer.
#
# Examples:
#   .\tools\mjpeg-player.ps1 play  -File .\out.mkv
#   .\tools\mjpeg-player.ps1 stream                       # desktop -> udp://127.0.0.1:3322
#   .\tools\mjpeg-player.ps1 stream -Seconds 10 -Quality 3 -Url udp://127.0.0.1:3322
#
# The receiver is started first so it binds the UDP port before the sender.
#
# gdigrab's -framerate is a requested upper bound, not a clock: on a 60 Hz
# desktop, 60 yields ~58 fps while 120 yields a steady ~60 fps. 1280x720
# MJPEG q=5 at 60 fps is ~34 Mbit/s and encodes at >20x realtime, so capture
# is the only real-time constraint.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('play', 'stream', 'bench')]
    [string] $Mode,

    [string] $File,

    [string] $Url = 'udp://127.0.0.1:3322',

    [ValidateRange(1, 240)]
    [int] $Framerate = 120,

    [ValidateRange(2, 31)]
    [int] $Quality = 5,

    [int] $Seconds = 0,

    # 0 = full desktop resolution. Scaling down is optional.
    [ValidateRange(0, 3840)]
    [int] $Width = 0,

    # normal = ffplay defaults (~0.4 s video queue at 720p60).
    # low    = no buffer + small probe (~0.07 s queue).
    # ultra  = low + video-clock sync.
    [ValidateSet('normal', 'low', 'ultra')]
    [string] $Latency = 'low',

    # Drop late frames instead of showing them slightly behind. Off by
    # default: the timing path uses wall-clock capture timestamps and
    # genpts so frames should not be late on an unloaded machine.
    [switch] $FrameDrop,

    [string] $Ffmpeg = 'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
    [string] $Ffplay = 'C:\Program Files\ffmpeg\bin\ffplay.exe'
)

# Native tools write informational text to stderr; don't turn that into a
# terminating PowerShell error. Explicit `throw` calls handle real failures.
$ErrorActionPreference = 'Continue'

# ffplay/ffmpeg do not report an opened UDP input until the first packet
# arrives, so poll the OS socket table instead of the log to avoid a send
# race (symptom: ffmpeg exits with AVERROR -10040, connection refused).
function Wait-UdpBound([string]$url, $proc) {
    $m = [regex]::Match($url, ':\s*(\d+)')
    if (-not $m.Success) { return $true }
    $port = [int]$m.Groups[1].Value
    for ($i = 0; $i -lt 100; $i++) {
        if ($proc -and $proc.HasExited) { return $false }
        if (Get-NetUDPEndpoint -LocalPort $port -ErrorAction SilentlyContinue) { return $true }
        Start-Sleep -Milliseconds 100
    }
    return $false
}

foreach ($exe in @($Ffmpeg, $Ffplay)) {
    if (-not (Test-Path $exe)) {
        throw "missing executable: $exe"
    }
}

if ($Mode -eq 'play') {
    if (-not $File -or -not (Test-Path $File)) {
        throw "play needs an existing -File (MJPEG in .mkv/.avi)"
    }
    Write-Host "Playing $File"
    & $Ffplay -autoexit $File
    exit $LASTEXITCODE
}

if ($Mode -eq 'bench') {
    # Decode the UDP stream with ffmpeg instead of a window so the frame
    # counter in its log tells us the delivered rate exactly.
    $recvLog = Join-Path ([System.IO.Path]::GetTempPath()) "mjxl-recv-$PID.log"
    $recvArgs = @('-hide_banner', '-loglevel', 'info', '-f', 'matroska',
                  '-i', $Url, '-f', 'null', '-')
    $recv = Start-Process -FilePath $Ffmpeg -ArgumentList $recvArgs -PassThru `
        -RedirectStandardError $recvLog -RedirectStandardOutput "$recvLog.out"
    try {
        if (-not (Wait-UdpBound $Url $recv)) { throw "receiver did not bind $Url within 10s" }
        $senderArgs = @(
            '-hide_banner', '-loglevel', 'error',
            '-probesize', $(if ($Latency -eq 'normal') { '1M' } else { '32k' })
        )
        if ($Latency -ne 'normal') {
            # gdigrab's own PTS can drift; wall-clock timestamps keep the
            # player's clock aligned with capture time so frames aren't late.
            $senderArgs += @('-use_wallclock_as_timestamps', '1')
        }
        $senderArgs += @(
            '-f', 'gdigrab', '-framerate', "$Framerate", '-i', 'desktop'
        )
        if ($Width -gt 0) { $senderArgs += @('-vf', "scale=${Width}:-2") }
        $senderArgs += @('-c:v', 'mjpeg', '-q:v', "$Quality", '-f', 'matroska')
        if ($Latency -ne 'normal') {
            $senderArgs += @('-cluster_time_limit', '50', '-flush_packets', '1')
        }
        if ($Seconds -gt 0) { $senderArgs += @('-t', "$Seconds") }
        $senderArgs += "$Url`?pkt_size=65536"
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $Ffmpeg @senderArgs
        $sw.Stop()
    } finally {
        Start-Sleep -Seconds 1
        if (-not $recv.HasExited) { Stop-Process -Id $recv.Id -Force }
    }
    $lines = Get-Content $recvLog | Select-String 'frame=\s*(\d+)'
    $frames = if ($lines) { [int]$lines[-1].Matches[0].Groups[1].Value } else { 0 }
    $fps = if ($sw.Elapsed.TotalSeconds -gt 0) { $frames / $sw.Elapsed.TotalSeconds } else { 0 }
    $size = if ($Width -gt 0) { "scaled to $Width" } else { "full desktop" }
    Write-Host ("Delivered {0} frames in {1:N2}s -> {2:N1} fps ({3}, q={4})" -f `
        $frames, $sw.Elapsed.TotalSeconds, $fps, $size, $Quality)
    exit 0
}

# stream: bind the player, then push the Windows desktop as MJPEG/Matroska.
Write-Host "Starting player on $Url"
$playerArgs = @('-autoexit', '-loglevel', 'info')
if ($Latency -ne 'normal') {
    $playerArgs += @('-fflags', '+genpts+nobuffer', '-probesize', '32768',
                     '-analyzeduration', '0', '-fpsprobesize', '0')
    if ($Latency -eq 'ultra') {
        $playerArgs += @('-sync', 'video')
    }
}
if ($FrameDrop) {
    $playerArgs += @('-framedrop')
}
$playerArgs += @('-f', 'matroska', '-i', $Url)
$playerLog = Join-Path ([System.IO.Path]::GetTempPath()) "mjxl-player-$PID.log"
$player = Start-Process -FilePath $Ffplay -ArgumentList $playerArgs -PassThru `
    -RedirectStandardError $playerLog -RedirectStandardOutput "$playerLog.out"
try {
    if (-not (Wait-UdpBound $Url $player)) {
        $tail = if (Test-Path $playerLog) { Get-Content $playerLog -Tail 8 } else { '' }
        throw "player did not bind $Url within 10s`n$tail"
    }

    $senderArgs = @(
        '-hide_banner', '-loglevel', 'warning',
        '-probesize', $(if ($Latency -eq 'normal') { '1M' } else { '32k' })
    )
    if ($Latency -ne 'normal') {
        $senderArgs += @('-use_wallclock_as_timestamps', '1')
    }
    $senderArgs += @(
        '-f', 'gdigrab', '-framerate', "$Framerate", '-i', 'desktop'
    )
    if ($Width -gt 0) {
        $senderArgs += @('-vf', "scale=${Width}:-2")
    }
    $senderArgs += @(
        '-c:v', 'mjpeg', '-q:v', "$Quality",
        '-f', 'matroska'
    )
    if ($Latency -ne 'normal') {
        $senderArgs += @('-cluster_time_limit', '50', '-flush_packets', '1')
    }
    if ($Seconds -gt 0) {
        $senderArgs += @('-t', "$Seconds")
    }
    $senderArgs += "$Url`?pkt_size=65536"

    Write-Host "Streaming desktop -> $Url (gdigrab upper bound ${Framerate} fps, q=$Quality; Ctrl+C to stop)"
    & $Ffmpeg @senderArgs
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg sender failed with exit $LASTEXITCODE"
    }
    Write-Host 'Sender finished.'
} finally {
    if (-not $player.HasExited) {
        Stop-Process -Id $player.Id -Force
    }
}
