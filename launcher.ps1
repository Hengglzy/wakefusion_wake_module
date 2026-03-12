# WakeFusion System Launcher (PowerShell)
# For Anaconda PowerShell Prompt
# Usage: Input 1 to start, 2 to stop

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "          WakeFusion Multi-modal Wake System" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if conda is available
$condaCheck = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCheck) {
    Write-Host "Error: Cannot find conda command. Please ensure Anaconda/Miniconda is installed." -ForegroundColor Red
    Write-Host "Tip: Please run this script in Anaconda PowerShell Prompt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Get conda base path (from current environment)
$condaBase = $env:CONDA_PREFIX
if (-not $condaBase) {
    # Try to get from conda info
    try {
        $condaInfo = conda info --json | ConvertFrom-Json
        $condaBase = $condaInfo.root_prefix
    } catch {
        # Fallback: try to find from conda executable
        $condaPath = (Get-Command conda).Source
        $condaBase = Split-Path (Split-Path $condaPath)
    }
}

# Build command to initialize conda in new PowerShell window
$condaInit = ""
if ($condaBase) {
    $condaExe = Join-Path $condaBase "Scripts\conda.exe"
    if (Test-Path $condaExe) {
        # Initialize conda in the new PowerShell session
        $condaInit = "& '$condaExe' shell.powershell hook | Out-String | Invoke-Expression; "
    }
}

# Menu
Write-Host "Please select an option:" -ForegroundColor Yellow
Write-Host "  1. Start all services" -ForegroundColor Green
Write-Host "  2. Stop all services" -ForegroundColor Red
Write-Host ""
$choice = Read-Host "Enter your choice (1 or 2)"

if ($choice -eq "1") {
    # Start all services
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Starting all services..." -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "[1/5] Starting Vision Service (env: wakefusion_vision)..." -ForegroundColor Green
    $visionCmd = "${condaInit}conda activate wakefusion_vision; python -m wakefusion.services.vision_service --fps 15"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $visionCmd -WindowStyle Normal
    Start-Sleep -Seconds 3
    
    Write-Host "[2/5] Starting Audio Service (env: wakefusion)..." -ForegroundColor Green
    $audioCmd = "${condaInit}conda activate wakefusion; python -m wakefusion.services.audio_service"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $audioCmd -WindowStyle Normal
    Start-Sleep -Seconds 3
    
    Write-Host "[3/5] Starting ASR Service (env: wakefusion_cosyvoice)..." -ForegroundColor Green
    $asrCmd = "${condaInit}conda activate wakefusion_cosyvoice; python -m wakefusion.services.asr_service"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $asrCmd -WindowStyle Normal
    Start-Sleep -Seconds 3
    
    Write-Host "[4/5] Starting TTS Service (env: wakefusion_cosyvoice)..." -ForegroundColor Green
    $ttsCmd = "${condaInit}conda activate wakefusion_cosyvoice; python -m wakefusion.services.tts_service"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $ttsCmd -WindowStyle Normal
    Start-Sleep -Seconds 3
    
    Write-Host "[5/5] Starting Core Server (env: wakefusion)..." -ForegroundColor Green
    $coreCmd = "${condaInit}conda activate wakefusion; python -m wakefusion.services.core_server"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $coreCmd -WindowStyle Normal
    Start-Sleep -Seconds 2
    
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "All services started!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Service List:" -ForegroundColor Yellow
    Write-Host "  - Vision Service (Port: 5555)"
    Write-Host "  - Audio Service (Port: 5556, 5557)"
    Write-Host "  - ASR Service  (Port: 5558, 8766)"
    Write-Host "  - TTS Service  (Port: 5559, 8767)"
    Write-Host "  - Core Server  (Port: 5561)"
    Write-Host ""
    Write-Host "Usage Tips:" -ForegroundColor Yellow
    Write-Host "  - Each service runs in a separate window"
    Write-Host "  - Close the window to stop that service"
    Write-Host "  - Closing this window will NOT stop services"
    Write-Host "  - Run this script again and select option 2 to stop all services"
    Write-Host ""
    
} elseif ($choice -eq "2") {
    # Stop all services
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Stopping all services..." -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Finding and stopping all WakeFusion service processes..." -ForegroundColor Yellow
    Write-Host ""
    
    # Stop all related PowerShell processes (containing wakefusion services)
    $processes = Get-Process | Where-Object {
        $_.ProcessName -eq "powershell" -or $_.ProcessName -eq "python"
    }
    
    $stopped = 0
    foreach ($proc in $processes) {
        try {
            $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
            if ($cmdline -match "wakefusion\.services\.(vision_service|audio_service|asr_service|tts_service|core_server)") {
                $serviceName = if ($cmdline -match "vision_service") { "Vision Service" }
                              elseif ($cmdline -match "audio_service") { "Audio Service" }
                              elseif ($cmdline -match "asr_service") { "ASR Service" }
                              elseif ($cmdline -match "tts_service") { "TTS Service" }
                              elseif ($cmdline -match "core_server") { "Core Server" }
                              else { "Unknown Service" }
                
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                Write-Host "Stopped $serviceName (PID: $($proc.Id))" -ForegroundColor Green
                $stopped++
            }
        } catch {
            # Ignore processes that cannot be accessed
        }
    }
    
    if ($stopped -eq 0) {
        Write-Host "No running services found" -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Stopped $stopped service(s)" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Cyan
    }
    Write-Host ""
    
} else {
    Write-Host "Invalid choice. Please run the script again and select 1 or 2." -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to close this window"
