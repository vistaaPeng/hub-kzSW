param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 8501
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ProjectDir

$EnvFile = Join-Path $ProjectDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $key, $value = $line.Split("=", 2)
            if (-not [Environment]::GetEnvironmentVariable($key, "Process")) {
                [Environment]::SetEnvironmentVariable($key, $value.Trim('"').Trim("'"), "Process")
            }
        }
    }
}

$env:PYTHONPATH = if ($env:RAG_PYTHONPATH) { $env:RAG_PYTHONPATH } else { "S:/condaEnvs/py312/Lib/site-packages" }
$env:HF_HOME = if ($env:RAG_HF_CACHE_DIR) { $env:RAG_HF_CACHE_DIR } else { "M:/huggingface_cache" }
$env:HF_HUB_OFFLINE = if ($env:RAG_HF_OFFLINE) { $env:RAG_HF_OFFLINE } else { "1" }
$env:RAG_HF_CACHE_DIR = $env:HF_HOME
$env:RAG_HF_OFFLINE = $env:HF_HUB_OFFLINE

$Python = if ($env:RAG_PYTHON) { $env:RAG_PYTHON } else { "S:/condaEnvs/py312/python.exe" }

# Kill old processes
Get-Job -Name "RAG_Backend" -ErrorAction SilentlyContinue | Stop-Job -PassThru | Remove-Job -Force
netstat -ano | Select-String ":$BackendPort" | ForEach-Object { $p = ($_ -split '\s+')[-1]; if ($p) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue } }

# Backend
if (-not $FrontendOnly) {
    Write-Host "Starting backend (FastAPI, port $BackendPort)..."

    $backendJob = Start-Job -Name "RAG_Backend" -ScriptBlock {
        param($py, $dir, $port, $pp, $hf, $offline)
        $env:PYTHONPATH = $pp
        $env:HF_HOME = $hf
        $env:HF_HUB_OFFLINE = $offline
        $env:RAG_HF_CACHE_DIR = $hf
        $env:RAG_HF_OFFLINE = $offline
        $env:PYTHONIOENCODING = "utf-8"
        Set-Location $dir
        & $py scripts/app.py --port $port --host 127.0.0.1 2>&1 | Out-File -FilePath "$dir/logs/backend.log" -Encoding UTF8
    } -ArgumentList $Python, $ProjectDir, $BackendPort, $env:PYTHONPATH, $env:HF_HOME, $env:HF_HUB_OFFLINE

    Write-Host "   Waiting for backend (max 90s)..." -NoNewline
    $ready = $false
    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 2
        Write-Host "." -NoNewline
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($r.status -eq "ok") { $ready = $true; break }
        } catch {}
    }
    Write-Host ""

    if (-not $ready) {
        Write-Host "Backend start FAILED. Check logs: type logs\backend.log" -ForegroundColor Red
        if ($backendJob.State -eq "Failed") {
            Write-Host "Job error: $($backendJob.ChildJobs[0].Error)"
        }
    } else {
        Write-Host "Backend ready: http://127.0.0.1:$BackendPort"
    }

    if ($BackendOnly) {
        Write-Host "Backend running. Ctrl+C to stop"
        try { Wait-Event } catch {}
        return
    }
}

# Frontend
if (-not $BackendOnly) {
    Write-Host "Starting frontend (StreamLit, port $FrontendPort)..."
    Write-Host "   Open: http://localhost:$FrontendPort"
    Write-Host "   Ctrl+C to stop all services"
    Write-Host ""

    try {
        & $Python -m streamlit run web/app.py `
            --server.port $FrontendPort `
            --server.address 127.0.0.1 `
            --browser.serverAddress localhost `
            --theme.base dark `
            --server.headless true
    } finally {
        if ($backendJob -and $backendJob.State -eq "Running") {
            Write-Host "Stopping backend..."
            Stop-Job -Name "RAG_Backend"
            Remove-Job -Name "RAG_Backend" -Force
        }
    }
}
