param(
    [string]$ModelId = "NousResearch/Hermes-3-Llama-3.2-1B",
    [string]$OutputDir = ".\\runs\\run_36h",
    [int]$ReportEverySec = 300
)

if (-not (Test-Path ".\\.venv\\Scripts\\python.exe")) {
    Write-Error "Virtual environment not found. Create it with: python -m venv .venv"
    exit 1
}

$pythonExe = ".\\.venv\\Scripts\\python.exe"
& $pythonExe .\\agentic_cpu_runner.py --model-id $ModelId --duration-hours 36 --report-every-sec $ReportEverySec --output-dir $OutputDir
