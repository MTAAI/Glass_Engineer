# Glass Expert AI — Full Stack Startup
# Run from project root

Write-Host "Starting Glass Expert AI..." -ForegroundColor Cyan

# Step 1: Start Docker services
Write-Host "Starting Docker services..." -ForegroundColor Yellow
docker-compose -f docker/docker-compose.yml up -d

# Step 2: Start model service (if weights exist)
$weightsPath = "models\glass-expert-v2\final"
if (Test-Path $weightsPath) {
    Write-Host "Starting LLM server on port 8000..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\python.exe model_service/serve.py"
} else {
    Write-Host "WARNING: Model weights not found at $weightsPath" -ForegroundColor Red
    Write-Host "         Copy weights from Engineer Z and re-run" -ForegroundColor Red
}

# Step 3: Start API server
Write-Host "Starting API server on port 8080..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Available at:
# http://localhost:8080        Chat UI
# http://localhost:8080/docs   API Docs
# http://localhost:8080/api/v1/health