# Start FastAPI server on port 8000
Write-Host "Starting Telco Churn FastAPI service on http://127.0.0.1:8000..." -ForegroundColor Cyan
Write-Host "Interactive Swagger API documentation: http://127.0.0.1:8000/docs" -ForegroundColor Green
$env:PYTHONPATH = ".;src"
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
