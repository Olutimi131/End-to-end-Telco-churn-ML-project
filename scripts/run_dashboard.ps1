# Start Streamlit dashboard on port 8501
Write-Host "Starting Telco Churn Streamlit Dashboard on http://localhost:8501..." -ForegroundColor Cyan
$env:PYTHONPATH = ".;src"
python -m streamlit run dashboard/app.py
