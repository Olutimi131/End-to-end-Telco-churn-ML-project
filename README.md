## Telco Customer Churn Prediction & Deployment Pipeline
An end-to-end Machine Learning and CI/CD project that predicts customer churn using modern ML techniques, served via a high-performance FastAPI REST interface, containerized with Docker, and automatically tested and deployed through GitHub Actions.
## Key Features
-End-to-End ML Workflow: Automated data preprocessing, feature engineering, and model inference pipelines.

-REST API Interface: Clean, production-ready FastAPI web service with interactive Swagger documentation.

-Containerized Infrastructure: Production-grade Dockerfile using lightweight Python bases for reliable deployment across any cloud or local environment.

-Automated CI/CD Pipeline: Fully automated testing (pytest) and multi-stage container assembly with automated deployments to Docker Hub on push to main.

-Automated Code Quality: Unit tests with coverage reporting integrated into the GitHub Actions workflow.

## Tech Stack & Tooling
Language: Python 3.12
Framework: FastAPI, Uvicorn
ML Libraries: Scikit-Learn, Pandas, NumPy, LightGBM / XGBoost
Testing: Pytest, Pytest-Cov
Containerization: Docker
CI/CD Automation: GitHub Actions, Docker Hub Registry

## 📊 Model Evaluation & Benchmarks

Models were evaluated using 5-fold Stratified Cross-Validation on 5,634 training records and tested on 1,409 held-out test records:

| Model Candidate | Cross-Validation ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.8476** | **0.8447** | **0.6589** | **0.6290** | 🏆 **Champion** |
| Random Forest | 0.8466 | 0.8421 | 0.6512 | 0.6184 | Candidate |
| Logistic Regression | 0.8453 | 0.8410 | 0.6480 | 0.6215 | Baseline |


## 🐳 Docker & Docker Compose

Deploy the API and Streamlit dashboard simultaneously in isolated containers:

```bash
# Build and run both services
docker compose up --build

# Run in background (detached mode)
docker compose up -d
```

- API container runs on port `8000`
- Streamlit dashboard container runs on port `8501`


---

## 📄 License

This project is licensed under the Apache 2.0 License.
