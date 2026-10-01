from setuptools import setup, find_packages

setup(
    name="telco_churn",
    version="1.0.0",
    author="ML Engineering Team",
    description="End-to-End Telco Customer Churn Prediction and Retention Optimization Platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.11.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "pydantic>=2.6.0",
        "streamlit>=1.32.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "httpx>=0.27.0",
        ]
    },
)
