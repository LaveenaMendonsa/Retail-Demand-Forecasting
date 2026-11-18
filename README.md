# Retail Demand Forecasting & Analytics Platform

An end-to-end Machine Learning project that analyzes retail sales data and predicts product demand across multiple stores. Built using Python, Pandas, Scikit-learn, Streamlit, MLflow, and Docker.

## Project Overview

This project uses the Store Sales – Time Series Forecasting dataset to understand retail sales patterns and predict future product demand.

**Objectives:**
- Analyze historical retail sales trends.
- Identify top-performing stores and product families.
- Understand the impact of promotions, holidays, and other factors.
- Build and compare Machine Learning regression models.
- Present insights and predictions through an interactive dashboard.
- Track experiments using MLflow and containerize the application using Docker.

## Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn
- **Models:** Random Forest Regressor, HistGradientBoosting Regressor
- **Dashboard:** Streamlit
- **Experiment Tracking:** MLflow
- **Containerization:** Docker
- **Development:** Jupyter Notebook, Git, GitHub

## Dataset

- Training records: 3,000,888
- Test records: 28,512
- Target variable: `sales`
- Main features: Date, Store Number, Product Family, Promotions, and additional store, oil, and holiday information.

## Machine Learning Models

Two regression models were trained and evaluated using a time-based validation approach.

| Metric | Random Forest | HistGradientBoosting |
|---|---:|---:|
| MAE | 109.28 | 134.14 |
| RMSE | 374.78 | 333.93 |
| R² Score | 0.9099 | 0.9284 |

Random Forest achieved lower MAE, while HistGradientBoosting achieved lower RMSE and higher R². Model selection depends on the evaluation metric and business objective.

## Streamlit Dashboard

The interactive dashboard includes:

- Overview of retail sales
- Sales trend analysis
- Product family analysis
- Store performance analysis
- Model comparison
- Demand forecasting results

## MLflow Experiment Tracking

MLflow is used to track model parameters, evaluation metrics, and model artifacts for experiment comparison.

## Docker Deployment

The application is containerized using Docker to support consistent execution across environments.

To run locally:

```bash
docker build -t retail-demand-forecasting .
docker run -p 8501:8501 retail-demand-forecasting
```

Open `http://localhost:8501` in your browser.

## Project Structure

```text
Retail_Demand_Forecasting/
│
├── app/
│   └── app.py
├── data/
├── models/
├── notebooks/
│   └── 01_Data_Understanding.ipynb
├── mlflow/
├── docker/
├── Dockerfile
├── requirements.txt
├── mlflow_tracking.py
├── README.md
└── .gitignore
```

## Key Learning Outcomes

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Time-based model validation
- Regression model training and evaluation
- Experiment tracking with MLflow
- Interactive dashboard development
- Docker-based application deployment

## Author

**Laveena Mendonsa**

BCA Graduate | Data Science & Machine Learning

GitHub: https://github.com/LaveenaMendonsa