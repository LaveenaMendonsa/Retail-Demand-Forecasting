\# Retail Demand Forecasting \& Analytics Platform



\## Project Overview



This project is an end-to-end Retail Demand Forecasting and Analytics Platform built using Python and Machine Learning.



The goal is to analyze historical retail sales data, identify sales trends, and predict product demand across different stores and product families.



The project combines data preprocessing, exploratory data analysis, machine learning, experiment tracking, interactive visualization, and Docker containerization.



\## Technologies Used



\- Python

\- Pandas and NumPy

\- Matplotlib and Seaborn

\- Scikit-learn

\- Streamlit

\- Plotly

\- MLflow

\- Docker

\- Jupyter Notebook



\## Dataset



The project uses the Store Sales - Time Series Forecasting dataset from Kaggle.



The dataset contains historical sales information, store details, product families, promotions, holidays, oil prices, and transactions.



\## Project Workflow



1\. Data collection and understanding

2\. Data cleaning and preprocessing

3\. Exploratory Data Analysis (EDA)

4\. Feature engineering

5\. Machine Learning model development

6\. Model evaluation and comparison

7\. Forecast generation

8\. MLflow experiment tracking

9\. Interactive Streamlit dashboard

10\. Docker containerization



\## Machine Learning Models



Two regression models were developed and evaluated:



\- Random Forest Regressor

\- HistGradientBoosting Regressor



\### Model Evaluation



| Model | MAE | RMSE | R² |

|---|---:|---:|---:|

| Random Forest | 109.28 | 374.78 | 0.9099 |

| HistGradientBoosting | 134.14 | 333.93 | 0.9284 |



The Random Forest model achieved a lower MAE, while HistGradientBoosting achieved a lower RMSE and higher R² on the validation data.



\## Streamlit Dashboard



The interactive dashboard includes:



\- Business overview and key performance indicators

\- Sales analysis

\- Product family analysis

\- Store performance analysis

\- Machine Learning model comparison

\- Demand forecasting with downloadable predictions



\## MLflow Experiment Tracking



MLflow is used to track model parameters, evaluation metrics, and model artifacts.



\## Docker Deployment



The Streamlit application is containerized using Docker.



\### Build the Docker Image



```bash

docker build -t retail-demand-forecasting .

```



\### Run the Docker Container



```bash

docker run -d --name retail-demand-app -p 8501:8501 retail-demand-forecasting

```



\### Open the Application



Visit:



http://localhost:8501



\## Project Structure



```text

Retail\_Demand\_Forecasting/

│

├── data/

├── notebooks/

├── models/

├── app/

│   └── app.py

├── mlflow/

│   └── tracking/

├── docker/

├── Dockerfile

├── requirements.txt

├── .dockerignore

├── mlflow\_tracking.py

└── README.md

```



\## Key Outcomes



\- Analyzed approximately 3 million historical sales records.

\- Developed and compared two regression models.

\- Generated predictions for the supplied test dataset.

\- Built an interactive Streamlit analytics dashboard.

\- Implemented MLflow experiment tracking.

\- Containerized the application using Docker.



\## Future Improvements



\- Cloud deployment using AWS.

\- Automated model retraining.

\- Improved time-series forecasting.

\- Continuous integration and deployment.



\## Author



\*\*Laveena Mendonsa\*\*



BCA Graduate | Data Science



GitHub: https://github.com/LaveenaMendonsa



