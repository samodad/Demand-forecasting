# 📈 Demand Forecasting Using Time Series Models

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

This project focuses on forecasting product demand using historical sales data through **time series analysis**. The system explores trends, identifies anomalies, and predicts future demand using statistical forecasting models.

The project also includes an **interactive Streamlit dashboard** that allows users to visualize historical sales data and forecast future demand.

The forecasting models implemented include:

* **ARIMA (Autoregressive Integrated Moving Average)**
* **SARIMA (Seasonal ARIMA)**
* **Prophet (Meta's Forecasting Model)**

These models help businesses improve **inventory planning, demand management, and decision-making**.

# 🎯 Problem Statement

Businesses rely on demand forecasting to maintain optimal inventory levels. However, sales data often contains:

* Trends
* Seasonal patterns
* Sudden anomalies
* Non-stationary behavior

These challenges make forecasting complex. This project applies **time series modeling techniques** to produce reliable demand predictions.

# 🎯 Objectives

* Analyze historical sales trends
* Perform **time series stationarity analysis**
* Detect anomalies in sales data
* Forecast future demand using **ARIMA, SARIMA, and Prophet**
* Track experiments using **MLflow**
* Develop an **interactive Streamlit dashboard**


# 📊 Dataset

Example dataset structure:

| Date       | Sales |
| ---------- | ----- |
| 2022-01-01 | 150   |
| 2022-01-02 | 180   |
| 2022-01-03 | 200   |

Dataset location:

Demand forecasting/sales.csv

# 🧠 Project Architecture

Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Stationarity Test (ADF)
   │
   ▼
ACF / PACF Analysis
   │
   ▼
ARIMA / SARIMA Models
   │
   ▼
Prophet Forecasting
   │
   ▼
Anomaly Detection
   │
   ▼
MLflow Experiment Tracking
   │
   ▼
Streamlit Interactive Dashboard(📊 Sales Trend

Actual historical demand visualization

🤖 Model Comparison

ARIMA vs Prophet

🏆 Automatic Best Model Selection

Based on RMSE

🔮 Forecast Comparison

Future demand predictions

🚨 Anomaly Detection

Detect unusual spikes or drops

📋 Forecast Tables

Model prediction outputs)


# 📂 Project Structure

```
demand-forecasting-project
│
──
└── sales.csv
│
├── notebooks
│   └── exploration.ipynb
│
├── src
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── arima_model.py
│   ├── prophet_model.py
│   ├── anomaly_detection.py
│   ├── run_arima.py
│   ├── run_prophet.py
│   └── run_anomaly.py
│
├── dashboard
│   └── app.py
│
├── models
│   
│
├── outputs
│   ├
│
├── requirements.txt
│
└── README.md



# ⚙️ Tools & Technologies

### Programming

* Python

### Data Analysis

* pandas
* numpy

### Visualization

* matplotlib
* seaborn
* plotly

### Forecasting

* statsmodels
* prophet

### Experiment Tracking

* MLflow

### Dashboard

* Streamlit

### Development

* VS Code
* Python Virtual Environment

---

# 🔬 Methodology

## 1️⃣ Data Preprocessing

* Convert date column to datetime
* Set date as index
* Ensure daily frequency
* Handle missing values

## 2️⃣ Exploratory Data Analysis

* Sales trend visualization
* Rolling mean analysis
* Rolling standard deviation

## 3️⃣ Stationarity Testing

Applied **Augmented Dickey-Fuller (ADF) test** to determine whether the time series is stationary.

## 4️⃣ Time Series Modeling

### ARIMA

Used for modeling non-seasonal time series patterns.

### SARIMA

Captures seasonal components in time series data.

### Prophet

Automatically models **trend and seasonality**.

## 5️⃣ Anomaly Detection

Anomalies were detected using rolling mean and standard deviation thresholds.

## 6️⃣ Experiment Tracking

MLflow was used to log:

* model parameters
* evaluation metrics
* experiment runs

## 7️⃣ Interactive Dashboard

Streamlit dashboard enables users to:

* View historical sales trends
* Forecast future demand
* Adjust forecast window interactively


# 📈 Example Forecast Visualization

Features included in the dashboard:

✔ Sales trend visualization
✔ Forecast demand prediction
✔ Interactive charts
✔ Adjustable forecast horizon


# 🚀 Running the Project

## 1️⃣ Create Virtual Environment

python -m venv .venv

Activate:

.\.venv\Scripts\Activate.ps1



## 2️⃣ Install Dependencies


pip install -r requirements.txt


## 3️⃣ Run Forecast Models
cd src
python run_arima.py

Run Prophet model:

python run_prophet.py


Run anomaly detection:
python run_anomaly.py

## 4️⃣ Run MLflow


mlflow ui


Open:

Local URL: http://localhost:8501
Network URL: http://192.168.1.29:8501


## 5️⃣ Launch Dashboard


cd dashboard
streamlit run app.py


Open:

http://localhost:8501




# 📊 Project Outputs


outputs/
│
├── forecast_results.csv
└── plots/


models/
└── saved_model.pkl

# 📚 Key Results

* Identified sales trends and seasonal patterns
* Detected anomalies in the dataset
* Generated demand forecasts using ARIMA and Prophet
* Developed an interactive forecasting dashboard


# 🎓 Learning Outcomes

Through this project, I gained experience in:

* Time series forecasting techniques
* ARIMA and SARIMA modeling
* Statistical stationarity testing
* Experiment tracking with MLflow
* Building interactive data dashboards
* Structuring real-world data science projects

# 🔮 Future Improvements

* Implement **Auto-ARIMA model tuning**
* Deploy model using **FastAPI**
* Add **real-time data streaming**
* Improve anomaly detection using machine learning

