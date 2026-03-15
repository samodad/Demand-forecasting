import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error

st.title("Demand Forecasting Analytics Dashboard")

# Load dataset
df = pd.read_csv("C:\\Samoda\\KDU\\3rd Year\\Project\\Demand forecasting\\sales.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")

st.subheader("Dataset Preview")
st.write(df.head())

# =========================
# Sales Trend
# =========================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['sales'],
        mode='lines',
        name='Actual Sales'
    )
)

fig.update_layout(title="Sales Trend")

st.plotly_chart(fig)

# =========================
# Forecast Settings
# =========================

st.subheader("Forecast Settings")

steps = st.slider("Forecast Days", 7, 60, 30)

ts = df.set_index("date")
ts = ts.asfreq("D")

# =========================
# ARIMA MODEL
# =========================

arima_model = ARIMA(ts['sales'], order=(1,1,1))
arima_fit = arima_model.fit()

arima_forecast_result = arima_fit.get_forecast(steps=steps)
arima_forecast = arima_forecast_result.predicted_mean
arima_ci = arima_forecast_result.conf_int()

# =========================
# PROPHET MODEL
# =========================

prophet_df = df.rename(columns={"date":"ds","sales":"y"})

prophet_model = Prophet()
prophet_model.fit(prophet_df)

future = prophet_model.make_future_dataframe(periods=steps)

prophet_forecast = prophet_model.predict(future)

# =========================
# MODEL COMPARISON (RMSE)
# =========================

arima_pred = arima_fit.predict()
prophet_pred = prophet_forecast['yhat'][:len(df)]

arima_rmse = np.sqrt(mean_squared_error(df['sales'], arima_pred))
prophet_rmse = np.sqrt(mean_squared_error(df['sales'], prophet_pred))

st.subheader("Model Comparison")

comparison = pd.DataFrame({
    "Model": ["ARIMA","Prophet"],
    "RMSE":[arima_rmse,prophet_rmse]
})

st.write(comparison)

# Automatic Best Model
best_model = "ARIMA" if arima_rmse < prophet_rmse else "Prophet"

st.success(f"Best Model Automatically Selected: {best_model}")

# =========================
# ANOMALY DETECTION
# =========================

rolling_mean = df['sales'].rolling(7).mean()
rolling_std = df['sales'].rolling(7).std()

upper = rolling_mean + (2*rolling_std)
lower = rolling_mean - (2*rolling_std)

df['anomaly'] = np.where(
    (df['sales'] > upper) |
    (df['sales'] < lower),
    1,0
)

anomalies = df[df['anomaly']==1]

# =========================
# ANOMALY VISUALIZATION
# =========================

st.subheader("Anomaly Detection")

fig_anomaly = go.Figure()

fig_anomaly.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['sales'],
        mode='lines',
        name='Sales'
    )
)

fig_anomaly.add_trace(
    go.Scatter(
        x=anomalies['date'],
        y=anomalies['sales'],
        mode='markers',
        marker=dict(color='red',size=8),
        name='Anomalies'
    )
)

st.plotly_chart(fig_anomaly)

# =========================
# FORECAST VISUALIZATION
# =========================

st.subheader("Forecast Visualization")

fig_forecast = go.Figure()

fig_forecast.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['sales'],
        mode='lines',
        name='Actual'
    )
)

# ARIMA forecast
arima_dates = pd.date_range(start=df['date'].max(), periods=steps+1)[1:]

fig_forecast.add_trace(
    go.Scatter(
        x=arima_dates,
        y=arima_forecast,
        mode='lines',
        name='ARIMA Forecast'
    )
)

# Prophet forecast
fig_forecast.add_trace(
    go.Scatter(
        x=prophet_forecast['ds'],
        y=prophet_forecast['yhat'],
        mode='lines',
        name='Prophet Forecast'
    )
)

st.plotly_chart(fig_forecast)

# =========================
# CONFIDENCE INTERVAL (ARIMA)
# =========================

st.subheader("Forecast Confidence Interval")

fig_ci = go.Figure()

fig_ci.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['sales'],
        mode='lines',
        name='Actual'
    )
)

fig_ci.add_trace(
    go.Scatter(
        x=arima_dates,
        y=arima_forecast,
        mode='lines',
        name='Forecast'
    )
)

fig_ci.add_trace(
    go.Scatter(
        x=arima_dates,
        y=arima_ci.iloc[:,0],
        line=dict(width=0),
        showlegend=False
    )
)

fig_ci.add_trace(
    go.Scatter(
        x=arima_dates,
        y=arima_ci.iloc[:,1],
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(width=0),
        name='Confidence Interval'
    )
)

st.plotly_chart(fig_ci)

st.success("Dashboard Loaded Successfully")