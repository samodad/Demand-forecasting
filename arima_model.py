from statsmodels.tsa.arima.model import ARIMA
import pandas as pd


def train_arima(series):

    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()

    return model_fit


def forecast_arima(model, steps=30):

    forecast = model.forecast(steps=steps)

    forecast_df = pd.DataFrame({
        "forecast": forecast
    })

    return forecast_df