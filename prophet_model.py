from prophet import Prophet
import pandas as pd

def prophet_forecast(df):

    prophet_df = df.reset_index()
    prophet_df.columns = ["ds", "y"]

    model = Prophet()

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=30)

    forecast = model.predict(future)

    return forecast