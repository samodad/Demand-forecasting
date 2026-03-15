from data_preprocessing import load_data
from prophet_model import prophet_forecast

df = load_data("C:\\Samoda\\KDU\\3rd Year\\Project\\Demand forecasting\\sales.csv")

forecast = prophet_forecast(df)

print(forecast.head())

from data_preprocessing import load_data
from prophet_model import prophet_forecast
import matplotlib.pyplot as plt

df = load_data("C:\\Samoda\\KDU\\3rd Year\\Project\\Demand forecasting\\sales.csv")

forecast = prophet_forecast(df)

print(forecast.head())

plt.figure(figsize=(12,6))
plt.plot(forecast['ds'], forecast['yhat'])
plt.title("Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.show()