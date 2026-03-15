from data_preprocessing import load_data
from arima_model import train_arima, forecast_arima

df = load_data("C:\\Samoda\\KDU\\3rd Year\\Project\\Demand forecasting\\sales.csv")

model = train_arima(df['sales'])

forecast = forecast_arima(model, 30)

print(forecast)