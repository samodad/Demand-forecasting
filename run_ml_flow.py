import mlflow
from sklearn.metrics import mean_squared_error
import numpy as np
from data_preprocessing import load_data
from arima_model import train_arima

df = load_data("C:\\Samoda\\KDU\\3rd Year\\Project\\Demand forecasting\\sales.csv")

mlflow.start_run()

model = train_arima(df['sales'])

predictions = model.predict()

rmse = np.sqrt(mean_squared_error(df['sales'], predictions))

mlflow.log_param("model", "ARIMA")
mlflow.log_metric("rmse", rmse)

mlflow.end_run()

print("RMSE:", rmse)