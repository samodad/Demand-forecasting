from data_preprocessing import load_data
from eda import plot_time_series, rolling_statistics

df = load_data("C:\\Samoda\\KDU\\3rd Year\\Project\\Demand forecasting\\sales.csv")

plot_time_series(df)

df = rolling_statistics(df)

print(df.head())