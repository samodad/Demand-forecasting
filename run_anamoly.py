from data_preprocessing import load_data
from anomaly_detection import detect_anomalies

df = load_data("C:\\Samoda\\KDU\\3rd Year\\Project\\Demand forecasting\\sales.csv")

df = detect_anomalies(df)

print(df[df['anomaly'] == 1])

import matplotlib.pyplot as plt
from data_preprocessing import load_data
from anomaly_detection import detect_anomalies

df = load_data("C:\\Samoda\\KDU\\3rd Year\\Project\\Demand forecasting\\sales.csv")

df = detect_anomalies(df)

anomalies = df[df['anomaly'] == 1]

plt.figure(figsize=(12,6))

plt.plot(df.index, df['sales'], label="Sales")
plt.scatter(anomalies.index, anomalies['sales'], color="red", label="Anomalies")

plt.title("Sales Anomaly Detection")
plt.legend()

plt.show()