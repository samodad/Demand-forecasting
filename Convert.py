import pandas as pd

# load files
sales = pd.read_csv("C:\\Samoda\\KDU\\3rd Year\\Project\\Datasets\\sales_train_validation.csv")
calendar = pd.read_csv("C:\\Samoda\\KDU\\3rd Year\\Project\\Datasets\\calendar.csv")

# select one product
series = sales.iloc[0,6:]

series = series.reset_index()
series.columns = ["d","sales"]

# merge with calendar
df = pd.merge(series, calendar, on="d")

# keep only date and sales
df = df[["date","sales"]]

df.to_csv("Demand forecasting/sales.csv", index=False)

print("sales.csv created successfully")

