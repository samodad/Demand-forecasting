import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_series(df):

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['sales'])

    plt.title("Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales")

    plt.show()


def rolling_statistics(df):

    df['rolling_mean'] = df['sales'].rolling(7).mean()
    df['rolling_std'] = df['sales'].rolling(7).std()

    return df