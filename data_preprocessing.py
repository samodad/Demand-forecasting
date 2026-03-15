import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values('date')

    df.set_index('date', inplace=True)

    df = df.asfreq('D')

    df['sales'] = df['sales'].fillna(method='ffill')

    return df
