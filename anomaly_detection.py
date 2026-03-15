import numpy as np

def detect_anomalies(df):

    df['rolling_mean'] = df['sales'].rolling(7).mean()
    df['std'] = df['sales'].rolling(7).std()

    df['upper'] = df['rolling_mean'] + (2 * df['std'])
    df['lower'] = df['rolling_mean'] - (2 * df['std'])

    df['anomaly'] = np.where(
        (df['sales'] > df['upper']) |
        (df['sales'] < df['lower']),
        1, 0
    )

    return df