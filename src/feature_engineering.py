def create_features(df):
    df['rolling_mean_7'] = df.groupby('product_id')['sales'].transform(lambda x: x.rolling(7).mean())
    df['rolling_std_7'] = df.groupby('product_id')['sales'].transform(lambda x: x.rolling(7).std())

    df['z_score'] = (df['sales'] - df['rolling_mean_7']) / (df['rolling_std_7'] + 1)

    df['lag_1'] = df.groupby('product_id')['sales'].shift(1)
    df['lag_7'] = df.groupby('product_id')['sales'].shift(7)

    return df.dropna()