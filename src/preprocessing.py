import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_timeseries(df):
    df = df.sort_values(['product_id', 'date'])
    return df

def fill_missing_dates(df):
    df = df.set_index('date')
    df = df.groupby('product_id').apply(
        lambda x: x.asfreq('D').fillna(method='ffill')
    )
    df = df.reset_index()
    return df