from sklearn.ensemble import IsolationForest

def detect_anomalies(df, feature_cols):
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(df[feature_cols])
    return df, model