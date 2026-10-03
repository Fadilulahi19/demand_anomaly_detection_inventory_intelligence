from sklearn.cluster import KMeans

def cluster_products(df):
    agg = df.groupby('product_id').agg({
        'sales': 'mean',
        'rolling_std_7': 'mean',
        'anomaly': lambda x: (x == -1).sum()
    }).reset_index()

    model = KMeans(n_clusters=3, random_state=42)
    agg['cluster'] = model.fit_predict(agg[['sales', 'rolling_std_7', 'anomaly']])

    return agg, model