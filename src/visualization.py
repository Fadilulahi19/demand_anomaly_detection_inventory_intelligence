import matplotlib.pyplot as plt

def plot_anomalies(df, product_id):
    product = df[df['product_id'] == product_id]

    plt.figure()
    plt.plot(product['date'], product['sales'], label='Sales')

    anomalies = product[product['anomaly'] == -1]
    plt.scatter(anomalies['date'], anomalies['sales'], label='Anomaly')

    plt.legend()
    plt.title(f"Product {product_id} Demand with Anomalies")
    plt.show()