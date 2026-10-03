import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import pandas as pd

# This gets the absolute path of the directory where app.py sits
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root, then down into data/raw/sales.csv
file_path = os.path.join(BASE_DIR, "..", "data", "raw", "sales.csv")


# Load data
df = pd.read_csv(file_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['sales'] = df['Quantity'] * df['UnitPrice']


# Sidebar
st.set_page_config(
    page_title="Inventory Intelligence",
    layout="wide"
)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview", 
    "Demand Analysis", 
    "Segmentation", 
    "Simulator"
])

# -------------------------
# PAGE 1: OVERVIEW
# -------------------------
if page == "Overview":
    st.title("📦 Inventory Intelligence Dashboard")

    total_products = df['StockCode'].nunique()
    total_records = len(df)

    st.metric("Total Products", total_products)
    st.metric("Total Records", total_records)

# -------------------------
# PAGE 2: DEMAND ANALYSIS
# -------------------------
elif page == "Demand Analysis":
    st.title("📈 Demand & Anomalies")

    StockCode = st.selectbox("Select Product", df['StockCode'].unique())

    product = df[df['StockCode'] == StockCode]

    fig = px.line(product, x='InvoiceDate', y='sales', title="Demand Over Time")

    st.plotly_chart(fig)

# -------------------------
# PAGE 3: SEGMENTATION
# -------------------------
elif page == "Segmentation":
    st.title("🧩 Product Segments")

    st.write("Segments based on demand behavior:")

    st.markdown("""
    - 🟢 Stable → predictable demand  
    - 🟡 Seasonal → periodic spikes  
    - 🔴 High-risk → frequent anomalies  
    """)

# -------------------------
# PAGE 4: SIMULATOR
# -------------------------
elif page == "Simulator":
    st.title("⚡ Inventory Risk Simulator")

    demand = st.slider("Average Demand", 0, 500, 100)
    volatility = st.slider("Demand Volatility", 0, 200, 50)
    safety_stock = st.slider("Safety Stock", 0, 500, 100)

    risk = (demand + volatility - safety_stock) / (demand + 1)

    st.subheader(f"Stockout Risk Score: {round(risk, 2)}")

    if risk > 0.7:
        st.error("High Risk → Increase safety stock")
    elif risk > 0.4:
        st.warning("Moderate Risk → Monitor closely")
    else:
        st.success("Low Risk → Efficient inventory")