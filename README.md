# Demand Anomaly Detection & Inventory Intelligence

## 🚀 Overview
This project detects unusual demand patterns in retail sales and segments products based on anomaly behavior to improve inventory decisions.

## 🎯 Business Problem
Traditional forecasting fails to detect sudden demand spikes or drops, leading to:
- Stockouts
- Overstock
- Revenue loss

## 🧠 Solution
We built an anomaly detection system using:
- Time-series feature engineering
- Isolation Forest for anomaly detection
- Clustering for product segmentation

## ⚙️ Techniques Used
- Rolling statistics (mean, std)
- Z-score anomaly detection
- Isolation Forest
- K-Means clustering

## 🔍 Key Insights
- Identified high-volatility products requiring safety stock
- Detected promo-driven anomalies
- Segmented products into stable vs risky groups

## 📈 Business Impact
- Reduced stockout risk
- Improved inventory planning
- Better demand visibility

## 🛠️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook

## 🌐 Streamlit Dashboard

Run locally:

```bash
streamlit run app/app.py