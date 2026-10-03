"""
Demand Anomaly Detection Dashboard
====================================
streamlit run app/anomaly_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import http.client
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings, io, urllib.request

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Demand Anomaly Detection", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');
html,body,[class*="css"] { font-family:'Space Grotesk',sans-serif; }
h1,h2,h3 { font-family:'Space Mono',monospace; }
.alert-card { border-radius:8px; padding:14px 18px; margin-bottom:10px;
              border-left:4px solid #ef4444; background:#fff5f5; }
.kpi { background:#f8fafc; border:1px solid #e2e8f0;
       border-radius:8px; padding:14px 18px; }
.kpi-val { font-family:'Space Mono',monospace; font-size:1.8rem;
           font-weight:700; line-height:1; }
.kpi-lbl { font-size:0.7rem; color:#94a3b8; text-transform:uppercase;
           letter-spacing:1px; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

CLUSTER_NAMES = {0:"Stable",1:"Seasonal",2:"Volatile",3:"Erratic"}
COLORS        = ["#10b981","#3b82f6","#ef4444","#f59e0b"]
RANDOM_STATE  = 42

@st.cache_data(show_spinner=False)
def load_and_build():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    try:
        # Attempt normal download
        raw = urllib.request.urlopen(url).read()
    except http.client.IncompleteRead as e:
        raw = e.partial
    except Exception as e:
        raw = None
    
    df  = pd.read_excel(io.BytesIO(raw))

    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"]>0) & (df["UnitPrice"]>0)]
    df = df[df["StockCode"].astype(str).str.len()>=5]
    df = df[~df["StockCode"].isin(["POST","D","M","BANK CHARGES","PADS","DOT"])]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Week"]        = df["InvoiceDate"].dt.to_period("W").dt.start_time

    weekly = (df.groupby(["StockCode","Week"])["Quantity"]
                .sum().reset_index().rename(columns={"Quantity":"units"}))

    wc = weekly.groupby("StockCode")["Week"].count()
    valid = wc[wc>=20].index
    weekly = weekly[weekly["StockCode"].isin(valid)].copy()

    all_weeks = pd.date_range(weekly["Week"].min(), weekly["Week"].max(), freq="W-MON")
    products  = weekly["StockCode"].unique()
    full_idx  = pd.MultiIndex.from_product([products, all_weeks], names=["StockCode","Week"])
    weekly    = weekly.set_index(["StockCode","Week"]).reindex(full_idx, fill_value=0).reset_index()

    weekly = weekly.sort_values(["StockCode","Week"])

    def add_features(g):
        s = g["units"]
        g["roll_mean_4"] = s.rolling(4, min_periods=2).mean()
        g["roll_std_4"]  = s.rolling(4, min_periods=2).std().fillna(0)
        g["z_score"]     = (s - g["roll_mean_4"]) / (g["roll_std_4"]+1e-9)
        g["lag_1"]       = s.shift(1)
        g["pct_change"]  = s.pct_change().replace([np.inf,-np.inf],0).fillna(0)
        pm = s[s>0].mean() if (s>0).any() else 1
        g["demand_vs_mean"] = s/(pm+1e-9)
        return g

    weekly = weekly.groupby("StockCode", group_keys=False).apply(add_features)
    weekly = weekly.dropna(subset=["roll_mean_4","lag_1"])

    FEAT = ["units","z_score","pct_change","demand_vs_mean","lag_1"]
    X    = weekly[FEAT].fillna(0)
    sc   = StandardScaler(); X_sc = sc.fit_transform(X)

    iforest = IsolationForest(n_estimators=150, contamination=0.05,
                              random_state=RANDOM_STATE, n_jobs=-1)
    weekly["anomaly_iforest"] = (iforest.fit_predict(X_sc)==-1).astype(int)
    weekly["anomaly_score"]   = -iforest.score_samples(X_sc)
    weekly["anomaly_zscore"]  = (np.abs(weekly["z_score"])>2.5).astype(int)
    weekly["anomaly"]         = ((weekly["anomaly_iforest"]+weekly["anomaly_zscore"])>=2).astype(int)

    # Product-level features for clustering
    pf = weekly.groupby("StockCode").agg(
        avg_demand=("units","mean"),
        demand_cv =("units",lambda x: x.std()/(x.mean()+1e-9)),
        anomaly_rate=("anomaly","mean"),
        avg_score =("anomaly_score","mean"),
        max_z     =("z_score",lambda x: x.abs().max()),
        zero_weeks=("units",lambda x: (x==0).sum()),
        peak_to_mean=("units",lambda x: x.max()/(x.mean()+1e-9)),
    ).reset_index()

    CF  = ["avg_demand","demand_cv","anomaly_rate","avg_score","max_z","zero_weeks","peak_to_mean"]
    Xcl = pf[CF].copy()
    for c in CF: Xcl[c] = Xcl[c].clip(upper=Xcl[c].quantile(0.99))
    scl = StandardScaler(); Xcl_sc = scl.fit_transform(Xcl)
    km  = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    pf["cluster"] = km.fit_predict(Xcl_sc)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xpca = pca.fit_transform(Xcl_sc)

    # Merge cluster back
    weekly = weekly.merge(pf[["StockCode","cluster"]], on="StockCode", how="left")

    price_map = (df.groupby("StockCode")["UnitPrice"].median().to_dict())
    pf["unit_price"] = pf["StockCode"].map(price_map).fillna(5.0)
    weekly["unit_price"] = weekly["StockCode"].map(price_map).fillna(5.0)

    desc_map  = df.groupby("StockCode")["Description"].first().to_dict()
    pf["description"] = pf["StockCode"].map(desc_map).fillna("")
    weekly["description"] = weekly["StockCode"].map(desc_map).fillna("")

    return weekly, pf, Xcl_sc, Xpca, km, pca, scl

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Controls")
    st.markdown("---")
    z_thresh    = st.slider("Z-Score threshold", 1.5, 4.0, 2.5, 0.1)
    contamination = st.slider("Expected anomaly rate (%)", 1, 15, 5, 1) / 100
    st.markdown("---")
    show_clusters = st.multiselect("Filter Clusters", list(range(4)), default=list(range(4)),
                                   format_func=lambda x: f"C{x} — {CLUSTER_NAMES[x]}")

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
with st.spinner("⚙️ Building demand time series & running anomaly detection…"):
    weekly, pf, Xcl_sc, Xpca, km, pca, scl = load_and_build()

weekly_f = weekly[weekly["cluster"].isin(show_clusters)]
pf_f     = pf[pf["cluster"].isin(show_clusters)]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 📡 Demand Anomaly Detection")
st.markdown("**Supply Chain Intelligence · UCI Online Retail · Isolation Forest + Z-Score + Product Clustering**")
st.markdown("---")

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
total_anom   = weekly_f["anomaly"].sum()
anom_rate    = weekly_f["anomaly"].mean()*100
n_products   = pf_f.shape[0]
high_risk    = (pf_f["anomaly_rate"] > 0.1).sum()
weeks_span   = weekly_f["Week"].nunique()

for col,(val,lbl,color) in zip(st.columns(5),[
    (f"{n_products:,}","Products Monitored","#1e293b"),
    (f"{weeks_span}","Weeks Analysed","#1e293b"),
    (f"{total_anom:,}","Anomaly Events","#ef4444"),
    (f"{anom_rate:.1f}%","Anomaly Rate","#ef4444"),
    (f"{high_risk}","High-Risk SKUs","#f59e0b"),
]):
    with col:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-val" style="color:{color}">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
st.markdown("")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📈 Anomaly Timeline","🗺️ Cluster Map","🔍 Product Inspector","📊 Segment Profiles","💼 Strategy"
])

# ── TAB 1: TIMELINE ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Portfolio-Level Anomaly Timeline")
    by_week = weekly_f.groupby("Week")["anomaly"].sum().reset_index()
    fig,ax = plt.subplots(figsize=(13,4))
    ax.fill_between(by_week["Week"], by_week["anomaly"], alpha=0.25, color="#ef4444")
    ax.plot(by_week["Week"], by_week["anomaly"], color="#ef4444", linewidth=1.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.set_ylabel("Anomalous SKUs"); ax.set_xlabel("")
    ax.set_title("Anomaly Events Across Portfolio by Week", fontsize=13, fontweight="bold")
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("### Top Anomaly Products This Week")
    latest = weekly_f[weekly_f["Week"]==weekly_f["Week"].max()]
    top    = latest[latest["anomaly"]==1].nlargest(10,"anomaly_score")[
        ["StockCode","description","units","roll_mean_4","z_score","anomaly_score","cluster"]]
    top.columns = ["SKU","Description","Units","4-wk Avg","Z-Score","Anomaly Score","Cluster"]
    st.dataframe(top.style.background_gradient(cmap="Reds",subset=["Anomaly Score","Z-Score"]),
                 use_container_width=True)

# ── TAB 2: CLUSTER MAP ────────────────────────────────────────────────────────
with tab2:
    col_l,col_r = st.columns([2,1])
    with col_l:
        st.markdown("### Product Behaviour Clusters (PCA)")
        fig,ax = plt.subplots(figsize=(9,6))
        for i in range(4):
            m = pf["cluster"]==i
            ax.scatter(Xpca[m,0], Xpca[m,1], c=COLORS[i], alpha=0.5, s=12,
                       label=CLUSTER_NAMES[i], edgecolors="none")
        cen = pca.transform(km.cluster_centers_)
        ax.scatter(cen[:,0],cen[:,1],c="black",marker="X",s=180,zorder=10)
        exp = pca.explained_variance_ratio_*100
        ax.set_xlabel(f"PC1 ({exp[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({exp[1]:.1f}%)")
        ax.set_title("Demand Behaviour Segments", fontweight="bold", fontsize=12)
        ax.legend(fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_r:
        st.markdown("### Cluster Split")
        counts = pf_f["cluster"].value_counts().sort_index()
        fig,ax = plt.subplots(figsize=(4.5,4))
        ax.pie(counts.values, labels=[CLUSTER_NAMES[i] for i in counts.index],
               autopct="%1.1f%%", colors=[COLORS[i] for i in counts.index],
               startangle=90, wedgeprops=dict(linewidth=2, edgecolor="white"))
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("### Avg Anomaly Rate by Cluster")
        ar = pf.groupby("cluster")["anomaly_rate"].mean()*100
        fig,ax = plt.subplots(figsize=(4.5,3.5))
        bars = ax.bar(range(4), ar.values, color=COLORS, width=0.5, edgecolor="white")
        for b,v in zip(bars,ar.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                    f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"C{i}" for i in range(4)])
        ax.set_ylabel("Anomaly Rate (%)"); ax.set_title("By Cluster", fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()

# ── TAB 3: PRODUCT INSPECTOR ──────────────────────────────────────────────────
with tab3:
    st.markdown("### Drill Down Into Any Product")
    cf1,cf2 = st.columns(2)
    with cf1:
        sel_cluster = st.selectbox("Filter by cluster", list(range(4)),
                                   format_func=lambda x: f"C{x} — {CLUSTER_NAMES[x]}")
    cluster_skus  = pf[pf["cluster"]==sel_cluster]["StockCode"].tolist()
    with cf2:
        sel_sku = st.selectbox("Select SKU", cluster_skus[:100])

    sub = weekly[weekly["StockCode"]==sel_sku].sort_values("Week")
    desc = pf[pf["StockCode"]==sel_sku]["description"].values[0] if len(pf[pf["StockCode"]==sel_sku])>0 else ""

    r1,r2,r3,r4 = st.columns(4)
    r1.metric("SKU", sel_sku)
    r2.metric("Total Anomalies", int(sub["anomaly"].sum()))
    r3.metric("Anomaly Rate", f"{sub['anomaly'].mean()*100:.1f}%")
    r4.metric("Max Z-Score", f"{sub['z_score'].abs().max():.2f}")
    st.caption(f"Description: {desc}")

    fig,axes = plt.subplots(2,1,figsize=(13,7),sharex=True)

    axes[0].plot(sub["Week"], sub["units"], color="steelblue", linewidth=1.5, label="Demand")
    axes[0].plot(sub["Week"], sub["roll_mean_4"], color="orange", linestyle="--",
                 linewidth=1.2, label="4-week avg")
    anom_pts = sub[sub["anomaly"]==1]
    axes[0].scatter(anom_pts["Week"], anom_pts["units"], color="red", s=60, zorder=5, label="Anomaly")
    axes[0].set_ylabel("Units"); axes[0].legend(fontsize=9)
    axes[0].set_title(f"SKU {sel_sku} — Demand & Anomalies", fontweight="bold")

    axes[1].bar(sub["Week"], sub["z_score"], color=sub["z_score"].apply(
        lambda x: "#ef4444" if abs(x)>2.5 else "#94a3b8"), width=5)
    axes[1].axhline(2.5, color="red", linestyle="--", alpha=0.6, label="+2.5σ")
    axes[1].axhline(-2.5, color="red", linestyle="--", alpha=0.6, label="-2.5σ")
    axes[1].set_ylabel("Z-Score"); axes[1].set_xlabel("Week"); axes[1].legend(fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=25)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ── TAB 4: PROFILES ───────────────────────────────────────────────────────────
with tab4:
    PCOLS = ["avg_demand","demand_cv","anomaly_rate","max_z","zero_weeks","peak_to_mean"]
    smry  = pf.groupby("cluster")[PCOLS].mean().round(3)
    smry["n_products"]  = pf.groupby("cluster").size()
    smry["pct_portfolio"] = (smry["n_products"]/len(pf)*100).round(1)

    col_a,col_b = st.columns(2)
    with col_a:
        st.markdown("### Feature Profile Heatmap")
        dh = smry[PCOLS].T
        nh = (dh-dh.min(axis=1).values[:,None])/(dh.max(axis=1)-dh.min(axis=1)+1e-9).values[:,None]
        fig,ax = plt.subplots(figsize=(8,5))
        sns.heatmap(nh,annot=dh.round(2),fmt="g",cmap="RdYlGn_r",ax=ax,
                    linewidths=0.6,linecolor="white",
                    xticklabels=[f"C{i}: {CLUSTER_NAMES[i]}" for i in range(4)],cbar=False)
        ax.set_title("Cluster Profiles",fontweight="bold",fontsize=12)
        plt.xticks(rotation=15)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        st.markdown("### Anomaly Score Distribution")
        fig,ax = plt.subplots(figsize=(8,5))
        for i in range(4):
            d = pf[pf["cluster"]==i]["anomaly_rate"]*100
            ax.hist(d, bins=20, alpha=0.6, color=COLORS[i], label=CLUSTER_NAMES[i], edgecolor="white")
        ax.set_xlabel("Anomaly Rate (%)"); ax.set_ylabel("Products")
        ax.legend(); ax.set_title("Anomaly Rate Distribution",fontweight="bold",fontsize=12)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.dataframe(smry.style.background_gradient(cmap="RdYlGn_r",subset=["anomaly_rate","max_z"]),
                 use_container_width=True)

# ── TAB 5: STRATEGY ───────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 💼 Inventory Strategy by Demand Behaviour Segment")
    strats = [
        {"c":"#10b981","e":"🟢","name":"Stable — Automatable",
         "p":"Low anomaly · Low CV · Consistent demand",
         "inv":"Automate EOQ replenishment. Minimal human oversight.",
         "det":"Z-score monitoring only. Alert at |z| > 3.",
         "kpi":"99%+ service level. Zero manual reorders."},
        {"c":"#3b82f6","e":"🔵","name":"Seasonal — Predictable",
         "p":"Recurring spikes · High peak-to-mean · Moderate anomaly rate",
         "inv":"Pre-build safety stock 2–3 weeks before known peaks.",
         "det":"Use Isolation Forest to separate true anomalies from seasonal peaks.",
         "kpi":"Reduce seasonal stockouts 40%. Pre-build cost <10% revenue."},
        {"c":"#ef4444","e":"🔴","name":"Volatile — High Risk",
         "p":"High anomaly rate · High CV · Large z-scores",
         "inv":"Increase safety stock 30–50%. Trigger emergency reorder on anomaly flag.",
         "det":"Ensemble detection required (Z + IF). Weekly human review.",
         "kpi":"Reduce stockout cost >25%. Bi-weekly review cadence."},
        {"c":"#f59e0b","e":"🟡","name":"Erratic — Unpredictable",
         "p":"High zero-weeks · No clear pattern · Intermittent demand",
         "inv":"Minimum viable stock. Make-to-order where feasible.",
         "det":"LOF best for sparse/intermittent demand. Alert on any non-zero week.",
         "kpi":"Minimise dead stock. Quarterly SKU rationalisation review."},
    ]
    for i,s in enumerate(strats):
        n_c  = (pf["cluster"]==i).sum()
        ar_c = pf[pf["cluster"]==i]["anomaly_rate"].mean()*100
        st.markdown(f"""
        <div style="border-radius:10px;padding:16px 20px;margin-bottom:12px;
                    border:1px solid #e2e8f0;border-left:4px solid {s['c']}">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <span style="font-size:1.3rem">{s['e']}</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.95rem;
                         font-weight:700;color:{s['c']}">C{i} — {s['name']}</span>
            <span style="margin-left:auto;font-size:0.78rem;color:#94a3b8">
              {n_c:,} products · {ar_c:.1f}% anomaly rate</span>
          </div>
          <div style="color:#64748b;font-size:0.82rem;margin-bottom:6px">📊 {s['p']}</div>
          <div style="color:#1e293b;font-size:0.88rem;margin-bottom:3px">📦 <b>Inventory:</b> {s['inv']}</div>
          <div style="color:#1e293b;font-size:0.88rem;margin-bottom:3px">🔍 <b>Detection:</b> {s['det']}</div>
          <div style="background:#f8fafc;border-radius:4px;padding:3px 10px;
                      font-size:0.78rem;color:#475569;display:inline-block;margin-top:6px">
            📈 {s['kpi']}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#cbd5e1;font-size:0.78rem;font-family:Space Mono,monospace'>Demand Anomaly Detection · UCI Online Retail · Isolation Forest + Z-Score + K-Means · Streamlit</div>", unsafe_allow_html=True)
