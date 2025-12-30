import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Accounting Risk Dashboard",
    layout="wide"
)

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.markdown("## Accounting Risk Analyzer")
st.sidebar.caption("Accrual-based risk analysis")

years = st.sidebar.slider("Analysis period (years)", 3, 8, 5)
manipulation_share = st.sidebar.slider(
    "Aggressive accounting probability", 0.1, 0.5, 0.3
)

st.sidebar.markdown("---")

# =====================================================
# COMPANY LIST
# =====================================================
COMPANY_NAMES = [
    "Tata Steel", "Reliance Industries", "Infosys", "HDFC Bank",
    "ICICI Bank", "Larsen and Toubro", "Bharti Airtel", "Axis Bank",
    "Hindustan Unilever", "ITC", "JSW Steel", "Adani Ports",
    "NTPC", "Power Grid Corporation", "Maruti Suzuki",
    "UltraTech Cement", "Asian Paints", "Sun Pharma",
    "Dr Reddy Laboratories", "Bajaj Auto", "Bajaj Finance",
    "State Bank of India", "Coal India", "ONGC", "Tata Motors"
]

# =====================================================
# DATA GENERATION
# =====================================================
@st.cache_data
def generate_data(companies, years, manipulation_share):
    np.random.seed(42)
    rows = []
    year_list = list(range(2019, 2019 + years))

    for company in companies:
        base_revenue = np.random.uniform(10000, 120000)
        base_ada = base_revenue * np.random.uniform(0.06, 0.14)
        aggressive = np.random.rand() < manipulation_share

        for year in year_list:
            revenue = base_revenue * (1 + np.random.uniform(0.04, 0.14)) ** (year - year_list[0])
            ada = base_ada * (1 + np.random.uniform(0.03, 0.12)) ** (year - year_list[0])

            if aggressive:
                ada *= np.random.uniform(1.3, 1.7)

            rows.append([company, year, revenue, ada])

    return pd.DataFrame(rows, columns=["Company", "Year", "Revenue", "ADA"])

df = generate_data(COMPANY_NAMES, years, manipulation_share)

# =====================================================
# FEATURE ENGINEERING + ANOMALY MODEL
# =====================================================
features = df.groupby("Company")[["Revenue", "ADA"]].mean().reset_index()
features["ADA_to_Revenue"] = features["ADA"] / features["Revenue"]

X = StandardScaler().fit_transform(
    features[["Revenue", "ADA", "ADA_to_Revenue"]]
)

model = IsolationForest(n_estimators=200, contamination=0.25, random_state=42)
features["Flag"] = model.fit_predict(X)
features["Risk Category"] = features["Flag"].map({-1: "High Risk", 1: "Normal"})

# =====================================================
# SIDEBAR COMPANY SELECTOR
# =====================================================
selected_company = st.sidebar.selectbox(
    "Select company for deep dive",
    sorted(features["Company"].unique())
)

# =====================================================
# HEADER
# =====================================================
st.markdown("# Accounting Risk Monitoring Dashboard")
st.caption("Structured 1–5 analysis of discretionary accrual risk")
st.divider()

# =====================================================
# KPI CARDS
# =====================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.container(border=True).metric("Total Companies", len(features))
with c2:
    st.container(border=True).metric(
        "High Risk Firms",
        (features["Risk Category"] == "High Risk").sum()
    )
with c3:
    st.container(border=True).metric(
        "Normal Firms",
        (features["Risk Category"] == "Normal").sum()
    )

st.divider()

# =====================================================
# 1. RISK MAP
# =====================================================
st.markdown("## 1. Risk Map: Revenue vs Discretionary Accruals")

fig1 = px.scatter(
    features,
    x="Revenue",
    y="ADA",
    color="Risk Category",
    hover_name="Company",
    hover_data={"ADA_to_Revenue": ":.2%"},
    color_discrete_map={"High Risk": "#d62728", "Normal": "#1f77b4"}
)

fig1.update_layout(height=500, plot_bgcolor="white")
st.plotly_chart(fig1, use_container_width=True)

# =====================================================
# SELECTED COMPANY DATA
# =====================================================
company_df = df[df["Company"] == selected_company].sort_values("Year")
base = company_df.iloc[0]

company_df["Revenue Index"] = company_df["Revenue"] / base["Revenue"] * 100
company_df["ADA Index"] = company_df["ADA"] / base["ADA"] * 100
company_df["ADA Ratio"] = company_df["ADA"] / company_df["Revenue"]
company_df["ADA YoY"] = company_df["ADA"].pct_change()

# =====================================================
# 2. JAWS EFFECT
# =====================================================
st.markdown("## 2. JAWS Effect: Revenue Growth vs ADA Growth")

fig2 = px.line(
    company_df,
    x="Year",
    y=["Revenue Index", "ADA Index"],
    markers=True
)

fig2.update_layout(height=450, hovermode="x unified", plot_bgcolor="white")
st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# 3. ADA INTENSITY
# =====================================================
st.markdown("## 3. ADA Intensity (ADA / Revenue)")

fig3 = px.bar(
    company_df,
    x="Year",
    y="ADA Ratio",
    text_auto=".2%"
)

fig3.update_layout(height=400, plot_bgcolor="white")
st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# 4. EARNINGS STABILITY
# =====================================================
st.markdown("## 4. Earnings Quality Stability (YoY ADA Change)")

fig4 = px.line(
    company_df,
    x="Year",
    y="ADA YoY",
    markers=True
)

fig4.update_layout(height=400, plot_bgcolor="white")
st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# 5. PEER POSITIONING
# =====================================================
st.markdown("## 5. Peer Positioning vs Market")

peer_df = features.copy()
peer_df["Selected"] = peer_df["Company"] == selected_company

fig5 = px.scatter(
    peer_df,
    x="Revenue",
    y="ADA_to_Revenue",
    size="Revenue",
    color="Selected",
    hover_name="Company",
    color_discrete_map={True: "#d62728", False: "#7f8c8d"}
)

fig5.update_layout(height=500, plot_bgcolor="white")
st.plotly_chart(fig5, use_container_width=True)

# =====================================================
# RAW DATA
# =====================================================
with st.expander("View company financials"):
    st.dataframe(company_df, use_container_width=True)

