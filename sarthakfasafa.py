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
# SIDEBAR
# =====================================================
st.sidebar.markdown("## Accounting Risk Analyzer")
st.sidebar.caption("Synthetic financial risk dashboard")

years = st.sidebar.slider(
    "Analysis period (years)",
    min_value=3,
    max_value=8,
    value=5
)

manipulation_share = st.sidebar.slider(
    "Aggressive accounting probability",
    min_value=0.1,
    max_value=0.5,
    value=0.3
)

st.sidebar.markdown("---")

# =====================================================
# COMPANY NAMES
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
# FEATURE ENGINEERING
# =====================================================
features = df.groupby("Company")[["Revenue", "ADA"]].mean().reset_index()
features["ADA_to_Revenue"] = features["ADA"] / features["Revenue"]

X = StandardScaler().fit_transform(
    features[["Revenue", "ADA", "ADA_to_Revenue"]]
)

model = IsolationForest(
    n_estimators=200,
    contamination=0.25,
    random_state=42
)

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
st.caption("Accrual-based early warning system for aggressive accounting")
st.divider()

# =====================================================
# KPIs
# =====================================================
k1, k2, k3 = st.columns(3)

with k1:
    st.container(border=True).metric("Total Companies", len(features))

with k2:
    st.container(border=True).metric(
        "High Risk Companies",
        (features["Risk Category"] == "High Risk").sum()
    )

with k3:
    st.container(border=True).metric(
        "Normal Companies",
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
# COMPANY DATA
# =====================================================
company_data = df[df["Company"] == selected_company].sort_values("Year")
base = company_data.iloc[0]

company_data["Revenue Index"] = company_data["Revenue"] / base["Revenue"] * 100
company_data["ADA Index"] = company_data["ADA"] / base["ADA"] * 100
company_data["ADA Ratio"] = company_data["ADA"] / company_data["Revenue"]

# =====================================================
# 2. JAWS EFFECT
# =====================================================
st.markdown("## 2. JAWS Effect: Revenue vs ADA Growth")

fig2 = px.line(
    company_data,
    x="Year",
    y=["Revenue Index", "ADA Index"],
    markers=True
)

fig2.update_layout(height=450, plot_bgcolor="white", hovermode="x unified")
st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# 3. ADA INTENSITY
# =====================================================
st.markdown("## 3. ADA Intensity (ADA / Revenue)")

fig3 = px.bar(
    company_data,
    x="Year",
    y="ADA Ratio",
    text_auto=".2%"
)

fig3.update_layout(height=400, plot_bgcolor="white")
st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# 4. TREND STABILITY
# =====================================================
st.markdown("## 4. Earnings Quality Stability")

company_data["ADA YoY Change"] = company_data["ADA"].pct_change()

fig4 = px.line(
    company_data,
    x="Year",
    y="ADA YoY Change",
    markers=True
)

fig4.update_layout(height=400, plot_bgcolor="white")
st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# 5. PEER POSITIONING
# =====================================================
st.markdown("## 5. Peer Positioning")

peer_data = features.copy()
peer_data["Selected"] = peer_data["Company"] == selected_company

fig5 = px.scatter(
    peer_data,
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
with st.expander("View financial data"):
    st.dataframe(company_data, use_container_width=True)
