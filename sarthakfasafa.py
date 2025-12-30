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
st.sidebar.caption("Synthetic demo for analysis")

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
st.sidebar.caption("Academic use only")

# =====================================================
# HEADER
# =====================================================
st.markdown("# Accounting Risk Monitoring Dashboard")
st.caption("Detecting aggressive accounting behavior using discretionary accruals")

st.divider()

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

            rows.append([
                company,
                year,
                round(revenue, 2),
                round(ada, 2)
            ])

    return pd.DataFrame(
        rows,
        columns=["Company", "Year", "Revenue", "ADA"]
    )

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
# KPI SECTION
# =====================================================
k1, k2, k3 = st.columns(3)

with k1:
    st.container(border=True).metric(
        "Total Companies",
        len(features)
    )

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
# TABS
# =====================================================
tab1, tab2 = st.tabs(["Risk Map", "Company Deep Dive"])

# =====================================================
# TAB 1: RISK MAP
# =====================================================
with tab1:
    st.markdown("## Revenue vs Discretionary Accruals")

    fig = px.scatter(
        features,
        x="Revenue",
        y="ADA",
        color="Risk Category",
        hover_name="Company",
        hover_data={"ADA_to_Revenue": ":.2%"},
        color_discrete_map={
            "High Risk": "#d62728",
            "Normal": "#1f77b4"
        }
    )

    fig.update_layout(
        height=550,
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2: COMPANY DEEP DIVE
# =====================================================
with tab2:
    st.markdown("## Accounting Trend Analysis")

    company = st.selectbox(
        "Select company",
        sorted(features["Company"].unique())
    )

    company_data = df[df["Company"] == company].sort_values("Year")
    base_year = company_data.iloc[0]

    company_data["Revenue Index"] = company_data["Revenue"] / base_year["Revenue"] * 100
    company_data["ADA Index"] = company_data["ADA"] / base_year["ADA"] * 100

    melted = company_data.melt(
        id_vars="Year",
        value_vars=["Revenue Index", "ADA Index"],
        var_name="Metric",
        value_name="Index (Base = 100)"
    )

    fig2 = px.line(
        melted,
        x="Year",
        y="Index (Base = 100)",
        color="Metric",
        markers=True
    )

    fig2.update_layout(
        height=500,
        hovermode="x unified",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View financial data"):
        st.dataframe(company_data, use_container_width=True)

