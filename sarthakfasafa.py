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
    page_title="AI Accounting Risk Analyzer",
    layout="wide"
)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("AI Risk Analyzer")
st.sidebar.caption("Accounting Quality Dashboard")

years = st.sidebar.slider(
    "Analysis Period (Years)",
    min_value=3,
    max_value=8,
    value=5
)

risk_cutoff = st.sidebar.slider(
    "Aggressive Accounting Share",
    min_value=0.1,
    max_value=0.5,
    value=0.3
)

st.sidebar.markdown("---")
st.sidebar.caption("Synthetic demo for academic use")

# =====================================================
# HEADER
# =====================================================
st.title("Accounting Risk Monitoring Dashboard")
st.caption("Identifying aggressive accounting behavior using accrual-based signals")

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

        for y in year_list:
            revenue = base_revenue * (1 + np.random.uniform(0.04, 0.14)) ** (y - year_list[0])
            ada = base_ada * (1 + np.random.uniform(0.03, 0.12)) ** (y - year_list[0])

            if aggressive:
                ada *= np.random.uniform(1.3, 1.7)

            rows.append([company, y, round(revenue, 2), round(ada, 2)])

    return pd.DataFrame(rows, columns=["Company", "Year", "Revenue", "ADA"])


df = generate_data(COMPANY_NAMES, years, risk_cutoff)

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
# KPI SECTION (VISIBLY DIFFERENT)
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
# TABS (MAJOR VISUAL CHANGE)
# =====================================================
tab1, tab2 = st.tabs([
    "Risk Map",
    "Company Deep Dive"
])

# =====================================================
# TAB 1: SCATTER
# =====================================================
with tab1:
st.markdown("### Revenue vs Discretionary Accruals")

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
# TAB 2: COMPANY TREND
# =====================================================
with tab2:
    st.markdown("### Accounting Trend Comparison")

    company = st.selectbox(
        "Select Company",
        sorted(features["Company"].unique())
    )

    data = df[df["Company"] == company].sort_values("Year")
    base = data.iloc[0]

    data["Revenue Index"] = data["Revenue"] / base["Revenue"] * 100
    data["ADA Index"] = data["ADA"] / base["ADA"] * 100

    melted = data.melt(
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

    with st.expander("View Financials"):
        st.dataframe(data, use_container_width=True)
