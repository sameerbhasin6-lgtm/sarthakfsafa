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
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CUSTOM CSS (SAFE, MINIMAL, ELEGANT)
# =====================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f2a44;
    }
    h2, h3 {
        color: #2c3e50;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.title("AI Accounting Risk Analyzer")
st.caption("Early warning system for aggressive accounting behavior using accrual analysis")

st.markdown("---")

# =====================================================
# REALISTIC COMPANY NAMES
# =====================================================
COMPANY_NAMES = [
    "Tata Steel",
    "Reliance Industries",
    "Infosys",
    "HDFC Bank",
    "ICICI Bank",
    "Larsen and Toubro",
    "Bharti Airtel",
    "Axis Bank",
    "Hindustan Unilever",
    "ITC",
    "JSW Steel",
    "Adani Ports",
    "NTPC",
    "Power Grid Corporation",
    "Maruti Suzuki",
    "UltraTech Cement",
    "Asian Paints",
    "Sun Pharma",
    "Dr Reddy Laboratories",
    "Bajaj Auto",
    "Bajaj Finance",
    "State Bank of India",
    "Coal India",
    "ONGC",
    "Tata Motors"
]

# =====================================================
# DATA GENERATION
# =====================================================
@st.cache_data
def generate_synthetic_data(companies, years=5):
    np.random.seed(42)

    year_list = list(range(2019, 2019 + years))
    rows = []

    for company in companies:
        base_revenue = np.random.uniform(8000, 120000)
        base_ada = base_revenue * np.random.uniform(0.06, 0.14)

        manipulated = np.random.rand() < 0.3  # 30 percent aggressive accounting

        for y in year_list:
            revenue_growth = np.random.uniform(0.04, 0.14)
            ada_growth = np.random.uniform(0.03, 0.12)

            revenue = base_revenue * (1 + revenue_growth) ** (y - year_list[0])
            ada = base_ada * (1 + ada_growth) ** (y - year_list[0])

            if manipulated:
                ada *= np.random.uniform(1.3, 1.7)

            rows.append([
                company,
                y,
                round(revenue, 2),
                round(ada, 2)
            ])

    return pd.DataFrame(
        rows,
        columns=["Company", "Year", "Revenue", "ADA"]
    )

df = generate_synthetic_data(COMPANY_NAMES)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
features = (
    df.groupby("Company")[["Revenue", "ADA"]]
    .mean()
    .reset_index()
)

features["ADA_to_Revenue"] = features["ADA"] / features["Revenue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    features[["Revenue", "ADA", "ADA_to_Revenue"]]
)

# =====================================================
# ANOMALY DETECTION
# =====================================================
model = IsolationForest(
    n_estimators=200,
    contamination=0.25,
    random_state=42
)

features["Anomaly_Score"] = model.fit_predict(X_scaled)
features["Risk Flag"] = features["Anomaly_Score"].map(
    {-1: "High Risk", 1: "Normal"}
)

# =====================================================
# METRICS
# =====================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='metric-box'><h3>Total Companies</h3><h2>{}</h2></div>".format(len(features)), unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-box'><h3>High Risk Flags</h3><h2>{}</h2></div>".format(
        (features["Risk Flag"] == "High Risk").sum()
    ), unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-box'><h3>Normal Companies</h3><h2>{}</h2></div>".format(
        (features["Risk Flag"] == "Normal").sum()
    ), unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# SCATTER PLOT
# =====================================================
st.subheader("Revenue vs Accounting Discretionary Accruals")

fig = px.scatter(
    features,
    x="Revenue",
    y="ADA",
    color="Risk Flag",
    hover_name="Company",
    hover_data={"ADA_to_Revenue": ":.2%"},
    color_discrete_map={
        "High Risk": "#c0392b",
        "Normal": "#2980b9"
    }
)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# COMPANY LEVEL ANALYSIS
# =====================================================
st.markdown("### Company-Level Trend Analysis")

selected_company = st.selectbox(
    "Select a company",
    options=features["Company"].sort_values()
)

company_data = df[df["Company"] == selected_company].sort_values("Year")

base_year = company_data.iloc[0]
company_data["Revenue Index"] = (company_data["Revenue"] / base_year["Revenue"]) * 100
company_data["ADA Index"] = (company_data["ADA"] / base_year["ADA"]) * 100

trend_data = company_data.melt(
    id_vars="Year",
    value_vars=["Revenue Index", "ADA Index"],
    var_name="Metric",
    value_name="Index (Base Year = 100)"
)

fig_trend = px.line(
    trend_data,
    x="Year",
    y="Index (Base Year = 100)",
    color="Metric",
    markers=True,
    color_discrete_map={
        "Revenue Index": "#2980b9",
        "ADA Index": "#c0392b"
    }
)

fig_trend.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified"
)

st.plotly_chart(fig_trend, use_container_width=True)

# =====================================================
# RAW DATA
# =====================================================
with st.expander("View Financial Data (INR Crore)"):
    st.dataframe(company_data, use_container_width=True)

