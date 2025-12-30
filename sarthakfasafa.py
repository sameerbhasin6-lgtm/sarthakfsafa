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

st.title("📊 AI Accounting Risk Analyzer (Synthetic Demo)")
st.caption("Detecting abnormal accounting behaviour using ADA & Revenue trends")

# =====================================================
# DATA GENERATION
# =====================================================
@st.cache_data
def generate_synthetic_data(n_companies=50, years=5):
    np.random.seed(42)

    companies = [f"Company_{i+1}" for i in range(n_companies)]
    year_list = list(range(2019, 2019 + years))

    rows = []

    for company in companies:
        base_revenue = np.random.uniform(800, 5000)
        base_ada = base_revenue * np.random.uniform(0.08, 0.15)

        manipulated = np.random.rand() < 0.3  # 30% manipulation

        for y in year_list:
            growth = np.random.uniform(0.05, 0.15)
            revenue = base_revenue * (1 + growth) ** (y - year_list[0])

            ada = base_ada * (1 + np.random.uniform(0.04, 0.12)) ** (y - year_list[0])

            if manipulated:
                ada *= np.random.uniform(1.3, 1.8)  # aggressive estimates

            rows.append([
                company, y, round(revenue, 2), round(ada, 2)
            ])

    df = pd.DataFrame(
        rows,
        columns=["Company", "Year", "Revenue", "ADA"]
    )

    return df


df = generate_synthetic_data()

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
X_scaled = scaler.fit_transform(features[["Revenue", "ADA", "ADA_to_Revenue"]])

# =====================================================
# ANOMALY DETECTION
# =====================================================
model = IsolationForest(
    n_estimators=200,
    contamination=0.25,
    random_state=42
)
features["Anomaly_Score"] = model.fit_predict(X_scaled)
features["Risk_Flag"] = features["Anomaly_Score"].map({-1: "High Risk", 1: "Normal"})

# =====================================================
# DASHBOARD METRICS
# =====================================================
col1, col2, col3 = st.columns(3)

col1.metric("Total Companies", len(features))
col2.metric("High Risk Flags", (features["Risk_Flag"] == "High Risk").sum())
col3.metric("Normal Companies", (features["Risk_Flag"] == "Normal").sum())

# =====================================================
# SCATTER PLOT
# =====================================================
fig = px.scatter(
    features,
    x="Revenue",
    y="ADA",
    color="Risk_Flag",
    hover_data=["Company", "ADA_to_Revenue"],
    title="Revenue vs Accounting Discretion Accruals (ADA)",
    color_discrete_map={
        "High Risk": "#C0392B",
        "Normal": "#2E86C1"
    }
)

fig.update_layout(plot_bgcolor="white")

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# COMPANY DEEP DIVE
# =====================================================
st.subheader("Company-Level Trend Analysis")

selected_company = st.selectbox(
    "Select a company",
    options=features["Company"].unique()
)

company_data = df[df["Company"] == selected_company].sort_values("Year")

base_year = company_data.iloc[0]
company_data["Revenue_Index"] = (company_data["Revenue"] / base_year["Revenue"]) * 100
company_data["ADA_Index"] = (company_data["ADA"] / base_year["ADA"]) * 100

melted = company_data.melt(
    id_vars="Year",
    value_vars=["Revenue_Index", "ADA_Index"],
    var_name="Metric",
    value_name="Index (Base = 100)"
)

fig_jaws = px.line(
    melted,
    x="Year",
    y="Index (Base = 100)",
    color="Metric",
    markers=True,
    title="JAWS Effect: ADA Growth vs Revenue Growth",
    color_discrete_map={
        "Revenue_Index": "#2E86C1",
        "ADA_Index": "#C0392B"
    }
)

fig_jaws.update_layout(plot_bgcolor="white", hovermode="x unified")

st.plotly_chart(fig_jaws, use_container_width=True)

# =====================================================
# RAW DATA
# =====================================================
with st.expander("📄 View Raw Financial Data (₹ Crores)"):
    st.dataframe(company_data, use_container_width=True)

