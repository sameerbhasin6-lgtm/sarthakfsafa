import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import math

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Accounting Risk Dashboard", layout="wide")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("## Accounting Risk Analyzer")
years = st.sidebar.slider("Analysis period (years)", 3, 8, 5)
manipulation_share = st.sidebar.slider("Aggressive accounting probability", 0.1, 0.5, 0.3)

# =====================================================
# COMPANY LIST
# =====================================================
COMPANIES = [
    "Tata Steel","Reliance Industries","Infosys","HDFC Bank","ICICI Bank",
    "Larsen and Toubro","Bharti Airtel","Axis Bank","Hindustan Unilever",
    "ITC","JSW Steel","Adani Ports","NTPC","Power Grid Corporation",
    "Maruti Suzuki","UltraTech Cement","Asian Paints","Sun Pharma",
    "Dr Reddy Laboratories","Bajaj Auto","Bajaj Finance",
    "State Bank of India","Coal India","ONGC","Tata Motors"
]

# =====================================================
# DATA GENERATION
# =====================================================
@st.cache_data
def generate_data(companies, years, manipulation):
    np.random.seed(42)
    rows = []
    years_list = list(range(2019, 2019 + years))

    for c in companies:
        base_rev = np.random.uniform(10000, 120000)
        base_ada = base_rev * np.random.uniform(0.06, 0.14)
        aggressive = np.random.rand() < manipulation

        for y in years_list:
            rev = base_rev * (1 + np.random.uniform(0.04, 0.14)) ** (y - years_list[0])
            ada = base_ada * (1 + np.random.uniform(0.03, 0.12)) ** (y - years_list[0])
            if aggressive:
                ada *= np.random.uniform(1.3, 1.7)
            rows.append([c, y, rev, ada])

    return pd.DataFrame(rows, columns=["Company","Year","Revenue","ADA"])

df = generate_data(COMPANIES, years, manipulation_share)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
features = df.groupby("Company")[["Revenue","ADA"]].mean().reset_index()
features["ADA_to_Revenue"] = features["ADA"] / features["Revenue"]

X = StandardScaler().fit_transform(features[["Revenue","ADA","ADA_to_Revenue"]])
iso = IsolationForest(n_estimators=200, contamination=0.25, random_state=42)
features["Anomaly"] = iso.fit_predict(X)
features["Risk Category"] = features["Anomaly"].map({-1:"High Risk",1:"Normal"})

# =====================================================
# SIDEBAR COMPANY SELECTOR
# =====================================================
company = st.sidebar.selectbox("Select company", sorted(features["Company"]))

company_df = df[df["Company"] == company].sort_values("Year")
base = company_df.iloc[0]

company_df["Revenue Index"] = company_df["Revenue"] / base["Revenue"] * 100
company_df["ADA Index"] = company_df["ADA"] / base["ADA"] * 100
company_df["ADA Ratio"] = company_df["ADA"] / company_df["Revenue"]
company_df["ADA YoY"] = company_df["ADA"].pct_change()

# =====================================================
# HEADER
# =====================================================
st.markdown("# Accounting Risk Monitoring Dashboard")
st.caption("Accrual quality, fraud risk and credit-style assessment")

# =====================================================
# KPI
# =====================================================
c1,c2,c3 = st.columns(3)
with c1: st.container(border=True).metric("Total Companies", len(features))
with c2: st.container(border=True).metric("High Risk Firms", (features["Risk Category"]=="High Risk").sum())
with c3: st.container(border=True).metric("Selected Firm Risk", features.loc[features["Company"]==company,"Risk Category"].iloc[0])

# =====================================================
# 1–5 CORE ANALYSIS
# =====================================================
st.markdown("## 1. Risk Map")
st.plotly_chart(px.scatter(features,x="Revenue",y="ADA",color="Risk Category",
                           hover_name="Company"),use_container_width=True)

st.markdown("## 2. JAWS Effect")
st.plotly_chart(px.line(company_df,x="Year",y=["Revenue Index","ADA Index"],markers=True),
                use_container_width=True)

st.markdown("## 3. ADA Intensity")
st.plotly_chart(px.bar(company_df,x="Year",y="ADA Ratio"),use_container_width=True)

st.markdown("## 4. Earnings Stability")
st.plotly_chart(px.line(company_df,x="Year",y="ADA YoY",markers=True),
                use_container_width=True)

st.markdown("## 5. Peer Positioning")
peer = features.copy()
peer["Selected"] = peer["Company"] == company
st.plotly_chart(px.scatter(peer,x="Revenue",y="ADA_to_Revenue",color="Selected",
                           hover_name="Company"),use_container_width=True)

# =====================================================
# 6. MONTIER C-SCORE (SIMPLIFIED)
# =====================================================
st.markdown("## 6. Montier C-Score")

c_score = 0
if company_df["ADA YoY"].std() < 0.05: c_score += 1
if company_df["ADA Ratio"].mean() > 0.12: c_score += 1
if company_df["ADA Index"].iloc[-1] > company_df["Revenue Index"].iloc[-1]: c_score += 1

st.metric("Montier C-Score (0–3)", c_score)

# =====================================================
# 7. BENFORD LAW
# =====================================================
st.markdown("## 7. Benford Law Check")

def leading_digit(x):
    x = abs(x)
    return int(str(x)[0]) if x > 0 else None

company_df["Leading Digit"] = company_df["Revenue"].apply(leading_digit)
benford = company_df["Leading Digit"].value_counts(normalize=True).sort_index()
expected = {d: math.log10(1 + 1/d) for d in range(1,10)}

benford_df = pd.DataFrame({
    "Actual": benford,
    "Expected": pd.Series(expected)
}).fillna(0)

st.plotly_chart(px.bar(benford_df,barmode="group"),use_container_width=True)

# =====================================================
# 8. CREDIT RATING
# =====================================================
st.markdown("## 8. Credit Style Risk Rating")

if c_score >= 2 and features.loc[features["Company"]==company,"Risk Category"].iloc[0]=="High Risk":
    rating = "C"
elif c_score >= 2:
    rating = "BB"
elif features.loc[features["Company"]==company,"Risk Category"].iloc[0]=="High Risk":
    rating = "B"
else:
    rating = "A"

st.metric("Implied Credit Rating", rating)

# =====================================================
# 9. PDF EXPORT
# =====================================================
st.markdown("## 9. Export Credit Memo")

if st.button("Generate PDF Credit Memo"):
    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()
    content = [
        Paragraph(f"Credit Risk Summary: {company}", styles["Title"]),
        Spacer(1,12),
        Paragraph(f"Risk Category: {features.loc[features['Company']==company,'Risk Category'].iloc[0]}", styles["Normal"]),
        Paragraph(f"Montier C-Score: {c_score}", styles["Normal"]),
        Paragraph(f"Credit Rating: {rating}", styles["Normal"])
    ]
    doc.build(content)
    with open(tmp.name,"rb") as f:
        st.download_button("Download PDF", f, file_name=f"{company}_credit_memo.pdf")


