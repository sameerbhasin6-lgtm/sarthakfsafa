import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# STEP 1: SYNTHETIC DATA GENERATION (INDIAN CONTEXT)
# ============================================
@st.cache_data
def generate_synthetic_data(n_companies=50, years=5):
    """
    Generates a synthetic dataset of Indian companies.
    Half normal, half with manipulated estimates.
    """
    np.random.seed(42) # Reproducible results
    
    # Generate realistic sounding Indian Company Names
    prefixes = ["Bharat", "Hindustan", "Deccan", "Mumbai", "Bengaluru", "Kolkata", "Indus", "Ganga", "Royal", "National", "Tata", "Reliance", "Adani", "Infosys", "Wipro"]
    suffixes = ["Motors", "Power", "Steel", "Infotech", "Pharma", "Textiles", "Logistics", "Chemicals", "Agro", "Cements", "Enterprises", "Energy"]
    
    company_names = []
    # Create unique names
    for _ in range(n_companies):
        p = np.random.choice(prefixes)
        s = np.random.choice(suffixes)
        # Add a random ID to ensure uniqueness if prefix+suffix repeat
        name = f"{p} {s} Ltd."
        if name in company_names:
            name = f"{p} {s} (India) Ltd."
        company_names.append(name)

    data = []

    for company in company_names:
        # Base financials (Randomized)
        revenue_base = np.random.uniform(500, 5000) # In Crores
        receivables_pct = np.random.uniform(0.15, 0.25)
        ppe_base = np.random.uniform(1000, 4000)
        
        # 50% chance of being a manipulator
        is_manipulator = np.random.rand() > 0.5
        company_type = "Manipulator" if is_manipulator else "Normal"

        for year in range(2019, 2019 + years):
            # Revenue Growth
            growth = np.random.normal(0.08, 0.03) # Indian markets often have higher nominal growth
            revenue = revenue_base * ((1 + growth) ** (year - 2019))
            
            # Normal noise
            gross_receivables = revenue * receivables_pct * np.random.normal(1, 0.05)
            gross_ppe = ppe_base * np.random.normal(1, 0.02) + (year-2019)*150

            # ---- THE MANIPULATION LOGIC ----
            if not is_manipulator:
                # Healthy: Provisions track assets closely
                ada_ratio = np.random.normal(0.05, 0.005) 
                depreciation_rate = np.random.normal(0.10, 0.01) 
            else:
                # Fraud: Revenue up, Provisions down (The "Cookie Jar" or aggressive accounting)
                drift_factor = (year - 2019) * 0.009 
                ada_ratio = max(0.01, np.random.normal(0.05, 0.005) - drift_factor)
                
                # Extending useful life artificially
                dep_drift = 0.035 if year > 2021 else 0
                depreciation_rate = max(0.04, np.random.normal(0.10, 0.01) - dep_drift)

            ada = gross_receivables * ada_ratio
            depreciation_expense = gross_ppe * depreciation_rate
            
            # Net Income Calculation
            expenses_other = revenue * 0.65
            net_income = revenue - expenses_other - depreciation_expense - (ada * 0.5) 

            data.append([company, year, company_type, revenue, gross_receivables, ada, gross_ppe, depreciation_expense, net_income])

    columns = ['Company Name', 'Year', 'Type', 'Revenue', 'Gross_Receivables', 'ADA', 'Gross_PPE', 'Depreciation_Exp', 'Net_Income']
    df = pd.DataFrame(data, columns=columns)
    return df

# ============================================
# STEP 2: FORENSIC FEATURE ENGINEERING
# ============================================
def calculate_forensic_features(df):
    df = df.copy()
    # Ratio 1: ADA to Gross Receivables (Coverage Ratio)
    df['ADA_Ratio'] = df['ADA'] / df['Gross_Receivables']
    
    # Ratio 2: Depreciation Rate 
    df['Depreciation_Rate'] = df['Depreciation_Exp'] / df['Gross_PPE']
    
    # Ratio 3: Net Income Margin 
    df['NI_Margin'] = df['Net_Income'] / df['Revenue']
    
    df.fillna(0, inplace=True)
    return df

# ============================================
# STEP 3: AI ENGINE (ISOLATION FOREST)
# ============================================
def run_anomaly_detection(df, contamination_percent=0.10):
    features = ['ADA_Ratio', 'Depreciation_Rate', 'NI_Margin']
    
    # Standardize Data 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # The Model: Isolation Forest
    model = IsolationForest(contamination=contamination_percent, random_state=42, n_estimators=100)
    
    df['Anomaly_Flag'] = model.fit_predict(X_scaled)
    
    # Calculate Risk Score (0-100)
    raw_scores = model.decision_function(X_scaled)
    normalized_scores = ((raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min())) * 100
    df['Risk_Score'] = normalized_scores

    df['Status'] = df['Anomaly_Flag'].apply(lambda x: 'High Risk' if x == -1 else 'Normal')
    
    return df

# ============================================
# STEP 4: STREAMLIT DASHBOARD UI/UX
# ============================================

st.set_page_config(page_title="Forensic Audit Dashboard", layout="wide", page_icon="🇮🇳")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Audit Controls")
    st.write("**Project:** Financial Statement Analysis & Forensic Auditing")
    st.caption("Focus: Provisioning & Estimate Manipulation")
    st.markdown("---")
    
    # Sensitivity Check 
    st.subheader("🤖 AI Sensitivity")
    contamination = st.slider("Anomaly Threshold (%)", min_value=1, max_value=25, value=10, step=1,
                              help="Higher % means the AI will flag more companies as suspicious.") / 100

    st.markdown("---")
    st.info("This dashboard analyzes synthetic financial data of Indian companies to detect accounting anomalies using Unsupervised Machine Learning.")

# --- Data Processing ---
raw_data = generate_synthetic_data()
processed_data = calculate_forensic_features(raw_data)
scored_data = run_anomaly_detection(processed_data, contamination_percent=contamination)

# Get latest year for summary views
latest_year = scored_data['Year'].max()
latest_data = scored_data[scored_data['Year'] == latest_year]

# --- Main Page Tabs ---
tab1, tab2 = st.tabs(["📊 Executive Summary", "🔍 Company Deep Dive"])

# === TAB 1: Executive Summary ===
with tab1:
    st.title("🇮🇳 Indian Corporate Forensic Dashboard")
    st.markdown(f"### Audit Overview: FY {latest_year}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Companies Audited", value=len(latest_data))
    with col2:
        high_risk_count = len(latest_data[latest_data['Status'] == 'High Risk'])
        st.metric(label="High Risk Flags", value=high_risk_count, delta="Attention Required", delta_color="inverse")
    with col3:
        avg_risk = latest_data['Risk_Score'].mean()
        st.metric(label="Avg Portfolio Risk Score", value=f"{avg_risk:.1f} / 100")

    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🚨 Top Risk Alerts")
        st.caption("Companies with the highest probability of accounting estimate manipulation.")
        
        # Sort by risk
        top_risks = latest_data.sort_values(by='Risk_Score', ascending=False).head(10)
        
        # FIXED: Using column_config to avoid Styler bugs
        st.dataframe(
            top_risks[['Company Name', 'Risk_Score', 'Status', 'ADA_Ratio', 'Depreciation_Rate']],
            column_config={
                "Company Name": st.column_config.TextColumn("Company Name"),
                "Risk_Score": st.column_config.ProgressColumn(
                    "Risk Score", 
                    format="%.1f", 
                    min_value=0, 
                    max_value=100,
                    help="AI calculated probability of anomaly"
                ),
                "ADA_Ratio": st.column_config.NumberColumn("ADA Ratio", format="%.2f%%"),
                "Depreciation_Rate": st.column_config.NumberColumn("Depr. Rate", format="%.2f%%"),
                "Status": st.column_config.TextColumn("Audit Status"),
            },
            use_container_width=True,
            hide_index=True
        )

    with col_right:
        # FIXED: Typo corrected (added parentheses)
        st.subheader("Peer Benchmarking") 
        
        fig_scatter = px.scatter(
            latest_data,
            x="NI_Margin",
            y="ADA_Ratio",
            color="Status",
            size="Risk_Score",
            hover_name="Company Name",
            color_discrete_map={"Normal": "#e0e0e0", "High Risk": "#ff4b4b"},
            title="Profitability vs. Provisions",
            labels={"NI_Margin": "Net Income Margin", "ADA_Ratio": "ADA / Receivables"}
        )
        fig_scatter.update_layout(plot_bgcolor="white", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_scatter, use_container_width=True)


# === TAB 2: Deep Dive Analysis ===
with tab2:
    st.title("🔍 Forensic Audit Report")
    
    # Select Company
    company_list = scored_data['Company Name'].unique()
    selected_company = st.selectbox("Select Company to Audit:", company_list)
    
    # Filter Data
    company_data = scored_data[scored_data['Company Name'] == selected_company].sort_values(by='Year')
    current_risk = company_data.iloc[-1]['Risk_Score']
    current_status = company_data.iloc[-1]['Status']

    # -- Status Header --
    c1, c2 = st.columns([1, 3])
    with c1:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_risk,
            title = {'text': "Risk Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "red" if current_risk > 65 else "orange" if current_risk > 40 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': '#f0f2f6'},
                    {'range': [40, 70], 'color': '#fff4e6'},
                    {'range': [70, 100], 'color': '#ffe6e6'}],
            }))
        fig_gauge.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        st.subheader(f"Audit Summary: {selected_company}")
        
        # Dynamic Text Logic
        latest_ada = company_data.iloc[-1]['ADA_Ratio']
        first_ada = company_data.iloc[0]['ADA_Ratio']
        ada_trend = "declining aggressively" if latest_ada < first_ada * 0.7 else "stable"
        
        latest_dep = company_data.iloc[-1]['Depreciation_Rate']
        first_dep = company_data.iloc[0]['Depreciation_Rate']
        dep_trend = "suspiciously low" if latest_dep < first_dep * 0.8 else "consistent"

        st.markdown(f"""
        * **Classification:** {current_status}
        * **Provisioning Analysis:** The Allowance for Doubtful Accounts (ADA) ratio is **{ada_trend}**.
        * **Asset Lifecycle:** Depreciation expenses appear **{dep_trend}** relative to Gross PP&E.
        """)
        
        if current_status == "High Risk":
            st.error("⚠️ FLAG: This company shows signs of earnings smoothing (increasing profit by reducing estimates).")
        else:
            st.success("✅ PASSED: Financial estimates appear consistent with industry peers.")

    st.markdown("---")

    # -- The Jaws Chart --
    st.subheader("📉 Divergence Analysis (The 'Jaws of Death')")
    st.caption("Red Flag: When Revenue (Activity) rises while Provisions (Caution) fall.")
    
    # Normalize for comparison
    base = company_data.iloc[0]
    norm = company_data.copy()
    norm['Revenue_Index'] = (norm['Revenue'] / base['Revenue']) * 100
    norm['ADA_Index'] = (norm['ADA'] / base['ADA']) * 100
    
    # Melt
    df_melt = norm.melt('Year', value_vars=['Revenue_Index', 'ADA_Index'], var_name='Metric', value_name='Index (Base=100)')
    
    fig_jaws = px.line(
        df_melt, x='Year', y='Index (Base=100)', color='Metric', markers=True,
        color_discrete_map={'Revenue_Index': '#2E86C1', 'ADA_Index': '#C0392B'}
    )
    fig_jaws.update_layout(plot_bgcolor="white", hovermode="x unified")
    st.plotly_chart(fig_jaws, use_container_width=True)

    with st.expander("📄 View Raw Financial Data (Crores)"):
        st.dataframe(company_data, hide_index=True)
