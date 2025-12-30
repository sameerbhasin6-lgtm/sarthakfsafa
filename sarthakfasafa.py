import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# STEP 1: SYNTHETIC DATA GENERATION
# ============================================
@st.cache_data # Cache data so it doesn't reload on every interaction
def generate_synthetic_data(n_companies=50, years=5):
    """
    Generates a synthetic dataset of financial statements.
    Half normal, half with manipulated estimates.
    """
    np.random.seed(42) # for reproducibility
    companies = [f"Tick-{i:02d}" for i in range(n_companies)]
    data = []

    for company in companies:
        # Base financials
        revenue_base = np.random.uniform(500, 2000)
        receivables_pct = np.random.uniform(0.15, 0.25)
        ppe_base = np.random.uniform(1000, 3000)
        
        is_manipulator = np.random.rand() > 0.5
        company_type = "Manipulator" if is_manipulator else "Normal"

        for year in range(2019, 2019 + years):
            # Revenue Growth with some noise
            growth = np.random.normal(0.05, 0.02)
            revenue = revenue_base * ((1 + growth) ** (year - 2019))
            gross_receivables = revenue * receivables_pct * np.random.normal(1, 0.05)
            gross_ppe = ppe_base * np.random.normal(1, 0.02) + (year-2019)*100

            # ---- THE MANIPULATION LOGIC ----
            if not is_manipulator:
                # Normal: Provisions track assets closely
                ada_ratio = np.random.normal(0.05, 0.005) # ~5% of receivables
                depreciation_rate = np.random.normal(0.10, 0.01) # ~10% of PPE
            else:
                # Manipulator: Provisions diverge to boost earnings
                # As time goes on, they under-provision more aggressively
                drift_factor = (year - 2019) * 0.008 
                ada_ratio = max(0.01, np.random.normal(0.05, 0.005) - drift_factor)
                
                # Suddenly extending useful life in later years
                dep_drift = 0.03 if year > 2021 else 0
                depreciation_rate = max(0.04, np.random.normal(0.10, 0.01) - dep_drift)

            ada = gross_receivables * ada_ratio
            depreciation_expense = gross_ppe * depreciation_rate
            
            # Net Income (Simplified)
            # Manipulators will have artificially higher net income due to lower provisions
            expenses_other = revenue * 0.7
            net_income = revenue - expenses_other - depreciation_expense - (ada * 0.5) 

            data.append([company, year, company_type, revenue, gross_receivables, ada, gross_ppe, depreciation_expense, net_income])

    columns = ['Ticker', 'Year', 'Type', 'Revenue', 'Gross_Receivables', 'ADA', 'Gross_PPE', 'Depreciation_Exp', 'Net_Income']
    df = pd.DataFrame(data, columns=columns)
    return df

# ============================================
# STEP 2: FORENSIC FEATURE ENGINEERING
# ============================================
def calculate_forensic_features(df):
    df = df.copy()
    # Ratio 1: ADA to Gross Receivables (Are they covering their bad debts?)
    df['ADA_Ratio'] = df['ADA'] / df['Gross_Receivables']
    
    # Ratio 2: Depreciation Rate (Are they changing useful lives?)
    df['Depreciation_Rate'] = df['Depreciation_Exp'] / df['Gross_PPE']
    
    # Ratio 3: Net Income Margin (For context)
    df['NI_Margin'] = df['Net_Income'] / df['Revenue']
    
    # Fill potential NaN from division by zero if data is bad
    df.fillna(0, inplace=True)
    return df

# ============================================
# STEP 3: AI ENGINE (ISOLATION FOREST)
# ============================================
def run_anomaly_detection(df, contamination_percent=0.10):
    # Select features for the AI model
    features = ['ADA_Ratio', 'Depreciation_Rate', 'NI_Margin']
    
    # Standardize Data (Important for ML)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # The Model: Isolation Forest
    # Contamination is an estimate of the percentage of outliers in dataset
    model = IsolationForest(contamination=contamination_percent, random_state=42, n_estimators=100)
    
    # -1 is outlier, 1 is normal
    df['Anomaly_Flag'] = model.fit_predict(X_scaled)
    
    # Decision function gives negative scores for outliers. We invert and normalize to 0-100 scale.
    # Higher score = Higher risk.
    raw_scores = model.decision_function(X_scaled)
    normalized_scores = ((raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min())) * 100
    df['Risk_Score'] = normalized_scores

    # Labeling for UI
    df['Status'] = df['Anomaly_Flag'].apply(lambda x: 'High Risk Flag' if x == -1 else 'Normal')
    
    return df

# ============================================
# STEP 4: STREAMLIT DASHBOARD UI/UX
# ============================================

st.set_page_config(page_title="Forensic Accounting AI", layout="wide", page_icon="🕵️‍♂️")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.write("Project: Financial Statement Analysis & Forensic Auditing")
    st.markdown("---")
    
    # Sensitivity Check (Robustness)
    st.subheader("🤖 AI Sensitivity Check")
    contamination = st.slider("Anomaly Threshold (%)", min_value=1, max_value=30, value=10, step=1,
                              help="Increasing this makes the AI more aggressive in flagging outliers.") / 100

    st.markdown("---")
    st.info("This tool uses an Unsupervised Isolation Forest algorithm to detect unusual patterns in accounting estimates derived from synthetic annual data.")


# --- Data Loading & Processing ---
raw_data = generate_synthetic_data()
processed_data = calculate_forensic_features(raw_data)
scored_data = run_anomaly_detection(processed_data, contamination_percent=contamination)

# Get latest year for summary views
latest_year = scored_data['Year'].max()
latest_data = scored_data[scored_data['Year'] == latest_year]

# --- Main Page Structure (Tabs) ---
tab1, tab2 = st.tabs(["🚁 Executive Overview", "🔍 Deep Dive Analysis"])

# === TAB 1: Landing Page / Overview ===
with tab1:
    st.title("🕵️‍♂️ Provisioning Manipulation Detection Dashboard")
    st.markdown("### AI-Driven Identification of Accounting Estimate Anomalies")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Companies Analyzed", value=len(latest_data))
    with col2:
        high_risk_count = len(latest_data[latest_data['Status'] == 'High Risk Flag'])
        st.metric(label="High Risk Flags (Latest Year)", value=high_risk_count, delta=f"{contamination*100:.0f}% Threshold Target", delta_color="inverse")
    with col3:
        avg_risk = latest_data['Risk_Score'].mean()
        st.metric(label="Average Industry Risk Score", value=f"{avg_risk:.1f} / 100")

    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader(f"🚨 Top Risk Alerts ({latest_year})")
        st.write("Companies displaying the highest statistical divergence in accounting estimates.")
        
        # Display top 10 riskiest companies
        top_risks = latest_data.sort_values(by='Risk_Score', ascending=False).head(10)
        
        # Styling the dataframe for visual impact
# Display top 10 riskiest companies (Fixed Version)
        top_risks = latest_data.sort_values(by='Risk_Score', ascending=False).head(10)
        
        st.dataframe(
            top_risks[['Ticker', 'Risk_Score', 'Status', 'ADA_Ratio', 'Depreciation_Rate']],
            column_config={
                "ADA_Ratio": st.column_config.NumberColumn(label="ADA Ratio", format="%.2f%%"),
                "Depreciation_Rate": st.column_config.NumberColumn(label="Depr. Rate", format="%.2f%%"),
                "Risk_Score": st.column_config.NumberColumn(label="Risk Score", format="%.1f"),
            },
            use_container_width=True
        )
    with col_right:
        st.subheader="Peer Comparison: Risk vs. Profitability"
        # Scatter plot to show outliers
        fig_scatter = px.scatter(
            latest_data,
            x="NI_Margin",
            y="ADA_Ratio",
            color="Status",
            size="Risk_Score",
            hover_data=['Ticker'],
            color_discrete_map={"Normal": "grey", "High Risk Flag": "red"},
            title=f"Peer Benchmarking ({latest_year})",
            labels={"NI_Margin": "Net Income Margin", "ADA_Ratio": "ADA / Gross Receivables"}
        )
        fig_scatter.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Note: High risk companies often appear in corners of the distribution plot.")


# === TAB 2: Deep Dive Analysis ===
with tab2:
    st.title("🔍 Individual Company Forensic Audit")
    
    # Selecting a company
    company_list = scored_data['Ticker'].unique()
    selected_ticker = st.selectbox("Select a Ticker for detailed review:", company_list)
    
    # Filtering data for that company
    company_data = scored_data[scored_data['Ticker'] == selected_ticker].sort_values(by='Year')
    current_risk = company_data.iloc[-1]['Risk_Score']
    current_status = company_data.iloc[-1]['Status']

    # -- Header Section --
    hl1, hl2 = st.columns([1, 3])
    with hl1:
        # Gauge Chart for Risk
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_risk,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Current Risk Score ({current_status})"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "red" if current_risk > 70 else "orange" if current_risk > 40 else "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#e6ffe6'},
                    {'range': [40, 70], 'color': '#ffebcc'},
                    {'range': [70, 100], 'color': '#ffcccc'}],
            }))
        fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with hl2:
        st.subheader("Red Flag Analysis Summary")
        
        # Simple logic to generate textual analysis based on the data
        latest_ada = company_data.iloc[-1]['ADA_Ratio']
        first_ada = company_data.iloc[0]['ADA_Ratio']
        ada_trend = "decreasing" if latest_ada < first_ada * 0.8 else "stable"
        
        latest_dep = company_data.iloc[-1]['Depreciation_Rate']
        first_dep = company_data.iloc[0]['Depreciation_Rate']
        dep_trend = "decreasing significantly" if latest_dep < first_dep * 0.7 else "stable"

        analysis_text = f"""
        Based on the multi-year analysis of {selected_ticker}:
        * The AI model has flagged this company as **{current_status}** based on peer deviation.
        * **Provisions for Doubtful Accounts (ADA):** The ratio of ADA to Gross Receivables is **{ada_trend}** over the 5-year period. { "This is a classic earnings management warning sign if revenue is growing." if ada_trend == "decreasing" else ""}
        * **Depreciation Estimates:** The effective depreciation rate appears to be **{dep_trend}**. {"A sudden drop often indicates management has extended asset useful lives to boost short-term income." if dep_trend != "stable" else ""}
        """
        st.info(analysis_text)

    st.markdown("---")

    # -- Visuals Section: The "Jaws of Death" --
    st.subheader("📉 Divergence Analysis (The 'Jaws' Chart)")
    st.write("Look for diverging lines. If Revenue (Activity) is going up, but Provisions (Estimates) are flat or going down, it indicates aggressive accounting.")
    
    # Normalize data to start at 100 for comparison
    base_year_data = company_data.iloc[0]
    norm_data = company_data.copy()
    norm_data['Norm_Revenue'] = (norm_data['Revenue'] / base_year_data['Revenue']) * 100
    norm_data['Norm_ADA'] = (norm_data['ADA'] / base_year_data['ADA']) * 100
    norm_data['Norm_Depreciation'] = (norm_data['Depreciation_Exp'] / base_year_data['Depreciation_Exp']) * 100

    # Melt for charting
    melted_norm = norm_data.melt('Year', value_vars=['Norm_Revenue', 'Norm_ADA', 'Norm_Depreciation'], var_name='Metric', value_name='Indexed Value (Base=100)')
    
    fig_jaws = px.line(
        melted_norm, 
        x='Year', 
        y='Indexed Value (Base=100)', 
        color='Metric',
        markers=True,
        color_discrete_map={
            'Norm_Revenue': 'blue',
            'Norm_ADA': 'red',
            'Norm_Depreciation': 'orange'
        },
        title=f"Revenue vs. Accounting Estimates Index ({selected_ticker})"
    )
    fig_jaws.update_layout(hovermode="x unified", plot_bgcolor="white")
    fig_jaws.add_vrect(x0=2021.5, x1=2023.5, annotation_text="High Divergence Zone", annotation_position="top left",
                       fillcolor="red", opacity=0.1, line_width=0)
    st.plotly_chart(fig_jaws, use_container_width=True)
    
    # -- Raw Data View --
    with st.expander("View Raw Financial Data for Selected Company"):
        st.dataframe(company_data.style.format("{:.2f}", subset=['Revenue', 'Gross_Receivables', 'ADA', 'Gross_PPE', 'Depreciation_Exp']))

# Run with: streamlit run app.py
