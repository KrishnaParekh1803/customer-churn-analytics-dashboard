import streamlit as st
import pandas as pd
import plotly.express as px

st.markdown("""
<style>
.stHeading a {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide")

# ------------------------
# Custom CSS
# ------------------------

st.markdown("""
<style>

.main {
    background-color: #0F172A;
}

.metric-card {
    background: linear-gradient(135deg,#1E293B,#334155);
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,0.3);
}

.metric-value {
    font-size:32px;
    font-weight:bold;
    color:#38BDF8;
}

.metric-title {
    font-size:16px;
    color:white;
}

.big-title {
    font-size:50px;
    font-weight:60px;
    background: linear-gradient(to right,#38BDF8,#8B5CF6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
            margin-bottom:0;
            }

.subtitle {
    color:#CBD5E1;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------
# Load Data
# ------------------------

df = pd.read_csv("artifacts/feature_engineered_data.csv")

# ------------------------
# Header
# ------------------------

st.markdown(
    '<p class="big-title">Customer Churn Intelligence Hub</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-Powered Customer Retention Analytics Platform</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------
# KPIs
# ------------------------
st.caption(
    "Currency: USD (IBM Telco Dataset)"
)
total_customers = len(df)

churn_rate = (df["Churn"] == 1).mean() * 100



avg_monthly = df["MonthlyCharges"].mean()

avg_clv = df["CLV"].mean()
retained = (df["Churn"]==0).sum()

c1,c2,c3,c4,c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">👥 Customers</div>
        <div class="metric-value">{total_customers:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">📉 Churn Rate</div>
        <div class="metric-value">{churn_rate:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        
 
        f"""
        <div class="metric-card">
        <div class="metric-title">💰 Avg Charge</div>
        <div class="metric-value">{avg_monthly:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">⭐ Avg CLV</div>
        <div class="metric-value">
        ${avg_clv:.0f}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c5:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">✅ Retained</div>
        <div class="metric-value">{retained:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------
# Charts Row 1
# ------------------------

left,right = st.columns(2)

with left:
    # create readable churn label
    df["ChurnLabel"] = df["Churn"].map({
        0: "Retained",
        1: "Churned"
    })

    churn_fig = px.pie(
    df,
    names="ChurnLabel",
    title="Customer Retention vs Churn",
    color="ChurnLabel",
    color_discrete_map={
        "Retained":"#22C55E",
        "Churned":"#EF4444"
    }
)

    st.plotly_chart(churn_fig, use_container_width=True)

with right:

    contract_fig = px.histogram(
    df,
    x="Contract",
    color="Contract",
    title="Contract Distribution"
)

    st.plotly_chart(
        contract_fig,
        use_container_width=True
    )

# ------------------------
# Charts Row 2
# ------------------------

left,right = st.columns(2)

with left:

    tenure_fig = px.histogram(
        df,
        x="TenureGroup",
        title="Customer Segments"
    )

    st.plotly_chart(
        tenure_fig,
        use_container_width=True
    )

with right:

    revenue_fig = px.histogram(
        df,
        x="MonthlyCharges",
        title="Revenue Distribution"
    )

    st.plotly_chart(
        revenue_fig,
        use_container_width=True
    )