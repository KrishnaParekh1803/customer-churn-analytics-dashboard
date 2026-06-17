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

st.markdown("""
            <style>
            .kpi-card{
            height: 180px;
            min-height: 180px;
            font-size: 2rem;
            background: linear-gradient(135deg,#1E293B,#334155);
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
            min-height:140px;
            display:flex;
            flex-direction:column;
            justify-content:center;
}
            </style>
            """, unsafe_allow_html=True
)
st.set_page_config(
    page_title="Model Intelligence",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# Load Data
# -------------------------

shap_df = pd.read_csv(
    "artifacts/shap_feature_importance.csv"
)

lr_df = pd.read_csv(
    "artifacts/logistic_regression_feature_importance.csv"
)

data_df = pd.read_csv(
    "artifacts/feature_engineered_data.csv"
)

# -------------------------
# Header
# -------------------------

st.markdown("""
# 🧠 Model Intelligence Hub

### Explainable AI & Churn Prediction Insights
            
"""
            )

st.markdown("---")

# -------------------------
# KPI Cards
# -------------------------

col1,col2,col3,col4,col5 = st.columns(5)

with col1:
    st.markdown(
    f"""
    <div class="kpi-card">
        <h4>Model</h4>
        <h2>Random Forest</h2>
    </div>
    """,
    unsafe_allow_html=True
)


with col2:
    st.markdown(
    f"""
    <div class="kpi-card">
        <h4>Customer</h4>
        <h2>{len(data_df)}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

with col3:
    st.markdown(
    f"""
    <div class="kpi-card">
        <h4>Accuracy</h4>
        <h2>79.4%</h2>
    </div>
    """,
    unsafe_allow_html=True
)


with col4:
    st.markdown(
    f"""
    <div class="kpi-card">
        <h4>Churn Rate</h4>
        <h2>{data_df['Churn'].mean()*100:.2f}%</h2>
    </div>
    """,
    unsafe_allow_html=True
)

with col5:
    st.markdown(
    f"""
    <div class="kpi-card">
        <h4>Business Features</h4>
        <h2>{data_df.shape[1]-1}</h2>
    </div>
    """,
    unsafe_allow_html=True
    )
    #data_df.shape[1]-1
        

top_shap = (
    shap_df
    .sort_values(shap_df.columns[1], ascending=False)
    .head(10)
    .sort_values(shap_df.columns[1])
)
fig_shap = px.bar(
    top_shap,
    x=top_shap.columns[1],
    y=top_shap.columns[0],
    orientation="h",
    title="Top Customer Churn Drivers"
)
    

st.markdown("---")

# -------------------------
# SHAP Importance
# -------------------------

st.subheader("🔥 Top Churn Drivers (SHAP)")

feature_map = {
    "cat__TenureGroup_Regular": "Regular Customers",
    "cat__TenureGroup_Very Loyal": "Very Loyal Customers",
    "cat__InternetService_DSL": "DSL Internet",
    "cat__InternetService_Fiber optic": "Fiber Internet",
    "cat__PaperlessBilling_No": "No Paperless Billing",
    "cat__MultipleLines_No": "No Multiple Lines",
    "num__ContractRiskScore": "Contract Risk Score",
    "num__EngagementScore": "Engagement Score",
    "num__ServiceCount": "Services Used",
    "num__tenure": "Customer Tenure",
    "num__CLV": "Customer Lifetime Value",
    "num__EngagementScore":"Engagement Score",
    "num__ContractRiskScore":"Contract Risk Score",
    "cat__StreamingTV_Yes":"Streaming TV",
    "cat__StreamingMovies_Yes":"Streaming Movies",
    "cat__OnlineSecurity_No":"No Online Security",
    "cat__TechSupport_No":"No Tech Support",
    "cat__PaymentMethod_Electronic check":"Electronic Check",
}
feature_map.update({
    "cat__PaymentMethod_Electronic check":"Electronic Check Users",
    "cat__TechSupport_No":"No Technical Support",
    "cat__OnlineSecurity_No":"No Online Security",
    "cat__StreamingTV_Yes":"Streaming TV Users",
    "cat__StreamingMovies_Yes":"Streaming Movie Users",
    "cat_TenureGroupe_nan": "Customer Tenure"
})
for df in (shap_df, lr_df):
    df["Feature"] = (
        df["Feature"]
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.replace("nan", "Unknown", regex=False)
        .str.replace("num", "", regex=False)
    )
    df["Feature"] = df["Feature"].replace({
        "TenureGroup Unknown": "Customer Tenure Group",
        "Fiber Internet": "Fiber Internet Service",
        "DSL Internet": "DSL Internet Service",
        "ContractRiskScore": "Contract Risk Score",
        "EngagementScore": "Customer Engagement"
    })




shap_df["Feature"] = shap_df["Feature"].replace(feature_map)
lr_df["Feature"] = lr_df["Feature"].replace(feature_map)

fig_shap = px.bar(
    shap_df.head(10).sort_values(
        shap_df.columns[1]
    ),
    x=shap_df.columns[1],
    y=shap_df.columns[0],
    orientation="h",
    title="Top Customer Churn Drivers"
)

st.plotly_chart(
    fig_shap,
    use_container_width=True,
    config={"displayModeBar": False}
)
st.info("""
Customer Tenure and Engagement Score are the
strongest churn indicators. Customers with
short tenure and low engagement show
significantly higher churn probability.
""")
# -------------------------
# Logistic Regression
# -------------------------

st.subheader("📊 Logistic Regression Feature Importance")

fig_lr = px.bar(
    lr_df.head(10).sort_values(
        lr_df.columns[1]
    ),
    x=lr_df.columns[1],
    y=lr_df.columns[0],
    orientation="h",
    title="Most Important Features"
)
feature_map.update({
    "cat__PaymentMethod_Electronic check":"Electronic Check Users",
    "cat__TechSupport_No":"No Technical Support",
    "cat__OnlineSecurity_No":"No Online Security",
    "cat__StreamingTV_Yes":"Streaming TV Users",
    "cat__StreamingMovies_Yes":"Streaming Movie Users"
})

st.plotly_chart(
    fig_lr,
    use_container_width=True,
    config={"displayModeBar": False}
)


# -------------------------
# Business Insights
# -------------------------

st.subheader("💡 Business Recommendations")

st.success(f"""
### Priority Actions

1. Retain low-tenure customers (<12 months)

2. Promote annual contracts

3. Increase engagement score through loyalty programs

4. Bundle internet and entertainment services

5. Target electronic-check customers with incentives
""")

# -------------------------
# Project Summary
# -------------------------

st.subheader("📋 Project Summary")

st.info("""
Dataset: IBM Telco Customer Churn

Algorithm: Random Forest

Feature Engineering:
• CLV
• TenureGroup
• ContractRiskScore
• ServiceCount
• EngagementScore

Deployment:
• Streamlit
• Scikit-Learn
• Plotly
""")