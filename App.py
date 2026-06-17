import streamlit as st

st.markdown("""
<style>
a[href^="#"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)




st.title("📊 Customer Churn Analytics Platform")

st.markdown("""
### AI-Powered Customer Retention System and Business Analytics Hub
            
Predict customer churn using machine learning,
analyze business performance, explore model
explainability and estimate revenue at risk.

This platform provides:

- Customer churn prediction
- Business analytics
- Model explainability
- Customer risk assessment

""")

kpi1, kpi2, kpi3, kpi4= st.columns(4)

with kpi1:
    st.info("🤖 Model\n\nRandom Forest")

with kpi2:
    st.info("📊 Dataset\n\nIBM Telco")

with kpi3:
    st.info("🎯 Model Accuracy\n\n79.4%")

with kpi4:
    st.info("👥 Customers\n\n7,043")

st.info(
    "Navigate using the sidebar to access Prediction, Analytics and Model Insights."
)
#st.markdown("---")

st.caption(
    "Customer Churn Analytics Platform | Built using Machine Learning, Streamlit and Explainable AI"
)


