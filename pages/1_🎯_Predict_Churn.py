import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

st.markdown("""
<style>
a[href^="#"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# PAGE CONFIG
# ------------------------

st.set_page_config(
    page_title="Churn Prediction Center",
    page_icon="🎯",
    layout="wide"

)

# ------------------------
# LOAD MODEL
# ------------------------

model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

MEDIAN_CLV = 1393.6

# ------------------------
# CUSTOM CSS
# ------------------------

st.markdown("""
<style>

.main {
    background-color: #0F172A;
}

.title {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg,#38BDF8,#8B5CF6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    color:#CBD5E1;
    font-size:18px;
}

.card{
    background:#1E293B;
    padding:20px;
    border-radius:18px;
    border:1px solid #334155;
}

.metric-card{
    background:linear-gradient(135deg,#1E293B,#334155);
    padding:20px;
    border-radius:18px;
    text-align:center;
}

.big-number{
    font-size:48px;
    font-weight:800;
}

.low{
    color:#10B981;
}

.medium{
    color:#F59E0B;
}

.high{
    color:#EF4444;
}

</style>
""", unsafe_allow_html=True)

# ------------------------
# HEADER
# ------------------------

st.markdown(
    '<p class="title">🎯 Churn Prediction Center</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">AI-Powered Customer Risk Assessment</p>',
    unsafe_allow_html=True,
)

st.markdown("---")
left,right = st.columns([1.4,1])

with left:

    st.markdown("## 👤 Customer Profile")

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    SeniorCitizen_display = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

    SeniorCitizen = (
    1 if SeniorCitizen_display == "Yes"
    else 0
)

    Partner = st.selectbox(
        "Partner",
        ["Yes","No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes","No"]
    )

    tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12,
    step=1
)

    service_col1, service_col2 = st.columns(2)

    with service_col1:
        PhoneService = st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )
        MultipleLines = st.selectbox(
            "Multiple Lines",
        [
            "No phone service",
            "No",
            "Yes"
        ]
    )
        InternetService = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )
        OnlineSecurity = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    with service_col2:
        OnlineBackup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )
        DeviceProtection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )
        TechSupport = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )
        StreamingTV = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )
        StreamingMovies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    st.markdown("## 💳 Billing & Contract")

    Contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

    predict = st.button(
        "🚀 Analyze Customer",
        use_container_width=True
    )
with right:

    st.subheader("📊 Prediction Results")

    if predict:

        CLV = MonthlyCharges * tenure

        if tenure <= 12:
            TenureGroup = "New"
        elif tenure <= 24:
            TenureGroup = "Regular"
        elif tenure <= 48:
            TenureGroup = "Loyal"
        else:
            TenureGroup = "Very Loyal"

        HighValueCustomer = (
            1 if CLV > MEDIAN_CLV else 0
        )

        risk_mapping = {
            "Month-to-month":3,
            "One year":2,
            "Two year":1
        }

        ContractRiskScore = risk_mapping[Contract]

        services = [
            PhoneService,
            OnlineSecurity,
            OnlineBackup,
            DeviceProtection,
            TechSupport,
            StreamingTV,
            StreamingMovies
        ]

        ServiceCount = sum(
            x == "Yes"
            for x in services
        )

        EngagementScore = (
            ServiceCount +
            (tenure / 12)
        )

        input_df = pd.DataFrame({

            "gender":[gender],
            "SeniorCitizen":[SeniorCitizen],
            "Partner":[Partner],
            "Dependents":[Dependents],
            "tenure":[tenure],
            "PhoneService":[PhoneService],
            "MultipleLines":[MultipleLines],
            "InternetService":[InternetService],
            "OnlineSecurity":[OnlineSecurity],
            "OnlineBackup":[OnlineBackup],
            "DeviceProtection":[DeviceProtection],
            "TechSupport":[TechSupport],
            "StreamingTV":[StreamingTV],
            "StreamingMovies":[StreamingMovies],
            "Contract":[Contract],
            "PaperlessBilling":[PaperlessBilling],
            "PaymentMethod":[PaymentMethod],
            "MonthlyCharges":[MonthlyCharges],
            "TotalCharges":[TotalCharges],
            "CLV":[CLV],
            "TenureGroup":[TenureGroup],
            "HighValueCustomer":[HighValueCustomer],
            "ContractRiskScore":[ContractRiskScore],
            "ServiceCount":[ServiceCount],
            "EngagementScore":[EngagementScore]
        })

        X = preprocessor.transform(
            input_df
        )

        probability = model.predict_proba(X)[0][1]
        churn_prob = probability * 100
        if churn_prob < 30:
            st.success("🟢 Low Churn Risk")
        elif churn_prob < 60:
            st.warning("🟡 Medium Churn Risk")
        else:
            st.error("🔴 High Churn Risk")
        
        prediction = model.predict(X)[0]
        if prediction == 1:
            st.error("⚠️ Customer Likely To Churn")
        else:
            st.success("✅ Customer Likely To Stay")
            
        st.metric(
                "Churn Probability",
                f"{churn_prob:.1f}%"
)
        reasons = []
        if Partner == "Yes":
            reasons.append("Has partner")
        if Dependents == "Yes":
            reasons.append("Has dependents")
        if ServiceCount >= 2:
            reasons.append("Uses multiple services")
        if Contract != "Month-to-month":
            reasons.append("Stable contract type")
        if churn_prob >= 30:
            reasons.append("Moderate churn probability")

        st.subheader("Prediction Factors")
        for reason in reasons:
            st.markdown(f"• {reason}")     
 

        

        churn_prob = probability * 100

        if churn_prob < 30:

            persona = "🟢 Loyal Customer"
            color = "low"

            advice = [
                "Maintain engagement",
                "Offer loyalty rewards",
                "Continue current service"
            ]

        elif churn_prob < 60:

            persona = "🟡 Growth Customer"
            color = "medium"

            advice = [
                "Monitor customer activity",
                "Promote additional services",
                "Increase engagement"
            ]

        elif churn_prob < 80:

            persona = "🟠 At-Risk Customer"
            color = "medium"

            advice = [
                "Offer retention discount",
                "Review customer concerns",
                "Promote annual contract"
            ]

        else:

            persona = "🔴 Churn Candidate"
            color = "high"

            advice = [
                "Immediate intervention",
                "Loyalty incentive",
                "Dedicated support",
                "Personalized retention plan"
            ]
        if churn_prob < 30:
                risk_level = "🟢 LOW RISK"
                risk_color = "#00C853"

        elif churn_prob < 60:
                risk_level = "🟡 MEDIUM RISK"
                risk_color = "#FFD600"

        else:
                risk_level = "🔴 HIGH RISK"
                risk_color = "#FF1744"


        # ----------------------------
        # CHURN RISK GAUGE
        # ----------------------------

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                title={"text": "Churn Risk %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 60], "color": "yellow"},
                        {"range": [60, 100], "color": "salmon"}
                    ]
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------
        # CHURN PROBABILITY CARD
        # ----------------------------

        st.markdown(
            f"""
            <div class="metric-card">
                <div>Churn Probability</div>
                <div class="big-number {color}">
                    {churn_prob:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------
        # CUSTOMER PERSONA
        # ----------------------------

        st.markdown("### 👤 Customer Persona")
        st.success(persona)

        # ----------------------------
        # RECOMMENDED ACTIONS
        # ----------------------------

        st.markdown("### 💡 Recommended Actions")

        for item in advice:
            st.write(f"✓ {item}")

        # ----------------------------
        # REVENUE AT RISK
        # ----------------------------

        revenue_at_risk = (
            MonthlyCharges * 12
        ) * (churn_prob / 100)

        st.metric(
            "💰 Revenue At Risk",
            f"${revenue_at_risk:,.2f}"
        )

        # ----------------------------
        # RETENTION CALCULATOR
        # ----------------------------

        st.subheader("💰 Retention Revenue Calculator")

        similar_customers = st.number_input(
            "Number of Similar Customers",
            value=100
        )

        retention_rate = st.slider(
            "Retention Success Rate (%)",
            0,
            100,
            20
        )

        total_risk = revenue_at_risk * similar_customers

        saved_revenue = (
            total_risk * retention_rate / 100
        )

        st.metric(
            "Potential Revenue Loss",
            f"${total_risk:,.0f}"
        )

        st.metric(
            "Revenue Saved",
            f"${saved_revenue:,.0f}"
        )

        # ----------------------------
        # DOWNLOAD REPORT
        # ----------------------------

        report = {
            "Churn Probability": round(churn_prob, 2),
            "Risk Level": risk_level,
            "Customer Persona": persona,
            "Revenue At Risk": round(revenue_at_risk, 2),
            "CLV": round(CLV, 2),
            "Tenure Group": TenureGroup
        }

        report_json = json.dumps(
            report,
            indent=4
        )

        st.download_button(
            label="📄 Download Customer Report",
            data=report_json,
            file_name="customer_report.json",
            mime="application/json"
        )

        # ----------------------------
        # CUSTOMER INSIGHTS
        # ----------------------------

        st.markdown("### 📈 Customer Insights")

        st.write(f"**CLV:** {CLV:,.2f}")
        st.write(f"**Service Count:** {ServiceCount}")
        st.write(f"**Engagement Score:** {EngagementScore:.2f}")
        st.write(f"**Tenure Group:** {TenureGroup}")
        