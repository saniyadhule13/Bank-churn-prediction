import streamlit as st
import numpy as np
import pickle

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ── Load Model ─────────────────────────────────────────
model = pickle.load(open("churn_model.pkl", "rb"))

# ── Custom CSS for Attractive UI ───────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
}
.card {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────
st.markdown('<h1 class="main-title">🏦 Bank Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown("---")

# ── Layout ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Customer Info")
    credit = st.slider("Credit Score", 300, 900, 600)
    age = st.slider("Age", 18, 90, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    geo = st.selectbox("Geography", ["Germany", "Spain"])

with col2:
    st.markdown("### 💳 Account Info")
    balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
    products = st.selectbox("Number of Products", [1, 2, 3, 4])
    tenure = st.slider("Tenure (Years)", 0, 10, 3)

with col3:
    st.markdown("### 📊 Activity Info")
    card = st.selectbox("Has Credit Card", [0, 1])
    active = st.selectbox("Is Active Member", [0, 1])
    salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)

st.markdown("---")

# ── Encoding ───────────────────────────────────────────
gender = 1 if gender == "Male" else 0
geo_ge = 1 if geo == "Germany" else 0
geo_sp = 1 if geo == "Spain" else 0

# ── Predict Button ─────────────────────────────────────
predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

if predict_btn:

    data = np.array([[credit, geo_ge, geo_sp, gender, age, tenure,
                      balance, products, card, active, salary]])

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to Churn")
        else:
            st.success("✅ Customer will Stay with Bank")

    with colB:
        churn_prob = probability[1] * 100
        stay_prob = probability[0] * 100
        st.metric("Churn Probability", f"{churn_prob:.2f}%")
        st.metric("Stay Probability", f"{stay_prob:.2f}%")

    st.markdown("---")
    st.info("💡 This prediction is based on customer financial and activity behavior.")
