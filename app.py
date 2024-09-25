import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="wide")


# Load the saved models
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("model.h5")
    with open("label_encode_gender.pkl", "rb") as file:
        label_encode_gender = pickle.load(file)
    with open("onehot_encode_geo.pkl", "rb") as file:
        onehot_encode_geo = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, label_encode_gender, onehot_encode_geo, scaler

model, label_encode_gender, onehot_encode_geo, scaler = load_models()

# Streamlit app
st.title("🔮 Customer Churn Prediction")
st.markdown("<p class='big-font'>Predict the probability of customer churn using ML model!</p>", unsafe_allow_html=True)

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics")
    geography = st.selectbox("📍 Country", onehot_encode_geo.categories_[0])
    gender = st.radio("👤 Gender", label_encode_gender.classes_)
    age = st.slider("🎂 Age", 18, 92, 30)
    
    st.subheader("Account Information")
    balance = st.number_input("💰 Account Balance", min_value=0.0, format="%.2f")
    credit_score = st.slider("📊 Credit Score", 350, 850, 600)
    
with col2:
    st.subheader("Customer Engagement")
    estimated_salary = st.slider("💼 Monthly Salary", 10, 200000, 50000, step=1000)
    tenure = st.slider("⏳ Credit Tenure (years)", 0, 10, 2)
    num_of_products = st.slider("🛍️ Number of Products", 1, 4, 1)
    has_cr_card = st.checkbox("💳 Has a Credit Card")
    is_active_member = st.checkbox("🏃‍♂️ Is an Active Member")

# Prediction button
if st.button(":red[Predict Churn Probability]👇🏻", key="predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encode_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [int(has_cr_card)],
        "IsActiveMember": [int(is_active_member)],
        "EstimatedSalary": [estimated_salary]
    })

    # One-hot encode "Geography"
    geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encode_geo.get_feature_names_out(["Geography"]))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Create a gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'cyan'},
                {'range': [50, 75], 'color': 'royalblue'},
                {'range': [75, 100], 'color': 'darkblue'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))

    fig.update_layout(height=300)

    # Display results
    st.plotly_chart(fig, use_container_width=True)

    if prediction_proba > 0.5:
        st.error("⚠️ Customer is likely to churn!")
        st.markdown("<p class='medium-font'>Consider implementing retention strategies.</p>", unsafe_allow_html=True)
    else:
        st.success("✅ Customer is unlikely to churn.")
        st.markdown("<p class='medium-font'>Keep up the good work!</p>", unsafe_allow_html=True)


# Add an explanation section
with st.expander("How does this prediction work?"):
    st.write("""
    This customer churn prediction tool uses a machine learning model trained on historical customer data. 
    It considers various factors such as demographics, account information, and customer engagement to estimate 
    the likelihood of a customer leaving the service.

    The model uses the following key features:
    - Demographics: Country, Gender, Age
    - Account Information: Balance, Credit Score
    - Customer Engagement: Salary, Tenure, Number of Products, Credit Card Ownership, and Active Membership Status

    The prediction is based on patterns learned from past customer behavior. A higher churn probability suggests 
    that a customer with similar characteristics has a higher likelihood of leaving the service.
    """)

# Add a footer
st.markdown("---")
