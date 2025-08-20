import streamlit as st
import pandas as pd
import joblib
import pickle

# Load model and column information
@st.cache_resource
def load_model():
    model = joblib.load("churn_model_pipeline.pkl")
    with open('model_columns.pkl', 'rb') as f:
        column_info = pickle.load(f)
    return model, column_info

# Streamlit App setup
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📞 Telco Customer Churn Prediction App")
st.markdown("Fill out the customer details below to predict whether they'll churn.")

try:
    model, column_info = load_model()
    
    # Input form
    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (in months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
        with col2:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=1500.0)
        
        submit_btn = st.form_submit_button("Predict Churn")

    # Prediction section
    if submit_btn:
        # Create input data with the exact same structure as training data
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,  # Keep as "Yes" or "No" to match training
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])
        
        # Use the pipeline for prediction which handles all preprocessing
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        # Display results
        st.markdown("## Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ The customer is likely to churn. (Confidence: {prediction_proba:.2%})")
            
            # Show risk factors
            st.markdown("### Risk Factors:")
            risk_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append("- Month-to-month contract (higher churn risk)")
            if tenure < 12:
                risk_factors.append("- Short tenure (less than 12 months)")
            if internet_service == "Fiber optic" and (online_security == "No" or tech_support == "No"):
                risk_factors.append("- Fiber optic without security features")
            if payment_method == "Electronic check":
                risk_factors.append("- Payment via electronic check")
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(factor)
            
            # Retention suggestions
            st.markdown("### Retention Suggestions:")
            st.markdown("- Offer a contract upgrade with discounted rates")
            st.markdown("- Provide complementary tech support or security features")
            st.markdown("- Consider a loyalty discount program")
        else:
            st.success(f"✅ The customer is likely to stay. (Confidence: {1 - prediction_proba:.2%})")
            
            # Show positive factors
            st.markdown("### Positive Factors:")
            positive_factors = []
            
            if contract in ["One year", "Two year"]:
                positive_factors.append("- Long-term contract")
            if tenure > 24:
                positive_factors.append("- Long tenure (over 24 months)")
            if tech_support == "Yes" and online_security == "Yes":
                positive_factors.append("- Using security and support services")
            
            if positive_factors:
                for factor in positive_factors:
                    st.markdown(factor)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.markdown("""
    ### Troubleshooting:
    1. Make sure you've run the model training script first
    2. Check that 'churn_model_pipeline.pkl' and 'model_columns.pkl' files exist
    3. Ensure that the same Python environment is used for both training and this app
    """)