import streamlit as st
import pandas as pd
import pickle
from src.pipeline.predict_pipeline import PredictPipeline

# Title of the app
st.title("Customer Churn Prediction")

# Form to input customer details
with st.form("customer_input"):
    st.header("Enter Customer Details")

    # Features captured in the form
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, step=1)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
    total_charges = st.number_input("Total Charges", min_value=0.0, step=0.01)

    # Submit button for form
    submit_button = st.form_submit_button("Predict")

# If the form is submitted
if submit_button:
    try:
        # Load selected features from the saved pkl file
        selected_features_path = 'artifacts/selected_features.pkl'
        with open(selected_features_path, 'rb') as f:
            selected_features = pickle.load(f)

        # Initialize input data with form values
        input_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
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
        }

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Reorder columns to match selected_features
        input_df = input_df.reindex(columns=selected_features, fill_value=None)

        # Display the DataFrame before prediction (optional for debugging)
        st.write("Final input DataFrame for prediction:", input_df)

        # Create an instance of the PredictPipeline and predict
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(input_df)

        # Show prediction result
        st.subheader("Prediction Result:")
        st.write(f"Churn Prediction: {result[0]}")

    except FileNotFoundError:
        st.error("Error: The file containing selected features was not found.")
    except KeyError as e:
        st.error(f"Error: Missing feature in input data - {str(e)}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
