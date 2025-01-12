import streamlit as st
import pickle
import pandas as pd

# File paths
model_file_path = 'artifacts/best_model.pkl'
features_file_path = 'artifacts/selected_features.pkl'

# Load selected features
try:
    with open(features_file_path, 'rb') as file:
        selected_features = pickle.load(file)
except Exception as e:
    st.error(f"Error loading selected features: {e}")
    selected_features = None

# Load model pipeline (preprocessing + trained model)
try:
    with open(model_file_path, 'rb') as file:
        model_pipeline = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    model_pipeline = None

# Streamlit app
def main():
    st.title("Telco Customer Churn Prediction")
    st.subheader("Enter customer details to predict churn probability:")

    if model_pipeline is not None and selected_features is not None:
        # Create input fields for features
        input_data = {}

        # Feature inputs
        if "Dependents" in selected_features:
            input_data["Dependents"] = st.selectbox("Dependents", ["Yes", "No"])
        if "tenure" in selected_features:
            input_data["tenure"] = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=0)
        if "OnlineSecurity" in selected_features:
            input_data["OnlineSecurity"] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        if "OnlineBackup" in selected_features:
            input_data["OnlineBackup"] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        if "DeviceProtection" in selected_features:
            input_data["DeviceProtection"] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        if "TechSupport" in selected_features:
            input_data["TechSupport"] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        if "Contract" in selected_features:
            input_data["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        if "PaperlessBilling" in selected_features:
            input_data["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"])
        if "MonthlyCharges" in selected_features:
            input_data["MonthlyCharges"] = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
        if "TotalCharges" in selected_features:
            input_data["TotalCharges"] = st.number_input("Total Charges", min_value=0.0, value=0.0)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Filter for selected features
        input_df = input_df[selected_features]

        if st.button("Predict"):
            try:
                # Use the model pipeline to preprocess and predict
                prediction = model_pipeline.predict(input_df)
                prediction_label = "Churn" if prediction[0] == 1 else "No Churn"
                st.success(f"Predicted outcome: {prediction_label}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.error("Ensure the model pipeline and selected features are loaded correctly.")

if __name__ == "__main__":
    main()
