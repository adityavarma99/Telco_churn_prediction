import pickle
import pandas as pd
from src.logger import logging

def predict(input_data_path, output_file_path):
    """
    Uses the trained model and preprocessor to make predictions on new data.
    """
    try:
        logging.info("Starting the prediction pipeline...")

        # Step 1: Load Simplified Selected Features
        logging.info("Loading simplified selected features...")
        with open("artifacts/selected_features.pkl", "rb") as file:
            simplified_selected_features = pickle.load(file)
        logging.info(f"Simplified selected features loaded: {simplified_selected_features}")

        # Step 2: Load Preprocessor
        logging.info("Loading the preprocessor...")
        with open("artifacts/preprocessor.pkl", "rb") as file:
            preprocessor = pickle.load(file)
        logging.info("Preprocessor loaded successfully.")

        # Step 3: Load Model
        logging.info("Loading the trained model...")
        with open("artifacts/best_model.pkl", "rb") as file:
            model = pickle.load(file)
        logging.info("Trained model loaded successfully.")

        # Step 4: Read Input Data
        logging.info("Reading input data...")
        data = pd.read_csv(input_data_path)
        logging.info(f"Input data loaded with shape: {data.shape}")

        # Drop unnecessary columns if present (e.g., ID, target variable)
        unnecessary_columns = ["customerID", "Churn"]
        data = data.drop(columns=[col for col in unnecessary_columns if col in data.columns], errors='ignore')

        # Step 5: Apply Preprocessing
        logging.info("Applying preprocessing on input data...")
        transformed_data = preprocessor.transform(data)

        # Handle sparse matrix, if applicable
        if hasattr(transformed_data, "toarray"):
            transformed_data = transformed_data.toarray()
            logging.info("Converted sparse matrix to dense array.")

        # Step 6: Create DataFrame with Transformed Features
        logging.info("Creating DataFrame with transformed feature names...")
        transformed_columns = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)

        # Step 7: Map Simplified Feature Names
        logging.info("Mapping simplified feature names to transformed data...")
        aligned_data = transformed_df[simplified_selected_features]
        logging.info(f"Aligned data shape: {aligned_data.shape}")

        # Step 8: Make Predictions
        logging.info("Generating predictions...")
        predictions = model.predict(aligned_data)

        # Step 9: Save Predictions
        logging.info("Saving predictions to output file...")
        pd.DataFrame({"Predictions": predictions}).to_csv(output_file_path, index=False)
        logging.info(f"Predictions saved to {output_file_path} successfully.")

    except Exception as e:
        logging.error(f"Error in prediction pipeline: {e}")
        raise e
