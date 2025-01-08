import pickle
import pandas as pd
import logging

def predict(input_data_path, output_file_path):
    """
    Uses the trained model and preprocessor to make predictions on new data.
    """
    try:
        logging.info("Starting the prediction pipeline...")

        # Step 1: Load Preprocessor
        logging.info("Loading the preprocessor...")
        with open("artifacts/preprocessor.pkl", "rb") as file:
            preprocessor = pickle.load(file)
        logging.info("Preprocessor loaded successfully.")

        # Step 2: Load Model
        logging.info("Loading the trained model...")
        with open("artifacts/best_model.pkl", "rb") as file:
            model = pickle.load(file)
        logging.info("Trained model loaded successfully.")

        # Step 3: Read Input Data
        logging.info("Reading input data...")
        data = pd.read_csv(input_data_path)
        logging.info(f"Input data loaded with shape: {data.shape}")

        # Step 4: Apply Preprocessing
        logging.info("Applying preprocessing on input data...")
        transformed_data = preprocessor.transform(data)
        logging.info("Preprocessing completed.")

        # Step 5: Make Predictions
        logging.info("Generating predictions...")
        predictions = model.predict(transformed_data)

        # Step 6: Save Predictions
        logging.info("Saving predictions to output file...")
        pd.DataFrame({"Predictions": predictions}).to_csv(output_file_path, index=False)
        logging.info(f"Predictions saved to {output_file_path} successfully.")

    except Exception as e:
        logging.error(f"Error in prediction pipeline: {e}")
        raise e
