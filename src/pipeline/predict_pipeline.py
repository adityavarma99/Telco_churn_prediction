import os
import pickle
import pandas as pd
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self, input_data_path=None, output_file_path=None):
        """
        Initialize paths for artifacts and optionally for input/output files.

        Args:
            input_data_path (str, optional): Path to the input data file.
            output_file_path (str, optional): Path to save the output predictions.
        """
        self.model_path = os.path.join("artifacts", "best_model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.selected_features_path = os.path.join("artifacts", "selected_features.pkl")

        # Initialize input and output paths if provided
        self.input_data_path = input_data_path
        self.output_file_path = output_file_path

        # Load the artifacts
        self.model = self._load_artifact(self.model_path)
        self.preprocessor = self._load_artifact(self.preprocessor_path)
        self.selected_features = self._load_artifact(self.selected_features_path)

        # Optionally load input data if path is provided
        if self.input_data_path:
            self.data = self._load_input_data(self.input_data_path)
        else:
            self.data = None

    def _load_artifact(self, file_path):
        """
        Loads a Python object from a file using pickle.
        """
        try:
            obj = load_object(file_path)
            logging.info(f"Artifact loaded successfully from {file_path}.")
            return obj
        except Exception as e:
            logging.error(f"Error loading artifact from {file_path}: {e}")
            raise e

    def _load_input_data(self, input_data_path):
        """
        Loads the input data (assuming it's in CSV format).

        Args:
            input_data_path (str): Path to the input data file.

        Returns:
            pd.DataFrame: Loaded input data.
        """
        try:
            data = pd.read_csv(input_data_path)
            logging.info(f"Input data loaded successfully from {input_data_path}.")
            return data
        except Exception as e:
            logging.error(f"Error loading input data from {input_data_path}: {e}")
            raise e

    def predict(self, data=None):
        """
        Predicts the output based on input data.

        Args:
            data (pd.DataFrame, optional): Input data as a Pandas DataFrame. If None, uses the loaded data.

        Returns:
            np.ndarray: Predictions.
        """
        try:
            logging.info("Starting the prediction process...")

            # Use data passed in or the data loaded earlier
            data_to_predict = data if data is not None else self.data

            # Step 1: Preprocess the input data
            logging.info("Applying preprocessing on input data...")
            transformed_data = self.preprocessor.transform(data_to_predict)

            # Handle sparse matrix, if applicable
            if hasattr(transformed_data, "toarray"):
                transformed_data = transformed_data.toarray()
                logging.info("Converted sparse matrix to dense array.")

            # Step 2: Create DataFrame with Transformed Features
            logging.info("Creating DataFrame with transformed feature names...")
            transformed_columns = self.preprocessor.get_feature_names_out()
            transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)

            # Step 3: Align Simplified Features
            logging.info("Aligning data with selected features...")
            aligned_data = transformed_df.loc[:, [
                col for col in self.selected_features if col in transformed_df.columns
            ]]

            # Check for missing features
            missing_features = [
                col for col in self.selected_features if col not in transformed_df.columns
            ]

            if missing_features:
                logging.warning(f"Missing features in transformed DataFrame: {missing_features}")
                raise ValueError(f"Missing features: {missing_features}")

            # Step 4: Generate Predictions
            logging.info("Generating predictions...")
            predictions = self.model.predict(aligned_data)

            # Save predictions if output path is provided
            if self.output_file_path:
                self._save_predictions(predictions)

            return predictions

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e

    def _save_predictions(self, predictions):
        """
        Save the predictions to the specified file.

        Args:
            predictions (np.ndarray): Predictions to be saved.
        """
        try:
            predictions_df = pd.DataFrame(predictions, columns=["Prediction"])
            predictions_df.to_csv(self.output_file_path, index=False)
            logging.info(f"Predictions saved to {self.output_file_path}.")
        except Exception as e:
            logging.error(f"Error saving predictions to {self.output_file_path}: {e}")
            raise e


# Example of the `load_object` utility function:
def load_object(file_path):
    """
    Utility function to load a Python object from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        Object loaded from the pickle file.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)
