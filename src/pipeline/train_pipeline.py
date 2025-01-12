from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

import os
import sys
import pickle

def start_pipeline(file_path: str):
    try:
        logging.info("Starting the training pipeline.")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion(file_path=file_path)
        logging.info(f"Data ingestion completed. Train file: {train_data}, Test file: {test_data}")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        X_train_resampled, y_train_resampled, X_test_selected, y_test, selected_features = data_transformation.initiate_data_transformation(
            train_data, test_data, target_column="Churn"
        )
        logging.info(f"Data transformation completed. Selected features: {selected_features}")

        # Save the selected features to a pickle file
        try:
            artifacts_dir = "artifacts"
            os.makedirs(artifacts_dir, exist_ok=True)
            features_file_path = os.path.join(artifacts_dir, "selected_features.pkl")
            with open(features_file_path, 'wb') as file:
                pickle.dump(selected_features, file)
            logging.info(f"Selected features saved to {features_file_path}.")
        except Exception as e:
            logging.error(f"Error saving selected features: {e}")
            raise CustomException(e, sys)

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        best_model, best_score = model_trainer.train_and_evaluate(
            X_train_resampled, y_train_resampled, X_test_selected, y_test, selected_features
        )
        logging.info(f"Model training completed successfully. Best Score: {best_score}")

    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise CustomException(e, sys)
