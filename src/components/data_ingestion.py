import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

    def initiate_data_ingestion(self, file_path: str):
        logging.info("Entered the data ingestion component")
        try:
            logging.info("Starting data ingestion...")

            # Load dataset
            df = pd.read_csv(file_path)
            logging.info(f"Dataset loaded with shape: {df.shape}")

            # Create directories if not already present
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully.")

            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Split data into training and testing sets.")

            # Save train and test datasets
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logging.info("Data ingestion completed successfully.")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.pipeline.train_pipeline import start_pipeline
    from src.pipeline.predict_pipeline import PredictPipeline

    try:
        logging.info("Pipeline execution started.")

        # Step 1: Data Ingestion
        logging.info("Starting data ingestion.")
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion(
            file_path="notebook/data/Telco-Customer-Churn.csv"
        )
        logging.info(f"Data ingestion completed. Train file: {train_data}, Test file: {test_data}")

        # Step 2: Training Pipeline
        logging.info("Starting training pipeline.")
        start_pipeline(file_path="notebook/data/Telco-Customer-Churn.csv")
        logging.info("Training pipeline executed successfully.")

        # Step 3: Prediction Pipeline
        logging.info("Starting prediction pipeline.")
        input_data_path = "notebook/data/new_data.csv"  # Path to new data for prediction
        output_file_path = "notebook/data/predictions.csv"  # Path to save predictions
        predict_pipeline = PredictPipeline(input_data_path=input_data_path, output_file_path=output_file_path)
        logging.info("Prediction pipeline executed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise



'''
if __name__ == "__main__":
        try:
            logging.info("Pipeline execution started.")

            # Step 1: Data Ingestion
            obj = DataIngestion()
            train_data, test_data = obj.initiate_data_ingestion(file_path="notebook/data/Telco-Customer-Churn.csv")
            logging.info(f"Data ingestion completed. Train: {train_data}, Test: {test_data}")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        X_train_resampled, y_train_resampled, X_test_selected, y_test, selected_features = data_transformation.initiate_data_transformation(
            train_data, test_data, target_column="Churn"
        )
        logging.info(f"Data transformation completed. Simplified selected features: {selected_features}")

        # Step 3: Model Trainer
        modeltrainer = ModelTrainer()
        best_model, best_score = modeltrainer.train_and_evaluate(
            X_train_resampled, y_train_resampled, X_test_selected, y_test, selected_features
        )
        logging.info(f"Model training completed. Best Score: {best_score}")
'''



'''
        # Step 2: Training Pipeline
        if __name__ == "__main__":
            file_path = "notebook/data/Telco-Customer-Churn.csv"
            from src.pipeline.train_pipeline import start_pipeline
            start_pipeline(file_path)


        # Step 3: Prediction Pipeline
        input_data_path = "notebook/data/new_data.csv"
        output_file_path = "notebook/data/predictions.csv"
        predict(input_data_path=input_data_path, output_file_path=output_file_path)
        logging.info("Prediction pipeline executed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise
    '''

'''
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.components.data_tranformation import DataTransformation
from src.components.data_tranformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

    def initiate_data_ingestion(self, file_path: str):
        logging.info("Entered the data ingestion component")
        try:
            logging.info("Starting data ingestion...")

            # Use the file_path argument
            df = pd.read_csv(file_path)
            logging.info(f"Dataset loaded with shape: {df.shape}")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully.")

            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Split data into training and testing sets.")

            # Save train and test datasets
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logging.info("Data ingestion completed successfully.")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise CustomException(str(e), e)


if __name__ == "__main__":
    try:
        logging.info("Pipeline execution started.")

        # Step 1: Data Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion(file_path="notebook/data/Telco-Customer-Churn.csv")
        logging.info(f"Data ingestion completed. Train: {train_data}, Test: {test_data}")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        X_train_resampled, y_train_resampled, X_test_selected, y_test, selected_features = data_transformation.initiate_data_transformation(
            train_data, test_data, target_column="Churn"
        )
        logging.info(f"Data transformation completed. Simplified selected features: {selected_features}")

        # Step 3: Model Trainer
        modeltrainer = ModelTrainer()
        best_model, best_score = modeltrainer.train_and_evaluate(
            X_train_resampled, y_train_resampled, X_test_selected, y_test, selected_features
        )
        logging.info(f"Model training completed. Best Score: {best_score}")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise
        
        '''
